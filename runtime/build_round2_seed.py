#!/usr/bin/env python3
"""D1: build the round-2 seed file from a round-1 run (+ optional lessons).

Reads a round-1 run's train/val trajectory jsonl, takes each row's LATEST
parsed skill (pos1, fallback pos0, skip unparsed), optionally composes in an
episode_skill lesson as a "## Lessons from past failures" section, and writes
{"train": [...], "val": [...]} for pure_rollout.py --seed-file.

Usage:
    ./runtime/venv/bin/python runtime/build_round2_seed.py \
        --data-dir runtime/outputs/pure_rollout/<round1_run> \
        [--lessons runtime/outputs/.../opid_lessons.jsonl] \
        [--val-lessons runtime/outputs/.../val_opid_lessons.jsonl] \
        --out runtime/outputs/round2_seed_<round1_run_id>.json
"""

import argparse
import json
from pathlib import Path

EMPTY_SKILL = "(no <skill> parsed)"


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def latest_parsed_skill(log) -> str:
    skills = log.get("skills", [])
    for sk in reversed(skills):
        doc = sk.get("skill", "")
        if doc and doc != EMPTY_SKILL:
            return doc
    return ""


def compose(skill: str, lesson: str, mistakes: str = "") -> str:
    parts = []
    if skill:
        parts.append(skill)
    if mistakes:
        parts.append("## Common mistakes\n" + mistakes)
    if lesson:
        parts.append("## Lessons from past failures\n" + lesson)
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="round-1 pure_rollout run dir")
    ap.add_argument("--lessons", default=None, help="train lessons jsonl (opid_lessons.jsonl)")
    ap.add_argument("--val-lessons", default=None, help="val lessons jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mistakes-bank", default=None,
                    help="EX: common-mistakes bank JSON {common_mistakes: {type: [...]}} "
                         "built by build_mistakes_bank.py; composed into the seed as "
                         "'## Common mistakes' (per-task-type for val)")
    ap.add_argument("--val-tasks-file", default=None,
                    help="EX: pinned val task list JSON {val: [{task_id, variation, task_type}]} "
                         "(row order must match the run's val rollout order)")
    ap.add_argument("--merged-skills-file", default=None,
                    help="EX: merged+validated skills JSON {type: {skill, score, status}} "
                         "built by merge_validate_skills.py; injected per type into val "
                         "(and train) seed entries")
    ap.add_argument("--merge-mode", choices=["replace", "stack"], default=None,
                    help="EX: replace -> merged skill substitutes the r1 val online skill; "
                         "stack -> merged skill appended to it. Only applies to types present "
                         "in the merged-skills file.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    train = load_jsonl(data_dir / "rollouts/traj_logs/train/rollout_000000.jsonl")
    val = load_jsonl(data_dir / "rollouts/traj_logs/val/rollout_000001.jsonl")
    train_lessons = load_jsonl(args.lessons) if args.lessons else []
    val_lessons = load_jsonl(args.val_lessons) if args.val_lessons else []

    bank = {}
    if args.mistakes_bank:
        bank = json.loads(Path(args.mistakes_bank).read_text()).get("common_mistakes", {})
    group_type = None
    if args.mistakes_bank or args.merged_skills_file:
        sel = json.loads((data_dir / "selected_group.json").read_text())
        group_type = sel["task_type"]
    val_types = []
    if args.val_tasks_file:
        val_types = [r["task_type"] for r in
                     json.loads(Path(args.val_tasks_file).read_text())["val"]]

    merged_skills = {}
    if args.merged_skills_file:
        ms = json.loads(Path(args.merged_skills_file).read_text())
        merged_skills = {t: v["skill"] for t, v in ms.items()
                         if v.get("status", "") != "dropped" and v.get("skill")}

    def fmt_mistakes(type_: str) -> str:
        ms = bank.get(type_, [])
        if not ms:
            return ""
        lines = []
        for i, m in enumerate(ms, 1):
            lines.append(f"{i}. {m.get('description', '')}")
            if m.get("how_to_avoid"):
                lines.append(f"   -> {m['how_to_avoid']}")
        return "\n".join(lines)

    def build(logs, lessons, per_row_mistakes_type, merged_row=None):
        # merged_row: per-row merged skill ("" = none). replace mode uses it in
        # place of the online skill; stack mode appends it to the online skill.
        out = []
        for i, log in enumerate(logs):
            skill = latest_parsed_skill(log)
            mg = merged_row[i] if merged_row else ""
            if args.merge_mode == "replace" and mg:
                skill = mg
            elif args.merge_mode == "stack" and mg:
                skill = f"{skill}\n\n{mg}" if skill else mg
            lesson = lessons[i].get("episode_skill", "") if i < len(lessons) else ""
            mt = None
            if per_row_mistakes_type:
                mt = fmt_mistakes(per_row_mistakes_type[i])
            out.append(compose(skill, lesson, mt or ""))
        return out

    train_merged = None
    if args.merge_mode and merged_skills.get(group_type):
        train_merged = [merged_skills[group_type]] * len(train)
    val_merged = None
    if args.merge_mode and val_types:
        val_merged = [merged_skills.get(t, "") for t in val_types]

    seed = {
        "train": build(train, train_lessons,
                       [group_type] * len(train) if group_type else None,
                       train_merged),
        "val": build(val, val_lessons, val_types if val_types else None, val_merged),
    }
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(seed, ensure_ascii=False, indent=2) + "\n")

    print(f"train seeds: {sum(1 for s in seed['train'] if s)}/{len(seed['train'])} non-empty")
    print(f"val seeds:   {sum(1 for s in seed['val'] if s)}/{len(seed['val'])} non-empty")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
