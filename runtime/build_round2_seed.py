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


def compose(skill: str, lesson: str) -> str:
    parts = []
    if skill:
        parts.append(skill)
    if lesson:
        parts.append("## Lessons from past failures\n" + lesson)
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="round-1 pure_rollout run dir")
    ap.add_argument("--lessons", default=None, help="train lessons jsonl (opid_lessons.jsonl)")
    ap.add_argument("--val-lessons", default=None, help="val lessons jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    train = load_jsonl(data_dir / "rollouts/traj_logs/train/rollout_000000.jsonl")
    val = load_jsonl(data_dir / "rollouts/traj_logs/val/rollout_000001.jsonl")
    train_lessons = load_jsonl(args.lessons) if args.lessons else []
    val_lessons = load_jsonl(args.val_lessons) if args.val_lessons else []

    def build(logs, lessons):
        out = []
        for i, log in enumerate(logs):
            skill = latest_parsed_skill(log)
            lesson = lessons[i].get("episode_skill", "") if i < len(lessons) else ""
            out.append(compose(skill, lesson))
        return out

    seed = {
        "train": build(train, train_lessons),
        "val": build(val, val_lessons),
    }
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(seed, ensure_ascii=False, indent=2) + "\n")

    print(f"train seeds: {sum(1 for s in seed['train'] if s)}/{len(seed['train'])} non-empty")
    print(f"val seeds:   {sum(1 for s in seed['val'] if s)}/{len(seed['val'])} non-empty")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
