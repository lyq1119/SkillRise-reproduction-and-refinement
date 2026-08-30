#!/usr/bin/env python3
"""E0: Fixed data-quality evaluation for SkillRise SFT data.

Judges generated trajectories + skill documents on a FIXED rubric via the
DeepSeek API (OpenAI-compatible), plus local structural metrics.  The rubric
prompts never change, so scores are comparable across data-generation runs.

Usage:
    ./venv/bin/python runtime/eval_data_quality.py \
        --data-dir runtime/outputs/pure_rollout/<run_dir> \
        [--out-dir runtime/eval_outputs] \
        [--split train] \
        [--dry-run]          # skip API calls, local structural metrics only

Config (runtime/.env):
    DEEPSEEK_API_KEY=...
    DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
    DEEPSEEK_MODEL=deepseek-chat
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# DeepSeek goes DIRECT (bypass the intermittently-dead local proxy).
for _k in ("NO_PROXY", "no_proxy"):
    _cur = os.environ.get(_k, "")
    if "api.deepseek.com" not in _cur:
        os.environ[_k] = (_cur + "," if _cur else "") + "api.deepseek.com"

# ---------------------------------------------------------------- #
# Fixed rubrics (do NOT change between runs)
# ---------------------------------------------------------------- #

TRAJECTORY_RUBRIC = """\
You are a strict data-quality judge for ScienceWorld agent trajectories. \
Score each trajectory on 4 dimensions as an integer 0-5. Be consistent and \
conservative: 3 = acceptable, 5 = excellent, 0 = broken.

Dimensions:
- T1 executability: the actions are valid, grammatically correct ScienceWorld commands the environment would accept.
- T2 task progress: the action sequence advances toward completing the task; for a failed trajectory, early steps should still be reasonable exploration toward the goal.
- T3 diversity: actions are not repetitive or looping; the agent explores meaningfully.
- T4 state coverage: the agent inspects relevant locations/objects and reacts to observations.

Return ONLY a JSON object, no other text: {"T1": <int>, "T2": <int>, "T3": <int>, "T4": <int>}
"""

SKILL_RUBRIC = """\
You are a strict data-quality judge for distilled skill documents in ScienceWorld. \
Score each skill on 5 dimensions as an integer 0-5. Be consistent and conservative: \
3 = acceptable, 5 = excellent, 0 = broken.

Dimensions:
- S1 structure: has clear sections (e.g. "When to use" / "Workflow" / "Pitfalls"), well organized.
- S2 abstraction: generalizes beyond the specific task instance (does not overfit to specific object names, room names, or numbers).
- S3 correctness: statements are plausible, actionable, and grounded in the observed trajectory (no invented steps).
- S4 transferability: would plausibly help solving a related task of the same family.
- S5 conciseness: concise, no verbatim trajectory copy, no filler.

Return ONLY a JSON object, no other text: {"S1": <int>, "S2": <int>, "S3": <int>, "S4": <int>, "S5": <int>}
"""

EMPTY_SKILL = "(no <skill> parsed)"

# ---------------------------------------------------------------- #
# .env loading (no external dep)
# ---------------------------------------------------------------- #

def load_env(env_path: Path):
    env = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


# ---------------------------------------------------------------- #
# Data loading / compact representation
# ---------------------------------------------------------------- #

def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def parse_reward(s):
    if isinstance(s, list):
        return [float(x) for x in s]
    return [float(x.strip()) for x in s.strip("[]").split(",")]


def extract_action(response: str, limit=100) -> str:
    m = re.search(r"<action>(.*?)</action>", response, re.S)
    if m:
        a = m.group(1).strip().replace("\n", " ")
        return a[:limit]
    # no action tag: keep the tail of the response (usually where the action sits)
    tail = response.strip().replace("\n", " ")[-limit:]
    return tail if tail else ""


def task_description_from_input(input_text: str) -> str:
    m = re.search(r"Your task is to:\s*([^.\n]{5,120})", input_text)
    if m:
        return m.group(1).strip()
    # fallback: the line right after "Your task is to"
    for line in input_text.split("\n"):
        if "task is to" in line.lower():
            return line.strip()[:120]
    return input_text.strip()[:120]


def compact_trajectory(log, max_steps=40):
    """Build a token-bounded representation of one trajectory."""
    play = [s for s in log["steps"] if s.get("phase") == "play"]
    task = task_description_from_input(log["steps"][0].get("input", ""))
    reward = parse_reward(log["reward"])
    # sample evenly if more than max_steps
    idxs = list(range(len(play)))
    if len(play) > max_steps:
        step = len(play) / max_steps
        idxs = [int(i * step) for i in range(max_steps)]
    actions = []
    for i in idxs:
        s = play[i]
        actions.append(f"[{i}] {extract_action(s['response'])}")
    return {"task": task, "reward": reward, "n_play_steps": len(play), "actions": actions}


def compact_skill_context(log, after_task_pos, max_steps=15):
    """Compact context for the task a skill was distilled from."""
    steps = [s for s in log["steps"] if s.get("phase") == "play"
             and s.get("task_pos") == after_task_pos]
    reward = parse_reward(log["reward"])
    out = [f"reward={reward}"]
    for i, s in enumerate(steps[:max_steps]):
        out.append(f"[{i}] {extract_action(s['response'])}")
    return "\n".join(out)


# ---------------------------------------------------------------- #
# Local structural metrics (no API)
# ---------------------------------------------------------------- #

def structural_stats(train, val):
    total_skills = 0
    parsed = 0
    complete = 0
    real_lens = []
    for log in train:
        for sk in log["skills"]:
            total_skills += 1
            doc = sk["skill"]
            if doc != EMPTY_SKILL:
                parsed += 1
                real_lens.append(len(doc))
                low = doc.lower()
                if "## when to use" in low and "## workflow" in low and "## pitfalls" in low:
                    complete += 1
    parse_rate = parsed / total_skills if total_skills else 0.0
    complete_rate = complete / parsed if parsed else 0.0

    # trajectory stats
    train_solve = []
    for log in train:
        rw = parse_reward(log["reward"])
        train_solve.append({"reward": rw, "solved_any": any(x > 0 for x in rw)})
    solved = sum(1 for t in train_solve if t["solved_any"])
    # diversity: distinct extracted actions / total play steps (train)
    div_total, div_uniq = 0, 0
    for log in train:
        acts = [extract_action(s["response"]) for s in log["steps"] if s.get("phase") == "play"]
        div_total += len(acts)
        div_uniq += len(set(acts))
    diversity = div_uniq / div_total if div_total else 0.0

    return {
        "n_train_trajectories": len(train),
        "n_val_trajectories": len(val),
        "skill_total": total_skills,
        "skill_parse_rate": round(parse_rate, 4),
        "skill_parse_failures": total_skills - parsed,
        "skill_structure_complete_rate": round(complete_rate, 4),
        "skill_real_doc_avg_len": round(sum(real_lens) / len(real_lens), 1) if real_lens else 0,
        "train_solved_any_rate": round(solved / len(train), 4) if train else 0.0,
        "train_action_diversity": round(diversity, 4),
    }


# ---------------------------------------------------------------- #
# DeepSeek judge
# ---------------------------------------------------------------- #

class Judge:
    def __init__(self, env):
        self.api_key = env.get("DEEPSEEK_API_KEY", "")
        self.base_url = env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
        self.model = env.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.dry_run = False
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            print(f"[warn] openai client init failed: {e}", file=sys.stderr)
            self._client = None

    def _chat(self, rubric, content, max_tokens=512):
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": rubric},
                {"role": "user", "content": content},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    def score(self, rubric, content, retries=3):
        if self.dry_run:
            return {}
        last = None
        for attempt in range(retries):
            try:
                text = self._chat(rubric, content)
                scores = extract_scores(text)
                if scores:
                    return scores
                last = f"unparseable response: {text[:200]}"
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
                time.sleep(1.5 * (attempt + 1))
        print(f"[error] judge failed for item: {last}", file=sys.stderr)
        return {}

    def close(self):
        pass


def extract_scores(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    out = {}
    for k, v in data.items():
        try:
            out[k] = int(v)
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------- #
# main
# ---------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="pure_rollout run output directory")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip DeepSeek API calls; local structural metrics only")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir or script_dir / "eval_outputs").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = load_env(script_dir / ".env")
    if not env.get("DEEPSEEK_API_KEY") and not args.dry_run:
        print("[fatal] DEEPSEEK_API_KEY not set. Fill runtime/.env first "
              "(or pass --dry-run for local metrics only).", file=sys.stderr)
        sys.exit(1)

    train = load_jsonl(data_dir / "rollouts/traj_logs/train/rollout_000000.jsonl")
    val = load_jsonl(data_dir / "rollouts/traj_logs/val/rollout_000001.jsonl")

    stats = structural_stats(train, val)
    print("=== local structural metrics ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if args.dry_run:
        print("\n[dry-run] skipping DeepSeek judge.")
        report = {"dry_run": True, "structural": stats}
        _save(out_dir, data_dir, report)
        return

    judge = Judge(env)
    judge.dry_run = False

    # ---- trajectory scoring (train split = the data we generate) ----
    logs = train if args.split == "train" else val
    traj_scores = []
    for i, log in enumerate(logs):
        compact = compact_trajectory(log)
        content = (
            f"## Trajectory\n"
            f"Task: {compact['task']}\n"
            f"Reward (per attempt): {compact['reward']}\n"
            f"Number of play steps: {compact['n_play_steps']}\n"
            f"Actions:\n" + "\n".join(compact["actions"])
        )
        sc = judge.score(TRAJECTORY_RUBRIC, content)
        traj_scores.append({"traj_idx": i, "scores": sc})
        print(f"[traj {i}] {sc}")

    # ---- skill scoring ----
    skill_scores = []
    for i, log in enumerate(train):
        for sk in log["skills"]:
            doc = sk["skill"]
            pos = sk["after_task_pos"]
            if doc == EMPTY_SKILL:
                sc = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "S5": 0, "unparsed": True}
            else:
                ctx = compact_skill_context(log, pos)
                content = (
                    f"## Skill Document\n{doc}\n\n"
                    f"## Task it was distilled from\n{ctx}"
                )
                sc = judge.score(SKILL_RUBRIC, content)
            skill_scores.append({"traj_idx": i, "after_task_pos": pos, "scores": sc})
            print(f"[skill traj{i} pos{pos}] {sc}")

    report = {
        "schema": "skillrise-data-quality-eval-v1",
        "data_dir": str(data_dir),
        "split_scored": args.split,
        "judge_model": judge.model,
        "rubric": {"trajectory": TRAJECTORY_RUBRIC, "skill": SKILL_RUBRIC},
        "structural": stats,
        "trajectory_scores": traj_scores,
        "skill_scores": skill_scores,
        "trajectory_means": _mean_scores(traj_scores),
        "skill_means": _mean_scores(skill_scores, unparsed_included=True),
    }
    path = _save(out_dir, data_dir, report)
    print(f"\nreport saved to {path}")


def _mean_scores(scored, unparsed_included=False):
    dims = {}
    for item in scored:
        for k, v in item["scores"].items():
            if k.startswith(("T", "S")):
                dims.setdefault(k, []).append(v)
    return {k: round(sum(v) / len(v), 3) for k, v in dims.items()}


def _save(out_dir, data_dir, report):
    import subprocess
    try:
        report["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=data_dir, text=True).strip()
    except Exception:
        report["git_commit"] = None
    ts = time.strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"eval_baseline_{ts}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return path


if __name__ == "__main__":
    main()
