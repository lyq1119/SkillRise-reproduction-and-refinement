#!/usr/bin/env python3
"""EX: build a type-level common-mistakes bank (a1) from r1 train trajectories.

Two steps, both reusing SkillRL's verbatim prompt skeletons
(github.com/aiming-lab/SkillRL, skill_generation/alfworld.py):

  Step 1 (per failure episode):  strategic_guidelines
      -> mistakes_to_avoid: [{trigger_condition, bad_action}]
  Step 2 (per task type):       generate_common_mistakes
      -> [{mistake_id, description, why_it_happens, how_to_avoid}]

Each episode = one task position within a train trial (task_pos in the steps).
Only FAILED positions (reward <= 0) feed the bank; successful positions are
skipped (they carry no failure knowledge).

Adapted from SkillRL for ScienceWorld: the "Focus on" failure-mode list in
step 2 is rewritten for ScienceWorld instrument/measurement tasks; the prompt
skeleton and JSON schemas are unchanged.

Usage:
    ./runtime/venv/bin/python runtime/build_mistakes_bank.py \
        --data-dirs <r1dir1>,<r1dir2>,<r1dir3> \
        --out runtime/outputs/mistakes_bank.json \
        [--max-steps 40] [--dry-run]
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# DeepSeek goes DIRECT (bypass the intermittently-dead local proxy).
for _k in ("NO_PROXY", "no_proxy"):
    _cur = __import__("os").environ.get(_k, "")
    if "api.deepseek.com" not in _cur:
        __import__("os").environ[_k] = (_cur + "," if _cur else "") + "api.deepseek.com"


def load_env(env_path: Path) -> dict:
    env = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def parse_reward(s):
    if isinstance(s, list):
        return [float(x) for x in s]
    return [float(x.strip()) for x in s.strip("[]").split(",")]


def extract_action(response: str, limit: int = 80) -> str:
    m = re.search(r"<action>(.*?)</action>", response, re.S)
    if m:
        a = m.group(1).strip().replace("\n", " ")
        return a[:limit]
    tail = response.strip().replace("\n", " ")[-limit:]
    return tail if tail else ""


def task_description_from_input(input_text: str) -> str:
    m = re.search(r"Your task is to:\s*([^.\n]{5,200})", input_text)
    if m:
        return m.group(1).strip()
    for line in input_text.split("\n"):
        if "task is to" in line.lower():
            return line.strip()[:200]
    return input_text.strip()[:200]


def split_episodes(log, max_steps=40):
    """Split one train trial into per-task-position episodes.

    Each episode = the play steps at one task_pos (0-based), with its own
    reward. Returns [(task_desc, reward, [compact actions]), ...].
    """
    play = [s for s in log["steps"] if s.get("phase") == "play"]
    rewards = parse_reward(log["reward"])
    episodes = []
    by_pos = defaultdict(list)
    for s in play:
        by_pos[s.get("task_pos", 0)].append(s)
    for pos in sorted(by_pos):
        steps = by_pos[pos]
        task = task_description_from_input(steps[0].get("input", ""))
        reward = rewards[pos] if pos < len(rewards) else 0.0
        idxs = list(range(len(steps)))
        if len(steps) > max_steps:
            step = len(steps) / max_steps
            idxs = [int(i * step) for i in range(max_steps)]
        actions = [f"[{i}] {extract_action(steps[i]['response'])}" for i in idxs]
        episodes.append({"task": task, "reward": float(reward), "actions": actions})
    return episodes


# ------------------------------------------------------------------ #
# Step 1: strategic_guidelines (SkillRL verbatim skeleton, ScienceWorld input)
# ------------------------------------------------------------------ #

STRATEGIC_GUIDELINES_PROMPT = """You are an expert **Strategic Analyst** for an autonomous agent.
Your goal is to extract high-level `strategic_guidelines` from a trajectory.

**Input Analysis:**
Check the `Outcome` of the trajectory first.

### CASE 1: If Outcome is SUCCESS
Focus on **Replicability**. How can we repeat this success?
1.  **`planning_pattern`** (The Skeleton):
    * Abstract the successful trajectory into a high-level chain of logic.
    * Format: `ActionType -> ActionType -> ActionType`.
2.  **`mistakes_to_avoid`:** Leave empty `[]`.

### CASE 2: If Outcome is FAILURE
Focus on **Error Avoidance**. What was the "Wrong Direction"?
1.  **`planning_pattern`:** Set to `null`.
2.  **`mistakes_to_avoid`** (The Core Task):
    * Identify the **Root Cause** of the failure.
    * **Generalization Rule:** Write universal rules. NEVER use specific object
      or location names (no "beaker 2", no "test tube"). Use abstract terms
      like `[Target_Object]`, `[Instrument]`, `[Substance]`, `[Location]`.
    * Formulate a **"Negative Constraint"** object:
      * `trigger_condition`: The abstract context where the agent went wrong.
      * `bad_action`: The abstract incorrect action taken.

**Output Structure (JSON):**
```json
{
  "strategic_guidelines": {
    "planning_pattern": "String or null",
    "mistakes_to_avoid": [
      {"trigger_condition": "String", "bad_action": "String"}
    ]
  }
}
```
"""


def build_strategic_prompt(episode):
    outcome = "success" if episode["reward"] > 0 else "failure"
    traj = "\n".join(episode["actions"]) or "(empty)"
    return f"""{STRATEGIC_GUIDELINES_PROMPT}

Task: {episode['goal']}
Outcome: {outcome}
Interaction trajectory (action per step):
{traj}

Return ONLY the JSON object.
"""


# ------------------------------------------------------------------ #
# Step 2: generate_common_mistakes (SkillRL skeleton, ScienceWorld focus list)
# ------------------------------------------------------------------ #

COMMON_MISTAKES_PROMPT = """You are an expert at analyzing agent failures and distilling them into avoidable mistakes.

Analyze these failure patterns from an embodied AI agent:

{failure_data}

Generate {n} COMMON MISTAKES to avoid. Format as JSON array:
[
    {{
        "mistake_id": "err_001",
        "description": "What the mistake is (1 sentence)",
        "why_it_happens": "Why agents make this mistake (1 sentence)",
        "how_to_avoid": "Concrete actionable fix (1-2 sentences)"
    }}
]

Focus on:
- Navigation / exploration failures (getting stuck, not finding the object or room)
- Instrument / measurement errors (reading before the reading stabilizes,
  wrong heating/cooling/boiling procedure, incorrect use of a thermometer/scale)
- State management errors (wrong focused object, forgetting what was picked up)
- Goal misunderstanding (wrong substance, wrong amount, incomplete procedure)

Return ONLY the JSON array, no other text."""


def build_mistakes_prompt(task_type, failure_data, n=4):
    return COMMON_MISTAKES_PROMPT.format(
        failure_data=json.dumps(failure_data[:15], indent=2, ensure_ascii=False),
        n=n,
    )


# ------------------------------------------------------------------ #
# client
# ------------------------------------------------------------------ #

def deepseek_chat(env, model, prompt_text, max_tokens=1024, retries=3):
    import api_health
    client = api_health.make_deepseek_client(env)
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "You are an expert agent-behavior analyst. "
                                "Follow the user's instruction exactly."},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise SystemExit(
        f"[build_mistakes_bank] DeepSeek API unreachable after {retries} retries "
        f"(last error: {last_err}) — aborting. Re-run when the API is back.")


def parse_json_response(response: str):
    m = re.search(r"\{.*\}|\[.*\]", response or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def strategic_for_episode(env, model, episode, max_tokens=600):
    raw = deepseek_chat(env, model, build_strategic_prompt(episode), max_tokens=max_tokens)
    data = parse_json_response(raw) or {}
    sg = data.get("strategic_guidelines") or {}
    return sg.get("mistakes_to_avoid") or []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dirs", required=True,
                    help="comma-separated r1 pure_rollout run dirs")
    ap.add_argument("--out", required=True, help="output bank JSON path")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--mistakes-per-type", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    env = load_env(script_dir / ".env")
    model = env.get("DEEPSEEK_MODEL", "deepseek-chat")

    dirs = [Path(d).resolve() for d in args.data_dirs.split(",")]
    failures_by_type = defaultdict(list)
    for d in dirs:
        sel = json.loads((d / "selected_group.json").read_text())
        task_type = sel["task_type"]
        trajs = load_jsonl(d / "rollouts/traj_logs/train/rollout_000000.jsonl")
        for log in trajs:
            for ep in split_episodes(log, max_steps=args.max_steps):
                if ep["reward"] <= 0:  # failure episodes only
                    failures_by_type[task_type].append({
                        "task_type": task_type,
                        "goal": ep["task"],
                        "reward": ep["reward"],
                        "actions": ep["actions"],
                    })

    print(f"failure episodes per type:")
    for t, f in sorted(failures_by_type.items()):
        print(f"  {t}: {len(f)}")

    if args.dry_run:
        for t, f in sorted(failures_by_type.items()):
            print(f"\n[dry-run] type={t} first failure strategic prompt:")
            print(build_strategic_prompt({
                "goal": f[0]["goal"], "reward": 0.0, "actions": f[0]["actions"]})[:400])
            print(f"[dry-run] type={t} mistakes prompt:")
            print(build_mistakes_prompt(t, f, args.mistakes_per_type)[:400])
        return

    if not env.get("DEEPSEEK_API_KEY"):
        raise SystemExit("[fatal] DEEPSEEK_API_KEY not set in runtime/.env")
    import api_health
    if not api_health.wait_until_ready(env, model=model, timeout_seconds=600):
        raise SystemExit("[build_mistakes_bank] DeepSeek API unreachable for 10 min at start.")

    # Step 1: per failure episode -> mistakes_to_avoid
    raw_by_type = defaultdict(list)
    jobs = [(t, i, ep) for t, eps in failures_by_type.items() for i, ep in enumerate(eps)]
    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as ex:
        futures = {ex.submit(strategic_for_episode, env, model, ep): (t, i)
                   for t, i, ep in jobs}
        for fut, (t, i) in futures.items():
            mistakes = fut.result()
            raw_by_type[t].append({
                "task_type": t,
                "goal": failures_by_type[t][i]["goal"],
                "mistakes": mistakes,
            })
            print(f"[step1] {t} ep{i}: {len(mistakes)} mistakes_to_avoid")

    # Step 2: per type -> common mistakes
    bank = {}
    for t, failure_data in sorted(raw_by_type.items()):
        prompt = build_mistakes_prompt(t, failure_data, args.mistakes_per_type)
        raw = deepseek_chat(env, model, prompt, max_tokens=1024)
        parsed = parse_json_response(raw)
        if isinstance(parsed, list):
            bank[t] = parsed
        else:
            print(f"[step2] WARNING unparsable common-mistakes for {t}: {raw[:120]!r}")
            bank[t] = []
        print(f"[step2] {t}: {len(bank[t])} common mistakes")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"common_mistakes": bank,
                                    "source": "a1: strategic_guidelines + generate_common_mistakes (SkillRL)",
                                    "model": model},
                                   indent=2, ensure_ascii=False) + "\n")
    print(f"\nsaved {sum(len(v) for v in bank.values())} mistakes to {out_path}")


if __name__ == "__main__":
    main()
