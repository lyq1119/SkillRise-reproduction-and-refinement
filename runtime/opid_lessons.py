#!/usr/bin/env python3
"""C2 (方案A): OPID-style failure-lesson extraction via DeepSeek.

Reads a pure_rollout run's TRAIN trajectories and produces, for each
trajectory, an OPID-format lesson record:

    {traj_idx, reward, episode_summary, episode_skill, step_skills}

Adapted from OPID's analyzer design (github.com/jinyangwu/OPID):
- episode_skill for a failed episode = avoidance rules
  (core mistake + warning signs the agent should avoid); for a successful
  episode = the workflow that made it work.
- step_skills = short, policy-facing, imperative lessons written at the
  critical steps of the episode.

方案A: critical steps are selected by DeepSeek itself from the compact
trajectory (no step-level rewards required).

Cost control: each step is reduced to its <action> text (plus step index),
and the trajectory is capped at --max-steps, so long episodes stay cheap.
The full prompt input / CoT reasoning is never sent.

Usage:
    ./runtime/venv/bin/python runtime/opid_lessons.py \
        --data-dir runtime/outputs/pure_rollout/<run_dir> \
        [--out runtime/outputs/opid_lessons/<run_id>_lessons.jsonl] \
        [--max-steps 40] [--dry-run]

Config (runtime/.env): DEEPSEEK_API_KEY / BASE_URL / MODEL.
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


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


def compact_trajectory(log, max_steps: int = 40):
    """Token-bounded per-step representation: only the action + step index."""
    play = [s for s in log["steps"] if s.get("phase") == "play"]
    task = task_description_from_input(log["steps"][0].get("input", ""))
    reward = parse_reward(log["reward"])
    idxs = list(range(len(play)))
    if len(play) > max_steps:
        step = len(play) / max_steps
        idxs = [int(i * step) for i in range(max_steps)]
    actions = [f"[{i}] {extract_action(play[i]['response'])}" for i in idxs]
    return {
        "task": task,
        "reward": reward,
        "n_play_steps": len(play),
        "actions": actions,
    }


# ------------------------------------------------------------------ #
# Prompt (adapted from OPID _build_episode_analysis_prompt)
# ------------------------------------------------------------------ #

def build_analysis_prompt(task, reward, actions, max_step_skills=5):
    any_success = any(r > 0 for r in reward)
    outcome_label = "success" if any_success else "failure"

    if any_success:
        episode_skill_instruction = (
            "Write one episode_skill that extracts the successful trajectory into a workflow: "
            "the core decision rule and action ordering that made this trajectory work."
        )
    else:
        episode_skill_instruction = (
            "Write one episode_skill that extracts the failed trajectory into avoidance rules: "
            "the core mistake and warning signs that the agent should avoid."
        )

    selection_instruction = (
        f"Provide concise, action-oriented decision skills for at most {max_step_skills} "
        "critical step(s) as entries in step_skills. Pick the steps where the agent's action "
        "was the wrong decision (the failure turning points). Use the full episode to infer "
        "the skill, but phrase each skill as one short imperative sentence the policy can "
        "act on at that step."
    )

    return f"""Analyze the following agent episode and return ONLY valid JSON.

You need to complete all three fields:
1. Write a concise episode_summary.
2. {episode_skill_instruction}
3. {selection_instruction}

Important constraints:
- Step indexing is 0-based and matches the [N] labels below.
- Use the task description together with the episode context to judge progress and mistakes.
- Each step_skills value should be one short imperative sentence for the policy at that step.
- Write step_skills as policy-facing advice, not as a retrospective explanation of the trajectory.
- Return only these top-level fields: episode_summary, episode_skill, step_skills.
- The chosen steps are exactly the keys present in step_skills.

Return format:
{{
  "episode_summary": "string",
  "episode_skill": "string",
  "step_skills": {{
    "0": "skill for step 0",
    "2": "skill for step 2"
  }}
}}

Episode context:
- Task description: {task}
- episode_success: {outcome_label}
- Interaction trajectory (action per step):
{chr(10).join(actions)}
"""


# ------------------------------------------------------------------ #
# DeepSeek client
# ------------------------------------------------------------------ #

def deepseek_chat(env, model, prompt_text, max_tokens=1024, retries=3):
    from openai import OpenAI
    client = OpenAI(
        api_key=env.get("DEEPSEEK_API_KEY", ""),
        base_url=env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/"),
    )
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
    print(f"[opid_lessons] error after {retries} retries: {last_err}", file=sys.stderr)
    return ""


def parse_lesson_response(response: str) -> dict:
    m = re.search(r"\{.*\}", response or "", re.S)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    step_skills = {}
    raw_ss = data.get("step_skills") or {}
    if isinstance(raw_ss, dict):
        for k, v in raw_ss.items():
            try:
                step_skills[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
    return {
        "episode_summary": str(data.get("episode_summary", "")),
        "episode_skill": str(data.get("episode_skill", "")),
        "step_skills": step_skills,
    }


def analyze_one(env, model, compact, max_step_skills):
    prompt_text = build_analysis_prompt(
        compact["task"], compact["reward"], compact["actions"], max_step_skills)
    raw = deepseek_chat(env, model, prompt_text)
    parsed = parse_lesson_response(raw)
    return {
        "episode_summary": parsed.get("episode_summary", ""),
        "episode_skill": parsed.get("episode_skill", ""),
        "step_skills": parsed.get("step_skills", {}),
        "_raw_response_len": len(raw),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="pure_rollout run output dir")
    ap.add_argument("--split", choices=["train", "val"], default="train",
                    help="which trajectory split to analyze")
    ap.add_argument("--out", default=None, help="output jsonl path")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--max-step-skills", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the compact trajectories + prompts without calling the API")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).resolve()

    env = load_env(script_dir / ".env")
    if not env.get("DEEPSEEK_API_KEY") and not args.dry_run:
        print("[fatal] DEEPSEEK_API_KEY not set in runtime/.env", file=sys.stderr)
        sys.exit(1)
    model = env.get("DEEPSEEK_MODEL", "deepseek-chat")

    split_file = "train/rollout_000000.jsonl" if args.split == "train" else "val/rollout_000001.jsonl"
    trajs = load_jsonl(data_dir / "rollouts/traj_logs" / split_file)

    compacts = [compact_trajectory(log, max_steps=args.max_steps) for log in trajs]
    for i, c in enumerate(compacts):
        print(f"[traj {i}] task={c['task'][:60]!r} reward={c['reward']} "
              f"play_steps={c['n_play_steps']} actions_shown={len(c['actions'])}")

    if args.dry_run:
        print("\n[dry-run] first prompt preview:")
        print(build_analysis_prompt(compacts[0]["task"], compacts[0]["reward"],
                                    compacts[0]["actions"], args.max_step_skills))
        return

    out_path = Path(args.out or data_dir / f"opid_lessons_{args.split}.jsonl").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = [None] * len(trajs)
    with ThreadPoolExecutor(max_workers=min(4, len(trajs))) as ex:
        futures = {i: ex.submit(analyze_one, env, model, compacts[i], args.max_step_skills)
                   for i in range(len(trajs))}
        for i, fut in futures.items():
            lesson = fut.result()
            results[i] = {
                "traj_idx": i,
                "reward": compacts[i]["reward"],
                "task": compacts[i]["task"],
                **lesson,
            }
            print(f"[traj {i}] summary={lesson['episode_summary'][:60]!r} "
                  f"steps={list(lesson['step_skills'])}")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nsaved {len(results)} lessons to {out_path}")


if __name__ == "__main__":
    main()
