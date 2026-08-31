#!/usr/bin/env python3
"""EX: evolve the merged skills from train-side failures (Recuris-adapted).

Adapted from Recuris (Gen-Verse/Recuris) metaagent protocols:
- diagnosis.md: evidence-bounded diagnosis — only failed-trace evidence, cite
  concrete moments, abstain over guess, classify the owner (simplified to
  skill-content / skill-when-to-use / model; only skill owners are patchable).
- patch.md: smallest connected fix — output the full revised skill but change
  ONLY the targeted section; preserve everything else verbatim; use placeholders.
- gate: judge-based validation (the merge_validate 4-dim rubric) as a cheap
  pre-filter; the statistical gate (repair + net>0 + regression cap) is applied
  AFTER the round's val run.

Input: merge-replace round-2 runs (whose train phase injected the merged skills)
+ the current merged_skills.json. Failed train episodes (per task position) are
diagnosed; per type the fixes are aggregated into one revised skill.

Usage:
    ./runtime/venv/bin/python runtime/evolve_skills.py \
        --r2-runs <r2dir_melting>,<r2dir_inclined> \
        --merged-skills runtime/outputs/exp_merge_validate/merged_skills.json \
        --out runtime/outputs/exp_skill_evolution/evolved_skills.json \
        [--dry-run]
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
    """Per-task-position episodes with own reward + compact actions."""
    play = [s for s in log["steps"] if s.get("phase") == "play"]
    rewards = parse_reward(log["reward"])
    by_pos = defaultdict(list)
    for s in play:
        by_pos[s.get("task_pos", 0)].append(s)
    episodes = []
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
# Diagnosis prompt (adapted from Recuris diagnosis.md)
# ------------------------------------------------------------------ #

DIAGNOSIS_PROMPT = """You are diagnosing why an agent failed a ScienceWorld task. The agent had the following skill injected into its prompt.

Evidence-bounded rules:
- Read the failed trajectory and locate the FIRST step where behavior diverged from the correct path. Cite that concrete step in your diagnosis.
- Separate observation from hypothesis. If the evidence is insufficient to implicate the skill, say so (owner: model).
- Do NOT fabricate a fix merely because the episode failed. Ambiguity is not a patch opportunity.

Classify the owner:
- skill-content: the skill's Workflow has a wrong or missing step that caused the failure.
- skill-when-to-use: the skill is correct but was applied in the wrong context (its trigger condition mismatches this task).
- model: the agent failed despite the skill being correct and applicable. fixes: none.

If owner is skill-content or skill-when-to-use, provide a targeted fix:
- Change ONLY the part of the skill that caused the failure. Preserve all other parts verbatim.
- Output the FULL revised skill document (only the targeted change applied).
- Use placeholders; never embed this task's specific IDs, values, or answers.

Skill injected:
{skill}

Failed task: {task}
Failed trajectory (action per step):
{actions}

Output JSON:
{{"owner": "skill-content"|"skill-when-to-use"|"model",
  "diagnosis": "concrete diagnosis citing the failing step",
  "targeted_section": "which section changed (or null)",
  "revised_skill": "full revised skill document or null"}}
"""


# ------------------------------------------------------------------ #
# Validation gate prompt (reuse the merge_validate 4-dim rubric)
# ------------------------------------------------------------------ #

VALIDATE_PROMPT = """You are a skill quality judge for an agent solving ScienceWorld tasks. Score this skill document 1-5 on each dimension:

1. Factual correctness — is every claim true for ScienceWorld?
2. Actionability — does it state clearly what to do?
3. Granularity — not too vague, not overfit to one instance?
4. Trajectory support — grounded in the source trajectory?

Skill document:
{skill}

Output JSON:
{{"score": <1-5>, "issues": ["..."]}}
Return ONLY the JSON object.
"""


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
        f"[evolve_skills] DeepSeek API unreachable after {retries} retries "
        f"(last error: {last_err}) — aborting. Re-run when the API is back.")


def parse_json_response(response: str):
    m = re.search(r"\{.*\}", response or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r2-runs", required=True,
                    help="comma-separated merge-replace r2 run dirs (train failures source)")
    ap.add_argument("--merged-skills", required=True,
                    help="current merged_skills.json (the skill being evolved)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-gate-score", type=float, default=3.0,
                    help="judge-based gate: accept revised skill if score >= this")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    env = load_env(script_dir / ".env")
    model = env.get("DEEPSEEK_MODEL", "deepseek-chat")

    merged = json.loads(Path(args.merged_skills).read_text())
    dirs = [Path(d).resolve() for d in args.r2_runs.split(",")]

    # collect failed episodes per type
    failed_by_type = defaultdict(list)
    for d in dirs:
        sel = json.loads((d / "selected_group.json").read_text())
        task_type = sel["task_type"]
        trajs = load_jsonl(d / "rollouts/traj_logs/train/rollout_000000.jsonl")
        for log in trajs:
            for ep in split_episodes(log):
                if ep["reward"] <= 0:
                    failed_by_type[task_type].append(ep)

    print("failed episodes per type:")
    for t, f in sorted(failed_by_type.items()):
        print(f"  {t}: {len(f)}")

    if args.dry_run:
        for t, f in sorted(failed_by_type.items()):
            print(f"\n[dry-run] {t} first diagnosis prompt:\n"
                  + DIAGNOSIS_PROMPT.format(
                      skill=merged.get(t, {}).get("skill", "(dropped)"),
                      task=f[0]["task"], actions="\n".join(f[0]["actions"]))[:300])
        return

    if not env.get("DEEPSEEK_API_KEY"):
        raise SystemExit("[fatal] DEEPSEEK_API_KEY not set in runtime/.env")
    import api_health
    if not api_health.wait_until_ready(env, model=model, timeout_seconds=600):
        raise SystemExit("[evolve_skills] DeepSeek API unreachable for 10 min at start.")

    # ---- diagnose each failed episode ----
    def diagnose(ep, skill):
        raw = deepseek_chat(env, model,
                            DIAGNOSIS_PROMPT.format(skill=skill, task=ep["task"],
                                                    actions="\n".join(ep["actions"])),
                            max_tokens=1200)
        return parse_json_response(raw) or {}

    evolved = {}
    for t, eps in sorted(failed_by_type.items()):
        cur = merged.get(t, {})
        skill = cur.get("skill", "")
        if not skill:
            print(f"[{t}] no merged skill (dropped) — skip evolution")
            evolved[t] = dict(cur, status="no-skill")
            continue

        print(f"[{t}] diagnosing {len(eps)} failed episodes...")
        fixes = []
        with ThreadPoolExecutor(max_workers=min(4, len(eps))) as ex:
            for d in ex.map(lambda ep: diagnose(ep, skill), eps):
                owner = d.get("owner", "model")
                rv = (d.get("revised_skill") or "").strip()
                print(f"  owner={owner} revised={'yes' if rv else 'no'} | {d.get('diagnosis','')[:80]}")
                if owner in ("skill-content", "skill-when-to-use") and rv:
                    fixes.append({"owner": owner, "diagnosis": d.get("diagnosis", ""),
                                  "revised_skill": rv})

        if not fixes:
            print(f"[{t}] no skill-attributable failures — keep original (model failures only)")
            evolved[t] = dict(cur, status="keep", evolution="none")
            continue

        # use the last skill-attributable revision (or most common); simple: last
        revised = fixes[-1]["revised_skill"]
        # ---- judge-based gate ----
        raw_v = deepseek_chat(env, model, VALIDATE_PROMPT.format(skill=revised), max_tokens=600)
        vd = parse_json_response(raw_v) or {}
        try:
            score = float(vd.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        if score >= args.min_gate_score:
            print(f"[{t}] GATE ACCEPT revised skill (score {score})")
            evolved[t] = {"skill": revised, "score": score, "status": "evolved",
                          "evolution": fixes[-1]["owner"], "diagnosis": fixes[-1]["diagnosis"]}
        else:
            print(f"[{t}] GATE REJECT revised skill (score {score}) — keep original")
            evolved[t] = dict(cur, status="keep", evolution="gate-rejected")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(evolved, indent=2, ensure_ascii=False) + "\n")
    print(f"\nsaved evolved skills to {out_path}")
    for t, r in evolved.items():
        print(f"  {t}: {r.get('status')} (evolution={r.get('evolution','-')})")


if __name__ == "__main__":
    main()
