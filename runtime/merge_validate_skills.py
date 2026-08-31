#!/usr/bin/env python3
"""EX: merge + validate the round-1 train skills (per group/type).

Method borrowed from the literature (no released scripts exist):
- Merge   — SkillComposer's "merge two related skills into a more general one"
            (we skip its embedding-based candidate search: the N skills of one
            group are already same-type, so we merge them directly).
- Validate — Parametric Skills' Validate dimensions (actionable / trajectory-
            backed / non-trivial) + factual correctness for ScienceWorld
            (ReMe's CORRECT/REFINE semantics: fix errors, add boundaries).

Flow per group: curate skills (one per trajectory) -> MERGE into 1 type-level
skill -> VALIDATE (score 1-5; <3 drop, 3-4 rewrite once, >=4 keep).

Output:
  { "<task_type>": {"skill": str, "score": float, "status": "keep|rewritten|dropped"} }

Usage:
    ./runtime/venv/bin/python runtime/merge_validate_skills.py \
        --data-dirs <r1dir1>,<r1dir2>,<r1dir3> \
        --out runtime/outputs/exp_merge/merged_skills.json [--dry-run]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

# DeepSeek goes DIRECT (bypass the intermittently-dead local proxy).
for _k in ("NO_PROXY", "no_proxy"):
    _cur = __import__("os").environ.get(_k, "")
    if "api.deepseek.com" not in _cur:
        __import__("os").environ[_k] = (_cur + "," if _cur else "") + "api.deepseek.com"

EMPTY_SKILL = "(no <skill> parsed)"


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


def latest_parsed_skill(log) -> str:
    skills = log.get("skills", [])
    for sk in reversed(skills):
        doc = sk.get("skill", "")
        if doc and doc != EMPTY_SKILL:
            return doc
    return ""


# ------------------------------------------------------------------ #
# Merge prompt (SkillComposer E.2 intent, self-written)
# ------------------------------------------------------------------ #

MERGE_PROMPT = """You are a skill curator. The following {n} skill documents were distilled from related tasks of the same type ({task_type}). Merge them into ONE concise, actionable skill document.

- Keep the shared core procedure that applies to all these tasks.
- Drop instance-specific details and redundant phrasing.
- Keep the format: "## When to use" then "## Workflow".
- Do not invent facts not supported by the input skills.

Skills to merge:
{skills}

Return ONLY the merged skill document.
"""


# ------------------------------------------------------------------ #
# Validate prompt (Parametric Skills dims + factual correctness)
# ------------------------------------------------------------------ #

VALIDATE_PROMPT = """You are a skill quality judge for an agent solving ScienceWorld tasks. Score this skill document 1-5 on each dimension:

1. Factual correctness — is every claim true for ScienceWorld? (e.g. melting point IS measured with a thermometer in a hot water bath; a multimeter measures conductivity)
2. Actionability — does it state clearly what to do?
3. Granularity — not too vague (empty advice) and not overfit to one instance?
4. Trajectory support — is it grounded in the source trajectory rather than hallucinated?

Skill document:
{skill}

Output JSON:
{{"score": <1-5>, "issues": ["..."], "rewritten": "<improved version or null>"}}

Rules:
- If score < 4, provide a "rewritten" version that fixes the issues.
- If score < 3, the skill is unusable and should be dropped (still give a rewritten attempt).
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
        f"[merge_validate] DeepSeek API unreachable after {retries} retries "
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
    ap.add_argument("--data-dirs", required=True, help="comma-separated r1 run dirs")
    ap.add_argument("--out", required=True, help="output merged+validated skills JSON")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    env = load_env(script_dir / ".env")
    model = env.get("DEEPSEEK_MODEL", "deepseek-chat")

    dirs = [Path(d).resolve() for d in args.data_dirs.split(",")]
    merged = {}
    for d in dirs:
        sel = json.loads((d / "selected_group.json").read_text())
        task_type = sel["task_type"]
        trajs = load_jsonl(d / "rollouts/traj_logs/train/rollout_000000.jsonl")
        skills = [latest_parsed_skill(t) for t in trajs]
        skills = [s for s in skills if s]
        print(f"[{task_type}] {len(skills)} source skills")

        if not skills:
            merged[task_type] = {"skill": "", "score": 0.0, "status": "dropped"}
            continue

        if args.dry_run:
            print(f"[dry-run] {task_type} merge prompt preview:\n"
                  + MERGE_PROMPT.format(n=len(skills), task_type=task_type,
                                        skills="\n\n".join(skills))[:300])
            continue

        if not env.get("DEEPSEEK_API_KEY"):
            raise SystemExit("[fatal] DEEPSEEK_API_KEY not set in runtime/.env")
        import api_health
        if not api_health.wait_until_ready(env, model=model, timeout_seconds=600):
            raise SystemExit("[merge_validate] DeepSeek API unreachable for 10 min at start.")

        # ---- merge ----
        raw = deepseek_chat(env, model,
                            MERGE_PROMPT.format(n=len(skills), task_type=task_type,
                                                skills="\n\n".join(skills)),
                            max_tokens=1024)
        merged_skill = raw.strip()
        print(f"[merge] {task_type}: merged len={len(merged_skill)}")

        # ---- validate ----
        raw_v = deepseek_chat(env, model, VALIDATE_PROMPT.format(skill=merged_skill),
                              max_tokens=1024)
        vd = parse_json_response(raw_v) or {}
        try:
            score = float(vd.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        issues = vd.get("issues") or []
        rewritten = (vd.get("rewritten") or "").strip()

        if score >= 4:
            status, final_skill = "keep", merged_skill
        elif score >= 3:
            status, final_skill = "rewritten", (rewritten or merged_skill)
        else:
            status, final_skill = "dropped", ""
        print(f"[validate] {task_type}: score={score} status={status} issues={len(issues)}")
        merged[task_type] = {"skill": final_skill, "score": score, "status": status}

    if args.dry_run:
        return
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    print(f"\nsaved merged+validated skills to {out_path}")
    for t, r in merged.items():
        print(f"  {t}: {r['status']} (score {r['score']}) len={len(r['skill'])}")


if __name__ == "__main__":
    main()
