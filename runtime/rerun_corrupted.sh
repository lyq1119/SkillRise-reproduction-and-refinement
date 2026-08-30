#!/usr/bin/env bash
# Re-run the conditions corrupted by the 2026-08-29/30 DeepSeek API outage.
#
# Waits for the API to be reachable, drops the corrupted round-1 state for
# b1_x2 / b1_lesson_x2 (so they re-run r1 fresh; baseline_lesson_x2 keeps its
# valid r1 and only re-extracts lessons + re-runs r2), then runs the harness
# for those three conditions. baseline_x2 was valid and is left alone.
set -euo pipefail

SKILLRISE_ROOT=/data/lanyuqi/skillrise
cd "$SKILLRISE_ROOT"
PY=./runtime/venv/bin/python

TS=$(date +%Y%m%dT%H%M%S)
LOG="$SKILLRISE_ROOT/runtime/outputs/rerun_corrupted_${TS}.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== corrupted-condition re-run started $(date) ==="

# ---- wait for DeepSeek API (poll every 60s, indefinitely) ----
echo "[rerun] waiting for DeepSeek API..."
while ! $PY -c "
import sys; sys.path.insert(0, 'runtime')
from pathlib import Path
from pure_rollout import load_env
import api_health
sys.exit(0 if api_health.check_deepseek(load_env(Path('runtime/.env'))) else 1)
"; do
    echo "[rerun] $(date +%H:%M:%S) API not reachable — retrying in 60s"
    sleep 60
done
echo "[rerun] DeepSeek API reachable at $(date)"

# ---- drop round-1 state entries whose dir has no manifest (e.g. an abort
#      message or a partial run) — valid r1 dirs are reused by the harness.
STATE="$SKILLRISE_ROOT/runtime/outputs/two_round_r1_state.tsv"
python3 - "$STATE" <<'EOF'
import os, sys
path = sys.argv[1]
kept = []
for l in open(path).read().splitlines():
    if not l or '\t' not in l:
        continue
    cond, r1 = l.split('\t', 1)
    if os.path.isfile(os.path.join(r1, 'manifest.json')):
        kept.append(f'{cond}\t{r1}')
    else:
        print(f"[rerun] dropping invalid r1 for {cond}: {r1[:70]}")
open(path, 'w').write('\n'.join(kept) + '\n')
print("[rerun] round-1 state after cleanup:")
print('\n'.join(kept))
EOF

# ---- re-run only the conditions that are still incomplete ----
# (baseline_x2 and baseline_lesson_x2 are already valid; b1_x2 reuses its
# valid r1 and re-runs r2; b1_lesson_x2 re-runs r1+r2)
bash runtime/run_2round_experiment.sh b1_x2 b1_lesson_x2

echo "=== corrupted re-run done $(date) ==="
