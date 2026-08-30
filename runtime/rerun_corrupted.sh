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

# ---- drop corrupted round-1 state so b1 conditions re-run r1 ----
STATE="$SKILLRISE_ROOT/runtime/outputs/two_round_r1_state.tsv"
python3 - "$STATE" <<'EOF'
import sys
path = sys.argv[1]
kept = [l for l in open(path).read().splitlines()
        if l and not l.startswith(('b1_x2\t', 'b1_lesson_x2\t'))]
open(path, 'w').write('\n'.join(kept) + '\n')
print("[rerun] round-1 state after cleanup:")
print('\n'.join(kept))
EOF

# ---- re-run only the corrupted conditions ----
bash runtime/run_2round_experiment.sh baseline_lesson_x2 b1_x2 b1_lesson_x2

echo "=== corrupted re-run done $(date) ==="
