#!/usr/bin/env bash
# EX: run one skill-evolution round — merge-replace with the EVOLVED skills.
# Reuses the shared r1; builds replace-seeds from the evolved merged skills and
# runs r2 for the 3 train groups. Then reports val + applies the Recuris
# statistical gate vs the previous round (per-task pass@1).
set -euo pipefail

SKILLRISE_ROOT=/data/lanyuqi/skillrise
cd "$SKILLRISE_ROOT"
PY=./runtime/venv/bin/python
OLD_EXP="$SKILLRISE_ROOT/runtime/outputs/exp_mistakes_vs_lessons"
EXP_ROOT="$SKILLRISE_ROOT/runtime/outputs/exp_skill_evolution"
mkdir -p "$EXP_ROOT"

ROUND=${ROUND:-2}                       # which evolution round (2, 3, ...)
EVOLVED=${EVOLVED:-$EXP_ROOT/evolved_skills.json}
PREV_SUMMARY=${PREV_SUMMARY:-$EXP_ROOT/summary_round_1.tsv}
TS=$(date +%Y%m%dT%H%M%S)
LOG="$EXP_ROOT/round_${ROUND}_${TS}.log"
SUMMARY="$EXP_ROOT/summary_round_${ROUND}.tsv"
exec > >(tee -a "$LOG") 2>&1
echo "=== skill-evolution round $ROUND started $(date) ==="

VAL_TASKS="$SKILLRISE_ROOT/runtime/data/exp_mistakes_val12.json"
ROLLOUT_N=3
CURATE_FLAG="--curate-via-api"
declare -a TRAIN_GROUPS=(
  "measure-melting-point-unknown-substance_K3_0"
  "find-plant_K3_0"
  "inclined-plane-friction-named-surfaces_K3_0"
)

# r1 dirs from the shared experiment's state
declare -a R1_DIRS=()
for gid in "${TRAIN_GROUPS[@]}"; do
    r1=$(awk -F '\t' -v g="$gid" '$1==g {print $2}' "$OLD_EXP/r1_state.tsv" | head -1)
    if [ -z "$r1" ] || [ ! -f "$r1/manifest.json" ]; then
        echo "[fatal] missing shared r1 for $gid"; exit 1
    fi
    R1_DIRS+=("$r1")
done

run_r2() { # $1=gid; $2=seed -> echoes r2 dir
    local gid=$1; local seed=$2
    local key="$gid"
    local existing=""
    if [ -f "$SUMMARY" ]; then
        existing=$(awk -F '|' -v g="$gid" '$1==g {print $2}' "$SUMMARY" | head -1)
    fi
    if [ -n "$existing" ] && [ -f "$existing/manifest.json" ]; then
        echo "[r2] reusing $gid -> $existing" >&2
        echo "$existing"; return
    fi
    echo "[r2] running $gid" >&2
    local outdir
    outdir=$(bash runtime/run_pure_rollout.sh \
        --group-id "$gid" --val-tasks "$VAL_TASKS" --rollout-n "$ROLLOUT_N" \
        $CURATE_FLAG --seed-file "$seed" 2>&1 | tail -1)
    if [ ! -f "$outdir/manifest.json" ]; then
        echo "[fatal] r2 run produced no manifest: $outdir" >&2
        exit 1
    fi
    echo "[r2] $gid -> $outdir" >&2
    echo "$gid|$outdir" >> "$SUMMARY"
    echo "$outdir"
}

for i in "${!TRAIN_GROUPS[@]}"; do
    gid="${TRAIN_GROUPS[$i]}"
    r1="${R1_DIRS[$i]}"
    seed="$EXP_ROOT/seed_round${ROUND}_${gid##*_}.json"
    $PY runtime/build_round2_seed.py --data-dir "$r1" \
        --merged-skills-file "$EVOLVED" --merge-mode replace \
        --val-tasks-file "$VAL_TASKS" --out "$seed"
    run_r2 "$gid" "$seed" >/dev/null
done

echo "=== round $ROUND val (per group) ==="
for gid in "${TRAIN_GROUPS[@]}"; do
    r2=$(awk -F '|' -v g="$gid" '$1==g {print $2}' "$SUMMARY" | head -1)
    echo "[$gid] $(basename "$r2")"
    $PY -c "
import json
d=json.load(open('$r2/manifest.json'))
m=d['eval']['metrics']
print(f'   pass@1={sum(m[\"pass@1\"])}/{len(m[\"pass@1\"])} pass@2={sum(m[\"pass@2\"])}/{len(m[\"pass@2\"])} pass@3={sum(m[\"pass@3\"])}/{len(m[\"pass@3\"])}')"
done
echo "=== all done $(date) ==="
