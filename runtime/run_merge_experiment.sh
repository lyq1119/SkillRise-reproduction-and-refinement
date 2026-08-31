#!/usr/bin/env bash
# EX: merge+validate skill experiment (merge-replace / merge-stack vs
#     previously-completed control / lesson).
#
# Reuses the SHARED r1 from exp_mistakes_vs_lessons (same 3 train groups,
# same pinned 12-task val, same B1). control/lesson r2 results are reused from
# that experiment's summary; only the two NEW conditions are run here.
#
# New step: merge_validate_skills.py merges each group's train skills into one
# type-level skill, then validates (score; <3 drop, 3-4 rewrite, >=4 keep).
# Seeds:
#   merge-replace -> merged skill replaces the r1 val online skill (trained types)
#   merge-stack   -> merged skill appended to the r1 val online skill (trained types)
set -euo pipefail

SKILLRISE_ROOT=/data/lanyuqi/skillrise
cd "$SKILLRISE_ROOT"
PY=./runtime/venv/bin/python
OLD_EXP="$SKILLRISE_ROOT/runtime/outputs/exp_mistakes_vs_lessons"
EXP_ROOT="$SKILLRISE_ROOT/runtime/outputs/exp_merge_validate"
mkdir -p "$EXP_ROOT"

TS=$(date +%Y%m%dT%H%M%S)
LOG="$EXP_ROOT/experiment_${TS}.log"
SUMMARY="$EXP_ROOT/summary_${TS}.tsv"
exec > >(tee -a "$LOG") 2>&1
echo "=== merge+validate experiment started $(date) ==="

VAL_TASKS="$SKILLRISE_ROOT/runtime/data/exp_mistakes_val12.json"
MERGED="$EXP_ROOT/merged_skills.json"
ROLLOUT_N=3
CURATE_FLAG="--curate-via-api"
declare -a TRAIN_GROUPS=(
  "measure-melting-point-unknown-substance_K3_0"
  "find-plant_K3_0"
  "inclined-plane-friction-named-surfaces_K3_0"
)
declare -a CONDS=(merge-replace merge-stack)

# r1 dirs from the shared experiment's state
declare -a R1_DIRS=()
for gid in "${TRAIN_GROUPS[@]}"; do
    r1=$(awk -F '\t' -v g="$gid" '$1==g {print $2}' "$OLD_EXP/r1_state.tsv" | head -1)
    if [ -z "$r1" ] || [ ! -f "$r1/manifest.json" ]; then
        echo "[fatal] missing shared r1 for $gid"; exit 1
    fi
    R1_DIRS+=("$r1")
    echo "[r1] reuse $gid -> $r1"
done

# ---- merge + validate the train skills ----
if [ -f "$MERGED" ]; then
    echo "[merge] reusing $MERGED"
else
    $PY runtime/merge_validate_skills.py \
        --data-dirs "$(IFS=,; echo "${R1_DIRS[*]}")" \
        --out "$MERGED" || { echo "[fatal] merge_validate_skills failed"; exit 1; }
fi

# ---- round 2 for the two new conditions ----
run_r2() { # $1=label; $2=gid; $3=seed -> echoes r2 dir
    local label=$1; local gid=$2; local seed=$3
    local key="$label|$gid"
    local existing=""
    if [ -f "$SUMMARY" ]; then
        existing=$(awk -F '|' -v l="$label" -v g="$gid" '$1==l && $2==g {print $3}' "$SUMMARY" | head -1)
    fi
    if [ -n "$existing" ] && [ -f "$existing/manifest.json" ]; then
        echo "[r2] reusing $key -> $existing" >&2
        echo "$existing"; return
    fi
    echo "[r2] running $label $gid" >&2
    local outdir
    outdir=$(bash runtime/run_pure_rollout.sh \
        --group-id "$gid" --val-tasks "$VAL_TASKS" --rollout-n "$ROLLOUT_N" \
        $CURATE_FLAG --seed-file "$seed" 2>&1 | tail -1)
    if [ ! -f "$outdir/manifest.json" ]; then
        echo "[fatal] r2 run produced no manifest: $outdir" >&2
        exit 1
    fi
    echo "[r2] $key -> $outdir" >&2
    echo "$key|$outdir" >> "$SUMMARY"
    echo "$outdir"
}

for cond in "${CONDS[@]}"; do
    for i in "${!TRAIN_GROUPS[@]}"; do
        gid="${TRAIN_GROUPS[$i]}"
        r1="${R1_DIRS[$i]}"
        seed="$EXP_ROOT/seed_${cond}_${gid##*_}.json"
        $PY runtime/build_round2_seed.py --data-dir "$r1" \
            --merged-skills-file "$MERGED" --merge-mode "${cond#merge-}" \
            --val-tasks-file "$VAL_TASKS" --out "$seed"
        run_r2 "$cond" "$gid" "$seed" >/dev/null
    done
done

echo "=== all done $(date) ==="
