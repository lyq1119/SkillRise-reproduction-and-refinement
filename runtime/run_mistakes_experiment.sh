#!/usr/bin/env bash
# EX: mistakes-vs-lessons experiment (finalized design).
#
# 3 conditions x 2 rounds on a fixed 12-task val, 3 train groups (types):
#   - r1 is SHARED across conditions (identical B1 rollout + curate).
#   - conditions diverge only in what failure knowledge rides into r2:
#       control  -> skills only
#       lesson   -> skills + per-trajectory OPID lessons
#       mistakes -> skills + type-level common-mistakes bank
#
# Train: 3 groups x 3 rollout samples (--rollout-n 3).
# Val:   12 pinned tasks (exp_mistakes_val12.json).
# Metric: round-2 val pass@k per condition (mean over the 3 groups).
#
# Resumable: r1/r2 dirs are recorded in a state file; completed runs are skipped.
set -euo pipefail

SKILLRISE_ROOT=/data/lanyuqi/skillrise
cd "$SKILLRISE_ROOT"
PY=./runtime/venv/bin/python
EXP_ROOT="$SKILLRISE_ROOT/runtime/outputs/exp_mistakes_vs_lessons"
mkdir -p "$EXP_ROOT"

TS=$(date +%Y%m%dT%H%M%S)
LOG="$EXP_ROOT/experiment_${TS}.log"
STATE="$EXP_ROOT/r1_state.tsv"
SUMMARY="$EXP_ROOT/summary_${TS}.tsv"
exec > >(tee -a "$LOG") 2>&1
echo "=== mistakes-vs-lessons experiment started $(date) ==="
echo "log: $LOG  state: $STATE"

VAL_TASKS="$SKILLRISE_ROOT/runtime/data/exp_mistakes_val12.json"
BANK="$EXP_ROOT/mistakes_bank.json"
ROLLOUT_N=3
CURATE_FLAG="--curate-via-api"
# train groups (one per type), pinned and non-overlapping with the val list
# NOTE: do NOT name this array GROUPS — GROUPS is a bash readonly builtin
# (supplementary GIDs), so assigning it silently fails.
declare -a TRAIN_GROUPS=(
  "measure-melting-point-unknown-substance_K3_0"
  "find-plant_K3_0"
  "inclined-plane-friction-named-surfaces_K3_0"
)
declare -a CONDS=(control lesson mistakes)

run_r1() { # $1=gid  -> echoes r1 dir
    local gid=$1
    local existing=""
    if [ -f "$STATE" ]; then
        existing=$(awk -F '\t' -v g="$gid" '$1==g {print $2}' "$STATE" | head -1)
    fi
    if [ -n "$existing" ] && [ -f "$existing/manifest.json" ]; then
        echo "[r1] reusing $gid -> $existing" >&2
        echo "$existing"; return
    fi
    echo "[r1] running $gid" >&2
    local outdir
    outdir=$(bash runtime/run_pure_rollout.sh \
        --group-id "$gid" --val-tasks "$VAL_TASKS" --rollout-n "$ROLLOUT_N" $CURATE_FLAG 2>&1 | tail -1)
    if [ ! -f "$outdir/manifest.json" ]; then
        echo "[fatal] r1 run produced no manifest: $outdir" >&2
        exit 1
    fi
    echo "[r1] $gid -> $outdir" >&2
    grep -v -P "^$gid\t" "$STATE" 2>/dev/null > "$STATE.tmp" || true
    printf '%s\t%s\n' "$gid" "$outdir" >> "$STATE.tmp"
    mv "$STATE.tmp" "$STATE"
    echo "$outdir"
}

run_r2() { # $1=label; $2=gid; $3=seed-file -> echoes r2 dir
    local label=$1; local gid=$2; local seed=$3
    local key="$label|$gid"
    local existing=""
    if [ -f "$SUMMARY" ]; then
        existing=$(awk -F '\t' -v k="$key" '$1==k {print $2}' "$SUMMARY" | head -1)
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

# ---- shared round 1 ----
declare -a R1_DIRS=()
for gid in "${TRAIN_GROUPS[@]}"; do
    R1_DIRS+=("$(run_r1 "$gid")")
done

# ---- build failure knowledge from the shared r1 ----
echo "=== building failure knowledge ==="
if [ -f "$BANK" ]; then
    echo "[bank] reusing existing $BANK"
else
    $PY runtime/build_mistakes_bank.py \
        --data-dirs "$(IFS=,; echo "${R1_DIRS[*]}")" \
        --out "$BANK" || { echo "[fatal] build_mistakes_bank failed"; exit 1; }
fi

# lessons are per-group; extract for each r1 dir
declare -a LESSONS_TRAIN=()
declare -a LESSONS_VAL=()
for d in "${R1_DIRS[@]}"; do
    $PY runtime/opid_lessons.py --data-dir "$d" --split train || exit 1
    $PY runtime/opid_lessons.py --data-dir "$d" --split val || exit 1
    LESSONS_TRAIN+=("$d/opid_lessons_train.jsonl")
    LESSONS_VAL+=("$d/opid_lessons_val.jsonl")
done

# ---- round 2 per condition per group ----
for cond in "${CONDS[@]}"; do
    for i in "${!TRAIN_GROUPS[@]}"; do
        gid="${TRAIN_GROUPS[$i]}"
        r1="${R1_DIRS[$i]}"
        seed="$EXP_ROOT/seed_${cond}_${gid##*_}.json"
        case "$cond" in
            control)
                $PY runtime/build_round2_seed.py --data-dir "$r1" --out "$seed" ;;
            lesson)
                $PY runtime/build_round2_seed.py --data-dir "$r1" \
                    --lessons "${LESSONS_TRAIN[$i]}" --val-lessons "${LESSONS_VAL[$i]}" \
                    --out "$seed" ;;
            mistakes)
                $PY runtime/build_round2_seed.py --data-dir "$r1" \
                    --mistakes-bank "$BANK" --val-tasks-file "$VAL_TASKS" \
                    --out "$seed" ;;
        esac
        run_r2 "$cond" "$gid" "$seed" >/dev/null
    done
done

# ---- aggregate r2 val pass@k per condition ----
echo "=== round-2 val pass@k per condition (mean over 3 groups) ==="
for cond in "${CONDS[@]}"; do
    for i in "${!TRAIN_GROUPS[@]}"; do
        gid="${TRAIN_GROUPS[$i]}"
        r2=$(awk -F '\t' -v k="$cond|$gid" '$1==k {print $2}' "$SUMMARY" | head -1)
        echo "[$cond/$gid] $(basename "$r2")"
        $PY -c "
import json
d=json.load(open('$r2/manifest.json'))
m=d['eval']['metrics']
print(f'   pass@1={sum(m[\"pass@1\"])}/{len(m[\"pass@1\"])} pass@2={sum(m[\"pass@2\"])}/{len(m[\"pass@2\"])} pass@3={sum(m[\"pass@3\"])}/{len(m[\"pass@3\"])}')"
    done
done

echo "=== all done $(date) ==="
