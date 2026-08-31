#!/usr/bin/env bash
# EX: skill-evolution experiments — A (merge-replace base, 3 rounds) and
#     B (merge-replace + lesson base, 3 rounds). Resumable per round.
set -euo pipefail

SKILLRISE_ROOT=/data/lanyuqi/skillrise
cd "$SKILLRISE_ROOT"
PY=./runtime/venv/bin/python
OLD_EXP="$SKILLRISE_ROOT/runtime/outputs/exp_mistakes_vs_lessons"
EVO_EXP="$SKILLRISE_ROOT/runtime/outputs/exp_skill_evolution"
mkdir -p "$EVO_EXP"

VAL_TASKS="$SKILLRISE_ROOT/runtime/data/exp_mistakes_val12.json"
MERGED="$SKILLRISE_ROOT/runtime/outputs/exp_merge_validate/merged_skills.json"
ROLLOUT_N=3
CURATE_FLAG="--curate-via-api"

TS=$(date +%Y%m%dT%H%M%S)
LOG="$EVO_EXP/evolution_experiments_${TS}.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== evolution experiments (A round3 + B round1/2/3) started $(date) ==="

declare -a GIDS=(
  "measure-melting-point-unknown-substance_K3_0"
  "find-plant_K3_0"
  "inclined-plane-friction-named-surfaces_K3_0"
)
declare -a SHORT=(melting findplant inclined)
declare -a R1=()
for g in "${GIDS[@]}"; do
    r1=$(awk -F '\t' -v g="$g" '$1==g {print $2}' "$OLD_EXP/r1_state.tsv" | head -1)
    [ -f "$r1/manifest.json" ] || { echo "[fatal] missing r1 for $g"; exit 1; }
    R1+=("$r1")
done

run_group() { # $1=state $2=gid $3=seed
    local state=$1; local gid=$2; local seed=$3
    local existing=""
    [ -f "$state" ] && existing=$(awk -F '|' -v g="$gid" '$1==g {print $2}' "$state" | head -1)
    if [ -n "$existing" ] && [ -f "$existing/manifest.json" ]; then
        echo "[run] reuse $gid -> $existing"; return
    fi
    echo "[run] running $gid"
    local outdir
    outdir=$(bash runtime/run_pure_rollout.sh --group-id "$gid" --val-tasks "$VAL_TASKS" \
        --rollout-n "$ROLLOUT_N" $CURATE_FLAG --seed-file "$seed" \
        --val-splits "${VAL_SPLITS:-2}" 2>&1 | tail -1)
    [ -f "$outdir/manifest.json" ] || { echo "[fatal] no manifest: $outdir"; exit 1; }
    echo "$gid|$outdir" >> "$state"
}

build_seed() { # $1=r1 $2=merged_file $3=with_lesson(0/1) $4=out
    local r1=$1; local mf=$2; local wl=$3; local out=$4
    local base="--data-dir $r1 --merged-skills-file $mf --merge-mode replace --val-tasks-file $VAL_TASKS"
    if [ "$wl" = "1" ]; then
        $PY runtime/build_round2_seed.py $base \
            --lessons "$r1/opid_lessons_train.jsonl" --val-lessons "$r1/opid_lessons_val.jsonl" \
            --out "$out"
    else
        $PY runtime/build_round2_seed.py $base --out "$out"
    fi
}

# ============ A: round3 (evolve round2's skills once more) ============
echo "### A round3"
A_R2_STATE="$EVO_EXP/summary_round_2.tsv"
A_R2_EVOLVED="$EVO_EXP/evolved_skills.json"
A_R3_EVOLVED="$EVO_EXP/evolved_skills_r3.json"
if [ ! -f "$A_R3_EVOLVED" ]; then
    A_R2_DIRS=$(cut -d'|' -f2 "$A_R2_STATE" | paste -sd, -)
    $PY runtime/evolve_skills.py --r2-runs "$A_R2_DIRS" \
        --merged-skills "$A_R2_EVOLVED" --out "$A_R3_EVOLVED" || exit 1
fi
A_R3_STATE="$EVO_EXP/A_r3_state.tsv"
for i in 0 1 2; do
    build_seed "${R1[$i]}" "$A_R3_EVOLVED" 0 "$EVO_EXP/seed_A_r3_${SHORT[$i]}.json"
    run_group "$A_R3_STATE" "${GIDS[$i]}" "$EVO_EXP/seed_A_r3_${SHORT[$i]}.json"
done

# ============ B: round1 (merge-replace + lesson baseline) ============
echo "### B round1 (merge-replace + lesson)"
B_R1_STATE="$EVO_EXP/B_r1_state.tsv"
for i in 0 1 2; do
    build_seed "${R1[$i]}" "$MERGED" 1 "$EVO_EXP/seed_B_r1_${SHORT[$i]}.json"
    run_group "$B_R1_STATE" "${GIDS[$i]}" "$EVO_EXP/seed_B_r1_${SHORT[$i]}.json"
done

# ============ B: round2 (evolve from B round1) ============
echo "### B round2"
B_R2_EVOLVED="$EVO_EXP/B_r2_evolved.json"
if [ ! -f "$B_R2_EVOLVED" ]; then
    B_R1_DIRS=$(cut -d'|' -f2 "$B_R1_STATE" | paste -sd, -)
    $PY runtime/evolve_skills.py --r2-runs "$B_R1_DIRS" \
        --merged-skills "$MERGED" --out "$B_R2_EVOLVED" || exit 1
fi
B_R2_STATE="$EVO_EXP/B_r2_state.tsv"
for i in 0 1 2; do
    build_seed "${R1[$i]}" "$B_R2_EVOLVED" 1 "$EVO_EXP/seed_B_r2_${SHORT[$i]}.json"
    run_group "$B_R2_STATE" "${GIDS[$i]}" "$EVO_EXP/seed_B_r2_${SHORT[$i]}.json"
done

# ============ B: round3 (evolve from B round2) ============
echo "### B round3"
B_R3_EVOLVED="$EVO_EXP/B_r3_evolved.json"
if [ ! -f "$B_R3_EVOLVED" ]; then
    B_R2_DIRS=$(cut -d'|' -f2 "$B_R2_STATE" | paste -sd, -)
    $PY runtime/evolve_skills.py --r2-runs "$B_R2_DIRS" \
        --merged-skills "$B_R2_EVOLVED" --out "$B_R3_EVOLVED" || exit 1
fi
B_R3_STATE="$EVO_EXP/B_r3_state.tsv"
for i in 0 1 2; do
    build_seed "${R1[$i]}" "$B_R3_EVOLVED" 1 "$EVO_EXP/seed_B_r3_${SHORT[$i]}.json"
    run_group "$B_R3_STATE" "${GIDS[$i]}" "$EVO_EXP/seed_B_r3_${SHORT[$i]}.json"
done

echo "=== all done $(date) ==="
