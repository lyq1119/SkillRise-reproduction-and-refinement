#!/usr/bin/env bash
# 2-round experiment: 4 conditions x (round1 -> lessons? -> seed -> round2).
# Conditions: baseline / baseline+lesson / B1 / B1+lesson.
# Primary metric: round-2 val pass@k.
#
# Resumable: round-1 dirs are recorded in two_round_r1_state.tsv
# (cond<TAB>dir). If a condition's r1 already exists there (with manifest),
# it is reused instead of re-run.
set -euo pipefail

SKILLRISE_ROOT=/data/lanyuqi/skillrise
cd "$SKILLRISE_ROOT"
PY=./runtime/venv/bin/python

TS=$(date +%Y%m%dT%H%M%S)
LOG="$SKILLRISE_ROOT/runtime/outputs/two_round_experiment_${TS}.log"
STATE="$SKILLRISE_ROOT/runtime/outputs/two_round_r1_state.tsv"
SUMMARY="$SKILLRISE_ROOT/runtime/outputs/two_round_summary_${TS}.tsv"
exec > >(tee -a "$LOG") 2>&1
echo "=== 2-round experiment (resume-aware) started $(date) ==="
echo "log: $LOG  state: $STATE"

get_r1() { # $1=cond -> echoes stored r1 dir or empty
    awk -F '\t' -v c="$1" '$1==c {print $2}' "$STATE" 2>/dev/null | head -1
}
set_r1() { # $1=cond $2=dir
    grep -v -P "^$1\t" "$STATE" 2>/dev/null > "$STATE.tmp" || true
    printf '%s\t%s\n' "$1" "$2" >> "$STATE.tmp"
    mv "$STATE.tmp" "$STATE"
}

run_round() {
    # $1 = label; remaining = extra flags for run_pure_rollout.sh
    # prints ONLY the output dir on stdout (info lines go to stderr)
    local label=$1; shift
    local outdir
    outdir=$(bash runtime/run_pure_rollout.sh "$@" 2>&1 | tail -1)
    echo "[$label] round output: $outdir" >&2
    echo "$outdir"
}

condition() {
    # $1 = condition name; $2 = curate flag; $3 = with_lesson (0/1)
    local cond=$1; local cflag=$2; local with_lesson=$3
    echo "########## CONDITION: $cond ##########"

    local r1
    r1=$(get_r1 "$cond")
    if [ -n "$r1" ] && [ -f "$r1/manifest.json" ]; then
        echo "[$cond] reusing stored r1: $r1"
    else
        echo "[$cond] running round 1"
        r1=$(run_round "$cond r1" $cflag)
        set_r1 "$cond" "$r1"
    fi
    echo "[$cond] r1 = $r1"

    local train_lessons="" val_lessons=""
    if [ "$with_lesson" = "1" ]; then
        echo "[$cond] extracting lessons (train + val)"
        $PY runtime/opid_lessons.py --data-dir "$r1" --split train
        $PY runtime/opid_lessons.py --data-dir "$r1" --split val
        train_lessons="$r1/opid_lessons_train.jsonl"
        val_lessons="$r1/opid_lessons_val.jsonl"
    fi

    local seed="$SKILLRISE_ROOT/runtime/outputs/seed_${cond}.json"
    if [ "$with_lesson" = "1" ]; then
        $PY runtime/build_round2_seed.py --data-dir "$r1" \
            --lessons "$train_lessons" --val-lessons "$val_lessons" \
            --out "$seed"
    else
        $PY runtime/build_round2_seed.py --data-dir "$r1" --out "$seed"
    fi
    echo "[$cond] seed = $seed"

    local r2
    r2=$(run_round "$cond r2" $cflag --seed-file "$seed")
    echo "[$cond] r2 = $r2"

    echo "[$cond] round-2 eval (val pass@k):"
    $PY -c "
import json
d = json.load(open('$r2/manifest.json'))
m = d['eval']['metrics']
print(f'  pass@1={sum(m[\"pass@1\"])}/8  pass@2={sum(m[\"pass@2\"])}/8  pass@3={sum(m[\"pass@3\"])}/8  wall={round(d[\"wall_time_seconds\"])}s')
" || echo "  (manifest missing or unreadable)"
    echo "$cond|$r1|$r2" >> "$SUMMARY"
}

condition baseline_x2 "" 0
condition baseline_lesson_x2 "" 1
condition b1_x2 "--curate-via-api" 0
condition b1_lesson_x2 "--curate-via-api" 1

echo "=== all done $(date) ==="
echo "summary: $SUMMARY"
