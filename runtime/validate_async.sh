#!/usr/bin/env bash
# EX: validate the async val pipeline — run 1 group with --val-splits 2 and
# sample GPU utilization every 3s to confirm the utilization pattern improves
# (no single-card-100%-others-idle; cards stay busy).
set -euo pipefail
SKILLRISE_ROOT=/data/lanyuqi/skillrise
cd "$SKILLRISE_ROOT"

GID=${1:-measure-melting-point-unknown-substance_K3_0}
SEED=${2:-}
OUT="$SKILLRISE_ROOT/runtime/outputs/exp_async_validate"
mkdir -p "$OUT"
TS=$(date +%Y%m%dT%H%M%S)

# GPU sampler in background: log utilization every 3s
( for i in $(seq 1 1200); do
    nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | tr '\n' ' ' >> "$OUT/gpu_util_${TS}.log"
    echo >> "$OUT/gpu_util_${TS}.log"
    sleep 3
  done ) &
SAMPLER_PID=$!
trap 'kill $SAMPLER_PID 2>/dev/null || true' EXIT

echo "=== async validation run started $(date) — gid=$GID val_splits=2 ==="
ARGS="--group-id $GID --val-tasks runtime/data/exp_mistakes_val12.json --rollout-n 3 --curate-via-api --val-splits 2"
[ -n "$SEED" ] && ARGS="$ARGS --seed-file $SEED"
OUTDIR=$(bash runtime/run_pure_rollout.sh $ARGS 2>&1 | tail -1)
echo "=== done $(date) — outdir=$OUTDIR ==="
echo "$OUTDIR" > "$OUT/last_run_${TS}.txt"

echo "=== GPU utilization stats ==="
awk -F', ' '{s=0; for(i=1;i<=NF;i++) s+=$i; n=NF; printf "avg=%.0f%% max=%.0f%% idle_count=%d/4 ", s/n, 0, 0}' "$OUT/gpu_util_${TS}.log" >/dev/null
python3 - "$OUT/gpu_util_${TS}.log" <<'EOF'
import sys
lines=[l.strip() for l in open(sys.argv[1]) if l.strip()]
import statistics
samples=[]
for l in lines:
    vals=[int(x.split()[0]) for x in l.split(',') if x.strip()]
    if len(vals)==8: samples.append(vals)
# we use GPUs 0-3
n=len(samples)
busy4=sum(1 for v in samples if all(x>=50 for x in v[:4]))/max(n,1)*100
allidle=sum(1 for v in samples if all(x<5 for x in v[:4]))/max(n,1)*100
avg=statistics.mean(sum(v[:4]) for v in samples) if samples else 0
maxb=statistics.mean(max(v[:4]) for v in samples) if samples else 0
print(f"samples={n}  avg(4 cards)={avg:.0f}%  all-4-idle={allidle:.0f}%  all-4-busy={busy4:.0f}%  avg-max-card={maxb:.0f}%")
EOF
