#!/usr/bin/env bash
# EX: pause the evolution driver right after B round1 completes (3 groups),
# before B round2 starts. Kills the driver and any pure_rollout child so the
# batch stops cleanly; B round2/3 resume later (resumable state files).
set -u

EVO_EXP=/data/lanyuqi/skillrise/runtime/outputs/exp_skill_evolution
STATE="$EVO_EXP/B_r1_state.tsv"
LOG="$EVO_EXP/pause_watcher.log"

while [ "$(wc -l < "$STATE" 2>/dev/null || echo 0)" -lt 3 ]; do
    sleep 10
done

echo "$(date) B round1 complete (3 groups) — pausing driver" >> "$LOG"
pkill -f "run_evolution_experiments.sh" 2>/dev/null || true
sleep 2
pkill -f "runtime/pure_rollout.py" 2>/dev/null || true
echo "$(date) driver paused" >> "$LOG"
