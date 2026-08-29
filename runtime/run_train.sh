#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/lanyuqi/skillrise
source "$ROOT/runtime/run_env.sh"
cd "$ROOT"

# No Hydra overrides: every algorithm/data/prompt/reward/training value comes
# directly from the pinned official script.
exec bash examples/skillrise_sciworld/skillrise_sciworld_qwen3_4b.sh
