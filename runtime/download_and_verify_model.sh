#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/lanyuqi/skillrise
source "$ROOT/runtime/run_env.sh"
export HF_HUB_DISABLE_XET=1
cd "$ROOT"

python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download(
    repo_id="Qwen/Qwen3.5-9B",
    revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    local_dir="/data/lanyuqi/skillrise/runtime/models/Qwen3.5-9B",
    max_workers=4,
))
PY

sha256sum runtime/models/Qwen3.5-9B/* | tee runtime/logs/qwen3.5-9b.sha256

python - <<'PY' | tee runtime/logs/qwen3.5-9b-tokenizer-validation.log
from transformers import AutoConfig, AutoTokenizer

path = "/data/lanyuqi/skillrise/runtime/models/Qwen3.5-9B"
config = AutoConfig.from_pretrained(path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
rendered = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Reply with OK."}],
    tokenize=False,
    add_generation_prompt=True,
)
print("model_type", config.model_type)
print("tokenizer_class", type(tokenizer).__name__)
print("vocab_size", len(tokenizer))
print("chat_template_ok", bool(rendered.strip()))
PY

echo "MODEL_DOWNLOAD_AND_VERIFY_COMPLETE"
