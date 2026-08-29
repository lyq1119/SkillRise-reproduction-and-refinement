# Validation record

## Passed

- Repository is detached at `2b38da3ad7b64414305d39609484803b07d3b9b0`.
- Eight GPUs are visible and all are NVIDIA GeForce RTX 4090 (24,564 MiB).
- NVIDIA driver is `580.173.02`.
- Python is CPython `3.11.15` in `runtime/venv`.
- Java is Eclipse Temurin JRE `11.0.32.1+1` in `runtime/java`.
- ScienceWorld `1.2.3` launched its JVM, loaded `boil` variation 3 with the
  `easy` simplification, reset successfully, and returned 173 valid actions.
- BEACON `L0_idx.json` and official `skillrise_sciworld_K3.jsonl` exist and
  their hashes are recorded in `README.md`.
- Torch `2.13.0+cu132` sees an RTX 4090 with compute capability 8.9.
- vLLM, Transformers, Ray, W&B, ScienceWorld, and the SkillRise ScienceWorld
  environment manager import successfully.
- W&B has an existing API credential in the server's current user context.

## Blocking failure

The Qwen3.5 model card requires vLLM from main/nightly. The pinned nightly used
for this reproduction is `0.28.1rc1.dev50+g94a54f581`. SkillRise's vendored veRL
rollout integration imports the removed legacy module:

```text
verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:29:
from vllm.worker.worker_base import WorkerWrapperBase

ModuleNotFoundError: No module named 'vllm.worker'
```

This vLLM build only contains the new worker package under `vllm.v1.worker`.
FSDP worker imports pass, but rollout worker initialization cannot start. Per the
reproduction constraint, no SkillRise or vendored veRL source was changed to
adapt the interface. Consequently the vLLM one-shot generation through veRL,
8-GPU rollout/FSDP initialization, smoke run, and formal run are not executed.

## Resource gates

- At audit time `/data` had about 105 GiB free and was 99% full.
- GPU 1 was occupied by an unrelated process (about 20 GiB, 80-100% compute),
  so the required all-eight-GPUs-idle gate was not satisfied.
- A 9B Adam/FSDP training checkpoint can consume a substantial fraction of the
remaining storage. Keeping checkpoints every five steps for 150 epochs is not
safe under the observed free-space budget without a retention/archive plan.
