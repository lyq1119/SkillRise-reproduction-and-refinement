# SkillRise

**SkillRise** is a cross-task meta-RL method for training LLM agents. Instead of
training on tasks independently, SkillRise groups **K related tasks** together and
has a single shared policy play them as an ordered sequence, distilling a reusable
**skill document** in-context that carries knowledge from one task to the next:

```
Solve(task_0) → Curate(0) → Solve(task_1) → Curate(1) → … → Solve(task_{K-1})
```

The policy plays two roles with the same weights:

- **SOLVE** — read the current skill document plus the environment observation,
  reason, and emit an `<action>`.
- **CURATE** — after finishing a task, read that task's trajectory and the old
  skill document, and rewrite an improved, task-agnostic skill document inside
  `<skill>…</skill>`.

The skill document evolves in-context across the K tasks of a group (it is not
persisted to disk during training; each group starts from an empty document). `N`
parallel trials replay the same group so that advantages can be normalized
group-relative across trials.

SkillRise is implemented on top of [verl](https://github.com/volcengine/verl) /
[verl-agent (GiGPO)](https://github.com/langfengQ/verl-agent) and is evaluated on
three text agent environments: **ALFWorld**, **WebShop**, and **ScienceWorld**.

## Method implementation

The SkillRise-specific code lives in:

| Component | Path |
|---|---|
| Environment managers (K-task groups, SOLVE/CURATE roles, skill evolution) | `agent_system/environments/skillrise_{alfworld,webshop,sciworld}/` |
| Group loader (serves groups of K related tasks) | `agent_system/environments/skillrise_*/group_loader.py` |
| Meta-RL rollout loop + credit assignment | `agent_system/multi_turn_rollout/skillrise_rollout_loop.py` |
| SkillRise advantage estimator | `verl/trainer/ppo/core_gigpo.py` (`compute_skillrise_outcome_advantage`) |
| Trainer wiring (`AdvantageEstimator.SKILLRISE`) | `verl/trainer/ppo/ray_trainer.py`, `verl/trainer/main_ppo.py` |
| Reward manager | `agent_system/reward_manager/episode.py` |

The task-sequence groups (K=3) for each environment are bundled under
`data/groups/`:

- `data/groups/skillrise_alfworld_K3.jsonl`
- `data/groups/skillrise_webshop_K3.jsonl`
- `data/groups/skillrise_sciworld_K3.jsonl`

Each line is one group of K related tasks (same task family).

## Installation

```bash
# Python 3.10, CUDA GPU(s) recommended.
pip install -r requirements.txt

# verl is vendored in this repo (verl/), no separate install needed.
```

Per-environment backends are installed by the run scripts (or manually):

- **ALFWorld**: `pip install alfworld`, then download game data with
  `alfworld-download` and set `ALFWORLD_DATA` to point at it.
- **WebShop**: needs Java 11 + `gym==0.26.2` + a spaCy model (the WebShop backend
  is vendored under `agent_system/environments/webshop/webshop/`). See
  `examples/skillrise_webshop/env.sh`.
- **ScienceWorld**: `pip install scienceworld gym` and a JDK 11 (`JAVA_HOME`).

## Running

Each environment has its own example directory with an `env.sh` (paths / keys) and
run scripts. Edit `env.sh` first to set `SKILLRISE_MODEL_PATH` (a local HF
checkpoint, e.g. Qwen3-4B), the data paths, and your `WANDB_API_KEY` (or
`export WANDB_MODE=offline`).

```bash
# Full 8-GPU SkillRise training
bash examples/skillrise_alfworld/skillrise_alfworld_qwen3_4b.sh
bash examples/skillrise_webshop/skillrise_webshop_qwen3_4b.sh
bash examples/skillrise_sciworld/skillrise_sciworld_qwen3_4b.sh

# Single-GPU smoke test (2 steps)
bash examples/skillrise_alfworld/debug/debug_skillrise_k3.sh
```

A GRPO baseline script (`grpo_<env>_qwen3_4b.sh`) is included in each directory for
comparison. The key SkillRise switches in the run scripts are:

```
algorithm.adv_estimator=skillrise
env.env_name="skillrise_alfworld/AlfredTWEnv"
+env.meta_mode=skillrise
+env.group_file="…/data/groups/skillrise_alfworld_K3.jsonl"
env.num_attempts=3          # K tasks per group
env.rollout.n=8             # N trials per group
```

## Attribution & license

This repository is a fork/derivative of verl and verl-agent (GiGPO), both licensed
under the Apache License 2.0. SkillRise is released under the same license — see
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
