# SkillRise

Large language model agents are increasingly deployed to solve complex,
long-horizon tasks. In practice, they often encounter streams of related yet
distinct tasks that share underlying regularities and reusable solution patterns.
However, standard agentic RL typically treats each task as an independent episode,
discarding the experience acquired during interaction and forcing the agent to
repeatedly explore from scratch. An ideal agent should not only solve the task at
hand, but also extract transferable skills from experience and continually reuse
and refine them across subsequent tasks, becoming increasingly capable over time.

**SkillRise** is an end-to-end reinforcement learning framework for cross-task
skill learning. SkillRise first constructs progressively challenging task
sequences by selecting similar yet distinct instances from the same task family
and ordering them by difficulty, so that experience from earlier tasks can support
later ones. During each rollout, a single policy alternates between **solving** the
current task with an evolving skill document and **curating** the document based on
the resulting trajectory before proceeding to the next task instance. SkillRise
employs decoupled cross-task credit assignment that assigns the current task reward
to task solving and a discounted return over subsequent task rewards to skill
curation. Group-relative advantages are then computed over trials sharing the same
task group, sequence stage, and behavioral phase. Together, these designs enable
the policy to continually extract, refine, and reuse transferable skills while
solving a sequence of tasks.

We evaluate SkillRise on **ALFWorld**, **WebShop**, and **ScienceWorld**. Across
all three agentic benchmarks, SkillRise consistently outperforms prompting-based
methods and standard reinforcement learning baselines, and exhibits cross-task
test-time scaling — improving as it encounters longer sequences of related tasks
at test time and progressively refines its skill document.

## Method implementation

SkillRise is built on top of [verl](https://github.com/volcengine/verl) /
[verl-agent (GiGPO)](https://github.com/langfengQ/verl-agent). The
SkillRise-specific code is:

| Component | Path |
|---|---|
| Environment managers (K-task sequences, solve/curate roles, evolving skill document) | `agent_system/environments/skillrise_{alfworld,webshop,sciworld}/` |
| Task-sequence group loader | `agent_system/environments/skillrise_*/group_loader.py` |
| Meta-RL rollout loop + decoupled cross-task credit assignment | `agent_system/multi_turn_rollout/skillrise_rollout_loop.py` |
| SkillRise advantage estimator (role-aware group-relative) | `verl/trainer/ppo/core_gigpo.py` (`compute_skillrise_outcome_advantage`) |
| Trainer wiring (`AdvantageEstimator.SKILLRISE`) | `verl/trainer/ppo/ray_trainer.py`, `verl/trainer/main_ppo.py` |

The pre-built task sequences (K=3 tasks per group) are bundled under `data/groups/`:
`skillrise_alfworld_K3.jsonl`, `skillrise_webshop_K3.jsonl`,
`skillrise_sciworld_K3.jsonl`. Each line is one group of K related tasks from the
same task family.

## Setup

The runtime environment follows **verl-agent**. Please set up the base framework
and the three environment backends by following the verl-agent installation guide:
https://github.com/langfengQ/verl-agent

In short:

```bash
# Base framework (Python 3.10, CUDA GPU).
pip install -r requirements.txt
# verl is vendored in this repo (verl/), so no separate verl install is needed.
```

Then install each environment backend as described by verl-agent:

- **ALFWorld**: `pip install alfworld`, then download the game files
  (`alfworld-download`) and set `ALFWORLD_DATA` to their location.
- **WebShop**: Java 11 + `gym==0.26.2` + a spaCy model (the WebShop backend is
  vendored under `agent_system/environments/webshop/`).
- **ScienceWorld**: `pip install scienceworld gym` and a JDK 11 (`JAVA_HOME`).

## Running

Each environment has its own directory under `examples/` with an `env.sh` (paths
and keys) and the run scripts. Edit `env.sh` first to set `SKILLRISE_MODEL_PATH` (a
local HF checkpoint, e.g. Qwen3-4B), the data paths, and your `WANDB_API_KEY` (or
`export WANDB_MODE=offline`).

```bash
# SkillRise training (8-GPU config)
bash examples/skillrise_alfworld/skillrise_alfworld_qwen3_4b.sh
bash examples/skillrise_webshop/skillrise_webshop_qwen3_4b.sh
bash examples/skillrise_sciworld/skillrise_sciworld_qwen3_4b.sh

# GRPO baseline (same env, for comparison)
bash examples/skillrise_alfworld/grpo_alfworld_qwen3_4b.sh
```

The key SkillRise switches in the run scripts are:

```
algorithm.adv_estimator=skillrise
env.env_name="skillrise_alfworld/AlfredTWEnv"
+env.meta_mode=skillrise
+env.group_file=".../data/groups/skillrise_alfworld_K3.jsonl"
env.num_attempts=3          # K tasks per group
env.rollout.n=8             # N trials per group
```

## License

Released under the Apache License 2.0. This repository is a derivative of verl and
verl-agent (GiGPO); see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
