# SkillRise

**SkillRise** is an end-to-end reinforcement learning framework for cross-task
skill learning. A single policy plays an ordered sequence of related tasks,
alternating between **solving** the current task with an evolving skill document
and **curating** that document from the resulting trajectory before moving to the
next task. Evaluated on ALFWorld, WebShop, and ScienceWorld.

Built on top of [verl](https://github.com/volcengine/verl) /
[verl-agent (GiGPO)](https://github.com/langfengQ/verl-agent).

## Setup

Follow the **verl-agent** guide to install the framework and environment backends:
https://github.com/langfengQ/verl-agent

```bash
pip install -r requirements.txt   # verl is vendored in verl/, no separate install
```

Then set up the backends: ALFWorld (`pip install alfworld` + `alfworld-download`,
set `ALFWORLD_DATA`), WebShop (Java 11 + `gym==0.26.2` + spaCy model),
ScienceWorld (`pip install scienceworld gym` + JDK 11).

## Running

Edit `examples/<env>/env.sh` (set `SKILLRISE_MODEL_PATH`, data paths, `WANDB_API_KEY`),
then:

```bash
bash examples/skillrise_alfworld/skillrise_alfworld_qwen3_4b.sh   # SkillRise
bash examples/skillrise_alfworld/grpo_alfworld_qwen3_4b.sh        # GRPO baseline
```

Same layout for `skillrise_webshop/` and `skillrise_sciworld/`. Task sequences
(K=3) are bundled under `data/groups/`.

## License

Apache License 2.0 (derivative of verl / verl-agent). See [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE).
