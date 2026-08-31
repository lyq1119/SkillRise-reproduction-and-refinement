# SkillRise reproduction-and-refinement

本仓库是 **SkillRise 的复现 + 实验改造项目**。

当前主要目标不是跑 RL 训练，而是 **生成 SFT 数据**：用本地 Qwen 模型做 rollout（推理），用 DeepSeek API 做 curate（精修/教学），并在此基础上做了一轮轮关于**技能演进、错误知识复用**的实验。

## 核心流程

```
Qwen3.5-9B (本地 vLLM rollout)  ──►  trajectory (manifest + jsonl)
        │
        ▼
DeepSeek API (--curate-via-api) ──►  curate 结果 → 技能 / 教训 / 错误库
        │
        ▼
下一轮 seed（技能携带 / merge / evolve）──► 再 rollout（2-round / 多轮）
```

- **rollout**：`runtime/pure_rollout.py`（vLLM，TP=1 / DP=4，默认 4 卡，可用 `GPUS`、`DATA_PARALLEL` 覆盖）。
- **curate**：DeepSeek API 对轨迹精修，产出技能文档、per-trajectory OPID 教训、type-level 错误库。
- **评估**：固定 12 个 val 任务（`runtime/data/exp_mistakes_val12.json`），指标为 round-2 val pass@k。

## 目录结构

```
agent_system/      # 多智能体系统（沿用上游 SkillRise）
data/groups/       # 任务序列分组（如 skillrise_sciworld_K3.jsonl）
examples/          # 上游训练脚本（verl RL，非当前主流程）
runtime/           # 本仓库实际工作区
  run_env.sh                        # 环境变量（模型路径、缓存、HOME 等，路径已硬编码到本机）
  run_pure_rollout.sh               # 单次 rollout 入口
  pure_rollout.py                   # rollout 主程序（vLLM + curate + async val + working memory）
  opid_lessons.py                   # 从轨迹抽取 OPID 教训
  build_round2_seed.py              # 用 r1 轨迹/教训构造 r2 seed
  build_mistakes_bank.py            # 构造 type-level 常见错误库
  merge_validate_skills.py          # 技能合并 + 打分校验（<3 丢弃 / 3-4 重写 / >=4 保留）
  evolve_skills.py                  # 技能进化（诊断 + 定向修复 + judge 门控）
  api_health.py                     # DeepSeek API 健康检查（E1）
  eval_data_quality.py              # 数据质量评估
  outputs/                          # 所有实验结果（git 跟踪部分 manifest/eval）
  models/  cache/  home/  sources/  # 本地依赖，gitignore，不入库
```

## 实验一览

所有实验脚本均在 `runtime/`，**可断点续跑**（每个 experiment 都写 state 文件，已完成的条件会跳过）。

| 分支 | 实验 | 脚本 |
|---|---|---|
| `d1-skill-carryover` | **2-round 技能沿用**：4 条件 × (r1 → 教训? → seed → r2)，条件为 baseline / baseline+lesson / b1 / b1+lesson | `run_2round_experiment.sh` |
| `e1-api-outage-protection` | **DeepSeek API 故障保护**：预检等待、部分条件运行、损坏条件自动重跑、等 8 卡空闲 | `rerun_corrupted.sh` 等 |
| `e2-api-client-ignore-proxy` | **绕过本地代理**：client `trust_env=False`，免疫失效的 `ALL_PROXY` | — |
| `exp-mistakes-vs-lessons` | **错误 vs 教训**：共享 r1，r2 分 control/lesson/mistakes 三条件 × 3 组 | `run_mistakes_experiment.sh` |
| `exp-merge-validate-skill` | **技能合并+校验**：merge-replace / merge-stack 两种 seed | `run_merge_experiment.sh` |
| `exp-skill-evolution` | **技能进化**：A（merge-replace）× 3 轮，B（merge+lesson）× 3 轮 | `run_evolution_experiments.sh` / `run_evolution_round.sh` |
| `opt-async-loop` | **异步验证**：`--val-splits N` 把 val 拆成子批次在不相交 worker 子集上并行，缓解单卡 100% 其余闲置 | `validate_async.sh` |
| `opt-agent-wm` | **agent 工作记忆**：solve prompt 中插入 task-state 块（`--working-memory`，默认关闭）；inference_worker 按 rank 映射父进程 `CUDA_VISIBLE_DEVICES` | — |

## 环境与运行

环境变量集中在 `runtime/run_env.sh`（路径已硬编码为本机 `/data/lanyuqi/skillrise`，含 venv、JAVA_HOME、模型路径 `runtime/models/Qwen3.5-9B`、各缓存目录）。

```bash
# 单次 rollout（r1 用）
bash runtime/run_pure_rollout.sh --group-id <group_id> \
  --val-tasks runtime/data/exp_mistakes_val12.json \
  --rollout-n 3 --curate-via-api

# 2-round 实验（D1）
bash runtime/run_2round_experiment.sh            # 默认跑全部 4 个条件

# 指定 GPU（默认 0,1,2,3）
GPUS=4,5,6,7 DATA_PARALLEL=4 bash runtime/run_pure_rollout.sh ...

# 启用 agent 工作记忆（默认关闭）
... bash runtime/run_pure_rollout.sh ... --working-memory
```

所有输出写入 `runtime/outputs/<实验名>/`，含 `manifest.json`（评估指标、wall time）、`run.log` 和轨迹 jsonl。

## 上游

复现自 [SkillRise](https://github.com/Within-yao/SkillRise)（框架：veRL / verl-agent）。原始论文方法见上游仓库。
