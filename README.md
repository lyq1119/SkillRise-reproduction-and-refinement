# SkillRise reproduction-and-refinement

在 ScienceWorld 上复现 SkillRise，目标是**生成高质量 SFT 数据**：用本地 Qwen3.5-9B 做 rollout（推理轨迹），用 DeepSeek API 做 curate（把轨迹精修成技能文档/教训），再让下一轮 rollout 复用这些知识。整个仓库是一系列"知识怎么带进下一轮"的对照实验。

## 核心流程

```
Qwen3.5-9B rollout ──► 轨迹 + manifest
      │
      ▼
DeepSeek curate ──► 技能 / 教训 / 错误库
      │
      ▼
构造下一轮 seed（携带 / 合并 / 进化）──► 再 rollout
```

指标统一用 **round-2 val pass@k**：早期 8 个任务，之后固定 12 个任务（`runtime/data/exp_mistakes_val12.json`）＋ 3 个任务类型组（每组 3 组重复）取均值。

## 实验逻辑链

总问题：**round-1 产生的知识（技能/教训/错误），怎么带进 round-2，val 才能更高？**

### 0. E0 数据质量基线 —— 先立度量标准
- **目的**：生成的数据好不好不能靠猜。用 DeepSeek 当 judge：轨迹打 T1–T4（可执行性/多样性/状态覆盖…），技能打 S1–S5（结构/抽象度/可迁移性）。
- **发现**：B1（DeepSeek curate 提炼技能）之后，技能抽象度 S2/S4 从 ~2.3 升到 ~3.8，技能 parse 率 0.63→1.0；带上技能后训练集 solved 从 0 升到 0.33~0.67（报告见 `runtime/eval_outputs/eval_baseline_*.json`）。
- **因此**：有了可信的数据质量度量，后续每个实验都用它评估。

### 1. D1 技能携带 —— 技能/教训能否帮到下一轮
- **目的**：2-round 实验，4 条件（baseline / baseline+lesson / b1 / b1+lesson）各跑 r1→r2，val=8。
- **发现**：携带技能小幅提升 pass@2/3（b1_lesson_x2 r2 pass@3=7/8 vs baseline 5/8），但 pass@1 有时还降；8 个任务噪声太大。
- **因此**：任务量不够 → 固定 12 任务 val ＋ 3 组重复，做统计上更可信的对照。

### 2. E1/E2 基建可靠性 —— 长实验必须先能跑完
- **目的/发现**：跑实验时 DeepSeek API 间歇性挂（本地代理死亡、连接超时），实验一挂就得重来。E1 加预检等待、支持部分条件运行、自动重跑损坏条件；E2 让 client 忽略环境代理（`trust_env=False`），免疫死的 `ALL_PROXY`。
- **因此**：API 稳了，后面的大实验才跑得完。

### 3. mistakes vs lessons —— 失败知识哪种给法有用
- **目的**：D1 里教训（lesson）表现不稳。失败知识有两种形态：**per-trajectory 的 OPID 教训** vs **type-level 的"常见错误库"**。3 条件 × 3 组，共享 r1。
- **发现**（r2 均值，val=12）：
  | 条件 | pass@1 | pass@2 | pass@3 |
  |---|---|---|---|
  | control | 0.31 | 0.42 | 0.42 |
  | **lesson** | **0.39** | **0.50** | **0.53** |
  | mistakes | 0.25 | 0.42 | 0.53 |
- **结论**：**lesson 最好**；type-level 错误库反而伤 pass@1（跨类型的"常见错误"对具体任务会误导）。
- **因此**：放弃"常见错误库"路线，保留教训。同时技能越攒越多（3 组 → 3 份技能），提示词开始膨胀。

### 4. 技能合并 —— 技能多到 prompt 装不下
- **目的**：把每组的技能合并成**一个 type-level 技能**（打分校验：<3 丢弃 / 3–4 重写 / ≥4 保留），然后 replace 或 stack 进 seed。
- **发现**（r2 均值）：
  | 条件 | pass@1 | pass@2 | pass@3 |
  |---|---|---|---|
  | lesson（上轮最好） | 0.39 | 0.50 | 0.53 |
  | merge-replace | 0.39 | 0.50 | **0.56** |
  | merge-stack | 0.42 | 0.47 | 0.50 |
- **结论**：合并成单一技能 ≈ lesson 水平，replace 还略优 pass@3，且 prompt 更精简 → **技能可无损合并**。
- **因此**：以 merge-replace 为 base，做"技能能不能跨轮自己进化"的实验。

### 5. 技能进化 —— 技能能否跨轮自我改进
- **目的**：对失败 episode 做诊断 → 定向修技能内容 → judge 门控（评分不过就丢弃，防止改坏）。A（merge-replace）3 轮，B（merge-replace+lesson）3 轮。
- **发现**：
  | run | pass@1 | pass@2 | pass@3 |
  |---|---|---|---|
  | A_r3 | 0.39 | 0.50 | 0.53 |
  | B_r1 | 0.44 | 0.50 | 0.50 |
  | B_r2 | 0.42 | 0.50 | 0.53 |
  | B_r3 | 0.39 | 0.44 | 0.44 |
- **结论**：进化能保持水平（judge 门控拦住了坏修改），但**不累积**，跨轮反而略降 → 只改技能内容已经到顶。
- **因此**：转向两个新方向——（a）**async val**：验证阶段 1 张卡忙、其余闲置，把 val 拆成 N 个子批次在不相交 worker 子集上并行，解决耗时瓶颈；（b）**agent 工作记忆**：让 agent 在 solve prompt 里自带 task-state 块（`--working-memory`，默认关），从"改技能"转向"改 agent 自身状态"。

### 6. 当前进行中
- `opt-async-loop`：async val 已接入 evolution driver（`--val-splits 2`）。
- `opt-agent-wm`：工作记忆 + inference_worker 按 rank 映射非零 GPU + RL 脚本 4 卡 FSDP 配置，正在用 `--working-memory` 跑测试。

## 目录结构

```
runtime/
  run_env.sh                  # 环境变量（模型路径/缓存/HOME，硬编码本机 /data/lanyuqi/skillrise）
  run_pure_rollout.sh         # 单次 rollout 入口（默认 4 卡，GPUS/DATA_PARALLEL 可覆盖）
  pure_rollout.py             # rollout 主程序（vLLM + curate + async val + working memory）
  opid_lessons.py             # 从轨迹抽 OPID 教训
  build_round2_seed.py        # r1 轨迹/教训 → r2 seed
  build_mistakes_bank.py      # type-level 常见错误库（实验3，已弃用）
  merge_validate_skills.py    # 技能合并 + 打分校验（实验4）
  evolve_skills.py            # 技能进化：诊断 + 定向修复 + judge 门控（实验5）
  api_health.py               # DeepSeek API 健康检查（实验2）
  eval_data_quality.py        # 数据质量评估（实验0）
  data/                       # 分组、12 任务 val 列表
  outputs/                    # 实验结果（gitignore，本地）
  eval_outputs/               # E0 质量报告（入库）
  models/ cache/ home/ sources/  # 本地依赖（gitignore）
```

## 运行

```bash
# 单次 rollout
bash runtime/run_pure_rollout.sh --group-id <group_id> \
  --val-tasks runtime/data/exp_mistakes_val12.json \
  --rollout-n 3 --curate-via-api

# 指定 GPU（默认 0,1,2,3）
GPUS=4,5,6,7 DATA_PARALLEL=4 bash runtime/run_pure_rollout.sh ...

# 启用 agent 工作记忆（默认关）
... bash runtime/run_pure_rollout.sh ... --working-memory
```

实验脚本均带**断点续跑**（state 文件记录已完成条件，跳过不重跑）。输出在 `runtime/outputs/<实验名>/`，含 `manifest.json`（pass@k、wall time）、`run.log`、轨迹 jsonl。

## 上游

复现自 [SkillRise](https://github.com/Within-yao/SkillRise)（框架 veRL / verl-agent），ScienceWorld 环境见 BEACON。原始论文方法见上游仓库。
