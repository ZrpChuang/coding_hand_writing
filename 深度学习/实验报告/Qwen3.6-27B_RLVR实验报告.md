# Qwen3.6-27B 数学 RLVR 实验报告

日期：2026-07-25 至 2026-07-27  
代码与产物：`/mnt/bn/vai-llm/zrp/RL`

## 1. 结论摘要

本实验使用 Qwen3.6-27B 进行小规模数学可验证强化学习（RLVR），比较了 Dr.GRPO、DAPO token-level loss、BNPO 和 OPSD。

主要结论：

1. 在线训练 reward 明显上升，但冻结评测没有统计显著提升。
2. 从 25 扩大到 200 个优化步（约 400–3200 条 rollout）仍未观察到显著规模收益。
3. 训练熵仅轻微变化，没有出现熵坍缩。
4. OOD 准确率基本保持不变，没有显著灾难性遗忘。
5. Dr.GRPO 数值最好，但相对基座仍不显著；DAPO-loss、BNPO 也没有显著增益。
6. 本轮 OPSD 没有增益，ID 准确率数值上低于基座。

因此，目前没有一个训练后 LoRA 能被严谨地称为显著优于基座。

## 2. 任务定义

模型针对数学题生成短答案：

```text
<answer>数值</answer>
```

程序自动解析并核验最终数值，以此提供无需人工标注和奖励模型的可验证奖励。

训练与 ID 评测题型：

- 四则运算
- 一次方程
- 分数
- 模运算
- 数位和

OOD 评测使用训练中未出现的题型：

- 最大公约数
- 等差数列
- 组合数
- 百分比
- 多边形内角和

## 3. 固定实验协议

| 项目 | 配置 |
|---|---|
| 基础模型 | Qwen3.6-27B text-only |
| 硬件 | 2×A800 80GB |
| 训练集 | 256 题 |
| ID 评测集 | 256 题 |
| OOD 评测集 | 256 题 |
| 评测采样 | 每题 4 次 |
| 主指标 | ID exact pass@1 |
| OOD 护栏 | OOD exact pass@1 |
| 置信区间 | 10,000 次配对 bootstrap，95% CI |
| 评测采样参数 | temperature=1.0，top-p=0.95，top-k=20 |
| 最大生成长度 | 32 token |
| 固定种子 | 训练 42；评测 20260725 |

冻结协议及文件哈希位于：

```text
/mnt/bn/vai-llm/zrp/RL/research/frozen/manifest.json
```

## 4. 共同训练配置

Dr.GRPO、DAPO-loss 和 BNPO 使用相同配置，仅替换 loss formulation：

| 配置项 | 值 |
|---|---|
| 并行方式 | Accelerate DDP |
| 量化 | 4-bit NF4、double quant、BF16 compute |
| LoRA | r=8，alpha=16，dropout=0 |
| LoRA 模块 | q_proj、k_proj、v_proj、o_proj |
| 每卡 batch | 2 |
| 梯度累积 | 4 |
| 每题 generations | 4 |
| 每步 rollout | 约 16 条 |
| 学习率 | 1e-5 |
| 优化器 | fused AdamW |
| 调度器 | 5% warmup + cosine |
| 最大梯度范数 | 1.0 |
| KL 系数 | beta=0 |
| Thinking | 关闭 |
| 检查点间隔 | 25 步 |

奖励组成：

- 精确答案：1.0
- 数值接近度：0.2
- 格式正确：0.1
- 奖励按 generation group 标准化

说明：本文的 DAPO 实验仅使用 TRL 提供的 token-level DAPO loss，不包含原始 DAPO 系统的 Dynamic Sampling、Clip-Higher 等完整组件。

## 5. 正式训练记录

| Run | 步数 | 约 rollout 数 | 总耗时 | 输出目录 |
|---|---:|---:|---:|---|
| 初始 Dr.GRPO | 100 | 1600 | 约 19 分钟 | `outputs/train-direct-20260725-direct-r1` |
| 规模 Dr.GRPO | 200 | 3200 | 39分49秒 | `outputs/scale-drgrpo-200` |
| DAPO-loss | 100 | 1600 | 20分40秒 | `outputs/method-dapo-100` |
| BNPO | 100 | 1600 | 20分20秒 | `outputs/method-bnpo-100` |
| OPSD | 100 | 1600 | 23分54秒 | `outputs/opsd-100` |

`smoke*` 和 `diagnostic-opsd*` 均为 1 步工程验证，不用于效果结论。

## 6. 规模实验

| 检查点 | 约 rollout 数 | ID exact pass@1 | OOD exact pass@1 |
|---|---:|---:|---:|
| Base | 0 | 14.16% | 76.95% |
| Step 25 | 400 | 13.77% | 76.95% |
| Step 50 | 800 | 14.55% | 76.56% |
| Step 75 | 1200 | 14.26% | 76.86% |
| Step 100 | 1600 | 14.55% | 76.66% |
| Step 125 | 2000 | **15.23%** | 76.66% |
| Step 150 | 2400 | 14.16% | 76.76% |
| Step 175 | 2800 | 14.26% | 76.76% |
| Step 200 | 3200 | 14.55% | 76.66% |

Step 125 数值最高，但所有相邻增量的 95% CI 均包含 0。结论是：在当前数据、奖励和评测精度下，从最早的 25 步开始就没有可确认的显著规模增益，继续扩大到 200 步也没有改变这一结论。

## 7. 熵变化

| 方法 | 前 20 步 | 后 20 步 | 变化 |
|---|---:|---:|---:|
| Dr.GRPO 200 | 0.2965 | 0.2884 | -0.0082 |
| DAPO-loss 100 | 0.2985 | 0.2953 | -0.0032 |
| BNPO 100 | 0.2974 | 0.2960 | -0.0015 |
| OPSD student | 0.3200 | 0.3479 | +0.0279 |
| OPSD teacher | 0.6988 | 0.6789 | -0.0200 |

GRPO 系列的熵基本稳定，没有观察到持续快速下降或熵坍缩。OPSD 学生熵反而有所上升，说明该负结果不是简单由策略过早变得确定导致的。

## 8. 方法比较

| 方法 | ID pass@1 | ID pass@4 | OOD pass@1 | 相对 Base |
|---|---:|---:|---:|---:|
| Base | 14.16% | 24.61% | 76.95% | — |
| Dr.GRPO 100 | **14.55%** | **26.56%** | 76.66% | +0.39 pp |
| DAPO-loss 100 | 14.36% | 25.39% | 76.95% | +0.20 pp |
| BNPO 100 | 14.26% | 25.78% | **77.05%** | +0.10 pp |
| OPSD 100 | 13.57% | 22.27% | 76.86% | -0.59 pp |

所有 ID pass@1 差异的 95% CI 都包含 0。Dr.GRPO 数值最好，但不能宣称显著优于基座。

## 9. OOD 退化检查

Base OOD 为 76.95%，Dr.GRPO Step 200 为 76.66%，变化 -0.29 个百分点，置信区间包含 0。其他方法也保持在 76.86%–77.05%。

结论：没有观察到显著 OOD 退化，但也没有获得可确认的 OOD 泛化提升。

## 10. OPSD 配置与结果

本轮按 on-policy self-distillation 实现：

- 学生只看到题目，并从自身策略采样轨迹。
- 同一个初始模型关闭 LoRA 后作为冻结教师。
- 教师额外看到该题标准答案。
- 教师和学生在学生 rollout token 上计算全词表 JSD 0.5。
- 梯度只通过学生 LoRA。
- batch、梯度累积、生成长度和种子与其他 100 步实验保持一致。

OPSD 的 ID pass@1 为 13.57%，Base 为 14.16%。差值 -0.59 个百分点，统计不显著，但方向上没有增益。

可能原因：

1. 教师只获得最终答案，而不是高质量完整推理轨迹。
2. 教师与学生是同一个初始模型，教师能力提升有限。
3. 32-token 短回答降低了 dense token supervision 的潜在优势。
4. 小型训练集和较低基座 ID 正确率使统计功效有限。

## 11. 最值得保留的学习结果

1. **训练 reward 与冻结能力评测必须分开。** 在线 reward 和在线正确率都上升，但固定评测没有显著改善。
2. **扩大训练步数不能自动解决问题。** 200 步没有比 25–125 步产生稳定收益。
3. **负结果不是熵坍缩导致。** 熵整体稳定。
4. **当前奖励可能过于容易被局部适应。** 后续应增加训练规模、提高题目难度，并强化独立评测。
5. **OPSD 对教师特权信息质量敏感。** 仅提供最终答案不足以保证获得有效的 dense supervision。

数值最好的候选检查点是：

```text
/mnt/bn/vai-llm/zrp/RL/outputs/scale-drgrpo-200/checkpoint-125
```

它适合继续复测，但不能作为“已验证提升”的部署模型。

## 12. 建议的下一轮实验

1. 将 ID/OOD 评测扩展到至少 1,000–2,000 题，提高显著性检验功效。
2. 使用更难且基座正确率约 30%–60% 的训练题，减少全错组和无梯度组。
3. 增加独立随机种子，至少运行 3 次并报告均值与方差。
4. OPSD 教师改用验证过的完整解题轨迹，或使用更强外部教师做真正 OPD。
5. 完整复现 DAPO 的 Dynamic Sampling、Clip-Higher 和 Overlong Reward Shaping，再与 Dr.GRPO 比较。

## 13. 结果文件

```text
/mnt/bn/vai-llm/zrp/RL/research/final/report.json
/mnt/bn/vai-llm/zrp/RL/research/final/research_summary.png
/mnt/bn/vai-llm/zrp/RL/research/eval-scale-200/predictions.jsonl
/mnt/bn/vai-llm/zrp/RL/research/eval-methods-100/predictions.jsonl
/mnt/bn/vai-llm/zrp/RL/research/results.tsv
```

参考：

- TRL GRPO Trainer：https://huggingface.co/docs/trl/grpo_trainer
- OPSD 论文：https://siyan-zhao.github.io/assets/papers/OPSD.pdf

