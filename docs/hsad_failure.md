# HSAD 失败分析（负面结果）

**复跑数据**：`outputs/hsad_full/`、`outputs/hsad_full_test/`、`outputs/hsad_ls_test/`

## 动机

最初设想：在已微调的 RoBERTa 上叠一个层级结构——token hidden states → 句子 mean-pooling
→ 2 层跨句 TransformerEncoder → 句级 + 文档级双分类头（MTL），冻结 RoBERTa 底 8 层。
目标是同时拿到更强分类性能 + 句级可视化（src/model_hsad.py, scripts/train_hsad.py）。

## 结果：比朴素 RoBERTa 更差

| 模型 | dev acc | test acc | test macro-F1 | **test AUROC** |
|---|---|---|---|---|
| RoBERTa-base（CLS+头） | 0.834 | 0.671 | 0.619 | **0.944** |
| HSAD full | **0.843** | 0.688 | 0.645 | **0.672** |
| HSAD + label smoothing | — | 0.697 | 0.657 | 0.682 |

训练 loss 降到 0.0075（epoch4），dev acc 0.843（epoch1 最佳）——**in-distribution 上看起来更好**。
但 **test AUROC 从 0.944 暴跌到 0.672**：排序能力几乎崩溃。

## 诊断：层级机器在单一分布上过拟合，域外失效

- dev（仅 bloomz + 5 域）上 dev acc 0.843，但 test（6 生成器 + outfox 域）AUROC 0.672。
- 对比：朴素 RoBERTa 同样经历 dev→test 偏移，但 AUROC **守住 0.944**——只是 operating point
  偏了（见 operating-point 分析）。
- 区别在于：HSAD 额外的跨句 Transformer + 双头 + 句级伪标签监督，给了模型更多自由度去
  拟合 dev 的单一分布（bloomz 文体），这些自由度在域外（GPT4/chatGPT 等未见生成器）
  无法泛化，反而破坏了 backbone 本来稳健的判别表示。
- 句级监督信号本身是脏的：subtask C 句子标签由"句子中点字符位置 < 边界"硬切伪标注，
  噪声大，sent_loss_weight=0.3 把噪声灌进了主任务。

## 结论（论文负面结果价值）

> 在强 backbone 已经提供高 AUROC 的前提下，叠加层级结构 + 额外监督，不仅没有提升、
> 反而因增加的拟合自由度在分布偏移下过拟合，使排序能力从 0.944 崩到 0.672。

这条负面结果支撑了本文的核心主张：**问题不在表示容量不足，而在 operating point 与分布偏移**。
正确的改进方向是轻量的校准 + 正交特征裁决（region-aware），而非加重模型。
HSAD 因此被弃用，仅作为对照保留。
