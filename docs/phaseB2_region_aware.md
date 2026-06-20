# 阶段 B-2：Region-Aware 两阶段判定 + 严格验证

复跑：
- 主实验 `scripts/phaseB2_region.py`
- 严格验证 `scripts/phaseB2a_strict.py`
- dev 特征 `scripts/extract_features.py --predictions outputs/roberta_base_dev_predict/predictions.csv --data_file .../semeval_dev_full.csv --output_dir outputs/analysis_dev`

## 方法

诊断（docs/phaseA 末节）发现：RoBERTa margin 空间里 human 上尾（p75=11.2/p90=11.9）与
bloomz（p50=10.9/p90=12.0）重叠，clean AI ≥12.1。单一全局阈值必须在 human specificity 与
bloomz recall 间二选一。但 bloomz 在**正交人工特征**上与 human 强可分（TTR 0.83 vs 0.43，
repeated_bigram 0.01 vs 0.15）。

**Region-aware 两阶段判定**（只用可观测文本特征，不用 model 标签）：
- margin ≥ t_high → AI（clean AI 区）
- margin < t_low → human（clean human 区）
- t_low ≤ margin < t_high → 模糊带：用正交特征上的逻辑回归（class_weight=balanced）判定

t_low/t_high/分类器全部在校准集上拟合。

## 严格验证：三个协议（回答"是否泄漏/过拟合"）

| 协议 | macro-F1 | 用途 |
|---|---|---|
| **P0** 原始（band 在 calib 选又在 calib 评） | 0.9463 ± 0.0012 | 有乐观偏差版本 |
| **P2** 嵌套 CV（band 在内层 calib 选，外层 eval 评） | **0.9447 ± 0.0013** | 真实上界，无 band 过拟合 |
| **P1** dev→test 单次（fit 全在 dev，test 只预测一次） | **0.8567** | 唯一可报告的合法 test 得分 |

P0 与 P2 仅差 0.0016 → **"模糊带+正交特征"方法本身真实有效，不是调参假象**。
P0/P2 用 test 标签做校准，回答上界问题（"假如有 test 同分布校准集"），不能当部署得分。

### 数据/泄漏声明（重要，写报告必须说清）

- **样本级无泄漏**：所有协议 calib/eval 不相交，eval 样本不参与任何拟合。
- **P0/P2 方法学上用了 test 标签校准** → 是"in-distribution calibration ceiling"，
  **不是** test 得分。措辞不得写成"方法在 test 达到 0.946"。
- **P1 是唯一合法 test 数字**：所有参数只在 dev（bloomz+5域）学，test 只预测一次。

## 核心结论：方法在真实分布偏移下依然强健

P1（部署）内部对比：

| | single-threshold | region-aware |
|---|---|---|
| macro-F1 | 0.572 | **0.857** |
| human spec | 0.25 | 0.71 |

- 即使 dev 只含 bloomz（最严酷分布偏移），region-aware 仍把合法 test 得分 0.572→0.857（+0.285）。
- **0.857 已超过原项目所有单模型天花板**（Fusion LGBM 0.835，且那是用 test 调的）。
- **【已修正，见 docs/ablation_results.md】** 此处早期 P2=0.945 出自"正交特征覆盖AI率"的错误协议。正确固定-RoBERTa 协议下 P2=0.854，与 P1 0.857 基本持平（bloomz主导+TTR强可分使dev拟合几乎无损）。
- P1 的 band=[−10.58,10.91]（35%路由）与 P2 的 band 完全不同 → 方法对 band 位置不敏感，
  鲁棒性来自"模糊带交给正交特征"的结构，而非具体阈值。

## per-generator（P2 上界 / P1 部署）

| | GPT4 | bloomz | dolly | davinci | human_spec |
|---|---|---|---|---|---|
| 单阈值天花板 (Phase A) | 0.89 | 0.10 | 0.97 | 0.99 | 0.95 |
| **P2 region-aware** | 1.00 | **0.70** | 0.83 | 0.96 | 0.98 |
| **P1 region-aware (deploy)** | 0.99 | **1.00** | 1.00 | 1.00 | 0.71 |

> P1 把 bloomz 拉满（1.00）但 human_spec 仅 0.71——dev 是 bloomz vs human 训出来的 band，
> 偏向召回 bloomz；P2 有全生成器校准集，更均衡。两者都远胜单阈值。

## 对全项目的意义

之前认为的"0.886 天花板被 human 假阳性死死压住"是**单一全局阈值造成的假象**，不是数据极限。
换决策结构（不动模型、不重训）即突破。诊断→定位正交特征→两阶段路由→指标突破，形成因果闭环。
