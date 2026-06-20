# 阶段 A：Operating-Point 天花板分析

复跑：`.conda/bin/python scripts/phaseA_report.py` → `outputs/phaseA_ceiling/ceiling.json`

## 要回答的问题

RoBERTa 在 test 上 macro-F1 只有 0.619，但 AUROC 高达 0.944。这个差距是
**判定阈值（operating point）选错**，还是**表示本身不行**？

## 方法

把 test 按 `model × label` 分层切 50/50（保证每个生成器在两半都出现）：
一半当校准集 calib（fit 温度 + 选 macro-F1 最优 margin 阈值），另一半 eval 评估。
5 个 seed 取均值。calib 与 eval 不相交，无泄漏。all-test oracle 作为严格上界参考。

> 注意：模型在 train 上训过，从 train 切 held-out 会过度自信；dev 只有 bloomz 单一
> 生成器，不能代表 test。所以用 test 自身分层切分来估计"假如有同分布标注校准集"
> 的天花板，这是最诚实的可达上界。

## 结果

| 模型 | baseline@0.5 macro-F1 | AUROC | 校准+阈值后 macro-F1 | oracle 上界 | ECE(校准后) |
|---|---|---|---|---|---|
| RoBERTa single | 0.619 | 0.944 | **0.886 ± 0.002** | 0.888 | 0.046 |
| RoBERTa chunked | 0.677 | 0.958 | 0.880 ± 0.002 | 0.881 | 0.215 |
| TF-IDF + LR | 0.828 | 0.897 | 0.830 ± 0.001 | 0.831 | 0.032 |

温度 ≈ 12.7，margin 阈值 ≈ +12.0（与阶段 0 的 all-test oracle +12.009 吻合，交叉验证通过）。

## 结论

1. **病在 operating point，不在表示。** RoBERTa 固定不动，只调温度+阈值，
   macro-F1 从 0.619 → 0.886，逼近 oracle 0.888，方差极小。这是免费的 +0.27。
2. **dev 不能用作校准集。** dev 只含 bloomz + 5 域；test 含 6 生成器（多 GPT4/bloomz
   两个未见）且全 outfox 域。dev↔test 双重分布偏移，dev 阈值迁移到 test 反而更差
   （阶段 0 已证实：dev 最优 margin −5.86 vs test oracle +12.0，方向相反）。
3. **TF-IDF 本就校准良好**（ECE 0.03，校准前后几乎不变）。浅层词汇特征跨生成器更稳，
   这是它在 test 上 baseline 反超 RoBERTa 的原因。

## 校准后每个生成器的真实表现（RoBERTa single）

human specificity = 0.954

| 生成器 | AI 检出率(recall) | 备注 |
|---|---|---|
| chatGPT / cohere / davinci | 0.99+ | 训练见过同族，完美 |
| dolly | 0.97 | 好 |
| GPT4 | 0.89 | **未见生成器，泛化成功** |
| **bloomz** | **0.096** | **几乎全漏** |

> per-generator 不能用 macro-F1（每个生成器子集单类别，macro-F1 退化为 ~0.5），
> 必须用 recall(TPR)，human 单独看 specificity(TNR)。

## 剩余唯一硬骨头：bloomz

bloomz 检出率仅 9.6%——模型几乎把所有 bloomz 判成 human。反直觉点在于 bloomz 是
**dev 唯一的 AI 生成器**。解释：bloomz 输出文体最接近人类自然写作，特征空间与 human
高度重叠；提高阈值修好整体后，bloomz 首当其冲被划到 human 侧。chunked 把 bloomz 拉到
0.56 但牺牲 GPT4 和 human specificity，整体反而更低（0.880 < 0.886）——说明 bloomz 漏检
与长文截断是两个独立问题。

→ 阶段 B 的明确目标：bloomz 这类"拟人文体"漏检（对应"AI 越像人越难检测"的攻防核心）。

---

## 阶段 B 第一步（修正阶段 A 的乐观假设）

把校准+同分布阈值方法套到 Fusion 上后，结论被修正：

| 模型 | 校准后 macro-F1 | bloomz recall | human spec |
|---|---|---|---|
| RoBERTa single | 0.886 | 0.096 | 0.954 |
| Fusion LGBM | 0.882 | 0.122 | 0.943 |
| Fusion LR | 0.838 | 0.166 | 0.872 |

**关键修正**：Fusion LGBM 未校准 @0.5 时 bloomz recall=0.410（人工特征如 TTR 确实携带正交信号），
但校准+选最优全局阈值后 bloomz 又掉回 0.122——因为为最大化整体 macro-F1，阈值被推高去压
human 假阳性，bloomz（仅 3000 样本 vs human 16272）被当作可接受牺牲品。

**根因定性**：bloomz 不是独立难题，而是"单一全局阈值 + 类别极度不平衡 + human 在 outfox 域
过度自信"三者的产物。任何以整体 macro-F1 为目标的单阈值方案都救不了它。校准后各模型都收敛到
~0.88，被同一个 human 假阳性天花板限制。

**margin 空间证据**（RoBERTa）：human p50=7.87 但 p75=11.21/p90=11.86（上半截探到 12）；
bloomz p50=10.88/p90=12.00 与 human 上半截几乎完全重叠；GPT4/seen_AI 干净地站在 12.1 以上。
阈值 12→human spec 0.95/bloomz 0.10；阈值 11→bloomz 0.48 但 human spec 0.72。bloomz recall 与
human spec 直接对冲。

→ 突破 0.886 只有两条路：(1) 放弃单一全局阈值，做 per-source/per-region 自适应判定；
(2) 重训降低 human 假阳性。**已选 (1) per-source 自适应阈值**作为下一步。

