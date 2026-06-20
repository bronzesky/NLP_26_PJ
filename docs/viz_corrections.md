# 阶段 C-可视化：口径修正记录（2026-06-17）

针对三处用户质疑做的修正。`src/detector_pipeline.py` + `scripts/render_report.py`。

## 修正1：头条数字 — 去掉误导性的离散 suspicion

**问题**：原 `doc_suspicion` 在 clean 区被硬编码为 1.0/0.0，在模糊带才用正交分类器概率。
导致"疑似度 100%→0% 突跳但 RoBERTa 仍 82%"——那是离散标签在跳，不是检测置信度。

**修正**：头条改用 RoBERTa **真实校准概率** `doc_prob_ai`（温度 5.37，连续不突跳）。
region-aware 判定降级为单独的 `doc_verdict` + `doc_region` 标签，与头条数字分离。
移除 `doc_suspicion` 字段。诚实暴露：human 样本头条 P(AI)=0.62（反映检测器对 outfox 域
人类文本的过度自信，见 [[阶段A]]），但 region-aware 仍正确判 human。

## 修正2：逐句高亮 — 改遮挡法，与总分联动

**问题**：原句子分是每句单独喂 RoBERTa，但模型在整篇上训练，单句信号弱→大多落模糊带→
全绿；而文档分是整篇一起算→margin 12+→判 AI。两套割裂打分导致"全绿却判 AI"。

**修正**：句子分改 **遮挡法（occlusion）**：`contrib = doc_prob(全文) − doc_prob(删去该句)`。
正值＝删掉它使 AI 概率下降＝该句推高 AI；负值＝推向人类。底色按 contrib 红/绿深浅。
高亮与文档总分**真正联动**——一次性算所有变体（n 句 = n 次前向）。

## 修正3：特征证据条 — 重画为人类↔AI判别轴

**问题**：原条形图把原始值归一化放单 marker，无刻度、贴边、看不出实际偏离。

**修正**：每个特征映射到 **高斯对数似然比判别轴**：
```
LLR  = logN(val; ai_mean, ai_std) − logN(val; human_mean, human_std)
disc = tanh(LLR/2)  ∈ [−1, +1]      (−1=确定人类, 0=决策边界, +1=确定AI)
```
渲染：左端"确定是人类"(绿)→正中决策边界→右端"确定是AI"(红)渐变轴；圆点落在 disc 位置；
两竖线标 human-mean / ai-mean 在轴上的落点；右列直接写"85% 偏AI"。比原始值位置科学——
同时考虑两类均值与方差。按 |disc| 排序，最具区分度特征在前。

## 修正4：头条概率必须与判定同源（修"70%却判human"矛盾）

**问题（用户指出）**：降AI样本头条显示 P(AI)=70% 却判 human——因为头条显示的是 RoBERTa 原始
校准概率，而判定来自正交特征裁决，两者不同源必然矛盾。根因：模糊带 [t_low,t_high] 在概率
空间是 [0.122, 0.884]，宽到任何 12%–88% 的文档都被正交特征(TTR)接管，RoBERTa 判断被丢弃。
band 在 bloomz-only dev 上拟合导致过宽（与 HSAD 同一过拟合陷阱）。

**试错**：在 dev 上重新拟合 logistic 融合(margin+6正交特征) → macro-F1 砸到 0.53，因 dev 只有
bloomz，融合学成"AI=高TTR"，chatGPT/cohere/davinci 召回全崩。否决。

**最终修法（最小、不重训、不掉指标）**：判定逻辑不动，只让头条 = 实际驱动判定的那个概率
`final_prob`：
- 干净区(margin≥t_high 或 <t_low)：显示校准 RoBERTa P(AI)（连续，**不再硬编码 1.0/0.0**——
  那个离散跳变是更早的 doc_suspicion bug）；
- 模糊带：显示正交分类器 P(AI)（就是那里真正做判定的信号）。
verdict = (final_prob ≥ 0.5)，与头条**永远一致**。RoBERTa 原始概率作为"RoBERTa 原始 P(AI)"
单独透明列在信号面板，暴露两阶段分歧而非隐藏。

**验证**：test macro-F1=0.8567（与旧 band 0.857 相同）、bloomz 召回 0.997（保住）、各生成器
0.99+、incoherent 文档=0。降AI样本：头条 90.6%→**0.0%**、判定 AI→human、区域"模糊带"、
RoBERTa 原始 70.2% 透明展示。"70%判human"矛盾消除。

## 验证

gallery（scripts/demo_reports.py → outputs/report_gallery/）：
human P(AI)=0.62 判human✓、bloomz 0.89 判AI✓（clean_ai，恢复）、GPT4 0.91 判AI✓。
特征轴 disc 实测范围 −0.40~+0.85，分布展开不贴边。
