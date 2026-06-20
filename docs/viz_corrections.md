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

## 验证

gallery（scripts/demo_reports.py → outputs/report_gallery/）：
human P(AI)=0.62 判human✓、bloomz 0.89 判AI✓（clean_ai，恢复）、GPT4 0.91 判AI✓。
特征轴 disc 实测范围 −0.40~+0.85，分布展开不贴边。
