# 实验方案设计：层级 AI 文本检测器

## 目标

1. **高分类性能**：超过 Fusion LightGBM（test macro-F1 = 0.835）
2. **层级可视化**：词/句/文档三级 AI 概率，类查重高亮展示
3. **自然语言解释**：输出 AI vs 人类写作风格的差异描述，可指导改写

---

## 模型架构：HSAD（Hierarchical Sentence-Aware Detector）

### 核心设计

```
输入文档（英文，≤512 token 截断）
  │
  ▼
RoBERTa-base（在 semeval 上已 fine-tuned，继续低 lr 训练后4层）
  │ 输出：所有 token hidden states（768维）
  │
  ├─────────── 词级 ──────────────────────────────
  │ Integrated Gradients 归因 → 每个 token 重要性分数
  │ 可视化：词级颜色深浅（不修改模型，推理时计算）
  │
  ▼ 按句子边界 mean pooling
每句向量（768维，上下文完整）
  │
  ▼ 2层 TransformerEncoder（hidden=768, heads=8, ffn=2048）
跨句上下文建模（捕捉句间顺序和依赖）
  │
  ├─────────── 句级 ──────────────────────────────
  │ 句子分类头：Linear(768→2) → 每句 prob_ai
  │ 监督信号：subtask C 句子级标注（混合文本边界）
  │ 可视化：句子背景色深浅
  │
  ▼ attention pooling（以句子 prob_ai 为权重）
文档向量（768维）
  │
  ├─────────── 文档级 ────────────────────────────
  │ 文档分类头：Linear(768→2) → 文档 prob_ai
  │ 监督信号：subtask A 文档级标注
  │ 输出：整体 AI 概率
  │
  ▼
风格解释器（解耦模块）
  对高 AI 概率句子 → features_v2（23维）→ 与训练集均值比较
  → 自然语言报告（模板生成，可指导改写）
```

### 为什么是两个分类头而不是三个

- 词级不需要单独的头：Integrated Gradients 从文档级 loss 反向传播即可
- 段落级对 semeval 无效（100% 单段），真实文章 demo 中段落 = 相邻句子聚合
- 两个有监督的头（句级 + 文档级）MTL 训练，梯度来源明确，叙事简洁

---

## 训练数据策略

### 可用数据（hf-mirror 可达，无需手动下载）

| 数据集 | 规模 | 用于 | 语言 |
|---|---|---|---|
| semeval M4 train（本地已有）| 119,756 docs | 文档级二分类 | 英文多域 |
| semeval subtask C（需下载）| 混合文本 + 句子边界标注 | 句级序列标注 | 英文 |
| HC3（本地已有）| 24,322 QA对 | 辅助文档级 | 英文 |
| RAID（hf-mirror 可达）| 600万+ docs，11模型，8域 | 扩展文档级泛化 | 英文为主 |

### 迁移可行性分析

**真正的问题**：不同数据集的文本分布差异显著

| 数据集 | 平均词数 | 域 | AI模型 |
|---|---|---|---|
| semeval outfox | 460 | wikihow/wiki/reddit/arxiv | chatGPT/cohere/davinci/dolly |
| HC3 | 134（human）/ 173（AI）| reddit/finance/medical/wiki | ChatGPT |
| RAID | 混合 | abstracts/news/wiki/recipes等 | GPT-4/Claude/Llama等11个 |

**迁移可行的部分**：
- RAID 的 GPT-4、ChatGPT 和 semeval 的 chatGPT/davinci 有重叠 → 有迁移性
- RAID 覆盖更多现代模型（Claude、Llama2）→ 提升泛化性

**迁移风险**：
- RAID 文本平均更短（abstracts 约200词），semeval wikihow 平均681词，分布差异大
- HC3 是 QA 格式（问题+回答），semeval 是独立文章，格式不一致
- **建议：先只用 semeval M4 train 做主训练，RAID 作为辅助域泛化验证，而非混入训练**

### 推荐训练配置

```
主训练（文档级）：semeval train 119K
  + MTL 辅助（句级）：semeval subtask C（待下载，约5-10K混合文档）
验证：semeval dev 5K
测试：semeval test 34K（官方）

可选扩展：RAID 英文 non-adversarial 子集 → 测试泛化性
```

---

## 多语言问题

**先只做英文，理由：**
- semeval subtask A 英文数据 119K，足够训练
- 多语言需要 mBERT 或 XLM-R，是另一个完整实验
- 叙事上"英文检测器 + 层级可视化"已经完整，多语言是扩展工作
- 如果要做多语言，subtask A 的 multilingual track 数据也需要单独分析

---

## 与现有方案的对比（叙事线）

```
baseline: RoBERTa-base（CLS → 分类头）
              ↓ 问题：截断丢失信息，只有文档级输出，无可解释性

本方案: RoBERTa-base（token hidden states → 句子聚合 → 跨句建模 → 两级输出）
              ↓ 改进：充分利用 token 级信息，句/文档双级监督，可视化完整
```

这是一条线的自然延伸，不是多个模型的拼接。

---

## SOTA 参考

- **Weighted Layer Averaging RoBERTa**（SemEval 2024 Task 8 参赛系统）：
  对 RoBERTa 各层 hidden states 加权平均，不只用最后一层，accuracy ~86.9%
  → 与本方案的"token hidden states 充分利用"思路一致

- **Fine-Grained Detection via Sentence-Level Segmentation**（2024）：
  句子序列标注检测 human/AI 转换边界
  → 对应本方案的句级分类头

- **StyleDecipher**（2024）：
  离散风格指标 + 连续语义表示联合建模，输出可解释风格差异
  → 对应本方案的风格解释器模块

---

## 下一步行动项

1. [ ] 下载 semeval subtask C 数据（hf-mirror 可达）
2. [ ] 确认 RAID 数据量和英文过滤后的规模
3. [ ] 实现 HSAD 模型（src/model_hsad.py）
4. [ ] 实现 MTL 训练脚本（scripts/train_hsad.py）
5. [ ] 集成 Integrated Gradients 词级归因
6. [ ] 更新 viz.py 支持三级可视化输出

---

## 状态

设计确认中，待实现。
