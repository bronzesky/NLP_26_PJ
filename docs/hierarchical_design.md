# 备选方案：层级 Transformer 检测器

## 核心思路

**从 CLS 点分类 → token-level 全信息层级聚合**

相对于 RoBERTa-base baseline（取 [CLS] 向量 → 分类头）的自然延伸，不引入新模型。

## 架构

```
输入文档（≤512 token，截断或完整）
 ↓
RoBERTa-base forward（fine-tuned，低 lr 继续训练）
 ↓
每个 token 的 hidden state（768维）← 词级
 ↓ 按句子边界 mean pooling
每个句子的向量（768维）← 句子级
 ↓ 跨句 Transformer（2层，hidden=768，heads=8）+ attention
每个句子的 AI 概率权重 ← 可视化高亮
 ↓ 按段落边界 attention pooling（semeval 单段则直接聚合）
段落向量 ← 段落级（对真实文章有意义）
 ↓ 文档级 attention pooling
文档向量 → 分类头 → AI 概率 ← 文档级
```

## 层级对应

| 层级 | 语言学单位 | 操作 |
|---|---|---|
| 词 | token | RoBERTa self-attention（内部） |
| 句子 | sentence | 句内 token mean pooling → 跨句 Transformer |
| 段落 | paragraph | 句子 attention pooling（真实文章有效） |
| 文档 | document | 段落 attention pooling → 分类头 |

## 局限说明

- semeval outfox 数据 100% 单段，段落层级退化为句子直接聚合
- 长文（>512 token，占 31%）在 RoBERTa 输入端截断，跨 chunk 上下文丢失
- 截断问题是所有短上下文 encoder 的共同限制，不影响层级结构叙事

## 预期性能

- 比 RoBERTa-base（0.619 test macro-F1）好
- 目标区间 0.75-0.82
- 不保证超过 Fusion LightGBM（0.835），原因：剩余错误主要是 bloomz 文体与 human 的特征空间重叠

## 状态

备选，未实现。
