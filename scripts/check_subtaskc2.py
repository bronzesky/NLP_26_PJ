import pandas as pd, re

df = pd.read_parquet('data/subtaskC/subtaskC/train-00000-of-00001.parquet')

# label 是字符级 boundary：text[:label] = human, text[label:] = AI
# label=0 → 全AI；label=len(text) → 全human
# avg label/text_len = 0.04，说明大部分文章开头很短的 human 部分，后面都是AI

# 转换成句子级标注
SENT_RE = re.compile(r'[^.!?\n]+[.!?\n]*')

def text_to_sent_labels(text, boundary):
    sents = [s.strip() for s in SENT_RE.findall(text) if s.strip()]
    labels = []
    pos = 0
    for s in sents:
        # 找这个句子在原文中的位置
        idx = text.find(s, pos)
        if idx == -1:
            idx = pos
        mid = idx + len(s) // 2
        labels.append(0 if mid < boundary else 1)  # 0=human, 1=AI
        pos = idx + len(s)
    return sents, labels

# 样例
for i in range(3):
    row = df.iloc[i]
    sents, labels = text_to_sent_labels(row['text'], int(row['label']))
    print(f'doc {i}: boundary={row["label"]}, {len(sents)} sents')
    for j, (s, l) in enumerate(zip(sents[:5], labels[:5])):
        print(f'  sent{j} [{l}]: {s[:60]}...')
    print()

# 统计：多少文档是纯AI (label=0)
print('label=0 (全AI):', (df['label']==0).sum(), '/', len(df))
print('混合文档:', (df['label']>0).sum())

# 平均每篇文档有多少 AI 句子
def count_ai_sents(row):
    _, labels = text_to_sent_labels(row['text'], int(row['label']))
    return sum(labels), len(labels)

sample_docs = df[df['label'] > 0].head(100)
ai_counts = sample_docs.apply(count_ai_sents, axis=1)
print('混合文档中 avg AI sents:', round(sum(x[0] for x in ai_counts)/len(ai_counts), 1))
print('混合文档中 avg total sents:', round(sum(x[1] for x in ai_counts)/len(ai_counts), 1))
