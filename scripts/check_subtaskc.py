import pandas as pd, re

df = pd.read_parquet('data/subtaskC/subtaskC/train-00000-of-00001.parquet')
SENT_RE = re.compile(r'[^.!?\n]+[.!?\n]*')

sample = df.iloc[0]
text = sample['text']
label = int(sample['label'])

print('text length (chars):', len(text))
print('label (boundary):', label)
print('label=0 count:', (df['label']==0).sum(), '→ 全AI')

print()
print('boundary 前（human）:', repr(text[:label][-80:]))
print('boundary 后（AI）:', repr(text[label:label+80]))

df['text_len'] = df['text'].str.len()
print()
print('avg text len:', round(df['text_len'].mean()))
print('avg label:', round(df['label'].mean()))
print('label/text_len ratio:', round((df['label']/df['text_len']).mean(), 2))

# 看看 label 是不是句子索引
sents = [s.strip() for s in SENT_RE.findall(text) if s.strip()]
print()
print('num sentences in sample:', len(sents))
print('sample label:', label, '— if sent index, boundary sentence:', sents[label] if label < len(sents) else 'out of range')
