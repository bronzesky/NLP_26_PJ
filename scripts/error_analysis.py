import pandas as pd, numpy as np, json

pred = pd.read_csv('outputs/fusion_lgbm_test/predictions.csv')
test_texts = {}
with open('data/official/test_sets/subtaskA_monolingual.jsonl') as f:
    for line in f:
        r = json.loads(line)
        test_texts[r['id']] = r['text']

wc_col = [len(test_texts.get(i,'').split()) for i in pred['id']]
text_col = [test_texts.get(i,'') for i in pred['id']]
pred = pred.assign(wc=wc_col, text=text_col)

fp = pred[(pred['correct']==False) & (pred['label']==0)]
fn = pred[(pred['correct']==False) & (pred['label']==1)]
tp = pred[(pred['correct']==True)  & (pred['label']==1)]

bl_fn = fn[fn['model']=='bloomz']
bl_tp = tp[tp['model']=='bloomz']
print(f'bloomz FN: n={len(bl_fn)}  wc mean={bl_fn["wc"].mean():.0f}  median={bl_fn["wc"].median():.0f}')
print(f'bloomz TP: n={len(bl_tp)}  wc mean={bl_tp["wc"].mean():.0f}  median={bl_tp["wc"].median():.0f}')

print('\n=== bloomz FN 样例 ===')
for _, row in bl_fn.sample(3, random_state=42).iterrows():
    print(f'wc={row["wc"]}  prob_ai={row["prob_ai"]:.3f}')
    print(row['text'][:400])
    print('---')

print('\n=== FP 样例（human→AI）===')
for _, row in fp.sample(3, random_state=42).iterrows():
    print(f'wc={row["wc"]}  prob_ai={row["prob_ai"]:.3f}')
    print(row['text'][:400])
    print('---')

print('\n=== 错误概率分布 ===')
print(f'FP prob_ai: mean={fp["prob_ai"].mean():.3f}  min={fp["prob_ai"].min():.3f}  max={fp["prob_ai"].max():.3f}')
print(f'FN prob_ai: mean={fn["prob_ai"].mean():.3f}  min={fn["prob_ai"].min():.3f}  max={fn["prob_ai"].max():.3f}')
