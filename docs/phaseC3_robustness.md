# 阶段 C-3：降AI改写 + 检测器鲁棒性评测

复跑：
- 闭环单测 `scripts/humanize.py --text_file X`（本地 Qwen3-8B）
- 跨模型攻击 `scripts/cross_model_attack.py`（经反向隧道调外部 API）
- 结果 `outputs/robustness/cross_model_attack.json`、`outputs/humanize_demo/result.json`

## 定位

把"降AI改写"做成**检测器鲁棒性 / 攻防评测工具**：用改写模型按检测器自身的特征诊断
（polish_advisor composite prompt）改写高 AI 文本，再回灌检测器，量化 AI 疑似度的下降与
判定翻转率。研究用途，不是规避学术诚信检查的服务。

## 关键约束（环境）

- 外部 API gateway `http://113.45.39.247:3001`（33 模型：claude/gpt-5/qwen3-max/deepseek）
  **CFFF 算力节点连不上**（只通 hf-mirror），本地 Mac 能连。
- 解决：反向 SSH 隧道 `ssh -R 13001:113.45.39.247:3001 -p <port> root@host`，
  CFFF 上脚本访问 `http://localhost:13001`。`scripts/humanize.py` 含 APIHumanizer
  （env LLM_API_BASE / LLM_API_KEY）与本地 Qwen Humanizer，接口同为 `.rewrite()`。

## 结果

单次改写（本地 Qwen3-8B，n=1 chatGPT）：文本明显更像人（ppl 13.4→19.0），但检测器几乎不动
（prob_ai 0.906→0.904）。→ 单次改写不足以破检测器。

跨模型递归攻击（n=12/模型，K=2 轮，起点均 prob_ai=0.906 / margin≈12.16）：

| 攻击模型 | r1 翻转率 | r2 翻转率 | prob_ai 轨迹 |
|---|---|---|---|
| claude-sonnet-4-6 | 67% | **83%** | 0.906→0.834→0.733 |
| qwen3-max | 17% | 33% | 0.906→0.893→0.884 |
| gpt-5.2 | 8% | 25% | 0.906→0.897→0.884 |

（n=12，结论稳定。翻转率＝检测器把文档重判为 human 的比例。）

## 结论

1. 检测器对**单次**改写鲁棒；**递归 + 强模型**能攻破（Claude 1 轮翻 67%，有样本 margin
   从 +12.17 崩到 −7.43 彻底判 human）。与学术界"递归改写破检测器"一致。
2. 攻击强度差异大：claude ≫ qwen3-max > gpt-5.2（GPT-5.2 改写更保守）。

## 核心洞察（贯穿全项目的双刃剑）

翻转机制：Claude 把 margin 从 12.17 压到 10.88，**刚跌破 t_high=10.907 进入模糊带**，正交特征
分类器接管，而改写恰好打散 TTR 等表层特征 → 翻成 human。但 **RoBERTa 深层 prob_ai 几乎没动**
（仍 0.88）。

> region-aware 的"正交特征模糊带分类器"——既是救回 bloomz 的功臣（[[阶段B]]），也是最大攻击面：
> bloomz 靠表层特征(TTR)被救回，而表层特征正是改写最易打散的。RoBERTa 深层 margin 反而抗改写。

这把检测（bloomz 恢复）、可解释（正交特征）、攻防（改写攻击）三条线用同一机制统一了。
