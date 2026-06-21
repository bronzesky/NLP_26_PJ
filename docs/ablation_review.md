# Ablation 复盘 + 调整结论（2026-06-21）

基于全套 ablation（stage1 / B1 / A3 / C1 / E-series / P1优化）的批判性复盘。

## 一、不符合预期的发现

| # | 发现 | 证据 | 影响 |
|---|---|---|---|
| 1 | TTR 单特征 = 6维全量 | LOO: 剔TTR唯一致跌; TTR-only=0.857 | 分类器本质是1维TTR判别,6维是过度修饰 |
| 2 | P2(0.854)≈P1(0.857) | 重测 | "ceiling"框架无信息量,应删 |
| 3 | bloomz @0.5召回0.963,非0.096 | B1校准表 | 0.096是高阈值下的数字;正文必须讲清阈值依赖 |
| 4 | 定点≈随机(降AI) | E系列 0.14 vs 0.16 | 遮挡定位在递归下无优势 |
| 5 | LGBM未过拟合(gap+0.007) | A5 | 因带内任务≈1维,与HSAD负面结果叙事需区分 |
| 6 | Isotonic拉低AUROC 0.939 | B1 | 非参校准在小dev上有害 |

## 二、可优化的（已验证）

- **human_spec 0.71 是真瓶颈**(剩余误差大头,模糊带29% human误判)。**P1实验结论:不可调** —— 扫ortho阈值/class_weight都撬不动(0.709-0.712),证明是bloomz↔human在TTR空间的本质重叠下界,非超参伪影。**诚实负面结果。**
- 主结果0.857 vs 旧SOTA 0.835仅+0.022,说服力靠per-generator(bloomz 0.10→1.0)+诊断叙事撑。

## 三、不需要的（删）

- 正交分类器的5个非TTR特征 → 坍缩为1维TTR(B1验证:0.8570=0.8569,免费)
- 温度校准对**判定**冗余(B1证明被阈值吸收),仅用于报告概率/ECE
- P2 ceiling叙事(≈P1)
- 检测侧(Fusion)手工特征(全去仅-0.003)

## 四、可替换的（验证后定）

- **6维LR → 1维TTR logistic**: 采纳(E1,免费且更优雅)
- 降AI定点 → 整篇递归(攻击更强); 定点保留作可视化
- 双区(去t_low): **不采纳** —— A1实测+0.0002噪声级,去t_low丢弃RoBERTa强置信human判断,保留3区更稳健有原则
- 仲裁器已是LR(最简),不换

## 五、调整后的方法（最终形态）

> RoBERTa 温度校准 margin：≥t_high 直判 AI，<t_low 直判 human，
> 中间模糊带用**单个 TTR→P(AI) logistic** 裁决。判定不依赖温度（温度仅供报告概率）。

P1实测数据：`outputs/ablation/p1_optimize.json`。1维TTR配置 macro-F1=0.8570、bloomz 0.998、human_spec 0.711。

## 六、执行项

- [x] 验证 E1(1维TTR免费)、P1(human_spec不可调)、A1(双区无收益)
- [ ] E1重构: fit_deploy_model + detector_pipeline 改1维TTR
- [ ] 论文: 删P2 ceiling; 加bloomz阈值依赖说明; human_spec瓶颈作诚实limitation; 方法降维表述
