# 简历描述（Résumé / CV Description）

以下提供多个版本，按篇幅和场景取用。所有数字均可通过仓库中的代码复现。

---

## 一、中文版

### 版本 A：标准条目式（适合技术岗 / 量化实习）

**基于随机矩阵理论去噪的协方差矩阵与层次风险平价组合** —— 独立研究项目

- 用 **Marchenko–Pastur 定律**从样本协方差矩阵中分离信号与噪声，实现 **7 种协方差估计量**（样本 / Ledoit–Wolf 线性 / 常数相关收缩 / RMT 硬阈值 / RMT 软阈值 / **RIE 非线性收缩** / PCA 因子模型）与 **6 种组合优化器**（最小方差 / 风险平价 / **层次风险平价 HRP** / 最大分散化 / 等权 / 逆方差）
- 建模发现美国行业 ETF 面板（N=16, T=252）中 **7/16 个特征值突破 MP 上界，承载 91.9% 的横截面方差**；最大特征值达上界 **29.5 倍**，其特征向量在 79 个滚动窗口中重叠度 **0.9987**——市场共同因子
- **提出并验证了评估范式的转换**：已实现夏普比率在三个市场上给出三个不同「赢家」（差异 <0.06，小于标准误），无法区分估计量；改用 **Patton (2011) QLIKE 损失**评估事前风险预测后，**RIE 在三个市场一致排名第一，损失降低 14.7% / 15.4% / 20.5%，且改善幅度随 $q=N/T$ 单调上升**
- 实现自洽 **QuEST Stieltjes 不动点迭代**求解非线性收缩；诊断并修复了朴素实现中正则化参数主导结果的问题（迹比从 1.02 崩至 0.0004），并纠正了「去噪必为单调收缩」的认知误区（RIE 实际会**上抬** bulk、**下压**顶部）
- 用蒙特卡洛在三类已知结构上验证 Ledoit–Wolf 常数相关估计量（真单位阵 → 强度 0.95；真等相关 → 强度 0.00；因子结构 → 代数最优 0.444 vs 蒙特卡洛真值 0.259）；撰写 **235 个单元测试**，其中抓出并修复了回撤序列遗漏首期回撤的真实缺陷
- 完成 **6 类稳健性扫描**（窗口 / 频率 / 阈值 / 成本 / 噪声尺度 / 跨市场），覆盖 3 个市场面板；输出 **22 张图表**与 10+ 节工作论文
- 技术栈：Python, NumPy, pandas, SciPy, scikit-learn, cvxpy, Matplotlib, pytest, akshare

### 版本 B：一段话（适合简历空间紧张时）

独立完成了一个从统计物理到量化资产配置的完整研究项目：用 Marchenko–Pastur 定律对协方差矩阵做随机矩阵去噪，实现了 7 种协方差估计量 × 6 种组合优化器的统一回测框架，并在 3 个市场面板（美国/中国行业 ETF、美股高维个股）上做滚动样本外检验。核心贡献是论证了**评估高维协方差估计量应当直接评估其风险预测而非绕道评估收益**——在此判据下，RMT 非线性收缩 (RIE) 相对样本矩阵将 QLIKE 损失降低 15–21% 且跨市场一致，而已实现夏普比率完全无法区分各估计量。项目含 235 个单元测试与 22 张图表。

### 版本 C：项目符号极简版（3 行）

- **随机矩阵理论去噪**：用 MP 定律识别信号/噪声边界，实现 7 种协方差估计量（含 QuEST 非线性收缩 RIE 与 HRP）
- **核心发现**：RIE 在 3 个市场的风险预测 QLIKE 损失上一致降低 **15–21%**，且改善随纵横比 $q=N/T$ 单调上升；而已实现夏普比率无法区分估计量
- **工程**：统一回测框架（成本+换手）、6 类稳健性扫描、235 个单元测试、22 张图表、10+ 节工作论文；Python/NumPy/SciPy/cvxpy

---

## 二、English Version

### Version A: Bullet points

**Random-Matrix-Theory Denoising of Covariance Matrices with Hierarchical Risk Parity** — Independent Research Project

- Applied the **Marchenko–Pastur law** to separate signal from noise in sample covariance matrices; implemented **7 covariance estimators** (sample, Ledoit–Wolf linear, constant-correlation shrinkage, RMT hard/soft thresholding, **RIE nonlinear shrinkage**, PCA factor model) and **6 portfolio allocators** (minimum variance, risk parity, **HRP**, maximum diversification, equal weight, inverse variance)
- Characterised the spectrum of a 16-asset US sector-ETF panel (T=252): **7 of 16 eigenvalues escape the MP band yet carry 91.9% of cross-sectional variance**; the top eigenvalue is **29.5×** the MP edge and its eigenvector has **0.9987** overlap across 79 rolling windows
- **Demonstrated that the evaluation metric determines the conclusion**: realised Sharpe ranks three different "winners" across three markets (spread <0.06, below the standard error) and cannot discriminate between estimators; using the **Patton (2011) QLIKE loss** on ex-ante risk forecasts, **RIE ranks first in all three markets, cutting loss by 14.7% / 15.4% / 20.5%**, with the gain increasing monotonically in the aspect ratio $q=N/T$
- Implemented a **self-consistent QuEST Stieltjes fixed-point** solve for nonlinear shrinkage, diagnosing and fixing a naive implementation in which the regularisation parameter dominated the output (trace ratio collapsing from 1.02 to 0.0004), and corrected the misconception that denoising must be monotone shrinkage (RIE **inflates** the bulk and **shrinks** the top)
- Validated the constant-correlation estimator against Monte-Carlo ground truth on three analytic structures (identity → intensity 0.95; exact equicorrelation → 0.00; factor structure → algebraic 0.444 vs MC truth 0.259); wrote **235 unit tests**, one of which uncovered a real defect whereby the drawdown series silently dropped the first period's drawdown
- Delivered **6 robustness sweeps** (window / frequency / threshold / cost / noise-scale estimator / cross-market) across 3 panels, **22 figures**, and a 10+ section working paper
- Stack: Python, NumPy, pandas, SciPy, scikit-learn, cvxpy, Matplotlib, pytest, akshare

### Version B: One paragraph

Independently completed an end-to-end research project spanning statistical physics and quantitative asset allocation: used the Marchenko–Pastur law to denoise covariance matrices and built a unified rolling out-of-sample framework covering 7 covariance estimators × 6 portfolio allocators across three market panels. The central contribution is the argument that **high-dimensional covariance estimators should be evaluated on their risk forecasts rather than on the returns they generate** — under that criterion RMT nonlinear shrinkage (RIE) reduces QLIKE loss by 15–21% versus the sample matrix, consistently across all three panels, whereas realised Sharpe ratios cannot distinguish between the estimators at all. The repository ships 235 unit tests and 22 publication-quality figures.

### Version C: Three-line summary

- **RMT denoising**: MP-law signal/noise separation; 7 covariance estimators (incl. QuEST nonlinear shrinkage and HRP) × 6 allocators
- **Key result**: RIE cuts risk-forecast QLIKE loss by **15–21%** consistently across 3 markets, with gains increasing in $q=N/T$; realised Sharpe ratios cannot discriminate between estimators
- **Engineering**: unified cost-aware backtest engine, 6 robustness sweeps, 235 unit tests, 22 figures, 10+ section working paper; Python/NumPy/SciPy/cvxpy

---

## 三、面试可能被追问的问题（自备答案）

**Q: 为什么去噪在已实现夏普上看不出效果？**
A: 已实现夏普是两个含噪量之比。7 年样本下月度平均收益的标准误约 0.04（年化 0.14），而估计量之间的夏普差异只有 0.06——信噪比不足一个数量级。风险预测 $\hat h = w^\top \tilde\Sigma w$ 是一个标量，平均掉 78 个月度观测后，真实的相对风险水平差异可以浮现。

**Q: RIE 为什么会「上抬」一些特征值？这不是把噪声放大了吗？**
A: 不是。非线性收缩的最优解在谱形上有特定结构：bulk 中偏大的特征值被样本估计**低估**（因为有限样本会把中间方向的方差低估），而顶部特征值被**高估**（最大样本特征值吸收了部分噪声）。所以正确的最优收缩是双向的。我最初的实现加了一句 `min(new, old)` 的「保险」，反而把 RIE 的精华抹掉了，退化成样本矩阵。

**Q: 为什么 Ledoit–Wolf 反而比样本矩阵差？**
A: 它向单位阵做**均匀**收缩，压低了所有方差项，引入系统性风险低估（偏差 +3.1%）。它降低条件数的代价是引入偏差，而在 QLIKE 下偏差是要被惩罚的。这是一个真实的 trade-off，不是实现问题。

**Q: 你的结果在别的市场成立吗？**
A: 在 QLIKE 上成立——美国行业 ETF、中国行业 ETF、美股高维个股三个面板上 RIE 都是一致第一。在夏普上不成立——三个市场的赢家各不相同（factor model / ledoit_wolf / sample）。这个反差本身就是我的论点。

**Q: 最大的局限是什么？**
A: $q$ 的上限只做到 0.25。RMT 的威力在 $q \to O(1)$ 时充分展现。我的结论「改善随 $q$ 单调上升」预测在 $N=400$、$T=252$（$q \approx 1.6$）时应该看到远超 20% 的改善——但受数据与算力限制没能验证。这是我列出的首要延伸方向。

---

## 四、关键词（用于简历关键词匹配）

`量化金融` `Quantitative Finance` `随机矩阵理论` `Random Matrix Theory` `Marchenko-Pastur` `协方差估计` `Covariance Estimation` `Ledoit-Wolf 收缩` `Nonlinear Shrinkage` `RIE` `QuEST` `最小方差组合` `Minimum Variance` `风险平价` `Risk Parity` `ERC` `层次风险平价` `Hierarchical Risk Parity` `HRP` `López de Prado` `最大分散化` `Maximum Diversification` `投资组合优化` `Portfolio Optimisation` `cvxpy` `滚动样本外回测` `Walk-Forward Backtest` `交易成本` `Turnover` `QLIKE` `Patton` `波动率预测` `Volatility Forecasting` `特征值去噪` `Eigenvalue Denoising` `有效下注数` `Effective Number of Bets` `Python` `NumPy` `pandas` `SciPy` `scikit-learn` `Matplotlib` `pytest`
