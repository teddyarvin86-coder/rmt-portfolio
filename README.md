# RMT Portfolio — 随机矩阵理论去噪下的协方差估计与组合构建

> **用 Marchenko–Pastur 定律识别并清理样本协方差矩阵中的噪声，构建最小方差 / 风险平价 / 分层风险平价组合，并在滚动样本外回测中与 Ledoit–Wolf 收缩、原始样本矩阵正面对比。**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-335%20passing-brightgreen.svg)](tests/)

---

## 一句话结论

**在一个 16 资产的行业 ETF 面板上，7 个特征值突破 Marchenko–Pastur 上界并承载 91.9% 的方差；在风险预测质量（QLIKE 损失）上，RMT 非线性收缩 (RIE) 在三个市场一致排名第一，相对样本矩阵降低 15–21% 的损失——而它在已实现夏普比率上并不突出。**
评估协方差矩阵，应当直接评估它的风险预测，而不是绕道看它带来的收益。

---

## 目录

- [快速开始](#快速开始)
- [这个项目在做什么](#这个项目在做什么)
- [核心结果](#核心结果)
- [方法总览](#方法总览)
- [项目结构](#项目结构)
- [完整复现流程](#完整复现流程)
- [配置说明](#配置说明)
- [开发过程中发现并修复的四个问题](#开发过程中发现并修复的四个问题)
- [局限与延伸](#局限与延伸)
- [引用与许可](#引用与许可)

---

## 快速开始

```bash
git clone https://github.com/your-username/rmt-portfolio.git
cd rmt-portfolio

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -e ".[all]"

# 一条命令跑通：谱分析 + 主回测 + 风险预测 + 全部图表
python experiments/run_figures.py --group all

# 看结果
ls figures/us_sector_etf/           # 22 张图
ls results/                         # 全部表格（CSV + Markdown 双份）
```

最小可运行示例：

```python
from rmt_portfolio.rmt.denoise import denoise
from rmt_portfolio.portfolio.optimizers import allocate, effective_number_of_bets
from rmt_portfolio.data.loader import load_returns

returns, _ = load_returns("us_sector_etf", start="2018-01-01", end="2025-12-31")

# 一步完成：估计 + RMT 去噪 + 全套诊断
res = denoise(returns, method="rmt_rie")

print(f"噪声尺度 σ          = {res.mp_law.sigma:.4f}")
print(f"MP 上界 λ₊          = {res.mp_law.lambda_plus:.4f}")
print(f"信号特征值个数       = {res.mp_law.n_signal_eigenvalues()}")
print(f"条件数  {res.info['cond_before']:.1f} → {res.info['cond_after']:.1f}")

# 去噪后的矩阵直接喂给任意优化器
port = allocate(res.covariance, method="hrp", max_weight=0.25)
print(f"有效下注数 ENB       = {effective_number_of_bets(port.weights, res.covariance):.2f}")
```

---

## 这个项目在做什么

马科维茨最优解 $w^\star \propto \Sigma^{-1}\mu$ 的根本困难在于 $\Sigma$ 不可观测。样本估计 $\hat\Sigma$ 在高维下病态：条件数动辄上千，最优权重对输入极度敏感，容易在少数方向上堆积巨额头寸。

**随机矩阵理论给出了一个精确的判决规则。** Marchenko 与 Pastur (1967) 证明：当 $N, T \to \infty$ 且 $N/T \to q$ 时，若真实协方差是单位阵（纯噪声），样本特征值必然落在紧支撑

$$
\lambda_\pm = \sigma^2\left(1 \pm \sqrt{q}\right)^2 , \qquad q = \frac{N}{T}
$$

之内。因此：

> 显著大于 $\lambda_+$ 的特征值承载**信号**；落在带内的特征值与噪声**不可区分**。

这把"如何正则化"这个模糊的统计问题，变成了"哪些特征值逃出了 MP 带"这个有闭式解的几何问题。

本项目实现了完整的实证链条：**数据管线 → RMT 谱分析 → 7 种协方差估计 → 6 种组合优化 → 滚动样本外回测（含成本与换手）→ 风险预测质量评估 → 稳健性扫描 → 图表与论文**。

---

## 核心结果

### 1. 谱结构高度稀疏（美国行业 ETF，$N=16$, $T=252$）

| 排名 | 特征值 | 方差占比 | 累计 | 突破 $\lambda_+$ | 相对 $\lambda_+$ |
|---:|---:|---:|---:|:--|---:|
| 1 | 8.5999 | 53.75% | 53.75% | ✅ | **29.5×** |
| 2 | 2.2957 | 14.35% | 68.10% | ✅ | 7.88× |
| 3 | 1.2060 | 7.54% | 75.63% | ✅ | 4.14× |
| 4 | 1.0367 | 6.48% | 82.11% | ✅ | 3.56× |
| 5 | 0.6084 | 3.80% | 85.92% | ✅ | 2.09× |
| 6 | 0.5595 | 3.50% | 89.41% | ✅ | 1.92× |
| 7 | 0.4011 | 2.51% | **91.92%** | ✅ | 1.38× |
| 8 | 0.2811 | 1.76% | 93.68% | ❌ | 0.97× |

**7 / 16 个特征值逃出噪声带，承载 91.9% 的方差。** 第 8 名跌入带内——那 8% 的方差不是"小信号"，而是**无法与噪声区分的方向**。
最大特征向量的滚动重叠度 **0.9987**（79 个窗口）——市场共同因子几乎完全不变。

### 2. 风险预测质量：RIE 是唯一的赢家（本文核心结果）

以 Patton (2011) **QLIKE 损失**评估事前风险预测（唯一在波动率代理含噪时仍稳健的排序标准）：

| 市场 | $N$ | $q$ | 样本 QLIKE | **RIE QLIKE** | **改善** | 样本偏差 → RIE 偏差 |
|---|---:|---:|---:|---:|---:|:--|
| 美国行业 ETF | 16 | 0.063 | 0.51309 | **0.43789** | **−14.7%** | +2.4% → −3.5% |
| 中国行业 ETF | 16 | 0.063 | 0.60385 | **0.51087** | **−15.4%** | +7.7% → +1.4% |
| 美股高维个股 | 50 | 0.25 | 0.44648 | **0.35478** | **−20.5%** | +8.5% → −1.5% |

三个市场、两个不同的 $q$、两种市场制度，**RIE 一致第一，且改善幅度随 $q$ 单调上升。**
硬阈值与软阈值在低 $q$ 下几乎等同于样本矩阵；Ledoit–Wolf 反而最差（均匀收缩引入系统性偏差）。

### 3. 而已实现夏普比率几乎无法区分估计量

| 市场 | 赢家 | 跨度 |
|---|---|---|
| 中国行业 ETF | `factor_model` | 0.163 – 0.179 |
| 美国行业 ETF | `ledoit_wolf` | 0.386 – 0.447 |
| 美股高维个股 | `sample` | 0.899 – 1.026 |

**三个市场三个不同的赢家，差异都在 0.06 以内。** 已实现夏普是两个含噪量之比，在有限样本中信噪比不足一个数量级——用它评价高维估计方法，效果会被淹没。

### 4. 稳健性：阈值位置不重要，噪声尺度的估计方法很重要

| $\lambda_+$ 倍数 | 0.85 | 0.90 | 1.00 | 1.10 | 1.15 |
|---|---:|---:|---:|---:|---:|
| `rmt_hard` 净夏普 | 0.4020 | 0.3980 | 0.4004 | 0.4030 | 0.4048 |

**平坦**——跨度不足 0.007，没有"幸运参数"。

| $\sigma$ 估计方法 | `median_bulk` | `mean_bulk` | `max_eigen` | `cdf_fit` |
|---|---:|---:|---:|---:|
| `rmt_hard` 净夏普 | 0.4004 | 0.3994 | **0.2431** ⚠️ | 0.4011 |
| `rmt_hard` 换手 | 5.3% | 5.1% | **13.8%** ⚠️ | 5.1% |

**锚定最大特征值的 `max_eigen` 方法崩溃**——市场因子波动大，阈值被推得过高，几乎所有权重都被判为噪声。**谱的极值不能用来估计噪声尺度，必须用 bulk 的整体形状。**

---

## 方法总览

### 7 种协方差估计量

| 名称 | 说明 |
|---|---|
| `sample` | 样本协方差（基准）|
| `ledoit_wolf` | Ledoit–Wolf 线性收缩到单位阵 |
| `constant_corr` | Ledoit–Wolf (2003) 收缩到常数相关目标 |
| `rmt_hard` | 硬阈值：带内特征值 → 其均值 |
| `rmt_soft` | 软阈值：$w_j=(\lambda_j/\lambda_+)^\beta$ 凸组合 |
| `rmt_rie` | **RIE / QuEST 非线性收缩** |
| `factor_model` | PCA 因子模型：$k$ 个主成分 + 对角残差 |

### 6 种组合优化器

`equal_weight` · `inverse_variance` · `minimum_variance` (cvxpy) · `risk_parity`/ERC · `hrp`（分层风险平价，相关距离 $d_{ij}=\sqrt{(1-\rho_{ij})/2}$）· `max_diversification`

### 关键实现约定

**所有 RMT 估计量在相关矩阵尺度上运算，而非协方差尺度。**
Stieltjes 变换的平衡条件 $1 - q + q\lambda m$ 只在单位方差假设下成立。流程是：标准化 $C = D^{-1}\Sigma D^{-1}$ → 去噪 → 还原 $\tilde\Sigma = D\tilde{C}D$。不影响权重，但让数学正确、数值可读。

---

## 项目结构

```
rmt-portfolio/
├── config/
│   └── config.yaml                  # 全部实验参数（唯一配置源）
├── src/rmt_portfolio/
│   ├── rmt/
│   │   ├── mp_law.py                # MP 定律：上下界、密度、数值 CDF、σ 拟合
│   │   └── denoise.py               # 7 种估计量 + QuEST Stieltjes 不动点求解
│   ├── portfolio/
│   │   ├── optimizers.py            # 6 种优化器 + RC / ENB / DR 诊断
│   │   └── clustering.py            # 相关距离、层次聚类、准对角化
│   ├── backtest/
│   │   ├── engine.py                # 滚动样本外引擎（成本、无交易带、权重漂移）
│   │   └── metrics.py               # 完整指标组（几何年化、Sharpe、回撤、VaR/CVaR…）
│   ├── data/
│   │   ├── loader.py                # akshare 管线 + 拆股检测与修复
│   │   └── universes.py             # 三个市场面板定义
│   └── config.py                    # 配置加载与深度合并
├── experiments/
│   ├── run_spectrum_analysis.py     # 谱分析（第 5 节结果）
│   ├── run_main_backtest.py         # 主回测（第 6 节结果）
│   ├── run_risk_forecast.py         # 风险预测评估（第 7 节，核心结果）
│   ├── run_robustness.py            # 6 类稳健性扫描（第 8 节）
│   └── run_figures.py               # 22 张图，分 A–E 五组
├── tests/                           # 335 个测试
│   ├── test_mp_law.py               # 42 个
│   ├── test_denoise.py              # 73 个
│   ├── test_optimizers.py           # 68 个
│   ├── test_metrics.py              # 53 个
│   └── test_backtest.py             # 99 个
├── docs/
│   └── working_paper.md             # 完整工作论文（10+ 节）
├── figures/                         # 22 张 PNG + 对应数据 CSV
├── results/                         # 全部表格（CSV + Markdown 双份）
└── notebooks/
```

---

## 完整复现流程

```bash
# 1. 谱分析 —— 生成第 5 节的表与 A/B 组图
python experiments/run_spectrum_analysis.py --universe us_sector_etf
python experiments/run_figures.py --group A --group B

# 2. 主回测 —— 第 6 节
python experiments/run_main_backtest.py --universe us_sector_etf --tag main
python experiments/run_main_backtest.py --universe cn_sector_etf  --tag main
python experiments/run_main_backtest.py --universe us_single_stock --window 120 --tag highq

# 3. 风险预测评估 —— 第 7 节（核心结果）
python experiments/run_risk_forecast.py --universe us_sector_etf
python experiments/run_risk_forecast.py --universe cn_sector_etf
python experiments/run_risk_forecast.py --universe us_single_stock --window 120 --tag highq

# 4. 稳健性扫描 —— 第 8 节（6 类：window/frequency/threshold/cost/sigma/universe）
python experiments/run_robustness.py --sweep all --universe us_sector_etf

# 5. 全部图表
python experiments/run_figures.py --group all

# 6. 测试
pytest                                  # 全部 335 个
pytest tests/test_backtest.py -q        # 单文件
pytest --cov=src/rmt_portfolio          # 带覆盖率
```

| 测试文件 | 数量 | 覆盖内容 |
|---|---:|---|
| `test_mp_law.py` | 42 | MP 密度/分布、数值积分、Stieltjes 变换、信号数判定 |
| `test_denoise.py` | 73 | 7 个估计量、相关矩阵标度、RIE 自洽方程、η 不敏感性 |
| `test_optimizers.py` | 68 | 6 个分配器、风险贡献、ENB、规模/置换等变性 |
| `test_metrics.py` | 53 | 年化、回撤（含首期）、VaR/CVaR、换手率 |
| `test_backtest.py` | 99 | 调仓日程、成本模型、无未来信息、无交易带、批量网格 |

### 图表分组

| 组 | 主题 | 张数 | 代表图 |
|---|---|---:|---|
| **A** | MP 谱结构 | 5 | `fig_spectrum_mp`, `fig_eigenvalue_stability` |
| **B** | 对矩阵的影响 | 3 | `fig_eigen_shrinkage`, `fig_condition_numbers` |
| **C** | 回测结果 | 6 | `fig_equity_grid`, `fig_turnover_sharpe` |
| **D** | 风险模型质量 | 5 | `fig_forecast_qlike`, `fig_forecast_bias_time` |
| **E** | 稳健性 | 3 | `fig_robustness_window`, `fig_robustness_threshold` |

---

## 配置说明

全部参数集中在 `config/config.yaml`，无需改代码。关键项：

```yaml
data:
  universe: us_sector_etf       # us_sector_etf | cn_sector_etf | us_single_stock
  start: "2018-01-01"
  repair_splits: true           # 检测并修复未复权拆股

rmt:
  sigma_method: median_bulk     # median_bulk | mean_bulk | max_eigen | cdf_fit
  #                            # ⚠️ max_eigen 在真实数据上会崩溃，仅作对照

backtest:
  window: 252                   # 估计窗口（交易日）
  frequency: monthly            # weekly | monthly | quarterly
  cost_bps: 10                  # 单边比例成本
  max_weight: 0.25              # ⚠️ 强烈建议保留，见下文
  no_trade_band: 0.0

robustness:
  windows: [120, 252, 500, 756]
  cost_bps: [0, 5, 10, 20, 40]
  sigma_methods: [median_bulk, mean_bulk, max_eigen, cdf_fit]
```

### 为什么 `max_weight` 必须设上限

无约束的最小方差组合会把 **83%** 的权重压在单一债券 ETF (IEF) 上，ENB 塌缩到 **1.84**。数学上这是最优解，但它不是一个分散化组合，而是一个债券头寸加装饰。设 `max_weight: 0.25` 后 ENB 升至 **5.3**，组合才真正可投资。

---

## 开发过程中发现并修复的四个问题

这四个问题都产生过**看起来合理但错误**的数字，值得记录。

### 1. RIE 的 Stieltjes 变换不是自洽求解

**症状：** `rmt_rie` 给出的条件数 1187.5，权重与样本矩阵逐位相同。

**根因：** 朴素实现直接算 $m(z)=N^{-1}\sum_j (z-\lambda_j+i\eta)^{-1}$。在离散谱上这个和对每个 $z=\lambda_i$ 都是奇异的，结果完全由正则化参数 $\eta$ 支配——扫描显示 $\eta$ 从 $10^{-1}$ 降到 $10^{-4}$ 时迹比从 1.02 崩到 0.0004。**一个本不该有物理意义的参数决定了全部结果。**

**修复：** 实现 QuEST 的**自洽不动点**

$$
m(z) = \frac{1}{N}\sum_{j=1}^{N}\left[z - \frac{\lambda_j}{1 - q + q\,\lambda_j\, m(z)}\right]^{-1}
$$

迭代求解（500 次上限，容差 $10^{-12}$）。验证：纯噪声上迹比稳定在 **0.971**；$\eta \in [10^{-4}, 10^{-2}]$ 内**不敏感**。

**附带发现：** 初版加了一句 `new_vals = min(new_vals, vals)` 的"保险"。**这是错的**——非线性收缩**不是**单调收缩，它会**上抬** bulk（中间部分被低估）并**下压**顶部（最大特征值被噪声抬高）。删掉这个 clip 之后 RIE 才真正工作。现在实测 12 个特征值被上抬、4 个被下压。

### 2. Ledoit–Wolf 常数相关目标的强度恒为 0

**症状：** 收缩强度 $\hat\delta$ 恰好为 0，估计量与样本矩阵完全相同。

**根因：** $\hat\rho$ 的公式高估约 6 倍，导致 $\hat\pi - \hat\rho < 0$，被截断到 0。

**修复：** 实现 Ledoit–Wolf (2003) 的**双指标 + 三指标**分解。用蒙特卡洛在三种已知结构上验证：真单位阵 → $\delta \approx 0.95$（目标正确）；真等相关 → $\delta \approx 0.00$（目标精确）；因子结构 → 代数最优 0.444 vs 蒙特卡洛真值 0.259。**定性顺序与教科书完全一致。**

### 3. 回撤序列漏掉了第一期的回撤 ⚠️

**症状：** 一个以 $-20\%$ 开局的序列，最大回撤报成 0。

**根因：** `wealth = cumprod(1 + r)` 从 $1+r_0$ 开始，**初始财富 1.0 从未进入滚动峰值**，所以第一期的回撤被静默丢弃。在 10 期连续 $-10\%$ 的例子中，报出 0.6126 而非正确的 0.6513。

**修复：** 在财富序列前补上起始值 1.0，输出再对齐回 `returns` 的长度。**低估风险指标是最危险的方向**，所以这个修复附带了专门的回归测试。

### 4. 60/40 基准在单股票面板上产生 34.5% 波动率

**根因：** 回退逻辑用了 `idxmax`/`idxmin`，产生 `0.6 × NVDA + 0.4 × 波动最低个股`。

**修复：** 单股票面板改为「等权股票篮子 + 波动最低的三分之一作为债券腿」，使 60/40 在任何面板上都保持股债结构。

### 另外两个方法论要点

- **MP 的解析 CDF 被删除。** 手写的闭式解在 $q=1/4$ 时给出 $F(\lambda_+)=1.5$、$F(\lambda_-)=0.5$——完全错误，被单元测试抓出。由于该路径未被生产代码使用，直接**删除**而不是修补，保留经过验证的数值积分器。
- **特征值偏差回归推翻了朴素假设。** 所有 42 个单元的 $R^2$ 都在 0.0005–0.040 之间——条件数几乎不能解释预测偏差。机制是：条件数高时最小方差类优化器会自动减少对病态方向的暴露，自我对冲了病态性。**病态矩阵的危险不在于预测偏差，而在于权重的不稳定性。**

---

## 局限与延伸

**局限**

1. **样本期有限。** 美国面板 7.5 年（2018–2025），月度频率下仅 78 个 QLIKE 观测。已用跨市场一致性加强结论，但单市场置信区间仍宽。
2. **单一数据源。** `akshare` 的复权处理不一致是已知风险。已做拆股检测与修复，并验证近期无残留异常，但更早的历史数据可能仍有污染。
3. **$q$ 的上限。** 最大只做到 $q=0.25$（$N=50$, $T=120$）。RMT 的威力在 $q \to O(1)$ 时充分展现（如 $N=500$, $T=252$）。**理论上那里收益应大得多**——这是最值得期待的延伸。
4. **未建模预期收益。** 全部组合是"仅风险"型，刻意避免污染风险模型的评估，但也就没有触及均值—方差框架的另一半。
5. **成本模型简化。** 比例成本 + 无交易带，未建模市场冲击的非线性、买卖价差、融券成本。

**延伸方向**

- **推高 $q$**：$N \approx 400$ 的美股全样本，$T=252$，考察 $q \approx 1.6$。若"改善随 $q$ 单调上升"成立，应看到远超 20% 的 QLIKE 改善。
- **尾部风险**：把 QLIKE 换成 VaR/CVaR 的预测评估（Christoffersen 检验），看 RIE 是否也改善尾部预测。
- **复合估计量**：RIE 去噪 + Ledoit–Wolf 收缩；RIE 去噪 + HRP（HRP 用原始相关距离，而 RIE 修改的正是小特征值方向，二者结构互补）。
- **时变 $\sigma$**：引入 GARCH 型时变波动率后再做 RMT。

---

## 引用与许可

**核心文献**

- Marchenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices.
- Laloux, L., Cizeau, P., Bouchaud, J.-P., & Potters, M. (1999). Noise dressing of financial correlation matrices. *PRL*.
- Ledoit, O., & Wolf, M. (2003, 2004, 2017, 2020). A well-conditioned estimator… / Honey, I shrunk the sample covariance matrix / Nonlinear shrinkage…
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*.
- López de Prado, M. (2016). Building diversification through hierarchical clustering. *JPM*.
- Maillard, S., Roncalli, T., & Teïletche, J. (2010). The properties of equally weighted risk contribution portfolios. *JPM*.

**许可：** MIT — 见 [LICENSE](LICENSE)

---

<p align="center">
<sub>全部数值结果均由本仓库代码直接生成，论文中的每一个数字都可通过上文的命令复现。</sub>
</p>
