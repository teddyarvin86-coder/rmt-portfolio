"""Backtesting engine and performance metrics."""

from rmt_portfolio.backtest.engine import (
    FREQUENCY_DAYS,
    TRADING_DAYS_PER_YEAR,
    BacktestConfig,
    BacktestResult,
    rebalance_dates,
    run_backtest,
    run_matrix,
)
from rmt_portfolio.backtest.metrics import (
    METRIC_LABELS,
    PerformanceMetrics,
    annualised_return,
    annualised_volatility,
    calmar_ratio,
    compute_metrics,
    conditional_value_at_risk,
    drawdown_series,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    turnover,
    ulcer_index,
    value_at_risk,
)

__all__ = [
    "BacktestConfig", "BacktestResult", "run_backtest", "run_matrix",
    "rebalance_dates", "FREQUENCY_DAYS", "TRADING_DAYS_PER_YEAR",
    "PerformanceMetrics", "compute_metrics", "METRIC_LABELS",
    "annualised_return", "annualised_volatility", "sharpe_ratio",
    "sortino_ratio", "max_drawdown", "drawdown_series", "calmar_ratio",
    "value_at_risk", "conditional_value_at_risk", "ulcer_index", "turnover",
]
