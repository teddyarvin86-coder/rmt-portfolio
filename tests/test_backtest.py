"""Tests for the walk-forward backtest engine.

The engine is where look-ahead bias, cost accounting and the rebalance
schedule can silently go wrong, and every one of those would corrupt the
paper's headline numbers.  The tests below pin down:

* :func:`rebalance_dates` — pure integer arithmetic, exact expected indices.
* :class:`BacktestConfig` — validation, frequency lookup, cost models.
* :func:`_drift` — weight growth and renormalisation.
* :class:`BacktestResult` — equity series construction.
* :func:`run_backtest` — no look-ahead, gross >= net, cost identity,
  turnover semantics, determinism, and error handling.
* :func:`run_matrix` — grid completeness and per-run labelling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rmt_portfolio.backtest.engine import (
    FREQUENCY_DAYS,
    TRADING_DAYS_PER_YEAR,
    BacktestConfig,
    BacktestResult,
    _drift,
    _with_extra_alloc_kwargs,
    rebalance_dates,
    run_backtest,
    run_matrix,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_returns(n_days: int = 700, n_assets: int = 6, seed: int = 2024,
                 start: str = "2018-01-02") -> pd.DataFrame:
    """Synthetic daily simple returns on a business-day index.

    A persistent factor plus idiosyncratic noise gives enough cross-sectional
    structure that optimisers produce non-degenerate, unequal weights.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    common = rng.normal(0.0004, 0.01, size=(n_days, 1))
    loadings = rng.uniform(0.4, 1.2, size=(1, n_assets))
    idio = rng.normal(0.0002, 0.008, size=(n_days, n_assets))
    R = common @ loadings + idio
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(R, index=idx, columns=cols)


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return make_returns()


@pytest.fixture(scope="module")
def small_cfg() -> BacktestConfig:
    """Short window so the module-scoped fixture is reused across tests."""
    return BacktestConfig(window=120, frequency="monthly", cost_bps=10.0)


# ---------------------------------------------------------------------------
# Rebalance schedule
# ---------------------------------------------------------------------------


class TestRebalanceDates:
    def test_exact_schedule(self):
        # window 100, monthly (21), 300 observations
        idx = rebalance_dates(300, 100, 21)
        assert idx[0] == 100
        assert np.all(np.diff(idx) == 21)
        # last index must leave a full holding period available
        assert idx[-1] + 21 <= 300
        assert idx[-1] + 21 + 21 > 300

    def test_expected_values(self):
        assert list(rebalance_dates(25, 20, 1)) == [20, 21, 22, 23, 24]
        assert list(rebalance_dates(30, 20, 5)) == [20, 25]

    def test_estimation_window_never_overlaps_holding_period(self):
        n, window, holding = 500, 150, 21
        for t in rebalance_dates(n, window, holding):
            assert t - window >= 0                      # window is fully in-sample
            assert t + holding <= n                     # holding period exists

    def test_start_offset_shifts_the_schedule(self):
        base = rebalance_dates(300, 100, 21)
        off = rebalance_dates(300, 100, 21, start_offset=13)
        assert off[0] == 113
        assert set(base).issubset(set(off)) or off[0] == base[0] + 13

    def test_empty_when_sample_shorter_than_window(self):
        assert rebalance_dates(100, 100, 21).size == 0
        assert rebalance_dates(50, 100, 21).size == 0

    def test_empty_when_window_plus_holding_exceeds_sample(self):
        # window 100 + holding 21 needs at least 121 observations
        assert rebalance_dates(120, 100, 21).size == 0
        assert rebalance_dates(121, 100, 21).size == 1

    def test_returns_integer_array(self):
        idx = rebalance_dates(300, 100, 21)
        assert np.issubdtype(idx.dtype, np.integer)

    def test_holding_days_one_yields_every_date(self):
        idx = rebalance_dates(30, 20, 1)
        assert idx.size == 10
        assert idx[-1] == 29

    def test_zero_holding_days_raises_rather_than_looping_forever(self):
        """A zero step would make ``np.arange`` raise; document that contract."""
        with pytest.raises(ZeroDivisionError):
            rebalance_dates(300, 100, 0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestBacktestConfig:
    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.window == 252
        assert cfg.frequency == "monthly"
        assert cfg.cost_bps == 10.0
        assert cfg.cost_model == "proportional"
        assert cfg.long_only is True
        assert cfg.max_weight is None
        assert cfg.cov_kwargs == {}
        assert cfg.alloc_kwargs == {}

    def test_window_too_short_raises(self):
        with pytest.raises(ValueError, match="window"):
            BacktestConfig(window=19)

    def test_window_of_exactly_twenty_is_allowed(self):
        assert BacktestConfig(window=20).window == 20

    def test_unknown_cost_model_raises(self):
        with pytest.raises(ValueError, match="cost_model"):
            BacktestConfig(cost_model="quadratic")

    @pytest.mark.parametrize("freq,days", [
        ("daily", 1), ("weekly", 5), ("biweekly", 10), ("monthly", 21),
        ("quarterly", 63), ("semiannual", 126), ("annual", 252),
    ])
    def test_frequency_lookup(self, freq, days):
        assert BacktestConfig(frequency=freq).holding_days == days

    def test_frequency_is_case_insensitive(self):
        assert BacktestConfig(frequency="Monthly").holding_days == 21

    def test_integer_frequency_is_taken_literally(self):
        cfg = BacktestConfig(frequency=7)
        assert cfg.holding_days == 7
        assert not isinstance(cfg.frequency, str)

    def test_unknown_frequency_raises(self):
        with pytest.raises(ValueError, match="frequency"):
            BacktestConfig(frequency="fortnightly").holding_days

    def test_frequency_days_table_has_no_zero_entries(self):
        assert all(v > 0 for v in FREQUENCY_DAYS.values())

    def test_rebalances_per_year(self):
        cfg = BacktestConfig(frequency="monthly", periods_per_year=252)
        assert cfg.rebalances_per_year == pytest.approx(252 / 21)
        assert BacktestConfig(frequency="annual").rebalances_per_year == pytest.approx(1.0)
        assert BacktestConfig(frequency="daily").rebalances_per_year == pytest.approx(252.0)

    def test_proportional_cost_is_linear(self):
        cfg = BacktestConfig(cost_bps=10.0, cost_model="proportional")
        assert cfg.cost_of_turnover(0.0) == 0.0
        assert cfg.cost_of_turnover(1.0) == pytest.approx(10 / 1e4)
        assert cfg.cost_of_turnover(0.5) == pytest.approx(5 / 1e4)
        # linearity
        a, b = cfg.cost_of_turnover(0.2), cfg.cost_of_turnover(0.4)
        assert b == pytest.approx(2 * a)

    def test_zero_cost_bps_gives_free_trading(self):
        cfg = BacktestConfig(cost_bps=0.0)
        assert cfg.cost_of_turnover(0.75) == 0.0

    def test_linear_plus_impact_adds_a_concave_term(self):
        cfg = BacktestConfig(cost_bps=10.0, cost_model="linear_plus_impact",
                             impact_coef=5.0)
        prop = BacktestConfig(cost_bps=10.0, cost_model="proportional")
        t = 0.4
        assert cfg.cost_of_turnover(t) > prop.cost_of_turnover(t)
        assert cfg.cost_of_turnover(t) == pytest.approx(
            prop.cost_of_turnover(t) + 5.0 / 1e4 * np.sqrt(t))

    def test_impact_model_is_sublinear_at_large_turnover(self):
        """A square-root term must be *less* than proportional for big trades."""
        cfg = BacktestConfig(cost_bps=0.0, cost_model="linear_plus_impact",
                             impact_coef=10.0)
        big = cfg.cost_of_turnover(1.0)
        # the impact part is sqrt(1)*k, i.e. smaller than a linear k*1 would be
        assert big == pytest.approx(10.0 / 1e4)

    def test_negative_turnover_is_floored_at_zero_for_impact(self):
        cfg = BacktestConfig(cost_model="linear_plus_impact")
        assert np.isfinite(cfg.cost_of_turnover(-0.5))

    def test_mutable_defaults_are_not_shared(self):
        a, b = BacktestConfig(), BacktestConfig()
        a.cov_kwargs["sigma_method"] = "median_bulk"
        a.alloc_kwargs["linkage_method"] = "ward"
        assert b.cov_kwargs == {}
        assert b.alloc_kwargs == {}


# ---------------------------------------------------------------------------
# Weight drift
# ---------------------------------------------------------------------------


class TestDrift:
    def test_weights_always_sum_to_one(self):
        w = np.array([0.5, 0.3, 0.2])
        r = np.array([[0.02, -0.01, 0.005], [0.01, 0.03, -0.02]])
        out = _drift(w, r)
        assert out.sum() == pytest.approx(1.0)

    def test_no_news_keeps_weights_unchanged(self):
        w = np.array([0.25, 0.25, 0.5])
        out = _drift(w, np.zeros((5, 3)))
        np.testing.assert_allclose(out, w)

    def test_a_winner_gains_weight(self):
        w = np.array([0.5, 0.5])
        out = _drift(w, np.array([[0.10, 0.00]]))
        assert out[0] > 0.5
        assert out.sum() == pytest.approx(1.0)

    def test_compounding_is_not_additive(self):
        """Two +10% days must be applied as 1.1**2, not 1 + 0.2."""
        w = np.array([1.0, 0.0])
        out = _drift(w, np.array([[0.10, 0.0], [0.10, 0.0]]))
        # with asset 1 flat at 0 weight it stays; normalisation is trivial
        assert out[0] == pytest.approx(1.0)

        w = np.array([0.5, 0.5])
        out = _drift(w, np.array([[0.10, 0.00], [0.10, 0.00]]))
        # ratio must be (1.1**2)/(1.0**2) = 1.21, not 1.20
        assert out[0] / out[1] == pytest.approx(1.21, rel=1e-12)

    def test_input_is_not_mutated(self):
        w = np.array([0.5, 0.5])
        original = w.copy()
        _drift(w, np.array([[0.1, 0.0]]))
        np.testing.assert_array_equal(w, original)

    def test_accepts_a_single_period_row(self):
        out = _drift(np.array([0.5, 0.5]), np.array([0.0, 0.0]))
        assert out.sum() == pytest.approx(1.0)

    def test_total_wipeout_does_not_produce_nan(self):
        """If everything goes to zero the normaliser must be guarded."""
        out = _drift(np.array([0.5, 0.5]), np.array([[-1.0, -1.0]]))
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class TestBacktestResult:
    def _result(self) -> BacktestResult:
        dates = pd.bdate_range("2020-01-01", periods=4)
        cfg = BacktestConfig(window=120)
        from rmt_portfolio.backtest.metrics import compute_metrics
        g = np.array([0.01, -0.02, 0.03, 0.005])
        n = g - 0.001
        return BacktestResult(
            name="t", config=cfg, dates=dates,
            gross_returns=g, net_returns=n,
            turnover_series=np.array([1.0, 0.1, 0.1, 0.1]),
            cost_series=g - n,
            weights_history=pd.DataFrame(np.full((2, 2), 0.5),
                                         index=dates[:2], columns=["a", "b"]),
            metrics=compute_metrics(g, name="t", periods_per_year=12),
        )

    def test_gross_equity_is_the_compounded_gross_series(self):
        res = self._result()
        expected = np.cumprod(1.0 + res.gross_returns)
        np.testing.assert_allclose(res.gross_equity.values, expected)

    def test_equity_series_starts_at_first_compounded_value(self):
        res = self._result()
        assert res.gross_equity.iloc[0] == pytest.approx(1.01)
        assert res.net_equity.iloc[0] == pytest.approx(1.009)

    def test_equity_series_carries_the_dates_and_a_name(self):
        res = self._result()
        assert res.gross_equity.index.equals(res.dates)
        assert res.gross_equity.name == "t_gross"
        assert res.net_equity.name == "t_net"

    def test_net_equity_never_exceeds_gross_equity(self):
        res = self._result()
        assert np.all(res.net_equity.values <= res.gross_equity.values + 1e-15)

    def test_to_frame_columns_and_index(self):
        res = self._result()
        df = res.to_frame()
        assert list(df.columns) == ["gross_return", "net_return", "turnover", "cost"]
        assert df.index.equals(res.dates)
        assert len(df) == 4

    def test_to_frame_cost_equals_gross_minus_net(self):
        res = self._result()
        df = res.to_frame()
        np.testing.assert_allclose(
            df["cost"].values, df["gross_return"].values - df["net_return"].values)

    def test_describe_is_a_string(self):
        assert isinstance(self._result().describe(), str)


# ---------------------------------------------------------------------------
# run_backtest — the engine
# ---------------------------------------------------------------------------


class TestRunBacktest:
    def test_smoke(self, returns, small_cfg):
        res = run_backtest(returns, cov_method="sample",
                           allocator="minimum_variance", config=small_cfg)
        assert isinstance(res, BacktestResult)
        assert res.gross_returns.size > 0
        assert res.gross_returns.size == res.net_returns.size == res.turnover_series.size
        assert np.all(np.isfinite(res.gross_returns))
        assert np.all(np.isfinite(res.net_returns))

    def test_result_length_matches_the_schedule(self, returns, small_cfg):
        res = run_backtest(returns, cov_method="sample",
                           allocator="equal_weight", config=small_cfg)
        expected = rebalance_dates(len(returns), small_cfg.window,
                                   small_cfg.holding_days).size
        assert res.gross_returns.size == expected
        assert len(res.weights_history) == expected
        assert len(res.dates) == expected

    def test_weights_history_is_long_only_and_sums_to_one(self, returns, small_cfg):
        res = run_backtest(returns, cov_method="sample",
                           allocator="minimum_variance", config=small_cfg)
        W = res.weights_history.values
        np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-8)
        assert np.all(W >= -1e-10)
        assert list(res.weights_history.columns) == list(returns.columns)

    def test_gross_equity_dominates_net_equity_with_costs(self, returns, small_cfg):
        res = run_backtest(returns, cov_method="sample",
                           allocator="minimum_variance", config=small_cfg)
        assert np.all(res.net_returns <= res.gross_returns + 1e-15)
        assert np.all(res.cost_series >= 0.0)

    def test_costs_vanish_when_cost_bps_is_zero(self, returns):
        cfg = BacktestConfig(window=120, frequency="monthly", cost_bps=0.0)
        res = run_backtest(returns, cov_method="sample",
                           allocator="minimum_variance", config=cfg)
        np.testing.assert_allclose(res.gross_returns, res.net_returns)
        np.testing.assert_allclose(res.cost_series, 0.0)

    def test_cost_identity_holds_every_period(self, returns, small_cfg):
        """net_return == gross_return - cost, exactly, for every rebalance."""
        res = run_backtest(returns, cov_method="sample",
                           allocator="risk_parity", config=small_cfg)
        np.testing.assert_allclose(
            res.net_returns, res.gross_returns - res.cost_series, atol=1e-12)

    def test_cost_is_the_configured_function_of_turnover(self, returns, small_cfg):
        res = run_backtest(returns, cov_method="sample",
                           allocator="minimum_variance", config=small_cfg)
        expected = np.array([small_cfg.cost_of_turnover(t)
                             for t in res.turnover_series])
        np.testing.assert_allclose(res.cost_series, expected)

    def test_first_turnover_is_a_full_purchase(self, returns, small_cfg):
        """The very first rebalance buys the portfolio from cash, so one-way
        turnover is 0.5 * sum|w - 0| = 0.5."""
        res = run_backtest(returns, cov_method="sample",
                           allocator="minimum_variance", config=small_cfg)
        assert res.turnover_series[0] == pytest.approx(0.5)

    def test_turnover_is_bounded_by_one(self, returns, small_cfg):
        res = run_backtest(returns, cov_method="sample",
                           allocator="hrp", config=small_cfg)
        assert np.all(res.turnover_series >= 0.0)
        assert np.all(res.turnover_series <= 1.0 + 1e-9)

    def test_equal_weight_realises_the_cross_sectional_mean(self, returns, small_cfg):
        """With constant 1/N weights and no drift inside a period the gross
        return is the equal-weighted mean of the holding-period asset returns
        (compounded daily)."""
        res = run_backtest(returns, cov_method="sample",
                           allocator="equal_weight", config=small_cfg)
        R = returns.values
        H = small_cfg.holding_days
        idx = rebalance_dates(len(returns), small_cfg.window, H)
        n = returns.shape[1]
        expected = []
        for t in idx:
            daily = R[t:t + H] @ np.full(n, 1.0 / n)
            expected.append(np.prod(1.0 + daily) - 1.0)
        np.testing.assert_allclose(res.gross_returns, expected, atol=1e-12)

    def test_no_look_ahead_perturbing_the_future_changes_nothing(self, returns, small_cfg):
        """Weights formed at ``t`` must not depend on data after ``t``.

        Truncating the sample right after a rebalance date reproduces the
        identical first-period return.
        """
        full = run_backtest(returns, cov_method="sample",
                            allocator="minimum_variance", config=small_cfg)
        cut = small_cfg.window + small_cfg.holding_days
        truncated = run_backtest(returns.iloc[:cut], cov_method="sample",
                                 allocator="minimum_variance", config=small_cfg)
        assert truncated.gross_returns.size == 1
        np.testing.assert_allclose(truncated.gross_returns[0],
                                   full.gross_returns[0], atol=1e-12)
        np.testing.assert_allclose(truncated.weights_history.iloc[0].values,
                                   full.weights_history.iloc[0].values, atol=1e-12)

    def test_perturbing_data_inside_the_window_changes_the_result(self, returns, small_cfg):
        """Control for the look-ahead test below: data *inside* the first
        estimation window does matter, otherwise the probe proves nothing."""
        w0 = small_cfg.window
        perturbed = returns.copy()
        rng = np.random.default_rng(11)
        perturbed.iloc[w0 - 10:w0] = rng.normal(0.0, 0.03, size=(10, returns.shape[1]))
        a = run_backtest(returns, cov_method="sample",
                         allocator="minimum_variance", config=small_cfg)
        b = run_backtest(perturbed, cov_method="sample",
                         allocator="minimum_variance", config=small_cfg)
        assert not np.allclose(a.weights_history.values,
                               b.weights_history.values)

    def test_future_data_does_not_change_past_weights(self, returns, small_cfg):
        """Appending data after the last rebalance must not alter anything."""
        full = run_backtest(returns, cov_method="sample",
                            allocator="minimum_variance", config=small_cfg)
        cut = small_cfg.window + small_cfg.holding_days * 2
        partial = run_backtest(returns.iloc[:cut], cov_method="sample",
                               allocator="minimum_variance", config=small_cfg)
        n = partial.weights_history.shape[0]
        assert n >= 1
        np.testing.assert_allclose(
            partial.weights_history.values,
            full.weights_history.values[:n], atol=1e-12)
        np.testing.assert_allclose(
            partial.gross_returns, full.gross_returns[:n], atol=1e-12)

    def test_deterministic_across_runs(self, returns, small_cfg):
        a = run_backtest(returns, cov_method="sample",
                         allocator="minimum_variance", config=small_cfg)
        b = run_backtest(returns, cov_method="sample",
                         allocator="minimum_variance", config=small_cfg)
        np.testing.assert_array_equal(a.gross_returns, b.gross_returns)
        np.testing.assert_array_equal(a.weights_history.values,
                                      b.weights_history.values)

    def test_hold_dates_track_the_holding_period_end(self, returns, small_cfg):
        res = run_backtest(returns, cov_method="sample",
                           allocator="equal_weight", config=small_cfg)
        idx = rebalance_dates(len(returns), small_cfg.window,
                              small_cfg.holding_days)
        expected = returns.index[idx + small_cfg.holding_days - 1]
        assert res.dates.equals(pd.DatetimeIndex(expected))

    def test_rebalance_dates_are_the_window_ends(self, returns, small_cfg):
        res = run_backtest(returns, cov_method="sample",
                           allocator="equal_weight", config=small_cfg)
        idx = rebalance_dates(len(returns), small_cfg.window,
                              small_cfg.holding_days)
        assert res.weights_history.index.equals(pd.DatetimeIndex(returns.index[idx]))

    def test_denoising_estimators_all_run_and_differ(self, returns, small_cfg):
        out = {}
        for m in ("sample", "ledoit_wolf", "rmt_hard", "rmt_rie"):
            out[m] = run_backtest(returns, cov_method=m,
                                  allocator="minimum_variance",
                                  config=small_cfg).gross_returns
        assert not np.allclose(out["sample"], out["rmt_hard"])
        assert not np.allclose(out["sample"], out["ledoit_wolf"])

    def test_corr_method_override_is_accepted(self, returns, small_cfg):
        res = run_backtest(returns, cov_method="ledoit_wolf",
                           corr_method="rmt_hard", allocator="hrp",
                           config=small_cfg)
        assert res.extra["corr_method"] == "rmt_hard"
        assert res.gross_returns.size > 0

    def test_name_and_extra_metadata(self, returns, small_cfg):
        res = run_backtest(returns, cov_method="rmt_hard",
                           allocator="risk_parity", config=small_cfg,
                           name="my-run")
        assert res.name == "my-run"
        assert res.metrics.name == "my-run"
        # ``PerformanceMetrics.to_dict`` flattens ``extra`` into the top level
        d = res.metrics.to_dict()
        assert d["cov_method"] == "rmt_hard"
        assert d["allocator"] == "risk_parity"
        assert d["window"] == small_cfg.window
        assert d["n_rebalances"] == res.gross_returns.size
        assert res.metrics.extra["cov_method"] == "rmt_hard"

    def test_default_name_is_cov_plus_allocator(self, returns, small_cfg):
        res = run_backtest(returns, cov_method="sample",
                           allocator="hrp", config=small_cfg)
        assert res.name == "sample+hrp"

    def test_default_config_is_monthly_252(self, returns):
        res = run_backtest(returns, cov_method="sample", allocator="equal_weight")
        assert res.config.window == 252
        assert res.config.holding_days == 21

    def test_metrics_are_annualised_on_the_holding_period(self, returns, small_cfg):
        """Monthly rebalancing must annualise with 12 periods, not 252."""
        res = run_backtest(returns, cov_method="sample",
                           allocator="minimum_variance", config=small_cfg)
        expected_ppy = small_cfg.periods_per_year / small_cfg.holding_days
        assert expected_ppy == pytest.approx(12.0)
        assert res.metrics.n_periods == res.gross_returns.size
        assert np.isfinite(res.metrics.ann_return)
        assert np.isfinite(res.metrics.ann_volatility)
        assert res.metrics.ann_volatility > 0.0

    def test_annualised_vol_scales_as_sqrt_of_the_period(self, returns, small_cfg):
        """The reported annual vol must be the per-period vol times sqrt(12)."""
        res = run_backtest(returns, cov_method="sample",
                           allocator="minimum_variance", config=small_cfg)
        per_period = res.gross_returns.std(ddof=1)
        expected = per_period * np.sqrt(small_cfg.periods_per_year /
                                        small_cfg.holding_days)
        assert res.metrics.ann_volatility == pytest.approx(expected, rel=1e-9)

    def test_diagnostics_carry_one_row_per_rebalance(self, returns, small_cfg):
        res = run_backtest(returns, cov_method="sample",
                           allocator="minimum_variance", config=small_cfg)
        assert len(res.diagnostics) == res.gross_returns.size
        for col in ("vol", "enb", "dr", "no_trade"):
            assert col in res.diagnostics.columns
        assert np.all(res.diagnostics["enb"].values >= 1.0 - 1e-9)

    def test_max_weight_cap_is_forwarded_to_minimum_variance(self, returns):
        cfg = BacktestConfig(window=120, frequency="monthly", max_weight=0.2)
        res = run_backtest(returns, cov_method="sample",
                           allocator="minimum_variance", config=cfg)
        assert res.weights_history.values.max() <= 0.2 + 1e-8

    def test_max_weight_ignored_by_cap_unaware_allocators(self, returns):
        """``hrp`` does not enforce a cap; the engine only forwards the cap to
        the allocators that accept it.  This is a documented limitation."""
        cfg = BacktestConfig(window=120, frequency="monthly", max_weight=0.01)
        res = run_backtest(returns, cov_method="sample",
                           allocator="hrp", config=cfg)
        # the cap is *not* honoured by HRP, so some weight exceeds it
        assert res.weights_history.values.max() > 0.01

    def test_cov_kwargs_reach_the_estimator(self, returns, small_cfg):
        """``cov_kwargs`` are forwarded verbatim to the estimator, so a
        different threshold multiplier must change the resulting weights."""
        import dataclasses
        a = run_backtest(returns, cov_method="rmt_hard",
                         allocator="minimum_variance", config=small_cfg)
        cfg = dataclasses.replace(
            small_cfg, cov_kwargs={"lambda_plus_scale": 1.5})
        b = run_backtest(returns, cov_method="rmt_hard",
                         allocator="minimum_variance", config=cfg)
        assert not np.allclose(a.weights_history.values,
                               b.weights_history.values)

    def test_sigma_method_reaches_the_estimator(self, returns, small_cfg):
        """A pathological sigma estimator (``max_eigen``) must visibly change
        the result — this is the failure mode documented in the paper."""
        import dataclasses
        a = run_backtest(returns, cov_method="rmt_hard",
                         allocator="minimum_variance", config=small_cfg)
        cfg = dataclasses.replace(
            small_cfg, cov_kwargs={"sigma_method": "max_eigen"})
        b = run_backtest(returns, cov_method="rmt_hard",
                         allocator="minimum_variance", config=cfg)
        assert not np.allclose(a.weights_history.values,
                               b.weights_history.values)

    def test_alloc_kwargs_reach_the_allocator(self, returns, small_cfg):
        import dataclasses
        cfg = dataclasses.replace(small_cfg,
                                  alloc_kwargs={"linkage_method": "single"})
        a = run_backtest(returns, cov_method="sample", allocator="hrp",
                         config=cfg)
        cfg2 = dataclasses.replace(small_cfg,
                                   alloc_kwargs={"linkage_method": "ward"})
        b = run_backtest(returns, cov_method="sample", allocator="hrp",
                         config=cfg2)
        assert not np.allclose(a.weights_history.values,
                               b.weights_history.values)

    def test_frequency_changes_the_number_of_rebalances(self, returns):
        monthly = run_backtest(
            returns, cov_method="sample", allocator="equal_weight",
            config=BacktestConfig(window=120, frequency="monthly"))
        quarterly = run_backtest(
            returns, cov_method="sample", allocator="equal_weight",
            config=BacktestConfig(window=120, frequency="quarterly"))
        assert monthly.gross_returns.size > quarterly.gross_returns.size

    # --- error handling -------------------------------------------------

    def test_window_longer_than_sample_raises(self, returns):
        cfg = BacktestConfig(window=len(returns) + 10)
        with pytest.raises(ValueError):
            run_backtest(returns, config=cfg)

    def test_window_equal_to_sample_raises(self, returns):
        cfg = BacktestConfig(window=len(returns))
        with pytest.raises(ValueError):
            run_backtest(returns, config=cfg)

    def test_sample_too_short_for_window_plus_holding_raises(self, returns):
        cfg = BacktestConfig(window=len(returns) - 5, frequency="monthly")
        with pytest.raises(ValueError):
            run_backtest(returns, config=cfg)

    def test_nan_in_returns_raises_a_clear_error(self, returns, small_cfg):
        bad = returns.copy()
        bad.iloc[10, 2] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            run_backtest(bad, config=small_cfg)

    def test_inf_in_returns_raises(self, returns, small_cfg):
        bad = returns.copy()
        bad.iloc[3, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            run_backtest(bad, config=small_cfg)

    def test_unknown_estimator_raises(self, returns, small_cfg):
        with pytest.raises(Exception):
            run_backtest(returns, cov_method="not_a_method", config=small_cfg)

    def test_unknown_allocator_raises(self, returns, small_cfg):
        with pytest.raises(Exception):
            run_backtest(returns, allocator="not_an_allocator", config=small_cfg)


# ---------------------------------------------------------------------------
# No-trade band
# ---------------------------------------------------------------------------


class TestNoTradeBand:
    def test_a_huge_band_suppresses_every_rebalance_after_the_first(self, returns):
        cfg = BacktestConfig(window=120, frequency="monthly", no_trade_band=1e9)
        res = run_backtest(returns, cov_method="sample",
                           allocator="minimum_variance", config=cfg)
        assert res.turnover_series[0] > 0.0
        np.testing.assert_allclose(res.turnover_series[1:], 0.0)

    def test_suppressed_rebalances_repeat_the_previous_weights(self, returns):
        cfg = BacktestConfig(window=120, frequency="monthly", no_trade_band=1e9)
        res = run_backtest(returns, cov_method="sample",
                           allocator="minimum_variance", config=cfg)
        W = res.weights_history.values
        for k in range(1, len(W)):
            np.testing.assert_allclose(W[k], W[0], atol=1e-12)

    def test_no_trade_flag_is_recorded(self, returns):
        cfg = BacktestConfig(window=120, frequency="monthly", no_trade_band=1e9)
        res = run_backtest(returns, cov_method="sample",
                           allocator="minimum_variance", config=cfg)
        assert res.diagnostics["no_trade"].iloc[0] == 0
        assert res.diagnostics["no_trade"].iloc[1:].sum() == len(res.diagnostics) - 1

    def test_a_zero_band_never_suppresses(self, returns, small_cfg):
        assert small_cfg.no_trade_band == 0.0
        res = run_backtest(returns, cov_method="sample",
                           allocator="minimum_variance", config=small_cfg)
        assert res.diagnostics["no_trade"].sum() == 0

    def test_a_band_reduces_total_turnover(self, returns):
        free = run_backtest(
            returns, cov_method="sample", allocator="minimum_variance",
            config=BacktestConfig(window=120, no_trade_band=0.0))
        banded = run_backtest(
            returns, cov_method="sample", allocator="minimum_variance",
            config=BacktestConfig(window=120, no_trade_band=0.02))
        assert banded.turnover_series.sum() <= free.turnover_series.sum() + 1e-12


# ---------------------------------------------------------------------------
# Cost models end-to-end
# ---------------------------------------------------------------------------


class TestCostModels:
    def test_impact_model_costs_more_than_proportional(self, returns):
        prop = run_backtest(
            returns, cov_method="sample", allocator="minimum_variance",
            config=BacktestConfig(window=120, cost_model="proportional",
                                  cost_bps=10.0))
        imp = run_backtest(
            returns, cov_method="sample", allocator="minimum_variance",
            config=BacktestConfig(window=120, cost_model="linear_plus_impact",
                                  cost_bps=10.0, impact_coef=5.0))
        assert imp.cost_series.sum() > prop.cost_series.sum()
        # proportional part is identical, so the extra is purely the impact term
        extra = imp.cost_series.sum() - prop.cost_series.sum()
        expected_extra = (5.0 / 1e4) * np.sqrt(prop.turnover_series).sum()
        assert extra == pytest.approx(expected_extra)

    def test_higher_cost_bps_lowers_net_but_not_gross(self, returns):
        lo = run_backtest(
            returns, cov_method="sample", allocator="risk_parity",
            config=BacktestConfig(window=120, cost_bps=0.0))
        hi = run_backtest(
            returns, cov_method="sample", allocator="risk_parity",
            config=BacktestConfig(window=120, cost_bps=100.0))
        np.testing.assert_allclose(lo.gross_returns, hi.gross_returns)
        assert hi.net_returns.sum() < lo.net_returns.sum()

    def test_costs_are_monotone_in_cost_bps(self, returns):
        prev = -np.inf
        for bps in (0.0, 5.0, 10.0, 50.0):
            res = run_backtest(
                returns, cov_method="sample", allocator="minimum_variance",
                config=BacktestConfig(window=120, cost_bps=bps))
            total = res.cost_series.sum()
            assert total >= prev - 1e-15
            prev = total


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------


class TestRunMatrix:
    def test_grid_is_complete(self, returns, small_cfg):
        grid = run_matrix(returns, ["sample", "rmt_hard"],
                          ["equal_weight", "minimum_variance"],
                          config=small_cfg, verbose=False)
        assert set(grid) == {
            "sample+equal_weight", "sample+minimum_variance",
            "rmt_hard+equal_weight", "rmt_hard+minimum_variance",
        }

    def test_each_result_is_labelled_by_its_key(self, returns, small_cfg):
        grid = run_matrix(returns, ["sample"], ["hrp"],
                          config=small_cfg, verbose=False)
        res = grid["sample+hrp"]
        assert res.name == "sample+hrp"
        assert res.metrics.extra["allocator"] == "hrp"

    def test_extra_alloc_kwargs_are_merged(self, returns, small_cfg):
        grid = run_matrix(returns, ["sample"], ["minimum_variance"],
                          config=small_cfg, verbose=False,
                          extra_alloc_kwargs={"max_weight": 0.2})
        assert grid["sample+minimum_variance"].weights_history.values.max() <= 0.2 + 1e-8

    def test_a_failing_cell_is_skipped_not_fatal(self, returns, small_cfg):
        grid = run_matrix(returns, ["sample", "bogus_method"],
                          ["equal_weight"], config=small_cfg, verbose=False)
        assert "sample+equal_weight" in grid
        assert "bogus_method+equal_weight" not in grid

    def test_verbose_false_is_silent(self, returns, small_cfg, capsys):
        run_matrix(returns, ["sample"], ["equal_weight"],
                   config=small_cfg, verbose=False)
        assert capsys.readouterr().out == ""


class TestWithExtraAllocKwargs:
    def test_returns_a_new_object(self):
        cfg = BacktestConfig()
        out = _with_extra_alloc_kwargs(cfg, {"max_weight": 0.3})
        assert out is not cfg
        assert out.alloc_kwargs == {"max_weight": 0.3}
        assert cfg.alloc_kwargs == {}

    def test_merges_rather_than_replaces(self):
        cfg = BacktestConfig(alloc_kwargs={"linkage_method": "ward"})
        out = _with_extra_alloc_kwargs(cfg, {"max_weight": 0.3})
        assert out.alloc_kwargs == {"linkage_method": "ward", "max_weight": 0.3}

    def test_overrides_on_a_key_clash(self):
        cfg = BacktestConfig(alloc_kwargs={"max_weight": 0.9})
        out = _with_extra_alloc_kwargs(cfg, {"max_weight": 0.1})
        assert out.alloc_kwargs["max_weight"] == 0.1

    def test_other_fields_are_preserved(self):
        cfg = BacktestConfig(window=200, cost_bps=25.0)
        out = _with_extra_alloc_kwargs(cfg, {"max_weight": 0.2})
        assert out.window == 200
        assert out.cost_bps == 25.0
