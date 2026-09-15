"""Tests for the performance metrics.

Most of these pin down *conventions*, because metric bugs are usually
convention bugs (annualised vs per-period, positive vs negative drawdown,
one-way vs two-way turnover) and they are silent: the number still looks
plausible, it is just wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from rmt_portfolio.backtest.metrics import (
    annualised_return,
    annualised_volatility,
    calmar_ratio,
    compute_metrics,
    conditional_value_at_risk,
    drawdown_series,
    kurtosis,
    max_drawdown,
    sharpe_ratio,
    skewness,
    sortino_ratio,
    turnover,
    ulcer_index,
    value_at_risk,
)


# ---------------------------------------------------------------------------
# annualised_return
# ---------------------------------------------------------------------------


def test_annualised_return_compounds_correctly():
    """Two years of a flat 1%/month must compound to ~12.68%/yr."""
    r = np.full(24, 0.01)
    got = annualised_return(r, periods_per_year=12, compounding=True)
    assert got == pytest.approx(1.01 ** 12 - 1.0, rel=1e-12)


def test_annualised_return_arithmetic_mode():
    """The arithmetic branch is simply mean * periods_per_year."""
    r = np.array([0.01, 0.02, 0.03, 0.04])
    got = annualised_return(r, periods_per_year=12, compounding=False)
    assert got == pytest.approx(np.mean(r) * 12, rel=1e-12)


def test_annualised_return_of_a_single_period():
    got = annualised_return(np.array([0.05]), periods_per_year=12)
    assert got == pytest.approx(1.05 ** 12 - 1.0, rel=1e-12)


def test_annualised_return_is_symmetric_in_sign():
    """A +10% then -10% round trip must have a negative geometric return."""
    r = np.array([0.10, -0.10] * 50)
    got = annualised_return(r, periods_per_year=12)
    assert got < 0.0, "volatility drag should make the geometric return negative"


def test_annualised_return_handles_empty_input():
    assert np.isnan(annualised_return(np.array([])))


def test_annualised_return_handles_wiped_out_portfolio():
    """A -100% period must not produce a complex number."""
    r = np.array([0.05, -1.0, 0.05])
    got = annualised_return(r, periods_per_year=12)
    assert np.isfinite(got)
    assert got < 0


# ---------------------------------------------------------------------------
# annualised_volatility
# ---------------------------------------------------------------------------


def test_annualised_volatility_scales_by_sqrt_time():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, size=5000)
    got = annualised_volatility(r, periods_per_year=252)
    assert got == pytest.approx(0.01 * np.sqrt(252), rel=0.05)


def test_annualised_volatility_of_constant_series_is_zero():
    assert annualised_volatility(np.full(50, 0.001)) == pytest.approx(0.0)


def test_annualised_volatility_uses_sample_std():
    """ddof=1 must be the default — using ddof=0 biases small samples."""
    r = np.array([0.01, 0.02, 0.03, 0.04])
    assert annualised_volatility(r, periods_per_year=1) == \
        pytest.approx(np.std(r, ddof=1))


def test_annualised_volatility_needs_two_points():
    assert np.isnan(annualised_volatility(np.array([0.01])))


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------


def test_sharpe_matches_manual_computation():
    r = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
    expected = np.mean(r) / np.std(r, ddof=1) * np.sqrt(252)
    assert sharpe_ratio(r) == pytest.approx(expected, rel=1e-12)


def test_sharpe_risk_free_is_annual_and_deannualised():
    """A 2% annual risk-free rate must be de-annualised, not subtracted raw."""
    r = np.full(100, 0.01)
    r[0] = 0.0            # introduce a little dispersion
    rf = 0.02
    got = sharpe_ratio(r, risk_free=rf)
    rf_period = (1.0 + rf) ** (1.0 / 252) - 1.0
    excess = r - rf_period
    expected = np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(252)
    assert got == pytest.approx(expected, rel=1e-10)


def test_sharpe_zero_risk_free_matches_plain_ratio():
    r = np.array([0.01, -0.01, 0.02, -0.005])
    assert sharpe_ratio(r, risk_free=0.0) == pytest.approx(
        np.mean(r) / np.std(r, ddof=1) * np.sqrt(252), rel=1e-12)


def test_sharpe_of_a_constant_series_is_finite():
    """A constant return has zero dispersion, so the Sharpe is unbounded.

    The implementation guards the division by returning a large finite value
    rather than +/-inf, so that downstream tables stay computable.  We only
    pin down that it does not produce NaN or inf.
    """
    got = sharpe_ratio(np.full(20, 0.001))
    assert not np.isnan(got)
    assert np.isfinite(got)
    assert abs(got) > 1e3


def test_sharpe_scales_with_periods_per_year():
    """The same data at monthly frequency must give a smaller annualisation."""
    r = np.array([0.01, -0.01, 0.02, -0.005, 0.015, 0.0])
    daily = sharpe_ratio(r, periods_per_year=252)
    monthly = sharpe_ratio(r, periods_per_year=12)
    assert daily > monthly
    assert daily / monthly == pytest.approx(np.sqrt(252 / 12), rel=1e-10)


# ---------------------------------------------------------------------------
# drawdown / max_drawdown
# ---------------------------------------------------------------------------


def test_max_drawdown_is_reported_positive():
    """A -25% drawdown must be returned as +0.25 (the magnitude)."""
    r = np.array([0.0, -0.25, 0.0])
    assert max_drawdown(r) == pytest.approx(0.25, rel=1e-12)


def test_max_drawdown_of_a_monotone_riser_is_zero():
    r = np.full(20, 0.01)
    assert max_drawdown(r) == pytest.approx(0.0, abs=1e-12)


def test_max_drawdown_of_a_monotone_faller_is_the_total_loss():
    """Ten consecutive -10% periods leave wealth at 0.9**10, so the drawdown
    from the initial peak of 1.0 is 1 - 0.9**10.

    REGRESSION GUARD.  An earlier implementation built the wealth index with
    ``cumprod(1 + r)`` and ran the maximum over those points only, which
    excluded the starting level of 1.0.  On a series that *opens* with a
    loss, the first period was silently dropped and the reported maximum
    drawdown came out as 0.6126 instead of 0.6513 here.  Understating a risk
    metric is the dangerous direction, so this is pinned down explicitly.
    """
    r = np.full(10, -0.10)
    assert max_drawdown(r) == pytest.approx(1.0 - 0.9 ** 10, rel=1e-9)


def test_drawdown_includes_the_first_period():
    """A single -20% period must give a 20% drawdown, not zero."""
    assert max_drawdown(np.array([-0.20])) == pytest.approx(0.20, rel=1e-12)


def test_drawdown_includes_the_first_period_in_a_mixed_series():
    """The opening -30% is the deepest drawdown, measured from wealth 1.0.

    Hand-built path (including the starting level of 1.0):
        wealth = [1.00, 0.70, 1.05, 0.945]
        peak   = [1.00, 1.00, 1.05, 1.05]
        dd     = [0.00, -0.30, 0.00, -0.10]
    so the maximum drawdown is 0.30 — set by the FIRST period.  An
    implementation that omits the starting level would miss it.
    """
    r = np.array([-0.30, 0.50, -0.10])
    np.testing.assert_allclose(drawdown_series(r), [-0.30, 0.0, -0.10],
                               atol=1e-12)
    assert max_drawdown(r) == pytest.approx(0.30, rel=1e-9)


def test_drawdown_series_length_matches_input():
    r = np.array([0.01, -0.02, 0.03])
    assert drawdown_series(r).shape == (3,)


def test_max_drawdown_matches_the_drawdown_series_minimum():
    """Cross-check the two functions agree — they are computed separately."""
    rng = np.random.default_rng(101)
    r = rng.normal(0.0003, 0.02, size=1000)
    assert max_drawdown(r) == pytest.approx(-drawdown_series(r).min(), rel=1e-12)


def test_drawdown_series_matches_manual_wealth_path():
    """Cross-check against a hand-built path that includes the level 1.0."""
    r = np.array([0.10, -0.20, 0.05, -0.10])
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + r)))
    peak = np.maximum.accumulate(wealth)
    expected = (wealth / peak - 1.0)[1:]
    np.testing.assert_allclose(drawdown_series(r), expected, rtol=1e-12)


def test_drawdown_series_is_non_positive():
    rng = np.random.default_rng(3)
    dd = drawdown_series(rng.normal(0.0005, 0.02, size=500))
    assert np.all(dd <= 1e-12)


def test_drawdown_series_starts_at_zero_when_the_first_return_is_positive():
    rng = np.random.default_rng(4)
    dd = drawdown_series(rng.normal(0.01, 0.001, size=100))
    assert dd[0] == pytest.approx(0.0, abs=1e-12)


def test_max_drawdown_never_exceeds_one():
    rng = np.random.default_rng(5)
    for _ in range(20):
        dd = max_drawdown(rng.normal(-0.01, 0.05, size=200))
        assert 0.0 <= dd <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# calmar
# ---------------------------------------------------------------------------


def test_calmar_is_return_over_max_drawdown():
    rng = np.random.default_rng(7)
    r = rng.normal(0.0006, 0.01, size=756)
    dd = max_drawdown(r)
    expected = annualised_return(r) / dd
    assert calmar_ratio(r) == pytest.approx(expected, rel=1e-10)


def test_calmar_of_a_monotone_riser_is_infinite_or_nan():
    """No drawdown means the ratio is undefined — must not silently be 0."""
    r = np.full(252, 0.001)
    got = calmar_ratio(r)
    assert not np.isfinite(got) or got > 1e6


# ---------------------------------------------------------------------------
# tail / shape statistics
# ---------------------------------------------------------------------------


def test_value_at_risk_is_a_positive_loss_at_the_5pct_quantile():
    """VaR is reported as a POSITIVE loss magnitude.

    For a standard normal-return fixture the 5% quantile is about -1.645
    sigma, so the reported VaR should be about +1.645 sigma.
    """
    rng = np.random.default_rng(11)
    r = rng.normal(0.0, 0.01, size=10000)
    got = value_at_risk(r, alpha=0.05)
    assert got == pytest.approx(-np.quantile(r, 0.05), rel=1e-10)
    assert got > 0


def test_value_at_risk_is_monotone_in_alpha():
    """A more extreme tail (smaller alpha) must give a larger loss."""
    rng = np.random.default_rng(12)
    r = rng.normal(0.0, 0.01, size=20000)
    assert value_at_risk(r, 0.01) > value_at_risk(r, 0.05) \
        > value_at_risk(r, 0.10)


def test_cvar_is_at_least_as_large_as_var():
    """CVaR (expected shortfall) must be at least as large as VaR.

    Both are positive loss magnitudes, so the expected loss *given* that we
    are in the tail is the bigger number.
    """
    rng = np.random.default_rng(13)
    r = rng.normal(0.0, 0.01, size=10000)
    assert conditional_value_at_risk(r, 0.05) >= value_at_risk(r, 0.05) - 1e-12


def test_cvar_is_close_to_the_gaussian_analytic_value():
    """For a normal, ES_5% = sigma * phi(z_5) / 0.05."""
    from scipy.stats import norm
    rng = np.random.default_rng(14)
    sigma = 0.01
    r = rng.normal(0.0, sigma, size=200000)
    analytic = sigma * norm.pdf(norm.ppf(0.05)) / 0.05
    assert conditional_value_at_risk(r, 0.05) == pytest.approx(analytic,
                                                              rel=0.03)


def test_skewness_of_a_symmetric_sample_is_near_zero():
    rng = np.random.default_rng(17)
    r = rng.normal(0.0, 1.0, size=200000)
    assert abs(skewness(r)) < 0.05


def test_skewness_is_negative_for_a_left_tailed_series():
    rng = np.random.default_rng(19)
    base = rng.normal(0.0, 1.0, size=50000)
    r = np.concatenate([base, np.full(2000, -8.0)])
    assert skewness(r) < 0


def test_kurtosis_is_near_zero_for_a_gaussian():
    rng = np.random.default_rng(23)
    r = rng.normal(0.0, 1.0, size=200000)
    # excess kurtosis convention
    assert abs(kurtosis(r)) < 0.1


def test_kurtosis_is_positive_for_a_fat_tailed_series():
    rng = np.random.default_rng(29)
    base = rng.normal(0.0, 1.0, size=50000)
    r = np.concatenate([base, rng.normal(0.0, 8.0, size=1000)])
    assert kurtosis(r) > 0


def test_ulcer_index_is_non_negative():
    rng = np.random.default_rng(31)
    idx = ulcer_index(rng.normal(0.0002, 0.015, size=500))
    assert idx >= 0.0


def test_ulcer_index_is_zero_for_a_monotone_riser():
    assert ulcer_index(np.full(100, 0.005)) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# turnover — convention matters
# ---------------------------------------------------------------------------


def test_turnover_is_one_way_half_the_l1_distance():
    new = np.array([0.5, 0.5])
    old = np.array([1.0, 0.0])
    # L1 distance is 1.0, so one-way turnover is 0.5
    assert turnover(new, old) == pytest.approx(0.5, rel=1e-12)


def test_turnover_is_zero_when_weights_are_unchanged():
    w = np.array([0.4, 0.35, 0.25])
    assert turnover(w, w) == pytest.approx(0.0, abs=1e-15)


def test_turnover_of_a_full_switch_is_one():
    new = np.array([0.0, 1.0])
    old = np.array([1.0, 0.0])
    assert turnover(new, old) == pytest.approx(1.0, rel=1e-12)


def test_turnover_never_exceeds_one_for_long_only_books():
    """Both vectors on the simplex, so the one-way turnover is capped at 1."""
    rng = np.random.default_rng(37)
    for _ in range(50):
        a = rng.dirichlet(np.ones(10))
        b = rng.dirichlet(np.ones(10))
        t = turnover(a, b)
        assert 0.0 <= t <= 1.0 + 1e-12


def test_turnover_requires_matching_lengths():
    with pytest.raises(ValueError):
        turnover(np.array([0.5, 0.5]), np.array([1.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# compute_metrics container
# ---------------------------------------------------------------------------


def test_compute_metrics_fills_the_expected_fields():
    rng = np.random.default_rng(41)
    r = rng.normal(0.0005, 0.01, size=756)
    m = compute_metrics(r, name="unit-test", periods_per_year=252)
    assert m.name == "unit-test"
    assert m.n_periods == 756
    for attr in ("ann_return", "ann_volatility", "sharpe", "sortino",
                 "max_drawdown", "calmar", "var_95", "cvar_95",
                 "avg_turnover", "sharpe_net", "ann_return_net"):
        assert hasattr(m, attr), f"missing metric {attr!r}"


def test_compute_metrics_net_metrics_are_weaker_than_gross():
    """Costs can only reduce performance."""
    rng = np.random.default_rng(43)
    r = rng.normal(0.0008, 0.01, size=504)
    costs = np.full(r.size, 0.0002)
    m = compute_metrics(r, periods_per_year=252,
                        returns_net=r - costs,
                        turnovers=np.full(r.size, 0.05))
    assert m.sharpe_net < m.sharpe
    assert m.ann_return_net < m.ann_return


def test_compute_metrics_turnover_summary():
    r = np.zeros(10)
    tn = np.array([0.0, 0.1, 0.2, 0.1, 0.0, 0.1, 0.2, 0.1, 0.0, 0.1])
    m = compute_metrics(r, turnovers=tn)
    assert m.avg_turnover == pytest.approx(tn.mean(), rel=1e-12)


def test_compute_metrics_handles_empty_input():
    m = compute_metrics(np.array([]), name="empty")
    assert m.n_periods == 0


def test_compute_metrics_ignores_nan():
    r = np.array([0.01, np.nan, 0.02, np.nan, 0.0])
    m = compute_metrics(r)
    assert m.n_periods == 3


def test_compute_metrics_carries_extra_metadata():
    r = np.zeros(5)
    m = compute_metrics(r, extra={"cov_method": "rmt_rie",
                                  "allocator": "hrp"})
    assert m.extra["cov_method"] == "rmt_rie"
    assert m.extra["allocator"] == "hrp"


def test_compute_metrics_to_dict_is_flat_and_finite():
    rng = np.random.default_rng(47)
    m = compute_metrics(rng.normal(0.0004, 0.01, size=504),
                        extra={"cov_method": "sample"})
    d = m.to_dict()
    assert isinstance(d, dict)
    assert d["cov_method"] == "sample"
    for k, v in d.items():
        if isinstance(v, float):
            assert not np.isinf(v), f"{k} is infinite"


# ---------------------------------------------------------------------------
# Metric coherence
# ---------------------------------------------------------------------------


def test_sharpe_and_sortino_agree_when_there_is_no_downside_asymmetry():
    """On a symmetric sample Sortino and Sharpe should be of similar size."""
    rng = np.random.default_rng(53)
    r = rng.normal(0.0005, 0.01, size=5000)
    s = sharpe_ratio(r)
    so = sortino_ratio(r)
    assert 0.5 < so / s < 2.0


def test_sortino_is_larger_when_the_downside_is_compressed():
    """Truncating losses should raise Sortino more than Sharpe."""
    rng = np.random.default_rng(59)
    r = rng.normal(0.0005, 0.01, size=5000)
    base_s, base_so = sharpe_ratio(r), sortino_ratio(r)
    clipped = np.maximum(r, -0.005)
    assert sortino_ratio(clipped) - base_so > sharpe_ratio(clipped) - base_s


def test_a_levered_series_has_the_same_sharpe():
    """Sharpe is scale invariant in the return stream (no risk-free)."""
    rng = np.random.default_rng(61)
    r = rng.normal(0.0004, 0.01, size=1000)
    assert sharpe_ratio(r * 2.0) == pytest.approx(sharpe_ratio(r), rel=1e-10)
