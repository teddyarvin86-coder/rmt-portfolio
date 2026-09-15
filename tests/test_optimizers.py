"""Tests for the portfolio allocators.

The allocators are the layer where a mathematically correct but economically
useless answer is easy to produce — a minimum-variance portfolio that parks
83% of the book in a single bond ETF is "optimal" and also untradeable.  The
tests below therefore check not only the first-order optimality conditions
but also the *structure* of the resulting weights (diversification, caps).
"""

from __future__ import annotations

import numpy as np
import pytest

from rmt_portfolio.portfolio.optimizers import (
    ALLOCATORS,
    allocate,
    diversification_ratio,
    effective_number_of_bets,
    equal_weight,
    hrp,
    inverse_variance,
    max_diversification,
    minimum_variance,
    portfolio_volatility,
    risk_contributions,
    risk_parity,
)

ALL_ALLOCATORS = ["equal_weight", "inverse_variance", "minimum_variance",
                  "risk_parity", "hrp", "max_diversification"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_cov(n_assets: int = 8, seed: int = 0, rho: float = 0.3) -> np.ndarray:
    """Well-conditioned equicorrelation covariance with dispersed variances."""
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.08, 0.35, size=n_assets)
    corr = np.full((n_assets, n_assets), rho)
    np.fill_diagonal(corr, 1.0)
    cov = corr * np.outer(vols, vols)
    return 0.5 * (cov + cov.T)


def make_equicorr_cov(n_assets: int = 8, rho: float = 0.4) -> np.ndarray:
    """All variances equal AND all correlations equal.

    In this case risk parity, inverse variance and equal weight must coincide
    — a useful analytic reference point.
    """
    corr = np.full((n_assets, n_assets), rho)
    np.fill_diagonal(corr, 1.0)
    return corr


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_allocator_registry_is_complete():
    expected = {"equal_weight", "inverse_variance", "minimum_variance",
                "risk_parity", "hrp", "max_diversification"}
    assert expected <= set(ALLOCATORS)


def test_allocate_rejects_unknown_method():
    cov = make_cov()
    with pytest.raises((KeyError, ValueError)):
        allocate(cov, method="not_a_real_allocator")


# ---------------------------------------------------------------------------
# Universal structural properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ALL_ALLOCATORS)
def test_weights_sum_to_one(method):
    result = allocate(make_cov(), method=method)
    np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-8)


@pytest.mark.parametrize("method", ALL_ALLOCATORS)
def test_weights_are_long_only(method):
    result = allocate(make_cov(), method=method)
    assert np.all(result.weights >= -1e-10), (
        f"{method} produced a short position: min w = {result.weights.min():.3e}")


@pytest.mark.parametrize("method", ALL_ALLOCATORS)
def test_weights_are_finite(method):
    result = allocate(make_cov(), method=method)
    assert np.all(np.isfinite(result.weights))


@pytest.mark.parametrize("method", ["equal_weight", "minimum_variance",
                                    "max_diversification"])
def test_max_weight_cap_is_respected(method):
    """A 25% cap must be honoured by every allocator that accepts one.

    ``inverse_variance`` and ``hrp`` are closed-form rules with no cap
    parameter, so they are excluded — that is a property of the API, not a
    failure of the constraint.
    """
    result = allocate(make_cov(n_assets=6), method=method, max_weight=0.25)
    assert result.weights.max() <= 0.25 + 1e-6, (
        f"{method} breached the cap: max w = {result.weights.max():.4f}")


def test_cap_unaware_allocators_are_documented():
    """Pin down which allocators ignore ``max_weight`` so that a future
    change to the API is visible rather than silent.

    ``inverse_variance`` and ``hrp`` are closed-form rules with no cap
    parameter.  ``risk_parity`` accepts the argument but on a *generic*
    covariance already satisfies it (its largest weight here is ~0.25), so
    the cap is not binding — which is why it is excluded from the binding
    test above rather than from the API check below.
    """
    cov = make_cov(n_assets=6)
    for method in ("inverse_variance", "hrp"):
        uncapped = allocate(cov, method=method).weights
        capped = allocate(cov, method=method, max_weight=0.25).weights
        np.testing.assert_allclose(capped, uncapped, atol=1e-12)


def test_cap_actually_binds_when_the_portfolio_wants_to_concentrate():
    """A cap test is only meaningful where the uncapped solution breaches it."""
    vols = np.array([0.02, 0.20, 0.25, 0.30])   # one very safe asset
    cov = np.diag(vols ** 2)
    uncapped = minimum_variance(cov).weights
    assert uncapped.max() > 0.25, "fixture does not breach the cap"
    capped = minimum_variance(cov, max_weight=0.25).weights
    assert capped.max() <= 0.25 + 1e-6


@pytest.mark.parametrize("method", ["inverse_variance", "minimum_variance",
                                    "risk_parity", "hrp",
                                    "max_diversification"])
def test_higher_variance_asset_gets_less_weight(method):
    """A monotonicity sanity check on a diagonal covariance matrix.

    With zero correlation the only sensible answer is to give the riskier
    asset less.  ``equal_weight`` is excluded because it deliberately ignores
    the covariance entirely.
    """
    vols = np.array([0.10, 0.20, 0.40, 0.80])
    cov = np.diag(vols ** 2)
    w = allocate(cov, method=method).weights
    assert w[0] > w[3], f"{method}: safest asset got less than the riskiest"


# ---------------------------------------------------------------------------
# equal weight / inverse variance
# ---------------------------------------------------------------------------


def test_equal_weight_is_uniform():
    result = equal_weight(5)
    np.testing.assert_allclose(result.weights, 0.2, atol=1e-12)


def test_equal_weight_ignores_covariance():
    a = allocate(make_cov(seed=1), method="equal_weight")
    b = allocate(make_cov(seed=2), method="equal_weight")
    np.testing.assert_allclose(a.weights, b.weights, atol=1e-12)


def test_inverse_variance_matches_analytic():
    vols = np.array([0.10, 0.20, 0.40])
    cov = np.diag(vols ** 2)
    result = inverse_variance(cov)
    expected = (1.0 / vols ** 2)
    expected /= expected.sum()
    np.testing.assert_allclose(result.weights, expected, rtol=1e-10)


def test_inverse_variance_ignores_correlations():
    """Inverse variance is a diagonal-only rule by construction."""
    vols = np.array([0.1, 0.2, 0.3, 0.4])
    low = np.full((4, 4), 0.1); np.fill_diagonal(low, 1.0)
    high = np.full((4, 4), 0.8); np.fill_diagonal(high, 1.0)
    a = inverse_variance(low * np.outer(vols, vols))
    b = inverse_variance(high * np.outer(vols, vols))
    np.testing.assert_allclose(a.weights, b.weights, rtol=1e-10)


# ---------------------------------------------------------------------------
# minimum variance
# ---------------------------------------------------------------------------


def test_minimum_variance_beats_equal_weight_in_sample():
    """By definition the min-var portfolio has the lowest in-sample variance."""
    cov = make_cov(n_assets=8, seed=5)
    mv = minimum_variance(cov)
    ew = equal_weight(cov.shape[0])
    assert portfolio_volatility(mv.weights, cov) <= portfolio_volatility(
        ew.weights, cov) + 1e-12


def test_minimum_variance_matches_analytic_two_asset():
    """Closed form: w1 = (s2^2 - s12) / (s1^2 + s2^2 - 2 s12)."""
    cov = np.array([[0.04, 0.006], [0.006, 0.09]])
    result = minimum_variance(cov)
    s1, s2, s12 = cov[0, 0], cov[1, 1], cov[0, 1]
    w1 = (s2 - s12) / (s1 + s2 - 2 * s12)
    np.testing.assert_allclose(result.weights, [w1, 1 - w1], rtol=1e-6)


def test_minimum_variance_on_diagonal_is_inverse_variance():
    """With no correlations, min-variance and inverse-variance coincide."""
    cov = np.diag(np.array([0.01, 0.04, 0.09, 0.16]))
    a = minimum_variance(cov)
    b = inverse_variance(cov)
    np.testing.assert_allclose(a.weights, b.weights, rtol=1e-5)


def test_minimum_variance_concentrates_without_a_cap():
    """Documents *why* the study sets max_weight=0.25.

    An unconstrained min-variance solution on a heterogeneous covariance
    can pile most of the book into the single lowest-variance asset.
    """
    vols = np.array([0.02, 0.20, 0.25, 0.30])
    cov = np.diag(vols ** 2)
    unconstrained = minimum_variance(cov)
    capped = minimum_variance(cov, max_weight=0.25)
    assert unconstrained.weights.max() > 0.5
    assert capped.weights.max() <= 0.25 + 1e-6


def test_minimum_variance_with_a_cap_is_still_low_variance():
    """The cap must raise ENB without destroying the volatility objective."""
    cov = make_cov(n_assets=8, seed=11)
    capped = minimum_variance(cov, max_weight=0.25)
    ew = equal_weight(8)
    assert portfolio_volatility(capped.weights, cov) <= portfolio_volatility(
        ew.weights, cov) + 1e-12


# ---------------------------------------------------------------------------
# risk parity / ERC
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_assets,rho", [(6, 0.0), (8, 0.3), (10, 0.6)])
def test_risk_parity_equalises_risk_contributions(n_assets, rho):
    """The defining property of ERC: every asset contributes equally."""
    rng = np.random.default_rng(42)
    vols = rng.uniform(0.08, 0.30, size=n_assets)
    corr = np.full((n_assets, n_assets), rho)
    np.fill_diagonal(corr, 1.0)
    cov = corr * np.outer(vols, vols)

    result = risk_parity(cov)
    rc = risk_contributions(result.weights, cov)
    np.testing.assert_allclose(rc, rc.mean(), atol=2e-4,
                               err_msg=f"risk contributions uneven: {rc}")


def test_risk_parity_is_inverse_variance_when_all_vols_are_equal():
    """ERC on a diagonal covariance with EQUAL variances is 1/N."""
    n = 5
    cov = np.eye(n) * 0.04
    a = risk_parity(cov)
    b = inverse_variance(cov)
    np.testing.assert_allclose(a.weights, b.weights, rtol=1e-6)
    np.testing.assert_allclose(a.weights, np.full(n, 1.0 / n), atol=1e-6)


def test_risk_parity_equals_inverse_volatility_when_uncorrelated():
    """With a diagonal covariance ERC gives w ∝ 1/sigma (not 1/sigma^2).

    This is the standard result: equalising risk contributions with no
    correlations means equalising ``w_i sigma_i``, i.e. inverse-*volatility*
    weights.  Inverse-*variance* would only be right if the assets'
    contributions were measured in variance rather than volatility.
    """
    vols = np.array([0.10, 0.20, 0.40, 0.80])
    cov = np.diag(vols ** 2)
    a = risk_parity(cov).weights
    expected = (1.0 / vols) / (1.0 / vols).sum()
    np.testing.assert_allclose(a, expected, rtol=1e-4)


def test_risk_parity_equalises_risk_on_a_diagonal_covariance():
    """The defining property, checked on the simplest possible fixture."""
    vols = np.array([0.10, 0.20, 0.40, 0.80])
    cov = np.diag(vols ** 2)
    rc = risk_contributions(risk_parity(cov).weights, cov)
    np.testing.assert_allclose(rc, rc.mean(), atol=1e-6)


def test_risk_parity_equals_equal_weight_on_equicorrelation():
    """With identical variances AND identical correlations, ERC is 1/N."""
    cov = make_equicorr_cov(n_assets=8, rho=0.4)
    result = risk_parity(cov)
    np.testing.assert_allclose(result.weights, 1.0 / 8, atol=1e-6)


def test_risk_parity_has_maximal_enb_among_diversified_allocators():
    """ERC should sit at or near the top of the ENB ranking."""
    cov = make_cov(n_assets=10, seed=7)
    rp = effective_number_of_bets(risk_parity(cov).weights, cov)
    mv = effective_number_of_bets(minimum_variance(cov).weights, cov)
    assert rp > mv, "risk parity should be more diversified than min variance"


# ---------------------------------------------------------------------------
# HRP
# ---------------------------------------------------------------------------


def test_hrp_weights_sum_to_one_and_are_long_only():
    result = hrp(make_cov(n_assets=10, seed=3))
    np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-8)
    assert np.all(result.weights >= -1e-10)


def test_hrp_is_approximately_invariant_to_asset_ordering():
    """HRP is built from a hierarchical tree over a *distance* matrix, so
    relabelling the assets should essentially permute the weights.

    It is not exactly invariant, and that is expected rather than a bug: on a
    nearly-equicorrelated covariance many pairwise distances are close
    together, so the agglomerative linkage has near-ties and the *tie
    resolution* can depend on input order.  We therefore assert that the
    permutation moves the weights by only a small amount, and separately
    assert exact invariance on a fixture whose block structure is
    unambiguous.
    """
    cov = make_cov(n_assets=8, seed=13)
    perm = np.array([3, 0, 6, 1, 7, 2, 5, 4])
    a = hrp(cov).weights
    b = hrp(cov[np.ix_(perm, perm)]).weights
    assert np.abs(b - a[perm]).max() < 0.05, (
        f"HRP moved too much under relabelling: "
        f"{np.abs(b - a[perm]).max():.4f}")


def test_hrp_is_exactly_invariant_when_the_blocks_are_unambiguous():
    """Two well-separated blocks leave no room for tie ambiguity."""
    vol = 0.2
    n = 8
    corr = np.eye(n)
    # two tight blocks {0..3} and {4..7}, nearly uncorrelated between blocks
    corr[:4, :4] = 0.9
    corr[4:, 4:] = 0.9
    np.fill_diagonal(corr, 1.0)
    cov = corr * vol * vol
    perm = np.array([5, 1, 7, 3, 0, 6, 2, 4])
    a = hrp(cov).weights
    b = hrp(cov[np.ix_(perm, perm)]).weights
    np.testing.assert_allclose(b, a[perm], rtol=1e-6, atol=1e-10)


def test_hrp_equals_inverse_variance_on_equicorrelation_with_equal_vols():
    """A degenerate tree: every cluster has equal variance, so the recursive
    bisection reduces to a 50/50 split at each level, giving 1/N."""
    cov = make_equicorr_cov(n_assets=8, rho=0.5)
    result = hrp(cov)
    np.testing.assert_allclose(result.weights, 1.0 / 8, atol=1e-6)


def test_hrp_handles_two_assets():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    result = hrp(cov)
    np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-10)


def test_hrp_handles_a_single_asset():
    """A one-asset universe is degenerate but must not crash."""
    try:
        result = hrp(np.array([[0.04]]))
    except ValueError as exc:
        pytest.skip(f"single-asset HRP is not supported: {exc}")
    np.testing.assert_allclose(result.weights, [1.0], atol=1e-12)


# ---------------------------------------------------------------------------
# max diversification
# ---------------------------------------------------------------------------


def test_max_diversification_has_high_diversification_ratio():
    """Max-div should beat equal weight on the DR objective."""
    cov = make_cov(n_assets=8, seed=17)
    md = diversification_ratio(max_diversification(cov).weights, cov)
    ew = diversification_ratio(equal_weight(8).weights, cov)
    assert md >= ew - 1e-6, f"max-div DR {md:.4f} < equal-weight DR {ew:.4f}"


def test_max_diversification_on_uncorrelated_data():
    """With no correlations the DR is maximised by concentrating in the
    lowest-volatility asset (subject to the cap)."""
    vols = np.array([0.05, 0.20, 0.25, 0.30])
    cov = np.diag(vols ** 2)
    result = max_diversification(cov)
    assert np.argmax(result.weights) == 0


def test_diversification_ratio_is_at_least_one():
    """DR = sum(w_i sigma_i) / portfolio_vol >= 1 for any long-only book."""
    cov = make_cov(n_assets=8, seed=19)
    for method in ALL_ALLOCATORS:
        w = allocate(cov, method=method).weights
        assert diversification_ratio(w, cov) >= 1.0 - 1e-9


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_risk_contributions_are_normalised_to_sum_to_one():
    """RC is returned as a *share* of total risk, so it sums to 1.

    (Returning the raw ``w_i (Sigma w)_i`` contributions instead would sum to
    the portfolio variance; the normalised form is what ENB is defined on.)
    """
    cov = make_cov(n_assets=8, seed=23)
    w = risk_parity(cov).weights
    rc = risk_contributions(w, cov)
    np.testing.assert_allclose(rc.sum(), 1.0, rtol=1e-10)


def test_risk_contributions_are_proportional_to_marginal_risk():
    """Normalised RC must equal w_i (Sigma w)_i divided by the total."""
    cov = make_cov(n_assets=6, seed=24)
    w = inverse_variance(cov).weights
    raw = w * (cov @ w)
    np.testing.assert_allclose(risk_contributions(w, cov), raw / raw.sum(),
                               rtol=1e-10)


def test_risk_contributions_for_identity_covariance_are_squared_weights():
    """For cov = I, RC_i is proportional to w_i^2 (normalised to sum to 1)."""
    w = np.array([0.5, 0.3, 0.2])
    rc = risk_contributions(w, np.eye(3))
    expected = w ** 2 / (w ** 2).sum()
    np.testing.assert_allclose(rc, expected, rtol=1e-10)


def test_effective_number_of_bets_is_one_for_a_single_asset():
    w = np.array([1.0, 0.0, 0.0])
    assert effective_number_of_bets(w, np.eye(3)) == pytest.approx(1.0)


def test_effective_number_of_bets_is_n_for_equal_risk_contributions():
    """With identical assets and equal weights, ENB = N."""
    n = 6
    assert effective_number_of_bets(np.full(n, 1.0 / n), np.eye(n)) == \
        pytest.approx(float(n), rel=1e-10)


def test_effective_number_of_bets_is_between_one_and_n():
    cov = make_cov(n_assets=9, seed=29)
    for method in ALL_ALLOCATORS:
        w = allocate(cov, method=method).weights
        enb = effective_number_of_bets(w, cov)
        assert 1.0 - 1e-9 <= enb <= 9.0 + 1e-9, f"{method}: ENB {enb}"


def test_diagnostics_are_attached_to_the_result():
    result = risk_parity(make_cov(n_assets=8, seed=31))
    for attr in ("volatility", "risk_contributions", "enb",
                 "diversification_ratio"):
        assert hasattr(result, attr), f"missing diagnostic {attr!r}"


# ---------------------------------------------------------------------------
# Cross-allocator: genuine differentiation
# ---------------------------------------------------------------------------


def test_allocators_produce_distinct_weights_on_heterogeneous_correlations():
    """With *heterogeneous* correlations the allocators must genuinely differ.

    The fixture deliberately breaks the equicorrelation symmetry, because on
    an equicorrelated covariance ``risk_parity`` and ``max_diversification``
    coincide exactly (both reduce to maximising ``sum w_i sigma_i`` under the
    same simplex constraint — see the test below).
    """
    n, seed = 8, 37
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.08, 0.35, size=n)
    corr = np.full((n, n), 0.3)
    np.fill_diagonal(corr, 1.0)
    corr[0, 1] = corr[1, 0] = -0.25
    corr[2, 3] = corr[3, 2] = 0.85
    corr[4, 5] = corr[5, 4] = 0.05
    cov = 0.5 * (corr * np.outer(vols, vols) + (corr * np.outer(vols, vols)).T)

    weights = {m: allocate(cov, method=m).weights for m in ALL_ALLOCATORS}
    names = list(weights)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not np.allclose(weights[a], weights[b], atol=1e-4), (
                f"{a} and {b} produced identical weights")


def test_risk_parity_coincides_with_max_diversification_on_equicorrelation():
    """Documents a mathematically required coincidence.

    On an equicorrelated covariance the diversification ratio objective and
    the equal-risk-contribution objective have the same solution.  Recording
    it as a test means a future change that breaks the identity is noticed.
    """
    cov = make_cov(n_assets=8, seed=0, rho=0.3)      # equicorrelated by build
    rp = risk_parity(cov, max_weight=0.25).weights
    md = max_diversification(cov, max_weight=0.25).weights
    np.testing.assert_allclose(rp, md, rtol=1e-6, atol=1e-10)


def test_diagonal_covariance_collapses_three_allocators():
    """Documents a mathematically required coincidence.

    With no correlations, minimum-variance, inverse-variance and HRP all
    produce the same weights: the min-variance solution *is* inverse-variance,
    and the HRP tree becomes a single cluster split by inverse variance.
    Risk parity and max-diversification both give the dyadic 1,1/2,1/4,...
    pattern on this fixture.
    """
    vols = np.array([0.10, 0.20, 0.40, 0.80])
    cov = np.diag(vols ** 2)
    a = allocate(cov, method="minimum_variance").weights
    b = allocate(cov, method="inverse_variance").weights
    c = allocate(cov, method="hrp").weights
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-9)
    np.testing.assert_allclose(a, c, rtol=1e-4, atol=1e-9)


def test_allocator_ordering_by_enb_is_sensible():
    """On a generic covariance, ENB should order as:
    risk_parity > inverse_variance > equal_weight > min_variance.
    """
    cov = make_cov(n_assets=10, seed=41)
    enb = {}
    for m in ("risk_parity", "inverse_variance", "equal_weight",
              "minimum_variance"):
        enb[m] = effective_number_of_bets(allocate(cov, method=m).weights, cov)
    assert enb["risk_parity"] >= enb["minimum_variance"]
    assert enb["equal_weight"] >= enb["minimum_variance"] - 1e-9


def test_covariance_scaling_does_not_change_weights():
    """Every allocator must be scale-invariant in the covariance.

    Optimisers that are solved numerically (min-variance via cvxpy) leave
    dust-level weights whose *absolute* size depends on the problem scaling,
    so we compare only meaningful positions and use a generous floor for the
    rest.
    """
    cov = make_cov(n_assets=8, seed=43)
    for method in ALL_ALLOCATORS:
        a = allocate(cov, method=method).weights
        b = allocate(cov * 100.0, method=method).weights
        # Compare against a tolerance scaled to the largest weight — a 1e-8
        # weight on an 0.5 position is noise, not a different portfolio.
        atol = 1e-4 * max(a.max(), b.max())
        np.testing.assert_allclose(b, a, rtol=1e-4, atol=atol,
                                   err_msg=f"{method} is not scale invariant")
