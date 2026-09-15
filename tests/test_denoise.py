"""Tests for the covariance denoising estimators.

These tests encode the *properties* each estimator must satisfy, not merely
that it runs.  Several of them are regression guards for bugs that actually
occurred during development and that produced plausible-looking but wrong
numbers — those are marked as such, because they are the tests with the
highest value.

Key API note: the estimators in this module are dispatched through
:func:`denoise` and :func:`estimate_covariance`, which take a ``T x N``
**returns** matrix rather than a pre-computed covariance matrix.
"""

from __future__ import annotations

import numpy as np
import pytest

from rmt_portfolio.rmt.denoise import (
    ESTIMATORS,
    RMT_METHODS,
    constant_correlation_cov,
    denoise,
    estimate_covariance,
    factor_model_cov,
    ledoit_wolf_constant_correlation,
    rmt_hard_threshold,
    rmt_soft_threshold,
    rmt_rie,
    sample_cov,
)

ALL = ["sample", "ledoit_wolf", "constant_corr", "rmt_hard", "rmt_soft",
       "rmt_rie", "factor_model"]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_data(n_assets: int = 12, n_obs: int = 252, n_factors: int = 2,
              seed: int = 0) -> np.ndarray:
    """Factor-structured returns: signal factors + idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    loadings = rng.normal(0.0, 1.0, size=(n_assets, n_factors))
    factors = rng.normal(0.0, 0.01, size=(n_obs, n_factors))
    idio = rng.normal(0.0, 0.01, size=(n_obs, n_assets))
    return factors @ loadings.T + idio


def min_eigenvalue(cov: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(cov).min())


def condition_number(cov: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(cov)
    return float(vals.max() / max(vals.min(), 1e-300))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_estimator_registry_is_complete():
    """The seven estimators promised by the paper must all be registered."""
    expected = {"sample", "ledoit_wolf", "constant_corr", "rmt_hard",
                "rmt_soft", "rmt_rie", "factor_model"}
    assert expected <= set(ESTIMATORS)


def test_rmt_method_set_is_a_subset_of_estimators():
    assert set(RMT_METHODS) <= set(ESTIMATORS)


def test_denoise_rejects_unknown_method():
    x = make_data()
    with pytest.raises((KeyError, ValueError)):
        estimate_covariance(x, method="not_a_real_method")


# ---------------------------------------------------------------------------
# Universal properties — every estimator must satisfy these
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ALL)
def test_output_is_symmetric(method):
    x = make_data()
    cov, _ = estimate_covariance(x, method=method, return_info=True)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)


@pytest.mark.parametrize("method", ALL)
def test_output_is_positive_semi_definite(method):
    """A single negative eigenvalue can break the downstream optimiser."""
    x = make_data()
    cov, _ = estimate_covariance(x, method=method, return_info=True)
    assert min_eigenvalue(cov) > -1e-10, (
        f"{method} produced a matrix with min eigenvalue "
        f"{min_eigenvalue(cov):.3e}")


@pytest.mark.parametrize("method", ALL)
def test_shape_is_preserved(method):
    x = make_data(n_assets=8, n_obs=200)
    cov, _ = estimate_covariance(x, method=method, return_info=True)
    assert cov.shape == (8, 8)


@pytest.mark.parametrize("method", ALL)
def test_diagonal_is_positive(method):
    x = make_data()
    cov, _ = estimate_covariance(x, method=method, return_info=True)
    assert np.all(np.diag(cov) > 0), f"{method} produced a non-positive variance"


@pytest.mark.parametrize("method", ALL)
def test_denoise_returns_consistent_diagnostics(method):
    """The DenoiseResult container must be internally consistent."""
    x = make_data()
    res = denoise(x, method=method)
    assert res.method == method
    assert res.covariance.shape == (12, 12)
    assert res.correlation.shape == (12, 12)
    np.testing.assert_allclose(np.diag(res.correlation), 1.0, atol=1e-8)
    assert res.eigenvalues_before.shape == (12,)
    assert res.eigenvalues_after.shape == (12,)
    # The public spectra are stored in ASCENDING order (they come straight
    # from ``eigen_decompose.eigenvalues``); the ranked view is exposed
    # separately as ``descending_eigenvalues``.  Pin the convention down so a
    # silent flip becomes a test failure rather than a wrong figure.
    assert np.all(np.diff(res.eigenvalues_before) >= -1e-12)
    assert np.all(np.diff(res.eigenvalues_after) >= -1e-12)
    # the spectra must therefore agree with the diagonalisation of the
    # covariance actually returned
    np.testing.assert_allclose(
        np.sort(np.linalg.eigvalsh(res.covariance)),
        res.eigenvalues_after, rtol=1e-8, atol=1e-15)


@pytest.mark.parametrize("method", ["rmt_hard", "rmt_soft", "rmt_rie"])
def test_rmt_methods_expose_an_mp_law(method):
    res = denoise(make_data(), method=method)
    assert res.mp_law is not None
    assert res.mp_law.lambda_plus > res.mp_law.lambda_minus


@pytest.mark.parametrize("method", ["sample", "ledoit_wolf"])
def test_non_rmt_methods_have_no_mp_law(method):
    res = denoise(make_data(), method=method)
    assert res.mp_law is None


# ---------------------------------------------------------------------------
# Thresholding
#
# IMPORTANT — the scale on which these operators act.
#
# ``rmt_hard_threshold`` / ``rmt_soft_threshold`` take a COVARIANCE matrix but
# internally convert to the CORRELATION scale (the returned ``info['scale']``
# is ``'correlation'``), apply the eigenvalue replacement there, and rescale
# back.  This matters for what is invariant:
#
#   * the CORRELATION trace is preserved exactly (it equals N by construction,
#     since the correlation diagonal is identically 1)
#   * the COVARIANCE trace is therefore only preserved *approximately*
#   * the flattening of the bulk is only visible on the correlation scale
#
# The tests below assert exactly these invariants.
# ---------------------------------------------------------------------------


def corr_scale(cov: np.ndarray) -> np.ndarray:
    """Return the correlation matrix implied by ``cov`` (unit diagonal)."""
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-300))
    corr = cov / np.outer(sd, sd)
    np.fill_diagonal(corr, 1.0)
    return 0.5 * (corr + corr.T)


def test_hard_threshold_preserves_correlation_trace():
    """The correlation trace is N by construction; it must survive."""
    x = make_data(n_assets=16, n_obs=120, n_factors=1)
    out = rmt_hard_threshold(sample_cov(x), n_obs=120)
    np.testing.assert_allclose(np.trace(corr_scale(out)), 16.0, rtol=1e-8)


def test_soft_threshold_preserves_correlation_trace():
    x = make_data(n_assets=16, n_obs=120, n_factors=1)
    out = rmt_soft_threshold(sample_cov(x), n_obs=120)
    np.testing.assert_allclose(np.trace(corr_scale(out)), 16.0, rtol=1e-8)


def test_hard_threshold_reports_the_correlation_scale():
    """Guards against a silent change of internal convention."""
    x = make_data(n_assets=16, n_obs=120)
    _, info = rmt_hard_threshold(sample_cov(x), n_obs=120, return_info=True)
    assert info["scale"] == "correlation"


def test_hard_threshold_compresses_the_bulk_spread():
    """On the correlation scale thresholding must *narrow* the noise band.

    It does not flatten it to a single value (a PSD ridge is added when
    cleaning the matrix, and the trace is restored), so we assert the weaker
    but still meaningful property: the spread of the noise eigenvalues
    shrinks.
    """
    x = make_data(n_assets=16, n_obs=120, n_factors=1)
    cov = sample_cov(x)
    out, info = rmt_hard_threshold(cov, n_obs=120, return_info=True)
    n_noise = int(info["n_noise"])
    assert n_noise >= 3, "fixture produced too few noise eigenvalues"

    raw = np.sort(np.linalg.eigvalsh(corr_scale(cov)))[:n_noise]
    new = np.sort(np.linalg.eigvalsh(corr_scale(out)))[:n_noise]
    assert (new.max() - new.min()) < (raw.max() - raw.min()), (
        "hard thresholding did not compress the bulk")


def test_hard_threshold_leaves_signal_eigenvalues_alone():
    """The top eigenvalue must pass through (up to the trace rescaling)."""
    x = make_data(n_assets=16, n_obs=120, n_factors=1)
    cov = sample_cov(x)
    out = rmt_hard_threshold(cov, n_obs=120)
    top_raw = np.linalg.eigvalsh(corr_scale(cov)).max()
    top_new = np.linalg.eigvalsh(corr_scale(out)).max()
    assert abs(top_new - top_raw) / top_raw < 0.25, (
        f"largest eigenvalue moved too much: {top_raw:.4f} -> {top_new:.4f}")


def test_soft_threshold_beta_zero_is_a_no_op():
    """beta -> 0 means w -> 1, i.e. no shrinkage at all."""
    x = make_data(n_assets=16, n_obs=120)
    cov = sample_cov(x)
    out = rmt_soft_threshold(cov, n_obs=120, beta=0.0)
    np.testing.assert_allclose(out, cov, rtol=1e-6, atol=1e-14)


def test_soft_and_hard_differ_but_stay_close():
    """Both are trace-preserving reshuffles of the same noise band."""
    x = make_data(n_assets=16, n_obs=120, n_factors=1)
    cov = sample_cov(x)
    hard = rmt_hard_threshold(cov, n_obs=120)
    soft = rmt_soft_threshold(cov, n_obs=120)
    assert not np.allclose(hard, soft, rtol=1e-8)
    # but neither should be wildly far from the other
    assert np.abs(hard - soft).max() / np.abs(cov).max() < 0.5


def test_threshold_scale_widens_the_band():
    """Raising lambda_plus_scale must classify more eigenvalues as noise."""
    x = make_data(n_assets=16, n_obs=120, n_factors=1)
    cov = sample_cov(x)
    _, info_lo = rmt_hard_threshold(cov, n_obs=120, lambda_plus_scale=1.0,
                                    return_info=True)
    _, info_hi = rmt_hard_threshold(cov, n_obs=120, lambda_plus_scale=5.0,
                                    return_info=True)
    assert info_hi["n_noise"] >= info_lo["n_noise"]


def test_signal_and_noise_counts_partition_the_spectrum():
    x = make_data(n_assets=16, n_obs=120, n_factors=1)
    _, info = rmt_hard_threshold(sample_cov(x), n_obs=120, return_info=True)
    assert info["n_signal"] + info["n_noise"] == 16


def test_signal_and_noise_variance_shares_sum_to_one():
    x = make_data(n_assets=16, n_obs=120, n_factors=1)
    _, info = rmt_hard_threshold(sample_cov(x), n_obs=120, return_info=True)
    total = info["signal_variance_share"] + info["noise_variance_share"]
    np.testing.assert_allclose(total, 1.0, rtol=1e-8)


# ---------------------------------------------------------------------------
# rmt_rie — the estimator with the trickiest numerics
# ---------------------------------------------------------------------------


def test_rie_is_insensitive_to_eta_scale():
    """The QuEST fixed point must converge regardless of the regulariser.

    A naive (non-self-consistent) implementation collapsed catastrophically as
    eta -> 0: the trace ratio fell from 1.02 to 0.0004 and the estimator
    degenerated onto the sample matrix.  A correct implementation is
    essentially eta-independent over a wide range, which is what this test
    pins down.  Measured relative deviations from eta=1e-3 are ~1e-3 or less
    across 1e-4 .. 1e-2.
    """
    x = make_data(n_assets=16, n_obs=120)
    cov = sample_cov(x)
    ref = rmt_rie(cov, n_obs=120, eta_scale=1e-3)
    scale = np.abs(ref).max()
    for eta in (1e-4, 5e-4, 1e-2):
        out = rmt_rie(cov, n_obs=120, eta_scale=eta)
        rel = np.abs(out - ref).max() / scale
        assert rel < 0.05, (
            f"eta_scale={eta} changed the result by {rel:.2e} (relative)")


def test_rie_trace_ratio_is_stable_across_eta():
    """The trace ratio itself must not be governed by the regulariser."""
    x = make_data(n_assets=16, n_obs=120)
    cov = sample_cov(x)
    ratios = [np.trace(rmt_rie(cov, n_obs=120, eta_scale=e)) / np.trace(cov)
              for e in (1e-4, 1e-3, 1e-2)]
    assert max(ratios) - min(ratios) < 0.05, f"trace ratios {ratios}"


def test_rie_preserves_trace_approximately():
    """Nonlinear shrinkage should retain most of the total variance."""
    x = make_data(n_assets=16, n_obs=120)
    cov = sample_cov(x)
    out = rmt_rie(cov, n_obs=120)
    ratio = np.trace(out) / np.trace(cov)
    assert 0.5 < ratio < 1.5, f"trace ratio {ratio:.4f} outside a sane band"


def test_rie_reports_both_inflation_and_shrinkage():
    """RIE is NOT monotone shrinkage: it lifts the bulk and lowers the top.

    This was a genuine surprise during development — the first implementation
    clipped the new eigenvalues to be <= the sample ones, which silently
    collapsed the estimator back onto the sample matrix.  The correct
    behaviour is two-sided, so both counters must be positive.
    """
    x = make_data(n_assets=16, n_obs=120)
    cov = sample_cov(x)
    _, info = rmt_rie(cov, n_obs=120, return_info=True)
    assert info["n_inflated"] > 0, "RIE should inflate some eigenvalues"
    assert info["n_shrunk"] > 0, "RIE should shrink some eigenvalues"
    assert info["n_inflated"] + info["n_shrunk"] == cov.shape[0]


def test_rie_differs_from_sample():
    """A regression guard.

    If RIE silently returns the sample matrix the study's headline result
    vanishes — which is exactly what happened when the naive Stieltjes
    transform was combined with a "never inflate" clip.
    """
    x = make_data(n_assets=16, n_obs=120, seed=5)
    out = rmt_rie(sample_cov(x), n_obs=120)
    assert not np.allclose(out, sample_cov(x), rtol=1e-6), (
        "rmt_rie returned the sample matrix unchanged")


def test_rie_trace_ratio_is_close_to_one_on_pure_noise():
    """On data with no signal the estimator should barely move the trace."""
    rng = np.random.default_rng(21)
    x = rng.normal(0.0, 1.0, size=(200, 20))
    cov = sample_cov(x)
    out = rmt_rie(cov, n_obs=200)
    ratio = np.trace(out) / np.trace(cov)
    assert 0.9 < ratio < 1.1, f"pure-noise trace ratio {ratio:.4f}"


# ---------------------------------------------------------------------------
# ledoit_wolf_constant_correlation — validated against analytic extremes
# ---------------------------------------------------------------------------


def test_lw_cc_shrinks_fully_on_identity_data():
    """If the truth is the identity, the constant-correlation target is wrong
    and the optimal shrinkage must be large (pull towards the target)."""
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, size=(60, 10))
    _, info = ledoit_wolf_constant_correlation(x, return_info=True)
    assert info["shrinkage_intensity"] > 0.3, (
        f"expected strong shrinkage on identity data, got "
        f"{info['intensity']:.3f}")


def test_lw_cc_shrinks_little_on_equicorrelation_data():
    """If the truth is exactly constant correlation the target is exact and
    the optimal shrinkage must be small."""
    rng = np.random.default_rng(11)
    n_assets, n_obs, rho = 12, 2000, 0.5
    common = rng.normal(0.0, 1.0, size=(n_obs, 1))
    idio = rng.normal(0.0, 1.0, size=(n_obs, n_assets))
    x = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idio
    _, info = ledoit_wolf_constant_correlation(x, return_info=True)
    assert info["shrinkage_intensity"] < 0.30, (
        f"expected weak shrinkage on exact equicorrelation, got "
        f"{info['intensity']:.3f}")


def test_lw_cc_shrinks_more_on_identity_than_equicorrelation():
    """The two extremes must be ordered correctly — this is the whole point
    of validating against analytic ground truth."""
    rng = np.random.default_rng(3)
    ident = rng.normal(0.0, 1.0, size=(400, 12))

    rng2 = np.random.default_rng(4)
    common = rng2.normal(0.0, 1.0, size=(400, 1))
    idio = rng2.normal(0.0, 1.0, size=(400, 12))
    equi = np.sqrt(0.5) * common + np.sqrt(0.5) * idio

    _, i_ident = ledoit_wolf_constant_correlation(ident, return_info=True)
    _, i_equi = ledoit_wolf_constant_correlation(equi, return_info=True)
    assert i_ident["shrinkage_intensity"] > i_equi["shrinkage_intensity"]


def test_lw_cc_intensity_is_in_unit_interval():
    for seed, (n_a, n_o) in enumerate([(8, 60), (16, 252), (30, 120)]):
        x = make_data(n_assets=n_a, n_obs=n_o, seed=seed)
        _, info = ledoit_wolf_constant_correlation(x, return_info=True)
        assert 0.0 <= info["shrinkage_intensity"] <= 1.0


def test_lw_cc_intensity_zero_is_a_symptom_not_a_result():
    """A regression guard.

    The first implementation over-counted ``rho`` by roughly 6x, which pushed
    ``pi - rho`` below zero so the intensity clipped to *exactly* 0 — making
    the estimator bit-identical to the sample matrix.  Exactly zero on
    factor-structured data is a bug signature.
    """
    x = make_data(n_assets=12, n_obs=252, n_factors=3)
    _, info = ledoit_wolf_constant_correlation(x, return_info=True)
    assert info["shrinkage_intensity"] > 1e-6, (
        "intensity is exactly zero — the rho decomposition is probably wrong")


def test_constant_correlation_target_has_unit_diagonal():
    x = make_data(n_assets=10, n_obs=200)
    corr = constant_correlation_cov(x)
    sd = np.sqrt(np.diag(corr))
    normed = corr / np.outer(sd, sd)
    np.testing.assert_allclose(np.diag(normed), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# factor model
# ---------------------------------------------------------------------------


def test_factor_model_concentrates_variance():
    x = make_data(n_assets=16, n_obs=252, n_factors=2)
    out, _ = factor_model_cov(x, n_factors=2, return_info=True)
    vals = np.sort(np.linalg.eigvalsh(out))[::-1]
    assert vals[2] < vals[1] * 0.5, "factor model did not concentrate variance"


def test_factor_model_with_all_factors_recovers_sample():
    x = make_data(n_assets=10, n_obs=252, n_factors=2)
    out = factor_model_cov(x, n_factors=10)
    np.testing.assert_allclose(out, sample_cov(x), rtol=1e-5, atol=1e-14)


def test_factor_model_outputs_are_distinct():
    """Different factor counts must give measurably different matrices."""
    x = make_data(n_assets=16, n_obs=252, n_factors=3)
    a = factor_model_cov(x, n_factors=1)
    b = factor_model_cov(x, n_factors=5)
    assert not np.allclose(a, b, rtol=1e-6)


# ---------------------------------------------------------------------------
# Cross-estimator: they must genuinely differ
# ---------------------------------------------------------------------------


def test_all_estimators_produce_distinct_matrices():
    """If any two estimators coincide the study is comparing nothing."""
    x = make_data(n_assets=16, n_obs=252, n_factors=3, seed=3)
    outs = {m: estimate_covariance(x, method=m) for m in ALL}
    names = list(outs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not np.allclose(outs[a], outs[b], rtol=1e-6, atol=1e-14), (
                f"{a} and {b} produced identical matrices")


def test_condition_numbers_are_distinct_across_estimators():
    """The estimators should not collapse onto the same conditioning.

    The spread is deliberately modest on a low-q fixture — with N=16 and
    T=252 the noise contamination is weak, so we assert distinctness rather
    than a large ratio.
    """
    x = make_data(n_assets=16, n_obs=252, n_factors=3, seed=3)
    conds = {m: condition_number(estimate_covariance(x, method=m))
             for m in ALL}
    lo, hi = min(conds.values()), max(conds.values())
    assert hi / lo > 1.2, f"condition numbers too clustered: {conds}"
    # and on a high-q fixture the spread must be wider still
    xh = make_data(n_assets=50, n_obs=120, n_factors=3, seed=3)
    condh = {m: condition_number(estimate_covariance(xh, method=m))
             for m in ALL}
    lo_h, hi_h = min(condh.values()), max(condh.values())
    assert hi_h / lo_h > 1.5, f"high-q condition numbers clustered: {condh}"


def test_sample_method_is_a_no_op():
    """`sample` must be the identity transform — it is the benchmark."""
    x = make_data()
    np.testing.assert_allclose(estimate_covariance(x, method="sample"),
                               sample_cov(x), rtol=1e-12)


def test_estimators_are_scale_equivariant():
    """Multiplying all returns by a constant must scale the covariance by its
    square.  RMT methods operate on the correlation scale, so this is a real
    test that the rescaling round-trip is correct."""
    x = make_data(n_assets=12, n_obs=252)
    for method in ALL:
        a = estimate_covariance(x, method=method)
        b = estimate_covariance(x * 3.0, method=method)
        np.testing.assert_allclose(b, a * 9.0, rtol=1e-6, atol=1e-16,
                                   err_msg=f"{method} is not scale equivariant")


def test_estimators_are_permutation_equivariant():
    """Reordering the assets must reorder the matrix, nothing more."""
    x = make_data(n_assets=12, n_obs=252, seed=9)
    perm = np.array([3, 0, 7, 11, 1, 5, 2, 9, 4, 10, 6, 8])
    for method in ALL:
        a = estimate_covariance(x, method=method)
        b = estimate_covariance(x[:, perm], method=method)
        np.testing.assert_allclose(b, a[np.ix_(perm, perm)], rtol=1e-6,
                                   atol=1e-14,
                                   err_msg=f"{method} is not permutation equivariant")
