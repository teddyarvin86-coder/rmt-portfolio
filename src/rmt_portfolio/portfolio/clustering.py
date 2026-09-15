"""Hierarchical clustering of assets from a correlation matrix.

The clustering step is what makes Hierarchical Risk Parity (HRP) different
from classical optimisers: instead of inverting a noisy covariance matrix, we
group assets by similarity and allocate risk across and within groups.

Two ingredients are provided:

1. :func:`correlation_distance` — the standard metric of Lopez de Prado
   (2016), :math:`d_{ij} = \\sqrt{\\tfrac{1}{2}(1-\\rho_{ij})}`, which maps
   correlation ``1`` to distance ``0`` and correlation ``-1`` to distance
   ``1``.  It is a proper metric on the space of correlation matrices.
2. :func:`hierarchical_cluster` — agglomerative clustering with a choice of
   linkage.  **Single linkage is the default** because it produces the
   "chained" dendrogram structure that HRP's recursive bisection assumes, and
   because it is invariant to monotone transformations of the distance.

An optional :func:`quasi_diagonalise` returns the leaf order that makes the
correlation matrix look block-diagonal — the visual signature of a
well-clustered universe.

References
----------
Lopez de Prado, M. (2016). Building diversification through hierarchical
clustering. *Journal of Portfolio Management*, 42(4), 74-85.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

__all__ = [
    "ClusterResult",
    "correlation_distance",
    "hierarchical_cluster",
    "quasi_diagonalise",
    "cluster_variance_share",
]


# ---------------------------------------------------------------------------
# Distance metric
# ---------------------------------------------------------------------------


def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Convert a correlation matrix to a distance matrix.

    .. math::

        d_{ij} = \\sqrt{\\tfrac{1}{2}\\left(1 - \\rho_{ij}\\right)}

    Properties used by HRP: symmetric, zero diagonal, and the resulting map
    is a valid metric (Lopez de Prado 2016, Appendix).  Negative correlations
    are clipped to ``[-1, 1]`` first to absorb floating point error.
    """
    c = np.asarray(corr, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("corr must be a square matrix")
    c = np.clip(0.5 * (c + c.T), -1.0, 1.0)
    np.fill_diagonal(c, 1.0)
    d = np.sqrt(np.maximum(0.5 * (1.0 - c), 0.0))
    np.fill_diagonal(d, 0.0)
    return d


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


@dataclass
class ClusterResult:
    """Output of :func:`hierarchical_cluster`.

    Attributes
    ----------
    linkage_matrix:
        The ``(N-1) x 4`` SciPy linkage matrix.
    order:
        Leaf order (indices into the original assets) that quasi-diagonalises
        the correlation matrix.
    labels:
        Flat cluster label per asset, from :func:`scipy.cluster.hierarchy.fcluster`.
    n_clusters:
        Number of flat clusters requested (or inferred from ``criterion``).
    method:
        Linkage method used.
    """

    linkage_matrix: np.ndarray
    order: np.ndarray
    labels: np.ndarray
    n_clusters: int
    method: str

    def cluster_members(self) -> Dict[int, List[int]]:
        """Map flat cluster label -> list of asset indices."""
        out: Dict[int, List[int]] = {}
        for asset, lab in enumerate(self.labels):
            out.setdefault(int(lab), []).append(int(asset))
        return out

    def sorted_labels(self) -> np.ndarray:
        """Cluster labels reordered to match :attr:`order`."""
        return self.labels[self.order]


def hierarchical_cluster(
    corr: np.ndarray,
    method: str = "single",
    n_clusters: Optional[int] = None,
    criterion: str = "maxclust",
    optimal_ordering: bool = True,
) -> ClusterResult:
    """Agglomerative hierarchical clustering of assets.

    Parameters
    ----------
    corr:
        ``N x N`` correlation matrix.
    method:
        Linkage method: ``'single'`` (default, HRP-standard), ``'average'``,
        ``'complete'``, ``'ward'``.  Ward requires a Euclidean distance, which
        the correlation metric of :func:`correlation_distance` approximates
        well enough for clustering purposes.
    n_clusters:
        Number of flat clusters.  Defaults to ``ceil(sqrt(N))``, a common
        heuristic that balances granularity and stability.
    criterion:
        Passed to :func:`scipy.cluster.hierarchy.fcluster`.  ``'maxclust'``
        with ``n_clusters`` is the natural choice.
    optimal_ordering:
        If ``True``, reorders leaves to minimise the distance between adjacent
        leaves — gives cleaner dendrograms and heatmaps.
    """
    n = corr.shape[0]
    dist = correlation_distance(corr)
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method=method, optimal_ordering=optimal_ordering)

    if n_clusters is None:
        n_clusters = int(np.ceil(np.sqrt(n)))
    n_clusters = max(1, min(int(n_clusters), n))

    labels = fcluster(z, t=n_clusters, criterion=criterion)
    order = np.array(dendrogram(z, no_plot=True)["leaves"], dtype=int)

    return ClusterResult(linkage_matrix=z, order=order, labels=labels,
                         n_clusters=n_clusters, method=method)


def quasi_diagonalise(corr: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Reorder a correlation matrix by ``order`` for visual inspection.

    A well-clustered universe produces a matrix that is approximately
    block-diagonal along the diagonal — the defining visual signature of
    HRP's premise that correlations have a hierarchical structure.
    """
    return np.asarray(corr, dtype=float)[np.ix_(order, order)]


def cluster_variance_share(
    corr: np.ndarray,
    cluster_result: ClusterResult,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Share of total portfolio risk contributed by each flat cluster.

    Uses the standard risk-contribution decomposition
    :math:`\\mathrm{RC}_i = w_i (\\Sigma w)_i / \\sqrt{w^T \\Sigma w}`; the
    cluster share is the sum of member contributions divided by total risk.
    """
    c = np.asarray(corr, dtype=float)
    n = c.shape[0]
    w = np.ones(n) / n if weights is None else np.asarray(weights, float).ravel()
    port_var = float(w @ c @ w)
    if port_var <= 0:
        return np.zeros(cluster_result.n_clusters)
    marginal = c @ w
    rc = w * marginal / np.sqrt(port_var)

    members = cluster_result.cluster_members()
    shares = np.zeros(cluster_result.n_clusters)
    for k, idx in members.items():
        shares[int(k) - 1] = float(np.sum(rc[idx])) / float(rc.sum())
    return shares
