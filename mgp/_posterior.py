"""Variational posterior moments of the model covariance matrix.

Under the mean-field CAVI posterior of :mod:`mgp._vi`

    q(B, Omega) = prod_j N(B_j; mu_j, V_j) prod_j IG(sigma_j^2; alpha_n, beta_j),

with B rows independent and independent of Omega, this module computes exact
moments of Sigma = B B^T + Omega (they are exact for the variational
approximation q, not for the true posterior):

    E_q[Sigma]      = mu mu^T + diag(tr(V_1) + E sigma_1^2, ...)
    Var_q(Sigma_ij) = tr(V_i V_j) + mu_i^T V_j mu_i + mu_j^T V_i mu_j   (i != j)
    Var_q(Sigma_ii) = 2 tr(V_i^2) + 4 mu_i^T V_i mu_i + Var_q(sigma_i^2)

and arbitrary entry-pair covariances through :func:`cov_sigma_pairs`.  The
derivation uses the independence of the B rows and the Gaussian moments

    A_i = E_q[b_i b_i^T] = V_i + mu_i mu_i^T,
    E_q[(b_i^T b_j)^2] = tr(A_i A_j)                         (i != j),
    Var_q(b_i^T b_i)   = 2 tr(V_i^2) + 4 mu_i^T V_i mu_i,

plus the inverse-gamma moments

    E_q[sigma_j^2] = beta_j / (alpha_n - 1)                    (alpha_n > 1)
    Var_q[sigma_j^2] = beta_j^2 / ((alpha_n-1)^2 (alpha_n-2))  (alpha_n > 2).

Entry-pair covariances, with Sigma_ij = b_i^T b_j + delta_ij sigma_i^2:

* the B part of Cov(Sigma_ij, Sigma_kl) only depends on how many row indices
  the two entries share;
* the Omega part contributes only when both entries are the same diagonal
  entry, through Var_q(sigma_i^2).

Concrete cases (pairs are unordered, Sigma_ij = Sigma_ji):

====================================================  =========================
row indices                                           Cov_q(Sigma_ij, Sigma_kl)
====================================================  =========================
all four distinct                                     0
share one row r, both entries off-diagonal            mu_s^T V_r mu_t
share one row r, one entry diagonal (r,r), (r,t)      2 mu_t^T V_r mu_r
same off-diagonal entry (i,j), i != j                 tr(V_i V_j) + mu_i^T V_j mu_i
                                                      + mu_j^T V_i mu_j
same diagonal entry (i,i)                             2 tr(V_i^2) + 4 mu_i^T V_i mu_i
                                                      + Var_q(sigma_i^2)
different diagonal entries (i,i), (j,j), i != j       0
====================================================  =========================

The full p^2 x p^2 covariance of vec(Sigma) is never formed: this module
returns the p x p entrywise variance matrix (and the pair API above), both in
float64.  Because the entrywise variance matrix has p^2 entries, computing it
is guarded by :data:`DEFAULT_MAX_ENTRIES` (160 MB in float64).

Semantics: these are moments of the mean-field variational approximation.
Mean-field typically underestimates posterior variance; they are not a
substitute for MCMC uncertainty and, in particular, are not moments of the
true posterior.

The second half of the module estimates the same quantity from Gibbs draws
instead: given draws (B_s, sigma_s^2), s = 1, ..., S,

    Sigma^(s) = B_s B_s^T + diag(sigma_s^2),

the functions below return the Monte Carlo sample mean, the entrywise sample
variance and arbitrary entry-pair sample covariances (ddof = 0).  These are
consistent estimates of the true posterior moments (up to MCMC error and
autocorrelation), unlike the mean-field moments above.  Because the full
p x p variance matrix costs O(S p^2 K) and becomes expensive for large S
(e.g. S = 10000 with p ~ 2000), it is guarded by a compute budget and can be
computed from a deterministic equidistant subsample; the cheap diagonal
variance and pair covariances always use all draws.

The formulas are prior-independent (mean-field Gaussian loading rows +
inverse-gamma noise variances on the CAVI side; plain draw streams on the
Gibbs side).
"""

from __future__ import annotations

import operator

import numpy as np

__all__ = [
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_MAX_DRAWS",
    "DEFAULT_MAX_MOMENT_FLOPS",
    "sigma_mean",
    "sigma_variance",
    "cov_sigma_pairs",
    "suggest_max_draws",
    "montecarlo_sigma_mean",
    "montecarlo_sigma_variance",
    "montecarlo_sigma_diag_variance",
    "montecarlo_cov_sigma_pairs",
]

# Default cap on the number of entries of a returned p x p matrix.
# 2e7 float64 entries ~ 160 MB, i.e. p <= 4472.
DEFAULT_MAX_ENTRIES = 20_000_000

# Budget for one row-block intermediate of sigma_variance (G, H or Ht).
_CHUNK_BYTES = 64 * 2 ** 20

# Gibbs-draw Monte Carlo moments: default cap on the number of draws used for
# the full p x p variance matrix, and the default compute budget.  A full
# scan costs about 2 p^2 K multiply-adds per draw plus O(p^2) accumulation;
# the budget keeps the default property from silently running for minutes on
# a large S (e.g. S = 10000, p ~ 2000).
DEFAULT_MAX_DRAWS = 2_000
DEFAULT_MAX_MOMENT_FLOPS = 200_000_000_000.0

# Target bytes for one (chunk_draws, p, p) matrix in float64.
_MOMENT_CHUNK_BYTES = 128 * 2 ** 20

_F64 = np.float64


def _prepare(mu, V, alpha_n, beta, *, need_variance):
    """Validate and normalize the variational parameters (float64 copies)."""
    mu = np.asarray(mu, dtype=_F64)
    V = np.asarray(V, dtype=_F64)
    beta = np.asarray(beta, dtype=_F64).reshape(-1)
    alpha = float(alpha_n)

    if mu.ndim != 2 or mu.shape[0] < 1 or mu.shape[1] < 1:
        raise ValueError(f"mu must have shape (p, K) with p, K >= 1, got {mu.shape}")
    p, K = mu.shape
    if V.shape != (p, K, K):
        raise ValueError(f"V must have shape ({p}, {K}, {K}), got {V.shape}")
    if beta.shape != (p,):
        raise ValueError(f"beta must have shape ({p},), got {beta.shape}")
    if not np.isfinite(mu).all():
        raise ValueError("mu must contain only finite values")
    if not np.isfinite(V).all():
        raise ValueError("V must contain only finite values")
    if not np.isfinite(beta).all():
        raise ValueError("beta must contain only finite values")
    if np.any(beta <= 0):
        raise ValueError("beta must be strictly positive")
    if not np.isfinite(alpha):
        raise ValueError("alpha_n must be finite")
    if alpha <= 1.0:
        raise ValueError(
            "the inverse-gamma mean E_q[sigma_j^2] = beta_j / (alpha_n - 1) "
            f"requires alpha_n > 1, got alpha_n = {alpha}"
        )
    if need_variance and alpha <= 2.0:
        raise ValueError(
            "the inverse-gamma variance Var_q[sigma_j^2] = "
            "beta_j^2 / ((alpha_n - 1)^2 (alpha_n - 2)) requires "
            f"alpha_n > 2 (alpha_n = a_sigma + n/2), got alpha_n = {alpha}"
        )
    # Symmetrize: the formulas assume symmetric V_j (CAVI returns it up to
    # float32 rounding, where V is computed from a Cholesky factor).
    V = 0.5 * (V + np.swapaxes(V, -1, -2))
    return mu, V, alpha, beta


def _check_entries(p, max_entries):
    """Guard the allocation of a p x p float64 matrix."""
    if max_entries is not None and p * p > int(max_entries):
        raise ValueError(
            f"the p x p matrix has {p * p} entries (p = {p}), above the "
            f"configured max_entries = {int(max_entries)}; pass a larger "
            "max_entries to the mgp._posterior function if the memory is "
            "available, or use cov_sigma_pairs for selected entry pairs"
        )


def _var_sigma2(beta, alpha):
    """Entrywise Var_q[sigma_j^2] for IG(alpha, beta_j) marginals."""
    return beta ** 2 / ((alpha - 1.0) ** 2 * (alpha - 2.0))


def _var_sigma2_scalar(beta_j, alpha):
    return float(beta_j ** 2 / ((alpha - 1.0) ** 2 * (alpha - 2.0)))


def sigma_mean(mu, V, alpha_n, beta, *, max_entries=DEFAULT_MAX_ENTRIES):
    """Variational posterior mean E_q[Sigma] of Sigma = B B^T + Omega.

    Parameters
    ----------
    mu : array-like, shape (p, K)
        Variational means mu_j of the loading rows (``CaviMGP.mu_``).
    V : array-like, shape (p, K, K)
        Variational covariances V_j of the loading rows (``CaviMGP.V_``).
    alpha_n : float
        Inverse-gamma shape a_sigma + n/2 of the noise variances;
        requires ``alpha_n > 1``.
    beta : array-like, shape (p,)
        Inverse-gamma rates beta_j (``CaviMGP.beta_``).
    max_entries : int or None, default DEFAULT_MAX_ENTRIES
        Safety cap on p * p; ``None`` disables the cap.

    Returns
    -------
    numpy.ndarray, shape (p, p), float64
        E_q[Sigma] = mu mu^T + diag(tr(V_j) + beta_j / (alpha_n - 1)).
        Compare with the plug-in ``CaviMGP.Sigma_``, which omits tr(V_j).
    """
    mu, V, alpha, beta = _prepare(mu, V, alpha_n, beta, need_variance=False)
    p = mu.shape[0]
    _check_entries(p, max_entries)

    out = mu @ mu.T
    trV = np.einsum("jkk->j", V)
    e_sigma2 = beta / (alpha - 1.0)
    out[np.diag_indices(p)] += trV + e_sigma2
    return out


def sigma_variance(
    mu,
    V,
    alpha_n,
    beta,
    *,
    max_entries=DEFAULT_MAX_ENTRIES,
    chunk_rows=None,
):
    """Variational posterior variance of every entry of Sigma = B B^T + Omega.

    Parameters
    ----------
    mu, V, alpha_n, beta
        Variational parameters, as in :func:`sigma_mean`; here
        ``alpha_n > 2`` is required.
    max_entries : int or None, default DEFAULT_MAX_ENTRIES
        Safety cap on p * p; ``None`` disables the cap.
    chunk_rows : int or None, default None
        Rows per row block of the Gram-matrix computation.  ``None`` selects
        a block size that keeps one intermediate near 64 MB; the values do
        not depend on it (the first ``chunk_rows`` rows are used only for
        memory).

    Returns
    -------
    numpy.ndarray, shape (p, p), float64
        Entrywise Var_q(Sigma_ij), symmetric.  Marginal (entrywise) variances
        only: see :func:`cov_sigma_pairs` for covariances between entries.
    """
    mu, V, alpha, beta = _prepare(mu, V, alpha_n, beta, need_variance=True)
    p, K = mu.shape
    _check_entries(p, max_entries)

    diag_var = (
        2.0 * np.einsum("jkl,jkl->j", V, V)
        + 4.0 * np.einsum("jk,jkl,jl->j", mu, V, mu)
        + _var_sigma2(beta, alpha)
    )

    Vf = V.reshape(p, K * K)
    U = (mu[:, :, None] * mu[:, None, :]).reshape(p, K * K)

    if chunk_rows is None:
        # Three identical-size intermediates (G, H, Ht) per block.
        chunk_rows = max(1, min(p, int(_CHUNK_BYTES / (3 * 8 * max(p, 1)))))
    chunk_rows = max(1, min(int(chunk_rows), p))

    var = np.empty((p, p), dtype=_F64)
    for start in range(0, p, chunk_rows):
        stop = min(start + chunk_rows, p)
        G = Vf[start:stop] @ Vf.T          # tr(V_i V_j)
        H = U[start:stop] @ Vf.T           # mu_i^T V_j mu_i
        Ht = Vf[start:stop] @ U.T          # mu_j^T V_i mu_j
        var[start:stop] = G + H + Ht

    var[np.diag_indices(p)] = diag_var
    # Remove any asymmetry introduced by the row-block assembly.
    return 0.5 * (var + var.T)


def _var_off(mu, V, i, j):
    """(i, j) entry (i != j) of Var_q(Sigma)."""
    return float(
        np.sum(V[i] * V[j])
        + mu[i] @ V[j] @ mu[i]
        + mu[j] @ V[i] @ mu[j]
    )


def _var_diag(mu, V, beta, alpha, i):
    """(i, i) entry of Var_q(Sigma)."""
    return float(
        2.0 * np.sum(V[i] * V[i])
        + 4.0 * (mu[i] @ V[i] @ mu[i])
        + _var_sigma2_scalar(beta[i], alpha)
    )


def _canonical_pair(pair, p):
    """Validate one (i, j) index pair and return (min, max)."""
    try:
        a, b = pair
    except (TypeError, ValueError):
        raise ValueError(f"each entry must be an (i, j) pair, got {pair!r}")
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        raise ValueError(
            f"pair indices must be integers (bool is not accepted), got {pair!r}"
        )
    try:
        i = operator.index(a)
        j = operator.index(b)
    except TypeError:
        raise ValueError(f"pair indices must be integers, got {pair!r}")
    if not (0 <= i < p and 0 <= j < p):
        raise ValueError(f"pair index out of range for p = {p}: {(i, j)}")
    return (i, j) if i <= j else (j, i)


def cov_sigma_pairs(mu, V, alpha_n, beta, pairs):
    """Cov_q(Sigma_ij, Sigma_kl) for selected entry pairs.

    Parameters
    ----------
    mu, V, alpha_n, beta
        Variational parameters, as in :func:`sigma_mean`; here
        ``alpha_n > 2`` is required.
    pairs : iterable of ((i, j), (k, l))
        Entry pairs, indices in ``range(p)``; order within each pair and
        between the two entries is irrelevant.

    Returns
    -------
    numpy.ndarray, shape (len(pairs),), float64
        The covariances, in the same order as ``pairs``.  The case structure
        is documented in the module docstring; pairs with no shared row have
        covariance zero, and the Omega part contributes only to the variance
        of a diagonal entry.
    """
    mu, V, alpha, beta = _prepare(mu, V, alpha_n, beta, need_variance=True)
    p = mu.shape[0]

    canonical = []
    for entry in pairs:
        if len(entry) != 2:
            raise ValueError(
                f"each entry must be ((i, j), (k, l)), got {entry!r}"
            )
        canonical.append(
            _canonical_pair(entry[0], p) + _canonical_pair(entry[1], p)
        )

    out = np.empty(len(canonical), dtype=_F64)
    for idx, (i, j, k, l) in enumerate(canonical):
        diag_a = i == j
        diag_b = k == l

        if diag_a and diag_b:
            out[idx] = _var_diag(mu, V, beta, alpha, i) if i == k else 0.0
            continue

        if not diag_a and not diag_b:
            if {i, j} == {k, l}:
                out[idx] = _var_off(mu, V, i, j)
            elif not ({i, j} & {k, l}):
                out[idx] = 0.0
            else:
                r = next(iter({i, j} & {k, l}))
                s = j if i == r else i
                t = l if k == r else k
                out[idx] = float(mu[s] @ V[r] @ mu[t])
            continue

        # Exactly one entry is diagonal; it shares row r with the off-diagonal
        # entry (r, t) or not at all.
        if diag_a:
            r, t = i, (l if k == i else k)
            if r not in (k, l):
                out[idx] = 0.0
            else:
                out[idx] = 2.0 * float(mu[t] @ V[r] @ mu[r])
        else:
            r, t = k, (j if i == k else i)
            if r not in (i, j):
                out[idx] = 0.0
            else:
                out[idx] = 2.0 * float(mu[t] @ V[r] @ mu[r])

    return out


# ============================================================================
# Monte Carlo posterior moments of Sigma from Gibbs draws
# ============================================================================


def _prepare_draws(B, sigma2):
    """Validate the Gibbs draws and return them as arrays."""
    B = np.asarray(B)
    sigma2 = np.asarray(sigma2)
    if B.ndim != 3 or min(B.shape) < 1:
        raise ValueError(
            f"B draws must have shape (S, p, K) with S, p, K >= 1, "
            f"got {B.shape}"
        )
    S, p, K = B.shape
    if sigma2.shape != (S, p):
        raise ValueError(
            f"sigma2 draws must have shape ({S}, {p}), got {sigma2.shape}"
        )
    if not np.isfinite(B).all():
        raise ValueError("B draws must contain only finite values")
    if not np.isfinite(sigma2).all():
        raise ValueError("sigma2 draws must contain only finite values")
    if np.any(sigma2 <= 0):
        raise ValueError("sigma2 draws must be strictly positive")
    return B, sigma2


def _subsample_indices(S, max_draws):
    """Deterministic equidistant draw indices (all draws when max_draws >= S)."""
    if max_draws is not None:
        if (isinstance(max_draws, (bool, np.bool_))
                or not isinstance(max_draws, (int, np.integer))):
            raise ValueError("max_draws must be an integer or None")
        max_draws = int(max_draws)
        if max_draws < 1:
            raise ValueError(f"max_draws must be >= 1, got {max_draws}")
    if max_draws is None or max_draws >= S:
        return np.arange(S, dtype=np.int64)
    idx = np.rint(np.linspace(0, S - 1, max_draws)).astype(np.int64)
    return np.unique(idx)


def _auto_chunk_draws(p, chunk_bytes=_MOMENT_CHUNK_BYTES):
    """Draw-block size keeping one (chunk, p, p) float64 matrix under budget."""
    return max(1, min(256, int(chunk_bytes / (8 * max(p * p, 1)))))


def _resolve_chunk(chunk_draws, p, S):
    if chunk_draws is None:
        chunk = _auto_chunk_draws(p)
    else:
        if (isinstance(chunk_draws, (bool, np.bool_))
                or not isinstance(chunk_draws, (int, np.integer))
                or int(chunk_draws) < 1):
            raise ValueError("chunk_draws must be a positive integer or None")
        chunk = int(chunk_draws)
    return max(1, min(chunk, S))


def suggest_max_draws(
    p,
    K,
    S,
    *,
    max_flops=DEFAULT_MAX_MOMENT_FLOPS,
    max_draws=DEFAULT_MAX_DRAWS,
):
    """Suggest how many draws to use for the full p x p variance matrix.

    The cost model is ``2 p^2 K`` multiply-adds per draw (forming
    ``B_s B_s^T``) plus ``O(p^2)`` accumulation.  All draws are suggested
    when the full scan fits ``max_flops`` (the ``max_draws`` cap is not
    binding then); when subsampling is required, the suggestion is
    ``min(max_draws, floor(max_flops / (2 p^2 K)))``, at least one draw
    (``max_draws=None`` removes that cap).

    Parameters
    ----------
    p, K, S : int
        Dimension, truncation level and number of available draws.
    max_flops : float, default DEFAULT_MAX_MOMENT_FLOPS
        Approximate compute budget.
    max_draws : int or None, default DEFAULT_MAX_DRAWS
        Cap applied when subsampling is required (all draws are used
        whenever the full scan fits ``max_flops``).

    Returns
    -------
    int
        Number of draws to use (``S`` when the full scan fits the budget).
    """
    for name, value in (("p", p), ("K", K), ("S", S)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 1):
            raise ValueError(f"{name} must be an integer >= 1, got {value}")
    p, K, S = int(p), int(K), int(S)
    if max_draws is None:
        cap = S
    else:
        if (isinstance(max_draws, (bool, np.bool_))
                or not isinstance(max_draws, (int, np.integer))
                or int(max_draws) < 1):
            raise ValueError("max_draws must be a positive integer or None")
        cap = min(int(max_draws), S)
    if not np.isfinite(max_flops) or max_flops <= 0:
        raise ValueError(f"max_flops must be positive and finite, got {max_flops}")
    flops_per_draw = 2.0 * p * p * K
    if flops_per_draw * S <= max_flops:
        return S
    budget_draws = max(1, int(max_flops // flops_per_draw))
    return int(max(1, min(cap, budget_draws)))


def montecarlo_sigma_mean(B, sigma2, *, chunk_draws=None):
    """Monte Carlo mean of Sigma over Gibbs draws.

    ``(1/S) sum_s (B_s B_s^T + diag(sigma_s^2))`` computed by streaming over
    draw blocks, so the full set of per-draw p x p matrices is never stored.

    Parameters
    ----------
    B : array-like, shape (S, p, K)
        Loading draws (``GibbsMGP.B_samples_``).
    sigma2 : array-like, shape (S, p)
        Noise-variance draws (``GibbsMGP.sigma2_samples_``).
    chunk_draws : int or None, default None
        Draws per block; ``None`` selects a block size from a memory budget.

    Returns
    -------
    numpy.ndarray, shape (p, p), float64
    """
    B, sigma2 = _prepare_draws(B, sigma2)
    S, p, K = B.shape
    chunk = _resolve_chunk(chunk_draws, p, S)
    total = np.zeros((p, p), dtype=_F64)
    for start in range(0, S, chunk):
        stop = min(start + chunk, S)
        Bc = B[start:stop]
        G = np.matmul(Bc, np.swapaxes(Bc, 1, 2))
        total += G.sum(axis=0, dtype=_F64)
    mean = total / S
    mean[np.diag_indices(p)] += sigma2.mean(axis=0, dtype=_F64)
    return mean


def montecarlo_sigma_variance(
    B,
    sigma2,
    *,
    max_draws=None,
    chunk_draws=None,
    max_entries=DEFAULT_MAX_ENTRIES,
):
    """Monte Carlo entrywise posterior variance of Sigma from Gibbs draws.

    Computes ``(1/S) sum_s Sigma_s^2 - ((1/S) sum_s Sigma_s)^2`` with
    ddof = 0, where ``Sigma_s = B_s B_s^T + diag(sigma_s^2)``, streaming over
    draw blocks.  Only the entrywise variance matrix is returned; the full
    p^2 x p^2 covariance of ``vec(Sigma)`` is never formed.

    Parameters
    ----------
    B, sigma2
        Gibbs draws, as in :func:`montecarlo_sigma_mean`.
    max_draws : int or None, default None
        Number of draws to use; ``None`` uses all draws.  A value below S
        selects a deterministic equidistant subsample (see
        :func:`_subsample_indices`), which reduces the Monte Carlo precision.
    chunk_draws : int or None, default None
        Draws per block; ``None`` selects a block size from a memory budget.
    max_entries : int or None, default DEFAULT_MAX_ENTRIES
        Safety cap on p * p; ``None`` disables the cap.

    Returns
    -------
    var : numpy.ndarray, shape (p, p), float64
        Entrywise sample variance, symmetric and nonnegative.
    n_used : int
        Number of draws actually used.
    """
    B, sigma2 = _prepare_draws(B, sigma2)
    S, p, K = B.shape
    _check_entries(p, max_entries)

    indices = _subsample_indices(S, max_draws)
    n_used = int(indices.size)
    chunk = _resolve_chunk(chunk_draws, p, n_used)

    total = np.zeros((p, p), dtype=_F64)
    total_sq = np.zeros((p, p), dtype=_F64)
    idx = np.arange(p)
    for start in range(0, n_used, chunk):
        sel = indices[start:start + chunk]
        Bc = B[sel]
        # Promote integral blocks before matmul to avoid integer overflow
        # and allow floating noise variances without copying all draws.
        if Bc.dtype.kind in "biu":
            Bc = Bc.astype(_F64)
        G = np.matmul(Bc, np.swapaxes(Bc, 1, 2))
        G[:, idx, idx] += sigma2[sel]
        total += G.sum(axis=0, dtype=_F64)
        G64 = G.astype(_F64, copy=False)
        np.square(G64, out=G64)
        total_sq += G64.sum(axis=0, dtype=_F64)

    mean = total / n_used
    var = total_sq / n_used - mean * mean
    np.maximum(var, 0.0, out=var)
    return 0.5 * (var + var.T), n_used


def montecarlo_sigma_diag_variance(B, sigma2):
    """Monte Carlo posterior variance of the diagonal entries of Sigma.

    Cheap: O(S p K) instead of O(S p^2 K).  Uses all draws.

    Returns
    -------
    numpy.ndarray, shape (p,), float64
        ``Var_q(Sigma_jj)`` sample variance with ddof = 0.
    """
    B, sigma2 = _prepare_draws(B, sigma2)
    S, p, K = B.shape
    chunk = max(1, min(S, int(_MOMENT_CHUNK_BYTES / (4 * max(p, 1)))))

    total = np.zeros(p, dtype=_F64)
    total_sq = np.zeros(p, dtype=_F64)
    for start in range(0, S, chunk):
        stop = min(start + chunk, S)
        Bc = B[start:stop]
        d = np.einsum("spk,spk->sp", Bc, Bc) + sigma2[start:stop]
        d64 = d.astype(_F64, copy=False)
        total += d64.sum(axis=0)
        np.square(d64, out=d64)
        total_sq += d64.sum(axis=0)

    mean = total / S
    var = total_sq / S - mean * mean
    return np.maximum(var, 0.0)


def montecarlo_cov_sigma_pairs(B, sigma2, pairs):
    """Monte Carlo posterior covariance between selected entries of Sigma.

    Uses all draws; cost O(S K) per pair, independent of p^2.

    Parameters
    ----------
    B, sigma2
        Gibbs draws, as in :func:`montecarlo_sigma_mean`.
    pairs : iterable of ((i, j), (k, l))
        Zero-based entry indices; order within and between entries is
        irrelevant.

    Returns
    -------
    numpy.ndarray, shape (len(pairs),), float64
        Sample covariances with ddof = 0.
    """
    B, sigma2 = _prepare_draws(B, sigma2)
    S, p, K = B.shape

    canonical = []
    for entry in pairs:
        if len(entry) != 2:
            raise ValueError(
                f"each entry must be ((i, j), (k, l)), got {entry!r}"
            )
        canonical.append(
            _canonical_pair(entry[0], p) + _canonical_pair(entry[1], p)
        )

    out = np.empty(len(canonical), dtype=_F64)
    for idx, (i, j, k, l) in enumerate(canonical):
        X = np.einsum("sk,sk->s", B[:, i, :], B[:, j, :]).astype(_F64)
        Y = np.einsum("sk,sk->s", B[:, k, :], B[:, l, :]).astype(_F64)
        if i == j:
            X += sigma2[:, i]
        if k == l:
            Y += sigma2[:, k]
        out[idx] = (X * Y).mean() - X.mean() * Y.mean()
    return out
