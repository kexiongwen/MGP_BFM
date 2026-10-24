"""ESS.py  -- effective sample size diagnostics for the Gibbs chain.

Computes per-entry effective sample sizes (ESS) of the kept draws of
the loading matrix B and the noise variances sigma2 produced by
:func:`mgp._gs.GS_MGP`, via ArviZ (:func:`arviz.ess`).

ArviZ is an OPTIONAL dependency of this package, imported lazily: every
other module here works without arviz except the functions of this
file, which raise ImportError with an install hint when it is missing.

Conventions
-----------
* Input arrays are the kept draws: B with shape (S, p, K) and sigma2
  with shape (S, p), as returned by GS_MGP (host-side NumPy arrays in
  the working precision: float32 by default).  Multi-chain arrays
  (C, S, p, K) / (C, S, p) are accepted as-is.
* ArviZ expects a (chain, draw, ...) layout; a single-chain input gets
  a chain axis prepended.  ESS then has the entry shape: (p, K) for B,
  (p,) for sigma2.
* ``method`` and ``prob`` are passed through to :func:`arviz.ess`:
  "bulk" (default; rank-normalized bulk ESS, the recommended summary),
  "tail" and "quantile" (both require ``prob``, e.g. prob=0.9 / 0.05),
  "mean", "sd", "median", ...  ``relative=True`` returns ESS per draw
  (ESS divided by the total number of draws).
* Rule of thumb (Vehtari et al., 2021, doi:10.1214/20-BA1221): bulk
  ESS above ~400 per entry makes the Monte Carlo error of posterior
  means negligible for most purposes.

This module provides optional effective-sample-size diagnostics for the
kept draws; ArviZ is an optional, lazily imported dependency.
"""

import numpy as np

__all__ = ["ess", "ess_summary"]


def _require_arviz():
    """Import arviz lazily, with an actionable error when missing."""
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError(
            "ESS diagnostics require arviz (e.g. pip install arviz, or "
            "pip install mgp-bfm[ess])"
        ) from exc
    return az


def _as_chain_draw(a: np.ndarray, name: str, entry_ndim: int) -> np.ndarray:
    """Validate kept draws and return the (chain, draw, ...) layout."""
    a = np.asarray(a)
    if a.ndim == entry_ndim + 1:        # single chain: (S, ...) -> (1, S, ...)
        a = a[None]
    elif a.ndim != entry_ndim + 2:      # multi-chain: (C, S, ...)
        raise ValueError(
            f"{name} must have shape (S, ...) or (C, S, ...) with "
            f"{entry_ndim} entry dimension(s), got shape {a.shape}"
        )
    if a.shape[1] < 4:
        raise ValueError(
            f"{name} has only {a.shape[1]} draws per chain; ESS needs at "
            "least 4 kept draws"
        )
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite draws; ESS is undefined")
    return a


def ess(B, sigma2=None, method: str = "bulk", prob=None, relative: bool = False):
    """Per-entry effective sample size of the kept B / sigma2 draws.

    Parameters
    ----------
    B : array-like, shape (S, p, K) or (C, S, p, K)
        Kept loading-matrix draws ("B" of the GS_MGP output).
    sigma2 : array-like, shape (S, p) or (C, S, p), optional
        Kept noise-variance draws ("sigma2" of the GS_MGP output).
    method : str, default "bulk"
        ESS kind, passed to :func:`arviz.ess` ("bulk", "tail", "mean",
        "sd", "median", "quantile", ...).  "tail" and "quantile" require
        ``prob``.
    prob : float, optional
        Probability for the "tail"/"quantile" methods (e.g. 0.9 / 0.05).
    relative : bool, default False
        If True, return ESS per draw (ESS / total draws).

    Returns
    -------
    dict
        {"B": (p, K) ndarray} plus {"sigma2": (p,) ndarray} when
        ``sigma2`` is given.
    """
    az = _require_arviz()
    out = {}
    B = _as_chain_draw(B, "B", entry_ndim=2)
    out["B"] = np.asarray(az.ess(B, method=method, prob=prob, relative=relative))
    if sigma2 is not None:
        s2 = _as_chain_draw(sigma2, "sigma2", entry_ndim=1)
        out["sigma2"] = np.asarray(
            az.ess(s2, method=method, prob=prob, relative=relative)
        )
    return out


def ess_summary(B, sigma2=None, method: str = "bulk"):
    """Compact ESS summary over the entries of B (and sigma2).

    The worst entries drive the reliability of the chain, so the
    minimum is the number to watch (see the module docstring for the
    ~400 rule of thumb).

    Parameters
    ----------
    B, sigma2 : array-like
        Kept draws, as in :func:`ess`.
    method : str, default "bulk"
        ESS kind (bulk ESS recommended).

    Returns
    -------
    dict
        One entry per input array, each a dict with keys "min",
        "median", "mean", "max" (floats) and "n_draws" (total draws
        behind the ESS values).
    """
    e = ess(B, sigma2, method=method)
    Bc = _as_chain_draw(B, "B", entry_ndim=2)
    n_draws = int(Bc.shape[0] * Bc.shape[1])
    return {
        key: {
            "min": float(a.min()),
            "median": float(np.median(a)),
            "mean": float(a.mean()),
            "max": float(a.max()),
            "n_draws": int(n_draws),
        }
        for key, a in e.items()
    }
