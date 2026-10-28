from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import numpy as np


@jax.jit
def _conditional_mean_batch(Xc, loadings, variances, transforms):
    def one_draw(args):
        B, sigma2, transform = args
        scale = jnp.sqrt(jnp.maximum(sigma2, jnp.finfo(sigma2.dtype).tiny ** 0.5))
        C = B.T / scale
        precision = C @ C.T + jnp.eye(B.shape[1], dtype=B.dtype)
        rhs = C @ (Xc.T / scale[:, None])
        factor = jnp.linalg.cholesky(precision)
        conditional_mean = jsla.cho_solve((factor, True), rhs)
        return transform.T @ conditional_mean

    return jax.lax.map(one_draw, (loadings, variances, transforms))


def conditional_factor_mean(
    X,
    loadings,
    variances,
    offset,
    *,
    scale=None,
    transforms=None,
    batch_size=32,
):
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(
        batch_size, (int, np.integer)
    ) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    loadings = np.asarray(loadings)
    variances = np.asarray(variances)
    if loadings.ndim != 3 or min(loadings.shape) < 1:
        raise ValueError("loadings must have nonempty shape (S, p, K)")
    S, p, K = loadings.shape
    if variances.shape != (S, p):
        raise ValueError(f"variances must have shape {(S, p)}")
    if not np.isfinite(loadings).all() or not np.isfinite(variances).all():
        raise ValueError("loadings and variances must be finite")
    if np.any(variances <= 0):
        raise ValueError("variances must be strictly positive")
    if loadings.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("loadings must use float32 or float64")
    work_np = loadings.dtype
    work_jax = jnp.float64 if work_np == np.dtype("float64") else jnp.float32

    X64 = np.asarray(X, dtype=np.float64)
    offset64 = np.asarray(offset, dtype=np.float64)
    if X64.ndim != 2 or X64.shape[0] < 1 or X64.shape[1] != p:
        raise ValueError(f"X must have shape (n, {p})")
    if offset64.shape != (p,) or not np.isfinite(offset64).all():
        raise ValueError("offset must be finite with shape (p,)")
    if not np.isfinite(X64).all():
        raise ValueError("X must contain only finite values")
    if scale is None:
        scale64 = np.ones(p, dtype=np.float64)
    else:
        scale64 = np.asarray(scale, dtype=np.float64)
    if scale64.shape != (p,) or not np.isfinite(scale64).all() or np.any(scale64 <= 0):
        raise ValueError("scale must be positive, finite and have shape (p,)")
    with np.errstate(over="ignore", invalid="ignore"):
        Xc = ((X64 - offset64) / scale64).astype(work_np)
    if not np.isfinite(Xc).all():
        raise ValueError("preprocessing X produced non-finite values")

    if transforms is None:
        transforms = np.broadcast_to(np.eye(K, dtype=work_np), (S, K, K))
    else:
        transforms = np.asarray(transforms, dtype=work_np)
    if transforms.shape != (S, K, K):
        raise ValueError(f"transforms must have shape {(S, K, K)}")
    if not np.isfinite(transforms).all():
        raise ValueError("transforms must be finite")

    n = X64.shape[0]
    n_batches = (S + int(batch_size) - 1) // int(batch_size)
    batch_means = np.empty((n_batches, n, K), dtype=np.float64)
    batch_indices = np.empty((n_batches, 2), dtype=np.int64)
    total = np.zeros((K, n), dtype=np.float64)
    Xc_j = jnp.asarray(Xc, dtype=work_jax)
    for batch_index, start in enumerate(range(0, S, int(batch_size))):
        stop = min(start + int(batch_size), S)
        aligned = np.asarray(_conditional_mean_batch(
            Xc_j,
            jnp.asarray(loadings[start:stop], dtype=work_jax),
            jnp.asarray(variances[start:stop], dtype=work_jax),
            jnp.asarray(transforms[start:stop], dtype=work_jax),
        ))
        if not np.isfinite(aligned).all():
            raise FloatingPointError(f"non-finite conditional means in batch {batch_index}")
        batch_total = aligned.sum(axis=0, dtype=np.float64)
        total += batch_total
        batch_means[batch_index] = (batch_total / (stop - start)).T
        batch_indices[batch_index] = (start, stop)
    return (total / S).T, batch_means, batch_indices


def interval_score_mean(batch_means, batch_indices, start, stop):
    total = None
    count = 0
    for batch_mean, (left, right) in zip(batch_means, batch_indices):
        overlap = max(0, min(int(right), stop) - max(int(left), start))
        if overlap:
            value = batch_mean * overlap
            total = value if total is None else total + value
            count += overlap
    if count != stop - start:
        raise ValueError("batch indices do not cover the requested draw interval")
    return total / count


def half_chain_score_diagnostics(batch_means, batch_indices, columns, norms):
    S = int(batch_indices[-1, 1])
    split = S // 2
    if split < 1 or split >= S:
        raise ValueError("half-chain diagnostics require at least two retained draws")
    columns = np.asarray(columns, dtype=np.int64)
    norms = np.asarray(norms, dtype=np.float64)
    first = interval_score_mean(batch_means, batch_indices, 0, split)
    second = interval_score_mean(batch_means, batch_indices, split, S)
    first = first[:, columns] * norms[columns]
    second = second[:, columns] * norms[columns]
    denominator = np.linalg.norm(first, axis=0) + np.linalg.norm(second, axis=0)
    relative = 2 * np.linalg.norm(second - first, axis=0) / np.maximum(
        denominator, np.finfo(np.float64).tiny
    )
    return {
        "split": split,
        "relative_l2_by_column": relative,
        "relative_l2_median": float(np.median(relative)),
        "relative_l2_max": float(np.max(relative)),
    }
