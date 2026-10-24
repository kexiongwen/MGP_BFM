"""Shared batched linear algebra with device-side validity flags.

The CAVI hot path factorizes I + PSD systems by construction (the eta
precision I_K + mu^T C mu + A, and the whitened B system
I_K + c_j (w_j w_j^T) . S), so a plain Cholesky is the happy path.  JAX
cholesky fills the factor with NaN on failure, so a cheap finite +
positive-diagonal flag is a reliable device-side factorization
diagnostic.  No jitter is added anywhere: a failed update is reported,
not silently repaired.

All numerical arrays use the package working precision: float32 by
default, switchable process-wide with :func:`mgp.set_precision` or per
estimator through its ``precision`` parameter.

For the batched B-block system the inverse is obtained with a
matmul-based blocked triangular inverse on CPU for K >= 32, where
batched ``cho_solve`` is much slower; below that threshold, and on GPU,
the plain ``cho_solve`` path is kept.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

__all__ = ["cholesky_checked", "cholesky_inverse_checked", "logdet_from_chol"]

# On CPU, batched cho_solve shows a large performance cliff around K ~ 32
# (p x 50 x 50 inverses: ~290 ms vs ~20 ms at K = 30 / K = 32 respectively).
# Above this threshold a blocked triangular inverse built from matmuls is
# 4-12x faster on CPU, so it is used there; below it cho_solve stays faster.
_BLOCKED_INVERSE_MIN_K = 32
_BLOCKED_INVERSE_BLOCK = 10


def _blocked_cholesky_inverse(L: jax.Array, block: int = _BLOCKED_INVERSE_BLOCK) -> jax.Array:
    """Inverse of M = L L^T from L, via a blocked triangular inverse.

    Block rows of Y = L^{-1} are computed by forward substitution: the
    diagonal blocks by a batched triangular solve, the off-diagonal blocks
    by matmul, then Minv = Y^T Y.  This replaces the batched cho_solve
    against the identity, which is much slower than matmul on XLA CPU for
    K >= 32.  Arbitrary batch dimensions are preserved; ``block`` need not
    divide K.
    """
    shape = L.shape
    K = shape[-1]
    L2 = L.reshape((-1, K, K))
    p = L2.shape[0]
    eye = jnp.eye(K, dtype=L.dtype)
    blocks = []
    for start in range(0, K, block):
        stop = min(start + block, K)
        rhs = jnp.broadcast_to(eye[start:stop], (p, stop - start, K))
        for j in range(0, start, block):
            jstop = min(j + block, K)
            rhs = rhs - jnp.matmul(L2[:, start:stop, j:jstop], blocks[j // block])
        blocks.append(
            jsla.solve_triangular(
                L2[:, start:stop, start:stop], rhs, lower=True
            )
        )
    Y = jnp.concatenate(blocks, axis=1)
    Minv = jnp.matmul(jnp.swapaxes(Y, -1, -2), Y)
    return Minv.reshape(shape)


def cholesky_checked(M: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Cholesky factor of a (batched) SPD matrix, with a validity flag.

    Parameters
    ----------
    M : jax.Array, shape (..., K, K)
        Symmetric positive definite matrix or batch (only the lower
        triangle is read).

    Returns
    -------
    L : jax.Array, shape (..., K, K)
        Lower Cholesky factor (NaN-filled where the factorization fails).
    ok : jax.Array, scalar bool
        True iff the factor is entirely finite with a strictly positive
        diagonal (over the whole batch).
    """
    L = jnp.linalg.cholesky(M)
    diag = jnp.diagonal(L, axis1=-2, axis2=-1)
    ok = jnp.isfinite(L).all() & (diag > 0).all()
    return L, ok


def cholesky_inverse_checked(
    L: jax.Array, ok: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Inverse of M from its Cholesky factor, folding in a validity flag.

    Solves against the identity (batched cho_solve; never a general
    matrix inversion), so the inverse shares the factor's numerical
    quality.  ``ok`` is ANDed with finiteness of the result.

    Parameters
    ----------
    L : jax.Array, shape (..., K, K)
        Lower Cholesky factor from :func:`cholesky_checked`.
    ok : jax.Array, scalar bool
        Validity flag from the Cholesky factorization.

    Returns
    -------
    Minv : jax.Array, shape (..., K, K)
    ok : jax.Array, scalar bool
    """
    K = L.shape[-1]
    # Only the batched p x K x K system (the B-block covariances) hits the
    # cho_solve performance cliff; the single eta precision stays on the
    # plain path.
    if (
        L.ndim >= 3
        and K >= _BLOCKED_INVERSE_MIN_K
        and jax.default_backend() == "cpu"
    ):
        Minv = _blocked_cholesky_inverse(L)
    else:
        eye = jnp.broadcast_to(jnp.eye(K, dtype=L.dtype), L.shape)
        Minv = jsla.cho_solve((L, True), eye)
    return Minv, ok & jnp.isfinite(Minv).all()


def logdet_from_chol(L: jax.Array) -> jax.Array:
    """log det M = 2 * sum(log diag L) from a (batched) Cholesky factor.

    Returns shape (...) -- the batch dimensions are preserved.
    """
    return 2.0 * jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)).sum(axis=-1)
