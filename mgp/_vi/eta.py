"""Latent factor scores eta (working precision).

Variational factor for the latent factors (MGP_CAVI.md, Component 1):

    q_i(eta_i) = N_K(w_i, Phi),  i = 1, ..., n,

with common covariance

    Phi = (I_K + mu^T C mu + A)^{-1},
    w_i = Phi mu^T C x_i,

where
    mu   : (p, K) variational means of the loading rows B_j,
    C    : diag(c), c_j = E_q[1 / sigma_j^2] = alpha_n / beta_j,
    A    : (K, K) sum_j c_j V_j.

The covariance is common to all individuals because
E_q(B)^T E_q(Omega)^{-1} E_q(B) + loading uncertainty does not depend
on i; Phi itself is updated at every iteration.

Tensor conventions used throughout the package:
    X   : (n, p)      data matrix,
    mu  : (p, K)      variational means of the loading rows,
    V   : (p, K, K)   variational covariances of the loading rows,
    c   : (p,)        E_q[1 / sigma_j^2],
    W   : (n, K)      row i is w_i^T,
    Phi : (K, K)      common covariance.

Reference: MGP_CAVI.md Section 3.1, Component 1.

ELBO byproduct: the eta block contributes

    (n/2) log det Phi - (1/2) tr(S),   S = W^T W + n Phi,

to the closed-form ELBO (Section 3.3, Terms 1/3 assembled).  log det Phi
comes free from the Cholesky factor of the precision matrix, and S is
shared with the B and sigma2 updates, so it is computed once per sweep.
"""

import jax
import jax.numpy as jnp

from ._linalg import cholesky_checked, cholesky_inverse_checked

__all__ = ["compute_A", "compute_Phi", "compute_W", "second_moment", "update_eta"]


def compute_A(c: jax.Array, V: jax.Array) -> jax.Array:
    """Compute A = sum_j c_j V_j.

    Parameters
    ----------
    c : jax.Array, shape (p,)
        E_q[1 / sigma_j^2] for j = 1, ..., p.
    V : jax.Array, shape (p, K, K)
        Variational covariances of the loading rows.

    Returns
    -------
    jax.Array, shape (K, K)
    """
    return jnp.einsum("j,jkl->kl", c, V)


def compute_Phi(
    mu: jax.Array,
    V: jax.Array,
    c: jax.Array,
    mu_c: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Precision and covariance of the eta block, with a validity flag.

    Computes Phi = (I_K + mu^T diag(c) mu + A)^{-1} together with its
    log determinant.  The matrix to invert is I + PSD by construction,
    so a plain Cholesky is used; ``ok`` is the device-side
    factorization/inversion validity flag (see _linalg).

    Parameters
    ----------
    mu : jax.Array, shape (p, K)
    V : jax.Array, shape (p, K, K)
    c : jax.Array, shape (p,)
    mu_c : jax.Array, shape (p, K), optional
        Precomputed mu * diag(c), shared with :func:`compute_W`.

    Returns
    -------
    Phi : jax.Array, shape (K, K)
    log_det_Phi : jax.Array, scalar
        log det Phi = -2 sum log diag(L), from the same Cholesky factor
        L of the precision matrix, hence consistent with Phi.
    ok : jax.Array, scalar bool
    """
    K = mu.shape[1]
    if mu_c is None:
        mu_c = mu * c[:, None]
    M = jnp.eye(K, dtype=mu.dtype) + mu_c.T @ mu + compute_A(c, V)
    # M = I + PSD by construction: plain Cholesky + flag, no fallback.
    L, ok = cholesky_checked(M)
    Phi, ok = cholesky_inverse_checked(L, ok)
    log_det_Phi = -2.0 * jnp.log(jnp.diagonal(L)).sum()
    return Phi, log_det_Phi, ok


def compute_W(
    X: jax.Array,
    mu: jax.Array,
    c: jax.Array,
    Phi: jax.Array,
    mu_c: jax.Array | None = None,
) -> jax.Array:
    """Variational means W = X diag(c) mu Phi.

    Row i of the result is w_i^T = x_i^T diag(c) mu Phi.

    Parameters
    ----------
    X : jax.Array, shape (n, p)
    mu : jax.Array, shape (p, K)
    c : jax.Array, shape (p,)
    Phi : jax.Array, shape (K, K)
    mu_c : jax.Array, shape (p, K), optional

    Returns
    -------
    jax.Array, shape (n, K)
    """
    # X diag(c) mu == X @ (c[:, None] * mu): scaling the (p, K) matrix
    # mu avoids materializing an (n, p) scaled copy of X at every sweep.
    if mu_c is None:
        mu_c = mu * c[:, None]
    return X @ mu_c @ Phi


def second_moment(W: jax.Array, Phi: jax.Array) -> jax.Array:
    """E_q[eta eta^T] = W^T W + n Phi.

    Parameters
    ----------
    W : jax.Array, shape (n, K)
    Phi : jax.Array, shape (K, K)

    Returns
    -------
    jax.Array, shape (K, K)
    """
    n = W.shape[0]
    return W.T @ W + n * Phi


def update_eta(
    X: jax.Array, mu: jax.Array, V: jax.Array, c: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """One CAVI update of the eta block (Component 1).

    Parameters
    ----------
    X : jax.Array, shape (n, p)
    mu : jax.Array, shape (p, K)
    V : jax.Array, shape (p, K, K)
    c : jax.Array, shape (p,)

    Returns
    -------
    W : jax.Array, shape (n, K)
    Phi : jax.Array, shape (K, K)
    S : jax.Array, shape (K, K)
        Second moment E_q[eta eta^T] = W^T W + n Phi.
    elbo_eta : jax.Array, scalar
        This block's ELBO contribution (n/2) log det Phi - (1/2) tr(S).
    ok : jax.Array, scalar bool
    """
    mu_c = mu * c[:, None]  # scaled loadings, shared by both updates
    Phi, log_det_Phi, ok = compute_Phi(mu, V, c, mu_c=mu_c)
    W = compute_W(X, mu, c, Phi, mu_c=mu_c)
    S = second_moment(W, Phi)
    elbo_eta = 0.5 * W.shape[0] * log_det_Phi - 0.5 * jnp.trace(S)
    return W, Phi, S, elbo_eta, ok
