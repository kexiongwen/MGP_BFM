"""Idiosyncratic noise variances sigma_j^2 (working precision).

Variational factor for the diagonal noise covariance Omega
(MGP_CAVI.md, Component 5):

    q_j(sigma_j^2) = IG(alpha_n, beta_j),

with
    alpha_n = a_sigma + n / 2,
    beta_j  = b_sigma + 1/2 * R_j,
    R_j     = ||X_.j - w mu_j||^2
              + n * mu_j^T Phi mu_j
              + Tr[V_j (w w^T + n Phi)].

With S = E_q[eta eta^T] = W^T W + n Phi (see eta.second_moment), the
residual term is computed as

    R_j = ||X_.j - W mu_j||^2 + n * mu_j^T Phi mu_j + Tr[V_j S],

where the plug-in residual ||X_.j - W mu_j||^2 is computed DIRECTLY (as
a squared norm of the difference), not by expanding it into
||X_.j||^2 - 2 mu_j^T (W^T X)_.j + mu_j^T (W^T W) mu_j: in single
precision (the default working precision) the expanded form
catastrophically cancels when the fitted values are large relative to
the residual, destroying the very quantity that drives beta_j.  The
direct form is used at every working precision: it is never worse, and
it stays accurate in float64 too.

Useful moments:
    E_q[1 / sigma_j^2]   = alpha_n / beta_j,
    E_q[log sigma_j^2]   = log(beta_j) - digamma(alpha_n).

Tensor conventions (as in eta.py):
    X : (n, p), W : (n, K), Phi : (K, K),
    mu : (p, K), V : (p, K, K), beta : (p,).

Reference: MGP_CAVI.md Section 3.1, Component 5.

ELBO byproduct: the sigma2 block contributes

    -alpha_n * sum_j log(beta_j)

to the closed-form ELBO (Section 3.3.7, Step 1).  This term already
merges the expected log-likelihood (Term 1) and the noise KL divergence
(Term 2) through the identity beta_j = b_sigma + R_j / 2, so it must
not be counted twice, and it is exact right after this block's update
at the current state -- which is why the sweep order of Section 3.1
makes Component 5 the last update among Components 1, 2 and 5.
"""

import jax
import jax.numpy as jnp

from .eta import second_moment

__all__ = ["compute_R", "compute_beta", "update_sigma2", "mean_inverse"]


def compute_R(
    X: jax.Array,
    W: jax.Array,
    Phi: jax.Array,
    mu: jax.Array,
    V: jax.Array,
    S: jax.Array | None = None,
) -> jax.Array:
    """Residual terms R_j for every coordinate j = 1, ..., p.

    R_j = ||X_.j - W mu_j||^2 + n * mu_j^T Phi mu_j + Tr[V_j S],
    with S = W^T W + n Phi.

    The plug-in residual is computed directly as a squared norm of the
    difference, never by expanding it (see module docstring).  The final
    maximum(., 0) is only a guard against roundoff-level negatives in
    the covariance terms; it cannot recover precision lost to
    cancellation, which is why the residual is never expanded.

    Parameters
    ----------
    X : jax.Array, shape (n, p)
    W : jax.Array, shape (n, K)
    Phi : jax.Array, shape (K, K)
    mu : jax.Array, shape (p, K)
    V : jax.Array, shape (p, K, K)
    S : jax.Array, shape (K, K), optional
        Precomputed second moment W^T W + n Phi.

    Returns
    -------
    jax.Array, shape (p,)
    """
    if S is None:
        S = second_moment(W, Phi)
    n = X.shape[0]
    resid = X - W @ mu.T                          # (n, p) direct residual
    plug_in = (resid * resid).sum(axis=0)         # ||X_.j - W mu_j||^2
    quad = jnp.einsum("jk,kl,jl->j", mu, n * Phi, mu)  # n mu_j^T Phi mu_j
    trace = jnp.einsum("jkl,kl->j", V, S)         # Tr[V_j S]
    return jnp.maximum(plug_in + quad + trace, 0.0)


def compute_beta(
    X: jax.Array,
    W: jax.Array,
    Phi: jax.Array,
    mu: jax.Array,
    V: jax.Array,
    b_sigma: jax.Array,
    S: jax.Array | None = None,
) -> jax.Array:
    """Rate parameters beta_j = b_sigma + R_j / 2.

    Parameters as in :func:`compute_R`; ``b_sigma`` is the
    noise-variance prior rate.

    Returns
    -------
    jax.Array, shape (p,)
    """
    return b_sigma + 0.5 * compute_R(X, W, Phi, mu, V, S=S)


def update_sigma2(
    X: jax.Array,
    W: jax.Array,
    Phi: jax.Array,
    mu: jax.Array,
    V: jax.Array,
    a_sigma: jax.Array,
    b_sigma: jax.Array,
    S: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One CAVI update of the sigma2 block (Component 5).

    Parameters as in :func:`compute_R`; ``a_sigma`` and ``b_sigma`` are
    the gamma-prior hyperparameters on the noise precision (shape and
    rate), which induce the inverse-gamma on sigma_j^2.

    Returns
    -------
    alpha_n : jax.Array, scalar
        Updated shape, a_sigma + n / 2.
    beta : jax.Array, shape (p,)
    elbo_sigma2 : jax.Array, scalar
        This block's ELBO contribution -alpha_n * sum_j log(beta_j).
    ok : (not returned; no factorization in this block)
    """
    n = X.shape[0]
    alpha_n = a_sigma + n / 2.0
    beta = compute_beta(X, W, Phi, mu, V, b_sigma, S=S)
    elbo_sigma2 = -alpha_n * jnp.log(beta).sum()
    return alpha_n, beta, elbo_sigma2


def mean_inverse(alpha_n: jax.Array, beta: jax.Array) -> jax.Array:
    """E_q[1 / sigma_j^2] = alpha_n / beta_j (the c_j of eta/B updates).

    Parameters
    ----------
    alpha_n : jax.Array, scalar
    beta : jax.Array, shape (p,)

    Returns
    -------
    jax.Array, shape (p,)
    """
    return alpha_n / beta
