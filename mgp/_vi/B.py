"""Factor loading matrix B (working precision).

Variational factor for the loading rows (MGP_CAVI.md, Component 2):

    q_j(B_j) = N_K(mu_j, V_j),  j = 1, ..., p,

with
    V_j^{-1} = M_j + c_j * sum_i (w_i w_i^T + Phi)
             = diag(rho_j . mbar) + c_j * S,
    mu_j     = c_j * V_j (W^T X)_.j,

where
    c_j  = E_q[1 / sigma_j^2] = alpha_n / beta_j     (see sigma2.py),
    S    = E_q[eta eta^T] = W^T W + n Phi             (see eta.second_moment),
    M_j  = diag(rho_j1 * mbar_1, ..., rho_jK * mbar_K) is the expected
           prior precision E_q[diag(omega_j . T)] of Section 3.1.

MGP note on the prior precision: the diagonal entry is the product
rho_jk * mbar_k with rho_jk = E_q[omega_jk] <= (nu+1)/nu and
mbar_k = E_q[T_k] = prod_l E_q[delta_l].  At the defaults (a2 = 3.1,
large p) the cumulative product can exceed the float32 range (and, for
large K, the float64 range as well), so the products are accumulated in
LOG space at every working precision (an implementation convention of
this package -- see shrinkage.py), and this module receives

    log_prec[j, k] = log rho_jk + log mbar_k

instead of the precision itself.  The whitening scales are then
w_jk = exp(-log_prec_jk / 2) and the covariance is computed in whitened
form

    V_j = W_j ( I_K + c_j (w_j w_j^T) . S )^{-1} W_j,

so the factorized matrix is I + PSD by construction with condition
number near 1 even on heavily shrunk coordinates (where log_prec is
large); the raw precision, with entries up to ~exp(80), is never
formed.

Tensor conventions (as in eta.py):
    X : (n, p), W : (n, K), Phi : (K, K), c : (p,),
    log_prec : (p, K), mu : (p, K), V : (p, K, K).

Reference: MGP_CAVI.md Section 3.1, Component 2 (eq. V_j^{-1}, mu_j).

ELBO byproduct: the B block contributes

    (1/2) sum_j log det V_j

to the closed-form ELBO (Section 3.3, Term 5).  Each log det V_j comes
free from the byproduct Cholesky factor of the whitened matrix
(log det V_j = -sum_k log_prec_jk - 2 sum_k log L_jkk), consistent with
V by construction.
"""

import jax
import jax.numpy as jnp

from ._linalg import cholesky_checked, cholesky_inverse_checked
from .eta import second_moment

__all__ = ["compute_V", "compute_mu", "update_B", "log_det_V"]


def compute_V(
    c: jax.Array, S: jax.Array, log_prec: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Covariances V_j = (diag(prec_j) + c_j * S)^{-1} for all rows.

    Computed in whitened form from log-space precisions: with
    w_jk = exp(-log_prec_jk / 2) and W_j = diag(w_j),

        V_j = W_j ( I_K + c_j (w_j w_j^T) . S )^{-1} W_j.

    The factorized matrix is I + PSD by construction, so its condition
    number stays near 1 even on heavily shrunk coordinates, and
    w_jk -> 0 decouples coordinate k exactly (M_j[k, k] -> 1), so no
    overflow can arise in the factorized system.

    Parameters
    ----------
    c : jax.Array, shape (p,)
        E_q[1 / sigma_j^2].
    S : jax.Array, shape (K, K)
        Second moment E_q[eta eta^T] = W^T W + n Phi.
    log_prec : jax.Array, shape (p, K)
        log of the expected prior precision, log rho_jk + log mbar_k
        (finite by the shrinkage module's contract).

    Returns
    -------
    V : jax.Array, shape (p, K, K)
    log_det_V : jax.Array, shape (p,)
        log det V_j = -sum_k log_prec_jk - 2 sum_k log L_jkk, from the
        same Cholesky factors, hence consistent with V by construction.
    ok : jax.Array, scalar bool
    """
    log_w = -0.5 * log_prec
    w = jnp.exp(log_w)
    # M_j = I_K + c_j (w_j w_j^T) . S, by pure broadcasting.
    M = c[:, None, None] * S * w[:, :, None] * w[:, None, :]
    M = M + jnp.eye(S.shape[0], dtype=S.dtype)
    # M_j = I + PSD by construction: plain batched Cholesky + flag.
    L, ok = cholesky_checked(M)
    # V_j = W_j M_j^{-1} W_j = (w_j w_j^T) . M_j^{-1}, via cho_solve
    # against the identity (never a general matrix inversion).
    V, ok = cholesky_inverse_checked(L, ok)
    V = V * w[:, :, None] * w[:, None, :]
    ldV = -log_prec.sum(axis=1) - 2.0 * jnp.log(
        jnp.diagonal(L, axis1=1, axis2=2)
    ).sum(axis=1)
    return V, ldV, ok


def compute_mu(c: jax.Array, V: jax.Array, WtX: jax.Array) -> jax.Array:
    """Means mu_j = c_j * V_j (W^T X)_.j for all rows.

    Parameters
    ----------
    c : jax.Array, shape (p,)
    V : jax.Array, shape (p, K, K)
    WtX : jax.Array, shape (K, p)
        Precomputed W^T X (column j is W^T X_.j).

    Returns
    -------
    jax.Array, shape (p, K)
    """
    mu = jnp.einsum("jkl,jl->jk", V, WtX.T)  # V_j (W^T X)_.j
    return c[:, None] * mu


def update_B(
    X: jax.Array,
    W: jax.Array,
    Phi: jax.Array,
    c: jax.Array,
    log_prec: jax.Array,
    WtX: jax.Array | None = None,
    S: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """One CAVI update of the B block (Component 2).

    Parameters
    ----------
    X : jax.Array, shape (n, p)
    W : jax.Array, shape (n, K)
        Variational means of the factor scores (current value).
    Phi : jax.Array, shape (K, K)
        Common covariance of the factor scores (current value).
    c : jax.Array, shape (p,)
    log_prec : jax.Array, shape (p, K)
        Log expected prior precision (see :func:`compute_V`).
    WtX : jax.Array, shape (K, p), optional
        Precomputed W^T X; computed internally when omitted.
    S : jax.Array, shape (K, K), optional
        Precomputed second moment; computed internally when omitted.

    Returns
    -------
    mu : jax.Array, shape (p, K)
    V : jax.Array, shape (p, K, K)
    elbo_B : jax.Array, scalar
        This block's ELBO contribution (1/2) sum_j log det V_j.
    ok : jax.Array, scalar bool
    """
    if S is None:
        S = second_moment(W, Phi)
    if WtX is None:
        WtX = W.T @ X
    V, ldV, ok = compute_V(c, S, log_prec)
    mu = compute_mu(c, V, WtX)
    elbo_B = 0.5 * ldV.sum()
    return mu, V, elbo_B, ok


def log_det_V(V: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Log determinants log det V_j for all rows (ELBO, Term 5).

    Diagnostic path: re-factorizes V with a plain batched Cholesky and
    returns the validity flag alongside (no jitter fallback anywhere in
    this package; a failed diagnostic factorization is reported).

    Parameters
    ----------
    V : jax.Array, shape (p, K, K)

    Returns
    -------
    log_det_V : jax.Array, shape (p,)
    ok : jax.Array, scalar bool
    """
    L, ok = cholesky_checked(V)
    return 2.0 * jnp.log(jnp.diagonal(L, axis1=1, axis2=2)).sum(axis=1), ok
