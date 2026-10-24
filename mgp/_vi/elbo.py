"""Closed-form ELBO of the mean-field approximation (Section 3.3.7).

With e_jk = mu_jk^2 + (V_j)_kk, alpha_n = a_sigma + n/2 and
c_j = alpha_n / beta_j, the compact closed form exact at every sweep
boundary (under the sweep order of Section 3.1) is

    L(Q) = C_ELBO
         - alpha_n * sum_j log(beta_j)
         + (n/2) log det Phi
         - (1/2) (||W||_F^2 + n Tr(Phi))
         + (1/2) sum_j log det V_j
         + S,

where S is the shrinkage block (3.3.7b).  Because every Gamma shape in
the MGP mean-field family is iteration-INDEPENDENT, all digamma and
log-Gamma terms of S are constants, and the variable/constant split of
shrinkage.py gives

    L(Q) = Const + L_var,

    L_var = (n/2) log det Phi - (1/2) tr(W^T W + n Phi)     [eta block]
          + (1/2) sum_j log det V_j                          [B block]
          - alpha_n sum_j log(beta_j)                        [sigma2 block]
          + S_var                                            [shrinkage],

    Const = C_ELBO + S_const    (host float64, once per run),

    C_ELBO = p [log Gamma(alpha_n) - log Gamma(a_sigma) + a_sigma log b_sigma]
             - (np/2) log(2 pi) + (pK/2)(1 + log(2 pi)) + (nK/2),

    S_const = -(pK/2) log(2 pi) + (pK/2) psi(alpha_omega)
              + (p/2) sum_l (K - l + 1) psi(alpha_delta_l)
              - pK [ (alpha_omega - nu/2) psi(alpha_omega)
                     + log Gamma(nu/2) - log Gamma(alpha_omega)
                     - (nu/2) log(nu/2) - alpha_omega ]
              - sum_l [ (alpha_delta_l - a_l) psi(alpha_delta_l)
                       + log Gamma(a_l) - log Gamma(alpha_delta_l)
                       - alpha_delta_l ].

(The -(pK/2) log(2 pi) normalization of the loading prior is included
in S_const; it is NOT subtracted a second time in C_ELBO -- Section
3.3.7's warning.)

Working precision (see mgp._precision; :class:`mgp.CaviMGP` defaults to
float64).  The stopping decision never
touches the absolute ELBO: it uses the variable part only (the block
byproducts, without Const), which keeps the large invariant constant
out of the floating-point difference.  The constant is combined on the
host, once per chunk, for the reported ELBO history.  With float64 the
evaluation resolution is about |L| * 1e-16, so rounding jitter no
longer binds at practical tolerances; the driver defaults to
tol = 1e-1 with an independent drop budget of 0.5.

This module provides the standalone diagnostic evaluator :func:`elbo`
(re-factorizes Phi and V at a saved state) and the host-side constant
:func:`elbo_const`.  The production path assembles the same value from
the update byproducts; the test-only independent float64 reference
(the five separate Section 3.3 terms and the general form (3.3.9))
lives in tests/reference_numpy.py.
"""

import math

import jax
import jax.numpy as jnp

from ._linalg import cholesky_checked, logdet_from_chol
from .B import log_det_V
from .shrinkage import elbo_shrink_variable, log_mean_delta, shapes

__all__ = ["elbo_const", "elbo"]


def _digamma(x: float) -> float:
    """Digamma for positive arguments, pure Python float64.

    Recurrence psi(x) = psi(x + 1) - 1/x shifts the argument above 10,
    then the standard asymptotic expansion
        psi(x) ~ log x - 1/(2x) - sum_n B_{2n} / (2n x^{2n})
    is evaluated to the B_10 term (truncation error ~1e-15 at x > 10).
    Used only by the host-side ELBO constant; the device path never
    needs a digamma because all Gamma shapes are iteration-independent.
    """
    if x <= 0.0:
        raise ValueError(f"_digamma requires x > 0, got {x}")
    acc = 0.0
    while x < 10.0:
        acc -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    series = inv2 * (
        1.0 / 12.0
        - inv2 * (1.0 / 120.0
                  - inv2 * (1.0 / 252.0
                            - inv2 * (1.0 / 240.0
                                      - inv2 * (1.0 / 132.0))))
    )
    return acc + math.log(x) - 0.5 * inv - series


def elbo_const(
    n: int,
    p: int,
    K: int,
    nu: float,
    a1: float,
    a2: float,
    a_sigma: float,
    b_sigma: float,
) -> float:
    """Constant term of the closed-form ELBO (Const = C_ELBO + S_const).

    Host-side Python float computation, called once per run.  It
    depends only on the dimensions and the hyperparameters, all fixed
    during a CAVI run.  (It cancels in the ELBO differences used for
    the stopping criterion, but is kept so that absolute ELBO values
    can be reported.)

    The Gamma shapes are recomputed here in pure float64 (the device
    path uses float32 arrays via shrinkage.shapes; at large p the
    float32 rounding of alpha_delta ~ 1e4 would leak ~1e-2 absolute
    error into the constant through digamma/log-Gamma).

    Parameters
    ----------
    n, p, K : int
        Observation count, dimension, truncation level.
    nu, a1, a2, a_sigma, b_sigma : float
        Prior hyperparameters (defaults: nu=3, a1=2.1, a2=3.1,
        a_sigma=1, b_sigma=0.3).

    Returns
    -------
    float
    """
    alpha_n = a_sigma + n / 2.0
    c_elbo = (
        p * (
            math.lgamma(alpha_n)
            - math.lgamma(a_sigma)
            + a_sigma * math.log(b_sigma)
        )
        - 0.5 * n * p * math.log(2.0 * math.pi)
        + 0.5 * p * K * (1.0 + math.log(2.0 * math.pi))
        + 0.5 * n * K
    )

    # float64 host recomputation of the (iteration-independent) shapes
    a_delta = [a1 if l == 1 else a2 for l in range(1, K + 1)]
    alpha_delta = [a_l + 0.5 * p * (K - l + 1)
                   for l, a_l in enumerate(a_delta, start=1)]
    alpha_omega = 0.5 * (nu + 1.0)
    half_ln2pi = 0.5 * math.log(2.0 * math.pi)
    s_const = (
        # loading-prior normalization (kept in S, NOT in C_ELBO)
        -p * K * half_ln2pi
        # log-precision constants: (1/2) sum_jk psi(alpha_omega)
        + 0.5 * p * K * _digamma(alpha_omega)
        # and (1/2) sum_jk sum_{l<=k} psi(alpha_delta_l)
        + 0.5 * p * sum(
            (K - l) * _digamma(alpha_delta[l])
            for l in range(K)
        )
        # KL(omega_jk || prior) constants, per (j, k); enters S with a
        # NEGATIVE sign (S subtracts the KL), unlike the C_ELBO terms
        - p * K * (
            (alpha_omega - 0.5 * nu) * _digamma(alpha_omega)
            + math.lgamma(0.5 * nu)
            - math.lgamma(alpha_omega)
            - 0.5 * nu * math.log(0.5 * nu)
            - alpha_omega
        )
        # KL(delta_l || prior) constants, per l
        - sum(
            (alpha_delta[l] - a_delta[l]) * _digamma(alpha_delta[l])
            + math.lgamma(a_delta[l])
            - math.lgamma(alpha_delta[l])
            - alpha_delta[l]
            for l in range(K)
        )
    )
    return c_elbo + s_const


def elbo(
    W: jax.Array,
    Phi: jax.Array,
    beta: jax.Array,
    V: jax.Array,
    mu: jax.Array,
    beta_omega: jax.Array,
    beta_delta: jax.Array,
    nu: float,
    a1: float,
    a2: float,
    a_sigma: float,
    const: float,
) -> tuple[jax.Array, jax.Array]:
    """Standalone closed-form ELBO at a saved state (diagnostic path).

    Recomputes log det Phi and log det V_j by fresh Cholesky
    factorizations of the saved covariances (no byproduct reuse), so it
    is an independent check of the production assembly.  The local
    shrinkage means are derived internally as rho = alpha_omega /
    beta_omega (never trusted from the caller), and the second moments
    e_jk = mu_jk^2 + (V_j)_kk are floored at the dtype minimum as in
    the production path.  The compact form assumes the state is a
    sweep boundary: beta_j = b_sigma + R_j / 2 and the shrinkage rates
    as stored (these identities were substituted in Section 3.3's
    assembly); evaluating it at non-synchronized states requires the
    (3.3.9) correction, provided by the test reference.

    Parameters
    ----------
    W : jax.Array, shape (n, K)
    Phi : jax.Array, shape (K, K)
    beta : jax.Array, shape (p,)
    V : jax.Array, shape (p, K, K)
    mu : jax.Array, shape (p, K)
        Loading means (e_jk = mu_jk^2 + V_jkk is recomputed here).
    beta_omega : jax.Array, shape (p, K)
    beta_delta : jax.Array, shape (K,)
    nu, a1, a2, a_sigma : float
        Prior hyperparameters.
    const : float
        Precomputed ELBO constant from :func:`elbo_const`.

    Returns
    -------
    L : jax.Array, scalar
        The absolute ELBO (Const included).
    ok : jax.Array, scalar bool
        Validity flags of the two diagnostic factorizations.
    """
    n = W.shape[0]
    p, K = V.shape[0], V.shape[1]
    alpha_n = a_sigma + n / 2.0
    alpha_omega, alpha_delta, _ = shapes(p, K, nu, a1, a2)

    L_phi, ok_phi = cholesky_checked(Phi)
    log_det_Phi = logdet_from_chol(L_phi)
    ldV, ok_V = log_det_V(V)

    e = mu * mu + jnp.diagonal(V, axis1=1, axis2=2)
    e = jnp.maximum(e, jnp.finfo(e.dtype).tiny)
    rho = jnp.asarray(alpha_omega, e.dtype) / beta_omega
    _, log_mbar = log_mean_delta(beta_delta, alpha_delta)
    s_var = elbo_shrink_variable(
        beta_omega, rho, beta_delta, e, log_mbar,
        jnp.asarray(nu, e.dtype), alpha_delta, jnp.asarray(alpha_omega, e.dtype),
    )

    L = (
        const
        + 0.5 * n * log_det_Phi
        - 0.5 * ((W * W).sum() + n * jnp.trace(Phi))
        + 0.5 * ldV.sum()
        - alpha_n * jnp.log(beta).sum()
        + s_var
    )
    return L, ok_phi & ok_V
