"""Fixed-structure state containers for the compiled CAVI loop.

Every leaf keeps a fixed shape/dtype across one compiled run: all
numerical leaves share the package working dtype (float32 as the
process-wide default; the :class:`mgp.CaviMGP` estimator defaults to
float64 for the duration of ``fit``; switchable with
:func:`mgp.set_precision` or per estimator through its ``precision``
parameter), iteration counters and error enumerations are int32,
logical flags are bool.  Both containers are plain NamedTuples, hence
JAX PyTrees by default.

The MGP recurrence state carries the two shrinkage rate vectors of the
multiplicative gamma process in addition to the Gaussian / noise
factors:

    beta_omega (p, K)  rates of q(omega_jk) = Gamma((nu+1)/2, beta_omega)
    beta_delta (K,)    rates of q(delta_l)  = Gamma(a_l + p(K-l+1)/2, ...)

(the shapes are iteration-independent, Section 3.1, so only the rates
are carried; the local means rho_jk and the cumulative products
T_k-hat = prod_l E[delta_l] are derived quantities).

Error codes (host maps them to exceptions at chunk boundaries):
    0  ERR_NONE
    1  ERR_NONFINITE        -- non-finite state / ELBO
    2  ERR_FACTORIZATION    -- Cholesky / inversion validity flag fired
    3  ERR_ELBO_DROP        -- negative ELBO change beyond the jitter budget

Module ids (which block first failed):
    0 none, 1 B, 2 eta, 3 sigma2, 4 omega, 5 delta, 6 ELBO/stopping
"""

from typing import NamedTuple

import jax

__all__ = [
    "Carry",
    "Inputs",
    "ERR_NONE",
    "ERR_NONFINITE",
    "ERR_FACTORIZATION",
    "ERR_ELBO_DROP",
    "MODULE_NAMES",
]

ERR_NONE = 0
ERR_NONFINITE = 1
ERR_FACTORIZATION = 2
ERR_ELBO_DROP = 3

MODULE_NAMES = {
    0: "none",
    1: "B",
    2: "eta",
    3: "sigma2",
    4: "omega",
    5: "delta",
    6: "elbo",
}


class Carry(NamedTuple):
    """The loop carry: minimal recurrence state plus outputs and control.

    mu, V, beta_omega, beta_delta : the recurrence core of one sweep.
    W, Phi, beta                  : outputs of the last completed sweep
                                    (kept so the returned state needs no
                                    recomputation).
    prev_elbo : variable part of the ELBO (no constant) of the last
                completed sweep, working-dtype scalar.
    t         : int32 global sweep count (completed sweeps).
    converged : bool, |delta| < tol at the last completed sweep.
    error, error_t, error_module : first-failure record.
    """

    mu: jax.Array          # (p, K)
    V: jax.Array           # (p, K, K)
    beta_omega: jax.Array  # (p, K)
    beta_delta: jax.Array  # (K,)
    W: jax.Array           # (n, K)
    Phi: jax.Array         # (K, K)
    beta: jax.Array        # (p,)
    prev_elbo: jax.Array   # working-dtype scalar
    t: jax.Array           # int32 scalar
    converged: jax.Array   # bool scalar
    error: jax.Array       # int32 scalar
    error_t: jax.Array     # int32 scalar
    error_module: jax.Array  # int32 scalar


class Inputs(NamedTuple):
    """Run-constant inputs (traced, so value changes do not recompile).

    X           : (n, p) working-dtype data (column-centered by the driver).
    alpha_omega : working-dtype scalar, shape (nu + 1)/2 of every q(omega_jk).
    alpha_delta : (K,) shapes a_l + (p/2)(K - l + 1), l = 1..K.
    nu          : working-dtype scalar, local shrinkage hyperparameter.
    alpha_n     : working-dtype scalar, a_sigma + n/2.
    a_sigma     : working-dtype scalar noise prior shape.
    b_sigma     : working-dtype scalar noise prior rate.
    tol         : working-dtype scalar absolute ELBO (variable part) tolerance.
    drop_budget : working-dtype scalar ELBO jitter budget.
    max_iter    : int32 scalar, global sweep cap.
    """

    X: jax.Array
    alpha_omega: jax.Array
    alpha_delta: jax.Array
    nu: jax.Array
    alpha_n: jax.Array
    a_sigma: jax.Array
    b_sigma: jax.Array
    tol: jax.Array
    drop_budget: jax.Array
    max_iter: jax.Array
