"""Multiplicative gamma process (MGP) shrinkage block (working precision).

The prior precision of a loading entry factorizes into a
LOCAL and a CUMULATIVE GLOBAL part,

    B_jk | omega_jk, T_k ~ N(0, (omega_jk * T_k)^{-1}),
    T_k = prod_{l<=k} delta_l,

and the mean-field family of MGP_CAVI.md Section 3.1 gives each
omega_jk and each delta_l its own Gamma factor:

    q(omega_jk) = Gamma(alpha_omega, beta_omega_jk)   (Component 3)
    q(delta_l)  = Gamma(alpha_delta_l, beta_delta_l)  (Component 4)

with iteration-INDEPENDENT shapes

    alpha_omega   = (nu + 1) / 2,
    alpha_delta_l = a_l + (p/2)(K - l + 1),   a_1 = a1, a_l = a2 (l >= 2),

and coordinate-optimal rates (the fixed-point conditions of Section 3.1)

    beta_omega_jk = (nu + mbar_k * e_jk) / 2,
    beta_delta_l  = 1 + (1/2) sum_{k>=l} (prod_{r<=k, r!=l} m_r)
                        * sum_j rho_jk e_jk,

where
    e_jk   = E_q[B_jk^2] = mu_jk^2 + (V_j)_kk,
    rho_jk = E_q[omega_jk] = alpha_omega / beta_omega_jk,
    m_l    = E_q[delta_l] = alpha_delta_l / beta_delta_l,
    mbar_k = E_q[T_k] = prod_{l<=k} m_l   (independence of the q(delta_l)).

The delta rates are COUPLED through the means of the other delta_r
(each beta_delta_l depends on prod_{r != l} m_r), so the K updates are
applied SEQUENTIALLY as coordinate updates, l = 1, ..., K, each using
the freshest means (Section 3.1, note at the end of Component 4; a
single parallel update with stale means need not increase the ELBO).
The K steps run as a lax.scan: the step body is compiled once, so the
XLA graph and the AOT compile time do not grow with K (a local
performance patch; each step computes exactly the former unrolled loop
body, so the numerical result is unchanged).

Numerical discipline (an implementation convention of this package, not
a section of MGP_CAVI.md: the products prod_l alpha_l/beta_l are
computed in log space), strengthened after an adversarial audit of this
module:

* the cumulative products mbar_k are carried as their LOGARITHMS,
  theta_l = log m_l and log_mbar_k = cumsum(theta); at the defaults
  (a2 = 3.1, large p) the materialized product can exceed the working
  dtype's range (float32 overflows at moderate K; float64 can overflow
  for large K or extreme draws) while its logarithm stays finite;
* every quantity of the form rho_jk * mbar_k * e_jk is assembled as
  exp(log rho + log mbar + log e), never by multiplying materialized
  factors; at the omega fixed point this quantity is bounded by
  nu + 1 (mbar e = 2 beta_omega - nu), so the exponent stays well
  inside the range of either working dtype in every self-consistent
  state;
* every SUM of exponentials (the delta rates and the ELBO quadratic
  term) is evaluated with logsumexp: the summands are combined as
  logarithms, so no intermediate ever passes through exp-overflow
  (a naive exp-then-sum produces inf for exponents beyond the exp
  ceiling -- ~88.7 in float32, ~709.8 in float64 -- even when the
  capped result is well defined), and the value is EXACT whenever the
  true sum lies below the cap;
* the rate caps act on the EXPONENT (exp(min(x, log cap))): beta_omega
  and beta_delta are capped at 1e30 and the summed contributions at
  2e30 -- caps that sit far beyond any statistically meaningful scale
  (a rate above 1e30 means a shrinkage mean below ~1e-30, i.e. a dead
  parameter) and only bound regimes unreachable through the CAVI
  dynamics; the log-space pipeline requires only that log(rate) and
  alpha/rate stay finite, which the caps guarantee;
* e_jk is floored at jnp.finfo(dtype).tiny before entering a logarithm
  (see second_moment_B).

ELBO contributions of this block (Section 3.3.5, equation (3.3.7b)).
Splitting the (3.3.7b) form into variable and constant parts (all
shapes are iteration-independent, so every digamma / log-Gamma term is
constant) gives the VARIABLE part carried per sweep:

    S_var = -alpha_omega * sum_jk log beta_omega_jk
            - (1/2) sum_jk rho_jk mbar_k e_jk
            - (nu/2) sum_jk rho_jk
            - sum_l alpha_delta_l log beta_delta_l
            - sum_l m_l,

and the CONSTANT part (host, once per run; see elbo.elbo_const):

    S_const = -(pK/2) log 2pi + (pK/2) psi(alpha_omega)
              + (p/2) sum_l (K - l + 1) psi(alpha_delta_l)
              - pK * [ (alpha_omega - nu/2) psi(alpha_omega)
                       + log Gamma(nu/2) - log Gamma(alpha_omega)
                       - (nu/2) log(nu/2) - alpha_omega ]
              - sum_l [ (alpha_delta_l - a_l) psi(alpha_delta_l)
                       + log Gamma(a_l) - log Gamma(alpha_delta_l)
                       - alpha_delta_l ].

Consistency check: at the Component 3 optimum, mbar_k e_jk =
2 beta_omega_jk - nu, so the first three terms collapse per (j, k) to
-alpha_omega (log beta_omega_jk + 1) -- the same shape as the sigma2
block's -alpha_n log beta_j (Section 3.3.5's remark).

Column indices l, k are 1-based throughout, as in MGP_CAVI.md.

Reference: MGP_CAVI.md Sections 2.1 (prior), 3.1 (Components 3-4),
3.3.5 (Term 4).  The log-space discipline above is an implementation
convention of this package (it is not prescribed by MGP_CAVI.md); the
default hyperparameters nu=3, a1=2.1, a2=3.1 follow Durante (2017)'s
practical recommendations rather than MGP_CAVI.md.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

__all__ = [
    "shapes",
    "initial_rates",
    "second_moment_B",
    "log_mean_delta",
    "log_prior_precision",
    "update_omega",
    "update_delta",
    "update_shrinkage",
    "elbo_shrink_variable",
]

_F64 = jnp.float64  # package working precision (x64 enabled at import)

# Rate / sum caps (linear values; the clamps act on log scale).  Beyond
# any statistically meaningful scale, existing only to bound regimes
# unreachable through the CAVI dynamics (see module docstring).  Both
# keep log(rate) <= 69.1 and alpha/rate > 0 finite.
BETA_CAP = 1e30
SUM_CAP = 2e30
LOG_BETA_CAP = jnp.float64(jnp.log(BETA_CAP))
LOG_SUM_CAP = jnp.float64(jnp.log(SUM_CAP))


def shapes(p: int, K: int, nu: float, a1: float, a2: float):
    """Iteration-independent shapes of the MGP Gamma factors.

    Parameters
    ----------
    p, K : int
        Dimension and truncation level.
    nu, a1, a2 : float
        MGP hyperparameters (nu for the local omega_jk; a1 for
        delta_1; a2 for delta_l, l >= 2).

    Returns
    -------
    alpha_omega : float
        (nu + 1) / 2, the shape of every q(omega_jk).
    alpha_delta : jax.Array, shape (K,)
        a_l + (p/2)(K - l + 1) for l = 1, ..., K (working dtype).
    a_delta : jax.Array, shape (K,)
        The prior shapes (a1, a2, ..., a2) (working dtype).
    """
    ell = jnp.arange(1, K + 1, dtype=_F64)          # 1-based column index
    a_delta = jnp.where(ell == 1, jnp.asarray(a1, _F64), jnp.asarray(a2, _F64))
    alpha_delta = a_delta + 0.5 * p * (K - ell + 1)
    return 0.5 * (nu + 1.0), alpha_delta, a_delta


def initial_rates(alpha_delta: jax.Array) -> jax.Array:
    """Initialization convention: shrinkage means equal to one.

    m_l = 1 amounts, with the shape fixed, to setting the rate to
    shape / mean:

        beta_delta_l = alpha_delta_l.

    (q(omega_jk) needs no initial rate: its optimal update is
    closed-form in (e, mbar) and never reads its previous rate, so the
    omega factor is always re-derived from the Gaussian state.)

    Returns
    -------
    beta_delta0 : jax.Array, shape (K,)
    """
    return alpha_delta


def second_moment_B(mu: jax.Array, V: jax.Array) -> jax.Array:
    """Second moments e_jk = E_q[B_jk^2] = mu_jk^2 + (V_j)_kk.

    The result is floored at jnp.finfo(dtype).tiny: it enters
    logarithms below, where a zero or subnormal entry would produce
    -inf / NaN.  The floor is far below any statistically relevant
    scale, so the guard has no statistical effect.

    Parameters
    ----------
    mu : jax.Array, shape (p, K)
    V : jax.Array, shape (p, K, K)

    Returns
    -------
    jax.Array, shape (p, K)
    """
    e = mu * mu + jnp.diagonal(V, axis1=1, axis2=2)
    return jnp.maximum(e, jnp.finfo(e.dtype).tiny)


def log_mean_delta(beta_delta: jax.Array, alpha_delta: jax.Array):
    """Log means of the global shrinkage increments, and log E[T_k].

    theta_l   = log m_l = log alpha_delta_l - log beta_delta_l,
    log_mbar_k = sum_{l<=k} theta_l   (= log E_q[T_k], the log-space
    cumulative product; implementation convention).

    Parameters
    ----------
    beta_delta : jax.Array, shape (K,)
        Rates of q(delta_l).
    alpha_delta : jax.Array, shape (K,)
        Shapes of q(delta_l).

    Returns
    -------
    theta : jax.Array, shape (K,)
    log_mbar : jax.Array, shape (K,)
    """
    theta = jnp.log(alpha_delta) - jnp.log(beta_delta)
    return theta, jnp.cumsum(theta)


def log_prior_precision(
    beta_omega: jax.Array, log_mbar: jax.Array, alpha_omega: jax.Array
) -> jax.Array:
    """log of the expected prior precision E_q[omega_jk T_k].

    log_prec[j, k] = log rho_jk + log mbar_k
                   = log alpha_omega - log beta_omega_jk + log_mbar_k,

    the quantity the whitened B update factorizes on (see B.compute_V).
    Finite by the shrinkage module's rate caps.

    Parameters
    ----------
    beta_omega : jax.Array, shape (p, K)
    log_mbar : jax.Array, shape (K,)
    alpha_omega : jax.Array, scalar

    Returns
    -------
    jax.Array, shape (p, K)
    """
    return jnp.log(alpha_omega) - jnp.log(beta_omega) + log_mbar[None, :]


def update_omega(
    e: jax.Array, log_mbar: jax.Array, nu: jax.Array, alpha_omega: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """One CAVI update of the local shrinkage factors (Component 3).

    beta_omega_jk = (nu + mbar_k e_jk) / 2
                  = (nu + exp(log_mbar_k + log e_jk)) / 2,
    rho_jk        = alpha_omega / beta_omega_jk.

    The exponential is assembled in log space (never as a materialized
    mbar_k * e_jk product) and clamped at the exponent, so the rate
    never passes through inf: beta_omega_jk <= (nu + BETA_CAP) / 2 and
    rho_jk >= alpha_omega / BETA_CAP > 0, keeping log(rho) finite for
    the downstream log-space pipeline.

    Parameters
    ----------
    e : jax.Array, shape (p, K)
        Second moments E_q[B_jk^2], strictly positive.
    log_mbar : jax.Array, shape (K,)
        Log expected global shrinkage, log E_q[T_k].
    nu : jax.Array, scalar
    alpha_omega : jax.Array, scalar

    Returns
    -------
    beta_omega : jax.Array, shape (p, K)
    rho : jax.Array, shape (p, K)
        Local means E_q[omega_jk].
    """
    mbar_e = jnp.exp(jnp.minimum(log_mbar[None, :] + jnp.log(e), LOG_BETA_CAP))
    beta_omega = 0.5 * (nu + mbar_e)
    rho = alpha_omega / beta_omega
    return beta_omega, rho


def update_delta(
    rho: jax.Array,
    e: jax.Array,
    beta_delta: jax.Array,
    alpha_delta: jax.Array,
) -> jax.Array:
    """One CAVI pass of the global shrinkage factors (Component 4).

    The K coordinate updates are applied SEQUENTIALLY, l = 1, ..., K,
    each at the exact conditional optimum given the current means of
    the other increments (fresh for r < l, stale for r > l):

        beta_delta_l = 1 + (1/2) sum_{k>=l} sum_j
                       exp(log rho_jk + log e_jk
                           + sum_{r<=k} theta_r - theta_l),

    where the inner exponential equals
    rho_jk e_jk prod_{r<=k, r!=l} m_r in linear space.  Everything is
    assembled in log space and the per-l sum of exponentials is
    evaluated with logsumexp, so no intermediate ever passes through
    exp-overflow (a naive exp-then-sum yields inf whenever a single
    exponent exceeds the working dtype's exp ceiling -- ~88.7 in
    float32, ~709.8 in float64 -- even when the capped rate is well
    defined); the summed contribution is bounded by exp(LOG_SUM_CAP).

    Parameters
    ----------
    rho : jax.Array, shape (p, K)
        Local shrinkage means from :func:`update_omega` (current).
    e : jax.Array, shape (p, K)
        Second moments E_q[B_jk^2], strictly positive.
    beta_delta : jax.Array, shape (K,)
        Current rates; the starting point of the sequential pass (the
        r > l entries keep these values until their own turn).
    alpha_delta : jax.Array, shape (K,)
        Iteration-independent shapes.

    Returns
    -------
    beta_delta : jax.Array, shape (K,)
        Updated rates.
    """
    log_alpha = jnp.log(alpha_delta)
    G = jnp.log(rho) + jnp.log(e)  # (p, K), l-independent part
    cols = jnp.arange(beta_delta.shape[0])
    neg_inf = jnp.asarray(-jnp.inf, dtype=G.dtype)

    def step(beta, l):
        """One coordinate update, l = 0, ..., K - 1 (compiled ONCE)."""
        theta = log_alpha - jnp.log(beta)
        cum = jnp.cumsum(theta)
        # exponent of rho_jk e_jk prod_{r<=k, r!=l} m_r, zero for k < l
        expo = G + (cum - theta[l])[None, :]
        expo = jnp.where(cols[None, :] < l, neg_inf, expo)
        contribution = jnp.exp(jnp.minimum(logsumexp(expo), LOG_SUM_CAP))
        beta = beta.at[l].set(1.0 + 0.5 * contribution)
        return beta, None

    # lax.scan over the K sequential coordinate updates: the step body is
    # compiled once instead of being unrolled at trace time, so the XLA
    # graph (and the AOT compile time) no longer grows with K.  Each step
    # is exactly the computation of the former unrolled loop body, so the
    # numerical result is unchanged.
    beta, _ = jax.lax.scan(step, beta_delta, cols)
    return beta


def update_shrinkage(
    mu: jax.Array,
    V: jax.Array,
    beta_delta: jax.Array,
    nu: jax.Array,
    alpha_delta: jax.Array,
    alpha_omega: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """One CAVI update of Components 3 and 4 (omega then delta).

    Parameters
    ----------
    mu : jax.Array, shape (p, K)
        Current loading means (from this sweep's B update).
    V : jax.Array, shape (p, K, K)
        Current loading covariances.
    beta_delta : jax.Array, shape (K,)
        Current delta rates: the starting point of the sequential
        Component 4 pass, and the source of the mbar used by the
        Component 3 update (coordinate ascent: omega is updated given
        the CURRENT q(delta)).
    nu, alpha_delta, alpha_omega : scalars / (K,) arrays
        Hyperparameter-derived constants from :func:`shapes`.

    Returns
    -------
    beta_omega : jax.Array, shape (p, K)
        Updated rates of q(omega_jk).
    rho : jax.Array, shape (p, K)
        Local means E_q[omega_jk].
    beta_delta : jax.Array, shape (K,)
        Updated rates of q(delta_l).
    log_mbar : jax.Array, shape (K,)
        log E_q[T_k] at the updated rates.
    elbo_shrink : jax.Array, scalar
        This block's VARIABLE ELBO contribution S_var (see module
        docstring), evaluated at the returned rates.
    """
    e = second_moment_B(mu, V)
    _, log_mbar = log_mean_delta(beta_delta, alpha_delta)
    beta_omega, rho = update_omega(e, log_mbar, nu, alpha_omega)
    beta_delta = update_delta(rho, e, beta_delta, alpha_delta)
    _, log_mbar = log_mean_delta(beta_delta, alpha_delta)
    elbo_shrink = elbo_shrink_variable(
        beta_omega, rho, beta_delta, e, log_mbar, nu, alpha_delta, alpha_omega
    )
    return beta_omega, rho, beta_delta, log_mbar, elbo_shrink


def elbo_shrink_variable(
    beta_omega: jax.Array,
    rho: jax.Array,
    beta_delta: jax.Array,
    e: jax.Array,
    log_mbar: jax.Array,
    nu: jax.Array,
    alpha_delta: jax.Array,
    alpha_omega: jax.Array,
) -> jax.Array:
    """Variable part S_var of the shrinkage block's ELBO contribution.

    S_var = -alpha_omega sum_jk log beta_omega_jk
            - (1/2) sum_jk rho_jk mbar_k e_jk
            - (nu/2) sum_jk rho_jk
            - sum_l alpha_delta_l log beta_delta_l
            - sum_l m_l,

    evaluated at the passed state (valid at ANY state: the constant /
    variable split is exact everywhere because all Gamma shapes are
    iteration-independent).  The quadratic sum is assembled in log
    space and evaluated with logsumexp (no inf intermediates; exact
    below the cap, bounded by exp(LOG_SUM_CAP) beyond it).

    Parameters
    ----------
    beta_omega, rho, e : jax.Array, shape (p, K)
    beta_delta : jax.Array, shape (K,)
    log_mbar : jax.Array, shape (K,)
    nu : jax.Array, scalar
    alpha_delta : jax.Array, shape (K,)
    alpha_omega : jax.Array, scalar

    Returns
    -------
    jax.Array, scalar
    """
    log_quad = logsumexp(jnp.log(rho) + jnp.log(e) + log_mbar[None, :])
    quad = jnp.exp(jnp.minimum(log_quad, LOG_SUM_CAP))
    m_delta = alpha_delta / beta_delta
    return (
        -alpha_omega * jnp.log(beta_omega).sum()
        - 0.5 * quad
        - 0.5 * nu * rho.sum()
        - (alpha_delta * jnp.log(beta_delta)).sum()
        - m_delta.sum()
    )
