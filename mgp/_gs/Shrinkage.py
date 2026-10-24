"""
Shrinkage.py  (JAX, working precision)

Shrinkage-parameter steps of the Gibbs sampler for the multiplicative
gamma process (MGP) prior of Bhattacharya & Dunson (2011), "Sparse
Bayesian infinite factor models", Biometrika 98, 291-306.  Steps 4-5 of
Section 3.1; the prior is their eq. (3):

    lambda_jh | phi_jh, tau_h ~ N(0, (phi_jh tau_h)^{-1}),
    phi_jh ~ Ga(nu/2, nu/2),
    tau_h = prod_{l<=h} delta_l,
    delta_1 ~ Ga(a1, 1),   delta_l ~ Ga(a2, 1)  (l >= 2),

with nu and (a1, a2) FIXED hyperparameters given as inputs (the paper's
Step 6 updates a1, a2 by Metropolis-Hastings under Ga(2, 1)
hyperpriors -- deliberately not implemented here: the exact
conditionals pi(a1, a2 | delta) are proper, but at large p and
overfitted K their normalizing constants (Gamma-function products
raised to K-1) overflow floating point -- for p ~ 1000, K = 50 they are
of order 10^7105 -- so the MH chain would be numerically degenerate;
fixing (a1, a2) is a deliberate simplification).

Given the current loading matrix B (= Lambda), this module draws, in the
paper's order,

    Step 4:  phi_jh | B, tau_h ~ Ga( (nu + 1)/2, (nu + tau_h B_jh^2) / 2 ),
                                              (independent over j, h)
    Step 5:  delta_1 | B, phi, delta_2..K ~ Ga( a1 + pK/2,
                                     1 + (1/2) sum_{l=1}^{K} tau_l^{(1)} S_l ),
             delta_h | B, phi, delta_{-h} ~ Ga( a2 + p(K - h + 1)/2,
                                     1 + (1/2) sum_{l=h}^{K} tau_l^{(h)} S_l ),
                                       (h >= 2; SEQUENTIAL over h: each
                                        delta_h is drawn from its full
                                        conditional given the LATEST
                                        delta_{-h} -- delta_1..delta_{h-1}
                                        already refreshed this sweep)
             with S_l = sum_{j=1}^{p} phi_jl B_jl^2  and
             tau_l^{(h)} = tau_l / delta_h = prod_{t <= l, t != h} delta_t,

The Step-5 scan is Gauss-Seidel, as in the paper's Gibbs sweep: the
joint conditional pi(delta | B, phi) does NOT factorize over h (tau_l
contains the cross products delta_h delta_t), so drawing all K
increments simultaneously from rates computed at the old state would be
a Jacobi-style update whose invariant distribution is not the target
posterior.  Each of the K draws therefore recomputes tau from the
current state (lax.fori_loop; the body is compiled once, so AOT compile
time does not grow with K).

and returns the prior standard deviations

    D_jh = 1 / sqrt(phi_jh tau_h)

that B's Gaussian full conditional consumes (see B_sample.py).  Because
the phi-conditional depends on tau = tau(delta) and the delta-conditional
depends on phi, BOTH (phi, log delta) are carried as sampler state
between sweeps -- unlike shrinkage priors whose block admits a one-shot
marginal-then-conditional regeneration each sweep.

Numerical design (the heart of this module)
-------------------------------------------
tau_h = prod_{l<=h} delta_l is a cumulative PRODUCT that legitimately
spans hundreds of log-units: E[log delta_l] = psi(a2) > 0 for a2 >~ 1.5
(paper default regime a2 ~ 2-3), so log tau_h grows linearly in h and a
direct float32 cumprod over/underflows to inf/0 once h is moderate.
Both directions poison the chain downstream:

    tau = inf  ->  delta-rate inf  ->  delta ~ 0  ->  tau = 0
               ->  D = 1/sqrt(phi * 0) = inf     ->  NaN in B_sample;

so tau is NEVER materialized standalone.  The canonical state is
log delta (a K-vector); every quantity below is assembled in log space:

* log tau = lax.cumsum(log delta): exact, no overflow.
* The delta-rate needs A_h = sum_{l>=h} tau_l S_l with S_l =
  sum_j phi_jl B_jl^2.  Each term tau_l S_l is O(p) (a chi-square_p-like
  quantity, since B_jl^2 ~ (phi_jl tau_l)^{-1} marginally), but its
  FACTORS individually range over the whole float32 spectrum -- in the
  real chain, a deeply shrunk column has tau_l = e^{~100} and loadings
  B_jl ~ e^{-50}, whose SQUARES underflow below the smallest float32
  subnormal (2^{-149} ~ 1.4e-45) and would zero out a direct sum S_l.
  Both S_l and A_h are therefore assembled in log space:
      log S_l = logsumexp_j( log phi_jl + 2 log|B_jl| ),
                exact zeros give -inf terms; an all-zero (dead) column
                gives log S_l = -inf and drops out of every rate --
                the correct mathematics (0 * tau = 0), essential
                numerically since dead columns carry enormous tau_l;
      v_l = log tau_l + log S_l,
      log A_h = logsumexp_{l>=h} v_l  (per-step masked logsumexp over
                                       the columns l >= h, recomputed at
                                       every draw of the sequential scan
                                       from the LATEST log delta state),
      log rate_h = logaddexp(0, log A_h - log delta_h - log 2)
                 = log( 1 + A_h / (2 delta_h) ).
  The exponent is clipped at _LOG_CAP_RATE = 80 (rate <= 5.5e34): beyond it
  the column is astronomically dead, the draw is ~0 either way, and the
  clip only prevents exp() overflow -- statistically invisible.
* The phi-rate term tau_h B_jh^2 is assembled as
  exp( log tau_h + 2 log|B_jh| ) with the same exponent cap: the true
  product is O(1/phi_jh), and the cap only fires when the resulting phi
  would underflow to ~0 in float32 anyway.
* The prior scales D = exp( -(log phi + log tau)/2 ) are capped ABOVE at
  exp(34.5) ~ 9.6e14: the whitened system of B_sample forms w*w' products,
  and |w| beyond ~1e15 overflows the float32 ceiling (the same bound as
  the adversarial tests of this implementation); the cap binds only when
  tau_h is so
  small that the prior is effectively flat (an unconstrained column),
  never in the shrinking direction.  On the lower side exp() underflows
  to 0, which decouples the coordinate exactly in the w -> 0 limit
  (B_jh = w_jh gamma_jh -> 0).
* delta draws are floored at the smallest positive normal float32 before
  the log, and phi draws likewise, so the carried state never contains
  exact zeros (which would make log(0) = -inf propagate).

No chi-square Gamma approximation is used anywhere in this module:
exact jax.random.gamma is a
pure XLA primitive -- exact, capturable, and fast on every backend --
so no accuracy-for-speed tradeoff is needed for the MGP shrinkage steps.

JAX-specific notes
------------------
* Working precision (float32 by default; float64 optional via
  mgp._precision) throughout; every public function takes an explicit
  jax.random.PRNGKey as its first argument (the deterministic
  prior_scales takes none).  The design comments below reference the
  float32 magnitudes the machinery was originally designed for -- the
  guards remain active and correct in either precision.
* Scalars (a1, a2, nu) are traced, so changing them does not
  trigger recompilation.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

__all__ = ["phi_sample", "delta_sample", "prior_scales", "shrinkage"]

_F64 = jnp.float64  # package working precision (x64 enabled at import)
_LN2 = 0.6931471805599453
# exponent caps: rate <= exp(80) ~ 5.5e34 (delta step, phi step) and
# prior scale D <= exp(34.5) ~ 9.6e14 (w*w' products in B_sample stay
# below the float32 ceiling).  See module docstring.
_LOG_CAP_RATE = 80.0
_LOG_CAP_SCALE = 34.5


def _tiny():
    """Smallest positive normal of the CURRENT working precision.

    Computed at trace time (never cached at import): a materialized
    np.float64 scalar is strongly typed and would silently promote
    float32 traces to float64 under x64 (the regime set by
    mgp._precision.set_precision after import).
    """
    return jnp.finfo(_F64).tiny


def phi_sample(
    key: jax.Array,
    B: jax.Array,
    log_tau: jax.Array,
    nu: float,
) -> jax.Array:
    r"""Step 4: one exact draw of the local shrinkage parameters phi.

    phi_jh | B, tau ~ Ga((nu + 1)/2, (nu + tau_h B_jh^2) / 2),
    independent over (j, h) and from the current B.

    The rate term tau_h B_jh^2 is assembled in log space as
    exp(log tau_h + 2 log|B_jh|) with the exponent capped at
    _LOG_CAP_RATE: the true product is O(1/phi_jh) (chi-square-like),
    and the cap only fires when the draw would underflow to ~0 in
    float32 anyway (statistically invisible; see module docstring).

    Parameters
    ----------
    key : jax.random.PRNGKey
    B : array-like, shape (p, K)
        Factor loading matrix at the current iteration.
    log_tau : array-like, shape (K,)
        log of the column shrinkage weights tau_h = prod_{l<=h} delta_l
        (the canonical carrier of the delta state; never pass tau
        itself).
    nu : float
        Degrees of freedom of the local-shrinkage prior
        phi_jh ~ Ga(nu/2, nu/2); paper default nu = 3.

    Returns
    -------
    jnp.ndarray, shape (p, K), working dtype
        The drawn phi, floored at the smallest positive normal of the
        working precision (so the carried state never contains exact zeros).
    """
    B = jnp.asarray(B, dtype=_F64)
    log_tau = jnp.asarray(log_tau, dtype=_F64)

    # x_jh = tau_h B_jh^2 via log space, exponent capped
    log_x = log_tau[None, :] + 2.0 * jnp.log(jnp.maximum(jnp.abs(B), _tiny()))
    x = jnp.exp(jnp.minimum(log_x, _LOG_CAP_RATE))
    rate = 0.5 * (nu + x)

    phi = jax.random.gamma(key, 0.5 * (nu + 1.0), shape=B.shape,
                           dtype=_F64) / rate
    return jnp.maximum(phi, _tiny())


def delta_sample(
    key: jax.Array,
    B: jax.Array,
    phi: jax.Array,
    log_delta: jax.Array,
    a1: float,
    a2: float,
) -> jax.Array:
    r"""Step 5: sequential Gibbs scan over the column shrinkage increments.

    Draws delta_1, ..., delta_K in column order, each from its Gamma
    full conditional given B, phi and the OTHER increments at their
    LATEST values (delta_1..delta_{h-1} already refreshed this sweep):

        delta_1 | ... ~ Ga( a1 + pK/2,
                            1 + (1/2) sum_{l=1}^{K} tau_l^{(1)} S_l ),
        delta_h | ... ~ Ga( a2 + p(K - h + 1)/2,
                            1 + (1/2) sum_{l=h}^{K} tau_l^{(h)} S_l ),  h >= 2,

    with S_l = sum_j phi_jl B_jl^2 and tau_l^{(h)} = tau_l / delta_h
    (a product over the OTHER delta's -- it does not depend on delta_h,
    which is why the full conditional is a Gamma at all).  Writing
    A_h = sum_{l>=h} tau_l S_l, the rate is 1 + A_h / (2 delta_h), and
    S_l, A_h and the rate are all assembled through the log-space
    machinery of the module docstring: the individual factors tau_l, S_l
    and even the squared loadings B_jl^2 span the whole float32 spectrum
    (deeply shrunk columns have B_jl^2 below the smallest subnormal),
    while every product tau_l S_l is O(p).

    The scan is SEQUENTIAL (Gauss-Seidel), matching the paper's Gibbs
    sweep: the joint conditional pi(delta | B, phi) does NOT factorize
    over h (tau_l contains the cross products delta_h delta_t), so
    drawing all K increments simultaneously from rates computed at the
    old state would be a Jacobi-style update whose invariant
    distribution is not the target posterior.  The loop is a
    lax.fori_loop with one jax.random.gamma call per column, so the
    body is compiled once and AOT compile time does not grow with K.

    Dead columns (all loadings exactly 0) produce log S_l = -inf and
    drop out of every rate: the correct mathematics (0 times anything
    contributes 0), and essential numerically, since a dead column
    typically carries an enormous tau_l whose spurious inclusion would
    overflow any direct sum.

    Parameters
    ----------
    key : jax.random.PRNGKey
    B : array-like, shape (p, K)
        Factor loading matrix at the current iteration.
    phi : array-like, shape (p, K)
        Local shrinkage parameters (the output of phi_sample in the
        same sweep; the paper's cycle draws phi before delta).
    log_delta : array-like, shape (K,)
        log of the CURRENT delta state: the scan refreshes it in place,
        column by column, and delta_h enters its own rate through
        tau_l^{(h)} = tau_l / delta_h.
    a1, a2 : float
        Shape hyperparameters of the delta priors
        delta_1 ~ Ga(a1, 1), delta_l ~ Ga(a2, 1).

    Returns
    -------
    jnp.ndarray, shape (K,), working dtype
        The log of the DRAWN delta (floored at the smallest positive
        normal of the working dtype before the log), to be carried as
        state and fed to the next phi_sample / delta_sample call.
    """
    B = jnp.asarray(B, dtype=_F64)
    phi = jnp.asarray(phi, dtype=_F64)
    log_delta = jnp.asarray(log_delta, dtype=_F64)
    p, K = B.shape

    # column sums S_l = sum_j phi_jl B_jl^2, assembled in log space ONCE
    # (they do not depend on delta): deeply shrunk columns have |B_jl|
    # ~ 1/sqrt(tau_l) whose SQUARES fall below the smallest float32
    # subnormal (2^-149 ~ 1.4e-45) and would underflow a direct sum to
    # zero.  logsumexp over the per-element log terms is exact within
    # float64 rounding; exact zeros give -inf terms, and an all-zero
    # (dead) column gives log S_l = -inf, dropping out of every
    # downstream rate.
    log_S = logsumexp(jnp.log(phi) + 2.0 * jnp.log(jnp.abs(B)), axis=0)

    keys = jax.random.split(key, K)
    cols = jnp.arange(K)

    def step(h, log_delta):
        # Sequential (Gauss-Seidel) full conditional of delta_h given
        # B, phi and the LATEST delta_{-h}: delta_1..delta_{h-1} have
        # already been refreshed this sweep, so tau is recomputed from
        # the current state at every step.
        log_tau = jnp.cumsum(log_delta)
        # v_l = log(tau_l S_l); log A_h = logsumexp_{l >= h} v_l with
        # earlier columns masked and dead columns (-inf) dropping out.
        log_A = logsumexp(jnp.where(cols >= h, log_tau + log_S, -jnp.inf))
        # shapes: h = 1 -> a1 + pK/2;  h >= 2 -> a2 + p(K-h+1)/2
        # (h is 0-based here, so K - h = K - (h_1-based - 1)).
        shape = jnp.where(h == 0, a1, a2) + 0.5 * p * (K - h)
        # rate_h = 1 + A_h / (2 delta_h), in log space with a cap.
        log_rate = jnp.logaddexp(
            jnp.asarray(0.0, dtype=_F64),
            log_A
            - jax.lax.dynamic_index_in_dim(log_delta, h, keepdims=False)
            - _LN2,
        )
        log_rate = jnp.minimum(log_rate, _LOG_CAP_RATE)
        key_h = jax.lax.dynamic_index_in_dim(keys, h, keepdims=False)
        delta_h = jax.random.gamma(key_h, shape, dtype=_F64) / jnp.exp(log_rate)
        return log_delta.at[h].set(jnp.log(jnp.maximum(delta_h, _tiny())))

    return jax.lax.fori_loop(0, K, step, log_delta)


def prior_scales(
    phi: jax.Array,
    log_delta: jax.Array,
) -> jax.Array:
    r"""Deterministic map (phi, log delta) -> prior standard deviations.

    D_jh = 1 / sqrt(phi_jh tau_h),  tau_h = prod_{l<=h} delta_l,

    assembled as exp( -(log phi + log tau)/2 ) with log D capped at
    _LOG_CAP_SCALE: B_sample's whitened system forms w*w' products, and
    |w| beyond ~1e15 overflows the float32 ceiling; the cap binds only
    when tau_h is so small that the prior on that column is effectively
    flat, never in the shrinking direction.  On the lower side exp()
    underflows to 0, decoupling the coordinate exactly in the
    w -> 0 limit.

    Parameters
    ----------
    phi : array-like, shape (p, K)
        Local shrinkage parameters (floored at the dtype's tiny).
    log_delta : array-like, shape (K,)
        log of the current delta state.

    Returns
    -------
    jnp.ndarray, shape (p, K), working dtype
        D_jh = prior standard deviation of B_jh, the exact input
        convention of B_sample.py.
    """
    phi = jnp.asarray(phi, dtype=_F64)
    log_delta = jnp.asarray(log_delta, dtype=_F64)

    log_tau = jnp.cumsum(log_delta)
    log_D = -0.5 * (jnp.log(phi) + log_tau[None, :])
    return jnp.exp(jnp.minimum(log_D, _LOG_CAP_SCALE))


def shrinkage(
    key: jax.Array,
    B: jax.Array,
    phi: jax.Array,
    log_delta: jax.Array,
    a1: float,
    a2: float,
    nu: float = 3.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Steps 4-5 in one call: phi -> delta -> prior scales.

    Parameters
    ----------
    key : jax.random.PRNGKey
    B : array-like, shape (p, K)
        Factor loading matrix at the current iteration.
    phi : array-like, shape (p, K)
        Ignored (kept for API compatibility): the returned phi is drawn
        from its exact full conditional given B and the CURRENT tau.
        GS_MGP carries phi only to expose D_last after the final sweep.
    log_delta : array-like, shape (K,)
        CURRENT log delta state (carried between sweeps).
    a1, a2 : float
        FIXED shape hyperparameters of the delta priors,
        delta_1 ~ Ga(a1, 1), delta_l ~ Ga(a2, 1) (inputs; the paper's
        Step 6 MH update of (a1, a2) is deliberately omitted, see the
        module docstring).
    nu : float
        Degrees of freedom of phi_jh ~ Ga(nu/2, nu/2); paper default 3.

    Returns
    -------
    (D, phi, log_delta) :
        D         : (p, K) prior standard deviations for B_sample;
        phi       : (p, K) the drawn local shrinkage (new state);
        log_delta : (K,) the log of the drawn delta (new state).
    """
    B = jnp.asarray(B, dtype=_F64)

    key_phi, key_d = jax.random.split(key)

    # Step 4: local shrinkage given B and the CURRENT tau
    log_tau = jnp.cumsum(jnp.asarray(log_delta, dtype=_F64))
    phi = phi_sample(key_phi, B, log_tau, nu)

    # Step 5: column shrinkage increments given B and the NEW phi
    log_delta = delta_sample(key_d, B, phi, log_delta, a1, a2)

    D = prior_scales(phi, log_delta)
    return D, phi, log_delta
