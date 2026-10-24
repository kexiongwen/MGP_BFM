"""
Sigma2_sample.py  (JAX, working precision)

Idiosyncratic-variance step of the Gibbs sampler (Step 2 of Section 3.1
in Bhattacharya & Dunson, 2011, "Sparse Bayesian infinite factor
models", Biometrika 98, 291-306): sample the idiosyncratic error
precisions sigma_j^{-2}, j = 1, ..., p, independently from their gamma
full conditional

    sigma_j^{-2} | X_{.j}, B_j, eta
        ~ Ga( a_sigma + n/2,
              b_sigma + (1/2) * || X_{.j} - eta^T B_j ||^2 ).

NOTE the parameterization: the MGP prior of the paper (their eq. (3))
puts a GAMMA prior on the PRECISION,
sigma_j^{-2} ~ Ga(a_sigma, b_sigma) (the paper's Section 4.1 defaults
are a_sigma = 1, b_sigma = 0.3).  An inverse-gamma prior on the variance
would induce the same sampling code through the c/g identity below, but
the interpretation of (a_sigma, b_sigma) would differ -- do not swap
hyperparameter values blindly.

Conventions (shared with the whole pipeline):
    B      : (p, K) loading matrix (as returned by B_sample.py).
    eta    : (K, n) latent factor matrix, column i is eta_i (as returned
             by eta_sample.py), so that eta^T B_j is row j of B @ eta.
    return : (p,) vector of VARIANCES sigma_j^2, the diagonal of Omega
             (the orientation eta_sample / B_sample consume).

Sampling scheme: reciprocal-gamma identity.  With residuals
RSS_j = || X_{.j} - eta^T B_j ||^2 and g_j ~ Ga(a_sigma + n/2, 1)
(shape-rate), independent across j,

    precision_j = g_j / (b_sigma + RSS_j / 2)  ~  Ga(a_sigma + n/2, b_sigma + RSS_j/2),
    sigma_j^2   = (b_sigma + RSS_j/2) / g_j,

since c / g ~ IG-like reciprocal: if g ~ Ga(alpha, 1) then c/g has
density proportional to g^{-(alpha+1)} e^{-c/g}, i.e. c/g is the
reciprocal of a Ga(alpha, c) draw, which is IG(alpha, c) -- and its
reciprocal is exactly the gamma precision above.  The residual sum of
squares is formed directly (no ||X||^2 - 2 B^T eta X + ... identity),
avoiding catastrophic cancellation when the fit is good; b_sigma > 0
keeps every draw strictly positive.

JAX-specific notes
------------------
* Working precision (float32 by default; switchable process-wide via
  ``mgp.set_precision`` or per estimator through its ``precision``
  parameter): array inputs are coerced to the working dtype on entry.
  Takes an explicit jax.random.PRNGKey as its first argument.
  (The design comments reference the float32 magnitudes the machinery
  was originally designed for; the guards remain active and correct in
  float64 as well.)
* Exact Gamma via jax.random.gamma (a pure XLA primitive); no
  chi-square approximation is used anywhere in this package -- exact
  Gamma needs no accuracy-for-speed tradeoff here (see Shrinkage.py).
"""

import jax
import jax.numpy as jnp

__all__ = ["sigma2_sample"]

_F64 = jnp.float64  # package working precision (x64 enabled at import)


def sigma2_sample(
    key: jax.Array,
    X: jax.Array,
    B: jax.Array,
    eta: jax.Array,
    a_sigma: float,
    b_sigma: float,
) -> jax.Array:
    r"""One exact draw of the diagonal variances sigma_j^2, j = 1..p.

    Parameters
    ----------
    key : jax.random.PRNGKey
    X : array-like, shape (n, p)
        Data matrix.
    B : array-like, shape (p, K)
        Loading matrix at the current iteration (output of B_sample()).
    eta : array-like, shape (K, n)
        Latent factors, column i is eta_i (output of eta_sample()).
    a_sigma, b_sigma : float
        Fixed hyperparameters of the gamma prior on the precision,
        sigma_j^{-2} ~ Ga(a_sigma, b_sigma), both > 0.  Paper defaults:
        a_sigma = 1, b_sigma = 0.3.

    Returns
    -------
    jnp.ndarray, shape (p,)
        The drawn variances sigma_j^2 (diagonal of Omega) in the working
        dtype (float32 by default), independent across j and strictly
        positive.
    """
    X = jnp.asarray(X, dtype=_F64)
    B = jnp.asarray(B, dtype=_F64)
    eta = jnp.asarray(eta, dtype=_F64)
    n = X.shape[0]

    # RSS_j = || X_{.j} - eta^T B_j ||^2, row j of the (p, n) residual
    rss = jnp.sum((X.T - B @ eta) ** 2, axis=1)

    rate = b_sigma + 0.5 * rss

    # precision_j = g_j / rate_j,  g_j ~ Ga(a_sigma + n/2, 1) iid across j
    # (exact Gamma; the downstream eta/B steps floor sigma_j^2 themselves)
    g = jax.random.gamma(key, a_sigma + 0.5 * n, shape=rate.shape,
                         dtype=_F64)
    return rate / g
