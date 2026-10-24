"""
B_sample.py  (JAX, working precision)

Loading-matrix step of the Gibbs sampler (Step 1 of Section 3.1 in
Bhattacharya & Dunson, 2011, "Sparse Bayesian infinite factor models",
Biometrika 98, 291-306): sample each row B_j (j = 1, ..., p) of the
factor loading matrix B = Lambda (p x K) independently from its Gaussian
full conditional

    B_j | X_{.j}, eta, D_j, sigma_j^2
        ~ N_K(Lam_j^{-1} b_j, Lam_j^{-1}),

    Lam_j = eta eta^T / sigma_j^2 + D_j^{-1},   D_j^{-1} = diag(phi_j1 tau_1, ..., phi_jK tau_K),
    b_j   = eta X_{.j} / sigma_j^2.

The multiplicative gamma process prior enters ONLY through the diagonal
prior precision D_j^{-1}; writing w_jk = 1/sqrt(phi_jk tau_k) for the
prior standard deviation of B_jk (so D_j^{-1} = diag(w_j)^{-2}), this
step is mathematically identical to the loading step under any
global-local normal prior, and the whitened perturb-and-solve scheme
below is the one shared with the whole pipeline.

Conventions (shared with the whole pipeline):
    eta    : (K, n) latent factor matrix, column i is eta_i (as returned
             by eta_sample.py; matches the paper's eta eta^T, eta X_{.j}
             notation in this step).
    D      : (p, K) matrix with entries w_jk = 1/sqrt(phi_jk tau_k), the
             prior standard deviations, i.e. EXACTLY the output of
             Shrinkage.prior_scales().  The prior precision enters as
             diag(w_j)^{-2} = diag(phi_jk tau_k).
    sigma2 : (p,) diagonal variances of Omega, as produced by
             Sigma2_sample.py.
    return : (p, K) matrix, row j is the drawn B_j.

Sampling scheme: whitened perturb-and-solve, batched over all p rows
(no Python loop).  Write W_j = diag(w_j) and G = eta eta^T.  G is formed
ONCE (O(n K^2)) and shared by all rows: the whitened system

    M_j = (w_j w_j^T) . (G / sigma_j^2) + I_K = W_j Lam_j W_j

is built by pure broadcasting -- O(p K^2) in memory and flops, versus
O(p K n) and O(p K^2 n) for the direct per-row outer products C_j C_j^T
with C_j = W_j eta / sigma_j.  The perturbed right-hand side

    phi_j = W_j eta (X_{.j} + sigma_j z_j) / sigma_j^2 + eps_j,
    z_j ~ N(0, I_n), eps_j ~ N(0, I_K),

has noise covariance (w_j w_j^T) . (G/sigma_j^2) + I_K = M_j (the term
W_j eta z_j / sigma_j is C_j z_j), and data and noise contributions are
folded into a single eta @ (X + Z^T sigma) matmul.  The draw is mapped
back by B_j = w_j . gamma_j with gamma_j = M_j^{-1} phi_j.  Exactness:

    mean: W M^{-1} W b = W (W Lam W)^{-1} W b = Lam^{-1} b,
    cov : W M^{-1} M M^{-1} W = W (W Lam W)^{-1} W = Lam^{-1}.

Numerical merits:
    * the raw precision Lam_j has diagonal entries phi_jk tau_k =
      1/w_jk^2 spanning the whole single-precision spectrum on heavily
      shrunk coordinates (tau_k = prod delta_l grows exponentially in
      k, up to overflow in direct float32 arithmetic -- the regime the
      pipeline was designed for; float64 postpones but does not remove
      the issue, and the whitened form is used at every working
      precision); whitening turns those into near-zero data terms in
      M_j = I + PSD;
    * w_jk -> 0 decouples coordinate k (M_j[k,k] -> 1, gamma_k ~ N(0,1),
      B_jk = w_jk gamma_k -> 0): exact zeros in D cannot produce inf/NaN;
    * M_j = I + PSD by construction, so the batched Cholesky needs no
      fallback branch.

JAX-specific notes
------------------
* Working precision (float32 by default; switchable process-wide via
  ``mgp.set_precision`` or per estimator through its ``precision``
  parameter): every array input is coerced to the working dtype on
  entry.  Takes an explicit jax.random.PRNGKey as its first argument.
  (The design comments reference the float32 magnitudes the machinery
  was originally designed for; the guards remain active and correct in
  float64 as well.)
* jnp.linalg.cholesky and jax.scipy.linalg.cho_solve both handle the
  leading batch dimension p natively (the rhs is passed with a trailing
  singleton axis, phi[..., None], the non-deprecated batched form).
* The whitened system below avoids forming the prior precision matrix
  explicitly; D holds the prior standard deviations
  w_jk = 1/sqrt(phi_jk tau_k).
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

__all__ = ["B_sample"]

_F64 = jnp.float64  # package working precision (x64 enabled at import)


def B_sample(
    key: jax.Array,
    X: jax.Array,
    D: jax.Array,
    eta: jax.Array,
    sigma2: jax.Array,
) -> jax.Array:
    r"""One exact draw of the loading matrix B given (eta, D, sigma2).

    Parameters
    ----------
    key : jax.random.PRNGKey
    X : array-like, shape (n, p)
        Data matrix.
    D : array-like, shape (p, K)
        Prior standard deviations w_jk = 1/sqrt(phi_jk tau_k), i.e. the
        output of Shrinkage.prior_scales().  Must be nonnegative
        (guaranteed by the producer); entries are clipped from above at
        exp(34.5) ~ 9.6e14 by the producer, and exact zeros decouple
        the coordinate gracefully (see module docstring).
    eta : array-like, shape (K, n)
        Latent factors, column i is eta_i (output of eta_sample()).
    sigma2 : array-like, shape (p,)
        Diagonal variances sigma_j^2 of Omega (output of
        Sigma2_sample.py).  Floored at tiny**0.5 before use (sigma_j >=
        tiny**0.25), so a perfect fit cannot overflow the divisions;
        the perturbation is invisible (binds only for sigma_j^2 <
        tiny**0.5, i.e. ~1.1e-19 in float32 / ~1.5e-154 in float64).

    Returns
    -------
    jnp.ndarray, shape (p, K)
        The drawn loading matrix in the working dtype (float32 by
        default); rows are independent across j.
    """
    X = jnp.asarray(X, dtype=_F64)
    D = jnp.asarray(D, dtype=_F64)
    eta = jnp.asarray(eta, dtype=_F64)
    sigma2 = jnp.asarray(sigma2, dtype=_F64)

    K, n = eta.shape
    p = X.shape[1]

    s2 = jnp.maximum(sigma2, jnp.finfo(_F64).tiny ** 0.5)

    # ---- whitened system M_j = (w_j w_j^T) . (G / sigma_j^2) + I_K --------
    # G = eta eta^T formed once and reused by all p rows (replaces the
    # O(p K^2 n) batched outer product and O(p K n) intermediate C).
    G = eta @ eta.T                                            # (K, K)
    M = D[:, :, None] * D[:, None, :] * (G / s2[:, None, None])
    M = M + jnp.eye(K, dtype=_F64)                             # M_j += I_K

    # ---- perturbed right-hand side -----------------------------------------
    # phi_j = w_j . eta (X_{.j} + sigma_j z_j) / sigma_j^2 + eps_j; the
    # noise w_j (eta z_j)/sigma_j + eps_j has covariance M_j.
    key_z, key_e = jax.random.split(key)
    Z = jax.random.normal(key_z, (p, n), dtype=_F64)
    E = X + Z.T * jnp.sqrt(s2)[None, :]      # X_{.j} + sigma_j z_j, (n, p)
    phi = D * ((eta @ E) / s2).T + jax.random.normal(
        key_e, (p, K), dtype=_F64
    )

    # ---- solve and map back: B_j = w_j . M_j^{-1} phi_j --------------------
    # Plain Cholesky: M_j = I + PSD by construction, so no fallback
    # branch.  The driver audits finiteness of the chain per chunk.
    L = jnp.linalg.cholesky(M)                                # (p, K, K)
    gamma = jsla.cho_solve((L, True), phi[..., None]).squeeze(-1)
    return D * gamma
