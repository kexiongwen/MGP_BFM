"""
GS_MGP.py  (JAX port, working precision)

Gibbs sampler driver for the Bayesian infinite factor model with the
multiplicative gamma process (MGP) shrinkage prior (Bhattacharya &
Dunson, 2011, "Sparse Bayesian infinite factor models", Biometrika 98,
291-306, Section 3.1, fixed truncation level).

Orchestrates one full sweep of the sampler by calling, in order (each
step is an exact full conditional):

    1. eta_sample    : eta_i | B, Omega, x_i            (i = 1..n)   [Step 3]
    2. shrinkage     : phi | B, tau  [Step 4];  delta | B, phi  [Step 5]
                       -> D = 1/sqrt(phi tau)
    3. sigma2_sample : sigma_j^{-2} | X_{.j}, B_j, eta  (j = 1..p)  [Step 2]
    4. B_sample      : B_j | X_{.j}, eta, D, sigma_j^2  (j = 1..p)  [Step 1]

Model:
    x_i = B eta_i + eps_i,
    eta_i ~ N_K(0, I_K),
    eps_i ~ N_p(0, Omega),  Omega = diag(sigma_1^2, ..., sigma_p^2).

Prior (paper eq. (3)):
    B_jh | phi_jh, tau_h ~ N(0, (phi_jh tau_h)^{-1}),
    phi_jh ~ Ga(nu/2, nu/2),  tau_h = prod_{l<=h} delta_l,
    delta_1 ~ Ga(a1, 1),  delta_l ~ Ga(a2, 1),
    sigma_j^{-2} ~ Ga(a_sigma, b_sigma).

The paper's Step 6 (a Metropolis-Hastings update of a1, a2 under Ga(2,1)
hyperpriors) is deliberately NOT implemented: a1 and a2 are fixed input
hyperparameters.  The exact conditionals pi(a1, a2 | delta) are proper,
but at large p and overfitted K their normalizing constants (products of
Gamma functions raised to K-1) overflow floating point -- for p ~ 1000
and K = 50 the constant is of order 10^7105 -- so an MH chain is
numerically degenerate; fixing (a1, a2) is a deliberate simplification
(see Shrinkage.py).

Chain structure: the Markov state carried between sweeps is
(B, sigma2, phi, log_delta).  The latent factors eta are regenerated
every sweep from their exact full conditional given (B, sigma2) and are
not sampler state; the shrinkage block (phi, delta) IS state, because
the phi- and delta-conditionals depend on each other (unlike shrinkage
priors whose block regenerates in one marginal-then-conditional pass).  delta is carried as log delta -- tau = prod delta grows
exponentially in the column index and overflows float32 if carried
directly (see Shrinkage.py).

Scope: FIXED truncation level K (an overfitted K is enough -- redundant
columns are shrunk toward zero by the prior, and the paper's Theorem 1
bounds the truncation error).  The adaptive choice of the number of
factors (paper Section 3.2) is NOT implemented; the effective number of
factors is read off the per-entry marginal credible intervals of B (see
mgp.base.BaseMGP.n_factors; this is not the paper's Section 3.2 rule).

JAX-specific notes
------------------
* The modules compute in the process-wide working precision (float32 by
  default, float64 optional; see mgp._precision): X is coerced on entry
  and every module derives its dtype from the mgp._precision constants.
  jax's x64 mode is enabled by the top-level mgp import (required for
  the float64 regime, and left enabled afterwards).
* Randomness is explicit: `seed` seeds one jax.random.PRNGKey that is
  split into one key per sweep up front; the same seed reproduces the
  same chain.
* The chain is executed as jax.lax.scan over the sweep keys, in chunks
  of at most _CHUNK sweeps: each chunk is ONE compiled XLA While loop
  (the full 4-step sweep inside), with no Python dispatch or host
  synchronization inside.  Between chunks the draws are copied to the
  host (bounded device memory, progress reports, finiteness audit).
  Two compiled runners are used: a burn-in runner that advances the
  state without collecting draws, and a collecting runner that returns
  only the KEPT draws of the chunk (the last sweep of every `thin`-sweep
  block; keep-phase chunks are multiples of `thin`, except the final
  remainder).  Thinning therefore costs no extra device memory or host
  traffic, and a device-side finiteness flag preserves the per-sweep
  audit.  Compilation is cached per (chunk length, thin, shapes) across
  GS_MGP calls and is done ahead of time (lower().compile()), so no
  compilation is counted in runtime_sec.  With graph=True (GPU only),
  the runners are compiled with XLA command buffers, so each chunk is
  additionally recorded as ONE CUDA graph and replayed.
* All Gamma draws are EXACT (jax.random.gamma is a pure XLA primitive,
  capturable by CUDA graphs on every backend); no chi-square
  approximation is offered -- exact Gamma needs no accuracy-for-speed
  tradeoff here (see Shrinkage.py).
* Posterior draws are stored on the host and returned as numpy arrays.

Implementation: JAX-based, functional style.
"""

import os
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from .B_sample import B_sample
from .eta_sample import eta_sample
from .Shrinkage import prior_scales, shrinkage
from .Sigma2_sample import sigma2_sample

__all__ = ["GS_MGP", "Initialization"]

_F64 = jnp.float64  # working precision of the whole package (x64 enabled at import)

# Sweeps per lax.scan chunk (shrunk adaptively for large p*K, see
# _chunk_sizes / _keep_chunk_sizes): gives one progress report and one
# finiteness audit per chunk.
_CHUNK = 500
# Cap on device-side collected draws per chunk, in bytes (the burn runner
# collects nothing; the keep runner collects ceil(chunk / thin) states).
_CHUNK_BYTES = 256 * 2 ** 20


def Initialization(X: jax.Array, r: int) -> tuple[jax.Array, float]:
    r"""Principal-factor initialization for (B, sigma2).

    Eigen-decomposition of the sample covariance (X is assumed
    column-centered, as the model has zero mean): with
    M = X.T / sqrt(n - 1), M M^T = X^T X / (n - 1) = Sigma_hat, and
    eigenvalues s_i^2 of Sigma_hat (descending),

        sigma2_hat = mean of the trailing eigenvalues {s_i^2}_{i > r},
        B0         = U_r diag( sqrt(s_1^2 - sigma2_hat, ...) ),

    so that B0 B0^T = U_r diag(s_i^2 - sigma2_hat) U_r^T is the rank-r
    PSD approximation of Sigma_hat - sigma2_hat I (the factor model
    decomposes Sigma = B B^T + sigma2 I at initialization).

    Implementation notes:
    * the top-r eigenpairs come from one jnp.linalg.eigh of the p x p
      covariance -- deterministic; runs ONCE per chain;
    * the trailing eigenvalues are summed DIRECTLY from the ascending
      eigh spectrum (every eigenvalue is already computed, so no
      identity is needed): the trace identity
      sum_{i > r} s_i^2 = tr(Sigma_hat) - sum_{i <= r} s_i^2
      is exact only in exact arithmetic and cancels catastrophically in
      float32 when the noise floor is small relative to the leading
      spectrum (the strong-signal regime; the same pattern as the _vi
      PPCA fix of 2026-10-22, here propagated to this initializer);
    * the denominator is min(n - 1, p) - r: column-centered X has rank
      at most min(n - 1, p), so this is exactly the number of trailing
      eigenvalues that can be nonzero;
    * s_i^2 - sigma2_hat is floored at 0: top-r components below the
      noise floor initialize to a zero column instead of NaN.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Column-centered data matrix.
    r : int
        Number of leading factors, 1 <= r <= min(n, p) - 2.

    Returns
    -------
    B0 : jnp.ndarray, shape (p, r), float64
        Initial loading matrix.
    sigma2_hat : float
        Noise-floor variance estimate (scalar; broadcast to (p,) by the
        caller for the initial Omega diagonal).
    """
    X = jnp.asarray(X, dtype=_F64)
    n, p = X.shape
    if not 1 <= r <= min(n, p) - 2:
        raise ValueError(
            f"r must be in [1, min(n, p) - 2] = [1, {min(n, p) - 2}], got {r}"
        )

    M = X.T / jnp.sqrt(jnp.asarray(n - 1, dtype=_F64))  # (p, n)
    C = M @ M.T                                          # sample cov (p, p)
    evals, evecs = jnp.linalg.eigh(C)                    # ascending order

    # top-r eigenpairs, descending
    s2 = evals[::-1][:r]
    U = evecs[:, ::-1][:, :r]

    # mean trailing eigenvalue, summed directly from the ascending
    # spectrum (eigh returns every eigenvalue; the trace identity
    # tr(C) - sum(top r) cancels catastrophically in float32 when the
    # trailing sum is small relative to the leading spectrum).
    # Exactly min(n - 1, p) - r trailing eigenvalues can be nonzero for
    # column-centered X.
    tail = evals[: p - r].sum()
    sigma2_hat = jnp.maximum(tail / (min(n - 1, p) - r), 0.0)

    B0 = U * jnp.sqrt(jnp.maximum(s2 - sigma2_hat, 0.0))[None, :]
    return B0, float(sigma2_hat)


def _sweep(state, key, X, nu, a1, a2, a_sigma, b_sigma):
    """One full Gibbs sweep: eta -> shrinkage -> sigma2 -> B
    (paper steps 3 -> 4,5 -> 2 -> 1).

    Scalar hyperparameters are traced, so changing them does not
    trigger recompilation.  This stays UN-jitted: it is compiled inline
    into the scan runners below.
    """
    B, sigma2, phi, log_delta = state
    k_eta, k_sh, k_s2, k_B = jax.random.split(key, 4)
    # 1. latent factors (paper step 3; fresh draw from the exact
    #    conditional given (B, sigma2), not carried state)
    eta = eta_sample(k_eta, X, B, sigma2)
    # 2. shrinkage parameters (paper steps 4, 5): phi, delta;
    #    D = prior standard deviations for the B step
    D, phi, log_delta = shrinkage(k_sh, B, phi, log_delta, a1, a2, nu)
    # 3. idiosyncratic variances (paper step 2, gamma on the precision)
    sigma2 = sigma2_sample(k_s2, X, B, eta, a_sigma, b_sigma)
    # 4. loading matrix (paper step 1, block draw of all p rows)
    B = B_sample(k_B, X, D, eta, sigma2)
    return (B, sigma2, phi, log_delta), (B, sigma2)


def _run_burn_impl(state, keys, X, nu, a1, a2, a_sigma, b_sigma):
    """Advance the chain over a chunk of burn-in keys, collecting
    nothing.  Returns only the final state."""
    def f(state, k):
        state, _ = _sweep(state, k, X, nu, a1, a2, a_sigma, b_sigma)
        return state, None

    state, _ = jax.lax.scan(f, state, keys)
    return state


def _run_keep_impl(state, keys, X, nu, a1, a2, a_sigma, b_sigma, thin):
    """Advance the chain over a keep-phase chunk, collecting only the kept
    sweeps.

    Keep-phase chunks are multiples of `thin` sweeps long (except the final
    remainder, whose trailing sweeps contain no kept draw), and every chunk
    starts on a keep boundary ((post-burn-in sweep index + 1) % thin == 0 is
    the keep rule).  The kept sweeps of a chunk are therefore exactly the
    last sweep of each `thin`-sweep block: only those states leave the
    device, so thinning costs no extra device memory and no extra host
    traffic.  A running finiteness flag preserves the per-sweep audit that
    the old collect-everything runner performed on the collected chunk.
    """

    def step(carry, k):
        st, ok = carry
        st_new, (B, s2) = _sweep(st, k, X, nu, a1, a2, a_sigma, b_sigma)
        ok = ok & jnp.isfinite(B).all() & jnp.isfinite(s2).all()
        return (st_new, ok), None

    def block(carry, kb):
        # One `thin`-sweep block; collect the state of its last (kept) sweep.
        (st, ok), _ = jax.lax.scan(step, carry, kb)
        return (st, ok), (st[0], st[1])

    c = keys.shape[0]
    nblocks = c // thin
    ntail = c - nblocks * thin
    ok = jnp.asarray(True)
    if nblocks:
        keys_b = keys[: nblocks * thin].reshape((nblocks, thin) + keys.shape[1:])
        (state, ok), (Bk, s2k) = jax.lax.scan(block, (state, ok), keys_b)
    else:
        Bk = jnp.zeros((0, state[0].shape[0], state[0].shape[1]),
                       dtype=state[0].dtype)
        s2k = jnp.zeros((0, state[1].shape[0]), dtype=state[1].dtype)
    if ntail:
        (state, ok), _ = jax.lax.scan(
            step, (state, ok), keys[nblocks * thin:]
        )
    return state, (Bk, s2k), ok


# XLA:GPU command-buffer (CUDA graph) capture of the whole scan While
# loop: with CONDITIONAL+WHILE enabled, the entire chunk executable is
# recorded as ONE CUDA graph on first execution and replayed afterwards,
# removing all per-kernel launch overhead between sweeps.  Every op in
# the sweep body (GPU RNG, cuBLAS/cuSOLVER custom calls, the
# associative_scan logaddexp reduction) is capturable, and the draws
# are bitwise-reproducible for a fixed seed.  Passed per-compile so
# graph mode is a per-call choice (no XLA_FLAGS needed before importing
# jax).
_GRAPH_COMPILER_OPTIONS = {
    "xla_gpu_enable_command_buffer":
        "FUSION,CUBLAS,CUBLASLT,CUDNN,CUSTOM_CALL,CONDITIONAL,WHILE",
    "xla_gpu_graph_min_graph_size": 1,
}

_run_burn = jax.jit(_run_burn_impl)
_run_keep = jax.jit(_run_keep_impl, static_argnames=("thin",))
_run_burn_graph = jax.jit(_run_burn_impl,
                          compiler_options=_GRAPH_COMPILER_OPTIONS)
_run_keep_graph = jax.jit(_run_keep_impl, static_argnames=("thin",),
                          compiler_options=_GRAPH_COMPILER_OPTIONS)


def _chunk_sizes(total: int, p: int, K: int) -> list[int]:
    """Split `total` sweeps into chunk lengths bounded by _CHUNK and by
    the per-chunk device memory cap _CHUNK_BYTES (the burn-in runner
    collects nothing)."""
    itemsize = 4 if _F64 == jnp.float32 else 8
    per_sweep = (p * K + p) * itemsize
    chunk = max(1, min(_CHUNK, _CHUNK_BYTES // max(per_sweep, 1)))
    sizes, left = [], total
    while left > 0:
        sizes.append(min(chunk, left))
        left -= sizes[-1]
    return sizes


def _keep_chunk_sizes(total: int, p: int, K: int, thin: int) -> list[int]:
    """Split the keep phase into chunk lengths that are multiples of `thin`
    (except a final remainder), so every chunk starts on a keep boundary and
    the collecting runner can return exactly the kept sweeps.

    The device-side buffer per chunk holds ceil(chunk / thin) states, so the
    _CHUNK_BYTES cap applies to the collected draws and `thin` proportionally
    increases the sweeps per chunk.  The final chunk may be shorter; its
    trailing sweeps contain no kept draw, so it needs no alignment.
    """
    itemsize = 4 if _F64 == jnp.float32 else 8
    per_kept = (p * K + p) * itemsize
    cap_kept = max(1, _CHUNK_BYTES // max(per_kept, 1))
    raw = min(_CHUNK, cap_kept * thin)
    step = max(thin, (raw // thin) * thin)
    sizes, left = [], total
    while left > 0:
        sizes.append(min(step, left))
        left -= sizes[-1]
    return sizes


def _audit_host(arrs: tuple[np.ndarray, ...], upto: int,
                device_ok: bool = True) -> None:
    """Finiteness audit of one host-side chunk.

    `device_ok` carries the device-side per-sweep finiteness flag of the
    collecting runner (it audits every sweep, not just the kept draws).
    All sampled systems are I + PSD by construction and the shrinkage
    block is assembled in log space, so a non-finite state signals a
    numerical anomaly OUTSIDE the guarded range; fail loudly rather
    than silently contaminating the chain.
    """
    if not bool(device_ok) or not all(np.isfinite(a).all() for a in arrs):
        raise FloatingPointError(
            f"non-finite state (B or sigma2) detected in the "
            f"chunk ending at sweep {upto}; the chain is numerically "
            "contaminated and must be discarded. This should not happen "
            "(all systems are I + PSD and the shrinkage state is "
            "log-space by construction) -- please report the data and "
            "hyperparameters."
        )


def GS_MGP(
    X: jax.Array,
    K: int,
    nu: float = 3.0,
    a1: float = 2.0,
    a2: float = 2.0,
    a_sigma: float = 1.0,
    b_sigma: float = 0.3,
    n_iter: int = 2000,
    burn_in: int = 1000,
    thin: int = 1,
    seed: int | None = None,
    verbose: bool = True,
    graph: bool = False,
) -> dict:
    r"""Gibbs sampler for the Bayesian factor model with the MGP
    shrinkage prior (Bhattacharya & Dunson, 2011, Section 3.1, fixed
    truncation).

    Runs n_iter sweeps on the truncated model with K columns.  Each
    sweep updates, in order (all exact full conditionals; cf. the
    paper's Section 3.1):

        1. eta    ~ pi(eta | B, sigma2, X)              [eta_sample]
        2. phi    ~ pi(phi | B, tau)                    [Shrinkage step 4]
           delta  ~ pi(delta | B, phi)                  [Shrinkage step 5]
        3. sigma2 ~ pi(sigma2 | B, eta, X)              [sigma2_sample]
        4. B      ~ pi(B | eta, sigma2, phi, tau, X)    [B_sample]

    Parameters
    ----------
    X : array-like, shape (n, p)
        Column-centered data matrix, coerced to the working precision
        (float32 by default).
    K : int
        Truncation level of the loading matrix (number of columns);
        must be an integer >= 1 (integral floats are coerced).  An
        overfitted K is enough: redundant columns are shrunk toward
        zero by the increasing shrinkage of the MGP prior.
    nu : float
        Degrees of freedom of the local shrinkage prior
        phi_jh ~ Ga(nu/2, nu/2).  Paper default nu = 3; the paper's
        theory (Theorem 1) assumes nu > 2, and 0 < nu <= 2 triggers a
        RuntimeWarning (the sampler remains valid).
    a1, a2 : float
        FIXED shape hyperparameters of the column shrinkage priors,
        delta_1 ~ Ga(a1, 1), delta_l ~ Ga(a2, 1); both > 0.  The paper
        (Section 4.1) instead places Ga(2, 1) hyperpriors on them and
        updates them by Metropolis-Hastings (their Step 6) -- omitted
        here by design (see the module docstring); the defaults equal
        the Ga(2, 1) prior means.  The tau_h are stochastically
        increasing under a2 > 1 (a RuntimeWarning fires otherwise) and
        the paper's Theorem 1 truncation bound additionally wants
        a2 > 2.
    a_sigma, b_sigma : float
        Hyperparameters of the gamma prior on the idiosyncratic
        PRECISION, sigma_j^{-2} ~ Ga(a_sigma, b_sigma); both > 0.
        Paper Section 4.1 defaults: a_sigma = 1, b_sigma = 0.3.
    n_iter, burn_in, thin : int
        Total sweeps, discarded initial sweeps, and keep-every-thin.
    seed : int or None
        Seed for jax.random.PRNGKey.  None draws a random seed.  The
        same seed (on the same backend) reproduces the same chain,
        including eta_last / D_last.
    verbose : bool
        Print progress about every 10% of the sweeps (reports are
        emitted at chunk boundaries, so small runs print less often).
    graph : bool
        If True and the backend is GPU, compile the scan runners with
        XLA command buffers (CUDA graphs): the whole chunk -- all its
        sweeps, RNG included -- is recorded as one CUDA graph on first
        execution and replayed afterwards.  The math is unchanged and a
        fixed seed still reproduces the chain.  Requires a GPU backend;
        on CPU/TPU a RuntimeWarning is issued and the run proceeds with
        the plain runners.

    Returns
    -------
    dict with keys
        "B"       : (S, p, K) kept loading matrices on the host in the
                    working precision, S = (n_iter - burn_in) // thin;
        "sigma2"  : (S, p) kept variance vectors, likewise;
        "eta_last": (K, n) latent factors drawn from the FINAL state
                    (one extra partial sweep; for diagnostics or warm
                    starts);
        "D_last"  : (p, K) prior scales 1/sqrt(phi tau) of the final
                    state (deterministic in the final shrinkage state);
        "seed"    : the seed actually used;
        "runtime_sec" : wall-clock seconds of the scan loop (excludes
                    the ahead-of-time JIT compilation).

    Initialization: principal-factor initialization (see Initialization)
    with r = min(K, min(n, p) - 2) leading factors; remaining columns of
    B start at zero, every sigma_j^2 starts at the noise-floor estimate,
    phi starts at 1 (the prior mean of Ga(nu/2, nu/2)), and log delta
    starts at log(a1, a2, ..., a2).
    Memory note: the B chain costs S*p*K*itemsize(working dtype) bytes on
    the host -- use thinning for very large p.
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    key = jax.random.PRNGKey(seed)

    # working precision throughout: every module derives its dtype from
    # the shared _F64 constant (flipped by mgp._precision.set_precision)
    X = jnp.asarray(X, dtype=_F64)
    if not bool(np.isfinite(np.asarray(X)).all()):
        raise ValueError(
            "X must contain only finite values (found NaN or inf); note that "
            "finite float64 inputs can overflow to inf in float32 working "
            "precision"
        )

    # ---- hyperparameter validation ----------------------------------------
    # integral floats (K = 10.0) are coerced; a fractional K would be
    # silently truncated by the jnp.arange below (K = 2.5 -> 3 columns)
    if K != int(K) or K < 1:
        raise ValueError(f"K must be an integer >= 1, got {K}")
    K = int(K)
    for _name, _val in (("n_iter", n_iter), ("burn_in", burn_in), ("thin", thin)):
        if isinstance(_val, bool) or not isinstance(_val, (int, np.integer)):
            raise ValueError(f"{_name} must be an integer, got {_val!r}")
    n_iter, burn_in, thin = int(n_iter), int(burn_in), int(thin)
    for name, val in (("nu", nu), ("a1", a1), ("a2", a2),
                      ("a_sigma", a_sigma), ("b_sigma", b_sigma)):
        if not val > 0:  # also rejects NaN
            raise ValueError(f"{name} must be > 0, got {name} = {val}")
    if nu <= 2:
        warnings.warn(
            f"nu = {nu} violates the condition nu > 2 used by the "
            "paper's theory (Theorem 1 is stated for nu = 3 and any "
            "nu > 2): the asymptotic truncation bound need not apply, "
            "although the sampler on the K-truncated model remains "
            "valid",
            RuntimeWarning,
        )
    if a2 <= 1:
        warnings.warn(
            f"a2 = {a2} violates the condition a2 > 1 under which the "
            "tau_h are stochastically increasing (paper Section 2.1); "
            "the increasing shrinkage that suppresses redundant columns "
            "is lost, although the sampler remains valid",
            RuntimeWarning,
        )

    if not 0 <= burn_in < n_iter:
        raise ValueError(f"need 0 <= burn_in < n_iter, got {burn_in}/{n_iter}")
    if thin < 1:
        raise ValueError(f"thin must be >= 1, got {thin}")
    if (n_iter - burn_in) // thin == 0:
        raise ValueError(
            f"no draws would be kept: (n_iter - burn_in) // thin = "
            f"({n_iter} - {burn_in}) // {thin} = 0; reduce thin or burn_in"
        )

    n, p = X.shape
    if min(n, p) < 3:
        raise ValueError(
            f"need min(n, p) >= 3 for initialization, got (n, p) = ({n}, {p})"
        )

    # ---- initialize ---------------------------------------------------------
    r = min(K, min(n, p) - 2)
    B0, s2_0 = Initialization(X, r)
    B = jnp.zeros((p, K), dtype=_F64).at[:, :r].set(B0)
    sigma2 = jnp.full((p,), s2_0, dtype=_F64)
    phi = jnp.ones((p, K), dtype=_F64)
    log_delta = jnp.log(jnp.concatenate([
        jnp.asarray([a1], dtype=_F64),
        jnp.full((K - 1,), a2, dtype=_F64),
    ]))

    n_keep = (n_iter - burn_in) // thin

    args = (X, nu, a1, a2, a_sigma, b_sigma)

    # one key per sweep, split up front: same seed -> same chain.  An
    # (n_iter + 1)-th child is reserved for the eta_last partial sweep
    # at the end of the run; children of one split are distinct by
    # construction, so it never collides with an in-chain sweep key.
    keys_all = jax.random.split(key, n_iter + 1)
    keys, key_extra = keys_all[:n_iter], keys_all[n_iter]

    # ---- runner selection (graph mode = XLA command buffers on GPU) ---------
    use_graph = graph and jax.default_backend() == "gpu"
    if graph and not use_graph:
        warnings.warn(
            f"graph=True requires a GPU backend (got "
            f"'{jax.default_backend()}'); running with the plain runners",
            RuntimeWarning,
        )
    run_burn = _run_burn_graph if use_graph else _run_burn
    run_keep = _run_keep_graph if use_graph else _run_keep

    # ---- ahead-of-time compilation (not timed) ------------------------------
    # One compilation per DISTINCT chunk length per runner (only shapes
    # are traced, so which key slice feeds lower() is irrelevant).
    state0 = (B, sigma2, phi, log_delta)
    burn_sizes = _chunk_sizes(burn_in, p, K)
    keep_sizes = _keep_chunk_sizes(n_iter - burn_in, p, K, thin)
    start = 0
    for c in set(burn_sizes):
        run_burn.lower(state0, keys[start:start + c], *args).compile()
        start += c
    start = burn_in
    for c in set(keep_sizes):
        run_keep.lower(state0, keys[start:start + c], *args,
                       thin=thin).compile()
        start += c

    # ---- run ------------------------------------------------------------------
    state = state0
    host_dtype = np.float32 if _F64 == jnp.float32 else np.float64
    B_keep = np.empty((n_keep, p, K), dtype=host_dtype)   # stored on host
    s2_keep = np.empty((n_keep, p), dtype=host_dtype)

    t_start = time.perf_counter()
    report = max(1, n_iter // 10)
    last_report = 0

    # burn-in: advance without collecting (audit the carried state per chunk)
    start = 0
    for c in burn_sizes:
        state = run_burn(state, keys[start:start + c], *args)
        start += c
        _audit_host((np.asarray(state[0]), np.asarray(state[1])), start)
        if verbose and (start - last_report >= report or start == n_iter):
            print(f"GS_MGP: sweep {start}/{n_iter}", flush=True)
            last_report = start

    # kept phase: the collecting runner returns only the kept draws of each
    # chunk (the last sweep of every `thin`-sweep block), so thinning costs
    # no extra device memory or host traffic.
    kept = 0
    for c in keep_sizes:
        state, (Bc, s2c), chunk_ok = run_keep(
            state, keys[start:start + c], *args, thin=thin
        )
        Bc, s2c = np.asarray(Bc), np.asarray(s2c)
        _audit_host((Bc, s2c), start + c, device_ok=chunk_ok)
        take = Bc.shape[0]
        B_keep[kept:kept + take] = Bc
        s2_keep[kept:kept + take] = s2c
        kept += take
        start += c
        if verbose and (start - last_report >= report or start == n_iter):
            print(f"GS_MGP: sweep {start}/{n_iter}", flush=True)
            last_report = start

    runtime = time.perf_counter() - t_start
    if kept != n_keep:
        raise RuntimeError(
            f"kept {kept} draws but expected {n_keep}: the thinning "
            "bookkeeping is inconsistent -- please report this bug"
        )

    # eta_last / D_last: one extra partial draw from the final state
    # (eta from its exact conditional; D is deterministic in the final
    # shrinkage state).  key_extra is the reserved (n_iter + 1)-th child
    # of the up-front split, so these draws are independent of every
    # sweep key and the whole run stays reproducible from `seed` alone.
    eta_last = eta_sample(key_extra, X, state[0], state[1])
    D_last = prior_scales(state[2], state[3])

    return {
        "B": B_keep,
        "sigma2": s2_keep,
        "eta_last": np.asarray(eta_last),
        "D_last": np.asarray(D_last),
        "seed": seed,
        "runtime_sec": runtime,
    }
