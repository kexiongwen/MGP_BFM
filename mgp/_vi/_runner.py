"""Compiled execution layer: pure sweep, chunked while_loop runners.

Three execution levels:

1. :func:`sweep` -- the pure CAVI sweep in the order of MGP_CAVI.md
   Section 3.1: B (Component 2) -> eta (1) -> sigma2 (5) -> omega (3)
   -> delta (4) -> ELBO + device-side diagnostics.  Never jitted on
   its own; it is inlined into the runners below.
2. ``sweep_jit`` -- single-sweep JIT kept for interactive debugging
   and fixed-count cross-backend comparisons.  Not exported and not
   used by the driver, which only runs the chunked runner.
3. :func:`make_run_chunk` -- a compiled lax.while_loop advancing the
   state until the chunk is full, the global max_iter is reached,
   convergence is detected, or a numerical failure is flagged.

Why this order (Section 3.1/3.2): sigma2 is the last update among
Components 1, 2 and 5, so beta_j = b_sigma + R_j / 2 holds with the
Gaussian moments current at the end of every sweep, and the subsequent
omega / delta updates only change quantities that do not enter R_j.
The compact ELBO (3.3.7) is therefore exact at every sweep boundary
and the recorded sequence is non-decreasing up to floating-point jitter.

Device-side diagnostics: each sweep checks the factorization validity
flags (B and eta), the finiteness of the state and of the ELBO
variable part, and that the ELBO did not drop by more than the jitter
budget.  A failing sweep does NOT overwrite the last valid state and
is NOT counted in the chunk's ELBO history; it only records (error,
error_t, error_module), which the host turns into an exception at the
chunk boundary.

The chunk runner writes the variable-part ELBO of each successful
sweep into a fixed-length device buffer with a dynamic count, so tail
chunks reuse the same compiled executable (no recompilation).  The
constant is added back on the host for the reported history.
"""

import jax
import jax.numpy as jnp

from ._state import (
    ERR_ELBO_DROP,
    ERR_FACTORIZATION,
    ERR_NONE,
    ERR_NONFINITE,
    Carry,
    Inputs,
)
from .B import update_B
from .eta import second_moment, update_eta
from .shrinkage import log_mean_delta, log_prior_precision, update_shrinkage
from .sigma2 import update_sigma2

__all__ = ["sweep", "make_run_chunk", "GRAPH_COMPILER_OPTIONS"]

# XLA:GPU command-buffer (CUDA graph) capture of the whole while_loop
# chunk: with CONDITIONAL+WHILE enabled, the entire chunk executable is
# recorded as ONE CUDA graph on first execution and replayed afterwards
# (GPU only, passed per-compile so graph mode is a per-call choice).
GRAPH_COMPILER_OPTIONS = {
    "xla_gpu_enable_command_buffer":
        "FUSION,CUBLAS,CUBLASLT,CUDNN,CUSTOM_CALL,CONDITIONAL,WHILE",
    "xla_gpu_graph_min_graph_size": 1,
}


def sweep(carry: Carry, inp: Inputs) -> Carry:
    """One full MGP CAVI sweep (B -> eta -> sigma2 -> omega -> delta).

    Same block order as MGP_CAVI.md Section 3.1: sigma2 comes last
    among Components 1, 2 and 5, and the coupled delta updates (applied
    sequentially inside update_shrinkage, Component 4's note) come
    last, so the evaluated variable part is exact at the current state
    (non-decreasing up to floating-point jitter).

    On numerical failure the returned carry keeps the OLD variational
    state (the failing candidate does not overwrite it) and only the
    error fields are set; t and prev_elbo stay at the last valid
    sweep.  The while_loop's cond then stops the chunk.
    """
    c = inp.alpha_n / carry.beta

    # ---- Component 2 (B), first: uses the previous sweep's W, Phi ----
    # and the shrinkage moments carried as their Gamma rates.
    S_prev = second_moment(carry.W, carry.Phi)
    _, log_mbar = log_mean_delta(carry.beta_delta, inp.alpha_delta)
    log_prec = log_prior_precision(carry.beta_omega, log_mbar, inp.alpha_omega)
    WtX = carry.W.T @ inp.X  # computed once per sweep, used by the B update
    mu, V, elbo_B, ok_B = update_B(
        inp.X, carry.W, carry.Phi, c, log_prec, WtX=WtX, S=S_prev
    )

    # ---- Component 1 (eta) ----
    W, Phi, S, elbo_eta, ok_eta = update_eta(inp.X, mu, V, c)

    # ---- Component 5 (sigma2): last among 1, 2, 5, so that
    # beta_j = b_sigma + R_j / 2 holds with the current moments ----
    _, beta, elbo_sigma2 = update_sigma2(
        inp.X, W, Phi, mu, V, inp.a_sigma, inp.b_sigma, S=S
    )

    # ---- Components 3-4 (omega, then the sequential delta pass) ----
    beta_omega, rho, beta_delta, _, elbo_shrink = update_shrinkage(
        mu, V, carry.beta_delta, inp.nu, inp.alpha_delta, inp.alpha_omega
    )

    elbo_var = elbo_B + elbo_eta + elbo_sigma2 + elbo_shrink
    delta = elbo_var - carry.prev_elbo

    # ---- device-side diagnostics -------------------------------------
    chol_ok = ok_B & ok_eta
    fin_ok = (
        jnp.isfinite(rho).all()
        & jnp.isfinite(beta_omega).all()
        & jnp.isfinite(beta_delta).all()
        & jnp.isfinite(beta).all()
        & jnp.isfinite(elbo_var)
    )
    drop_ok = delta >= -inp.drop_budget
    ok = chol_ok & fin_ok & drop_ok

    error = jnp.where(
        chol_ok,
        jnp.where(fin_ok, jnp.where(drop_ok, ERR_NONE, ERR_ELBO_DROP), ERR_NONFINITE),
        ERR_FACTORIZATION,
    ).astype(jnp.int32)
    omega_bad = ~(jnp.isfinite(rho).all() & jnp.isfinite(beta_omega).all())
    delta_bad = ~jnp.isfinite(beta_delta).all()
    sigma2_bad = ~jnp.isfinite(beta).all()
    fin_module = jnp.where(
        omega_bad, 4, jnp.where(delta_bad, 5, jnp.where(sigma2_bad, 3, 6))
    )
    module = (
        jnp.where(~ok_B, 1, 0)
        + jnp.where(ok_B & ~ok_eta, 2, 0)
        + jnp.where(chol_ok & ~fin_ok, fin_module, 0)
        + jnp.where(chol_ok & fin_ok & ~drop_ok, 6, 0)
    ).astype(jnp.int32)

    new_t = carry.t + jnp.int32(1)
    converged = jnp.abs(delta) < inp.tol

    # On failure keep the OLD variational state; set only error fields.
    def pick(new, old):
        return jnp.where(ok, new, old)

    return Carry(
        mu=pick(mu, carry.mu),
        V=pick(V, carry.V),
        beta_omega=pick(beta_omega, carry.beta_omega),
        beta_delta=pick(beta_delta, carry.beta_delta),
        W=pick(W, carry.W),
        Phi=pick(Phi, carry.Phi),
        beta=pick(beta, carry.beta),
        prev_elbo=pick(elbo_var, carry.prev_elbo),
        t=pick(new_t, carry.t),
        converged=pick(converged, carry.converged),
        error=jnp.where(ok, carry.error, error),
        error_t=jnp.where(ok, carry.error_t, new_t),
        error_module=jnp.where(ok, carry.error_module, module),
    )


# Single-sweep JIT for debugging and fixed-count cross-backend checks.
sweep_jit = jax.jit(sweep)


def _run_chunk_impl(carry: Carry, buf: jax.Array, inp: Inputs):
    """Advance over one chunk of sweeps (compiled lax.while_loop).

    Stops at the first of: chunk full (``i == len(buf)``), a numerical
    error flag, convergence, or the global max_iter.  Only successful
    sweeps are written into ``buf`` (their variable-part ELBO) and
    counted; the returned count is the valid prefix length.
    """
    chunk_len = buf.shape[0]

    def cond(cs):
        c, b, i = cs
        return (
            (i < chunk_len)
            & (c.error == ERR_NONE)
            & (~c.converged)
            & (c.t < inp.max_iter)
        )

    def body(cs):
        c, b, i = cs
        c = sweep(c, inp)
        good = c.error == ERR_NONE
        b = b.at[i].set(jnp.where(good, c.prev_elbo, 0.0))
        i = i + good.astype(jnp.int32)
        return c, b, i

    return jax.lax.while_loop(cond, body, (carry, buf, jnp.int32(0)))


def make_run_chunk(graph: bool = False):
    """Return the compiled chunk runner (plain JIT or XLA command buffers).

    ``graph=True`` requests XLA command buffers (CUDA graphs): the
    whole while_loop chunk is recorded as one CUDA graph and replayed.
    It is a per-compile choice for GPU backends; callers must only pass
    graph=True when jax.default_backend() == "gpu" (the driver warns
    and falls back otherwise).
    """
    if graph:
        return jax.jit(_run_chunk_impl, compiler_options=GRAPH_COMPILER_OPTIONS)
    return jax.jit(_run_chunk_impl)
