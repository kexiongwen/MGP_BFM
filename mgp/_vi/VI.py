"""Coordinate ascent variational inference (CAVI) driver -- JAX, working precision.

Implements the single-study CAVI algorithm of MGP_CAVI.md: Bayesian
factor analysis with the multiplicative gamma process (MGP) shrinkage
prior, mean-field family and optimal variational posteriors of Section
3.1, closed-form ELBO of Section 3.3.

One sweep updates, in the order of Section 3.1, q(B) (Component 2),
q(eta) (1), q(sigma2) (5), q(omega) (3), and the coupled q(delta)
factors (4, applied sequentially inside update_shrinkage), then
evaluates the variable part of the closed-form ELBO; the constant is
added back on the host for the reported history.  This block order is
the note's: sigma2 is the last update among Components 1, 2 and 5, so
beta_j = b_sigma + R_j / 2 holds with the Gaussian moments current at
the end of every sweep, and the subsequent omega / delta updates only
change quantities that do not enter R_j -- the compact ELBO (3.3.7) is
exact at every sweep boundary and the recorded sequence is
non-decreasing.

The modules compute in the working precision selected through
:mod:`mgp._precision`: the :class:`mgp.CaviMGP` estimator defaults to
float64 and applies it for the duration of ``fit`` (direct calls to this
driver use the process-wide default, initially float32); X is coerced on
entry and every module derives its dtype from the working-precision
constants.  The stopping criterion uses the
VARIABLE PART of the ELBO only (delta_t = L_var_t - L_var_{t-1},
|delta_t| < tol), which keeps the large invariant constant out of the
floating-point difference -- the main cancellation defence.  With
float64 the ELBO resolution is about |L| * 1e-16, so rounding jitter
no longer binds at practical tolerances; the default tol of 1e-1 and
the independent absolute drop_budget of 0.5 are deliberate choices for
speed and behavioural continuity.

Execution: the pure sweep is inlined into a compiled lax.while_loop
chunk runner; chunks sync with the host only at their boundaries (ELBO
history transfer, audits, progress).  AOT compilation
(lower().compile()) happens before the timed loop, so compile time is
never counted in runtime_sec.  graph=True on a GPU backend compiles
the same runner with XLA command buffers (CUDA graphs); elsewhere it
warns and falls back.  jit=False runs the sweep eagerly for debugging.

Device-side numerical diagnostics: each sweep checks the factorization
validity flags, the finiteness of the state and of the ELBO variable
part, and that the ELBO did not drop by more than the jitter budget.
A failing sweep keeps the last valid state, records (error, sweep,
module), and the host raises at the chunk boundary -- failures are
never returned as successful results.
"""

import math
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from ._linalg import cholesky_checked, logdet_from_chol
from ._runner import make_run_chunk, sweep
from ._state import (
    ERR_ELBO_DROP,
    ERR_FACTORIZATION,
    ERR_NONFINITE,
    MODULE_NAMES,
    Carry,
    Inputs,
)
from .B import log_det_V
from .elbo import elbo, elbo_const
from .eta import second_moment
from .initialization import PPCA, initialization
from .shrinkage import (
    initial_rates,
    log_mean_delta,
    shapes,
    update_shrinkage,
)
from .sigma2 import mean_inverse, update_sigma2

__all__ = ["CAVI", "PPCA", "elbo", "elbo_const", "plot_elbo", "initialization"]

_F64 = jnp.float64  # package working precision (x64 enabled at import)

_ERROR_MESSAGES = {
    ERR_NONFINITE: "non-finite state / ELBO",
    ERR_FACTORIZATION: "Cholesky factorization or inversion failed",
    ERR_ELBO_DROP: "ELBO dropped by more than the jitter budget (numerical anomaly)",
}


def _validate_args(
    X: jax.Array,
    K: int,
    nu: float,
    a1: float,
    a2: float,
    a_sigma: float,
    b_sigma: float,
    tol: float | None,
    max_iter: int,
    v0: float,
    chunk_size: int,
    audit_every: int,
) -> None:
    """Validate the CAVI arguments (host wrapper layer).

    Raises
    ------
    ValueError
        On non-2-D or non-finite data, non-positive hyperparameters,
        or invalid tol / max_iter / K / chunk_size / audit_every.
    UserWarning
        If the hyperparameters violate the moment conditions
        (nu > 2 for a finite E[omega^{-1}], a1, a2 > 1 for
        a finite E[T_k^{-1}]; the algorithm still runs).
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {tuple(X.shape)}")
    if not bool(np.isfinite(np.asarray(X)).all()):
        raise ValueError("X must contain only finite values (found NaN or inf)")
    if K != int(K) or K < 1:
        raise ValueError(f"K must be an integer >= 1, got {K}")
    for name, val in (
        ("nu", nu), ("a1", a1), ("a2", a2),
        ("a_sigma", a_sigma), ("b_sigma", b_sigma), ("v0", v0),
    ):
        if not (math.isfinite(val) and val > 0):
            raise ValueError(f"{name} must be a positive finite number, got {val}")
    if tol is not None and not (math.isfinite(tol) and tol > 0):
        raise ValueError(f"tol must be positive, got {tol}")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError(f"max_iter must be an integer >= 1, got {max_iter}")
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError(f"chunk_size must be an integer >= 1, got {chunk_size}")
    if not isinstance(audit_every, (int, np.integer)) or audit_every < 0:
        raise ValueError(
            f"audit_every must be an integer >= 0 (0 disables), got {audit_every}"
        )
    if nu <= 2.0 or a1 <= 1.0 or a2 <= 1.0:
        warnings.warn(
            f"nu={nu}, a1={a1}, a2={a2}: the moment conditions "
            f"(nu > 2 for a finite E[1/omega_jk], a1, a2 > 1 for a finite "
            f"E[1/T_k]) are violated; the algorithm still runs, but these "
            f"expectations do not exist and the prior interpretation "
            f"weakens",
            UserWarning,
            stacklevel=3,
        )


def _make_carry(state: dict) -> Carry:
    """Pack an initialization/warm-start dict into the loop carry."""
    return Carry(
        mu=state["mu"],
        V=state["V"],
        beta_omega=state["beta_omega"],
        beta_delta=state["beta_delta"],
        W=state["W"],
        Phi=state["Phi"],
        beta=state["beta"],
        prev_elbo=jnp.asarray(state["elbo_var"], dtype=_F64),
        t=jnp.int32(0),
        converged=jnp.bool_(False),
        error=jnp.int32(0),
        error_t=jnp.int32(0),
        error_module=jnp.int32(0),
    )


def _normalize_warm_start(
    X: jax.Array,
    initial_state: dict,
    K: int,
    nu: jax.Array,
    alpha_omega: jax.Array,
    alpha_delta: jax.Array,
    a_sigma: float,
    b_sigma: float,
) -> dict:
    """Validate and normalize an external initial state.

    Required fields: mu, V, W, Phi.  Optional field: beta_delta (K,) --
    the starting rates of the sequential delta pass below (a warm start
    from a previous run's rates); it must be positive, finite, and of
    shape (K,), and when omitted the mean-one initialization
    convention is used.  A beta_omega field is REJECTED: the
    optimal q(omega_jk) update is closed-form in (e_jk, mbar_k) and
    never reads its previous rate, so it is always re-derived from the
    Gaussian state during normalization.  After shape / finiteness /
    SPD checks, S, the shrinkage rates, the noise rates, c and the
    initial ELBO are recomputed so the state is self-consistent in the
    end-of-sweep sense (cached values are never authoritative).
    """
    n, p = X.shape
    missing = {"mu", "V", "W", "Phi"} - set(initial_state)
    if missing:
        raise ValueError(f"initial_state is missing required fields: {sorted(missing)}")
    mu = jnp.asarray(initial_state["mu"], dtype=_F64)
    V = jnp.asarray(initial_state["V"], dtype=_F64)
    W = jnp.asarray(initial_state["W"], dtype=_F64)
    Phi = jnp.asarray(initial_state["Phi"], dtype=_F64)
    if mu.shape != (p, K):
        raise ValueError(f"initial_state['mu'] must have shape {(p, K)}, got {mu.shape}")
    if V.shape != (p, K, K):
        raise ValueError(
            f"initial_state['V'] must have shape {(p, K, K)}, got {V.shape}"
        )
    if W.shape != (n, K):
        raise ValueError(f"initial_state['W'] must have shape {(n, K)}, got {W.shape}")
    if Phi.shape != (K, K):
        raise ValueError(
            f"initial_state['Phi'] must have shape {(K, K)}, got {Phi.shape}"
        )
    for name, arr in (("mu", mu), ("V", V), ("W", W), ("Phi", Phi)):
        if not bool(jnp.isfinite(arr).all()):
            raise ValueError(f"initial_state['{name}'] contains non-finite values")
    if not bool(jnp.allclose(V, jnp.swapaxes(V, -1, -2))):
        raise ValueError("initial_state['V'] must be symmetric")
    if not bool(jnp.allclose(Phi, Phi.T)):
        raise ValueError("initial_state['Phi'] must be symmetric")

    if "beta_omega" in initial_state:
        raise ValueError(
            "initial_state['beta_omega'] is not accepted: the optimal "
            "q(omega_jk) update is closed-form and does not depend on its "
            "previous rate, so it is always re-derived during normalization "
            "(pass only beta_delta)"
        )
    if "beta_delta" in initial_state:
        beta_delta0 = jnp.asarray(initial_state["beta_delta"], dtype=_F64)
        if beta_delta0.shape != (K,):
            raise ValueError(
                f"initial_state['beta_delta'] must have shape {(K,)}, "
                f"got {beta_delta0.shape}"
            )
        if not bool(jnp.isfinite(beta_delta0).all()) or not bool(
            (beta_delta0 > 0).all()
        ):
            raise ValueError(
                "initial_state['beta_delta'] must contain only positive "
                "finite values (Gamma rates)"
            )
    else:
        beta_delta0 = initial_rates(alpha_delta)

    # SPD validity of the supplied covariances (checked, no jitter).
    ldV, ok_V = log_det_V(V)
    L_phi, ok_Phi = cholesky_checked(Phi)
    if not bool(ok_V & ok_Phi):
        raise ValueError(
            "initial_state covariances are not positive definite "
            "(V and Phi must be SPD)"
        )

    S = second_moment(W, Phi)
    beta_omega, rho, beta_delta, log_mbar, elbo_shrink = update_shrinkage(
        mu, V, beta_delta0, nu, alpha_delta, alpha_omega
    )
    alpha_n, beta, elbo_sigma2 = update_sigma2(X, W, Phi, mu, V, a_sigma, b_sigma)
    c = mean_inverse(alpha_n, beta)
    elbo_eta = 0.5 * n * logdet_from_chol(L_phi) - 0.5 * jnp.trace(S)
    elbo_var = elbo_eta + 0.5 * ldV.sum() + elbo_shrink + elbo_sigma2
    # meaningful validity flag: the renormalized quantities must be finite
    ok = jnp.bool_(
        jnp.isfinite(rho).all()
        & jnp.isfinite(beta).all()
        & jnp.isfinite(beta_delta).all()
        & jnp.isfinite(log_mbar).all()
        & jnp.isfinite(elbo_var)
    )

    return {
        "mu": mu,
        "V": V,
        "W": W,
        "Phi": Phi,
        "S": S,
        "alpha_n": alpha_n,
        "beta": beta,
        "c": c,
        "beta_omega": beta_omega,
        "rho": rho,
        "beta_delta": beta_delta,
        "log_mbar": log_mbar,
        "elbo_var": elbo_var,
        "ok": ok,
    }


def _audit_host(state_arrays: dict, upto: int) -> None:
    """Full finiteness audit of the state at a chunk boundary (host).

    Covers mu, V, W, Phi, beta, beta_omega, beta_delta, rho and c.  A
    non-finite state signals a numerical anomaly outside the guarded
    range; fail loudly.
    """
    bad = [k for k, a in state_arrays.items() if not np.isfinite(np.asarray(a)).all()]
    if bad:
        raise FloatingPointError(
            f"non-finite variational state ({', '.join(bad)}) detected in "
            f"the chunk ending at sweep {upto}; the run is numerically "
            f"contaminated and must be discarded. This should not happen "
            f"(all factorized systems are I + PSD by construction and the "
            f"device-side flags did not fire) -- please report the data "
            f"and hyperparameters."
        )


def CAVI(
    X: jax.Array,
    K: int,
    nu: float = 3.0,
    a1: float = 2.1,
    a2: float = 3.1,
    a_sigma: float = 1.0,
    b_sigma: float = 0.3,
    tol: float | None = 1e-1,
    max_iter: int = 1000,
    r: int | None = None,
    v0: float = 1.0,
    center: bool = True,
    initial_state: dict | None = None,
    jit: bool = True,
    chunk_size: int = 50,
    graph: bool = False,
    audit_every: int = 50,
    drop_budget: float | None = 0.5,
    verbose: bool = False,
    plot: bool = False,
) -> dict:
    """Coordinate ascent variational inference for the MGP factor model.

    Parameters
    ----------
    X : jax.Array, shape (n, p)
        Data matrix.  The model has zero mean, so columns are centered
        by default (``center=True``).  Coerced to the working precision
        (see :mod:`mgp._precision`; :class:`mgp.CaviMGP` defaults to
        float64).
    K : int
        Truncation level (computational upper bound on the number of
        factors; no smaller than the number of active factors, which
        can be thresholded post hoc from the shrunken columns).
    nu, a1, a2 : float
        MGP hyperparameters: nu for the local omega_jk (prior
        Gamma(nu/2, nu/2)), a1 for delta_1, a2 for delta_l (l >= 2).
        Defaults (Durante 2017): nu=3, a1=2.1, a2=3.1 (1 < a1 < a2, so
        E[T_k] = a1 a2^{k-1} increases and shrinkage strengthens with
        the column index).
    a_sigma, b_sigma : float
        Inverse-gamma hyperparameters of the noise variances
        (paper Section 4.1 defaults: 1 and 0.3).
    tol : float or None, default 1e-1
        Absolute ELBO (variable part) stopping tolerance.  After each
        complete sweep, stop when abs(Lvar_t - Lvar_prev) < tol, where
        Lvar excludes the invariant ELBO constant.  Must be positive
        and finite; None also resolves to 1e-1.  The float32 ELBO
        resolution is of the order of |ELBO_var| * 1e-7; tolerances
        near that scale can respond to rounding jitter.
    max_iter : int, default 1000
        Maximum number of CAVI sweeps.
    r : int or None, default None
        Number of leading factors used by the PPCA initialization;
        defaults to min(K, min(n, p) - 2).  Values larger than K are
        clipped to K.  Ignored with ``initial_state``.
    v0 : float, default 1.0
        Initial loading covariance scale, V_j = v0 * I_K.  Ignored
        with ``initial_state``.
    center : bool, default True
        Subtract the column means of X before running.
    initial_state : dict or None, default None
        Warm start: must contain mu (p, K), V (p, K, K), W (n, K),
        Phi (K, K); optionally beta_delta (K,) -- positive finite Gamma
        rates used as the starting point of the sequential delta pass.
        (A beta_omega field is rejected: q(omega)'s optimal update is
        closed-form and never reads its previous rate.)  Validated
        (shape, finiteness, SPD) and normalized into a self-consistent
        state.
    jit : bool, default True
        Compile the sweep into chunked lax.while_loop runners.  False
        runs the sweep eagerly, one host-visible step at a time (for
        debugging and fixed-count comparisons).
    chunk_size : int, default 50
        Sweeps per compiled chunk: bounds the host sync frequency
        (ELBO history transfer, audits, progress).  Does NOT change
        the stopping point -- convergence is checked on the device
        after every sweep.
    graph : bool, default False
        On a GPU backend, compile the chunk runner with XLA command
        buffers (CUDA graphs).  Elsewhere a RuntimeWarning is issued
        and the run proceeds with the plain runner.
    audit_every : int, default 50
        Every this many sweeps (checked at chunk boundaries), verify
        on the host that the whole state is finite.  0 disables the
        periodic audit; the mandatory init / final audits and the
        per-sweep device-side checks are unaffected.
    drop_budget : float or None, default 0.5
        Maximum permitted absolute ELBO decrease per sweep, independent
        of tol.  A larger decrease raises FloatingPointError,
        even if it satisfies the stopping threshold.  Must be finite
        and nonnegative; 0 requires non-decreasing ELBO.  None also
        resolves to the fixed default 0.5.
    verbose : bool, default False
        Print the ELBO at chunk boundaries; warn if it decreases
        beyond the jitter budget (a numerical issue).
    plot : bool, default False
        After the loop ends, plot the ELBO trajectory with
        :func:`plot_elbo` (requires matplotlib).

    Raises
    ------
    ValueError
        On invalid data, hyperparameters, or run parameters, or on an
        invalid ``initial_state``.
    UserWarning
        If the hyperparameters violate the moment conditions
        (the algorithm still runs).
    FloatingPointError
        On a device-side numerical failure (non-finite state, failed
        factorization, ELBO drop beyond the jitter budget) or a failed
        host audit; the message carries the sweep and module.

    Returns
    -------
    dict with keys
        mu : (p, K) variational means of the loading rows,
        V : (p, K, K) variational covariances of the loading rows,
        W : (n, K) variational means of the factor scores,
        Phi : (K, K) common covariance of the factor scores,
        alpha_n : float, noise inverse-gamma shape a_sigma + n/2,
        beta : (p,) noise inverse-gamma rates,
        c : (p,) E_q[1 / sigma_j^2] = alpha_n / beta_j,
        beta_omega : (p, K) rates of q(omega_jk),
        rho : (p, K) local shrinkage means E_q[omega_jk],
        beta_delta : (K,) rates of q(delta_l),
        m_delta : (K,) global increment means E_q[delta_l],
        log_T : (K,) log E_q[T_k] (log of the cumulative product;
        materialized E_q[T_k] can exceed the float32 range at large
        p, K),
        T : (K,) numpy float64 array, E_q[T_k] = prod_l m_l,
        elbo : list of float, absolute ELBO after initialization and
        each sweep (constant included; length n_iter + 1),
        n_iter : int, number of sweeps performed,
        converged : bool, whether the ELBO tolerance was reached,
        stop_reason : "tol" or "max_iter",
        execution_mode : "while_loop", "command_buffer" or "eager",
        backend, device : execution environment summary,
        compile_sec : AOT compilation wall time (0 in eager mode),
        runtime_sec : sweep-loop wall time (excludes compilation).
    """
    _validate_args(X, K, nu, a1, a2, a_sigma, b_sigma, tol, max_iter, v0,
                   chunk_size, audit_every)
    if tol is None:
        tol = 1e-1
    tol_work = jnp.asarray(tol, dtype=_F64)
    if float(tol_work) == 0.0:
        raise ValueError(
            f"tol={tol!r} underflows the working precision; use a larger "
            "absolute tolerance or precision='float64'"
        )
    if drop_budget is None:
        drop_budget = 0.5
    if not (math.isfinite(drop_budget) and drop_budget >= 0):
        raise ValueError(
            f"drop_budget must be a nonnegative finite number, got {drop_budget}"
        )

    X = jnp.asarray(X, dtype=_F64)
    n, p = X.shape
    if center:
        X = X - X.mean(axis=0)
    K = int(K)

    alpha_omega, alpha_delta, _ = shapes(p, K, nu, a1, a2)
    nu_j = jnp.asarray(nu, dtype=_F64)
    alpha_omega_j = jnp.asarray(alpha_omega, dtype=_F64)
    const = elbo_const(n, p, K, nu, a1, a2, a_sigma, b_sigma)

    # --- initialization ------------------------------------------------
    if initial_state is None:
        state = initialization(
            X, K, nu_j, alpha_omega_j, alpha_delta,
            a_sigma, b_sigma, r=r, v0=v0,
        )
    else:
        state = _normalize_warm_start(
            X, initial_state, K, nu_j, alpha_omega_j, alpha_delta,
            a_sigma, b_sigma,
        )
    if not bool(state["ok"]):
        raise FloatingPointError(
            "initialization produced a numerically invalid state "
            "(factorization flags fired); please report the data."
        )
    # mandatory init audit (covers states that never reach a periodic point)
    _audit_host(
        {"mu": state["mu"], "V": state["V"], "W": state["W"], "Phi": state["Phi"],
         "beta": state["beta"], "beta_omega": state["beta_omega"],
         "beta_delta": state["beta_delta"], "rho": state["rho"],
         "c": state["c"]},
        0,
    )

    carry = _make_carry(state)
    inp = Inputs(
        X=X,
        alpha_omega=alpha_omega_j,
        alpha_delta=alpha_delta,
        nu=nu_j,
        alpha_n=jnp.asarray(a_sigma + n / 2.0, dtype=_F64),
        a_sigma=jnp.asarray(a_sigma, dtype=_F64),
        b_sigma=jnp.asarray(b_sigma, dtype=_F64),
        tol=jnp.asarray(tol, dtype=_F64),
        drop_budget=jnp.asarray(drop_budget, dtype=_F64),
        max_iter=jnp.int32(max_iter),
    )

    history = [const + float(state["elbo_var"])]
    if verbose:
        print(f"[CAVI] init      ELBO = {history[0]:.6f}")

    # --- runner selection ----------------------------------------------
    use_graph = graph and jax.default_backend() == "gpu"
    if graph and not use_graph:
        warnings.warn(
            f"graph=True requires a GPU backend (got "
            f"'{jax.default_backend()}'); running with the plain runner",
            RuntimeWarning,
        )
    execution_mode = "command_buffer" if use_graph else ("while_loop" if jit else "eager")

    def _raise_on_error(c: Carry) -> None:
        err = int(c.error)
        if err != 0:
            msg = _ERROR_MESSAGES.get(err, f"unknown numerical error {err}")
            mod = MODULE_NAMES.get(int(c.error_module), "?")
            raise FloatingPointError(
                f"{msg} at sweep {int(c.error_t)} (module: {mod}); the run "
                f"is numerically contaminated and must be discarded."
            )

    def _state_arrays(c: Carry) -> dict:
        return {"mu": c.mu, "V": c.V, "W": c.W, "Phi": c.Phi, "beta": c.beta,
                "beta_omega": c.beta_omega, "beta_delta": c.beta_delta,
                "rho": inp.alpha_omega / c.beta_omega, "c": inp.alpha_n / c.beta}

    # --- CAVI sweeps ----------------------------------------------------
    compile_sec = 0.0
    last_audit = 0
    if not jit:
        # eager debug path: one host-visible sweep at a time
        t_start = time.perf_counter()
        while (
            int(carry.t) < max_iter
            and not bool(carry.converged)
            and int(carry.error) == 0
        ):
            carry = sweep(carry, inp)
            _raise_on_error(carry)
            history.append(const + float(carry.prev_elbo))
            if verbose:
                v, prev = history[-1], history[-2]
                print(f"[CAVI] sweep {len(history) - 1:4d}  ELBO = {v:.6f}  "
                      f"d = {v - prev:.3e}")
                if v < prev - drop_budget:
                    print("[CAVI] warning: ELBO decreased (numerical issue)")
            if audit_every and int(carry.t) - last_audit >= audit_every:
                _audit_host(_state_arrays(carry), int(carry.t))
                last_audit = int(carry.t)
        runtime_sec = time.perf_counter() - t_start
    else:
        run_chunk = make_run_chunk(graph=use_graph)
        buf = jnp.zeros((chunk_size,), dtype=_F64)
        # AOT compilation (not timed): one executable, reused by every
        # chunk including the shorter tail (dynamic count).
        t_compile = time.perf_counter()
        exe = run_chunk.lower(carry, buf, inp).compile()
        compile_sec = time.perf_counter() - t_compile

        t_start = time.perf_counter()
        while (
            int(carry.t) < max_iter
            and not bool(carry.converged)
            and int(carry.error) == 0
        ):
            carry, buf, used = exe(carry, buf, inp)
            used = int(used)
            _raise_on_error(carry)
            if used:
                vals = np.asarray(buf[:used]).astype(np.float64) + const
                base = len(history) - 1
                history.extend(vals.tolist())
                if verbose:
                    for k, v in enumerate(vals):
                        prev = history[base + k]
                        print(f"[CAVI] sweep {base + k + 1:4d}  ELBO = {v:.6f}  "
                              f"d = {v - prev:.3e}")
                        if v < prev - drop_budget:
                            print("[CAVI] warning: ELBO decreased (numerical issue)")
            if audit_every and int(carry.t) - last_audit >= audit_every:
                _audit_host(_state_arrays(carry), int(carry.t))
                last_audit = int(carry.t)
        runtime_sec = time.perf_counter() - t_start

    stop_reason = "tol" if bool(carry.converged) else "max_iter"

    # mandatory final audit (covers runs that end before any periodic point)
    _audit_host(_state_arrays(carry), int(carry.t))

    c = mean_inverse(inp.alpha_n, carry.beta)
    rho = inp.alpha_omega / carry.beta_omega
    m_delta = inp.alpha_delta / carry.beta_delta
    _, log_T = log_mean_delta(carry.beta_delta, inp.alpha_delta)
    out = {
        "mu": carry.mu,
        "V": carry.V,
        "W": carry.W,
        "Phi": carry.Phi,
        "alpha_n": float(inp.alpha_n),
        "beta": carry.beta,
        "c": c,
        "beta_omega": carry.beta_omega,
        "rho": rho,
        "beta_delta": carry.beta_delta,
        "m_delta": m_delta,
        "log_T": log_T,
        "T": np.exp(np.asarray(log_T, dtype=np.float64)),
        "elbo": history,
        "n_iter": int(carry.t),
        "converged": bool(carry.converged),
        "stop_reason": stop_reason,
        "execution_mode": execution_mode,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "compile_sec": compile_sec,
        "runtime_sec": runtime_sec,
    }

    if plot:
        plot_elbo(history, show=True)
    return out


def plot_elbo(history, ax=None, logx=False, show=False, save=None, title=None):
    """Plot the ELBO trajectory of a CAVI run against the iteration count.

    Iteration 0 is the ELBO after initialization; iteration t is the
    ELBO after sweep t.  matplotlib is imported lazily, so VI.py does
    not depend on it unless this function (or CAVI with ``plot=True``)
    is used.

    Parameters
    ----------
    history : dict or sequence of float
        The dict returned by :func:`CAVI` (its ``"elbo"`` entry is
        used) or the ELBO sequence itself.
    ax : matplotlib.axes.Axes, optional
    logx : bool, default False
        Logarithmic iteration axis; sweep 0 is dropped in that case.
    show : bool, default False
    save : str or Path, optional
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the history is empty.
    ImportError
        If matplotlib is not installed.
    """
    if isinstance(history, dict):
        history = history["elbo"]
    history = list(history)
    if len(history) == 0:
        raise ValueError("empty ELBO history")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_elbo requires matplotlib (e.g. pip install matplotlib)"
        ) from exc

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    xs = list(range(len(history)))
    if logx:
        xs, history = xs[1:], history[1:]
        ax.set_xscale("log")
    ax.plot(xs, history, lw=1.5)
    ax.set_xlabel("sweep" + (" (log scale)" if logx else ""))
    ax.set_ylabel("ELBO")
    ax.set_title(
        title
        or f"CAVI ELBO trajectory (final {history[-1]:.6g}, "
           f"{len(history) - (0 if logx else 1)} sweeps shown)"
    )
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
