"""CAVI estimator for the MGP-shrunk Bayesian factor model.

sklearn-style wrapper of the JAX coordinate-ascent variational
inference driver :func:`mgp._vi.CAVI` (the algorithm specified in
MGP_CAVI.md: mean-field family and optimal variational posteriors of
Section 3.1, closed-form ELBO of Section 3.3).  All numerical semantics
(the working precision -- this estimator defaults to **float64** (double
precision) because the absolute ELBO stopping criterion is sensitive to
float32 jitter; ``precision=None`` follows the process-wide default
instead --, the variable-part ELBO stopping criterion, device-side
diagnostics, chunked while_loop, XLA command buffers) are those of the
underlying driver; this class only adds the estimator interface and the
shared preprocessing of :class:`mgp.base.BaseMGP`.
"""

import warnings

import numpy as np

from ._posterior import cov_sigma_pairs, sigma_mean, sigma_variance
from ._vi import CAVI
from .base import BaseMGP

__all__ = ["CaviMGP"]


class CaviMGP(BaseMGP):
    """Coordinate ascent variational inference (fast posterior approximation).

    Parameters
    ----------
    K : int
        Truncation level (upper bound on the number of factors).
    nu, a1, a2 : float
        MGP hyperparameters; the defaults nu=3, a1=2.1, a2=3.1 follow
        Durante (2017) and satisfy the paper's moment conditions
        (a1 > 2, a2 > 3 for the second moments; a2 > 2 for the Theorem 1
        truncation bound).
    a_sigma, b_sigma : float
        Hyperparameters of the noise-variance prior.
    center : bool, default True
        Subtract the column means of X before fitting.
    precision : {"float32", "float64"} or None, default "float64"
        Working precision of the fit.  CAVI defaults to **double
        precision**: its stopping criterion compares absolute ELBO
        differences, and float32 rounding jitter at large |ELBO| can
        dominate that difference (and can trip the numerical-anomaly
        guard), so float32 is opt-in for speed.  ``None`` inherits the
        process-wide default (:func:`mgp.get_precision`, initially
        "float32"); an explicit value overrides the default for this
        estimator only.  The resolved value is stored in
        ``precision_used_`` and is followed by :meth:`align_factors`.
    standardize : bool, default False
        Center and divide each column by its sample standard deviation
        (stored in ``x_scale_``) before fitting.  Strongly recommended:
        the MGP prior is calibrated for unit-variance columns.  Requires
        ``center=True``.  Fitted quantities are then in the standardized
        coordinate system (see :class:`mgp.base.BaseMGP`).
    tol : float or None, default None
        Absolute ELBO (variable part) stopping tolerance.  After each
        complete sweep, stop when abs(Lvar_t - Lvar_prev) < tol, where
        Lvar excludes the invariant ELBO constant.  None resolves
        per resolved working precision: 1e-1 in float64 (the ELBO
        resolution is about |ELBO| * 1e-16, so jitter does not bind) and
        1e-3 in float32, where the recomputation jitter is about
        |ELBO| * 1e-7 and already reaches 1e-1 at |ELBO| ~ 1e5.  This
        jitter sensitivity is why the estimator defaults to float64;
        explicit float32 CAVI at very large |ELBO| can still stop early
        on noise.
    max_iter : int, default 1000
        Maximum number of CAVI sweeps.
    r : int or None, default None
        Number of leading factors used by the PPCA initialization;
        defaults to min(K, min(n, p) - 2).  Ignored with
        ``initial_state``.
    v0 : float, default 1.0
        Initial loading covariance scale, V_j = v0 * I_K.  Ignored with
        ``initial_state``.
    initial_state : dict or None, default None
        Warm start with mu (p, K), V (p, K, K), W (n, K), Phi (K, K);
        optionally beta_delta (K,).  Validated and normalized by the
        underlying driver.
    jit : bool, default True
        Compile the sweep into chunked lax.while_loop runners; False
        runs eagerly for debugging.
    chunk_size : int, default 50
        Sweeps per compiled chunk (host-sync frequency; does not change
        the stopping point).
    graph : bool, default False
        On a GPU backend, compile the chunk runner with XLA command
        buffers (CUDA graphs); elsewhere a warning is issued and the
        plain runner is used.
    audit_every : int, default 50
        Host-side finiteness audit frequency (0 disables the periodic
        audit; the mandatory init/final audits are unaffected).
    drop_budget : float or None, default None
        Maximum permitted absolute ELBO decrease per sweep, independent
        of tol; a larger decrease raises FloatingPointError.  None
        resolves to the fixed default 0.5 (0 requires a non-decreasing
        ELBO).
    verbose : bool, default False
        Print the ELBO at chunk boundaries.

    Attributes
    ----------
    mu_, V_ : (p, K) and (p, K, K) numpy arrays
        Variational means and covariances of the loading rows.
    W_ : (n, K) numpy array
        Variational means of the factor scores.
    Phi_ : (K, K) numpy array
        Common covariance of the factor scores.
    alpha_n_ : float
        Noise inverse-gamma shape a_sigma + n/2.
    beta_, c_ : (p,) numpy arrays
        Noise inverse-gamma rates and E_q[1/sigma_j^2].
    rho_, beta_omega_ : (p, K) numpy arrays
        Local shrinkage means E_q[omega_jk] and their Gamma rates.
    beta_delta_, m_delta_ : (K,) numpy arrays
        Global increment Gamma rates and means E_q[delta_l].
    log_T_ : (K,) numpy array
        log E_q[T_k] (the log-space cumulative product; the
        materialized product can exceed the float32 range at large
        p, K).
    T_ : (K,) numpy float64 array
        E_q[T_k] = prod_l m_l.
    elbo_ : list of float
        Absolute ELBO after initialization and each sweep.
    converged_, stop_reason_, n_iter_ : bool, str, int
        Stopping outcome ("tol" or "max_iter") and sweep count.
    execution_mode_, backend_, device_ : str
        Execution environment summary.
    compile_sec_, runtime_sec_ : float
        AOT compilation and sweep-loop wall times.
    B_hat_, sigma2_hat_, Sigma_ : numpy arrays
        Point estimates (see :class:`mgp.base.BaseMGP`); sigma2_hat_ is
        the inverse-gamma posterior mean beta / (alpha_n - 1).  With
        ``standardize=True`` these are on the standardized scale (use
        ``x_scale_`` to convert); alignment does not change them.
    Sigma_mean_, Sigma_var_, Sigma_sd_ : (p, p) numpy arrays
        Variational posterior moments of Sigma = B B^T + Omega, computed
        lazily from ``mu_``, ``V_``, ``beta_`` and ``alpha_n_``:
        ``Sigma_mean_ = E_q[Sigma] = Sigma_ + diag(tr V_1, ..., tr V_p)``,
        ``Sigma_var_[i, j] = Var_q(Sigma_ij)`` and ``Sigma_sd_`` its square
        root.  ``Sigma_var_``/``Sigma_sd_`` require ``alpha_n > 2`` (where
        ``alpha_n = a_sigma + n/2``).  These are moments of the mean-field
        approximation, not of the true posterior; see the module notes in
        :mod:`mgp._posterior` and :meth:`cov_Sigma` for entry-pair
        covariances.
    rsp_result_, B_aligned_
        Produced by :meth:`align_factors` from ``mu_[None, :, :]``.
        ``B_aligned_`` has shape (1, p, K); its single matrix is the
        transformed VI mean, not an MCMC draw.
    mu_aligned_, V_aligned_ : (p, K) and (p, K, K) numpy arrays
        Transformed loading means and covariances: mu R and R^T V_j R.
    W_aligned_, Phi_aligned_ : (n, K) and (K, K) numpy arrays
        Transformed score means and covariance: W R and R^T Phi R.
    eta_aligned_ : (K, n) numpy array
        Transpose view of W_aligned_: aligned score posterior MEANS,
        not random draws.  Together with mu_aligned_, preserves the
        fitted common component mu @ W.T up to numerical error.
    raw_output_ : dict
        The unmodified return value of the underlying driver (JAX
        device arrays).

    Notes
    -----
    Alignment outputs are separate from the fitted VI parameters,
    shrinkage moments (``rho_``, ``beta_omega_``, ``beta_delta_``,
    ``m_delta_``, ``log_T_``, ``T_``), and ELBO; a successful refit
    clears them all.  The column-dependent MGP shrinkage prior is not
    invariant to general rotations or permutations.  Transformed moments
    are useful for interpretation and comparison, not an equivalent CAVI
    optimum in the original parameterization.  ``n_factors()`` uses the
    original variational parameters unless ``use_aligned=True``.
    Repeated alignment starts from ``mu_``.

    ``Sigma_`` is the plug-in estimate ``mu mu^T + diag(beta/(alpha_n-1))``;
    ``Sigma_mean_`` is the variational posterior mean and therefore adds
    ``diag(tr V_j)`` on the diagonal.  The entrywise posterior variance
    ``Sigma_var_`` is guarded by a default limit of 2e7 entries (p <= 4472);
    for larger p use :meth:`cov_Sigma` on selected entry pairs.
    ``align_factors()`` does not change these original-coordinate moments
    (they are computed from the fitted variational parameters, which the
    alignment never modifies).

    Examples
    --------
    Align the VI means and covariances consistently::

        vi = CaviMGP(K=5).fit(X).align_factors()
        vi.mu_aligned_       # (p, K), also vi.B_aligned_[0]
        vi.eta_aligned_      # (K, n), posterior means
        vi.V_aligned_        # (p, K, K), loading covariances
        vi.Phi_aligned_      # (K, K), score covariance
    """

    _ALIGN_UPDATES_B_HAT = False
    _ALIGN_ATTRS = BaseMGP._ALIGN_ATTRS + (
        "mu_aligned_", "V_aligned_", "W_aligned_", "Phi_aligned_",
        "eta_aligned_",
    )

    def __init__(
        self,
        K,
        nu=3.0,
        a1=2.1,
        a2=3.1,
        a_sigma=1.0,
        b_sigma=0.3,
        center=True,
        precision="float64",
        standardize=False,
        tol=None,
        max_iter=1000,
        r=None,
        v0=1.0,
        initial_state=None,
        jit=True,
        chunk_size=50,
        graph=False,
        audit_every=50,
        drop_budget=None,
        verbose=False,
    ):
        super().__init__(
            K, nu=nu, a1=a1, a2=a2, a_sigma=a_sigma, b_sigma=b_sigma,
            center=center, precision=precision, standardize=standardize,
        )
        self.tol = tol
        self.max_iter = max_iter
        self.r = r
        self.v0 = v0
        self.initial_state = initial_state
        self.jit = jit
        self.chunk_size = chunk_size
        self.graph = graph
        self.audit_every = audit_every
        self.drop_budget = drop_budget
        self.verbose = verbose

    def _effective_tol(self):
        """Resolve tol=None per working precision.

        The absolute tolerance is compared with |Delta ELBO|; in float32
        (opt-in -- the estimator defaults to float64) the ELBO
        recomputation jitter is about |ELBO| * 1e-7, which at
        |ELBO| ~ 1e5 already reaches the float64 default 1e-1 and stops
        the iteration early at a worse optimum (measured on both demos:
        209/275 sweeps instead of 864/1379, degraded Sigma recovery).
        The float32 default 1e-3 sits below that jitter floor.
        """
        if self.tol is not None:
            return self.tol
        resolved = getattr(self, "precision_used_", None) or self.precision
        if resolved is None:
            from ._precision import get_precision

            resolved = get_precision()
        return 1e-3 if resolved == "float32" else 1e-1

    def _fit_arrays(self, Xc):
        # Posterior-moment cache (Sigma_mean_/Sigma_var_); a new fit must not
        # reuse moments of the previous fit.
        self._sigma_cache = {}
        n_obs = Xc.shape[0]
        alpha_n = self.a_sigma + n_obs / 2.0
        if not np.isfinite(alpha_n) or alpha_n <= 1.0:
            raise ValueError(
                f"alpha_n = a_sigma + n/2 = {alpha_n:.6g} must be > 1 so "
                "that the inverse-gamma posterior mean beta/(alpha_n-1) "
                "exists; use more observations, a larger a_sigma, or a "
                "warm start with n >= 2"
            )
        with warnings.catch_warnings():
            # BaseMGP._warn_theory_conditions already reports the moment
            # conditions once; suppress the identical driver warning.
            warnings.filterwarnings(
                "ignore",
                message=r"nu=.*, a1=.*, a2=.*: the moment conditions",
            )
            out = CAVI(
                Xc,
                self.K,
                nu=self.nu,
                a1=self.a1,
                a2=self.a2,
                a_sigma=self.a_sigma,
                b_sigma=self.b_sigma,
                tol=self._effective_tol(),
                max_iter=self.max_iter,
                r=self.r,
                v0=self.v0,
                center=False,  # centering is done once by BaseMGP
                initial_state=self.initial_state,
                jit=self.jit,
                chunk_size=self.chunk_size,
                graph=self.graph,
                audit_every=self.audit_every,
                drop_budget=self.drop_budget,
                verbose=self.verbose,
                plot=False,
            )
        self.raw_output_ = out
        self.mu_ = np.asarray(out["mu"])
        self.V_ = np.asarray(out["V"])
        self.W_ = np.asarray(out["W"])
        self.Phi_ = np.asarray(out["Phi"])
        self.alpha_n_ = out["alpha_n"]
        self.beta_ = np.asarray(out["beta"])
        self.c_ = np.asarray(out["c"])
        self.beta_omega_ = np.asarray(out["beta_omega"])
        self.rho_ = np.asarray(out["rho"])
        self.beta_delta_ = np.asarray(out["beta_delta"])
        self.m_delta_ = np.asarray(out["m_delta"])
        self.log_T_ = np.asarray(out["log_T"])
        self.T_ = out["T"]
        self.elbo_ = list(out["elbo"])
        self.n_iter_ = out["n_iter"]
        self.converged_ = out["converged"]
        self.stop_reason_ = out["stop_reason"]
        self.execution_mode_ = out["execution_mode"]
        self.backend_ = out["backend"]
        self.device_ = out["device"]
        self.compile_sec_ = out["compile_sec"]
        self.runtime_sec_ = out["runtime_sec"]
        self._B_hat = self.mu_
        # posterior mean of IG(alpha_n, beta_j): beta_j / (alpha_n - 1)
        self._sigma2_hat = self.beta_ / (self.alpha_n_ - 1.0)

    def _alignment_loadings(self):
        self._check_fitted()
        return self.mu_[None, :, :]

    def _alignment_outputs(self, result):
        outputs = super()._alignment_outputs(result)
        p, K = self.mu_.shape
        transforms = np.asarray(result.transforms)
        if (transforms.shape != (1, K, K) or
                outputs["B_aligned_"].shape != (1, p, K)):
            raise ValueError("VI alignment requires one (p, K) loading matrix and (K, K) transform")
        R = transforms[0]
        with np.errstate(over="ignore", invalid="ignore"):
            outputs.update({
                "mu_aligned_": outputs["B_aligned_"][0],
                "W_aligned_": self.W_ @ R,
                "V_aligned_": R.T @ self.V_ @ R,
                "Phi_aligned_": R.T @ self.Phi_ @ R,
            })
        outputs["eta_aligned_"] = outputs["W_aligned_"].T
        if (not np.isfinite(R).all() or any(
            not np.isfinite(outputs[name]).all()
            for name in self._ALIGN_ATTRS if name != "rsp_result_"
        )):
            exc = FloatingPointError(
                "VI alignment produced non-finite moments; no estimator state was changed"
            )
            exc.rsp_result = result
            raise exc
        return outputs

    # ------------------------------------------------------------------
    # factor-dimension selection (Gaussian intervals or loading magnitude)
    # ------------------------------------------------------------------
    def n_factors(
        self, level=0.95, *, interval="hpd", use_aligned=False,
        method="credible", threshold=0.1,
    ):
        """Estimate the number of active factor columns.

        Parameters
        ----------
        level : float, default 0.95
            Credibility level in (0, 1) for ``method="credible"``.
            Ignored by ``method="linf"``.
        interval : {"hpd", "quantile"}, default "hpd"
            Gaussian HPD and equal-tailed intervals coincide.  Ignored
            by ``method="linf"``.
        use_aligned : bool, default False
            Use ``mu_aligned_`` (and ``V_aligned_`` for credible intervals)
            instead of the fitted ``mu_``/``V_``.  Requires a successful
            :meth:`align_factors` call first.
        method : {"credible", "linf"}, default "credible"
            ``"credible"`` retains the marginal credible-interval rule:
            a column is active if any loading interval excludes zero.
            ``"linf"`` uses the infinity norm of each loading-mean column:
            column k is active iff ``max_j(abs(mu_jk)) > threshold``.
        threshold : float, default 0.1
            Finite, nonnegative loading magnitude for ``method="linf"``.
            Equality is inactive.  Ignored by ``method="credible"``.

        Returns
        -------
        int
            Number of active columns.

        Notes
        -----
        The infinity-norm rule uses variational means only, without
        accounting for posterior uncertainty.  The threshold is on the
        fitted loading scale (standardized when ``standardize=True``).
        Both rules are invariant to sign flips and column permutations,
        but a general alignment rotation can change the count.
        ``active_entries()`` always uses credible intervals.
        """
        self._check_fitted()
        if not isinstance(method, str) or method not in ("credible", "linf"):
            raise ValueError(
                f"method must be 'credible' or 'linf', got {method!r}"
            )
        if method == "credible":
            return super().n_factors(
                level=level, interval=interval, use_aligned=use_aligned
            )
        if (
            isinstance(threshold, (bool, np.bool_))
            or not isinstance(threshold, (int, float, np.integer, np.floating))
            or not np.isfinite(threshold)
            or threshold < 0.0
        ):
            raise ValueError(
                f"threshold must be a finite nonnegative real number, got {threshold!r}"
            )
        if use_aligned:
            if not hasattr(self, "mu_aligned_"):
                raise RuntimeError(
                    "use_aligned=True requires a successful "
                    "align_factors() call first"
                )
            mu = self.mu_aligned_
        else:
            mu = self.mu_
        return int(np.count_nonzero(np.max(np.abs(mu), axis=0) > threshold))

    def _credible_active(self, level, interval, use_aligned):
        """CAVI path: closed-form Gaussian intervals from (mu, V).

        For q(B_jk) = N(mu_jk, V_jkk) the highest-density interval is the
        equal-tailed one, so ``interval`` has no effect here.  Active iff
        |mu_jk| > z_(1/2 + level/2) * sqrt(V_jkk).
        """
        if use_aligned:
            if not hasattr(self, "mu_aligned_"):
                raise RuntimeError(
                    "use_aligned=True requires a successful "
                    "align_factors() call first"
                )
            mu, V = self.mu_aligned_, self.V_aligned_
        else:
            mu, V = self.mu_, self.V_
        from statistics import NormalDist

        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        sd = np.sqrt(np.diagonal(np.asarray(V, dtype=np.float64),
                                 axis1=1, axis2=2))
        return np.abs(np.asarray(mu, dtype=np.float64)) > z * sd

    # ------------------------------------------------------------------
    # variational posterior covariance of Sigma = B B^T + Omega
    # ------------------------------------------------------------------
    @property
    def Sigma_mean_(self):
        """(p, p) variational posterior mean E_q[Sigma] of the covariance.

        Computed lazily from the fitted ``mu_``, ``V_``, ``beta_`` and
        ``alpha_n_`` as ``mu mu^T + diag(tr(V_j) + beta_j/(alpha_n-1))``,
        i.e. the plug-in :attr:`Sigma_` plus the diagonal correction
        ``diag(tr V_j)``.  float64; cached until the next successful fit.
        """
        self._check_fitted()
        cache = self._sigma_cache
        if "mean" not in cache:
            cache["mean"] = sigma_mean(
                self.mu_, self.V_, self.alpha_n_, self.beta_
            )
        return cache["mean"]

    @property
    def Sigma_var_(self):
        """(p, p) entrywise variational posterior variance of the covariance.

        ``Sigma_var_[i, j] = Var_q(Sigma_ij)`` with ``Sigma = B B^T + Omega``
        under the mean-field posterior; float64, computed lazily and cached
        until the next successful fit.  Requires ``alpha_n > 2``.  The
        returned matrix has p * p entries (guard: 2e7 entries by default);
        for larger p use :meth:`cov_Sigma` on selected entry pairs.
        """
        self._check_fitted()
        cache = self._sigma_cache
        if "var" not in cache:
            cache["var"] = sigma_variance(
                self.mu_, self.V_, self.alpha_n_, self.beta_
            )
        return cache["var"]

    @property
    def Sigma_sd_(self):
        """(p, p) entrywise posterior standard deviation sqrt(Sigma_var_)."""
        self._check_fitted()
        cache = self._sigma_cache
        if "sd" not in cache:
            cache["sd"] = np.sqrt(self.Sigma_var_)
        return cache["sd"]

    def cov_Sigma(self, pairs):
        """Cov_q(Sigma_ij, Sigma_kl) for selected entry pairs.

        Parameters
        ----------
        pairs : iterable of ((i, j), (k, l))
            Zero-based entry indices; order within each pair and between the
            two entries does not matter.

        Returns
        -------
        numpy.ndarray, shape (len(pairs),), float64
            The covariances in the order given.  Pairs with no shared row
            have covariance zero.  Requires ``alpha_n > 2``.

        Notes
        -----
        This is the mean-field variational posterior covariance, computed in
        closed form for the requested pairs; the full p^2 x p^2 covariance of
        ``vec(Sigma)`` is never formed.  See :mod:`mgp._posterior` for the
        case structure and the formulas.
        """
        self._check_fitted()
        return cov_sigma_pairs(
            self.mu_, self.V_, self.alpha_n_, self.beta_, pairs
        )

    def plot_elbo(self, **kwargs):
        """Plot the ELBO trajectory of this run (requires matplotlib).

        Keyword arguments are passed to :func:`mgp._vi.plot_elbo`
        (``ax``, ``logx``, ``show``, ``save``, ``title``).
        """
        self._check_fitted()
        from ._vi import plot_elbo

        return plot_elbo(self.elbo_, **kwargs)
