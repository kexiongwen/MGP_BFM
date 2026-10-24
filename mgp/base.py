"""Shared base class for the MGP estimators.

:class:`BaseMGP` holds the hyperparameters shared by the Gibbs sampler
(:class:`mgp.GibbsMGP`) and the CAVI approximation (:class:`mgp.CaviMGP`),
and implements the common preprocessing (validation, working-precision
coercion, column centering) and the common fitted accessors (``B_hat_``,
``sigma2_hat_``, ``Sigma_``, ``n_factors``).

The model is the latent factor model of Bhattacharya & Dunson (2011),
"Sparse Bayesian infinite factor models", Biometrika 98(2), 291-306:

    x_i = B eta_i + eps_i,
    eta_i ~ N_K(0, I_K),
    eps_i ~ N_p(0, Omega),  Omega = diag(sigma_1^2, ..., sigma_p^2),

with the multiplicative gamma process (MGP) shrinkage prior on the
loadings,

    B_jk | omega_jk, T_k ~ N(0, (omega_jk T_k)^{-1}),
    omega_jk ~ Ga(nu/2, nu/2),  T_k = prod_{l<=k} delta_l,
    delta_1 ~ Ga(a1, 1),  delta_l ~ Ga(a2, 1),

and a gamma prior on the noise precisions, sigma_j^{-2} ~ Ga(a_sigma,
b_sigma) (the CAVI side parameterizes the same prior through the
induced inverse-gamma on sigma_j^2).
"""

import warnings

import numpy as np

__all__ = ["BaseMGP"]


class BaseMGP:
    """Shared hyperparameters and preprocessing for the MGP estimators.

    Parameters
    ----------
    K : int
        Truncation level of the loading matrix (number of columns); an
        overfitted K is enough (redundant columns are shrunk toward
        zero by the increasing shrinkage of the prior).
    nu, a1, a2 : float
        MGP hyperparameters: nu for the local omega_jk (prior
        Gamma(nu/2, nu/2)); a1 for delta_1 and a2 for delta_l (l >= 2),
        both Gamma(shape, rate=1).  The defaults nu=3, a1=2.1, a2=3.1
        follow Durante (2017) (1 < a1 < a2, so
        E[T_k] = a1 a2^{k-1} increases and shrinkage strengthens with
        the column index).  Note that the underlying Gibbs driver has
        its OWN defaults a1=a2=2 (the Bhattacharya-Dunson Section 4.1
        Ga(2,1) hyperprior means); the estimators always pass a1/a2
        explicitly, so the effective defaults through this interface
        are 2.1/3.1 for BOTH algorithms.
    a_sigma, b_sigma : float
        Hyperparameters of the gamma prior on the noise precisions,
        sigma_j^{-2} ~ Ga(a_sigma, b_sigma); both must be > 0.
        Paper Section 4.1 defaults: a_sigma = 1, b_sigma = 0.3.
    center : bool, default True
        Subtract the column means of X before fitting (the model has
        zero mean).  The column means are stored in ``x_offset_``.
    standardize : bool, default False
        Divide each column of X by its sample standard deviation after
        centering and store the scales in ``x_scale_``.  The MGP prior
        is calibrated for unit-variance columns (``b_sigma=0.3`` and a
        fixed unit loading scale), so ``standardize=True`` is strongly
        recommended for data in arbitrary units.  Fitted quantities are
        then expressed in the standardized coordinate system:
        ``B_orig[j, k] = x_scale_[j] * B_std[j, k]``,
        ``sigma2_orig[j] = x_scale_[j]**2 * sigma2_std[j]`` and
        ``Sigma_orig = diag(x_scale_) @ Sigma_std @ diag(x_scale_)``.
        Requires ``center=True``.  When ``standardize=False`` a warning
        is emitted if column standard deviations deviate strongly from 1.
    precision : {"float32", "float64"} or None
        Working precision of the fit.  The default is estimator-specific:
        ``GibbsMGP`` uses ``None`` (inherit the process-wide default at
        fit time, initially "float32"), while ``CaviMGP`` defaults to
        "float64" because its absolute ELBO stopping criterion is
        sensitive to float32 jitter.  An explicit string overrides the
        default for this estimator only; ``None`` always means "follow
        :func:`mgp.get_precision`".  ``fit`` applies the resolved
        precision only for the duration of the operation and restores
        the previous global value afterwards, so estimator calls never
        permanently mutate the process-wide default.  The resolved value
        is stored in ``precision_used_`` and is followed by
        :meth:`align_factors` and (for Gibbs)
        :meth:`~mgp.GibbsMGP.sample_factors`.

    Notes
    -----
    This class only stores parameters and provides shared machinery;
    use :class:`mgp.GibbsMGP` or :class:`mgp.CaviMGP` for fitting.
    Algorithm-specific validation is left to the underlying drivers so
    that the wrapper never drifts from their rules.

    With ``standardize=True`` every fitted array (``B_hat_``,
    ``B_samples_``, ``sigma2_hat_``, ``Sigma_``, posterior moments, ...)
    lives in the standardized coordinate system; multiply/divide by
    ``x_scale_`` as documented above to convert to original units.
    ``precision_used_`` records the working precision of the last
    successful fit.
    """

    def __init__(
        self,
        K,
        nu=3.0,
        a1=2.1,
        a2=3.1,
        a_sigma=1.0,
        b_sigma=0.3,
        center=True,
        precision=None,
        standardize=False,
    ):
        self.K = K
        self.nu = nu
        self.a1 = a1
        self.a2 = a2
        self.a_sigma = a_sigma
        self.b_sigma = b_sigma
        self.center = center
        self.precision = precision
        self.standardize = standardize

    def _resolve_precision(self):
        """Resolve this estimator's precision against the global default.

        ``precision=None`` inherits :func:`mgp.get_precision` at call time;
        an explicit value overrides it for this estimator only.
        """
        from ._precision import PRECISIONS, get_precision

        if self.precision is None:
            return get_precision()
        if self.precision not in PRECISIONS:
            raise ValueError(
                f"precision must be None or one of {PRECISIONS}, "
                f"got {self.precision!r}"
            )
        return self.precision

    # ------------------------------------------------------------------
    # fitting (template method)
    # ------------------------------------------------------------------
    # Alignment outputs produced by align_factors(); removed by fit() AFTER
    # the new fit succeeds, so a fit that fails (invalid X, numerical
    # error) keeps the previous state -- raw or aligned -- fully
    # consistent instead of half-cleared.
    _ALIGN_ATTRS = ("rsp_result_", "B_aligned_")
    # Gibbs summarizes aligned draws; VI keeps its fitted estimate and
    # exposes transformed moments separately.
    _ALIGN_UPDATES_B_HAT = True

    def fit(self, X):
        """Fit the model to the data matrix X and return ``self``.

        The resolved working precision (``precision`` if not None,
        otherwise the process-wide default from
        :func:`mgp.get_precision`) is applied only for the duration of
        the fit and restored afterwards.  X is validated, coerced to that
        precision, column-centered (unless ``center=False``) and, with
        ``standardize=True``, scaled to unit sample variance; the
        concrete algorithm then runs in :meth:`_fit_arrays`.  Results of
        a previous :meth:`align_factors` call are discarded only after
        the new fit succeeds: a fit that raises leaves the estimator
        exactly as it was (including the process-wide precision).
        """
        # Fitting replaces attributes, so preserving their references is
        # enough to restore the previous fit without copying large draws.
        previous_state = self.__dict__.copy()
        from ._precision import get_precision, set_precision

        resolved = self._resolve_precision()
        previous_precision = get_precision()
        try:
            if previous_precision != resolved:
                set_precision(resolved)
            self.precision_used_ = resolved
            self._warn_theory_conditions()
            Xc = self._prepare_X(X)
            self.n_samples_, self.n_features_in_ = Xc.shape
            self._fit_arrays(Xc)
            for attr in self._ALIGN_ATTRS:
                if hasattr(self, attr):
                    delattr(self, attr)
        except BaseException:
            # Restore preprocessing and partially assigned results even
            # when a long-running fit is interrupted, then re-raise.
            self.__dict__.clear()
            self.__dict__.update(previous_state)
            raise
        finally:
            if previous_precision != resolved:
                set_precision(previous_precision)
        return self

    def _fit_arrays(self, Xc):
        """Run the concrete algorithm on the prepared data matrix.

        Replace fitted attributes rather than modifying existing arrays
        or containers in place, so :meth:`fit` can restore them on failure.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # shared preprocessing
    # ------------------------------------------------------------------
    def _warn_theory_conditions(self):
        """Warn when the hyperparameters violate the paper's conditions."""
        if not (self.nu > 2.0) or not (self.a1 > 1.0) or not (self.a2 > 1.0):
            warnings.warn(
                f"nu={self.nu}, a1={self.a1}, a2={self.a2}: the MGP moment "
                "conditions are violated (the paper needs nu > 2 for a "
                "finite E[1/omega_jk] and a1, a2 > 1 for a finite E[1/T_k]); "
                "the algorithm still runs, but these expectations do not "
                "exist and the prior interpretation weakens (for the Gibbs "
                "sampler, a2 > 1 is additionally what makes the tau_k "
                "stochastically increasing, i.e. the increasing shrinkage "
                "that suppresses redundant columns)",
                UserWarning,
                stacklevel=3,
            )
        elif self.a2 <= 2.0:
            warnings.warn(
                f"a2={self.a2} <= 2: the Bhattacharya-Dunson (2011) "
                "Theorem 1 truncation-error bound assumes a2 > 2, so the "
                "finite-truncation accuracy guarantee does not apply (the "
                "sampler / CAVI itself remains valid).  The prior's second "
                "moments additionally require a1 > 2 and a2 > 3.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _prepare_X(self, X):
        """Validate, coerce, center and (optionally) standardize X."""
        from ._precision import work_dtypes

        _, np_dtype = work_dtypes()
        X = np.asarray(X, dtype=np_dtype)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-dimensional, got shape {tuple(X.shape)}"
            )
        if X.shape[0] < 1 or X.shape[1] < 1:
            raise ValueError(
                f"X must have at least one row and one column, got {X.shape}"
            )
        if not np.isfinite(X).all():
            raise ValueError(
                "X must contain only finite values (found NaN or inf); "
                "note that finite float64 inputs can overflow to inf when "
                "the working precision is float32"
            )
        if self.standardize and not self.center:
            raise ValueError("standardize=True requires center=True")
        if self.center:
            self.x_offset_ = X.mean(axis=0)
            X = X - self.x_offset_
        else:
            self.x_offset_ = np.zeros(X.shape[1], dtype=X.dtype)
        if self.standardize:
            if X.shape[0] > 1:
                sd = X.std(axis=0, ddof=1)
            else:
                sd = np.ones(X.shape[1], dtype=X.dtype)
            sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
            self.x_scale_ = sd
            X = X / sd
        else:
            self.x_scale_ = np.ones(X.shape[1], dtype=X.dtype)
            if X.shape[0] > 1:
                sd = X.std(axis=0, ddof=1)
                positive = sd[sd > 0]
                if positive.size and (
                    positive.max() > 10.0 or positive.min() < 0.1
                ):
                    warnings.warn(
                        "column standard deviations are far from 1 (range "
                        f"{positive.min():.3g} to {positive.max():.3g}); the "
                        "MGP prior is calibrated for unit-variance columns. "
                        "Consider standardize=True (or pass pre-standardized X).",
                        UserWarning,
                        stacklevel=3,
                    )
        return X

    # ------------------------------------------------------------------
    # fitted accessors
    # ------------------------------------------------------------------
    def _check_fitted(self):
        if not hasattr(self, "_B_hat"):
            raise RuntimeError(
                f"{type(self).__name__} is not fitted yet; call fit(X) first"
            )

    @property
    def B_hat_(self):
        """(p, K) point estimate of the loading matrix.

        Posterior mean over the kept draws for :class:`GibbsMGP`; the
        variational mean ``mu`` for :class:`CaviMGP`.  After
        :meth:`align_factors` (GibbsMGP), this is the posterior mean of
        the ALIGNED draws; the mean of the raw draws remains available
        as ``B_samples_.mean(axis=0)``.  For CaviMGP it always remains
        ``mu_``; the transformed mean is available as ``mu_aligned_``.
        With ``standardize=True`` this estimate is on the standardized
        scale (multiply rows by ``x_scale_`` for original units).
        """
        self._check_fitted()
        return self._B_hat

    @property
    def sigma2_hat_(self):
        """(p,) point estimate of the idiosyncratic variances.

        Posterior mean over the kept draws for :class:`GibbsMGP`; the
        inverse-gamma posterior mean beta / (alpha_n - 1) for
        :class:`CaviMGP`.  With ``standardize=True`` multiply by
        ``x_scale_**2`` for original units.
        """
        self._check_fitted()
        return self._sigma2_hat

    @property
    def Sigma_(self):
        """(p, p) estimated covariance B_hat B_hat^T + diag(sigma2_hat_).

        With ``standardize=True`` this is the covariance of the
        standardized variables; the original-scale covariance is
        ``diag(x_scale_) @ Sigma_ @ diag(x_scale_)``.
        """
        self._check_fitted()
        return self._B_hat @ self._B_hat.T + np.diag(self._sigma2_hat)

    # ------------------------------------------------------------------
    # factor-dimension selection: marginal credible intervals
    # ------------------------------------------------------------------
    def n_factors(self, level=0.95, *, interval="hpd", use_aligned=False):
        """Estimate the number of active factor columns.

        Entry-wise rule: for every loading entry B_jk, compute its
        smallest (highest-density) marginal credible interval at
        credibility ``level``; if 0 lies inside the interval the entry is
        treated as zero.  A factor column k is ACTIVE iff at least one
        entry j in it is nonzero by this rule, and the estimated factor
        count is the number of active columns:

            r_hat = #{ k : exists j with 0 notin CI(B_jk) }.

        GibbsMGP builds the intervals empirically from the posterior
        draws (``B_samples_``, or ``B_aligned_`` with
        ``use_aligned=True``); CaviMGP uses the closed-form Gaussian
        interval ``|mu_jk| > z * sqrt(V_jkk)`` of its variational
        posterior (for a symmetric Gaussian the highest-density interval
        is the equal-tailed one).

        Parameters
        ----------
        level : float, default 0.95
            Credibility level in (0, 1); larger is more conservative
            (more entries called zero).
        interval : {"hpd", "quantile"}, default "hpd"
            Interval construction for the empirical (Gibbs) path:
            "hpd" is the shortest interval covering ``level`` of the
            draws (robust to multimodal marginals, e.g. residual sign
            switching in the chain); "quantile" is the equal-tailed
            interval.  The Gaussian (CAVI) path is identical under both.
        use_aligned : bool, default False
            Gibbs: use ``B_aligned_`` instead of the raw ``B_samples_``
            (requires a successful :meth:`align_factors` first).
            CAVI: use ``mu_aligned_``/``V_aligned_`` instead of
            ``mu_``/``V_``.  Note that the marginal intervals are
            invariant to sign flips and column permutations but NOT to
            the varimax rotation: RSP's default rotation redistributes
            mass across columns, so the two variants answer slightly
            different questions (raw = the prior's coordinate system,
            aligned = the displayed orientation).

        Returns
        -------
        int

        Notes
        -----
        Unlike a point-estimate rule this uses the full marginal
        posterior, so a weak residual column whose draws straddle zero is
        correctly called inactive.  The draws are autocorrelated; the
        interval quality depends on mixing (see ``GibbsMGP.ess``).
        """
        return int(
            self.active_entries(level=level, interval=interval,
                                use_aligned=use_aligned).any(axis=0).sum()
        )

    def active_entries(self, level=0.95, *, interval="hpd", use_aligned=False):
        """(p, K) bool mask of active loading entries (see :meth:`n_factors`).

        Entry (j, k) is active iff 0 lies outside its smallest marginal
        credible interval at credibility ``level``.
        """
        self._check_fitted()
        if (
            isinstance(level, (bool, np.bool_))
            or not isinstance(level, (int, float, np.integer, np.floating))
            or not np.isfinite(level)
            or not 0.0 < level < 1.0
        ):
            raise ValueError(f"level must be a real number in (0, 1), got {level!r}")
        if interval not in ("hpd", "quantile"):
            raise ValueError(
                f"interval must be 'hpd' or 'quantile', got {interval!r}"
            )
        return self._credible_active(level, interval, use_aligned)

    def _credible_draws(self, use_aligned):
        """Gibbs path: the draw array to build empirical intervals from."""
        self._check_fitted()
        draws = getattr(self, "B_samples_", None)
        if draws is None:
            raise RuntimeError(
                f"{type(self).__name__} has no posterior draws for the "
                "credible-interval rule"
            )
        if use_aligned:
            aligned = getattr(self, "B_aligned_", None)
            if aligned is None:
                raise RuntimeError(
                    "use_aligned=True requires a successful "
                    "align_factors() call first"
                )
            return aligned
        return draws

    def _credible_active(self, level, interval, use_aligned):
        """Gibbs path: empirical marginal intervals over the draws."""
        draws = np.asarray(self._credible_draws(use_aligned), dtype=np.float64)
        S = draws.shape[0]
        if S < 2:
            raise ValueError(
                f"the credible rule needs at least 2 draws, got S = {S}"
            )
        if interval == "quantile":
            a = 1.0 - level
            lo, hi = np.quantile(draws, [a / 2.0, 1.0 - a / 2.0], axis=0)
        else:
            # HPD: shortest window covering m = ceil(level * S) sorted
            # draws (a window of m consecutive points is s[i+m-1] - s[i])
            s = np.sort(draws, axis=0)
            m = int(np.ceil(level * S))
            if m >= S:
                lo, hi = s[0], s[-1]
            else:
                w = s[m - 1:] - s[: S - m + 1]             # (S - m + 1, p, K)
                i = np.argmin(w, axis=0)                   # (p, K)
                lo = np.take_along_axis(s, i[None], axis=0)[0]
                hi = np.take_along_axis(s, (i + m - 1)[None], axis=0)[0]
        return (lo > 0) | (hi < 0)

    # ------------------------------------------------------------------
    # factor switching (RSP post-processing)
    # ------------------------------------------------------------------
    def align_factors(
        self,
        *,
        rotate=True,
        sp_path="elim",
        init="auto",
        reference=None,
        threshold=1e-6,
        max_iter=100,
        polar="qdwh",
        assignment_backend="auto",
        precision=None,
        verbose=False,
        metadata=None,
    ):
        """Align Gibbs draws or the VI loading mean using RSP.

        Runs the Rotation-Sign-Permutation algorithm implemented in
        :mod:`mgp._rsp` (see ``Factor_Switching.md``).

        GibbsMGP aligns the kept ``B_samples_`` and updates ``B_hat_``
        to their aligned mean.  Consequently the plug-in ``Sigma_``
        can change, while the raw draws used by ``ess()`` remain
        unchanged.

        CaviMGP supplies ``mu_[None, :, :]`` as one loading matrix,
        not an MCMC draw.  It exposes the rotated Gaussian means and
        covariances separately; ``B_hat_``, the fitted VI parameters,
        shrinkage moments, and ELBO remain unchanged.  These are
        representations for interpretation/comparison, not another
        optimum under the original coordinate-dependent shrinkage prior.

        Repeated calls always start from the original draws or VI mean
        and replace the previous alignment outputs.  A successful new
        ``fit`` discards them; a failed fit or alignment preserves them.

        Parameters
        ----------
        rotate, sp_path, init, reference, threshold, max_iter, polar,
        assignment_backend, precision, verbose
            Passed through to :func:`mgp._rsp.rsp_exact` (see its
            docstring / the Factor_Switching.md notes for details).
            ``precision=None`` (the default) follows the estimator's own
            ``precision``.  ``return_transforms`` is forced on for
            consistent score and covariance transforms.
            ``assignment_backend="auto"`` needs optax for K > 7 (install
            with ``pip install -e ".[rsp]"``); the "scipy" assignment
            backend additionally needs scipy.
        metadata : mapping or None, default None
            Optional caller metadata attached to ``rsp_result_`` (e.g.
            chain name, seed, model settings).  Only hashable scalar
            values or sequences of them are accepted (the RSP result is
            a pytree); the kernels never read it.

        Returns
        -------
        self
            New attributes:

            - ``rsp_result_``: the full :class:`mgp._rsp.RSPResult`
              (aligned draws, signs, permutations, loss histories,
              diagnostics, per-draw transforms);
            - ``B_aligned_``: (S, p, K) aligned loadings, with S=1 for VI.

            Gibbs scores are returned on request by
            :meth:`GibbsMGP.sample_factors`.  VI also exposes
            ``mu_aligned_``, ``V_aligned_``, ``W_aligned_``,
            ``Phi_aligned_``, and ``eta_aligned_ = W_aligned_.T``;
            see :class:`CaviMGP` for shapes and semantics.

        Raises
        ------
        RuntimeError
            If the estimator is unfitted or has no supported alignment input.
        FloatingPointError
            If RSP ended with status_code 4 or the transformed VI
              moments are non-finite: NO estimator attribute changes, the
              previous estimates (raw or from a previous successful
              alignment) are kept, and the failed run is attached to
              the exception as ``rsp_result`` for diagnostics.
        UserWarning
            If the run ended with status_code 3 (the exact objective
              increased beyond the numerical tolerance); the finite
              aligned results are still accepted.
        """
        if precision is None:
            _resolved = getattr(self, "precision_used_", None)
            precision = (
                _resolved if _resolved is not None else self._resolve_precision()
            )
        draws = self._alignment_loadings()
        try:
            from ._rsp import rsp_exact
        except ImportError as exc:  # pragma: no cover - optional import
            raise ImportError(
                "the RSP module mgp._rsp failed to import: "
                f"{exc}"
            ) from exc

        try:
            result = rsp_exact(
                draws,
                rotate=rotate,
                sp_path=sp_path,
                init=init,
                reference=reference,
                threshold=threshold,
                max_iter=max_iter,
                polar=polar,
                assignment_backend=assignment_backend,
                precision=precision,
                return_transforms=True,
                verbose=verbose,
                metadata=metadata,
            )
        except ImportError as exc:
            raise ImportError(
                f"the RSP alignment backend is missing a dependency "
                f"({exc}); install it with: pip install -e \".[rsp]\" "
                "(optax, needed for K > 7); the \"scipy\" assignment "
                "backend additionally needs scipy"
            ) from exc

        status = int(result.status_code)
        if status == 4:
            # Zero estimator state change on failure: the failed run is
            # attached to the exception (exc.rsp_result) for diagnostics
            # and NO attribute is written.  A failed run must not
            # create alignment outputs or overwrite a successful one.
            exc = FloatingPointError(
                "rsp_exact ended with status_code 4 (non-finite "
                "objective, intermediate, or returned state); no "
                "estimator state was changed and the previous estimates "
                "are kept (the failed run is attached as exc.rsp_result)"
            )
            exc.rsp_result = result
            raise exc
        # Build and validate every output before publishing any of them.
        updates = self._alignment_outputs(result)
        if status == 3:
            warnings.warn(
                "rsp_exact ended with status_code 3 (the exact alignment "
                "objective increased beyond the numerical tolerance); "
                "accepting the finite aligned results -- inspect "
                "rsp_result_.diagnostics",
                UserWarning,
                stacklevel=2,
            )

        self.__dict__.update(updates)
        return self

    def _alignment_loadings(self):
        """Return original loadings with a leading alignment-sample axis."""
        self._check_fitted()
        draws = getattr(self, "B_samples_", None)
        if draws is None:
            raise RuntimeError(f"{type(self).__name__} has no alignment input")
        return draws

    def _alignment_outputs(self, result):
        """Prepare alignment attributes without changing estimator state."""
        outputs = {
            "rsp_result_": result,
            "B_aligned_": np.asarray(result.aligned_loadings),
        }
        if self._ALIGN_UPDATES_B_HAT:
            outputs["_B_hat"] = np.asarray(result.mean_loadings)
        return outputs
