"""Gibbs sampler estimator for the MGP-shrunk Bayesian factor model.

sklearn-style wrapper of the JAX Gibbs sampler driver
:func:`mgp._gs.GS_MGP` (Bhattacharya & Dunson 2011, "Sparse Bayesian
infinite factor models", Section 3.1, fixed truncation level).  All
numerical semantics (the working precision -- resolved from the
estimator precision, defaulting to the process-wide default (initially
float32); float64 optional --, the log-space shrinkage state, chunked scan
compilation, XLA command buffers, exact Gamma draws, per-chunk
finiteness audits) are those of the underlying driver; this class only
adds the estimator interface and the shared preprocessing of
:class:`mgp.base.BaseMGP`.
"""

import warnings

import numpy as np

from ._gs import GS_MGP
from ._posterior import (
    montecarlo_cov_sigma_pairs,
    montecarlo_sigma_diag_variance,
    montecarlo_sigma_mean,
    montecarlo_sigma_variance,
    suggest_max_draws,
)
from .base import BaseMGP

__all__ = ["GibbsMGP"]


class GibbsMGP(BaseMGP):
    """Gibbs sampler for the Bayesian factor model (exact MCMC inference).

    Parameters
    ----------
    K : int
        Truncation level of the loading matrix (number of columns); an
        overfitted K is enough (redundant columns are shrunk toward
        zero by the increasing shrinkage of the prior).
    nu, a1, a2 : float
        MGP hyperparameters; the defaults nu=3, a1=2.1, a2=3.1 follow
        Durante (2017) and satisfy the paper's moment conditions (see
        :class:`mgp.base.BaseMGP` for the note on the Gibbs driver's own
        a1=a2=2 defaults).
    a_sigma, b_sigma : float
        Hyperparameters of the gamma prior on the noise precisions.
    center : bool, default True
        Subtract the column means of X before fitting.
    precision : {"float32", "float64"} or None, default None
        Working precision of the fit; None inherits the process-wide
        default (:func:`mgp.get_precision`, initially "float32"), an
        explicit value overrides it for this estimator only.  The
        resolved value is stored in ``precision_used_`` and followed by
        :meth:`align_factors` and :meth:`sample_factors`.
    standardize : bool, default False
        Center and divide each column by its sample standard deviation
        (stored in ``x_scale_``) before fitting.  Strongly recommended:
        the MGP prior is calibrated for unit-variance columns.  Requires
        ``center=True``.  Fitted quantities are then in the standardized
        coordinate system (see :class:`mgp.base.BaseMGP`).
    n_iter, burn_in, thin : int
        Total sweeps, discarded initial sweeps, and keep-every-thin.
    seed : int or None
        Seed for jax.random.PRNGKey; None draws a random seed.  The same
        seed (on the same backend) reproduces the same chain.
    graph : bool, default False
        On a GPU backend, record each compiled chunk as one CUDA graph
        (XLA command buffers); elsewhere a warning is issued and the
        plain runner is used.
    verbose : bool, default True
        Print progress about every 10% of the sweeps.

    Attributes
    ----------
    B_samples_ : (S, p, K) numpy array
        Kept loading-matrix draws, S = (n_iter - burn_in) // thin.
        These RAW draws are never modified by :meth:`align_factors`;
        ``ess()`` also continues to operate on them.
    sigma2_samples_ : (S, p) numpy array
        Kept noise-variance draws.
    eta_last_ : (K, n) numpy array
        Latent factors drawn from the final state.  Always RAW:
        :meth:`align_factors` never transforms it (paired aligned
        scores for every kept draw are available on demand through
        :meth:`sample_factors`).
    D_last_ : (p, K) numpy array
        Prior scales 1/sqrt(phi tau) drawn from the final state.
        Never aligned: the column-wise shrinkage scales have no
        counterpart under the varimax rotation.
    seed_ : int
        The seed actually used.
    runtime_sec_ : float
        Wall-clock seconds of the scan loop (excludes AOT compilation).
    B_hat_, sigma2_hat_, Sigma_ : numpy arrays
        Posterior-mean point estimates (see :class:`mgp.base.BaseMGP`);
        with ``standardize=True`` they are on the standardized scale
        (convert with ``x_scale_``).  After :meth:`align_factors`,
        ``B_hat_``/``Sigma_`` reflect the ALIGNED draws.
    Sigma_mc_ : (p, p) numpy array
        Monte Carlo posterior mean of Sigma = B B^T + Omega,
        (1/S) sum_s (B_s B_s^T + diag(sigma2_s)); invariant to
        per-draw orthogonal transforms, hence identical before and after
        :meth:`align_factors`.  Aliased as ``Sigma_mean_``.
    Sigma_mean_ : (p, p) numpy array
        Alias of :attr:`Sigma_mc_` (Monte Carlo posterior mean of Sigma).
    Sigma_var_, Sigma_sd_ : (p, p) numpy arrays
        Monte Carlo entrywise posterior variance and standard deviation of
        Sigma, using ddof = 0.  ``Sigma_var_`` may use a deterministic
        equidistant subsample of the draws when the full scan exceeds the
        default compute budget; see ``Sigma_draws_used_`` and
        :meth:`posterior_sigma_var`.
    Sigma_diag_var_ : (p,) numpy array
        Entrywise posterior variance of the diagonal of Sigma; cheap
        (O(S p K)) and always computed from all draws.
    Sigma_draws_used_ : int or None
        Number of draws used by the most recent ``Sigma_var_`` computation
        (``None`` before it is computed).
    rsp_result_, B_aligned_
        Produced by :meth:`align_factors` (see
        :meth:`mgp.base.BaseMGP.align_factors`); removed by the next
        successful ``fit`` call.
    raw_output_ : dict
        The unmodified return value of the underlying driver.

    Examples
    --------
    Resolve factor switching before summarizing the posterior::

        gs = GibbsMGP(K=10).fit(X)
        gs.align_factors()          # RSP: varimax + sign/permutation
        gs.B_hat_                   # mean of the ALIGNED draws
        gs.rsp_result_.diagnostics  # status, stop reason, losses

    Return factor scores only when needed::

        scores = gs.sample_factors(X, return_eta=True,
                                   return_eta_aligned=True, seed=123)
        scores["eta_sample"]          # (S, K, n), paired to B_samples_
        scores["eta_sample_aligned"]  # (S, K, n), paired to B_aligned_

    Notes
    -----
    The paper's Step 6 (a Metropolis-Hastings update of a1, a2) is
    deliberately not implemented by the underlying driver.  The
    conditionals are proper, but at large p and overfitted K their
    normalizing constants (products of Gamma functions raised to K-1)
    overflow floating point --- for p ~ 1000 and K = 50 the constant is
    of order 10^7105 --- so an MH chain is numerically degenerate; a1
    and a2 are fixed hyperparameters instead.

    ``Sigma_`` is the plug-in estimate ``B_hat B_hat^T + diag(sigma2_hat_)``;
    ``Sigma_mc_``/``Sigma_mean_`` is the Monte Carlo posterior mean of
    ``Sigma = B B^T + Omega``.  ``Sigma_var_``, ``Sigma_sd_``,
    ``Sigma_diag_var_`` and :meth:`cov_Sigma` estimate the posterior
    covariance of Sigma from the draws with ddof = 0; they are noisy
    Monte Carlo estimates (and the draws are autocorrelated), not exact
    moments.  The full p x p variance scan costs O(S p^2 K); when it exceeds
    the default compute budget, ``Sigma_var_`` uses a deterministic
    equidistant subsample (with a warning) and records the count in
    ``Sigma_draws_used_``.  Use :meth:`posterior_sigma_var` to control this
    explicitly, or the cheap ``Sigma_diag_var_`` / :meth:`cov_Sigma` for
    selected entries on large problems.  As with the other covariance
    summaries, the moments are computed from the raw ``B_samples_`` and
    ``sigma2_samples_`` and are invariant to :meth:`align_factors`.
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
        n_iter=2000,
        burn_in=1000,
        thin=1,
        seed=None,
        graph=False,
        verbose=True,
    ):
        super().__init__(
            K, nu=nu, a1=a1, a2=a2, a_sigma=a_sigma, b_sigma=b_sigma,
            center=center, precision=precision, standardize=standardize,
        )
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.thin = thin
        self.seed = seed
        self.graph = graph
        self.verbose = verbose

    def _fit_arrays(self, Xc):
        # Posterior-moment cache (Sigma_mc_/Sigma_var_/...); a new fit must
        # not reuse moments of the previous fit.
        self._sigma_cache = {}
        self.Sigma_draws_used_ = None
        with warnings.catch_warnings():
            # BaseMGP._warn_theory_conditions already reports the moment
            # conditions once; suppress the duplicate driver warnings.
            warnings.filterwarnings(
                "ignore", message=r"nu = .* violates the condition nu > 2"
            )
            warnings.filterwarnings(
                "ignore", message=r"a2 = .* violates the condition a2 > 1"
            )
            out = GS_MGP(
                Xc,
                self.K,
                nu=self.nu,
                a1=self.a1,
                a2=self.a2,
                a_sigma=self.a_sigma,
                b_sigma=self.b_sigma,
                n_iter=self.n_iter,
                burn_in=self.burn_in,
                thin=self.thin,
                seed=self.seed,
                verbose=self.verbose,
                graph=self.graph,
            )
        self.raw_output_ = out
        self.B_samples_ = out["B"]
        self.sigma2_samples_ = out["sigma2"]
        self.eta_last_ = out["eta_last"]
        self.D_last_ = out["D_last"]
        self.seed_ = out["seed"]
        self.runtime_sec_ = out["runtime_sec"]
        self._B_hat = self.B_samples_.mean(axis=0)
        self._sigma2_hat = self.sigma2_samples_.mean(axis=0)

    def sample_factors(
        self,
        X,
        *,
        return_eta=True,
        return_eta_aligned=False,
        seed=None,
        batch_size=32,
    ):
        """Return optional factor-score draws paired to all kept samples.

        For every kept ``(B_samples_[s], sigma2_samples_[s])``, draw a
        fresh ``eta_s`` from its conditional distribution given X.  For
        the original training data, these are joint posterior samples,
        not the historical intermediate scores from the Gibbs sweeps.
        They are correctly paired even when thinning omitted the final
        sweep.  This method does not change or store any fitted attributes.

        Parameters
        ----------
        X : array-like, shape (n, p)
            Uncentered observations in the fitted feature order.  The
            fitted ``x_offset_`` (and ``x_scale_`` when standardize=True)
            is applied internally; do not transform X yourself.
            Pass the original training data for posterior factor scores.
            New rows produce conditional scores under the already-fitted
            parameter draws, without updating the loading posterior.
        return_eta : bool, default True
            Include raw scores under the key ``"eta_sample"``.
        return_eta_aligned : bool, default False
            Include ``"eta_sample_aligned"`` using the same raw draws and
            the latest successful :meth:`align_factors` transforms.
            Requires alignment first.  Setting both flags to False
            returns an empty dictionary without sampling.
        seed : int or None, default None
            Unsigned 32-bit seed for this separate sampling stream.
            None selects a fresh random seed.  A fixed seed gives the
            same draws on the same backend, independently of the output
            flags and batching (up to floating-point rounding).
        batch_size : int, default 32
            Maximum number of retained draws processed per batch.
            Requested output arrays are allocated on the host; when only
            aligned scores are requested, raw scores are held one batch
            at a time.

        Returns
        -------
        dict
            Only the requested keys, each a NumPy array of shape
            ``(S, K, n)``.  Raw scores use the working precision; aligned
            scores use float64 (the alignment matmul promotes the working
            precision; see ``mgp._factor_scores``).
            For each s, ``B_aligned_[s] @ eta_sample_aligned[s]`` equals
            ``B_samples_[s] @ eta_sample[s]`` up to numerical error.
        """
        self._check_fitted()
        from ._factor_scores import sample_factor_scores
        from ._precision import get_precision, set_precision

        resolved = getattr(self, "precision_used_", None) or self._resolve_precision()
        previous = get_precision()
        if previous != resolved:
            set_precision(resolved)
        try:
            result = getattr(self, "rsp_result_", None)
            transforms = None if result is None else result.transforms
            return sample_factor_scores(
                X, self.B_samples_, self.sigma2_samples_, self.x_offset_,
                scale=getattr(self, "x_scale_", None),
                return_eta=return_eta,
                return_eta_aligned=return_eta_aligned,
                seed=seed,
                batch_size=batch_size,
                transforms=transforms,
            )
        finally:
            if previous != resolved:
                set_precision(previous)

    @property
    def Sigma_mc_(self):
        """(p, p) Monte Carlo posterior mean of Sigma = B B^T + Omega.

        Computed from the raw draws as
        ``(1/S) sum_s (B_s B_s^T + diag(sigma2_s))``, equivalently
        ``(1/S) sum_s B_s B_s^T + diag(sigma2_hat_)``; cached until the next
        successful fit.  Each per-draw covariance is invariant to that
        draw's own orthogonal transform, so this quantity is identical
        before and after :meth:`align_factors` (it is computed from the raw
        ``B_samples_``).  Contrast with the plug-in
        :attr:`mgp.base.BaseMGP.Sigma_` = B_hat B_hat^T + diag(
        sigma2_hat_), whose value DOES change under alignment (e.g. two
        draws B and -B have mean 0 raw but mean B after sign alignment):
        ``Sigma_mc_`` summarizes the posterior mean of the covariance, while
        ``Sigma_`` is the covariance of the (aligned or raw) loading point
        estimate.  Aliased as :attr:`Sigma_mean_`.
        """
        self._check_fitted()
        cache = self._sigma_cache
        if "mc" not in cache:
            cache["mc"] = montecarlo_sigma_mean(
                self.B_samples_, self.sigma2_samples_
            )
        return cache["mc"]

    @property
    def Sigma_mean_(self):
        """(p, p) Monte Carlo posterior mean of Sigma; alias of Sigma_mc_."""
        return self.Sigma_mc_

    @property
    def Sigma_var_(self):
        """(p, p) Monte Carlo entrywise posterior variance of Sigma.

        ``Sigma_var_[i, j]`` estimates ``Var(Sigma_ij)`` from the draws with
        ddof = 0, where ``Sigma = B B^T + Omega`` and each draw contributes
        ``Sigma_s = B_s B_s^T + diag(sigma2_s)``.  Cached until the next
        successful fit.  The computation costs O(S p^2 K); when it exceeds
        the default compute budget, a deterministic equidistant
        subsample is used and a ``UserWarning`` is emitted; the actual number
        of draws is stored in :attr:`Sigma_draws_used_`.  Use
        :meth:`posterior_sigma_var` to control the draw count explicitly, or
        :attr:`Sigma_diag_var_`/:meth:`cov_Sigma` for cheap summaries.
        """
        self._check_fitted()
        if self.B_samples_.shape[0] < 2:
            raise ValueError(
                "Sigma_var_ needs at least 2 kept draws (ddof=0 variance is "
                "undefined for a single draw); increase n_iter - burn_in or "
                "reduce thin"
            )
        cache = self._sigma_cache
        if "var" not in cache:
            S, p, K = self.B_samples_.shape
            n_used = suggest_max_draws(p, K, S)
            if n_used < S:
                warnings.warn(
                    f"Sigma_var_: using {n_used} of {S} draws (deterministic "
                    "equidistant subsample) because the full p x p variance "
                    "scan exceeds the default compute budget; the Monte "
                    "Carlo error is correspondingly larger.  Call "
                    "posterior_sigma_var(max_draws=None) for all draws, or "
                    "use cov_Sigma(pairs)/Sigma_diag_var_ for cheap entry "
                    "summaries.",
                    UserWarning,
                    stacklevel=2,
                )
            var, used = montecarlo_sigma_variance(
                self.B_samples_, self.sigma2_samples_, max_draws=n_used
            )
            cache["var"] = var
            cache.pop("sd", None)
            self.Sigma_draws_used_ = used
        return cache["var"]

    @property
    def Sigma_sd_(self):
        """(p, p) Monte Carlo entrywise posterior standard deviation of Sigma."""
        self._check_fitted()
        cache = self._sigma_cache
        if "sd" not in cache:
            cache["sd"] = np.sqrt(self.Sigma_var_)
        return cache["sd"]

    @property
    def Sigma_diag_var_(self):
        """(p,) Monte Carlo posterior variance of the diagonal of Sigma.

        Cheap (O(S p K)); always uses all draws and is not subject to the
        full-matrix compute budget.
        """
        self._check_fitted()
        cache = self._sigma_cache
        if "diag_var" not in cache:
            cache["diag_var"] = montecarlo_sigma_diag_variance(
                self.B_samples_, self.sigma2_samples_
            )
        return cache["diag_var"]

    def cov_Sigma(self, pairs):
        """Monte Carlo posterior covariance between selected entries of Sigma.

        Parameters
        ----------
        pairs : iterable of ((i, j), (k, l))
            Zero-based entry indices; order within each pair and between the
            two entries does not matter.

        Returns
        -------
        numpy.ndarray, shape (len(pairs),), float64
            Sample covariances with ddof = 0.  Uses all draws; the cost is
            O(S K) per pair, independent of p^2, so this is the recommended
            summary when p is large.  Requires no subsampling.
        """
        self._check_fitted()
        return montecarlo_cov_sigma_pairs(
            self.B_samples_, self.sigma2_samples_, pairs
        )

    def posterior_sigma_var(self, max_draws=None, chunk_draws=None):
        """Compute the (p, p) Monte Carlo variance matrix with explicit controls.

        Parameters
        ----------
        max_draws : int or None, default None
            Number of draws to use; ``None`` uses all draws.  A value below
            the number of kept draws selects a deterministic equidistant
            subsample, with correspondingly larger Monte Carlo error.
        chunk_draws : int or None, default None
            Draws per processing block; ``None`` selects a block size from a
            memory budget.  Does not change the result.

        Returns
        -------
        numpy.ndarray, shape (p, p), float64
            Entrywise posterior variance with ddof = 0.  Unlike the
            ``Sigma_var_`` property this method does not apply the default
            compute budget (``None`` means all draws; an explicit
            ``max_draws`` selects the deterministic equidistant subsample).
            With ``max_draws=None`` the result is also cached for
            ``Sigma_var_`` and ``Sigma_draws_used_`` is updated to the number
            of draws used; a subsampled call leaves the cached property
            result and that attribute untouched.
        """
        self._check_fitted()
        var, used = montecarlo_sigma_variance(
            self.B_samples_, self.sigma2_samples_,
            max_draws=max_draws, chunk_draws=chunk_draws,
        )
        if max_draws is None:
            self._sigma_cache["var"] = var
            self._sigma_cache.pop("sd", None)
            self.Sigma_draws_used_ = used
        return var

    def ess(self, method="bulk", prob=None, relative=False):
        """Per-entry effective sample size of the kept draws (ArviZ).

        Returns a dict {"B": (p, K), "sigma2": (p,)} of ESS values for
        the kept loading-matrix and noise-variance draws.  ``method``
        and ``prob`` are passed to :func:`arviz.ess` ("bulk" default;
        "tail"/"quantile" require ``prob``); ``relative=True`` returns
        ESS per draw.  Requires the optional arviz dependency.
        """
        self._check_fitted()
        from ._gs import ess

        return ess(self.B_samples_, self.sigma2_samples_, method=method,
                   prob=prob, relative=relative)

    def ess_summary(self, method="bulk"):
        """Compact ESS summary (min/median/mean/max per array).

        The minimum is the number to watch; bulk ESS above ~400 per
        entry is the usual rule of thumb for reliable posterior means.
        Requires the optional arviz dependency.
        """
        self._check_fitted()
        from ._gs import ess_summary

        return ess_summary(self.B_samples_, self.sigma2_samples_, method=method)
