<div align="center">

# MGP

**Sparse Bayesian factor models with the multiplicative gamma process prior — in JAX**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-CPU%20%7C%20GPU-orange.svg)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/license-GPL--2.0--only-blue.svg)](LICENSE)

</div>

`mgp` is a self-contained Python package for Bayesian factor analysis with the
**multiplicative gamma process (MGP) shrinkage prior** of
Bhattacharya & Dunson (2011), *Sparse Bayesian infinite factor models*.
It provides a unified, sklearn-style interface to two JAX inference engines —
an exact Gibbs sampler and deterministic CAVI — designed for high-dimensional
data (`p ≫ n`), with configurable working precision, optional column
standardization, and rotation–sign–permutation (RSP) post-processing of the
factor loadings.

## Features

- **Two inference engines** — exact MCMC (`GibbsMGP`) and fast mean-field CAVI
  (`CaviMGP`), both fitting the same MGP prior.
- **Overfitted truncation, sparse solutions** — choose a generous `K`;
  increasing column-wise shrinkage suppresses redundant factors.
- **Efficient JAX** — compiled chunked sweeps, exact Gamma draws, log-space
  shrinkage state, XLA command buffers on GPU.
- **per-estimator precision** — Gibbs defaults to float32 (fast); CAVI
  defaults to float64 (its absolute ELBO stopping rule is jitter-sensitive);
  each can be overridden per estimator or process-wide, without silently
  changing global state.
- **Standardization built in** — `standardize=True` matches the unit-variance
  calibration of the MGP prior.
- **Factor switching** — RSP alignment of posterior draws and of the CAVI mean
  for interpretable loading matrices.
- **Posterior summaries** — credible-interval factor counts, effective sample
  sizes, and Monte Carlo or mean-field moments of `Σ = ΛΛᵀ + Ω`.

## Install

```bash
pip install -e .                    # runtime: jax, jaxlib, numpy
pip install -e ".[plot,ess,rsp]"    # optional: matplotlib, arviz, optax
pip install -e ".[test]"            # optional: scipy (numerical checks, scipy RSP backend)
```

Optional dependencies are imported lazily. You can also simply put the
repository on `sys.path` and `import mgp` — no build step is required.

## Quick start

```python
import numpy as np
from mgp import GibbsMGP, CaviMGP

X = np.random.default_rng(0).normal(size=(200, 50))  # (n, p) observations

# Exact MCMC (default working precision: float32)
gs = GibbsMGP(K=10, n_iter=4000, burn_in=2000, seed=2025,
              standardize=True).fit(X)

gs.B_hat_          # (p, K) posterior-mean loadings (standardized scale)
gs.Sigma_          # (p, p) plug-in covariance estimate
gs.n_factors()     # active columns (marginal credible intervals)
gs.ess_summary()   # per-entry effective sample size (needs arviz)

# Fast deterministic variational approximation (default working precision: float64)
vi = CaviMGP(K=10, max_iter=1000).fit(X)
vi.elbo_           # absolute ELBO history
vi.converged_, vi.stop_reason_, vi.n_iter_
vi.Sigma_var_      # mean-field posterior variance of Σ

# Optional factor switching (RSP)
gs.align_factors()     # aligns B_samples_; updates B_hat_
vi.align_factors()     # transforms the CAVI Gaussian moments
```

## Model

For observations `x_i ∈ R^p` and `K` latent factors:

$$
x_i = \Lambda \eta_i + \varepsilon_i, \qquad
\eta_i \sim N_K(0, I_K), \qquad
\varepsilon_i \sim N_p(0, \Sigma), \quad
\Sigma = \mathrm{diag}(\sigma_1^2, \dots, \sigma_p^2).
$$

The loading matrix `Λ` (`p × K`) receives the MGP shrinkage prior

$$
\lambda_{jk} \mid \phi_{jk}, \tau_k \sim N(0, (\phi_{jk}\tau_k)^{-1}), \qquad
\phi_{jk} \sim \mathrm{Ga}(\nu/2, \nu/2), \qquad
\tau_k = \prod_{l \le k} \delta_l,
$$

with `δ₁ ~ Ga(a₁, 1)`, `δ_l ~ Ga(a₂, 1)` for `l ≥ 2`, and
`σ_j^{-2} ~ Ga(a_σ, b_σ)`. Because `a₂ > 1` makes `τ_k` stochastically
increasing, later columns are shrunk progressively harder — the mechanism
behind automatic factor selection with an overfitted `K`.

Unified defaults: `ν = 3`, `a₁ = 2.1`, `a₂ = 3.1`, `a_σ = 1`, `b_σ = 0.3`
(`a₁/a₂` follow Durante (2017); `a_σ/b_σ` are the paper's Section 4.1 values).

## Working precision

Defaults are per estimator: **`GibbsMGP` inherits the process-wide default**
(`precision=None`, initially **float32**), while **`CaviMGP` defaults to
`"float64"`**. CAVI's stopping rule compares absolute ELBO differences against
`tol`, and float32 jitter at large `|ELBO|` can dominate that difference (and
can trip the numerical-anomaly guard), so double precision is the CAVI default;
float32 is opt-in for speed. Passing `precision=None` to either estimator
follows the process-wide default.

```python
GibbsMGP(K=10)                          # inherits mgp.get_precision() → "float32"
GibbsMGP(K=10, precision="float64")     # override for this estimator only
CaviMGP(K=10)                           # float64 (double precision) by default
CaviMGP(K=10, precision="float32")      # opt into the faster, jitter-prone path
CaviMGP(K=10, precision=None)           # follow the process-wide default
import mgp
mgp.set_precision("float64")            # process-wide default for precision=None fits
mgp.get_precision()                     # -> "float64"
```

- `fit()` applies the resolved precision only for the duration of the call and
  restores the previous global value afterwards (also on failure). The resolved
  value is stored in `precision_used_`.
- Fitted arrays are stored in the resolved precision; `align_factors()` and
  `sample_factors()` follow it automatically.
- float32 is the native, faster regime (especially on GPU) and remains the
  default for Gibbs. CAVI already defaults to float64; explicit float32 CAVI at
  large `|ELBO|` can stop early on ELBO jitter.
- JAX's x64 mode is enabled at import (needed for float64; harmless for float32).

## Standardization

The MGP prior is calibrated for **unit-variance columns** (fixed unit loading
scale, `b_σ = 0.3`); `MGP_CAVI.md` states the same requirement. Use
`standardize=True` unless your columns already have unit sample variance:

```python
gs = GibbsMGP(K=10, standardize=True).fit(X)
gs.x_offset_   # (p,) column means
gs.x_scale_    # (p,) sample standard deviations (ddof=1)
```

- `standardize=False` (default) only centers the data and warns when column
  standard deviations deviate strongly from 1 (max > 10 or min < 0.1).
- `standardize=True` centers and scales each column (`ddof=1`; zero-variance
  columns keep scale 1) and requires `center=True`.
- With `standardize=True` **all fitted quantities are in the standardized
  coordinate system**. Back-transform with

  ```python
  B_orig      = gs.x_scale_[:, None] * gs.B_hat_                  # (p, K)
  sigma2_orig = gs.x_scale_ ** 2 * gs.sigma2_hat_                 # (p,)
  Sigma_orig  = np.diag(gs.x_scale_) @ gs.Sigma_ @ np.diag(gs.x_scale_)
  ```

  and analogously for `B_samples_`, `B_aligned_`, and posterior-moment arrays.

## API overview

| | `GibbsMGP` | `CaviMGP` |
| --- | --- | --- |
| Inference | exact Gibbs sampling | mean-field coordinate-ascent VI |
| Point estimate | posterior mean of draws | variational mean `mu_` |
| Key outputs | `B_samples_`, `sigma2_samples_`, `eta_last_`, `D_last_`, `ess()` | `mu_`, `V_`, `W_`, `Phi_`, `elbo_`, `T_` |
| Cost | slower, exact | fast, approximate |

Both estimators share `B_hat_`, `sigma2_hat_`, `Sigma_`, `n_factors()`,
`active_entries()`, `x_offset_`, `x_scale_`, `precision_used_`,
`raw_output_`, `align_factors()`, and `fit(X)` returning `self`.

- **`n_factors()`** counts columns with at least one loading whose marginal
  credible interval excludes zero (empirical HPD/quantile for Gibbs, closed-form
  Gaussian for CAVI). This is *not* the paper's Section 3.2 rule, and MGP factor
  splitting can make the count exceed the true number of factors.
- **Gibbs** additionally offers `Sigma_mc_`/`Sigma_mean_`, `Sigma_var_`,
  `Sigma_diag_var_`, `cov_Sigma(pairs)`, and on-demand
  `sample_factors(X, seed=..., return_eta=True)`.
- **CAVI** additionally offers `Sigma_mean_`, `Sigma_var_`, `cov_Sigma(pairs)`,
  `plot_elbo()`, and `n_factors(method="linf", threshold=0.1)`. The latter
  counts columns whose loading-mean infinity norm `max_j(abs(mu_[j, k]))`
  is strictly greater than the threshold; equality is inactive. It uses
  `mu_aligned_` with `use_aligned=True` after `align_factors()`. This rule
  ignores posterior uncertainty and uses the fitted loading scale
  (standardized when `standardize=True`); rotation can change the count.
  `n_factors()` retains the credible-interval method (`method="credible"`),
  and `active_entries()` continues to use credible intervals.

## Factor switching (RSP)

MCMC draws and CAVI means are only identified up to orthogonal transformations.
RSP post-processing chooses a common orientation (varimax rotation followed by
sign/permutation alignment) for display and comparison:

```python
gs.align_factors()          # aligns B_samples_; updates B_hat_
gs.align_factors(metadata={"chain": "c0", "seed": 2025})   # optional bookkeeping
gs.rsp_result_              # losses, signs, permutations, transforms, diagnostics

vi.align_factors()          # transforms the CAVI Gaussian moments
vi.mu_aligned_, vi.W_aligned_, vi.V_aligned_, vi.Phi_aligned_
```

`align_factors(metadata={...})` attaches hashable scalar metadata (chain name,
seed, model settings, ...) to `rsp_result_.metadata`.

Alignment does not make the model identifiable — it selects a representation.
A run ending with RSP `status_code == 4` raises `FloatingPointError` with no
estimator state change (`exc.rsp_result` carries the diagnostics); only an
explicitly supplied non-finite `reference=` is a caller `ValueError`. With
`assignment_backend="auto"`, RSP alignment for `K > 7` needs `optax`
(`pip install -e ".[rsp]"`).

## Repository layout

```text
mgp/                 package: estimators and inference engines (_gs, _vi, _rsp)
examples/            demo1.ipynb, demo2.ipynb (executed notebooks)
MGP_CAVI.md          mean-field CAVI specification note
Factor_Switching.md  rotation-sign-permutation alignment note
pyproject.toml       packaging metadata (version 0.2.1)
```

## Scope and limitations

- **Step 6 of the paper** (Metropolis–Hastings updates of `a₁`, `a₂`) is not
  implemented: the conditionals are proper, but at large `p`/overfitted `K`
  their normalizing constants overflow floating point, making the MH chain
  numerically degenerate. `a₁`, `a₂` are fixed hyperparameters.
- **Adaptive truncation (paper §3.2)** is not implemented; `K` is a fixed
  user-chosen truncation level, and the effective factor count is read off
  post hoc.
- This is research software: numerical behavior is documented in the module
  docstrings and the algorithm notes in [`MGP_CAVI.md`](MGP_CAVI.md).

## Citation

If you use this package in academic work, please cite the MGP prior paper.
If you use the CAVI implementation, please also cite the variational inference
paper.

```bibtex
@article{bhattacharya2011sparse,
  title   = {Sparse {B}ayesian infinite factor models},
  author  = {Bhattacharya, Anirban and Dunson, David B.},
  journal = {Biometrika},
  volume  = {98},
  number  = {2},
  pages   = {291--306},
  year    = {2011},
  doi     = {10.1093/biomet/asr013}
}
```

```bibtex
@article{hansen2025fast,
  title   = {Fast Variational Inference for {B}ayesian Factor Analysis in
             Single and Multi-Study Settings},
  author  = {Hansen, Blake and Avalos-Pacheco, Alejandra and
             Russo, Massimiliano and De Vito, Roberta},
  journal = {Journal of Computational and Graphical Statistics},
  volume  = {34},
  number  = {1},
  pages   = {96--108},
  year    = {2025},
  doi     = {10.1080/10618600.2024.2356173}
}
```

## License

This package is distributed under the **GNU General Public License, version 2
only** (GPL-2.0-only) — see [`LICENSE`](LICENSE).

## Acknowledgements

The Gibbs sampler follows Bhattacharya & Dunson (2011); the CAVI algorithm
follows Hansen et al. (2025). The algorithm notes live in
[`MGP_CAVI.md`](MGP_CAVI.md) and [`Factor_Switching.md`](Factor_Switching.md).
