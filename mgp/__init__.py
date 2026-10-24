"""mgp: unified estimators for the MGP-shrunk Bayesian factor model.

sklearn-style interface to the two JAX inference algorithms for the
multiplicative gamma process (MGP) shrinkage prior of Bhattacharya &
Dunson (2011), "Sparse Bayesian infinite factor models":

* :class:`GibbsMGP` -- the Gibbs sampler of the paper's Section 3.1
  (exact MCMC), implemented in :mod:`mgp._gs`;
* :class:`CaviMGP` -- the coordinate ascent variational inference
  specified in MGP_CAVI.md (fast approximation, closed-form ELBO),
  implemented in :mod:`mgp._vi`.

Both estimators share the hyperparameters of the prior (K, nu, a1, a2,
a_sigma, b_sigma), the preprocessing (column centering in the working
precision) and the fitted accessors B_hat_ / sigma2_hat_ / Sigma_ /
n_factors().  Both also support RSP factor-switching post-processing
through :meth:`BaseMGP.align_factors` (implemented in
:mod:`mgp._rsp`); :meth:`GibbsMGP.sample_factors` returns
on-demand factor-score draws, raw or aligned.

Runtime dependencies: JAX/jaxlib and NumPy (matplotlib only for
CaviMGP.plot_elbo, arviz only for GibbsMGP.ess / ess_summary, optax
only for RSP assignment with K > 7 -- all imported lazily).

Precision: :class:`GibbsMGP` takes ``precision=None`` by default and
therefore inherits the process-wide default (:func:`get_precision`,
initially "float32") at :meth:`~mgp.base.BaseMGP.fit` time, while
:class:`CaviMGP` defaults to **double precision** ("float64") because
its absolute ELBO stopping criterion is sensitive to float32 jitter.
Passing ``precision=None`` to either estimator follows the process-wide
default.  An explicit "float32"/"float64" overrides the default for that
estimator only and does not permanently change the global one.
:func:`set_precision` sets the process-wide default.  JAX's x64 mode is
enabled at import and left on: it is required for the float64 regime and
harmless for float32.

Standardization: the MGP prior is calibrated for unit-variance columns.
Estimators accept ``standardize=True`` (scale stored in ``x_scale_``)
and warn when unstandardized columns deviate strongly from unit scale.
"""

import jax as _jax

# x64 stays enabled process-wide: the float64 regime needs it (JAX
# silently downcasts float64 requests otherwise) and float32 is
# unaffected by it.  The working dtype itself lives in the per-module
# constants flipped by mgp._precision.set_precision.
_jax.config.update("jax_enable_x64", True)

from .base import BaseMGP
from .cavi import CaviMGP
from .gibbs import GibbsMGP
from ._precision import get_precision, set_precision

# Package default: single precision (see module docstring and
# mgp._precision).  set_precision("float64") restores double precision.
set_precision("float32")

__version__ = "0.2.1"

__all__ = [
    "BaseMGP",
    "GibbsMGP",
    "CaviMGP",
    "set_precision",
    "get_precision",
]
