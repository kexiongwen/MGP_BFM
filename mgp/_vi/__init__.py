"""CAVI for the MGP-shrunk Bayesian factor model
(working precision; float64 through :class:`mgp.CaviMGP` by default).

JAX implementation of the single-study CAVI algorithm of MGP_CAVI.md:
Bayesian factor analysis with the multiplicative gamma process (MGP)
shrinkage prior of Bhattacharya & Dunson (2011), the mean-field family
and optimal variational posteriors of Section 3.1, and the closed-form
ELBO of Section 3.3.

All modules compute in the package working precision: float32 as the
process-wide default, switchable with :func:`mgp.set_precision`; the
:class:`mgp.CaviMGP` estimator defaults to float64 and applies it for the
duration of ``fit``.  JAX's x64 mode is enabled by the top-level mgp
import.  The sweep is deterministic -- no RNG anywhere;
initialization is a deterministic thin SVD.  The likelihood-side blocks
eta / B / sigma2 are standard Gaussian updates; all MGP-specific work
lives in shrinkage.py (local omega_jk + coupled global delta_l Gamma
factors, cumulative products carried in log space) and in the ELBO's
shrinkage block.

Public API
----------
CAVI           : full CAVI driver (chunked lax.while_loop; optional
                 XLA command buffers on GPU)
PPCA           : deterministic thin-SVD probabilistic-PCA initialization
elbo           : standalone closed-form ELBO at a saved state (diagnostic)
elbo_const     : constant term of the closed-form ELBO (host, once per run)
plot_elbo      : ELBO trajectory plot (lazy matplotlib import)
initialization : build the self-consistent initial variational state

Runtime dependencies: JAX/jaxlib and NumPy only.
"""

from .VI import CAVI, PPCA, elbo, elbo_const, plot_elbo, initialization

__all__ = ["CAVI", "PPCA", "elbo", "elbo_const", "plot_elbo", "initialization"]
