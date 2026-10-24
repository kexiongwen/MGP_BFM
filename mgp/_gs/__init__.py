"""
Gibbs sampler for the MGP-shrunk Bayesian factor model
(working precision; float32 by default)

JAX implementation of the Gibbs sampler for the Bayesian infinite factor
model with the multiplicative gamma process (MGP) shrinkage prior:
Bhattacharya & Dunson (2011), "Sparse Bayesian infinite factor models",
Biometrika 98(2), 291-306, Section 3.1 (fixed truncation level).

All modules compute in the package working precision: float32 by default,
switchable process-wide with :func:`mgp.set_precision` or per estimator
through its ``precision`` parameter.  The log-space shrinkage machinery was
designed for float32's narrower range and remains active and correct in
either precision.  All sampler functions take an explicit
jax.random.PRNGKey as first argument.  The driver GS_MGP returns posterior
draws as host-side numpy arrays in the working precision.

Public API
----------
GS_MGP        : full Gibbs sampler driver (chunked jax.lax.scan chain)
Initialization: principal-factor initialization for (B, sigma2)
shrinkage    : MGP shrinkage draw given B -- phi [step 4], delta
               [step 5] -> D = 1/sqrt(phi tau); (a1, a2) are fixed
               input hyperparameters (the paper's Step 6 MH update of
               them is deliberately omitted -- see GS_MGP's docstring)
phi_sample    : local-shrinkage draw phi | B, tau
delta_sample  : column-shrinkage draw delta | B, phi (log-space state)
prior_scales  : deterministic (phi, log delta) -> prior stds D
eta_sample    : latent-factor draw eta | B, Omega, X
B_sample      : loading-matrix draw B | eta, D, sigma2, X
sigma2_sample : idiosyncratic-variance draw sigma2 | B, eta, X
ess, ess_summary : effective-sample-size diagnostics of the kept draws
               (ArviZ, optional dependency)

Design notes: the eta / B / sigma2 steps are standard Gaussian updates; the
MGP-specific work lives in Shrinkage.py.  There, tau_h = prod_{l<=h} delta_l
is a cumulative product that overflows float32, while the quantities the
delta-rates need (sum_{l>=h} tau_l S_l) remain O(p): the state is carried as
log delta and every rate is assembled with suffix-logsumexp / logaddexp
machinery (see Shrinkage.py's module docstring).  No chi-square Gamma
approximation is offered: exact jax.random.gamma is capturable on every
backend, so the approximation is unnecessary.
"""

from .eta_sample import eta_sample
from .Sigma2_sample import sigma2_sample
from .Shrinkage import delta_sample, phi_sample, prior_scales, shrinkage
from .B_sample import B_sample
from .GS_MGP import GS_MGP, Initialization
from .ESS import ess, ess_summary

__all__ = [
    "GS_MGP",
    "Initialization",
    "shrinkage",
    "phi_sample",
    "delta_sample",
    "prior_scales",
    "eta_sample",
    "B_sample",
    "sigma2_sample",
    "ess",
    "ess_summary",
]
