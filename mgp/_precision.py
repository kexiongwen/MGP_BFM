"""Working-precision configuration for the mgp package.

The GS (_gs) and VI (_vi) engines derive their computation dtypes from
the per-module ``_F64`` constants (a historical name; it is simply "the
module's working dtype").  This module flips those constants as a group
so the whole pipeline runs in a single, coherent precision.

Default: **float32** (single precision) -- the engines' log-space
shrinkage machinery was designed for the float32 range, and it is
meaningfully faster (GPU in particular).  ``set_precision("float64")``
restores the
double-precision regime and enables JAX's x64 mode (required, or JAX
silently downcasts float64 requests); it is never disabled afterwards:
float32 works fine with x64 on, and user arrays may rely on it.

Notes
-----
* The switch is process-global.  An estimator with ``precision=None``
  resolves to the current global default at fit time: ``GibbsMGP``
  follows this default (initially "float32"), while ``CaviMGP`` defaults
  to "float64" and only follows the global default when ``precision=None``
  is passed explicitly.  The resolved precision is applied for the
  duration of the operation and the previous global value is restored
  afterwards; an explicit estimator ``precision`` never permanently
  changes the process default.
* ``mgp/_vi/shrinkage.py`` materializes its log-space caps
  (``LOG_BETA_CAP`` / ``LOG_SUM_CAP``) at import; they are re-created in
  the working dtype here so no float64 scalar promotes float32 traces
  back to float64 under x64.
* Host-side accumulators that are precision-critical in the other
  direction (the ELBO host accumulation in ``_vi/VI.py``) stay float64
  regardless; host storage buffers follow the working dtype.
"""

import jax
import jax.numpy as jnp

__all__ = ["set_precision", "get_precision", "work_dtypes", "PRECISIONS"]

PRECISIONS = ("float32", "float64")

# the algorithm-side modules carrying a working-dtype constant
_MODULE_NAMES = (
    "mgp._gs.GS_MGP",
    "mgp._gs.eta_sample",
    "mgp._gs.B_sample",
    "mgp._gs.Sigma2_sample",
    "mgp._gs.Shrinkage",
    "mgp._vi.VI",
    "mgp._vi.initialization",
    "mgp._vi.shrinkage",
)

# Engines initialize their module-level working-dtype constants as
# jnp.float64; the package __init__ then calls set_precision("float32")
# (the documented default), which rewrites them.  Importing any mgp
# submodule executes that __init__ first, so this initialization value is
# never user-observable -- it only reflects the pre-flip module state.
_current = "float64"


def _jdtype(name):
    return jnp.float32 if name == "float32" else jnp.float64


def work_dtypes():
    """(jax dtype, numpy dtype) of the current working precision."""
    import numpy as np

    dt = _jdtype(_current)
    return dt, np.dtype(dt).type


def get_precision():
    """The current working precision: "float32" or "float64"."""
    return _current


def set_precision(name):
    """Set the package working precision; returns the previous one.

    Parameters
    ----------
    name : {"float32", "float64"}
        The regime every subsequent fit runs in.  "float64" additionally
        enables (and leaves enabled) JAX's x64 mode.
    """
    global _current
    if name not in PRECISIONS:
        raise ValueError(
            f"precision must be one of {PRECISIONS}, got {name!r}"
        )
    from importlib import import_module

    previous, _current = _current, name
    dt = _jdtype(name)
    if name == "float64":
        # float64 requests are silently downcast without x64; enabling
        # it is process-wide and left on (harmless for float32).
        jax.config.update("jax_enable_x64", True)
    for module_name in _MODULE_NAMES:
        module = import_module(module_name)
        module._F64 = dt
    # re-create the import-time log-space caps in the working dtype so a
    # float64 scalar cannot promote float32 traces back to float64.
    shrink = import_module("mgp._vi.shrinkage")
    shrink.LOG_BETA_CAP = jnp.asarray(jnp.log(shrink.BETA_CAP), dtype=dt)
    shrink.LOG_SUM_CAP = jnp.asarray(jnp.log(shrink.SUM_CAP), dtype=dt)
    return previous
