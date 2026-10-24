"""Rotation-Sign-Permutation (RSP) alignment of posterior loadings.

Implements the RSP post-processing described in ``Factor_Switching.md``:
a per-draw varimax rotation followed by recursive sign/permutation
alignment to a common reference matrix.  Default precision is float32;
precision="float64" enables double precision.

Public API
----------
rsp_exact       : full RSP alignment (varimax + SP/RLME alternation)
transform_scores: apply per-draw orthogonal transforms to factor scores
from_flat       : parse flat (T, p*q) LambdaV layouts into (T, p, q)
to_flat         : (T, p, q) -> (T, p*q) flat layout
lambda_colnames : LambdaV column names for a (p, q) loading matrix
RSPResult       : result container (aligned draws, signs, permutations,
                  losses, transforms, diagnostics)

optax (assignment problems with q > 7) and scipy (the "scipy" assignment
backend) are imported lazily inside the functions that need them.
"""
from .api import rsp_exact
from .chains import transform_scores
from .io import from_flat, lambda_colnames, to_flat
from .types import RSPResult

__all__ = [
    "rsp_exact",
    "transform_scores",
    "from_flat",
    "to_flat",
    "lambda_colnames",
    "RSPResult",
]
