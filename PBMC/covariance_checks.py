from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mgp._posterior import cov_sigma_pairs, montecarlo_cov_sigma_pairs

MARKER_PAIRS = (
    ("IL7R", "CD3D"),
    ("CD79A", "CD79B"),
    ("NKG7", "GNLY"),
    ("GZMB", "PRF1"),
    ("S100A8", "S100A9"),
    ("FCGR3A", "LST1"),
    ("C1QA", "C1QB"),
    ("PPBP", "PF4"),
    ("ISG15", "IFI6"),
    ("LYZ", "S100A8"),
)


def _covariance_diagnostics(cavi_model, gibbs_model, n, a_sigma):
    with np.load(cavi_model, allow_pickle=False) as z:
        mu = z["mu"]
        V = z["V"]
        beta = z["beta"]
    alpha_n = a_sigma + n / 2.0
    cavi_diag = (
        np.sum(mu * mu, axis=1)
        + np.diagonal(V, axis1=1, axis2=2).sum(axis=1)
        + beta / (alpha_n - 1.0)
    )
    with np.load(gibbs_model, allow_pickle=False) as z:
        diagnostics = json.loads(z["diagnostics_json"].item())["diagnostics"]
    gibbs_diag = np.asarray(diagnostics["sampling"]["covariance_diag_mean"])
    if cavi_diag.shape != gibbs_diag.shape:
        raise ValueError("covariance diagonal shapes differ")
    difference = cavi_diag - gibbs_diag
    return {
        "pearson_correlation": float(np.corrcoef(cavi_diag, gibbs_diag)[0, 1]),
        "relative_l2_difference": float(np.linalg.norm(difference) / np.linalg.norm(gibbs_diag)),
        "median_absolute_relative_difference": float(np.median(
            np.abs(difference) / np.maximum(np.abs(gibbs_diag), np.finfo(np.float64).eps)
        )),
        "cavi_diagonal": cavi_diag,
        "gibbs_diagonal": gibbs_diag,
    }


def check_marker_covariances(cavi_path, gibbs_path):
    with np.load(cavi_path, allow_pickle=False) as z:
        mu = z["mu"]
        V = z["V"]
        beta = z["beta"]
        hvg_cavi = z["hvg_names"]
        config = json.loads(z["config_json"].item())
    with np.load(gibbs_path, allow_pickle=False) as z:
        B = z["B_samples"]
        sigma2 = z["sigma2_samples"]
        hvg_gibbs = z["hvg_names"]
        n_cells = len(z["cell_names"])
    if not np.array_equal(hvg_cavi, hvg_gibbs):
        raise ValueError("HVG identifiers differ between CAVI and Gibbs")
    lookup = {name: index for index, name in enumerate(hvg_cavi)}
    pairs = [(lookup[left], lookup[right]) for left, right in MARKER_PAIRS if left in lookup and right in lookup]
    names = [(left, right) for left, right in MARKER_PAIRS if left in lookup and right in lookup]
    missing = [pair for pair in MARKER_PAIRS if pair[0] not in lookup or pair[1] not in lookup]
    if not pairs:
        raise ValueError(f"no marker pairs available in the HVGs: {missing}")
    alpha_n = config["a_sigma"] + n_cells / 2.0
    variational_covariance = cov_sigma_pairs(mu, V, alpha_n, beta, [
        (pair, pair) for pair in pairs
    ])
    gibbs_covariance = montecarlo_cov_sigma_pairs(B, sigma2, [
        (pair, pair) for pair in pairs
    ])
    rows = []
    for (left, right), (i, j), cavi_var, gibbs_var in zip(names, pairs, variational_covariance, gibbs_covariance):
        cavi_mean = float(mu[i] @ mu[j])
        gibbs_mean = float(np.einsum("sk,sk->s", B[:, i, :], B[:, j, :]).mean())
        rows.append({
            "gene_i": left,
            "gene_j": right,
            "cavi_mean": cavi_mean,
            "gibbs_mean": gibbs_mean,
            "cavi_sd": float(np.sqrt(max(cavi_var, 0.0))),
            "gibbs_sd": float(np.sqrt(max(gibbs_var, 0.0))),
            "mean_difference": cavi_mean - gibbs_mean,
        })
    return rows, missing
