from __future__ import annotations

import numpy as np

ALIGNED_REPRESENTATION = "d20_norm"
REPRESENTATION_NAMES = (ALIGNED_REPRESENTATION,)


def build_representations(adata, estimator, aligned_scores, run_id, *, n_columns=20):
    scores = np.asarray(aligned_scores, dtype=np.float64)
    K = int(estimator.K)
    if scores.ndim != 2 or scores.shape[1] != K or not np.isfinite(scores).all():
        raise ValueError(f"aligned scores must be finite with shape (n, {K})")
    if hasattr(estimator, "mu_aligned_"):
        loadings = estimator.mu_aligned_
    else:
        loadings = estimator.B_hat_
    loading_norm = np.linalg.norm(np.asarray(loadings, dtype=np.float64), axis=0)
    active_mask = np.asarray(
        estimator.active_entries(level=0.95, interval="hpd", use_aligned=True),
        dtype=bool,
    )
    active_columns = active_mask.any(axis=0)
    columns_by_name = {
        "d20_norm": np.arange(min(n_columns, K), dtype=np.int64),
    }
    weighted_by_name = {"d20_norm": True}

    outputs = {}
    diagnostics = {}
    for name in REPRESENTATION_NAMES:
        columns = columns_by_name[name]
        weighted = scores * loading_norm if weighted_by_name[name] else scores
        selected = np.asarray(weighted[:, columns], dtype=np.float64)
        key = f"X_{run_id}_{name}"
        adata.obsm[key] = selected
        singular = np.linalg.svd(selected, compute_uv=False) if selected.shape[1] else np.empty(0)
        zero_variance = int(np.count_nonzero(selected.std(axis=0) == 0)) if selected.size else 0
        outputs[name] = {
            "key": key,
            "scores": selected,
            "columns": columns,
            "loading_norm": loading_norm[columns],
        }
        diagnostics[name] = {
            "key": key,
            "n_columns": int(selected.shape[1]),
            "columns": columns,
            "weighting": "loading_column_l2_norm" if weighted_by_name[name] else "none",
            "selected_loading_column_norm": loading_norm[columns],
            "selected_active_factor_count": int(active_columns[columns].sum()),
            "score_singular_values": singular,
            "zero_variance_columns": zero_variance,
            "degenerate": bool(selected.shape[1] == 0 or zero_variance == selected.shape[1]),
            "finite": bool(np.isfinite(selected).all()),
        }
    summary = {
        "active_mask": active_mask,
        "active_factor_count": int(active_columns.sum()),
        "loading_column_norm": loading_norm,
        "representations": diagnostics,
    }
    existing = adata.uns.get(run_id, {})
    existing["representation"] = summary
    adata.uns[run_id] = existing
    return outputs, summary


def representation_batch_diagnostics(representations, batch_means, batch_indices):
    from .scores import half_chain_score_diagnostics

    result = {}
    for name, representation in representations.items():
        norms = representation["loading_norm"]
        columns = representation["columns"]
        full_norms = np.zeros(batch_means.shape[-1], dtype=np.float64)
        full_norms[columns] = norms
        norms = full_norms
        result[name] = half_chain_score_diagnostics(
            batch_means,
            batch_indices,
            representation["columns"],
            norms,
        )
    return result
