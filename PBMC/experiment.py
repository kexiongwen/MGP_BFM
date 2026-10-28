from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import arviz as az
import jax
import numpy as np

from mgp import CaviMGP, GibbsMGP

from .data_processing import DATA_DIR, jsonable, prepare_model_input, software_versions
from .downstream import DownstreamConfig, run_representation_suite, write_top_markers
from .representations import build_representations, representation_batch_diagnostics
from .scores import conditional_factor_mean

_MARKER_CANDIDATES = (
    "IL7R", "CD79A", "MS4A1", "CD8A", "CD8B", "LYZ", "CD14",
    "GNLY", "NKG7", "FCGR3A", "MS4A7", "FCER1A", "CST3", "PPBP",
)


@dataclass
class ModelRun:
    prepared: object
    estimator: object
    config: object
    run_id: str
    representations: dict
    downstream: dict
    diagnostics: dict
    pilot: bool = False
    starts: dict | None = None


def subset_prepared(prepared, n_cells, n_genes):
    if not 3 <= n_cells <= prepared.X.shape[0]:
        raise ValueError("n_cells must be in [3, n]")
    if not 3 <= n_genes <= prepared.X.shape[1]:
        raise ValueError("n_genes must be in [3, p]")
    prepared.adata = prepared.adata[:n_cells].copy()
    prepared.X = np.ascontiguousarray(prepared.X[:n_cells, :n_genes])
    prepared.hvg_names = prepared.hvg_names[:n_genes]
    return prepared


def estimator_kwargs(config):
    return {
        "K": config.K,
        "nu": config.nu,
        "a1": config.a1,
        "a2": config.a2,
        "a_sigma": config.a_sigma,
        "b_sigma": config.b_sigma,
        "center": False,
        "precision": config.precision,
        "standardize": config.standardize,
    }


def run_stem(config, method, pilot=False):
    stem = f"{method}_{config.input_tag}_k{config.K}"
    if method == "gibbs":
        stem += f"_n{config.n_iter}_b{config.burn_in}_t{config.thin}"
    stem += "_d20"
    return stem + ("_pilot" if pilot else "")


def _fit_cavi_diagnostics(estimator, wall_sec):
    elbo = np.asarray(estimator.elbo_, dtype=np.float64)
    delta = np.diff(elbo)
    return {
        "n_iter": int(estimator.n_iter_),
        "converged": bool(estimator.converged_),
        "stop_reason": str(estimator.stop_reason_),
        "final_elbo": float(elbo[-1]),
        "final_delta": float(delta[-1]) if len(delta) else 0.0,
        "negative_delta_count": int((delta < 0).sum()),
        "max_drop": float(max(0.0, -delta.min())) if len(delta) else 0.0,
        "compile_sec": float(estimator.compile_sec_),
        "runtime_sec": float(estimator.runtime_sec_),
        "fit_wall_sec": float(wall_sec),
        "execution_mode": str(estimator.execution_mode_),
        "backend": str(estimator.backend_),
        "device": str(estimator.device_),
        "precision": str(estimator.precision_used_),
    }


def _rsp_diagnostics(estimator, warning_messages, wall_sec):
    rsp = estimator.rsp_result_
    n_iter = int(rsp.n_iter)
    return {
        "status_code": int(rsp.status_code),
        "converged": bool(rsp.converged),
        "n_iter": n_iter,
        "stop_reason": str(rsp.diagnostics["stop_reason"]),
        "nonfinite_fields": list(rsp.diagnostics["nonfinite_fields"]),
        "loss_initial": float(rsp.loss_initial),
        "loss_final": float(rsp.loss_final),
        "loss_history": np.asarray(rsp.loss_history[:n_iter]).copy(),
        "loss_history_post": np.asarray(rsp.loss_history_post[:n_iter]).copy(),
        "varimax_n_iter": np.asarray(rsp.varimax_n_iter).copy(),
        "varimax_svd_fallbacks": int(rsp.diagnostics["varimax_svd_fallbacks"]),
        "assignment_backend": str(rsp.diagnostics["assignment_backend"]),
        "wall_sec": float(wall_sec),
        "warnings": warning_messages,
    }


def _align(estimator):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        estimator.align_factors()
        wall_sec = time.perf_counter() - start
    return _rsp_diagnostics(
        estimator, [str(item.message) for item in caught], wall_sec
    )


def _cavi_alignment_diagnostics(estimator):
    transform = np.asarray(estimator.rsp_result_.transforms[0])
    K = estimator.K
    max_error = 0.0
    max_scale = 0.0
    for start in range(0, estimator.W_.shape[0], 128):
        stop = min(start + 128, estimator.W_.shape[0])
        raw = estimator.mu_ @ estimator.W_[start:stop].T
        aligned = estimator.mu_aligned_ @ estimator.W_aligned_[start:stop].T
        max_error = max(max_error, float(np.max(np.abs(raw - aligned))))
        max_scale = max(max_scale, float(np.max(np.abs(raw))))
    return {
        "orthogonality_error": float(np.linalg.norm(transform.T @ transform - np.eye(K))),
        "score_transform_max_error": float(np.max(np.abs(
            estimator.W_aligned_ - estimator.W_ @ transform
        ))),
        "common_component_max_error": max_error,
        "common_component_max_scale": max_scale,
        "common_component_relative_error": max_error / max(max_scale, np.finfo(np.float64).eps),
    }


def run_cavi(prepared, config, *, pilot=False, downstream_enabled=True, warm_start=None):
    prepared = prepare_model_input(prepared, config.input_mode)
    run_id = run_stem(config, "cavi", pilot)
    estimator = CaviMGP(
        **estimator_kwargs(config),
        tol=config.tol,
        max_iter=config.max_iter,
        drop_budget=config.drop_budget,
        jit=True,
        verbose=False,
        initial_state=warm_start,
    )
    start = time.perf_counter()
    estimator.fit(prepared.X)
    wall_sec = time.perf_counter() - start
    fit = _fit_cavi_diagnostics(estimator, wall_sec)
    fit["start_index"] = 0
    fit["initialization"] = (
        "external_warm_start" if warm_start is not None else "deterministic_ppca"
    )
    fit["seed"] = None
    fit["initial_state_path"] = None
    starts = {
        "start_0": {
            "fit": fit,
            "elbo": np.asarray(estimator.elbo_, dtype=np.float64),
        }
    }
    print(
        f"[{run_id}] start=0 sweeps={fit['n_iter']} "
        f"ELBO={fit['final_elbo']:.6f} stop={fit['stop_reason']}",
        flush=True,
    )
    rsp = _align(estimator)
    representations, representation_summary = build_representations(
        prepared.adata,
        estimator,
        estimator.W_aligned_,
        run_id,
        n_columns=config.n_columns,
    )
    downstream = {}
    if downstream_enabled:
        downstream = run_representation_suite(
            prepared.adata,
            run_id,
            representations,
            DownstreamConfig(
                n_neighbors=config.n_neighbors,
                leiden_resolution=config.leiden_resolution,
                random_state=config.random_state,
            ),
        )
    diagnostics = {
        "selected_start": 0,
        "selection_rule": "single deterministic start or explicit warm-start continuation",
        "fit": fit,
        "rsp": rsp,
        "alignment": _cavi_alignment_diagnostics(estimator),
        "representation": representation_summary,
        "downstream": downstream,
        "pipeline_complete": bool(not pilot and estimator.converged_ and rsp["status_code"] == 1),
    }
    prepared.adata.uns[run_id] = {
        "config_json": json.dumps(jsonable(asdict(config)), sort_keys=True),
        "pilot": pilot,
        "input_mode": config.input_mode,
        "source": prepared.source,
        "selected_start": 0,
        "diagnostics": diagnostics,
        "representation": representation_summary,
        "downstream": downstream,
        "software": software_versions(),
    }
    return ModelRun(
        prepared, estimator, config, run_id, representations, downstream,
        diagnostics, pilot, starts,
    )


def _ess_summary(values, n_draws):
    return {
        key: {
            "min": float(array.min()),
            "median": float(np.median(array)),
            "mean": float(np.mean(array)),
            "max": float(array.max()),
            "n_draws": int(n_draws),
        }
        for key, array in values.items()
    }


def _sampling_diagnostics(estimator, hvg_names, active_mask):
    B = estimator.B_samples_
    sigma2 = estimator.sigma2_samples_
    S = B.shape[0]
    bulk = estimator.ess(method="bulk")
    tail = estimator.ess(method="tail", prob=0.05)
    aligned = np.asarray(estimator.B_aligned_)
    mcse_B = np.asarray(az.mcse(aligned[None], method="mean"))
    covariance_diag = np.einsum("spk,spk->sp", B, B) + sigma2
    covariance_mean = covariance_diag.mean(axis=0)
    mcse_covariance = np.asarray(az.mcse(covariance_diag[None], method="mean"))
    mcse_sigma2 = np.asarray(az.mcse(sigma2[None], method="mean"))
    marker_indices = np.array(
        [i for i, name in enumerate(hvg_names) if name in _MARKER_CANDIDATES],
        dtype=np.int64,
    )
    bulk_B_aligned = np.asarray(az.ess(aligned[None], method="bulk"))
    active_bulk = bulk_B_aligned[active_mask]
    active_summary = {
        "count": int(active_bulk.size),
        "median": float(np.median(active_bulk)) if active_bulk.size else None,
        "below_400_count": int(np.count_nonzero(active_bulk < 400)),
        "below_400_fraction": float(np.mean(active_bulk < 400)) if active_bulk.size else 1.0,
        "coverage_at_400": float(np.mean(active_bulk >= 400)) if active_bulk.size else 0.0,
    }
    marker_mcse_relative = (
        mcse_covariance[marker_indices] / np.maximum(np.abs(covariance_mean[marker_indices]), np.finfo(np.float64).tiny)
        if len(marker_indices) else np.empty(0)
    )
    near_constant = np.std(B, axis=0) <= np.finfo(B.dtype).eps
    diagnostics = {
        "bulk_summary": _ess_summary(bulk, S),
        "tail_summary": _ess_summary(tail, S),
        "bulk_B": bulk["B"],
        "bulk_sigma2": bulk["sigma2"],
        "tail_B": tail["B"],
        "tail_sigma2": tail["sigma2"],
        "active_loading_bulk_ess": active_summary,
        "mcse_B_aligned": mcse_B,
        "mcse_sigma2": mcse_sigma2,
        "mcse_covariance_diag": mcse_covariance,
        "covariance_diag_mean": covariance_mean,
        "marker_indices": marker_indices,
        "marker_names": hvg_names[marker_indices],
        "marker_covariance_mcse_relative": marker_mcse_relative,
        "marker_covariance_mcse_relative_max": float(marker_mcse_relative.max()) if marker_mcse_relative.size else None,
        "near_constant_count": int(near_constant.sum()),
    }
    diagnostics["review_required"] = bool(
        active_summary["coverage_at_400"] < 0.9
        or (marker_mcse_relative.size and np.max(marker_mcse_relative) > 0.01)
        or not np.isfinite(mcse_covariance).all()
    )
    return diagnostics


def _log_stage(run_id, message):
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 1e9
        memory = f", RSS={rss:.2f} GB"
    except ImportError:
        memory = ""
    print(f"[{run_id}] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}{memory}", flush=True)


def save_gibbs_chain_checkpoint(
    estimator, config, fit_wall_sec, baseline_sha256, path, *, input_sha256=None
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        B_samples=estimator.B_samples_,
        sigma2_samples=estimator.sigma2_samples_,
        x_offset=estimator.x_offset_,
        x_scale=estimator.x_scale_,
        eta_last=estimator.eta_last_,
        D_last=estimator.D_last_,
        seed=estimator.seed_,
        runtime_sec=estimator.runtime_sec_,
        fit_wall_sec=fit_wall_sec,
        config_json=json.dumps(jsonable(asdict(config)), sort_keys=True),
        baseline_sha256=baseline_sha256,
        input_sha256=input_sha256 or baseline_sha256,
    )


def load_gibbs_chain_checkpoint(prepared, config, path):
    path = Path(path)
    estimator = GibbsMGP(
        **estimator_kwargs(config),
        n_iter=config.n_iter,
        burn_in=config.burn_in,
        thin=config.thin,
        seed=config.seed,
        graph=False,
        verbose=False,
    )
    with np.load(path, allow_pickle=False) as z:
        stored_config = json.loads(z["config_json"].item())
        stored_config.setdefault("representation", None)
        if stored_config != jsonable(asdict(config)):
            raise ValueError("chain checkpoint configuration does not match the requested run")
        if str(z["baseline_sha256"]) != prepared.source["baseline_sha256"]:
            raise ValueError("chain checkpoint was produced from a different baseline")
        stored_input = str(z["input_sha256"]) if "input_sha256" in z.files else str(z["baseline_sha256"])
        expected_input = prepared.source.get("model_input", {}).get(
            "sha256", prepared.source["baseline_sha256"]
        )
        if stored_input != expected_input:
            raise ValueError("chain checkpoint was produced from a different model input")
        B = z["B_samples"].copy()
        sigma2 = z["sigma2_samples"].copy()
        offset = z["x_offset"].copy()
        scale = z["x_scale"].copy()
        eta_last = z["eta_last"].copy()
        D_last = z["D_last"].copy()
        seed = int(z["seed"])
        runtime_sec = float(z["runtime_sec"])
        fit_wall_sec = float(z["fit_wall_sec"])
    n, p = prepared.X.shape
    expected = (config.n_kept, p, config.K)
    if B.shape != expected or sigma2.shape != (config.n_kept, p):
        raise ValueError(f"chain shapes must be {expected} and {(config.n_kept, p)}")
    if B.dtype != np.dtype(config.precision) or sigma2.dtype != np.dtype(config.precision):
        raise ValueError("chain precision does not match the requested run")
    if offset.shape != (p,) or scale.shape != (p,):
        raise ValueError("invalid preprocessing arrays in chain checkpoint")
    if not np.isfinite(B).all() or not np.isfinite(sigma2).all() or np.any(sigma2 <= 0):
        raise ValueError("chain checkpoint contains invalid draws")
    estimator.B_samples_ = B
    estimator.sigma2_samples_ = sigma2
    estimator.eta_last_ = eta_last
    estimator.D_last_ = D_last
    estimator.seed_ = seed
    estimator.runtime_sec_ = runtime_sec
    estimator.x_offset_ = offset
    estimator.x_scale_ = scale
    estimator.n_samples_, estimator.n_features_in_ = n, p
    estimator.precision_used_ = config.precision
    estimator._B_hat = B.mean(axis=0)
    estimator._sigma2_hat = sigma2.mean(axis=0)
    estimator._sigma_cache = {}
    estimator.Sigma_draws_used_ = None
    estimator.raw_output_ = {"D_last": D_last}
    return estimator, fit_wall_sec


def run_gibbs(
    prepared,
    config,
    *,
    pilot=False,
    downstream_enabled=True,
    chain_path=None,
    reuse_chain=False,
    checkpoint_overwrite=False,
):
    prepared = prepare_model_input(prepared, config.input_mode)
    run_id = run_stem(config, "gibbs", pilot)
    chain_path = Path(chain_path) if chain_path is not None else DATA_DIR / f"{run_id}_chain.npz"
    if reuse_chain:
        estimator, fit_wall_sec = load_gibbs_chain_checkpoint(prepared, config, chain_path)
        _log_stage(run_id, f"loaded completed chain from {chain_path}; skipping sampling")
    else:
        if chain_path.exists() and not checkpoint_overwrite:
            raise FileExistsError(f"chain checkpoint exists: {chain_path}")
        estimator = GibbsMGP(
            **estimator_kwargs(config),
            n_iter=config.n_iter,
            burn_in=config.burn_in,
            thin=config.thin,
            seed=config.seed,
            graph=False,
            verbose=True,
        )
        _log_stage(run_id, f"starting {config.n_iter} Gibbs sweeps")
        start = time.perf_counter()
        estimator.fit(prepared.X)
        fit_wall_sec = time.perf_counter() - start
        _log_stage(run_id, f"sampling finished in {fit_wall_sec:.1f}s; saving chain checkpoint")
        save_gibbs_chain_checkpoint(
            estimator, config, fit_wall_sec,
            prepared.source["baseline_sha256"], chain_path,
            input_sha256=prepared.source.get("model_input", {}).get("sha256"),
        )
        _log_stage(run_id, "chain checkpoint saved; starting RSP")
    rsp = _align(estimator)
    _log_stage(run_id, f"RSP finished with status={rsp['status_code']} in {rsp['wall_sec']:.1f}s")

    start = time.perf_counter()
    aligned_scores, batch_means, batch_indices = conditional_factor_mean(
        prepared.X,
        estimator.B_samples_,
        estimator.sigma2_samples_,
        estimator.x_offset_,
        scale=estimator.x_scale_,
        transforms=estimator.rsp_result_.transforms,
        batch_size=config.batch_size,
    )
    score_wall_sec = time.perf_counter() - start
    _log_stage(run_id, f"conditional means finished in {score_wall_sec:.1f}s; building representations")
    representations, representation_summary = build_representations(
        prepared.adata,
        estimator,
        aligned_scores,
        run_id,
        n_columns=config.n_columns,
    )
    score_stability = representation_batch_diagnostics(
        representations, batch_means, batch_indices
    )
    _log_stage(run_id, f"representations finished; active factors={representation_summary['active_factor_count']}; starting ESS/MCSE")
    S = estimator.B_samples_.shape[0]
    fit = {
        "n_iter": int(config.n_iter),
        "burn_in": int(config.burn_in),
        "thin": int(config.thin),
        "n_kept": int(S),
        "retained_sweeps": config.burn_in + config.thin * np.arange(1, S + 1),
        "seed": int(estimator.seed_),
        "runtime_sec": float(estimator.runtime_sec_),
        "fit_wall_sec": float(fit_wall_sec),
        "score_wall_sec": float(score_wall_sec),
        "backend": str(jax.default_backend()),
        "precision": str(estimator.precision_used_),
        "gamma_mode": "exact",
    }
    sampling = _sampling_diagnostics(
        estimator, prepared.hvg_names, representation_summary["active_mask"]
    )
    sampling["score_stability_thresholds"] = {"relative_l2_median": 0.02, "relative_l2_max": 0.05}
    sampling["review_required"] = bool(
        sampling["review_required"] or any(
            values["relative_l2_median"] > 0.02 or values["relative_l2_max"] > 0.05
            for values in score_stability.values()
        )
    )
    _log_stage(run_id, f"ESS/MCSE finished; review_required={sampling['review_required']}; starting downstream")
    downstream = {}
    if downstream_enabled:
        downstream = run_representation_suite(
            prepared.adata,
            run_id,
            representations,
            DownstreamConfig(
                n_neighbors=config.n_neighbors,
                leiden_resolution=config.leiden_resolution,
                random_state=config.random_state,
            ),
        )
    _log_stage(run_id, "downstream finished; assembling outputs")
    diagnostics = {
        "fit": fit,
        "sampling": sampling,
        "rsp": rsp,
        "representation": representation_summary,
        "score_stability": score_stability,
        "downstream": downstream,
        "conditional_mean": {
            "batch_size": int(config.batch_size),
            "n_batches": int(batch_indices.shape[0]),
            "working_dtype": str(estimator.B_samples_.dtype),
            "accumulation_dtype": "float64",
            "score_dtype": str(aligned_scores.dtype),
            "full_score_array_allocated": False,
            "aligned": True,
        },
        "pipeline_complete": bool(not pilot and rsp["status_code"] == 1),
    }
    prepared.adata.uns[run_id] = {
        "config_json": json.dumps(jsonable(asdict(config)), sort_keys=True),
        "pilot": pilot,
        "input_mode": config.input_mode,
        "source": prepared.source,
        "diagnostics": diagnostics,
        "representation": representation_summary,
        "downstream": downstream,
        "software": software_versions(),
    }
    result = ModelRun(
        prepared, estimator, config, run_id, representations, downstream,
        diagnostics, pilot,
    )
    result.batch_means = batch_means
    result.batch_indices = batch_indices
    result.aligned_scores = aligned_scores
    return result


def _check_output_paths(paths, overwrite):
    existing = [str(path) for path in paths if Path(path).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output files already exist; choose a new run_id/output directory "
            f"or explicitly enable overwrite: {existing}"
        )


def _save_common(result, arrays):
    for name, representation in result.representations.items():
        arrays[f"score_{name}"] = representation["scores"]
        arrays[f"columns_{name}"] = representation["columns"]
    arrays.update({
        "hvg_names": result.prepared.hvg_names,
        "cell_names": result.prepared.cell_names,
        "x_offset": result.estimator.x_offset_,
        "x_scale": result.estimator.x_scale_,
        "active_mask": result.diagnostics["representation"]["active_mask"],
        "loading_column_norm": result.diagnostics["representation"]["loading_column_norm"],
        "config_json": json.dumps(jsonable(asdict(result.config)), sort_keys=True),
        "diagnostics_json": json.dumps(jsonable({
            "run_id": result.run_id,
            "pilot": result.pilot,
            "config": asdict(result.config),
            "diagnostics": result.diagnostics,
            "starts": result.starts,
        }), sort_keys=True),
    })
    return arrays


def save_cavi_run(result, output_dir=DATA_DIR, *, overwrite=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5ad_path = output_dir / f"{result.run_id}.h5ad"
    model_path = output_dir / f"{result.run_id}_model.npz"
    _check_output_paths((h5ad_path, model_path), overwrite)
    estimator = result.estimator
    arrays = _save_common(result, {
        "mu": estimator.mu_,
        "V": estimator.V_,
        "W": estimator.W_,
        "Phi": estimator.Phi_,
        "beta": estimator.beta_,
        "c": estimator.c_,
        "rho": estimator.rho_,
        "beta_delta": estimator.beta_delta_,
        "m_delta": estimator.m_delta_,
        "log_T": estimator.log_T_,
        "T": estimator.T_,
        "elbo": np.asarray(estimator.elbo_),
        "mu_aligned": estimator.mu_aligned_,
        "V_aligned": estimator.V_aligned_,
        "W_aligned": estimator.W_aligned_,
        "Phi_aligned": estimator.Phi_aligned_,
        "B_aligned": estimator.B_aligned_,
        "transforms": estimator.rsp_result_.transforms,
        "signs": estimator.rsp_result_.signs,
        "permutations": estimator.rsp_result_.permutations,
    })
    np.savez_compressed(model_path, **arrays)
    result.prepared.adata.uns[result.run_id]["model_npz"] = str(model_path.resolve())
    result.prepared.adata.write_h5ad(h5ad_path, compression="gzip")
    marker_paths = []
    for name, run in result.downstream.items():
        if run.get("status") == "complete":
            marker_paths.append(write_top_markers(
                result.prepared.adata,
                run["markers"],
                f"CAVI_{name}",
                h5ad_path.with_name(f"{h5ad_path.stem}_{name}_top5_markers"),
            )[1:])
    return h5ad_path, model_path, marker_paths


def save_gibbs_run(result, output_dir=DATA_DIR, *, overwrite=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5ad_path = output_dir / f"{result.run_id}.h5ad"
    model_path = output_dir / f"{result.run_id}_model.npz"
    _check_output_paths((h5ad_path, model_path), overwrite)
    estimator = result.estimator
    sampling = result.diagnostics["sampling"]
    arrays = _save_common(result, {
        "B_samples": estimator.B_samples_,
        "sigma2_samples": estimator.sigma2_samples_,
        "B_posterior_mean": estimator.B_hat_,
        "D_last": estimator.D_last_,
        "eta_last": estimator.eta_last_,
        "score_aligned": result.aligned_scores,
        "batch_means": result.batch_means,
        "batch_indices": result.batch_indices,
        "transforms": estimator.rsp_result_.transforms,
        "signs": estimator.rsp_result_.signs,
        "permutations": estimator.rsp_result_.permutations,
        "bulk_B": sampling["bulk_B"],
        "bulk_sigma2": sampling["bulk_sigma2"],
        "tail_B": sampling["tail_B"],
        "tail_sigma2": sampling["tail_sigma2"],
        "mcse_B_aligned": sampling["mcse_B_aligned"],
        "mcse_sigma2": sampling["mcse_sigma2"],
        "mcse_covariance_diag": sampling["mcse_covariance_diag"],
    })
    _log_stage(result.run_id, "saving compressed model NPZ")
    np.savez_compressed(model_path, **arrays)
    _log_stage(result.run_id, "model NPZ saved; writing H5AD")
    result.prepared.adata.uns[result.run_id]["model_npz"] = str(model_path.resolve())
    result.prepared.adata.write_h5ad(h5ad_path, compression="gzip")
    _log_stage(result.run_id, "H5AD saved; writing marker tables")
    marker_paths = []
    for name, run in result.downstream.items():
        if run.get("status") == "complete":
            marker_paths.append(write_top_markers(
                result.prepared.adata,
                run["markers"],
                f"Gibbs_{name}",
                h5ad_path.with_name(f"{h5ad_path.stem}_{name}_top5_markers"),
            )[1:])
    return h5ad_path, model_path, marker_paths
