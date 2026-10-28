from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scanpy as sc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PBMC.data_processing import (
    DATA_DIR,
    DEFAULT_BASELINE,
    jsonable,
    load_baseline,
    prepare_model_input,
    software_versions,
)
from PBMC.downstream import DownstreamConfig, run_downstream, write_top_markers


@dataclass(frozen=True)
class PCAConfig:
    n_comps: int = 50
    n_neighbors: int = 10
    leiden_resolution: float = 0.7
    random_state: int = 0


def run_pca(prepared, config=None):
    if config is None:
        config = PCAConfig()
    run_id = "pca_mgp_seurat_aligned"
    prepared = prepare_model_input(prepared, "seurat")
    X = prepared.X.copy()
    model = prepared.adata[:, prepared.hvg_names].copy()
    model.X = X
    sc.pp.pca(
        model,
        n_comps=config.n_comps,
        zero_center=True,
        svd_solver="arpack",
        random_state=config.random_state,
        mask_var=None,
        dtype="float32",
    )
    scores = np.asarray(model.obsm["X_pca"]).copy()
    prepared.adata.obsm[f"X_{run_id}"] = scores
    representation = f"X_{run_id}_d50"
    prepared.adata.obsm[representation] = scores
    runs = {
        "d50": run_downstream(
            prepared.adata,
            representation=representation,
            name=f"{run_id}_d50",
            config=DownstreamConfig(
                n_neighbors=config.n_neighbors,
                leiden_resolution=config.leiden_resolution,
                random_state=config.random_state,
            ),
        )
    }
    prepared.adata.uns[run_id] = {
        "config_json": json.dumps(jsonable(asdict(config)), sort_keys=True),
        "input_mode": "seurat",
        "source": prepared.source,
        "variance": np.asarray(model.uns["pca"]["variance"]),
        "variance_ratio": np.asarray(model.uns["pca"]["variance_ratio"]),
        "loadings": np.asarray(model.varm["PCs"]),
        "params_json": json.dumps(jsonable(model.uns["pca"]["params"])),
        "runs": runs,
        "software": software_versions(),
    }
    return run_id, runs


def save_pca(prepared, run_id, runs, output_dir=DATA_DIR, *, overwrite=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5ad_path = output_dir / f"{run_id}.h5ad"
    model_path = output_dir / f"{run_id}_model.npz"
    if not overwrite:
        existing = [str(path) for path in (h5ad_path, model_path) if path.exists()]
        if existing:
            raise FileExistsError(f"outputs already exist: {existing}")
    metadata = prepared.adata.uns[run_id]
    np.savez_compressed(
        model_path,
        scores=prepared.adata.obsm[f"X_{run_id}"],
        loadings=metadata["loadings"],
        variance=metadata["variance"],
        variance_ratio=metadata["variance_ratio"],
        hvg_names=prepared.hvg_names,
        cell_names=prepared.cell_names,
        config_json=metadata["config_json"],
        source_json=json.dumps(jsonable(prepared.source), sort_keys=True),
    )
    prepared.adata.uns[run_id]["model_npz"] = str(model_path.resolve())
    prepared.adata.write_h5ad(h5ad_path, compression="gzip")
    marker_paths = []
    for dimension, run in runs.items():
        marker_paths.append(write_top_markers(
            prepared.adata,
            run["markers"],
            f"PCA_{dimension}",
            h5ad_path.with_name(f"{h5ad_path.stem}_{dimension}_top5_markers"),
        )[1:])
    return h5ad_path, model_path, marker_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-source-check", action="store_true")
    args = parser.parse_args()
    prepared = load_baseline(args.baseline, verify_source=not args.skip_source_check)
    run_id, runs = run_pca(prepared)
    paths = save_pca(prepared, run_id, runs, args.output_dir, overwrite=args.overwrite)
    print(json.dumps({
        "run_id": run_id,
        "runs": runs,
        "h5ad": str(paths[0]),
        "model": str(paths[1]),
    }, indent=2))


if __name__ == "__main__":
    main()
