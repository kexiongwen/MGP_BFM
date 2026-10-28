from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BASELINE = (
    PROJECT_ROOT.parent / "LH_BFM" / "PBMC" / "data" / "notebook_scaled_pca_baseline.h5ad"
)
DATA_DIR = PROJECT_ROOT / "PBMC" / "data"


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def software_versions():
    packages = (
        "mgp-bfm", "numpy", "scipy", "jax", "jaxlib", "scanpy", "anndata",
        "scikit-misc", "scikit-learn", "optax", "arviz", "igraph", "umap-learn",
        "python-igraph", "psutil",
    )
    result = {"python": platform.python_version()}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    try:
        import mgp
        result["mgp_source_version"] = mgp.__version__
        result["mgp_source_path"] = str(Path(mgp.__file__).resolve().parent)
    except ImportError:
        pass
    return result


def file_sha256(path):
    path = Path(path)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class PBMCInput:
    adata: ad.AnnData
    X: np.ndarray
    hvg_names: np.ndarray
    baseline_path: Path
    source: dict

    @property
    def cell_names(self):
        return self.adata.obs_names.to_numpy(dtype=str)


def prepare_model_input(prepared, input_mode):
    if input_mode != "seurat":
        raise ValueError("the fixed workflow requires input_mode='seurat'")
    missing = {"total_counts", "pct_counts_mt"}.difference(prepared.adata.obs.columns)
    if missing:
        raise ValueError(f"missing Seurat regression covariates: {sorted(missing)}")
    model = prepared.adata[:, prepared.hvg_names].copy()
    model.layers["scaled"] = prepared.X.copy()
    sc.pp.regress_out(
        model,
        ["total_counts", "pct_counts_mt"],
        layer="scaled",
    )
    sc.pp.scale(model, max_value=10, layer="scaled")
    X = np.asarray(model.layers["scaled"], dtype=np.float64, order="C")
    expected_shape = (prepared.adata.n_obs, len(prepared.hvg_names))
    if X.shape != expected_shape or not np.isfinite(X).all():
        raise ValueError("invalid Seurat model input")
    if np.max(np.abs(X)) > 10.0 + 1e-10:
        raise ValueError("Seurat clipping was not applied")
    source = dict(prepared.source)
    source["model_input"] = {
        "name": "X_scaled",
        "sha256": hashlib.sha256(X.view(np.uint8)).hexdigest(),
        "shape": list(X.shape),
        "regress_keys": ["total_counts", "pct_counts_mt"],
        "scale_max_value": 10.0,
        "estimator_center": False,
        "estimator_standardize": False,
    }
    return PBMCInput(
        adata=prepared.adata,
        X=X,
        hvg_names=prepared.hvg_names,
        baseline_path=prepared.baseline_path,
        source=source,
    )


def load_baseline(path=DEFAULT_BASELINE, *, verify_source=True):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PCA baseline does not exist: {path}")
    adata = sc.read_h5ad(path)
    if adata.shape != (2638, 13714):
        raise ValueError(f"expected (2638, 13714), got {adata.shape}")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("cell and gene identifiers must be unique")
    missing = {"counts", "lognorm"}.difference(adata.layers)
    if missing:
        raise ValueError(f"baseline lacks required layers: {sorted(missing)}")
    if "highly_variable" not in adata.var:
        raise ValueError("baseline lacks the HVG mask")
    mask = adata.var["highly_variable"].to_numpy(dtype=bool)
    if int(mask.sum()) != 2000:
        raise ValueError(f"expected 2000 HVGs, got {int(mask.sum())}")
    matrix = adata.layers["lognorm"][:, mask]
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    X = np.asarray(matrix, dtype=np.float64, order="C")
    if X.shape != (2638, 2000) or not np.isfinite(X).all():
        raise ValueError("invalid log-normalized HVG matrix")

    source = {
        "baseline_path": str(path),
        "baseline_sha256": file_sha256(path),
        "baseline_shape": list(adata.shape),
    }
    if verify_source:
        recorded = adata.uns["preprocessing"]["source"]["files"]["pbmc3k_raw.h5ad"]
        raw_path = Path(adata.uns["pbmc3k_source_path"])
        actual_sha = file_sha256(raw_path)
        if actual_sha != recorded["sha256"]:
            raise ValueError("raw PBMC3k SHA-256 differs from the recorded value")
        if raw_path.stat().st_size != recorded["size_bytes"]:
            raise ValueError("raw PBMC3k size differs from the recorded value")
        source.update({
            "raw_path": str(raw_path.resolve()),
            "raw_sha256_recorded": recorded["sha256"],
            "raw_sha256_actual": actual_sha,
            "raw_size_bytes": raw_path.stat().st_size,
        })
    return PBMCInput(
        adata=adata,
        X=X,
        hvg_names=adata.var_names[mask].to_numpy(dtype=str),
        baseline_path=path,
        source=source,
    )


def input_summary(prepared):
    sd = prepared.X.std(axis=0, ddof=1)
    return {
        "shape": list(prepared.X.shape),
        "dtype": str(prepared.X.dtype),
        "finite": bool(np.isfinite(prepared.X).all()),
        "minimum": float(prepared.X.min()),
        "maximum": float(prepared.X.max()),
        "gene_sd_quantiles": dict(zip(
            ["min", "q05", "median", "q95", "max"],
            np.quantile(sd, [0, 0.05, 0.5, 0.95, 1]).tolist(),
        )),
        "zero_sd_genes": int(np.count_nonzero(sd == 0)),
        "hvg_first10": prepared.hvg_names[:10].tolist(),
        "cell_first3": prepared.cell_names[:3].tolist(),
    }


def save_prepared_input(prepared, output_dir=DATA_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "mgp_input.npz"
    manifest_path = output_dir / "mgp_input_manifest.json"
    np.savez_compressed(
        arrays_path,
        X_lognorm=prepared.X,
        cell_names=prepared.cell_names,
        hvg_names=prepared.hvg_names,
    )
    manifest = {
        "source": prepared.source,
        "input": input_summary(prepared),
        "arrays_path": str(arrays_path.resolve()),
        "arrays_sha256": file_sha256(arrays_path),
        "software": software_versions(),
    }
    manifest_path.write_text(
        json.dumps(jsonable(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return arrays_path, manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--skip-source-check", action="store_true")
    args = parser.parse_args()
    prepared = load_baseline(args.baseline, verify_source=not args.skip_source_check)
    arrays_path, manifest_path = save_prepared_input(prepared, args.output_dir)
    print(json.dumps(input_summary(prepared), indent=2))
    print(f"Saved {arrays_path}")
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    main()
