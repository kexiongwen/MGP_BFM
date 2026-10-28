from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PBMC.data_processing import DATA_DIR, jsonable
from PBMC.downstream import write_top_markers
from PBMC.seurat_analysis import CAVI_RUN, GIBBS_RUN, REPRESENTATION

LEIDEN_RESOLUTIONS = {
    "cavi": 0.425,
    "gibbs": 0.5,
}
RUN_IDS = {
    "cavi": CAVI_RUN,
    "gibbs": GIBBS_RUN,
}
RESOLUTION_HITS = {
    "cavi": [0.3, 0.35, 0.375, 0.4, 0.425],
    "gibbs": [0.5, 0.525],
}


def _tuning_payload(method, resolution):
    return {
        "target_n_clusters": 9,
        "original_resolution": 0.45,
        "selected_resolution": resolution,
        "selection_rule": "9 clusters, then closest to the original resolution 0.45",
        "resolution_hits": RESOLUTION_HITS[method],
        "neighbor_graph_reused": True,
        "umap_reused": True,
        "random_state": 0,
        "flavor": "igraph",
        "n_iterations": 2,
        "directed": False,
    }


def _update_model_archive(run_id, method, resolution, tuning):
    model_path = DATA_DIR / f"{run_id}_model.npz"
    with np.load(model_path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    config = json.loads(arrays["config_json"].item())
    config["leiden_resolution"] = resolution
    arrays["config_json"] = json.dumps(config, sort_keys=True)
    payload = json.loads(arrays["diagnostics_json"].item())
    payload["config"]["leiden_resolution"] = resolution
    payload["diagnostics"]["leiden_tuning"] = tuning
    downstream = payload["diagnostics"]["downstream"][REPRESENTATION]
    downstream["leiden_resolution"] = resolution
    downstream["n_clusters"] = 9
    arrays["diagnostics_json"] = json.dumps(
        jsonable(payload), ensure_ascii=False, sort_keys=True
    )
    np.savez_compressed(model_path, **arrays)


def tune_leiden():
    summary = {}
    search_rows = []
    for method, run_id in RUN_IDS.items():
        resolution = LEIDEN_RESOLUTIONS[method]
        tuning = _tuning_payload(method, resolution)
        cluster_key = f"leiden_{run_id}_{REPRESENTATION}"
        marker_key = f"markers_{run_id}_{REPRESENTATION}"
        neighbors_key = f"neighbors_{run_id}_{REPRESENTATION}"
        backup_cluster = f"{cluster_key}_r0p45"
        backup_marker = f"{marker_key}_r0p45"
        path = DATA_DIR / f"{run_id}.h5ad"
        adata = sc.read_h5ad(path)
        if backup_cluster not in adata.obs:
            adata.obs[backup_cluster] = adata.obs[cluster_key].copy()
        if backup_marker not in adata.uns:
            adata.uns[backup_marker] = copy.deepcopy(adata.uns[marker_key])
        sc.tl.leiden(
            adata,
            resolution=resolution,
            random_state=0,
            flavor="igraph",
            n_iterations=2,
            directed=False,
            neighbors_key=neighbors_key,
            key_added=cluster_key,
        )
        if int(adata.obs[cluster_key].nunique()) != 9:
            raise RuntimeError(f"{method} did not produce 9 clusters")
        sc.tl.rank_genes_groups(
            adata,
            groupby=cluster_key,
            mask_var=None,
            use_raw=False,
            method="wilcoxon",
            corr_method="benjamini-hochberg",
            key_added=marker_key,
        )
        run = adata.uns[run_id]["downstream"][REPRESENTATION]
        run["leiden_resolution"] = resolution
        run["n_clusters"] = 9
        adata.uns[run_id]["diagnostics"]["downstream"][REPRESENTATION].update({
            "leiden_resolution": resolution,
            "n_clusters": 9,
        })
        adata.uns[run_id]["diagnostics"]["leiden_tuning"] = tuning
        adata.uns[run_id]["leiden_tuning"] = tuning
        config = json.loads(adata.uns[run_id]["config_json"])
        config["leiden_resolution"] = resolution
        adata.uns[run_id]["config_json"] = json.dumps(config, sort_keys=True)
        adata.write_h5ad(path, compression="gzip")
        _update_model_archive(run_id, method, resolution, tuning)
        write_top_markers(
            adata,
            marker_key,
            method.upper(),
            path.with_name(f"{path.stem}_{REPRESENTATION}_top5_markers"),
        )
        summary[method] = {
            "run_id": run_id,
            "n_clusters": 9,
            **tuning,
        }
        for hit in RESOLUTION_HITS[method]:
            search_rows.append({
                "method": method,
                "resolution": hit,
                "n_clusters": 9,
                "selected": hit == resolution,
            })
    pd.DataFrame(search_rows).to_csv(
        DATA_DIR / "leiden9_resolution_search.csv", index=False
    )
    (DATA_DIR / "leiden9_tuning.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main():
    print(json.dumps(tune_leiden(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
