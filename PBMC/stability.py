from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PBMC.data_processing import DATA_DIR, jsonable
from PBMC.downstream import DownstreamConfig, require_igraph
from PBMC.representations import REPRESENTATION_NAMES


def leiden_seed_stability(adata, run_id, representation, config, seeds=(0, 1, 2, 3, 4)):
    base = f"leiden_{run_id}_{representation}"
    neighbors = f"neighbors_{run_id}_{representation}"
    initial = adata.obs[base].astype(str)
    keys = []
    rows = []
    for seed in seeds:
        key = f"{base}_seed{seed}"
        sc.tl.leiden(
            adata,
            resolution=config.leiden_resolution,
            random_state=seed,
            flavor="igraph",
            n_iterations=2,
            directed=False,
            neighbors_key=neighbors,
            key_added=key,
        )
        labels = adata.obs[key].astype(str)
        keys.append(key)
        rows.append({
            "seed": int(seed),
            "key": key,
            "n_clusters": int(labels.nunique()),
            "smallest_cluster": int(labels.value_counts().min()),
            "ari_to_initial": float(adjusted_rand_score(initial, labels)),
            "ami_to_initial": float(adjusted_mutual_info_score(initial, labels)),
        })
    ari = np.eye(len(keys))
    ami = np.eye(len(keys))
    for i, left in enumerate(keys):
        for j, right in enumerate(keys):
            if j <= i:
                continue
            ari[i, j] = ari[j, i] = adjusted_rand_score(adata.obs[left].astype(str), adata.obs[right].astype(str))
            ami[i, j] = ami[j, i] = adjusted_mutual_info_score(adata.obs[left].astype(str), adata.obs[right].astype(str))
    upper = np.triu_indices(len(keys), 1)
    return {
        "runs": rows,
        "pairwise_ari": ari,
        "pairwise_ami": ami,
        "pairwise_ari_min": float(ari[upper].min()),
        "pairwise_ari_median": float(np.median(ari[upper])),
        "pairwise_ami_min": float(ami[upper].min()),
        "pairwise_ami_median": float(np.median(ami[upper])),
    }


def neighbor_stability(adata, run_id, representation, config, neighbors_values=(10, 20, 45)):
    score_key = f"X_{run_id}_{representation}"
    scores = np.asarray(adata.obsm[score_key])
    initial = adata.obs[f"leiden_{run_id}_{representation}"].astype(str)
    rows = []
    for k in neighbors_values:
        neighbors_key = f"neighbors_{run_id}_{representation}_k{k}"
        cluster_key = f"leiden_{run_id}_{representation}_k{k}"
        sc.pp.neighbors(
            adata,
            n_neighbors=k,
            n_pcs=scores.shape[1],
            use_rep=score_key,
            metric="euclidean",
            random_state=config.random_state,
            key_added=neighbors_key,
        )
        sc.tl.leiden(
            adata,
            resolution=config.leiden_resolution,
            random_state=config.random_state,
            flavor="igraph",
            n_iterations=2,
            directed=False,
            neighbors_key=neighbors_key,
            key_added=cluster_key,
        )
        labels = adata.obs[cluster_key].astype(str)
        rows.append({
            "n_neighbors": int(k),
            "neighbors": neighbors_key,
            "leiden": cluster_key,
            "n_clusters": int(labels.nunique()),
            "smallest_cluster": int(labels.value_counts().min()),
            "ari_to_k10": float(adjusted_rand_score(initial, labels)),
            "ami_to_k10": float(adjusted_mutual_info_score(initial, labels)),
        })
    return rows


def run_stability(adata, run_id, config=None, representations=REPRESENTATION_NAMES):
    require_igraph()
    config = config or DownstreamConfig()
    results = {"leiden_seeds": {}, "neighbors": {}}
    for representation in representations:
        results["leiden_seeds"][representation] = leiden_seed_stability(
            adata, run_id, representation, config
        )
        results["neighbors"][representation] = neighbor_stability(
            adata, run_id, representation, config
        )
    adata.uns[run_id].pop("stability", None)
    adata.uns[run_id]["stability_json"] = json.dumps(jsonable(results))
    return results


def save_stability(adata, run_id, results, output_dir=DATA_DIR, *, overwrite=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5ad_path = output_dir / f"{run_id}_stability.h5ad"
    json_path = output_dir / f"{run_id}_stability.json"
    csv_path = output_dir / f"{run_id}_stability_summary.csv"
    if not overwrite:
        existing = [str(path) for path in (h5ad_path, json_path, csv_path) if path.exists()]
        if existing:
            raise FileExistsError(f"stability outputs already exist: {existing}")
    adata.write_h5ad(h5ad_path, compression="gzip")
    json_path.write_text(json.dumps(jsonable(results), indent=2), encoding="utf-8")
    rows = []
    for representation in results["leiden_seeds"]:
        seed = results["leiden_seeds"][representation]
        neighbors = results["neighbors"][representation]
        rows.append({
            "run_id": run_id,
            "representation": representation,
            "seed_pairwise_ari_min": seed["pairwise_ari_min"],
            "seed_pairwise_ari_median": seed["pairwise_ari_median"],
            "seed_pairwise_ami_min": seed["pairwise_ami_min"],
            "seed_pairwise_ami_median": seed["pairwise_ami_median"],
            "neighbor_ari_min": min(row["ari_to_k10"] for row in neighbors),
            "neighbor_ari_median": float(np.median([row["ari_to_k10"] for row in neighbors])),
            "neighbor_cluster_counts": [row["n_clusters"] for row in neighbors],
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return h5ad_path, json_path, csv_path
