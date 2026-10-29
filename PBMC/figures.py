from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scanpy as sc

from .representations import REPRESENTATION_NAMES

PBMC_MARKERS = {
    "T": ["CD3D", "CD3E", "IL7R", "LTB"],
    "B": ["CD79A", "CD79B", "MS4A1"],
    "NK": ["NKG7", "GNLY", "GZMB", "XCL1", "XCL2"],
    "Monocyte": ["LYZ", "S100A8", "S100A9", "FCGR3A", "LST1"],
    "DC": ["CST3", "FCER1A", "C1QA", "C1QB"],
    "Platelet": ["PPBP", "PF4", "NRGN"],
    "Interferon": ["ISG15", "IFI6", "MX1"],
}


def save_cavi_elbo(model_path, output_path):
    import matplotlib.pyplot as plt

    with np.load(model_path, allow_pickle=False) as z:
        payload = json.loads(z["diagnostics_json"].item())
    starts = payload["starts"]
    selected = payload["diagnostics"]["selected_start"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for index, (key, value) in enumerate(starts.items()):
        elbo = np.asarray(value["elbo"], dtype=np.float64)
        ax.plot(
            np.arange(len(elbo)),
            elbo,
            linewidth=1.8 if index == selected else 1.0,
            label=f"start {index}" + (" (selected)" if index == selected else ""),
        )
    ax.set_xlabel("sweep")
    ax.set_ylabel("ELBO")
    ax.set_title("MGP CAVI: deterministic start")
    ax.legend()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_gibbs_ess(model_path, output_path):
    import matplotlib.pyplot as plt

    with np.load(model_path, allow_pickle=False) as z:
        bulk_B = z["bulk_B"]
        bulk_sigma2 = z["bulk_sigma2"]
        diagnostics = json.loads(z["diagnostics_json"].item())["diagnostics"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].hist(bulk_B.ravel(), bins=40)
    axes[0].set_title("Raw-loadings bulk ESS")
    axes[1].hist(bulk_sigma2, bins=40)
    axes[1].set_title("Noise-variance bulk ESS")
    for ax in axes:
        ax.axvline(400, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("ESS")
        ax.set_ylabel("entries")
    active = diagnostics["sampling"]["active_loading_bulk_ess"]
    fig.suptitle(
        f"Aligned active entries: ESS>=400 coverage {active['coverage_at_400']:.1%}"
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_marker_dotplots(adata, run_id, output_dir, representations=REPRESENTATION_NAMES):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for representation in representations:
        groupby = f"leiden_{run_id}_{representation}"
        display_groupby = "cluster_1_based"
        adata.obs[display_groupby] = (
            adata.obs[groupby].astype(int) + 1
        ).astype(str).astype("category")
        plot = sc.pl.dotplot(
            adata,
            PBMC_MARKERS,
            groupby=display_groupby,
            use_raw=False,
            show=False,
            return_fig=True,
        )
        path = output_dir / f"{run_id}_{representation}_marker_dotplot.png"
        plot.savefig(path, dpi=150, bbox_inches="tight")
        paths.append(path)
    return paths


def save_loading_norm_figures(cavi_model, gibbs_model, output_dir):
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    specifications = (
        ("cavi", cavi_model, "loading_column_norms_cavi.png"),
        ("gibbs", gibbs_model, "loading_column_norms_gibbs.png"),
    )
    for method, model_path, filename in specifications:
        with np.load(model_path, allow_pickle=False) as z:
            if method == "cavi":
                raw = np.linalg.norm(z["mu"], axis=0)
                aligned = np.linalg.norm(z["mu_aligned"], axis=0)
            else:
                raw = np.linalg.norm(z["B_samples"].mean(axis=0), axis=0)
                aligned = np.linalg.norm(z["B_posterior_mean"], axis=0)
        columns = np.arange(1, len(raw) + 1)
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        ax.plot(columns, raw, marker=".", linewidth=1, label="Before RSP")
        ax.plot(columns, aligned, marker=".", linewidth=1, label="After RSP")
        ax.set_xlabel("factor column")
        ax.set_ylabel("loading column L2 norm")
        title = "CAVI" if method == "cavi" else "GS"
        ax.set_title(title)
        ax.legend()
        path = output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths
