from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PBMC.covariance_checks import _covariance_diagnostics, check_marker_covariances
from PBMC.data_processing import DATA_DIR, jsonable
from PBMC.figures import (
    save_cavi_elbo,
    save_gibbs_ess,
    save_loading_norm_figures,
    save_marker_dotplots,
)
from PBMC.stability import run_stability, save_stability

CAVI_RUN = "cavi_seurat_k50_d20"
GIBBS_RUN = "gibbs_seurat_k50_n20000_b10000_t10_d20"
PCA_RUN = "pca_mgp_seurat_aligned"
REPRESENTATION = "d20_norm"
LEIDEN_RESOLUTIONS = {
    CAVI_RUN: 0.425,
    GIBBS_RUN: 0.5,
    PCA_RUN: 0.7,
}
FIGURE_DIR = PROJECT_ROOT / "PBMC" / "figures" / "mgp_seurat"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_payload(run_id):
    with np.load(DATA_DIR / f"{run_id}_model.npz", allow_pickle=False) as z:
        return json.loads(z["diagnostics_json"].item())


def labels(adata, key):
    if key not in adata.obs or adata.obs[key].isna().any():
        raise ValueError(f"missing or invalid cluster labels: {key}")
    return adata.obs[key].astype(str)


def metric(left, right):
    return {
        "ari": float(adjusted_rand_score(left, right)),
        "ami": float(adjusted_mutual_info_score(left, right)),
    }


def rank_99(representation_info):
    singular = np.asarray(representation_info["score_singular_values"], dtype=np.float64)
    squared = singular * singular
    total = squared.sum()
    if not np.isfinite(total) or total < 0:
        raise ValueError("invalid score spectrum")
    return int(np.searchsorted(np.cumsum(squared), 0.99 * total) + 1) if total > 0 else 0


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def markdown_table(rows, columns):
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(key)) for key, _ in columns) + " |")
    return "\n".join(lines)


def gibbs_review_rows(run_ids):
    rows = []
    for run_id in run_ids:
        diagnostics = model_payload(run_id)["diagnostics"]
        sampling = diagnostics["sampling"]
        coverage = sampling["active_loading_bulk_ess"]["coverage_at_400"]
        marker_mcse = sampling["marker_covariance_mcse_relative_max"]
        finite_covariance_mcse = bool(np.isfinite(sampling["mcse_covariance_diag"]).all())
        thresholds = sampling["score_stability_thresholds"]
        score_stability = diagnostics["score_stability"]
        median_max = max(value["relative_l2_median"] for value in score_stability.values())
        half_chain_max = max(value["relative_l2_max"] for value in score_stability.values())
        exceeded = [
            name for name, value in score_stability.items()
            if value["relative_l2_median"] > thresholds["relative_l2_median"]
            or value["relative_l2_max"] > thresholds["relative_l2_max"]
        ]
        reasons = []
        if coverage < 0.9:
            reasons.append("活动载荷ESS覆盖率<90%")
        if marker_mcse is not None and marker_mcse > 0.01:
            reasons.append("marker协方差MCSE>1%")
        if not finite_covariance_mcse:
            reasons.append("协方差MCSE含非有限值")
        if exceeded:
            reasons.append("半链差异超限:" + ",".join(exceeded))
        review_required = bool(reasons)
        if review_required != sampling["review_required"]:
            raise RuntimeError(f"{run_id} has inconsistent sampling review diagnostics")
        rows.append({
            "run_id": run_id,
            "ess_coverage": coverage,
            "ess_median": sampling["active_loading_bulk_ess"]["median"],
            "marker_mcse_max": marker_mcse,
            "half_chain_median_max": median_max,
            "half_chain_max": half_chain_max,
            "review_required": review_required,
            "review_reasons": "；".join(reasons) if reasons else "上述阈值均未触发",
        })
    return rows


def load_models():
    run_ids = {"cavi": CAVI_RUN, "gibbs": GIBBS_RUN, "pca": PCA_RUN}
    adatas = {
        method: sc.read_h5ad(DATA_DIR / f"{run_id}.h5ad")
        for method, run_id in run_ids.items()
    }
    payloads = {
        method: model_payload(run_id)
        for method, run_id in (("cavi", CAVI_RUN), ("gibbs", GIBBS_RUN))
    }
    cell_names = adatas["cavi"].obs_names
    for adata in adatas.values():
        if not adata.obs_names.equals(cell_names):
            raise ValueError("cell identifiers or order differ across Seurat runs")
    for method, precision in (("cavi", "float64"), ("gibbs", "float32")):
        payload = payloads[method]
        config = payload["config"]
        diagnostics = payload["diagnostics"]
        if config["input_mode"] != "seurat" or config["precision"] != precision:
            raise ValueError(f"unexpected input or precision for {method}")
        if config["representation"] != REPRESENTATION or config["K"] != 50:
            raise ValueError(f"unexpected model or representation for {method}")
        if not diagnostics["pipeline_complete"] or diagnostics["rsp"]["status_code"] != 1:
            raise ValueError(f"{method} is not complete and cannot enter the Seurat report")
        if diagnostics["downstream"].get(REPRESENTATION, {}).get("status") != "complete":
            raise ValueError(f"{method} lacks complete d20_norm downstream output")
        run_id = run_ids[method]
        if config["leiden_resolution"] != LEIDEN_RESOLUTIONS[run_id]:
            raise ValueError(f"unexpected Leiden resolution for {method}")
        cluster_key = f"leiden_{run_id}_{REPRESENTATION}"
        if int(adatas[method].obs[cluster_key].nunique()) != 9:
            raise ValueError(f"{method} must have 9 tuned clusters")
        score = adatas[method].obsm[f"X_{run_id}_{REPRESENTATION}"]
        if score.shape != (2638, 20) or not np.isfinite(score).all():
            raise ValueError(f"invalid {method} d20_norm scores")
    if not payloads["cavi"]["diagnostics"]["fit"]["converged"]:
        raise ValueError("formal Seurat CAVI has not reached its stopping criterion")
    expected_prior = {"nu": 3.0, "a1": 2.1, "a2": 3.1, "a_sigma": 1.0, "b_sigma": 1.0}
    for key, expected in expected_prior.items():
        if payloads["cavi"]["config"][key] != payloads["gibbs"]["config"][key]:
            raise ValueError(f"CAVI and Gibbs prior mismatch: {key}")
        if payloads["cavi"]["config"][key] != expected:
            raise ValueError(f"unexpected aligned prior value: {key}")
    fit = payloads["gibbs"]["diagnostics"]["fit"]
    if (fit["n_iter"], fit["burn_in"], fit["thin"], fit["n_kept"]) != (20000, 10000, 10, 1000):
        raise ValueError("unexpected formal Gibbs settings")
    hashes = []
    for method in ("cavi", "gibbs"):
        source = adatas[method].uns[run_ids[method]]["source"]
        model_input = source["model_input"]
        if model_input["name"] != "X_scaled" or model_input["estimator_center"]:
            raise ValueError(f"unexpected input preprocessing for {method}")
        hashes.append(model_input["sha256"])
    if len(set(hashes)) != 1:
        raise ValueError("CAVI and Gibbs used different Seurat matrices")
    pca_scores = adatas["pca"].obsm[f"X_{PCA_RUN}"]
    pca_downstream = adatas["pca"].obsm[f"X_{PCA_RUN}_d50"]
    if pca_scores.shape != (2638, 50) or pca_downstream.shape != (2638, 50):
        raise ValueError("Seurat PCA must compute and use 50 PCs")
    if set(adatas["pca"].uns[PCA_RUN]["runs"]) != {"d50"}:
        raise ValueError("unexpected Seurat PCA downstream dimensions")
    pca_source = adatas["pca"].uns[PCA_RUN]
    if pca_source["input_mode"] != "seurat":
        raise ValueError("unexpected PCA input mode")
    if pca_source["source"]["model_input"]["sha256"] != hashes[0]:
        raise ValueError("PCA and MGP used different Seurat matrices")
    return adatas, payloads


def compare(output_dir=DATA_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adatas, payloads = load_models()
    pca_labels = labels(adatas["pca"], f"leiden_{PCA_RUN}_d50")
    if not pca_labels.index.equals(adatas["cavi"].obs_names):
        raise ValueError("PCA and MGP cell identifiers are not aligned")
    rows = []
    cluster_rows = []
    label_runs = {}
    for method, run_id in (("cavi", CAVI_RUN), ("gibbs", GIBBS_RUN)):
        adata = adatas[method]
        current = labels(adata, f"leiden_{run_id}_{REPRESENTATION}")
        label_runs[method] = current
        info = adata.uns[run_id]["representation"]["representations"][REPRESENTATION]
        rows.append({
            "method": method,
            "precision": payloads[method]["config"]["precision"],
            "representation": REPRESENTATION,
            "n_columns": int(info["n_columns"]),
            "rank_99_percent": rank_99(info),
            "active_factors": int(adata.uns[run_id]["representation"]["active_factor_count"]),
            "selected_active_factors": int(info["selected_active_factor_count"]),
            "n_clusters": int(current.nunique()),
            **{f"{key}_vs_pca50": value for key, value in metric(pca_labels, current).items()},
        })
        for cluster, count in current.value_counts().sort_index().items():
            cluster_rows.append({
                "method": method,
                "representation": REPRESENTATION,
                "cluster": str(cluster),
                "n_cells": int(count),
            })
    for cluster, count in pca_labels.value_counts().sort_index().items():
        cluster_rows.append({
            "method": "pca50",
            "representation": "d50",
            "cluster": str(cluster),
            "n_cells": int(count),
        })
    label_runs["pca50"] = pca_labels
    inference = metric(label_runs["cavi"], label_runs["gibbs"])
    cavi_model = DATA_DIR / f"{CAVI_RUN}_model.npz"
    gibbs_model = DATA_DIR / f"{GIBBS_RUN}_model.npz"
    covariance = _covariance_diagnostics(
        cavi_model, gibbs_model, adatas["cavi"].n_obs, payloads["cavi"]["config"]["a_sigma"]
    )
    public_covariance = {
        key: value for key, value in covariance.items() if not key.endswith("_diagonal")
    }
    marker_rows, missing_pairs = check_marker_covariances(cavi_model, gibbs_model)
    marker_summary = {
        "rows": marker_rows,
        "missing_pairs": missing_pairs,
        "max_absolute_mean_difference": max(abs(row["mean_difference"]) for row in marker_rows),
        "mean_absolute_mean_difference": float(np.mean([
            abs(row["mean_difference"]) for row in marker_rows
        ])),
        "note": "Raw-coordinate off-diagonal covariances. CAVI is mean-field; Gibbs uses posterior draws. Precision differs between the two workflows.",
    }
    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "runs": {"cavi": CAVI_RUN, "gibbs": GIBBS_RUN, "pca": PCA_RUN},
        "input": adatas["cavi"].uns[CAVI_RUN]["source"]["model_input"],
        "config": {method: payloads[method]["config"] for method in payloads},
        "cluster_metrics": rows,
        "cavi_vs_gibbs": inference,
        "pca50": {
            "n_columns": 50,
            "n_clusters": int(pca_labels.nunique()),
        },
        "covariance_diagonal_comparison": public_covariance,
        "covariance_marker_pairs": marker_summary,
        "interpretation": "Aligned MGP workflows use CAVI float64 and Gibbs float32. ARI/AMI measure clustering agreement, not cell-type accuracy. No shared subsampling was run.",
    }
    (output_dir / "seurat_comparison.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(output_dir / "seurat_cluster_metrics.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(output_dir / "seurat_cluster_sizes.csv", index=False)
    (output_dir / "seurat_covariance_diagonal.json").write_text(
        json.dumps(jsonable(public_covariance), indent=2), encoding="utf-8"
    )
    (output_dir / "seurat_covariance_marker_pairs.json").write_text(
        json.dumps(jsonable(marker_summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(marker_rows).to_csv(output_dir / "seurat_covariance_marker_pairs.csv", index=False)
    return summary


def stability(output_dir=DATA_DIR, overwrite=False, method="all"):
    specifications = (
        (CAVI_RUN, (REPRESENTATION,)),
        (GIBBS_RUN, (REPRESENTATION,)),
        (PCA_RUN, ("d50",)),
    )
    if method != "all":
        specifications = tuple(
            item for item in specifications if item[0].startswith(f"{method}_")
        )
    paths = []
    for run_id, representations in specifications:
        expected = tuple(
            Path(output_dir) / f"{run_id}_stability{suffix}"
            for suffix in (".h5ad", ".json", "_summary.csv")
        )
        if not overwrite and all(path.is_file() for path in expected):
            frame = pd.read_csv(expected[2])
            if set(frame["representation"]) != set(representations):
                raise ValueError(f"stability outputs have unexpected representations: {run_id}")
            paths.extend(expected)
            print(json.dumps({"run_id": run_id, "status": "already_complete"}, indent=2), flush=True)
            continue
        if not overwrite and any(path.exists() for path in expected):
            raise FileExistsError(f"incomplete stability outputs exist for {run_id}")
        adata = sc.read_h5ad(DATA_DIR / f"{run_id}.h5ad")
        from PBMC.downstream import DownstreamConfig

        config = DownstreamConfig(leiden_resolution=LEIDEN_RESOLUTIONS[run_id])
        result = run_stability(
            adata, run_id, config=config, representations=representations
        )
        saved = save_stability(
            adata, run_id, result, output_dir=output_dir, overwrite=overwrite
        )
        paths.extend(saved)
        print(json.dumps({
            "run_id": run_id,
            "representations": list(representations),
            "stability_outputs": [str(path) for path in saved],
        }, indent=2), flush=True)
    return paths


def figures(output_dir=FIGURE_DIR):
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adatas, _ = load_models()
    paths = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, (title, method, run_id, representation) in zip(axes, (
        ("CAVI f64: d20_norm", "cavi", CAVI_RUN, REPRESENTATION),
        ("Gibbs f32: d20_norm", "gibbs", GIBBS_RUN, REPRESENTATION),
        ("Seurat PCA: 50 PCs", "pca", PCA_RUN, "d50"),
    )):
        cluster_key = f"leiden_{run_id}_{representation}"
        display_key = "cluster_1_based"
        adatas[method].obs[display_key] = (
            adatas[method].obs[cluster_key].astype(int) + 1
        ).astype(str).astype("category")
        sc.pl.embedding(
            adatas[method],
            basis=f"umap_{run_id}_{representation}",
            color=display_key,
            title=title,
            frameon=False,
            show=False,
            ax=ax,
        )
    path = output_dir / "comparison_umap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    paths.append(save_cavi_elbo(
        DATA_DIR / f"{CAVI_RUN}_model.npz", output_dir / "cavi_elbo.png"
    ))
    paths.append(save_gibbs_ess(
        DATA_DIR / f"{GIBBS_RUN}_model.npz", output_dir / "gibbs_ess.png"
    ))
    paths.extend(save_loading_norm_figures(
        DATA_DIR / f"{CAVI_RUN}_model.npz",
        DATA_DIR / f"{GIBBS_RUN}_model.npz",
        output_dir,
    ))
    for run_id, representation in (
        (CAVI_RUN, REPRESENTATION),
        (GIBBS_RUN, REPRESENTATION),
        (PCA_RUN, "d50"),
    ):
        paths.extend(save_marker_dotplots(
            adatas["pca" if run_id == PCA_RUN else "cavi" if run_id == CAVI_RUN else "gibbs"],
            run_id,
            output_dir,
            representations=(representation,),
        ))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, (title, method, run_id) in zip(axes, (
        ("CAVI f64", "cavi", CAVI_RUN),
        ("Gibbs f32", "gibbs", GIBBS_RUN),
    )):
        info = adatas[method].uns[run_id]["representation"]["representations"][REPRESENTATION]
        singular = np.asarray(info["score_singular_values"])
        ax.plot(np.arange(1, len(singular) + 1), singular, marker=".")
        if len(singular) and np.all(singular > 0):
            ax.set_yscale("log")
        ax.set_title(f"{title}: d20_norm")
        ax.set_xlabel("score column")
        ax.set_ylabel("singular value")
    path = output_dir / "score_spectra.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    cluster_sizes = pd.read_csv(DATA_DIR / "seurat_cluster_sizes.csv")
    methods = list(cluster_sizes["method"].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4), constrained_layout=True)
    for ax, method in zip(np.atleast_1d(axes), methods):
        subset = cluster_sizes[cluster_sizes["method"] == method]
        display_clusters = (subset["cluster"].astype(int) + 1).astype(str)
        ax.bar(display_clusters, subset["n_cells"])
        ax.set_title(method)
        ax.set_xlabel("Leiden cluster")
        ax.set_ylabel("cells")
        ax.tick_params(axis="x", rotation=90)
    path = output_dir / "cluster_sizes.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def report(output_path=PROJECT_ROOT / "PBMC" / "seurat_report.md"):

    summary = read_json(DATA_DIR / "seurat_comparison.json")
    cavi = model_payload(CAVI_RUN)
    gibbs = model_payload(GIBBS_RUN)
    stability_rows = []
    for method, run_id in (("cavi_f64", CAVI_RUN), ("gibbs_f32", GIBBS_RUN), ("pca50", PCA_RUN)):
        frame = pd.read_csv(DATA_DIR / f"{run_id}_stability_summary.csv")
        frame.insert(0, "method", method)
        stability_rows.extend(frame.to_dict("records"))
    sampling = gibbs["diagnostics"]["sampling"]
    selected = cavi["diagnostics"]["selected_start"]
    parts = [
        "# Seurat 输入下的 MGP 对齐分析报告",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat()}",
        "",
        "## 1. 配置与解释边界",
        "",
        "- 输入为 2000 HVG 的 lognorm，经 total_counts、pct_counts_mt 回归及 sc.pp.scale(max_value=10) 处理；MGP 不再居中或标准化。",
        f"- 输入矩阵 SHA-256：`{summary['input']['sha256']}`。CAVI 使用 float64，Gibbs 采样使用 float32，跨样本累计使用 float64。",
        "- 两种 MGP 均为 K=50，先验为 nu=3、a1=2.1、a2=3.1、a_sigma=1、b_sigma=1；主表示为 RSP 后的 d20_norm。",
        "- Seurat PCA 计算并全部使用 50 个 PC，仅作参照。未执行共享子抽样，也未运行额外的 Gibbs float64 对照链。",
        "- ARI/AMI 是聚类一致性而非细胞类型准确率；UMAP 只用于展示。按用户指定目标调整 Leiden 分辨率：CAVI=0.425、Gibbs=0.500，两种方法均为 9 簇；原 0.45 标签另行保留。",
        "",
        "## 2. CAVI float64",
        "",
        markdown_table(
            [
                {
                    "start": key,
                    "n_iter": value["fit"]["n_iter"],
                    "elbo": f"{value['fit']['final_elbo']:.6f}",
                    "stop": value["fit"]["stop_reason"],
                    "selected": key == f"start_{selected}",
                }
                for key, value in cavi["starts"].items()
            ],
            [("start", "起点"), ("n_iter", "轮数"), ("elbo", "最终ELBO"), ("stop", "停止"), ("selected", "被选择")],
        ),
        "",
        f"拟合墙钟时间约 {cavi['diagnostics']['fit']['fit_wall_sec'] / 60:.2f} 分钟。",
        "",
        "## 3. Gibbs float32",
        "",
        "n_iter=20000、burn_in=10000、thin=10、seed=2026、保留1000个样本、exact Gamma。",
        "",
        f"活动载荷 ESS≥400 覆盖率 {sampling['active_loading_bulk_ess']['coverage_at_400']:.2%}；marker 协方差 MCSE/均值最大值 {sampling['marker_covariance_mcse_relative_max']:.4g}。",
        "",
        markdown_table(
            gibbs_review_rows([GIBBS_RUN]),
            [("run_id", "结果"), ("ess_coverage", "ESS覆盖率"), ("ess_median", "ESS中位数"), ("marker_mcse_max", "marker MCSE最大值"), ("half_chain_median_max", "半链中位差异最大值"), ("half_chain_max", "半链最大差异"), ("review_required", "需复核"), ("review_reasons", "原因")],
        ),
        "",
        f"拟合墙钟时间约 {gibbs['diagnostics']['fit']['fit_wall_sec'] / 60:.2f} 分钟。单条长链不证明整个后验已充分探索，review_required=True 时必须保留相应限制。",
        "",
        "## 4. d20_norm 聚类结果",
        "",
        markdown_table(
            summary["cluster_metrics"],
            [("method", "方法"), ("precision", "精度"), ("n_columns", "列数"), ("rank_99_percent", "99%秩"), ("active_factors", "活动列"), ("selected_active_factors", "前20列活动列"), ("n_clusters", "簇数"), ("ari_vs_pca50", "ARI vs PCA50"), ("ami_vs_pca50", "AMI vs PCA50")],
        ),
        "",
        f"Seurat PCA50 为 {summary['pca50']['n_clusters']} 簇。",
        "",
        f"CAVI 与 Gibbs 的 d20_norm 聚类一致性：ARI={summary['cavi_vs_gibbs']['ari']:.4f}、AMI={summary['cavi_vs_gibbs']['ami']:.4f}。",
        "",
        "### 协方差核查",
        "",
        "协方差对角线比较：" + json.dumps(summary["covariance_diagonal_comparison"], ensure_ascii=False),
        "",
        f"非对角 marker 协方差均值差的平均绝对值为 {summary['covariance_marker_pairs']['mean_absolute_mean_difference']:.4g}；缺失于 HVG 的基因对：{summary['covariance_marker_pairs']['missing_pairs']}。CAVI 的 mean-field 后验 SD 不能解释为对真实后验更确定。",
        "",
        "活动列数、实际建图列数、99%秩和簇数是不同概念。",
        "",
        "## 5. Leiden 种子与近邻数稳定性",
        "",
        markdown_table(
            stability_rows,
            [("method", "方法"), ("representation", "表示"), ("seed_pairwise_ari_min", "种子ARI最小"), ("seed_pairwise_ari_median", "种子ARI中位数"), ("neighbor_ari_min", "近邻ARI最小"), ("neighbor_cluster_counts", "k=10/20/45簇数")],
        ),
        "",
        "## 6. 图形与数据",
        "",
        "- [三种工作流的 UMAP](figures/mgp_seurat/comparison_umap.png)",
        "- [CAVI ELBO](figures/mgp_seurat/cavi_elbo.png)、[Gibbs ESS](figures/mgp_seurat/gibbs_ess.png)",
        "- [CAVI 载荷范数](figures/mgp_seurat/loading_column_norms_cavi.png)、[Gibbs 载荷范数](figures/mgp_seurat/loading_column_norms_gibbs.png)",
        "- [d20_norm 得分谱](figures/mgp_seurat/score_spectra.png)、[簇大小](figures/mgp_seurat/cluster_sizes.png)",
        "- [比较汇总](data/seurat_comparison.json)、[聚类指标](data/seurat_cluster_metrics.csv)、[逐簇细胞数](data/seurat_cluster_sizes.csv)",
        "- [Leiden 调参记录](data/leiden9_tuning.json)、[分辨率搜索](data/leiden9_resolution_search.csv)",
        "- [协方差对角线](data/seurat_covariance_diagonal.json)、[非对角协方差](data/seurat_covariance_marker_pairs.csv)",
        "- 各方法的 d20_norm/PCA50 marker dotplot 位于 `PBMC/figures/mgp_seurat/`。",
        "",
    ]
    output_path = Path(output_path)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path


def validate_outputs():
    import re

    from PBMC.data_processing import load_baseline, prepare_model_input

    adatas, payloads = load_models()
    expected_hash = adatas["cavi"].uns[CAVI_RUN]["source"]["model_input"]["sha256"]
    prepared = prepare_model_input(load_baseline(), "seurat")
    assert prepared.source["model_input"]["sha256"] == expected_hash
    for method, run_id, dtype in (
        ("cavi", CAVI_RUN, np.float64),
        ("gibbs", GIBBS_RUN, np.float32),
    ):
        with np.load(DATA_DIR / f"{run_id}_model.npz", allow_pickle=False) as z:
            assert np.array_equal(z["cell_names"], adatas[method].obs_names.to_numpy(dtype=str))
            assert np.array_equal(
                z["hvg_names"],
                adatas["cavi"].var_names[adatas["cavi"].var["highly_variable"]].to_numpy(dtype=str),
            )
            full = z["W_aligned"] if method == "cavi" else z["score_aligned"]
            assert full.shape == (2638, 50)
            assert z["score_d20_norm"].shape == (2638, 20)
            expected = (full * z["loading_column_norm"])[:, :20]
            assert np.array_equal(expected, z["score_d20_norm"])
            assert np.array_equal(
                z["score_d20_norm"], adatas[method].obsm[f"X_{run_id}_{REPRESENTATION}"]
            )
            assert np.isfinite(z["score_d20_norm"]).all()
            if method == "cavi":
                assert z["mu"].dtype == dtype and z["V"].dtype == dtype
                assert np.isfinite(z["mu"]).all() and np.isfinite(z["V"]).all()
            else:
                B = z["B_samples"]
                sigma2 = z["sigma2_samples"]
                assert B.shape == (1000, 2000, 50) and B.dtype == dtype
                assert sigma2.shape == (1000, 2000) and sigma2.dtype == dtype
                assert np.isfinite(B).all() and np.isfinite(sigma2).all() and (sigma2 > 0).all()
                conditional = payloads[method]["diagnostics"]["conditional_mean"]
                assert conditional["working_dtype"] == "float32"
                assert conditional["accumulation_dtype"] == "float64"
                assert conditional["score_dtype"] == "float64"
                assert not conditional["full_score_array_allocated"]
                with np.load(DATA_DIR / f"{run_id}_chain.npz", allow_pickle=False) as chain:
                    assert np.array_equal(chain["B_samples"], B)
                    assert np.array_equal(chain["sigma2_samples"], sigma2)
                    assert str(chain["input_sha256"]) == expected_hash
    for run_id, representation in (
        (CAVI_RUN, REPRESENTATION),
        (GIBBS_RUN, REPRESENTATION),
        (PCA_RUN, "d50"),
    ):
        summary = pd.read_csv(DATA_DIR / f"{run_id}_stability_summary.csv")
        assert set(summary["representation"]) == {representation}
        assert np.isfinite(summary.select_dtypes(include=[np.number]).to_numpy()).all()
        markers = pd.read_csv(DATA_DIR / f"{run_id}_{representation}_top5_markers.csv")
        adata = adatas["pca" if run_id == PCA_RUN else run_id.split("_")[0]]
        n_clusters = labels(adata, f"leiden_{run_id}_{representation}").nunique()
        assert len(markers) == n_clusters * 5
    report_path = PROJECT_ROOT / "PBMC" / "seurat_report.md"
    text = report_path.read_text(encoding="utf-8")
    assert all(token in text for token in (
        "float64", "float32", "d20_norm", "50 个 PC", "未执行共享子抽样", "b_sigma=1"
    ))
    links = re.findall(r"\[[^]]+\]\(([^)]+)\)", text)
    for link in links:
        assert (report_path.parent / link).is_file(), f"missing Seurat report link: {link}"
    result = {
        "status": "passed",
        "validated_at": datetime.now().astimezone().isoformat(),
        "cavi_precision": "float64",
        "gibbs_precision": "float32",
        "gibbs_chain_shape": [1000, 2000, 50],
        "score_shape": [2638, 20],
        "pca_dimensions": [2638, 50],
        "input_sha256": expected_hash,
        "report_links": len(links),
        "report": str(report_path),
    }
    (DATA_DIR / "seurat_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "compare", "stability", "figures", "report", "validate"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--method", choices=("all", "cavi", "gibbs", "pca"), default="all", help="method for the stability stage")
    args = parser.parse_args()
    if args.stage in ("all", "compare"):
        summary = compare()
        print(json.dumps({
            "cavi_vs_gibbs": summary["cavi_vs_gibbs"],
            "cluster_metrics": summary["cluster_metrics"],
        }, indent=2), flush=True)
    if args.stage in ("all", "stability"):
        stability(overwrite=args.overwrite, method=args.method)
    if args.stage in ("all", "figures"):
        paths = figures()
        print(json.dumps({"figures": [str(path) for path in paths]}, indent=2), flush=True)
    if args.stage in ("all", "report"):
        print(report())
    if args.stage in ("all", "validate"):
        print(json.dumps(validate_outputs(), indent=2), flush=True)


if __name__ == "__main__":
    main()
