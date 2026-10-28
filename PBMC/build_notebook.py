from pathlib import Path
import textwrap

import nbformat
from nbclient import NotebookClient


def build_and_execute():
    root = Path(__file__).resolve().parents[1]
    nb = nbformat.v4.new_notebook()
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    })

    def md(text):
        nb.cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def code(text):
        nb.cells.append(nbformat.v4.new_code_cell(textwrap.dedent(text).strip()))

    md("""
    # PBMC3k：PCA、CAVI 与 Gibbs 最终结果汇报

    本 notebook 汇总当前固定工作流程的最终结果：`X_scaled`、`K=50`、`b_sigma=1`、RSP、载荷列 L2 范数加权、前 20 列 `d20_norm`。CAVI 使用 float64，Gibbs 使用 float32；Leiden 分辨率分别为 **0.425** 和 **0.500**，两种方法均为 **9 簇**。

    图片中的簇编号从 1 开始，底层结果文件保留原始从 0 开始的编号。ARI/AMI 仅表示聚类一致性，不表示细胞类型准确率。本 notebook 只读取已保存结果，不重新拟合模型。
    """)

    code("""
    from pathlib import Path
    import json
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from IPython.display import Image, display
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.width', 180)

    ROOT = Path.cwd()
    if not (ROOT / 'PBMC').is_dir():
        ROOT = ROOT.parent
    DATA = ROOT / 'PBMC' / 'data'
    FIGURES = ROOT / 'PBMC' / 'figures' / 'mgp_seurat'

    RUNS = {
        'PCA': 'pca_mgp_seurat_aligned',
        'CAVI': 'cavi_seurat_k50_d20',
        'GS': 'gibbs_seurat_k50_n20000_b10000_t10_d20',
    }
    REPRESENTATIONS = {'PCA': 'd50', 'CAVI': 'd20_norm', 'GS': 'd20_norm'}
    ORDER = ['PCA', 'CAVI', 'GS']

    adatas = {
        name: sc.read_h5ad(DATA / f'{run_id}.h5ad')
        for name, run_id in RUNS.items()
    }
    payloads = {}
    for name in ('CAVI', 'GS'):
        with np.load(DATA / f'{RUNS[name]}_model.npz', allow_pickle=False) as z:
            payloads[name] = json.loads(z['diagnostics_json'].item())
    comparison = json.loads((DATA / 'seurat_comparison.json').read_text(encoding='utf-8'))
    validation = json.loads((DATA / 'seurat_validation.json').read_text(encoding='utf-8'))

    def labels(name):
        key = f"leiden_{RUNS[name]}_{REPRESENTATIONS[name]}"
        raw = adatas[name].obs[key].astype(int)
        return pd.Series((raw + 1).astype(str), index=raw.index, dtype='category')

    print({name: adatas[name].shape for name in ORDER})
    print({'validation': validation['status'], 'input_sha256': validation['input_sha256']})
    """)

    md("""
    ## 1. 聚类数量与 ARI/AMI

    PCA 仅作参照，不作为细胞类型真值。AMI 为 adjusted mutual information。
    """)

    code("""
    rows = []
    for name in ORDER:
        rows.append({
            'Method': name,
            'Clusters': labels(name).nunique(),
            'ARI vs PCA50': adjusted_rand_score(labels('PCA'), labels(name)),
            'AMI vs PCA50': adjusted_mutual_info_score(labels('PCA'), labels(name)),
        })
    metrics = pd.DataFrame(rows).set_index('Method')
    display(metrics.round(4))
    print(f"CAVI vs GS: ARI={comparison['cavi_vs_gibbs']['ari']:.4f}, AMI={comparison['cavi_vs_gibbs']['ami']:.4f}")
    """)

    code("""
    cluster_sizes = pd.DataFrame({name: labels(name).value_counts().sort_index() for name in ORDER})
    cluster_sizes = cluster_sizes.astype('Int64')
    cluster_sizes.index.name = 'Cluster'
    display(cluster_sizes)
    """)

    md("""
    ## 2. UMAP 聚类图

    UMAP 仅用于展示结构，不能单独作为算法优劣的证据。
    """)

    code("""
    display(Image(filename=str(FIGURES / 'comparison_umap.png'), width=1200))
    """)

    md("""
    ## 3. 每簇 Top 5 markers

    Marker 分析使用全部质控基因的 `lognorm`、Wilcoxon 检验和 BH 校正。完整 score、log fold change 和 p 值保存在对应 CSV 中。
    """)

    code("""
    marker_files = {
        'PCA': DATA / 'pca_mgp_seurat_aligned_d50_top5_markers.csv',
        'CAVI': DATA / 'cavi_seurat_k50_d20_d20_norm_top5_markers.csv',
        'GS': DATA / 'gibbs_seurat_k50_n20000_b10000_t10_d20_d20_norm_top5_markers.csv',
    }
    marker_rows = []
    for name in ORDER:
        frame = pd.read_csv(marker_files[name])
        for raw_cluster, group in frame.groupby('cluster', sort=True):
            marker_rows.append({
                'Method': name,
                'Cluster': int(raw_cluster) + 1,
                'Top 5 markers': ', '.join(group['gene'].head(5).astype(str)),
            })
    marker_table = pd.DataFrame(marker_rows)
    display(marker_table)
    """)

    md("""
    ## 4. 模型、调参与采样诊断

    CAVI 的轮数包含首次运行与 warm-start 续跑。Gibbs 的 ESS、MCSE 和半链诊断均保留；单条长链不证明整个后验已充分探索。
    """)

    code("""
    cavi_d = payloads['CAVI']['diagnostics']
    gs_d = payloads['GS']['diagnostics']
    cavi_fit = cavi_d['fit']
    gs_fit = gs_d['fit']
    gs_sampling = gs_d['sampling']
    cavi_config = payloads['CAVI']['config']
    gs_config = payloads['GS']['config']

    rows = [
        ('Input', 'X_scaled', 'X_scaled'),
        ('K', 50, 50),
        ('Precision', 'float64', 'float32'),
        ('Prior', 'nu=3, a1=2.1, a2=3.1, a_sigma=1, b_sigma=1', 'same'),
        ('RSP status', cavi_d['rsp']['status_code'], gs_d['rsp']['status_code']),
        ('Score weighting', 'loading_column_l2_norm', 'loading_column_l2_norm'),
        ('Report columns', 20, 20),
        ('Leiden resolution', cavi_config['leiden_resolution'], gs_config['leiden_resolution']),
        ('Report clusters', labels('CAVI').nunique(), labels('GS').nunique()),
        ('Fit sweeps', cavi_fit.get('total_sweeps', cavi_fit['n_iter']), gs_fit['n_iter']),
        ('Burn-in', '—', gs_fit['burn_in']),
        ('Thin', '—', gs_fit['thin']),
        ('Kept draws', '—', gs_fit['n_kept']),
        ('Converged / Gamma', f"{cavi_fit['converged']} / {cavi_fit['stop_reason']}", gs_fit['gamma_mode']),
        ('Active factors (full)', cavi_d['representation']['active_factor_count'], gs_d['representation']['active_factor_count']),
        ('Active factors (first 20)', cavi_d['representation']['representations']['d20_norm']['selected_active_factor_count'], gs_d['representation']['representations']['d20_norm']['selected_active_factor_count']),
        ('Active-loading ESS>=400 coverage', '—', gs_sampling['active_loading_bulk_ess']['coverage_at_400']),
        ('Median bulk ESS (active loadings)', '—', gs_sampling['active_loading_bulk_ess']['median']),
        ('Minimum bulk ESS (loadings)', '—', gs_sampling['bulk_summary']['B']['min']),
        ('Minimum bulk ESS (noise variances)', '—', gs_sampling['bulk_summary']['sigma2']['min']),
        ('Review required', '—', gs_sampling['review_required']),
        ('Validation', validation['status'], validation['status']),
    ]
    diagnostics = pd.DataFrame(rows, columns=['Item', 'CAVI', 'GS'])
    display(diagnostics)
    """)

    md("""
    ## 5. RSP 对齐前后载荷列 L2 范数

    CAVI 比较 `mu_` 与 `mu_aligned_`；Gibbs 比较原始载荷后验均值与 RSP 对齐后验均值。
    """)

    code("""
    print('CAVI')
    display(Image(filename=str(FIGURES / 'loading_column_norms_cavi.png'), width=1000))
    print('GS')
    display(Image(filename=str(FIGURES / 'loading_column_norms_gibbs.png'), width=1000))
    """)

    md("""
    ## 6. Leiden 种子与近邻数稳定性
    """)

    code("""
    stability_frames = []
    for name in ORDER:
        frame = pd.read_csv(DATA / f'{RUNS[name]}_stability_summary.csv')
        frame.insert(0, 'Method', name)
        stability_frames.append(frame)
    stability = pd.concat(stability_frames, ignore_index=True)
    stability = stability[[
        'Method', 'representation', 'seed_pairwise_ari_min',
        'seed_pairwise_ari_median', 'neighbor_ari_min', 'neighbor_cluster_counts'
    ]]
    display(stability.round(4))
    """)

    md("""
    ## 7. 其他诊断图
    """)

    code("""
    for filename in ('cluster_sizes.png', 'score_spectra.png', 'cavi_elbo.png', 'gibbs_ess.png'):
        print(filename)
        display(Image(filename=str(FIGURES / filename), width=1000))
    """)

    md("""
    ## 8. 最终验收

    - CAVI 满足 `tol`，两种 RSP 均为 `status_code=1`。
    - 两种 score 均先按载荷列范数加权，再取前 20 列。
    - CAVI 与 Gibbs 均为 9 簇，每簇恰好 5 个 marker。
    - 图片簇编号从 1 开始。
    - `seurat_analysis.py --stage validate` 返回 `passed`，报告内链接全部有效。

    复现命令和固定配置见 `PBMC/reproduce.md`，唯一工作流程见 `PBMC/plan.md`。
    """)

    code("""
    summary = pd.DataFrame([
        ('Status', validation['status']),
        ('CAVI precision', validation['cavi_precision']),
        ('Gibbs precision', validation['gibbs_precision']),
        ('Gibbs chain shape', tuple(validation['gibbs_chain_shape'])),
        ('Score shape', tuple(validation['score_shape'])),
        ('PCA dimensions', tuple(validation['pca_dimensions'])),
        ('Report links', validation['report_links']),
    ], columns=['Item', 'Value'])
    display(summary)
    """)

    path = root / "PBMC.ipynb"
    nbformat.write(nb, path)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    nbformat.write(nb, path)
    print(f"Executed and saved {path} with {len(nb.cells)} cells")


if __name__ == "__main__":
    build_and_execute()
