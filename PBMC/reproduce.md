# PBMC3k 固定工作流程复现说明

本文件仅对应 `PBMC/plan.md` 中固定的当前流程。其他输入模式、表示方法、K=40、共享子抽样及历史比较报告已移除，历史数据仅作存档。

## 1. 固定配置

- 输入：`../LH_BFM/PBMC/data/notebook_scaled_pca_baseline.h5ad` 中 2000 HVG 的 `lognorm`，经协变量回归和 `sc.pp.scale(max_value=10)` 得到 `X_scaled`。
- MGP：`K=50`、`center=False`、`standardize=False`。
- 先验：`nu=3, a1=2.1, a2=3.1, a_sigma=1, b_sigma=1`。
- CAVI：float64、单个确定性 PPCA 起点、`tol=0.1`、`drop_budget=0.5`、`max_iter=1000`。
- Gibbs：float32，`n_iter=20000, burn_in=10000, thin=10, seed=2026`，保留 1000 个样本，跨样本累计使用 float64。
- 表示：RSP 后先在全部 50 列上按载荷列 L2 范数加权，再取前 20 列，得到 `d20_norm`。
- PCA：计算并全部使用 50 个 PC，Leiden `resolution=0.7`，仅作参照。
- Leiden：CAVI `resolution=0.425`，Gibbs `resolution=0.500`，均为 9 簇；原 0.45 结果以 `_r0p45` 后缀保留。
- marker：全部质控基因的 `lognorm`，Wilcoxon + BH，每簇 top 5。
- 图片簇编号从 1 开始，底层结果文件保留原始从 0 开始的编号。

## 2. 环境与输入检查

在仓库根目录执行：

```bash
python -m pip install -e ".[pbmc]"
python PBMC/data_processing.py
```

加载器核对 baseline、原始 PBMC3k SHA-256、2638×13714 的表达数据、2000 HVG 及细胞和基因顺序。模型拟合时重新构建并记录实际 `X_scaled` 的 SHA-256。

## 3. Smoke 检查

```bash
python PBMC/run_cavi.py --smoke
python PBMC/run_gibbs.py --smoke
```

smoke/pilot 只用于功能、编译和资源检查，不进入正式报告。

## 4. 正式运行

```bash
python PBMC/PCA.py
python PBMC/run_cavi.py --max-iter 1000
python PBMC/run_gibbs.py
python PBMC/tune_leiden.py
python PBMC/seurat_analysis.py --stage stability --overwrite
python PBMC/seurat_analysis.py --stage all
python PBMC/build_notebook.py
```

固定 run id：

- `pca_mgp_seurat_aligned`
- `cavi_seurat_k50_d20`
- `gibbs_seurat_k50_n20000_b10000_t10_d20`

CAVI 达到 `max_iter` 但未满足 `tol` 时不能标记正式完成。使用模型 NPZ 中的 `mu`、`V`、`W`、`Phi`、`beta_delta` warm start 继续，并记录 `previous_sweeps`、`additional_sweeps` 和 `total_sweeps`。只有续跑满足 `tol` 后才能保存为正式结果。

Gibbs 后处理若中断，可从同一配置、输入哈希和 float32 精度的完整 checkpoint 恢复：

```bash
python PBMC/run_gibbs.py \
  --reuse-chain \
  --chain-path PBMC/data/gibbs_seurat_k50_n20000_b10000_t10_d20_chain.npz
```

这不是断点续采样，其他配置或历史链不可复用。

## 5. 分阶段执行与验收

```bash
python PBMC/seurat_analysis.py --stage compare
python PBMC/seurat_analysis.py --stage stability
python PBMC/seurat_analysis.py --stage figures
python PBMC/seurat_analysis.py --stage report
python PBMC/seurat_analysis.py --stage validate
```

主要交付物：

- H5AD、模型 NPZ 和 Gibbs 原始链 checkpoint；
- `d20_norm` score、活动因子、ELBO、RSP、ESS/MCSE 和半链稳定性诊断；
- ARI/AMI 比较、逐簇细胞数及 top 5 marker CSV/Markdown；
- UMAP、ELBO、ESS、score 谱、RSP 前后载荷列范数、marker dotplot 和簇大小图；
- `leiden9_tuning.json`、`leiden9_resolution_search.csv`；
- `PBMC/seurat_report.md`、`PBMC.ipynb` 和 `PBMC/data/seurat_validation.json`。

验收要求：CAVI 满足 `tol`，两种 RSP 均为 `status_code=1`，两种 score 均先加权再取前 20 列，CAVI/Gibbs 均为 9 簇，每簇恰好 5 个 marker，图片编号从 1 开始，最终验证返回 `passed` 且报告链接全部有效。
