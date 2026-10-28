# PBMC3k 固定工作流程

本文件固定当前已验证并获得满意结果的 PBMC 分析流程。此后仅维护这一套流程，不再保留其他输入模式、表示方法或历史分析分支。

## 1. 唯一工作流程

```text
固定的 X_scaled
→ PCA（计算 50 个 PC，全部 50 个用于邻域图，仅作参照）
→ CAVI / Gibbs（K=50，center=False，standardize=False）
→ RSP 对齐
→ 全部 50 列 score 按载荷列 L2 范数加权
→ 取前 20 列（d20_norm）
→ 欧氏邻域图（n_neighbors=10）
→ Leiden 调参（CAVI=0.425，Gibbs=0.500）
→ CAVI 与 Gibbs 均为 9 簇
→ Wilcoxon top 5 markers、UMAP、诊断、稳定性与报告
```

这里的“对齐”指数据表示、score 构造、下游分析、诊断和存档约定一致，不表示把 MGP 改成 LH-BFM。

## 2. 固定输入与预处理

- 上游 baseline：`../LH_BFM/PBMC/data/notebook_scaled_pca_baseline.h5ad`。
- 从 2000 个 HVG 的 `lognorm` 出发，回归 `total_counts` 和 `pct_counts_mt`，再执行 `sc.pp.scale(max_value=10)`，得到 `X_scaled`。
- 输入形状固定为 `2638 × 2000`，所有值有限且位于 `[-10,10]`。
- 核对 baseline、原始 PBMC3k 文件、实际 `X_scaled` 的 SHA-256，以及细胞和 HVG 的顺序。
- PCA、CAVI、Gibbs 必须使用同一份 `X_scaled`。
- MGP estimator 不再居中或标准化：`center=False`、`standardize=False`。
- marker 分析只使用全部质控基因的 `lognorm`，不使用截断后的模型输入。

## 3. 固定模型设置

| 项目 | CAVI | Gibbs |
|---|---|---|
| 截断上限 | `K=50` | `K=50` |
| 工作精度 | float64 | float32 |
| 先验 | `nu=3, a1=2.1, a2=3.1, a_sigma=1, b_sigma=1` | 同左 |
| 初始化 | 单个确定性 PPCA 起点 | 主因子初始化 |
| 正式运行 | `tol=0.1, drop_budget=0.5, max_iter=1000` | `n_iter=20000, burn_in=10000, thin=10, seed=2026` |
| 保留结果 | 达到 `tol` 的变分状态 | 1000 个参数样本 |
| 后处理 | 默认 RSP | 默认 RSP |

补充规则：

- CAVI 达到 `max_iter` 但未满足 `tol` 时不能标记正式完成；使用保存的 `mu`、`V`、`W`、`Phi`、`beta_delta` warm start 继续，并记录累计轮数。
- Gibbs 跨样本累计使用 float64。
- Gibbs checkpoint 只能在配置、输入哈希和 float32 精度完全一致时复用，不能复用其他历史链。
- 不修改 `mgp/` 推断引擎，不复制 LH-BFM 的 `a,c1,c2` 参数化。

## 4. RSP 与 score 的固定构造

1. 对 CAVI 的变分均值和 Gibbs 保留载荷样本分别执行默认 RSP。
2. CAVI 使用 `W_aligned_` 和 `mu_aligned_` 的列 L2 范数。
3. Gibbs 对每个参数样本计算条件均值，应用该样本的 RSP 变换，再分批平均；不使用随机 factor-score 抽样，不保存完整 `(S,K,n)` 数组。
4. 先在全部 50 列上乘载荷列 L2 范数，再取列 `0:20`，顺序不得交换。
5. 最终 score 形状固定为 `2638 × 20`。
6. 分别保存完整活动因子数和前 20 列活动因子数，不把活动列数解释为独立生物因子数。

## 5. PCA 与 Leiden 的固定设置

### PCA 参照

- 计算 50 个 PC，并将全部 50 个用于邻域图。
- `zero_center=True`、`svd_solver="arpack"`、`random_state=0`。
- Leiden `resolution=0.7`，仅作参照，不作为细胞类型真值。

### MGP 的 9 簇调参

- 复用既有邻域图和 UMAP，仅调整 Leiden，不重新拟合 MGP。
- CAVI：`resolution=0.425`。
- Gibbs：`resolution=0.500`。
- 两者均为 9 簇；选择规则为“达到 9 簇后，最接近原 0.45”。
- 固定 `random_state=0`、`flavor="igraph"`、`n_iterations=2`、`directed=False`。
- 原 0.45 标签和 marker 结果以 `_r0p45` 后缀保留。
- 调参由 `PBMC/tune_leiden.py` 固化，搜索记录保存为 `leiden9_tuning.json` 和 `leiden9_resolution_search.csv`。

## 6. Marker、图片和解释约定

- 在全部质控基因的 `lognorm` 上运行 Wilcoxon，`use_raw=False`，BH 校正，每簇输出 top 5。
- CSV 同时保存 score、log fold change、原始 p 值和校正后 p 值。
- UMAP 只用于展示；ARI/AMI 只表示聚类一致性，不称为准确率。
- 图片中的簇编号从 1 开始；底层 H5AD、NPZ 和 marker 文件保留原始从 0 开始的编号。
- 保留 Gibbs 的 ESS、tail ESS、MCSE、近常数列、半链 score 稳定性和 `review_required` 诊断。
- 单条长链不证明整个后验已充分探索，CAVI mean-field 后验也不能解释为对真实后验更确定。

## 7. 固定 run id 与当前确认结果

Run id：

- `pca_mgp_seurat_aligned`
- `cavi_seurat_k50_d20`
- `gibbs_seurat_k50_n20000_b10000_t10_d20`

当前确认结果：

| 项目 | PCA | CAVI | Gibbs |
|---|---:|---:|---:|
| 簇数 | 7 | 9 | 9 |
| 建图列数 | 50 | 20 | 20 |
| 完整活动因子数 | — | 50 | 50 |
| 前 20 列活动因子数 | — | 20 | 20 |
| RSP `status_code` | — | 1 | 1 |

- CAVI 与 Gibbs 的聚类一致性：ARI=0.7776、AMI=0.8460。
- 当前 Gibbs `review_required=False`，其 ESS/MCSE 和半链诊断仍完整保留。
- 上述数值是本次固定输入与配置的确认结果，不作为其他数据或模型上的强制目标。

## 8. 唯一复现入口

在仓库根目录执行：

```bash
python -m pip install -e ".[pbmc]"
python PBMC/data_processing.py

python PBMC/run_cavi.py --smoke
python PBMC/run_gibbs.py --smoke

python PBMC/PCA.py
python PBMC/run_cavi.py --max-iter 1000
python PBMC/run_gibbs.py

python PBMC/tune_leiden.py
python PBMC/seurat_analysis.py --stage stability --overwrite
python PBMC/seurat_analysis.py --stage all
python PBMC/build_notebook.py
```

smoke/pilot 只用于功能检查，不进入正式报告。正式结果默认不被静默覆盖；续跑和恢复必须遵循第 3 节的配置与输入约束。

## 9. 固定交付物与验收

- PCA、CAVI、Gibbs 的 H5AD 和模型 NPZ。
- Gibbs 原始链 checkpoint。
- `d20_norm` score、活动因子、ELBO、RSP、ESS/MCSE 和稳定性诊断。
- ARI/AMI 比较、逐簇细胞数及每簇 top 5 marker CSV/Markdown。
- UMAP、ELBO、ESS、score 谱、RSP 前后载荷列范数、marker dotplot 和簇大小图。
- `PBMC/seurat_report.md`、`PBMC/reproduce.md`、`PBMC.ipynb` 和 `PBMC/data/seurat_validation.json`。

验收必须同时满足：

- 输入哈希、配置、链形状和 checkpoint 一致；
- CAVI 满足 `tol`，两种 RSP 均为 `status_code=1`；
- 两种 score 都是先加权再取前 20 列；
- CAVI 和 Gibbs 均为 9 簇，每簇恰好 5 个 marker；
- 图片簇编号从 1 开始；
- `seurat_analysis.py --stage validate` 返回 `passed`，报告内链接全部有效。

## 10. 已移除的其他流程

以下不再属于项目活动流程：

- standardized/log 输入分支；
- K=40 和其他历史模型配置；
- `full`、`full_norm`、`d10_norm`、`active_norm` 表示；
- 历史多起点主分析、共享子抽样及旧版比较报告；
- 其他 legacy 输入、输出和分析入口。

历史数据文件仅作为存档，不作为可继续执行的工作流程。后续修改必须在本固定流程上完成，并重新通过全部验收。
