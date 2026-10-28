# Seurat 输入下的 MGP 对齐分析报告

生成时间：2026-10-28T16:47:12.164520+08:00

## 1. 配置与解释边界

- 输入为 2000 HVG 的 lognorm，经 total_counts、pct_counts_mt 回归及 sc.pp.scale(max_value=10) 处理；MGP 不再居中或标准化。
- 输入矩阵 SHA-256：`689dc2d0912196ff1b0883d6caef514e56e39dc10f3cf3eaad3e23eb3653dbeb`。CAVI 使用 float64，Gibbs 采样使用 float32，跨样本累计使用 float64。
- 两种 MGP 均为 K=50，先验为 nu=3、a1=2.1、a2=3.1、a_sigma=1、b_sigma=1；主表示为 RSP 后的 d20_norm。
- Seurat PCA 计算并全部使用 50 个 PC，仅作参照。未执行共享子抽样，也未运行额外的 Gibbs float64 对照链。
- ARI/AMI 是聚类一致性而非细胞类型准确率；UMAP 只用于展示。按用户指定目标调整 Leiden 分辨率：CAVI=0.425、Gibbs=0.500，两种方法均为 9 簇；原 0.45 标签另行保留。

## 2. CAVI float64

| 起点 | 轮数 | 最终ELBO | 停止 | 被选择 |
|---|---|---|---|---|
| start_0 | 256 | -6682220.394691 | tol | True |

拟合墙钟时间约 0.67 分钟。

## 3. Gibbs float32

n_iter=20000、burn_in=10000、thin=10、seed=2026、保留1000个样本、exact Gamma。

活动载荷 ESS≥400 覆盖率 95.01%；marker 协方差 MCSE/均值最大值 0.004496。

| 结果 | ESS覆盖率 | ESS中位数 | marker MCSE最大值 | 半链中位差异最大值 | 半链最大差异 | 需复核 | 原因 |
|---|---|---|---|---|---|---|---|
| gibbs_seurat_k50_n20000_b10000_t10_d20 | 0.9501 | 869.2 | 0.004496 | 0.01278 | 0.02796 | False | 上述阈值均未触发 |

拟合墙钟时间约 64.51 分钟。单条长链不证明整个后验已充分探索，review_required=True 时必须保留相应限制。

## 4. d20_norm 聚类结果

| 方法 | 精度 | 列数 | 99%秩 | 活动列 | 前20列活动列 | 簇数 | ARI vs PCA50 | AMI vs PCA50 |
|---|---|---|---|---|---|---|---|---|
| cavi | float64 | 20 | 17 | 50 | 20 | 9 | 0.6148 | 0.7732 |
| gibbs | float32 | 20 | 17 | 50 | 20 | 9 | 0.527 | 0.7434 |

Seurat PCA50 为 7 簇。

CAVI 与 Gibbs 的 d20_norm 聚类一致性：ARI=0.7776、AMI=0.8460。

### 协方差核查

协方差对角线比较：{"pearson_correlation": 0.9999759180711525, "relative_l2_difference": 0.0019400062462048779, "median_absolute_relative_difference": 0.0008794887002678575}

非对角 marker 协方差均值差的平均绝对值为 0.006023；缺失于 HVG 的基因对：[['IL7R', 'CD3D']]。CAVI 的 mean-field 后验 SD 不能解释为对真实后验更确定。

活动列数、实际建图列数、99%秩和簇数是不同概念。

## 5. Leiden 种子与近邻数稳定性

| 方法 | 表示 | 种子ARI最小 | 种子ARI中位数 | 近邻ARI最小 | k=10/20/45簇数 |
|---|---|---|---|---|---|
| cavi_f64 | d20_norm | 0.7144 | 0.8046 | 0.6862 | [9, 8, 6] |
| gibbs_f32 | d20_norm | 0.7041 | 0.7669 | 0.7182 | [9, 8, 8] |
| pca50 | d50 | 0.9561 | 0.985 | 0.9699 | [7, 7, 7] |

## 6. 图形与数据

- [三种工作流的 UMAP](figures/mgp_seurat/comparison_umap.png)
- [CAVI ELBO](figures/mgp_seurat/cavi_elbo.png)、[Gibbs ESS](figures/mgp_seurat/gibbs_ess.png)
- [CAVI 载荷范数](figures/mgp_seurat/loading_column_norms_cavi.png)、[Gibbs 载荷范数](figures/mgp_seurat/loading_column_norms_gibbs.png)
- [d20_norm 得分谱](figures/mgp_seurat/score_spectra.png)、[簇大小](figures/mgp_seurat/cluster_sizes.png)
- [比较汇总](data/seurat_comparison.json)、[聚类指标](data/seurat_cluster_metrics.csv)、[逐簇细胞数](data/seurat_cluster_sizes.csv)
- [Leiden 调参记录](data/leiden9_tuning.json)、[分辨率搜索](data/leiden9_resolution_search.csv)
- [协方差对角线](data/seurat_covariance_diagonal.json)、[非对角协方差](data/seurat_covariance_marker_pairs.csv)
- 各方法的 d20_norm/PCA50 marker dotplot 位于 `PBMC/figures/mgp_seurat/`。
