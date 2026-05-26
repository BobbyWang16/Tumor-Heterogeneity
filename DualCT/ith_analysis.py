import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import os

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.exceptions import ConvergenceWarning
from scipy import ndimage as ndi

warnings.filterwarnings("ignore")


class NSCLCHeterogeneityAnalyzer:
    """
    非小细胞肺癌(NSCLC)双期CT异质性分析

    基于NSCLC病理生物学机制设计的三类聚类：
        Class 1: 坏死/低活性区 (Necrosis/Low-activity)
                 - 平扫期呈软组织密度(N ~ 20-60 HU)
                 - 增强后几乎不增强(delta ~ -10 to +10 HU)
                 - 局部纹理均匀(low local_std)
        Class 2: 间质/混合区 (Stroma/Mixed)
                 - 平扫期软组织密度(N ~ 30-80 HU)
                 - 中等程度增强(delta ~ 10-40 HU)
                 - 纹理可能不均匀
        Class 3: 富血供/高活性区 (Rich-vascular/High-activity)
                 - 平扫期软组织密度(N ~ 40-100 HU)
                 - 明显增强(delta > 40 HU)
                 - 代表活性肿瘤实质

    核心设计思想:
    1. 密度预滤波: 排除N < -50 HU的极低密度体素(肺组织/空气混入)
    2. 三维特征空间: [平扫密度, 绝对增强值, 局部纹理异质性]
    3. 以"增强程度"为核心的聚类策略: 坏死<->间质<->富血供
    4. 自适应空间约束: 根据肿瘤体积调整连通域过滤阈值
    5. NSCLC专用描述符: 纹理异质性、增强梯度、交界带特征等
    """

    def __init__(self, noncontrast_ct, contrast_ct, tumor_mask, density_thresh=-50):
        """
        Args:
            noncontrast_ct: 平扫期CT (HU)
            contrast_ct: 增强期CT (HU)
            tumor_mask: ROI二值掩膜 (bool或0/1)
            density_thresh: 密度预滤波阈值, 低于此值的体素排除
        """
        self.noncontrast = noncontrast_ct.astype(np.float32)
        self.contrast = contrast_ct.astype(np.float32)
        self.mask_raw = tumor_mask.astype(bool)
        self.density_thresh = density_thresh

        # 密度预滤波: 排除极低密度体素(肺组织/空气/伪影)
        self.mask = self.mask_raw & (self.noncontrast >= density_thresh)
        self.mask_excluded = self.mask_raw & (~self.mask)  # 被排除的体素

        self.labels = None           # 聚类标签(对有效体素)
        self.labels_3d = None        # 3D标签图
        self.features = None         # 特征矩阵
        self.local_std = None        # 局部标准差图(3D)
        self.dist_to_boundary = None # 到肿瘤边界的距离

    def _kernel_smooth_masked(self, data, mask, size=3):
        """
        只对mask内邻居做3D均值平滑

        关键改进: 不同于uniform_filter(cval=0)会在边界处引入外部0值,
        此方法仅对mask内的有效邻居做平均, 避免边界体素被mask外区域拉低。

        Args:
            data: 3D CT数据
            mask: 有效区域掩膜(bool)
            size: kernel大小
        Returns:
            smoothed: mask内体素的邻域均值
        """
        kernel = np.ones((size, size, size), dtype=np.float32)
        masked_data = data.astype(np.float32) * mask.astype(np.float32)
        # 邻域内有效体素数量
        neighbor_count = ndi.convolve(mask.astype(np.float32), kernel,
                                      mode='constant', cval=0)
        # 邻域内有效体素之和
        neighbor_sum = ndi.convolve(masked_data, kernel,
                                    mode='constant', cval=0)
        # 避免除零: 无有效邻居时保持原值
        result = np.where(neighbor_count > 0,
                          neighbor_sum / neighbor_count,
                          data.astype(np.float32))
        return result

    def _compute_local_std(self, data, size=3):
        """
        计算3D局部标准差, 反映纹理异质性

        NSCLC生物学意义:
        - 高局部std: 坏死-活性交界区、纤维化与肿瘤混合区
        - 低局部std: 纯坏死区(均匀低密度)、纯富血供区(均匀高密度)
        """
        mean = ndi.uniform_filter(data, size=size, mode='constant', cval=0)
        mean_sq = ndi.uniform_filter(data ** 2, size=size, mode='constant', cval=0)
        var = np.maximum(mean_sq - mean ** 2, 0)
        return np.sqrt(var)

    def _compute_dist_to_boundary(self):
        """计算肿瘤内部每个体素到边界的距离"""
        return ndi.distance_transform_edt(self.mask)

    def extract_features(self, kernel_size=3):
        """
        基于kernel统计的特征提取

        核心改进：每个体素的特征不是其自身像素值，而是3x3x3邻域内的
        统计量。这可以显著降低配准不精细带来的像素级偏差，使聚类更鲁棒。

        特征维度:
        1. n_mean: 3x3x3邻域内平扫期HU均值
           - 平滑了N期的配准噪声和局部伪影
        2. delta_mean: 3x3x3邻域内增强差均值 (C_mean - N_mean)
           - NSCLC核心区分特征，但基于邻域平均而非单像素
           - 配准偏移在邻域内被平均化
        3. local_std: 3x3x3邻域内增强期标准差
           - 反映局部纹理异质性

        Args:
            kernel_size: 邻域核大小(默认3，即3x3x3)

        Returns:
            features: (N_valid, 3) 特征矩阵
            indices: 有效体素在3D数组中的扁平索引
        """
        # --- Step 1: Kernel平滑(仅考虑ROI内邻居) ---
        # 对N和C分别做3D均值滤波，用邻域统计替代单像素值
        # 关键：只对mask内邻居平均，避免边界处被外部0值拉低
        n_smooth = self._kernel_smooth_masked(
            self.noncontrast, self.mask, size=kernel_size
        )
        c_smooth = self._kernel_smooth_masked(
            self.contrast, self.mask, size=kernel_size
        )

        # 只对有效体素(ROI内且密度满足阈值)提取kernel特征
        n_mean = n_smooth[self.mask]
        c_mean = c_smooth[self.mask]

        # --- Step 2: Kernel增强差 ---
        delta_mean = c_mean - n_mean

        # --- Step 3: Kernel纹理异质性 ---
        # 在平滑后的C上计算局部标准差
        self.local_std = self._compute_local_std(c_smooth, size=kernel_size)
        local_std_vals = self.local_std[self.mask]

        # 四维特征空间: [N_mean, C_mean, Delta_mean, LocalStd]
        # 同时利用N和C的绝对HU值，以及Delta和Texture
        self.features = np.column_stack([
            n_mean.astype(np.float64),
            c_mean.astype(np.float64),
            delta_mean.astype(np.float64),
            local_std_vals.astype(np.float64)
        ])

        # 保存有效体素的3D索引
        self.indices = np.argwhere(self.mask)

        # 计算到边界的距离
        self.dist_to_boundary = self._compute_dist_to_boundary()

        return self.features, self.indices

    def cluster(self, features, method='clinical_prior'):
        """三类聚类算法"""
        if len(features) < 30:
            return np.ones(len(features), dtype=int)

        if method == 'clinical_prior':
            labels = self._clinical_prior_clustering(features)
        elif method == 'gmm':
            labels = self._gmm_clustering_nsclc(features)
        else:
            labels = self._kmeans_clustering_nsclc(features)
        return labels.astype(int)

    def _clinical_prior_clustering(self, features):
        """
        基于N和Delta联合的临床先验分类

        核心思想:
        同时利用N(平扫密度)和Delta(增强差)两个维度做分类:
            坏死   = N高 + Delta低 (高密度但不增强 = 纤维化/凝固坏死)
            间质   = N中 + Delta中 (软组织 + 中等增强 = 混合间质)
            富血供 = N低 + Delta高 (软组织 + 明显增强 = 活性肿瘤)

        分类方法:
        在标准化[N, Delta]空间中, 计算每个体素到三类"临床理想点"的加权距离,
        分配到距离最近的类。Delta权重更高(核心区分维度)。
        """
        n_hu = features[:, 0]
        delta = features[:, 2]

        # 标准化N和Delta
        n_mean, n_std = n_hu.mean(), n_hu.std() + 1e-6
        d_mean, d_std = delta.mean(), delta.std() + 1e-6
        n_z = (n_hu - n_mean) / n_std
        d_z = (delta - d_mean) / d_std

        # === 修正的临床理想点 ===
        # 之前版本把"高N+低Delta"错误地归为坏死(这是纤维化!)
        # 正确的临床先验:
        #   坏死 = 低N + 低Delta (软组织低密度 + 不增强 = 液化/凝固坏死)
        #   间质 = 高N + 低Delta (软组织高密度 + 不增强 = 纤维化)
        #   富血供 = 中N + 高Delta (软组织 + 明显增强 = 活性肿瘤)
        n_p25, n_p50, n_p75 = np.percentile(n_z, [25, 50, 75])
        d_p25, d_p50, d_p75 = np.percentile(d_z, [25, 50, 75])

        ideal_necrosis = np.array([n_p25, d_p25])   # 坏死: N低 + Delta低
        ideal_stroma   = np.array([n_p75, d_p50])   # 间质: N高 + Delta中(低)
        ideal_rich     = np.array([n_p50, d_p75])   # 富血供: N中 + Delta高

        # 加权距离 (Delta权重更高, 是核心血供指标; N权重辅助区分坏死vs纤维化)
        w_n, w_d = 0.6, 1.4
        dist_necrosis = w_n * (n_z - ideal_necrosis[0])**2 + w_d * (d_z - ideal_necrosis[1])**2
        dist_stroma   = w_n * (n_z - ideal_stroma[0])**2   + w_d * (d_z - ideal_stroma[1])**2
        dist_rich     = w_n * (n_z - ideal_rich[0])**2     + w_d * (d_z - ideal_rich[1])**2

        # 分配到距离最近的类
        dists = np.stack([dist_necrosis, dist_stroma, dist_rich], axis=1)
        labels = np.argmin(dists, axis=1)

        # 标签语义: 0=坏死(低N+低Delta), 1=间质(高N+低Delta), 2=富血供(中N+高Delta)
        # 已经是临床正确排序, 无需重排
        return labels.astype(int)

    def _gmm_clustering_nsclc(self, features, min_frac=0.05):
        """
        基于临床先验的GMM聚类

        核心思想:
        临床先验定义了三类在特征空间中的"理想位置"，而非让GMM从数据中自由发现。
        GMM的作用是在临床先验锚定的基础上，根据实际数据微调边界。

        NSCLC临床先验定义:
        - 坏死: 低/无增强 (delta最低区间) + 软组织密度
        - 间质: 中等增强 (delta中间区间) + 软组织密度
        - 富血供: 明显增强 (delta最高区间) + 软组织密度

        初始化策略:
        1. 根据delta的分布范围，将空间三等分
        2. 每类的N中心 = 全局N中位数 (坏死和活性肿瘤的N值重叠)
        3. 每类的delta中心 = 对应分段的中点
        4. 用这个临床锚定初始化GMM
        """
        N = len(features)
        n_hu = features[:, 0]
        delta = features[:, 2]

        # GMM输入: [N_mean, delta_mean]
        X = np.column_stack([n_hu, delta])

        # === 临床先验锚定初始化 ===
        # 核心思想: 根据delta分位数将数据空间三等分,
        # 每类的初始中心落在对应分位段的统计中心,
        # 确保GMM从合理位置开始微调, 而非自由发现。
        #
        # 临床语义:
        #   坏死 = 相对最低增强 (delta最低的33%)
        #   间质 = 相对中等增强 (delta中间的34%)
        #   富血供 = 相对最高增强 (delta最高的33%)
        d_q33 = np.percentile(delta, 33)
        d_q67 = np.percentile(delta, 67)

        low_mask = delta <= d_q33
        mid_mask = (delta > d_q33) & (delta < d_q67)
        high_mask = delta >= d_q67

        init_means = np.array([
            [np.median(n_hu[low_mask]),  np.mean(delta[low_mask])],   # 坏死
            [np.median(n_hu[mid_mask]),  np.mean(delta[mid_mask])],   # 间质
            [np.median(n_hu[high_mask]), np.mean(delta[high_mask])],  # 富血供
        ])

        # RobustScaler标准化
        scaler = RobustScaler(quantile_range=(5, 95))
        X_scaled = scaler.fit_transform(X)
        init_means_scaled = scaler.transform(init_means)

        # GMM参数搜索
        best_labels = None
        best_score = -np.inf

        for cov in ['diag', 'full']:
            for reg in [1e-4, 5e-4, 1e-3, 1e-2]:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        gmm = GaussianMixture(
                            n_components=3,
                            covariance_type=cov,
                            reg_covar=reg,
                            means_init=init_means_scaled,
                            n_init=1,
                            max_iter=400,
                            random_state=42
                        )
                        labels = gmm.fit_predict(X_scaled)

                    unique, counts = np.unique(labels, return_counts=True)
                    if len(unique) == 3 and (counts.min() / N >= min_frac):
                        # 使用BIC评估模型
                        score = -gmm.bic(X_scaled)
                        if score > best_score:
                            best_score = score
                            best_labels = labels
                except Exception:
                    continue

        if best_labels is not None:
            return self._reorder_labels_by_enhancement(best_labels, features)

        return self._kmeans_clustering_nsclc(features)

    def _kmeans_clustering_nsclc(self, features):
        """NSCLC优化的KMeans种子初始化 (四维特征)"""
        # 使用delta分位数选择种子点 (delta在四维特征中是第3列)
        delta = features[:, 2]
        seeds_idx = [
            np.argmin(np.abs(delta - np.percentile(delta, p)))
            for p in [15, 50, 85]
        ]
        km = KMeans(n_clusters=3, init=features[seeds_idx], n_init=1, random_state=42)
        labels = km.fit_predict(features)
        return self._reorder_labels_by_enhancement(labels, features)

    @staticmethod
    def _reorder_labels_by_enhancement(labels, features):
        """
        按增强程度(delta_mean)排序标签

        NSCLC生物学语义:
        - Class 1: 最低增强(坏死/低活性)
        - Class 2: 中等增强(间质/混合)
        - Class 3: 最高增强(富血供/高活性)
        """
        # delta在四维特征中是第3列 (index 2)
        delta = features[:, 2]
        means = [delta[labels == lb].mean() for lb in np.unique(labels)]
        mapping = {old: new for new, old in enumerate(np.argsort(means))}
        return np.vectorize(mapping.get)(labels).astype(int)

    def apply_necrosis_prior(self, labels_1d, interior_q=0.4,
                             keep_top_k=1, kernel_size=3):
        """
        NSCLC自适应空间先验约束

        与原版相比的优化:
        1. 自适应连通域过滤阈值: 根据肿瘤总体素数调整
           - 小肿瘤(<5000体素): min_cc_vox = max(5, 0.3% * total)
           - 中肿瘤(5000-30000): min_cc_vox = max(10, 0.5% * total)
           - 大肿瘤(>30000): min_cc_vox = max(20, 1.0% * total)
        2. 对Class 1(坏死)应用空间约束, 但对极小碎片更宽松
        3. 边界重分配: 仅重分配远离中心的坏死碎片

        NSCLC病理学基础:
        - 坏死可位于肿瘤中央(大肿瘤)或散布(小肿瘤)
        - 鳞癌常有中央坏死伴空洞形成
        - 腺癌坏死相对少见但可呈散在分布
        """
        n_voxels = len(labels_1d)

        # 自适应连通域阈值
        if n_voxels < 5000:
            min_cc_vox = max(3, int(n_voxels * 0.003))
        elif n_voxels < 30000:
            min_cc_vox = max(5, int(n_voxels * 0.005))
        else:
            min_cc_vox = max(10, int(n_voxels * 0.01))

        labels0 = labels_1d.copy()
        labels3d = np.zeros(self.mask.shape, dtype=np.int16)
        labels3d[self.mask] = labels0

        dist = self.dist_to_boundary
        d_thr = np.quantile(dist[self.mask], interior_q)

        k = max(3, int(kernel_size))
        kernel = np.ones((k, k, k))
        kernel[k // 2, k // 2, k // 2] = 0

        # --- Step 1: 边界坏死重分配 ---
        nec = (labels3d == 0)
        boundary_nec = nec & (dist < d_thr)
        if np.any(boundary_nec):
            n1 = ndi.convolve((labels3d == 1).astype(int), kernel, mode='constant')
            n2 = ndi.convolve((labels3d == 2).astype(int), kernel, mode='constant')
            labels3d[boundary_nec] = np.where(n1[boundary_nec] >= n2[boundary_nec], 1, 2)

        # --- Step 2: 连通域过滤 ---
        nec = (labels3d == 0)
        if np.any(nec):
            struct = ndi.generate_binary_structure(3, 1)
            cc, num = ndi.label(nec, structure=struct)
            sizes = np.bincount(cc.ravel()) if num > 0 else np.array([0])
            sizes[0] = 0

            keep = np.zeros_like(sizes, dtype=bool)
            if keep_top_k and keep_top_k > 0:
                keep[np.argsort(sizes)[-keep_top_k:]] = True
            keep[sizes >= max(1, int(min_cc_vox))] = True

            nec_clean = nec & keep[cc]
            nec_clean = ndi.binary_closing(nec_clean, structure=struct, iterations=1)

            drop = nec & (~nec_clean)
            if np.any(drop):
                n1 = ndi.convolve((labels3d == 1).astype(int), kernel, mode='constant')
                n2 = ndi.convolve((labels3d == 2).astype(int), kernel, mode='constant')
                labels3d[drop] = np.where(n1[drop] >= n2[drop], 1, 2)

        return labels3d[self.mask].astype(np.uint16)

    def compute_descriptors(self):
        """
        计算NSCLC优化的异质性描述符

        核心指标:
        1. 体积与比例 (基础)
        2. 坏死中心性 (空间分布)
        3. 聚类多样性 (Shannon熵)
        4. 增强异质性 (增强值变异)
        5. 纹理异质性指数 (局部std均值)
        6. 增强梯度 (中心vs边缘差异)
        7. 坏死-活性交界带特征
        8. 增强均匀性指数
        9. 密度分布偏度
        10. 富血供集中度
        """
        if self.labels is None:
            raise ValueError("Need to run clustering first!")

        n_voxels = self.mask.sum()
        excluded_voxels = self.mask_excluded.sum()
        total_roi = self.mask_raw.sum()

        # 基础特征 (四维: [N_mean, C_mean, Delta_mean, LocalStd])
        features = self.features
        n_hu = features[:, 0]
        c_hu = features[:, 1]
        delta = features[:, 2]
        localstd = features[:, 3]
        dist = self.dist_to_boundary[self.mask]

        n1 = np.sum(self.labels == 0)  # Class 1: 坏死
        n2 = np.sum(self.labels == 1)  # Class 2: 间质
        n3 = np.sum(self.labels == 2)  # Class 3: 富血供
        total_valid = n1 + n2 + n3

        # 获取体素尺寸(如果可用)
        voxel_vol = 1.0

        desc = {
            # === 基础体积指标 ===
            'total_roi_voxels': int(total_roi),
            'valid_voxels': int(total_valid),
            'excluded_voxels': int(excluded_voxels),
            'exclusion_fraction': float(excluded_voxels / total_roi) if total_roi > 0 else 0.0,
            'tumor_volume_mm3': float(n_voxels * voxel_vol),

            # === 三类聚类比例 ===
            'class1_necrosis_voxels': int(n1),
            'class1_necrosis_fraction': float(n1 / total_valid) if total_valid > 0 else 0.0,
            'class2_stroma_voxels': int(n2),
            'class2_stroma_fraction': float(n2 / total_valid) if total_valid > 0 else 0.0,
            'class3_rich_voxels': int(n3),
            'class3_rich_fraction': float(n3 / total_valid) if total_valid > 0 else 0.0,

            # === 坏死中心性 ===
            'necrosis_centrality': float(
                np.mean(dist[self.labels == 0]) / (np.max(dist) + 1e-6)
            ) if n1 > 0 else 0.0,

            # === 聚类多样性 (Shannon熵) ===
            'cluster_diversity': float(
                -np.sum((np.bincount(self.labels, minlength=3) / total_valid) *
                        np.log2(np.bincount(self.labels, minlength=3) / total_valid + 1e-10))
            ) if total_valid > 0 else 0.0,

            # === 增强异质性 (变异系数) ===
            'enhancement_heterogeneity': float(
                np.std(delta) / (np.mean(np.abs(delta)) + 1e-6)
            ),

            # === 纹理异质性指数 ===
            'texture_heterogeneity': float(np.mean(localstd)),
            'texture_heterogeneity_cv': float(
                np.std(localstd) / (np.mean(localstd) + 1e-6)
            ),

            # === 增强梯度 (中心 vs 边缘) ===
            'enhancement_gradient': self._compute_enhancement_gradient(delta, dist),

            # === 坏死-活性交界带 ===
            'necrosis_rich_interface': self._compute_interface_ratio(),

            # === 增强均匀性指数 ===
            'enhancement_uniformity': float(
                1.0 / (1.0 + np.std(delta) / (np.abs(np.mean(delta)) + 1e-6))
            ),

            # === 密度分布偏度 ===
            'density_skewness': float(pd.Series(n_hu).skew()),

            # === 富血供集中度 ===
            'rich_vessel_centrality': float(
                np.mean(dist[self.labels == 2]) / (np.max(dist) + 1e-6)
            ) if n3 > 0 else 0.0,

            # === 各类特征统计 ===
            'necrosis_mean_nhu': float(np.mean(n_hu[self.labels == 0])) if n1 > 0 else 0.0,
            'necrosis_mean_delta': float(np.mean(delta[self.labels == 0])) if n1 > 0 else 0.0,
            'stroma_mean_nhu': float(np.mean(n_hu[self.labels == 1])) if n2 > 0 else 0.0,
            'stroma_mean_delta': float(np.mean(delta[self.labels == 1])) if n2 > 0 else 0.0,
            'rich_mean_nhu': float(np.mean(n_hu[self.labels == 2])) if n3 > 0 else 0.0,
            'rich_mean_delta': float(np.mean(delta[self.labels == 2])) if n3 > 0 else 0.0,
        }
        return desc

    def _compute_enhancement_gradient(self, delta, dist):
        """
        计算增强梯度: 中心区域vs边缘区域的增强差异

        NSCLC生物学意义:
        - 正梯度(中心>边缘): 中央富血供模式(中央型肺癌常见)
        - 负梯度(边缘>中心): 周围富血供模式(周围型肺癌/腺癌常见)
        - 接近0: 增强均匀分布
        """
        if len(delta) < 10:
            return 0.0

        d_max = np.max(dist)
        if d_max < 1e-6:
            return 0.0

        # 定义中心和边缘区域
        center_mask = dist >= (d_max * 0.5)
        edge_mask = dist <= (d_max * 0.3)

        center_delta = np.mean(delta[center_mask]) if np.any(center_mask) else 0.0
        edge_delta = np.mean(delta[edge_mask]) if np.any(edge_mask) else 0.0

        # 归一化梯度
        denom = np.abs(edge_delta) + 1e-6
        gradient = (center_delta - edge_delta) / denom

        return float(np.clip(gradient, -5.0, 5.0))

    def _compute_interface_ratio(self):
        """
        计算坏死-活性交界带比例

        在3D标签图中, 统计Class 1(坏死)与Class 3(富血供)之间的交界体素数
        占总有效体素的比例

        NSCLC生物学意义:
        - 高比值: 坏死与富血供区域紧密相邻, 提示快速生长导致的中心坏死
        - 低比值: 坏死与富血供区域分离, 可能为散在坏死或治疗相关坏死
        """
        if self.labels_3d is None:
            return 0.0

        labels = self.labels_3d.copy()
        # 只考虑肿瘤内部的有效区域
        labels[~self.mask] = -1

        # 创建二值图: 坏死(0)和富血供(2)
        nec = (labels == 0)
        rich = (labels == 2)

        if not (np.any(nec) and np.any(rich)):
            return 0.0

        # 对坏死区域进行膨胀, 然后与富血供区域求交
        struct = ndi.generate_binary_structure(3, 1)
        nec_dilated = ndi.binary_dilation(nec, structure=struct, iterations=1)

        # 交界体素 = 坏死膨胀区域 ∩ 富血供区域
        interface = nec_dilated & rich
        n_interface = np.sum(interface)
        n_valid = np.sum(self.mask)

        return float(n_interface / n_valid) if n_valid > 0 else 0.0

    def build_3d_labels(self, labels_1d):
        """构建完整3D标签图(含被排除的体素)"""
        labels3d = np.zeros(self.mask_raw.shape, dtype=np.uint16)
        labels3d[self.mask] = labels_1d + 1  # 0->1, 1->2, 2->3
        # 被排除的体素保持为0(背景)
        return labels3d


def batch_process_clustering(input_dir="./data", output_dir="./cluster/nsclc",
                             hu_range=(-150, 300), density_thresh=-50):
    """
    NSCLC双期CT异质性批量聚类

    输出结构:
        cluster/nsclc/
        ├─ clustering_table.csv
        ├─ summary_visualization.png
        ├─ clustering_labels/
        │  └─ *.nii.gz
        └─ patient_analysis/
            └─ *.png
    """
    clustering_dir = os.path.join(output_dir, "clustering_labels")
    analysis_dir = os.path.join(output_dir, "patient_analysis")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(clustering_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    n_dir = f"{input_dir}/N"
    c_dir = f"{input_dir}/C"
    m_dir = f"{input_dir}/ROI"

    for d in [n_dir, c_dir, m_dir]:
        if not os.path.exists(d):
            print(f"[ERR] Dir not found: {d}")
            return

    files = sorted([f for f in os.listdir(n_dir) if f.endswith('.nii.gz')])

    if not files:
        print("[ERR] No .nii.gz files found")
        return

    print(f"\n{'='*80}")
    print(f"NSCLC Dual-phase CT Heterogeneity Clustering (N-C)")
    print(f"{'='*80}\n")

    results_list = []
    successful = 0

    for idx, file in enumerate(files, 1):
        try:
            patient_id = file.replace('.nii.gz', '')
            print(f"[{idx:3d}/{len(files)}] {patient_id:<50}", end=" ", flush=True)

            N = nib.load(f"{n_dir}/{file}").get_fdata(dtype=np.float32)
            C = nib.load(f"{c_dir}/{file}").get_fdata(dtype=np.float32)
            M = (nib.load(f"{m_dir}/{file}").get_fdata() == 1)

            N = np.clip(N, hu_range[0], hu_range[1])
            C = np.clip(C, hu_range[0], hu_range[1])

            analyzer = NSCLCHeterogeneityAnalyzer(N, C, M, density_thresh=density_thresh)
            features, indices = analyzer.extract_features()
            labels = analyzer.cluster(features, method='clinical_prior')
            analyzer.labels = labels

            labels_refined = analyzer.apply_necrosis_prior(
                labels, interior_q=0.4, keep_top_k=1, kernel_size=3
            )
            analyzer.labels = labels_refined

            # 构建3D标签图
            labels_3d = analyzer.build_3d_labels(labels_refined)
            analyzer.labels_3d = labels_3d

            # 保存NIfTI
            N_img = nib.load(f"{n_dir}/{file}")
            hdr = nib.Nifti1Header()
            hdr.set_data_dtype(np.uint16)
            hdr.set_qform(N_img.affine, code=1)
            labels_img = nib.Nifti1Image(labels_3d, affine=N_img.affine, header=hdr)
            nib.save(labels_img, os.path.join(clustering_dir, f"{patient_id}_clustering.nii.gz"))

            # 计算描述符
            desc = analyzer.compute_descriptors()
            results_list.append({'patient_id': patient_id, **desc})

            # 绘制分析图
            plot_single_patient_nsclc(analyzer, patient_id, analysis_dir)

            print("OK")
            successful += 1

        except Exception as e:
            print(f"ERR {str(e)[:50]}")
            continue

    if results_list:
        df_all = pd.DataFrame(results_list)
        table_path = os.path.join(output_dir, "clustering_table.csv")
        df_all.to_csv(table_path, index=False)

        print(f"\n{'='*80}")
        print(f"[OK] Done: {successful}/{len(files)} samples")
        print(f"{'='*80}\n")

        print("Clustering Summary:")
        print("-" * 130)
        print(df_all.to_string(index=False))
        print("-" * 130 + "\n")

        plot_summary_visualization_nsclc(df_all, output_dir)

        print(f"\n{'='*80}")
        print(f"Output: {output_dir}")
        print(f"  - clustering_table.csv")
        print(f"  - summary_visualization.png")
        print(f"  - clustering_labels/")
        print(f"  - patient_analysis/")
        print(f"{'='*80}\n")
    else:
        print("\n[ERR] No samples processed successfully")


def plot_single_patient_nsclc(analyzer, patient_id, output_dir):
    """NSCLC single-patient analysis plot"""
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f'{patient_id}', fontsize=13, fontweight='bold', y=0.98)

    features = analyzer.features
    labels = analyzer.labels
    n_hu = features[:, 0]     # N_mean
    c_hu = features[:, 1]     # C_mean
    delta = features[:, 2]    # Delta_mean
    localstd = features[:, 3] # LocalStd

    # Color-blind friendly palette: blue / orange / green
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    class_names = ['Necrosis', 'Stroma', 'Rich']

    # 1. N vs C scatter
    ax1 = axes[0, 0]
    for k in [0, 1, 2]:
        mask = labels == k
        ax1.scatter(n_hu[mask], c_hu[mask], c=colors[k],
                    label=f'C{k+1}', alpha=0.5, s=20)
    ax1.set_xlabel('N (HU)', fontsize=9)
    ax1.set_ylabel('C (HU)', fontsize=9)
    ax1.set_title('N-C Scatter', fontsize=10)
    ax1.legend(fontsize=8, framealpha=0.7)
    ax1.grid(alpha=0.25)

    # 2. Enhancement distribution
    ax2 = axes[0, 1]
    for k in [0, 1, 2]:
        mask = labels == k
        ax2.hist(delta[mask], bins=25, alpha=0.55, label=f'C{k+1}',
                color=colors[k], edgecolor='white', linewidth=0.5)
    ax2.axvline(x=0, color='#666666', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Delta (C-N, HU)', fontsize=9)
    ax2.set_ylabel('Count', fontsize=9)
    ax2.set_title('Enhancement', fontsize=10)
    ax2.legend(fontsize=8, framealpha=0.7)
    ax2.grid(alpha=0.25, axis='y')

    # 3. Texture distribution
    ax3 = axes[0, 2]
    for k in [0, 1, 2]:
        mask = labels == k
        ax3.hist(localstd[mask], bins=25, alpha=0.55, label=f'C{k+1}',
                color=colors[k], edgecolor='white', linewidth=0.5)
    ax3.set_xlabel('Local Std', fontsize=9)
    ax3.set_ylabel('Count', fontsize=9)
    ax3.set_title('Texture', fontsize=10)
    ax3.legend(fontsize=8, framealpha=0.7)
    ax3.grid(alpha=0.25, axis='y')

    # 4. Spatial pattern
    ax4 = axes[1, 0]
    dist = analyzer.dist_to_boundary[analyzer.mask]
    dist_norm = dist / (np.max(dist) + 1e-6)
    for k in [0, 1, 2]:
        mask = labels == k
        ax4.scatter(dist_norm[mask], delta[mask], c=colors[k],
                    alpha=0.35, s=12)
    ax4.set_xlabel('Norm. Distance', fontsize=9)
    ax4.set_ylabel('Delta (HU)', fontsize=9)
    ax4.set_title('Spatial Pattern', fontsize=10)
    ax4.grid(alpha=0.25)

    # 5. Composition
    ax5 = axes[1, 1]
    class_counts = [np.sum(labels == k) for k in [0, 1, 2]]
    class_fracs = [c / len(labels) * 100 for c in class_counts]
    bars = ax5.bar(class_names, class_fracs, color=colors, alpha=0.75,
                   edgecolor='white', linewidth=1)
    for bar, frac, count in zip(bars, class_fracs, class_counts):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{frac:.1f}%',
                ha='center', va='bottom', fontsize=9)
    ax5.set_ylabel('Fraction (%)', fontsize=9)
    ax5.set_title('Composition', fontsize=10)
    ax5.set_ylim([0, max(class_fracs) * 1.25])
    ax5.grid(alpha=0.25, axis='y')

    # 6. Statistics table
    ax6 = axes[1, 2]
    ax6.axis('off')
    stats_text = "Statistics\n" + "-" * 22 + "\n"
    for k in [0, 1, 2]:
        mask = labels == k
        stats_text += f"\nC{k+1} ({class_names[k]}):\n"
        stats_text += f"  N:    {np.mean(n_hu[mask]):.1f}±{np.std(n_hu[mask]):.1f}\n"
        stats_text += f"  D:    {np.mean(delta[mask]):.1f}±{np.std(delta[mask]):.1f}\n"
        stats_text += f"  Tex:  {np.mean(localstd[mask]):.1f}±{np.std(localstd[mask]):.1f}\n"

    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8, edgecolor='#cccccc'))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, f"{patient_id}_analysis.png"),
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_summary_visualization_nsclc(df, output_dir):
    """Summary visualization"""
    sns.set_style("whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Summary', fontsize=13, fontweight='bold', y=0.98)

    n_patients = len(df)
    x = np.arange(n_patients)
    width = 0.55

    # Color-blind friendly palette for 3 classes
    class_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # 1. Cluster proportion (stacked)
    ax1 = axes[0, 0]
    frac_cols = ['class1_necrosis_fraction', 'class2_stroma_fraction', 'class3_rich_fraction']
    frac_data = df[frac_cols].values
    for i in range(3):
        bottom = np.sum(frac_data[:, :i], axis=1) if i > 0 else np.zeros(n_patients)
        ax1.bar(x, frac_data[:, i], width, bottom=bottom,
                label=f'C{i+1}', color=class_colors[i], alpha=0.7, edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Patient', fontsize=9)
    ax1.set_ylabel('Fraction', fontsize=9)
    ax1.set_title('Cluster %', fontsize=10)
    ax1.set_ylim([0, 1])
    ax1.legend(fontsize=8, framealpha=0.7)
    ax1.grid(axis='y', alpha=0.25)
    if n_patients <= 20:
        ax1.set_xticks(x)
        ax1.set_xticklabels([pid[:8] for pid in df['patient_id']], rotation=45, ha='right', fontsize=7)

    # 2. Exclusion fraction
    ax2 = axes[0, 1]
    ax2.bar(x, df['exclusion_fraction'].values, width=width,
            color='#888888', alpha=0.6, edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Patient', fontsize=9)
    ax2.set_ylabel('Fraction', fontsize=9)
    ax2.set_title('Excluded', fontsize=10)
    ax2.grid(axis='y', alpha=0.25)
    if n_patients <= 20:
        ax2.set_xticks(x)
        ax2.set_xticklabels([pid[:8] for pid in df['patient_id']], rotation=45, ha='right', fontsize=7)

    # 3. Heterogeneity metrics (twin axis)
    ax3 = axes[0, 2]
    ax3_twin = ax3.twinx()
    ax3.plot(x, df['cluster_diversity'].values, 'o-', linewidth=1.5,
             markersize=5, label='Diversity', color='#d62728')
    ax3_twin.plot(x, df['texture_heterogeneity'].values, 's-', linewidth=1.5,
                  markersize=5, label='Texture', color='#9467bd')
    ax3.set_xlabel('Patient', fontsize=9)
    ax3.set_ylabel('Diversity', fontsize=9, color='#d62728')
    ax3_twin.set_ylabel('Texture', fontsize=9, color='#9467bd')
    ax3.set_title('Heterogeneity', fontsize=10)
    ax3.tick_params(axis='y', labelcolor='#d62728')
    ax3_twin.tick_params(axis='y', labelcolor='#9467bd')
    ax3.grid(alpha=0.25)
    ax3.legend(fontsize=7, loc='upper left', framealpha=0.7)
    ax3_twin.legend(fontsize=7, loc='upper right', framealpha=0.7)
    if n_patients <= 20:
        ax3.set_xticks(x)
        ax3.set_xticklabels([pid[:8] for pid in df['patient_id']], rotation=45, ha='right', fontsize=7)

    # 4. Enhancement gradient
    ax4 = axes[1, 0]
    ax4.bar(x, df['enhancement_gradient'].values, width=width,
            color='#17becf', alpha=0.6, edgecolor='white', linewidth=0.5)
    ax4.axhline(y=0, color='#666666', linestyle='--', linewidth=0.8)
    ax4.set_xlabel('Patient', fontsize=9)
    ax4.set_ylabel('Gradient', fontsize=9)
    ax4.set_title('Center-Edge', fontsize=10)
    ax4.grid(axis='y', alpha=0.25)
    if n_patients <= 20:
        ax4.set_xticks(x)
        ax4.set_xticklabels([pid[:8] for pid in df['patient_id']], rotation=45, ha='right', fontsize=7)

    # 5. Necrosis-Rich interface
    ax5 = axes[1, 1]
    ax5.bar(x, df['necrosis_rich_interface'].values, width=width,
            color='#bcbd22', alpha=0.6, edgecolor='white', linewidth=0.5)
    ax5.set_xlabel('Patient', fontsize=9)
    ax5.set_ylabel('Ratio', fontsize=9)
    ax5.set_title('Necrosis-Rich', fontsize=10)
    ax5.grid(axis='y', alpha=0.25)
    if n_patients <= 20:
        ax5.set_xticks(x)
        ax5.set_xticklabels([pid[:8] for pid in df['patient_id']], rotation=45, ha='right', fontsize=7)

    # 6. Enhancement uniformity vs centrality (twin axis)
    ax6 = axes[1, 2]
    ax6_twin = ax6.twinx()
    ax6.plot(x, df['enhancement_uniformity'].values, 'D-', linewidth=1.5,
             markersize=5, label='Uniformity', color='#8c564b')
    ax6_twin.plot(x, df['rich_vessel_centrality'].values, '^-', linewidth=1.5,
                  markersize=5, label='Centrality', color='#e377c2')
    ax6.set_xlabel('Patient', fontsize=9)
    ax6.set_ylabel('Uniformity', fontsize=9, color='#8c564b')
    ax6_twin.set_ylabel('Centrality', fontsize=9, color='#e377c2')
    ax6.set_title('Pattern', fontsize=10)
    ax6.tick_params(axis='y', labelcolor='#8c564b')
    ax6_twin.tick_params(axis='y', labelcolor='#e377c2')
    ax6.grid(alpha=0.25)
    ax6.legend(fontsize=7, loc='upper left', framealpha=0.7)
    ax6_twin.legend(fontsize=7, loc='upper right', framealpha=0.7)
    if n_patients <= 20:
        ax6.set_xticks(x)
        ax6.set_xticklabels([pid[:8] for pid in df['patient_id']], rotation=45, ha='right', fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "summary_visualization.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Summary saved: {os.path.join(output_dir, 'summary_visualization.png')}")


# ============ Main ============
if __name__ == "__main__":
    batch_process_clustering(
        input_dir="./data",
        output_dir="./cluster/nsclc",
        hu_range=(-150, 300),
        density_thresh=-100
    )
