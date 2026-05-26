# NSCLC双期CT异质性聚类分析

## 1. 项目概述

本项目针对**非小细胞肺癌(NSCLC)**的CT影像进行**异质性分析与聚类**。输入数据为每位患者的三类体数据：

-   **平扫期CT**：`N/` 目录（NIfTI格式, Non-contrast）
-   **增强期CT**：`C/` 目录（NIfTI格式, Contrast-enhanced）
-   **肿瘤ROI掩膜**：`ROI/` 目录（NIfTI格式, 体素值=1表示肿瘤区域）

系统在肿瘤ROI内部提取双期影像特征，基于NSCLC病理生物学机制设计**四维特征空间**和**N+Delta联合临床先验分类**，将肿瘤体素分为：

-   **Class 1：坏死/低活性区域（Necrosis / Low-activity）**
    - 平扫期呈低密度（N ~ -50 to 40 HU）
    - 增强后几乎不增强（Delta ~ -50 to +10 HU）
    - 代表液化坏死、凝固坏死或水肿区
-   **Class 2：间质/纤维化区域（Stroma / Fibrosis）**
    - 平扫期呈高密度（N ~ 40-90 HU）
    - 不增强或低增强（Delta ~ -40 to +10 HU）
    - 代表纤维化、胶原沉积或致密间质
-   **Class 3：富血供/高活性区域（Rich-vascular / High-activity）**
    - 平扫期软组织密度（N ~ 20-60 HU）
    - 明显增强（Delta > +10 HU）
    - 代表活性肿瘤实质

---

## 2. NSCLC病理生物学基础

### 2.1 为什么需要NSCLC专用优化？

原版A-V双期算法（动脉-静脉）针对胰腺肿瘤设计，但**NSCLC具有独特的病理特征**：

| 特征 | 胰腺癌 | NSCLC |
|------|--------|-------|
| 肿瘤位置 | 腹膜后 | 肺实质内 |
| 背景密度 | 软组织(~40-60 HU) | 空气(~-800 HU) |
| 坏死模式 | 中央坏死为主 | 中央/散在/空洞 |
| 增强模式 | 动脉期明显增强 | 多呈渐进性增强 |
| 间质比例 | 极高(Desmoplasia) | 中等 |
| ROI混杂 | 较少 | 肺组织/血管易混入 |

### 2.2 NSCLC的CT密度特征

NSCLC肿瘤ROI内常包含多种组织成分：

-   **肺组织混入**（N < -100 HU）：肺实质、支气管、血管周围空气
-   **坏死液化/空洞**（N ~ -100 to -50 HU）：坏死区液化形成低密度区，部分可形成空洞（HU接近-800）
-   **凝固坏死/水肿**（N ~ -50 to 40 HU）：低密度软组织，对比剂不渗透
-   **活性肿瘤**（N ~ 20-60 HU）：软组织密度，增强后明显强化
-   **纤维化**（N ~ 60-100 HU）：较高密度，增强弱或不增强
-   **钙化/出血**（N > 100 HU）：高密度灶

### 2.3 NSCLC的增强特征

NSCLC增强遵循以下规律：

-   **坏死区**：对比剂不渗透，Delta ≈ 0（甚至为负，由于噪声或运动伪影）
-   **纤维化区**：血供稀少，Delta ≈ 0-10 HU
-   **间质区**：中等血供，Delta ≈ 10-40 HU
-   **富血供区**：大量新生血管，Delta > 40 HU

**核心洞察**：在NSCLC中，**"增强程度"（Delta = C - N）是区分活性与坏死的核心维度**，但**平扫密度（N）同样关键**——高N+低Delta=纤维化，低N+低Delta=坏死，两者临床意义完全不同。

---

## 3. 算法总体流程

```
对每个患者：
  1. 读取N、C、ROI三期数据
  2. HU值裁剪（hu_range = [-150, 300]）
  3. 【密度预滤波】排除N < -100 HU的体素（保留坏死/液化，排除空气）
  4. 【Kernel平滑】对N和C做3x3x3邻域均值滤波（仅mask内邻居）
  5. 【特征提取】构建四维特征空间：[N_mean, C_mean, Delta_mean, LocalStd]
  6. 【N+Delta联合分类】在标准化[N, Delta]空间中，基于临床先验理想点做加权距离分类
  7. 【空间先验】对坏死类应用自适应空间约束
  8. 【描述符计算】计算18+维度异质性指标
  9. 保存聚类标签图(NIfTI)和分析图(PNG)

汇总所有患者的聚类结果与异质性指标
```

---

## 4. 密度预滤波机制（Density Pre-filtering）

### 4.1 为什么需要预滤波？

在NSCLC的ROI掩膜中，由于分割不精确、肿瘤边界过渡区、支气管/血管穿过肿瘤等因素，常混入大量**肺组织**（N ~ -800 HU）和**空气**（N ~ -1000 HU）。这些体素不是肿瘤组织，不应参与异质性聚类。

### 4.2 预滤波规则

```python
有效体素 = ROI_mask AND (N >= density_thresh)
排除体素 = ROI_mask AND (N < density_thresh)
```

默认阈值 `density_thresh = -100 HU`：

| 密度范围 | 典型组织 | 处理 |
|---------|---------|------|
| N < -100 HU | 肺组织、极端空气、大空洞 | **排除**（label=0） |
| -100 <= N < -50 HU | 坏死液化、水肿、小气腔 | **保留，参与聚类** |
| N >= -50 HU | 软组织、肿瘤、纤维化 | **保留，参与聚类** |

**为什么-100 HU而不是-50 HU？**
- -50 HU排除了大量真正的坏死/液化体素（N=-50~-100）
- -100 HU在排除空气（N~-800）的同时，保留了坏死液化区（N=-100~-50）
- NSCLC中液化坏死的典型密度：-100到0 HU（介于水和空气之间）

### 4.3 预滤波效果

以测试样本为例：
-   总ROI体素：16,019
-   排除体素（N < -100）：2,692（16.8%）
-   有效体素：13,327（83.2%）

---

## 5. 四维特征空间与Kernel平滑

### 5.1 特征维度总览

对有效体素（N >= -100 HU）提取**四维特征向量**，**每个体素的特征基于其3x3x3邻域(kernel)的统计量，而非单像素值**：

| 维度 | 索引 | 符号 | 计算方式 | NSCLC生物学意义 |
|------|------|------|---------|----------------|
| 平扫密度 | 0 | `n_mean` | 3x3x3邻域内N均值 | 区分低密度坏死 vs 软组织 vs 高密度纤维化 |
| 增强密度 | 1 | `c_mean` | 3x3x3邻域内C均值 | 增强后的实际HU值 |
| 绝对增强 | 2 | `delta_mean` | c_mean - n_mean | **核心维度**：血供程度 |
| 局部纹理 | 3 | `local_std` | 3x3x3邻域内C期标准差 | 反映微观异质性 |

### 5.2 Kernel平滑算法（_kernel_smooth_masked）

**问题**：`uniform_filter(cval=0)`在边界处会引入mask外区域的0值，导致边界体素被拉低。

**解决方案**：仅对mask内的有效邻居做平均。

```python
def _kernel_smooth_masked(data, mask, size=3):
    """
    只对mask内邻居做3D均值平滑

    Args:
        data: 3D CT数据 (shape: D x H x W)
        mask: 有效区域掩膜(bool)
        size: kernel大小 (默认3, 即3x3x3)

    Returns:
        smoothed: mask内体素的邻域均值 (shape: D x H x W)
    """
    # 构建3x3x3全1卷积核
    kernel = np.ones((size, size, size), dtype=np.float32)

    # 将data在mask外置0，避免mask外值参与求和
    masked_data = data.astype(np.float32) * mask.astype(np.float32)

    # 计算每个体素的3x3x3邻域内有效体素数量
    # neighbor_count[i,j,k] = sum(mask[i±1, j±1, k±1])
    neighbor_count = ndi.convolve(
        mask.astype(np.float32), kernel,
        mode='constant', cval=0
    )

    # 计算每个体素的3x3x3邻域内有效体素之和
    # neighbor_sum[i,j,k] = sum(data * mask)[i±1, j±1, k±1]
    neighbor_sum = ndi.convolve(
        masked_data, kernel,
        mode='constant', cval=0
    )

    # 均值 = 和 / 数量，无有效邻居时保持原值
    result = np.where(
        neighbor_count > 0,
        neighbor_sum / neighbor_count,   # 正常均值
        data.astype(np.float32)           # 无邻居时fallback
    )
    return result
```

**关键优势**：
- 边界体素只与mask内的邻居平均，不会被外部空气（-1000 HU）拉低
- 肿瘤边缘的过渡区更加平滑、自然
- 配准偏移1体素导致的像素级delta波动被邻域平均降低约60%

### 5.3 局部纹理标准差（_compute_local_std）

```python
def _compute_local_std(data, size=3):
    """
    计算3D局部标准差

    原理: var = E[X^2] - (E[X])^2
    """
    # E[X]: 邻域均值
    mean = ndi.uniform_filter(data, size=size, mode='constant', cval=0)

    # E[X^2]: 邻域二阶矩
    mean_sq = ndi.uniform_filter(data ** 2, size=size, mode='constant', cval=0)

    # var = E[X^2] - (E[X])^2，避免数值误差导致的负值
    var = np.maximum(mean_sq - mean ** 2, 0)

    return np.sqrt(var)
```

**生物学意义**：
-   **低local_std（< 15）**：区域内均匀——纯坏死区（均匀低密度）、纯纤维化（均匀高密度）
-   **高local_std（> 30）**：区域内异质——坏死与活性肿瘤交界区、纤维化与肿瘤混合区
-   **中等local_std（15-30）**：过渡区

### 5.4 特征提取完整流程（extract_features）

```python
def extract_features(self, kernel_size=3):
    """
    基于kernel统计的特征提取

    Step 1: Kernel平滑
        n_smooth = _kernel_smooth_masked(N, mask, size=3)
        c_smooth = _kernel_smooth_masked(C, mask, size=3)

    Step 2: 提取有效体素特征
        n_mean = n_smooth[mask]          # (N_valid,)
        c_mean = c_smooth[mask]          # (N_valid,)
        delta_mean = c_mean - n_mean     # (N_valid,)

    Step 3: 局部纹理
        local_std = _compute_local_std(c_smooth, size=3)
        local_std_vals = local_std[mask]  # (N_valid,)

    Step 4: 构建四维特征矩阵
        features = [n_mean, c_mean, delta_mean, local_std_vals]
                   # shape: (N_valid, 4)

    Step 5: 计算到边界距离（用于后续空间先验）
        dist_to_boundary = distance_transform_edt(mask)
    """
```

### 5.5 为什么需要N_mean和C_mean同时存在？

Delta = C - N 是增强差，但**N和C的绝对值各自携带独立信息**：

| 组织类型 | N_mean | C_mean | Delta | N-C联合解读 |
|---------|--------|--------|-------|-----------|
| 液化坏死 | **-50~-20** | **-20~+20** | ~+20~+40 | 两者都低，Delta因N极低而"虚高" |
| 凝固坏死 | **0~30** | **0~30** | ~-10~+10 | 软组织密度，不增强 |
| 纤维化 | **60~90** | **40~60** | ~-30~-10 | 高密度，不增强 |
| 活性肿瘤 | **20~50** | **40~80** | ~+10~+50 | 软组织，明显增强 |

**关键洞察**：
- 仅用Delta会混淆"液化坏死"（N=-50, C=-20, Delta=+30）和"活性肿瘤"（N=30, C=60, Delta=+30）
- 加入N后：N=-50是坏死，N=30是活性肿瘤——清晰区分
- 高N+低C+低Delta = **纤维化**（不是坏死！）

---

## 6. N+Delta联合临床先验分类算法

### 6.1 为什么放弃GMM和纯Delta分类？

| 问题 | 原因 | 后果 |
|------|------|------|
| GMM假设失效 | 数据不是三个分离的高斯分布 | 三类中心挤在一起，某类被压到极端 |
| 纯Delta分类 | 忽略N的关键信息 | 纤维化（N=70, D=-30）被误分为坏死 |
| KMeans距离主导 | N和C的绝对大值主导欧氏距离 | Delta维度被淹没 |
| 初始化敏感 | GMM初始中心微小变化导致结果剧变 | 跨样本不可比 |

### 6.2 核心算法：标准化[N, Delta]空间中的加权距离分类

```python
def _clinical_prior_clustering(self, features):
    """
    基于N和Delta联合的临床先验分类

    输入: features (N_valid, 4) = [N_mean, C_mean, Delta_mean, LocalStd]
    输出: labels (N_valid,) = {0:坏死, 1:间质, 2:富血供}
    """
    # === Step 1: 提取分类用的两个维度 ===
    n_hu = features[:, 0]    # N_mean
    delta = features[:, 2]   # Delta_mean (C_mean - N_mean)

    # === Step 2: Z-score标准化 ===
    # 目的: 让N和Delta有相同尺度，避免N的绝对大值主导距离
    n_mean, n_std = n_hu.mean(), n_hu.std() + 1e-6
    d_mean, d_std = delta.mean(), delta.std() + 1e-6
    n_z = (n_hu - n_mean) / n_std   # 标准化后的N
    d_z = (delta - d_mean) / d_std  # 标准化后的Delta

    # === Step 3: 定义动态临床理想点 ===
    # 基于数据分布的分位数，而非固定绝对值
    # 这样可适应不同扫描时相（动脉期/静脉期/延迟期）
    n_p25 = np.percentile(n_z, 25)   # N的低分位
    n_p50 = np.percentile(n_z, 50)   # N的中位数
    n_p75 = np.percentile(n_z, 75)   # N的高分位
    d_p25 = np.percentile(d_z, 25)   # Delta的低分位
    d_p50 = np.percentile(d_z, 50)   # Delta的中位数
    d_p75 = np.percentile(d_z, 75)   # Delta的高分位

    # 三类临床理想点（标准化空间中）
    ideal_necrosis = np.array([n_p25, d_p25])
    #   坏死 = N低(P25) + Delta低(P25)
    #   临床: 低密度 + 不增强 = 液化/凝固坏死

    ideal_stroma = np.array([n_p75, d_p50])
    #   间质 = N高(P75) + Delta中(P50)
    #   临床: 高密度 + 不/低增强 = 纤维化

    ideal_rich = np.array([n_p50, d_p75])
    #   富血供 = N中(P50) + Delta高(P75)
    #   临床: 软组织 + 明显增强 = 活性肿瘤

    # === Step 4: 计算加权距离 ===
    # Delta权重更高(1.4)，因为血供是核心区分维度
    # N权重较低(0.6)，作为辅助区分坏死vs纤维化
    w_n, w_d = 0.6, 1.4

    dist_necrosis = w_n * (n_z - ideal_necrosis[0])**2 + \
                    w_d * (d_z - ideal_necrosis[1])**2

    dist_stroma   = w_n * (n_z - ideal_stroma[0])**2 + \
                    w_d * (d_z - ideal_stroma[1])**2

    dist_rich     = w_n * (n_z - ideal_rich[0])**2 + \
                    w_d * (d_z - ideal_rich[1])**2

    # === Step 5: 分配到距离最近的类 ===
    dists = np.stack([dist_necrosis, dist_stroma, dist_rich], axis=1)
    labels = np.argmin(dists, axis=1)

    # 标签语义:
    #   0 = 坏死 (低N + 低Delta)
    #   1 = 间质/纤维化 (高N + 低Delta)
    #   2 = 富血供/活性 (中N + 高Delta)
    return labels.astype(int)
```

### 6.3 临床理想点的生物学依据

**为什么坏死 = N低 + Delta低？**

| 错误理解 | 正确理解 | 临床依据 |
|---------|---------|---------|
| "不增强 = 坏死" | "低N + 不增强 = 坏死" | 液化坏死密度接近水(0-20 HU)，纤维化密度高(60-90 HU) |
| "高N + 不增强 = 坏死" | "高N + 不增强 = 纤维化" | 胶原沉积、瘢痕组织的平扫密度高于正常软组织 |

**典型误判案例**：
- 纤维化：N=75 HU, C=50 HU, Delta=-25 HU → **高N+低Delta = 间质（纤维化）**
- 液化坏死：N=10 HU, C=5 HU, Delta=-5 HU → **低N+低Delta = 坏死**
- 两者Delta相近，但N值截然不同

### 6.4 权重选择依据

`w_n = 0.6`, `w_d = 1.4` 的选择理由：

| 维度 | 权重 | 理由 |
|------|------|------|
| Delta | 1.4 (高) | 血供是区分活性肿瘤的核心指标 |
| N | 0.6 (低) | 辅助区分坏死vs纤维化，但不应主导分类 |

如果 `w_n = w_d = 1.0`：N和Delta等权重，N的方差较大时可能主导分类
如果 `w_n > w_d`：N主导，可能将大量体素错误地按密度而非血供分类

---

## 7. 空间先验约束算法

### 7.1 自适应连通域过滤

```python
def apply_necrosis_prior(self, labels_1d, interior_q=0.4,
                         keep_top_k=1, kernel_size=3):
    """
    NSCLC自适应空间先验约束

    输入: labels_1d (N_valid,) — 聚类后的标签 {0:坏死, 1:间质, 2:富血供}
    输出: labels_refined (N_valid,) — 空间约束后的标签
    """
    n_voxels = len(labels_1d)

    # === 自适应连通域阈值 ===
    # 根据肿瘤总体素数动态调整
    if n_voxels < 5000:       # 小肿瘤
        min_cc_vox = max(3, int(n_voxels * 0.003))
    elif n_voxels < 30000:   # 中等肿瘤
        min_cc_vox = max(5, int(n_voxels * 0.005))
    else:                     # 大肿瘤
        min_cc_vox = max(10, int(n_voxels * 0.01))

    # 将1D标签映射回3D
    labels0 = labels_1d.copy()
    labels3d = np.zeros(self.mask.shape, dtype=np.int16)
    labels3d[self.mask] = labels0

    # 计算到边界的距离
    dist = self.dist_to_boundary
    d_thr = np.quantile(dist[self.mask], interior_q)

    # 构建3x3x3邻域核（中心点除外）
    k = max(3, int(kernel_size))
    kernel = np.ones((k, k, k))
    kernel[k // 2, k // 2, k // 2] = 0

    # === Step 1: 边界坏死重分配 ===
    # 边界附近的坏死碎片可能是分割噪声
    nec = (labels3d == 0)
    boundary_nec = nec & (dist < d_thr)

    if np.any(boundary_nec):
        # 计算边界坏死点邻域中间质(1)和富血供(2)的数量
        n1 = ndi.convolve((labels3d == 1).astype(int), kernel, mode='constant')
        n2 = ndi.convolve((labels3d == 2).astype(int), kernel, mode='constant')
        # 多数表决：邻域中哪类多，就分配给哪类
        labels3d[boundary_nec] = np.where(
            n1[boundary_nec] >= n2[boundary_nec], 1, 2
        )

    # === Step 2: 连通域过滤 ===
    nec = (labels3d == 0)
    if np.any(nec):
        struct = ndi.generate_binary_structure(3, 1)
        cc, num = ndi.label(nec, structure=struct)
        sizes = np.bincount(cc.ravel())
        sizes[0] = 0

        # 保留策略:
        # (a) 保留最大的keep_top_k个连通块
        keep = np.zeros_like(sizes, dtype=bool)
        if keep_top_k and keep_top_k > 0:
            keep[np.argsort(sizes)[-keep_top_k:]] = True
        # (b) 保留体素数 >= min_cc_vox的连通块
        keep[sizes >= max(1, int(min_cc_vox))] = True

        # 形态学闭运算：弥合小的断裂
        nec_clean = nec & keep[cc]
        nec_clean = ndi.binary_closing(nec_clean, structure=struct, iterations=1)

        # 被过滤掉的坏死体素重新分配
        drop = nec & (~nec_clean)
        if np.any(drop):
            n1 = ndi.convolve((labels3d == 1).astype(int), kernel, mode='constant')
            n2 = ndi.convolve((labels3d == 2).astype(int), kernel, mode='constant')
            labels3d[drop] = np.where(n1[drop] >= n2[drop], 1, 2)

    return labels3d[self.mask].astype(np.uint16)
```

### 7.2 空间先验的生物学依据

| 约束 | 目的 | NSCLC病理基础 |
|------|------|--------------|
| 边界重分配 | 去除边界噪声碎片 | 分割误差常导致边界模糊 |
| 连通域过滤 | 保留主要坏死区，去除孤立点 | 真实坏死应成团，散在点多为噪声 |
| 形态学闭运算 | 弥合坏死区内部小断裂 | 坏死区内部可能有血管穿过形成"桥" |
| 多数表决重分配 | 被过滤的体素归到周围主导类 | 邻近组织类型决定归属 |

---

## 8. 异质性描述符体系（完整计算算法）

```python
def compute_descriptors(self):
    """
    计算NSCLC优化的18维度异质性描述符

    输入: self.features (N_valid, 4) = [N_mean, C_mean, Delta_mean, LocalStd]
          self.labels (N_valid,) = {0:坏死, 1:间质, 2:富血供}
          self.dist_to_boundary (D x H x W)
    输出: desc (dict) — 18个异质性指标
    """
```

### 8.1 基础体积指标

```python
n_voxels = self.mask.sum()                # 有效体素数
excluded_voxels = self.mask_excluded.sum() # 被排除体素数
total_roi = self.mask_raw.sum()           # ROI总 voxel数
voxel_vol = 1.0  # 若头文件有体素尺寸信息则使用

desc = {
    'total_roi_voxels': int(total_roi),
    'valid_voxels': int(n_voxels),
    'excluded_voxels': int(excluded_voxels),
    'exclusion_fraction': float(excluded_voxels / total_roi),
    'tumor_volume_mm3': float(n_voxels * voxel_vol),
}
```

### 8.2 聚类比例指标

```python
n1 = np.sum(self.labels == 0)   # 坏死体素数
n2 = np.sum(self.labels == 1)   # 间质体素数
n3 = np.sum(self.labels == 2)   # 富血供体素数
total_valid = n1 + n2 + n3

desc['class1_necrosis_voxels'] = int(n1)
desc['class1_necrosis_fraction'] = float(n1 / total_valid)
desc['class2_stroma_voxels'] = int(n2)
desc['class2_stroma_fraction'] = float(n2 / total_valid)
desc['class3_rich_voxels'] = int(n3)
desc['class3_rich_fraction'] = float(n3 / total_valid)
```

### 8.3 空间分布指标

**坏死中心性（necrosis_centrality）**
```python
# 原理: 坏死体素到边界的平均距离 / 最大距离
# 值域: [0, 1]
# 越大说明坏死越靠近肿瘤中心
if n1 > 0:
    desc['necrosis_centrality'] = float(
        np.mean(dist[self.labels == 0]) / (np.max(dist) + 1e-6)
    )
else:
    desc['necrosis_centrality'] = 0.0
```

**富血供中心性（rich_vessel_centrality）**
```python
# 原理: 富血供体素到边界的平均距离 / 最大距离
# 值域: [0, 1]
if n3 > 0:
    desc['rich_vessel_centrality'] = float(
        np.mean(dist[self.labels == 2]) / (np.max(dist) + 1e-6)
    )
else:
    desc['rich_vessel_centrality'] = 0.0
```

### 8.4 聚类多样性指标

**聚类多样性（cluster_diversity）——Shannon熵**
```python
# 原理: 三类分布的Shannon熵，衡量分布均匀性
# H = -sum(p_i * log2(p_i))
# 值域: [0, log2(3)≈1.585]
# 越大说明三类分布越均衡，异质性越强
p = np.bincount(self.labels, minlength=3) / total_valid
desc['cluster_diversity'] = float(
    -np.sum(p * np.log2(p + 1e-10))
)
```

### 8.5 增强特征指标

**增强异质性（enhancement_heterogeneity）——变异系数**
```python
# 原理: Delta的标准差 / Delta的绝对均值
# 值域: [0, +inf)
# 越大说明增强越不均匀
desc['enhancement_heterogeneity'] = float(
    np.std(delta) / (np.mean(np.abs(delta)) + 1e-6)
)
```

**增强梯度（enhancement_gradient）——中心vs边缘**
```python
# 原理: (中心区域平均Delta - 边缘区域平均Delta) / |边缘平均Delta|
# 定义中心区域: 距离 > 最大距离的50%
# 定义边缘区域: 距离 < 最大距离的30%
d_max = np.max(dist)
center_mask = dist >= (d_max * 0.5)
edge_mask = dist <= (d_max * 0.3)

center_delta = np.mean(delta[center_mask]) if np.any(center_mask) else 0.0
edge_delta = np.mean(delta[edge_mask]) if np.any(edge_mask) else 0.0

gradient = (center_delta - edge_delta) / (np.abs(edge_delta) + 1e-6)
desc['enhancement_gradient'] = float(np.clip(gradient, -5.0, 5.0))

# 生物学解读:
#   > 0: 中心增强 > 边缘 → 中央富血供模式（中央型肺癌）
#   < 0: 边缘增强 > 中心 → 周围富血供模式（周围型/腺癌）
#   ≈ 0: 增强均匀
```

**增强均匀性（enhancement_uniformity）**
```python
# 原理: 1 / (1 + CV)，CV = 变异系数
# 值域: [0, 1]
# 越接近1说明增强越均匀
desc['enhancement_uniformity'] = float(
    1.0 / (1.0 + np.std(delta) / (np.abs(np.mean(delta)) + 1e-6))
)
```

### 8.6 纹理异质性指标

**纹理异质性（texture_heterogeneity）**
```python
# 原理: 局部标准差的均值
# 反映肿瘤内部微观结构复杂性
desc['texture_heterogeneity'] = float(np.mean(localstd))
```

**纹理异质性变异系数（texture_heterogeneity_cv）**
```python
# 原理: 局部标准差自身的变异系数
# 反映纹理异质性在空间上的不均匀程度
desc['texture_heterogeneity_cv'] = float(
    np.std(localstd) / (np.mean(localstd) + 1e-6)
)
```

### 8.7 交界带特征

**坏死-富血供交界带（necrosis_rich_interface）**
```python
# 原理: 坏死区域膨胀1体素后与富血供区域的交集 / 总有效体素
# 值域: [0, 1]
# 越高说明坏死与活性肿瘤越紧密相邻
labels = self.labels_3d.copy()
labels[~self.mask] = -1

nec = (labels == 0)      # 坏死二值图
rich = (labels == 2)     # 富血供二值图

struct = ndi.generate_binary_structure(3, 1)
nec_dilated = ndi.binary_dilation(nec, structure=struct, iterations=1)

interface = nec_dilated & rich
n_interface = np.sum(interface)

desc['necrosis_rich_interface'] = float(
    n_interface / np.sum(self.mask)
)

# 生物学解读:
#   高比值: 坏死与富血供紧密相邻 → 快速生长导致的"生长性坏死"
#   低比值: 坏死与富血供分离 → 治疗相关坏死或散在坏死
```

### 8.8 密度分布指标

**密度分布偏度（density_skewness）**
```python
# 原理: 平扫期N值的偏度（三阶标准化矩）
# 值域: (-inf, +inf)
#   > 0: 右偏（大量低密度体素，提示坏死/水肿多）
#   < 0: 左偏（大量高密度体素，提示纤维化/钙化多）
desc['density_skewness'] = float(pd.Series(n_hu).skew())
```

### 8.9 各类特征统计

```python
# 各类的N、Delta均值（用于验证分类合理性）
desc['necrosis_mean_nhu'] = float(np.mean(n_hu[self.labels == 0])) if n1 > 0 else 0.0
desc['necrosis_mean_delta'] = float(np.mean(delta[self.labels == 0])) if n1 > 0 else 0.0
desc['stroma_mean_nhu'] = float(np.mean(n_hu[self.labels == 1])) if n2 > 0 else 0.0
desc['stroma_mean_delta'] = float(np.mean(delta[self.labels == 1])) if n2 > 0 else 0.0
desc['rich_mean_nhu'] = float(np.mean(n_hu[self.labels == 2])) if n3 > 0 else 0.0
desc['rich_mean_delta'] = float(np.mean(delta[self.labels == 2])) if n3 > 0 else 0.0
```

---

## 9. 输出文件解释

```
cluster/nsclc/
├── clustering_table.csv              # 汇总表格（18个描述符）
├── summary_visualization.png         # 汇总可视化图表
├── clustering_labels/
│   └── *_clustering.nii.gz          # 3D聚类标签图
│       # 标签值：0=背景/排除, 1=坏死, 2=间质, 3=富血供
└── patient_analysis/
    └── *_analysis.png               # 单患者分析图（6子图）
```

### 单患者分析图（6子图）

1. **N-C散点图**：N_mean vs C_mean，三类颜色区分
2. **Delta分布**：三类增强差的直方图
3. **纹理分布**：三类局部标准差的直方图
4. **空间模式**：肿瘤内部位置 vs Delta的空间分布
5. **类别比例**：三类体素数量与百分比
6. **统计表**：各类的N、C、Delta、Texture均值与标准差

---

## 10. 运行方式

### 10.1 数据准备

```
data/
├── N/      # 平扫期CT影像（.nii.gz）
├── C/      # 增强期CT影像（.nii.gz，与N同名）
└── ROI/    # ROI掩膜影像（.nii.gz，与N同名，体素值=1为肿瘤）
```

### 10.2 运行命令

```bash
python ith_analysis.py
```

### 10.3 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_dir` | "./data" | 数据目录 |
| `output_dir` | "./cluster/nsclc" | 输出目录 |
| `hu_range` | (-150, 300) | HU裁剪范围 |
| `density_thresh` | **-100** | 密度预滤波阈值 |

---

## 11. 算法设计要点总结

### 11.1 与原版算法的核心区别

| 维度 | 原版（A-V，胰腺） | NSCLC优化版（N-C） |
|------|----------------|-------------------|
| 密度预滤波 | 无 | 有（N < **-100** HU，保留坏死/液化） |
| 特征提取 | 单像素值 | **3x3x3 Kernel统计量**（仅mask内邻居） |
| 特征空间 | [A, V, V-A] | **[N_mean, C_mean, Delta_mean, LocalStd]**（四维） |
| 聚类方法 | 数据驱动GMM | **N+Delta联合加权距离分类**（动态临床理想点） |
| 坏死定义 | 低mean_hu | **低N + 低Delta**（低密度不增强） |
| 间质定义 | 中mean_hu | **高N + 低Delta**（高密度不增强=纤维化） |
| 连通域阈值 | 固定200 | 自适应（按肿瘤体积） |
| 描述符数量 | 6个 | **18个** |

### 11.2 关键设计决策的验证

| 决策 | 验证结果 | 临床合理性 |
|------|---------|-----------|
| 坏死 = 低N + 低Delta | 坏死N=23 HU（低密度） | 液化坏死密度接近水，符合 |
| 间质 = 高N + 低Delta | 间质N=61 HU（高密度） | 纤维化密度高于软组织，符合 |
| 富血供 = 中N + 高Delta | 富血供N=31 HU, D=+18 | 软组织增强，符合 |
| Kernel平滑替代单像素 | Delta变异降低~60% | 配准鲁棒性提升 |
| 密度预滤波-100 HU | 保留627体素(-100<N<-50) | 包含坏死液化区，合理 |

### 11.3 局限性

1. **单一样本验证**：当前仅在一个样本上测试，多样本验证待进行
2. **时相敏感性**：不区分动脉期/静脉期/延迟期，不同期相的Delta分布差异大
3. **坏死比例偏低**：当前样本坏死仅4%，可能反映真实病理，也可能与阈值设置有关
4. **权重固定**：w_n=0.6, w_d=1.4对所有样本统一，不同肿瘤的最优权重可能不同
