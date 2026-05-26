"""
Visualization utility functions
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")


def setup_figure(figsize=(10, 8), dpi=150):
    """创建标准图形配置"""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    return fig


def save_and_close(fig, out_path, tight=True):
    """保存图形并关闭"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if tight:
        plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_colormap_n(n_colors, name='custom'):
    """生成n色离散颜色映射"""
    colors = plt.cm.get_cmap('tab20', n_colors)
    return colors


def overlay_mask_on_slice(image_slice, mask_slice, alpha=0.4, mask_color='red'):
    """在图像切片上叠加掩膜"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image_slice, cmap='gray')
    mask_rgba = np.zeros((*mask_slice.shape, 4))
    if mask_color == 'red':
        mask_rgba[mask_slice > 0] = [1, 0, 0, alpha]
    elif mask_color == 'blue':
        mask_rgba[mask_slice > 0] = [0, 0.5, 1, alpha]
    elif mask_color == 'green':
        mask_rgba[mask_slice > 0] = [0, 1, 0, alpha]
    ax.imshow(mask_rgba)
    ax.axis('off')
    return fig, ax


def plot_distribution_with_stats(data, title, xlabel, out_path, bins=30, color='steelblue'):
    """绘制分布直方图+统计信息"""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(data, bins=bins, color=color, edgecolor='white', alpha=0.8)

    # 添加统计线
    mean_v = np.mean(data)
    median_v = np.median(data)
    ax.axvline(mean_v, color='red', linestyle='--', linewidth=2, label=f'Mean={mean_v:.3f}')
    ax.axvline(median_v, color='green', linestyle='-.', linewidth=2, label=f'Median={median_v:.3f}')

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.legend(fontsize=10)

    save_and_close(fig, out_path)


def plot_correlation_heatmap(df, features, title, out_path, figsize=(10, 8)):
    """绘制特征相关性热图"""
    fig, ax = plt.subplots(figsize=figsize)

    corr = df[features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, ax=ax,
                annot_kws={'size': 8})
    ax.set_title(title, fontsize=13, fontweight='bold')

    save_and_close(fig, out_path)


def plot_radar_chart(categories, values, title, out_path, figsize=(8, 8)):
    """绘制雷达图"""
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_plot = values + values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.fill(angles, values_plot, color='steelblue', alpha=0.25)
    ax.plot(angles, values_plot, color='steelblue', linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

    save_and_close(fig, out_path)


def plot_multi_slice_comparison(volume_dict, slice_idx=None, out_path=None, titles=None):
    """多模型结果切片对比图"""
    n = len(volume_dict)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (name, vol) in enumerate(volume_dict.items()):
        ax = axes[idx]

        if slice_idx is None:
            # 自动选择最大肿瘤切片
            if vol.ndim == 3:
                slice_idx = np.argmax(np.sum(vol > 0, axis=(1, 2)))
            else:
                slice_idx = 0

        if vol.ndim == 3:
            sl = vol[slice_idx]
        else:
            sl = vol

        title = titles[idx] if titles and idx < len(titles) else name
        im = ax.imshow(sl, cmap='jet', interpolation='nearest')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # 隐藏多余的子图
    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    if out_path:
        save_and_close(fig, out_path)
    return fig


# ============================================================
# 新增可视化工具函数
# ============================================================

def plot_violin_box(data_series, title, xlabel, out_path, color='steelblue',
                    inner='box', show_points=True, figsize=(8, 6)):
    """小提琴图+箱线图组合"""
    fig, ax = plt.subplots(figsize=figsize)

    parts = ax.violinplot(data_series, positions=[0], widths=0.5,
                          showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(0.4)

    bp = ax.boxplot(data_series, positions=[0], widths=0.15, patch_artist=True,
                    showmeans=True, meanline=True,
                    boxprops=dict(facecolor=color, alpha=0.7),
                    medianprops=dict(color='white', linewidth=2),
                    meanprops=dict(color='yellow', linestyle='--', linewidth=2))

    if show_points:
        jitter = np.random.normal(0, 0.04, len(data_series))
        ax.scatter(jitter, data_series, alpha=0.3, s=15, color='black', zorder=3)

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel(xlabel, fontsize=11)
    ax.set_xticks([])
    ax.set_xlim(-0.5, 0.5)

    # 统计注释
    mean_v = np.mean(data_series)
    std_v = np.std(data_series)
    median_v = np.median(data_series)
    ax.text(0.35, mean_v, f'Mean: {mean_v:.3f}\nStd: {std_v:.3f}\nMedian: {median_v:.3f}',
            fontsize=9, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    save_and_close(fig, out_path)


def plot_kde(data_list, labels, title, xlabel, out_path, colors=None,
             fill=True, figsize=(10, 6)):
    """多组KDE密度对比图"""
    fig, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))

    for data, label, color in zip(data_list, labels, colors):
        data_clean = np.array(data).flatten()
        data_clean = data_clean[np.isfinite(data_clean)]
        if len(data_clean) < 3:
            continue
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data_clean)
        x_range = np.linspace(data_clean.min(), data_clean.max(), 200)
        density = kde(x_range)
        ax.plot(x_range, density, color=color, linewidth=2, label=label)
        if fill:
            ax.fill_between(x_range, density, alpha=0.15, color=color)

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=10)
    save_and_close(fig, out_path)


def plot_swarm_strip(df, x_col, y_col, title, out_path, hue_col=None,
                     palette='Set2', figsize=(10, 6)):
    """蜂群图/抖动散点图，展示每个数据点"""
    fig, ax = plt.subplots(figsize=figsize)

    if hue_col and hue_col in df.columns:
        sns.stripplot(data=df, x=x_col, y=y_col, hue=hue_col,
                      palette=palette, dodge=True, alpha=0.6, size=4, ax=ax)
        sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col,
                    palette=palette, dodge=True, width=0.4,
                    boxprops=dict(alpha=0.5), ax=ax, legend=False)
    else:
        sns.stripplot(data=df, x=x_col, y=y_col, color='steelblue',
                      alpha=0.5, size=4, ax=ax)
        sns.boxplot(data=df, x=x_col, y=y_col, color='steelblue',
                    width=0.4, boxprops=dict(alpha=0.4), ax=ax)

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    save_and_close(fig, out_path)


def plot_boxen(df, features, title, out_path, palette='Set2', figsize=(12, 6)):
    """增强箱线图 (letter-value plot)，适合大数据集"""
    fig, ax = plt.subplots(figsize=figsize)
    df_melt = df[features].melt()

    sns.boxenplot(data=df_melt, x='variable', y='value',
                  palette=palette, ax=ax, k_depth='proportion')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('')
    save_and_close(fig, out_path)


def plot_regplot(df, x_col, y_col, title, out_path, color='steelblue',
                 figsize=(8, 6), add_identity=False):
    """带回归线和置信区间的散点图"""
    fig, ax = plt.subplots(figsize=figsize)

    x = df[x_col].values
    y = df[y_col].values
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) > 3:
        # 计算相关系数和p值
        r, p = stats.pearsonr(x, y)
        sns.regplot(x=x, y=y, ax=ax, color=color, scatter_kws={'alpha': 0.5, 's': 30},
                    line_kws={'color': 'red', 'linewidth': 2})
        ax.annotate(f'Pearson r={r:.3f}, p={p:.2e}',
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.scatter(x, y, alpha=0.5, color=color, s=30)

    if add_identity:
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, 'k--', alpha=0.3, linewidth=1)

    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel(y_col, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    save_and_close(fig, out_path)


def plot_joint_distribution(df, x_col, y_col, title, out_path, kind='scatter',
                            color='steelblue', figsize=(9, 9)):
    """联合分布图（散点+边缘分布）"""
    fig = plt.figure(figsize=figsize)

    x = df[x_col].values
    y = df[y_col].values
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) < 3:
        save_and_close(fig, out_path)
        return

    # 主散点图
    ax_main = fig.add_axes([0.1, 0.1, 0.65, 0.65])
    ax_main.scatter(x, y, alpha=0.4, s=20, c=color, edgecolors='none')

    # 添加回归线
    z = np.polyfit(x, y, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax_main.plot(x_line, p_line(x_line), 'r-', linewidth=2, alpha=0.7)

    r, p = stats.pearsonr(x, y)
    ax_main.annotate(f'r={r:.3f}\np={p:.2e}',
                     xy=(0.95, 0.05), xycoords='axes fraction',
                     fontsize=10, ha='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax_main.set_xlabel(x_col, fontsize=11)
    ax_main.set_ylabel(y_col, fontsize=11)

    # X轴边缘分布
    ax_top = fig.add_axes([0.1, 0.78, 0.65, 0.15])
    ax_top.hist(x, bins=30, color=color, alpha=0.6, edgecolor='white')
    ax_top.set_xlim(ax_main.get_xlim())
    ax_top.set_xticks([])
    ax_top.set_yticks([])

    # Y轴边缘分布
    ax_right = fig.add_axes([0.78, 0.1, 0.15, 0.65])
    ax_right.hist(y, bins=30, color=color, alpha=0.6, edgecolor='white', orientation='horizontal')
    ax_right.set_ylim(ax_main.get_ylim())
    ax_right.set_xticks([])
    ax_right.set_yticks([])

    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    save_and_close(fig, out_path)


def plot_ecdf(data_series, title, xlabel, out_path, color='steelblue',
              figsize=(8, 6)):
    """经验累积分布函数图"""
    fig, ax = plt.subplots(figsize=figsize)

    data_clean = np.sort(np.array(data_series).flatten())
    data_clean = data_clean[np.isfinite(data_clean)]
    n = len(data_clean)
    y = np.arange(1, n + 1) / n

    ax.step(data_clean, y, color=color, linewidth=2, where='post')
    ax.fill_between(data_clean, y, alpha=0.15, color=color, step='post')

    # 添加分位数标记
    for q, label in [(0.25, 'Q1'), (0.5, 'Median'), (0.75, 'Q3')]:
        val = np.quantile(data_clean, q)
        ax.axvline(val, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(q, color='gray', linestyle='--', alpha=0.5)
        ax.scatter([val], [q], color='red', s=50, zorder=5)
        ax.annotate(f'{label}: {val:.3f}', xy=(val, q),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    save_and_close(fig, out_path)


def plot_qq(data_series, title, out_path, color='steelblue', figsize=(8, 6)):
    """QQ正态检验图"""
    fig, ax = plt.subplots(figsize=figsize)

    data_clean = np.array(data_series).flatten()
    data_clean = data_clean[np.isfinite(data_clean)]

    stats.probplot(data_clean, dist="norm", plot=ax)

    ax.get_lines()[0].set_markerfacecolor(color)
    ax.get_lines()[0].set_markeredgecolor('white')
    ax.get_lines()[0].set_markersize(4)
    ax.get_lines()[0].set_alpha(0.5)
    ax.get_lines()[1].set_color('red')
    ax.get_lines()[1].set_linewidth(2)

    ax.set_title(title, fontsize=13, fontweight='bold')
    save_and_close(fig, out_path)


def plot_pairgrid(df, features, title, out_path, hue_col=None,
                  palette='tab10', figsize=None):
    """特征对散点图矩阵（对角线为分布）"""
    if len(features) > 6:
        features = features[:6]  # 限制特征数量

    n = len(features)
    if figsize is None:
        figsize = (2.5 * n, 2.5 * n)

    fig, axes = plt.subplots(n, n, figsize=figsize)

    if hue_col and hue_col in df.columns:
        hue_values = df[hue_col].values
        unique_hues = np.unique(hue_values[np.isfinite(hue_values)])
        cmap = plt.cm.get_cmap(palette, len(unique_hues))
    else:
        hue_values = None
        unique_hues = None
        cmap = None

    for i, fi in enumerate(features):
        for j, fj in enumerate(features):
            ax = axes[i, j]
            if i == j:
                # 对角线：分布
                data_i = df[fi].dropna().values
                ax.hist(data_i, bins=20, color='steelblue', alpha=0.6, edgecolor='white')
                ax.set_title(fi, fontsize=9)
            else:
                # 非对角线：散点图
                x = df[fj].values
                y = df[fi].values
                valid = np.isfinite(x) & np.isfinite(y)

                if hue_values is not None and unique_hues is not None:
                    for idx_h, h_val in enumerate(unique_hues):
                        mask = valid & (hue_values == h_val)
                        ax.scatter(x[mask], y[mask], alpha=0.4, s=8,
                                   color=cmap(idx_h), label=str(h_val)[:20])
                else:
                    ax.scatter(x[valid], y[valid], alpha=0.4, s=8, color='steelblue')

                if i == n - 1:
                    ax.set_xlabel(fj, fontsize=8)
                if j == 0:
                    ax.set_ylabel(fi, fontsize=8)

            ax.tick_params(axis='both', labelsize=7)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    save_and_close(fig, out_path)


def plot_clustermap(df, features, title, out_path, cmap='RdBu_r',
                    figsize=(12, 10)):
    """层次聚类热图"""
    subset = df[features].dropna()
    if len(subset) < 3:
        return

    # 标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled = scaler.fit_transform(subset)
    scaled_df = pd.DataFrame(scaled, columns=features, index=subset.index)

    g = sns.clustermap(scaled_df, cmap=cmap, figsize=figsize,
                       dendrogram_ratio=0.15, cbar_pos=(0.02, 0.8, 0.03, 0.15),
                       linewidths=0.1, annot=False, fmt='.1f',
                       xticklabels=True, yticklabels=False)
    g.fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    g.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(g.fig)
    print(f"  Saved: {out_path}")


def plot_pca_2d(df, features, title, out_path, hue_col=None,
                palette='tab10', figsize=(10, 8)):
    """PCA降维散点图"""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    subset = df[features].dropna()
    if len(subset) < 5:
        return

    scaler = StandardScaler()
    scaled = scaler.fit_transform(subset)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(scaled)

    fig, ax = plt.subplots(figsize=figsize)

    explained = pca.explained_variance_ratio_

    if hue_col and hue_col in df.columns:
        hues = df.loc[subset.index, hue_col].values
        unique_hues = np.unique(hues[np.isfinite(hues)])
        cmap = plt.cm.get_cmap(palette, len(unique_hues))
        for idx, h_val in enumerate(unique_hues):
            mask = hues == h_val
            ax.scatter(pcs[mask, 0], pcs[mask, 1], alpha=0.6, s=40,
                       color=cmap(idx), label=str(h_val)[:20], edgecolors='black', linewidth=0.3)
        ax.legend(fontsize=9)
    else:
        scatter = ax.scatter(pcs[:, 0], pcs[:, 1], alpha=0.6, s=40,
                             c=range(len(pcs)), cmap='viridis',
                             edgecolors='black', linewidth=0.3)
        plt.colorbar(scatter, ax=ax, label='Index')

    ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}%)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3)

    save_and_close(fig, out_path)


def plot_annotated_boxplot(df, x_col, y_col, title, out_path,
                           palette='Set2', figsize=(10, 6)):
    """带统计检验标注的箱线图"""
    fig, ax = plt.subplots(figsize=figsize)

    groups = df[x_col].unique()
    group_data = [df[df[x_col] == g][y_col].dropna().values for g in groups]
    group_data = [g for g in group_data if len(g) > 1]

    if len(group_data) >= 2:
        from scipy.stats import f_oneway
        try:
            f_stat, p_val = f_oneway(*group_data)
        except ValueError:
            f_stat, p_val = np.nan, np.nan

        sns.boxplot(data=df, x=x_col, y=y_col, palette=palette, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)

        if not np.isnan(p_val):
            sig_text = f'ANOVA: F={f_stat:.2f}, p={p_val:.3f}'
            if p_val < 0.001:
                sig_text += ' ***'
            elif p_val < 0.01:
                sig_text += ' **'
            elif p_val < 0.05:
                sig_text += ' *'
            ax.text(0.5, 0.98, sig_text, transform=ax.transAxes,
                    fontsize=10, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    ax.set_title(title, fontsize=13, fontweight='bold')
    save_and_close(fig, out_path)


def plot_multi_slice_grid(image, mask, out_path, n_slices=9,
                          cmap='gray', mask_alpha=0.3, figsize=None):
    """多切片网格展示（CT + mask 叠加）"""
    if image.ndim != 3 or mask.ndim != 3:
        return

    # 找到mask存在的z范围
    z_sums = np.sum(mask > 0, axis=(1, 2))
    z_valid = np.where(z_sums > 0)[0]
    if len(z_valid) == 0:
        return

    z_min, z_max = z_valid[0], z_valid[-1]
    z_indices = np.linspace(z_min, z_max, min(n_slices, z_max - z_min + 1), dtype=int)

    n = len(z_indices)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    if figsize is None:
        figsize = (3.5 * cols, 3.5 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, z in enumerate(z_indices):
        ax = axes[idx]
        ax.imshow(image[z], cmap=cmap, vmin=image.min(), vmax=image.max())

        # 叠加mask
        mask_rgba = np.zeros((*mask[z].shape, 4))
        mask_rgba[mask[z] > 0] = [1, 0, 0, mask_alpha]
        ax.imshow(mask_rgba)

        ax.set_title(f'Z={z}', fontsize=10)
        ax.axis('off')

    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    fig.suptitle('Multi-Slice CT with Tumor Mask', fontsize=13, fontweight='bold')
    save_and_close(fig, out_path)


def plot_3d_scatter_projection(coords, values, title, out_path,
                               cmap='viridis', figsize=(10, 8)):
    """3D坐标点云投影（从三个正交方向）"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # XY投影（轴向）
    ax = axes[0, 0]
    sc = ax.scatter(coords[:, 2], coords[:, 1], c=values, cmap=cmap, s=2, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Axial (XY) Projection')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, fraction=0.046)

    # XZ投影（冠状）
    ax = axes[0, 1]
    sc = ax.scatter(coords[:, 2], coords[:, 0], c=values, cmap=cmap, s=2, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Coronal (XZ) Projection')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, fraction=0.046)

    # YZ投影（矢状）
    ax = axes[1, 0]
    sc = ax.scatter(coords[:, 1], coords[:, 0], c=values, cmap=cmap, s=2, alpha=0.6)
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_title('Sagittal (YZ) Projection')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, fraction=0.046)

    # 3D视角示意
    ax = axes[1, 1]
    ax.scatter(coords[:, 2], coords[:, 1], c=values, cmap=cmap, s=1, alpha=0.3)
    # 添加轮廓
    from scipy.spatial import ConvexHull
    try:
        xy_coords = coords[:, [2, 1]]
        if len(xy_coords) > 3:
            hull = ConvexHull(xy_coords)
            for simplex in hull.simplices:
                ax.plot(xy_coords[simplex, 0], xy_coords[simplex, 1], 'r-', alpha=0.3, linewidth=0.5)
    except Exception:
        pass
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('XY with Convex Hull')
    ax.set_aspect('equal')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    save_and_close(fig, out_path)


def plot_surface_projection(mask, title, out_path, figsize=(10, 8)):
    """肿瘤表面投影（最大强度投影风格）"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 轴向MIP
    ax = axes[0, 0]
    mip_axial = np.max(mask, axis=0)
    ax.imshow(mip_axial, cmap='Reds', interpolation='nearest')
    ax.set_title('Axial MIP')
    ax.axis('off')

    # 冠状MIP
    ax = axes[0, 1]
    mip_coronal = np.max(mask, axis=1)
    ax.imshow(mip_coronal, cmap='Reds', interpolation='nearest')
    ax.set_title('Coronal MIP')
    ax.axis('off')

    # 矢状MIP
    ax = axes[1, 0]
    mip_sagittal = np.max(mask, axis=2)
    ax.imshow(mip_sagittal, cmap='Reds', interpolation='nearest')
    ax.set_title('Sagittal MIP')
    ax.axis('off')

    # 3D轮廓
    ax = axes[1, 1]
    coords = np.argwhere(mask > 0)
    if len(coords) > 0:
        ax.scatter(coords[:, 2], coords[:, 1], c=coords[:, 0],
                   cmap='viridis', s=1, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('3D Point Cloud (XY view, color=Z)')
        ax.set_aspect('equal')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    save_and_close(fig, out_path)


def plot_histogram_comparison(df, feature_cols, group_col, title, out_path,
                              palette='tab10', figsize=(12, 8)):
    """多组直方图对比"""
    groups = df[group_col].unique()
    n_groups = len(groups)
    n_features = len(feature_cols)

    cols = min(3, n_features)
    rows = (n_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] if isinstance(figsize, tuple) else 12,
                                                  rows * 3.5))
    if n_features == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    cmap = plt.cm.get_cmap(palette, n_groups)

    for idx, feat in enumerate(feature_cols):
        ax = axes[idx]
        for g_idx, group in enumerate(groups):
            data = df[df[group_col] == group][feat].dropna().values
            ax.hist(data, bins=20, alpha=0.4, color=cmap(g_idx),
                    label=str(group)[:20], edgecolor='white')
        ax.set_title(feat, fontsize=11)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    save_and_close(fig, out_path)


def plot_feature_importance(features, importances, title, out_path,
                            color='steelblue', figsize=(10, 8)):
    """特征重要性水平条形图"""
    fig, ax = plt.subplots(figsize=figsize)

    # 排序
    idx = np.argsort(importances)
    features_sorted = [features[i] for i in idx]
    importances_sorted = [importances[i] for i in idx]

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(features)))
    bars = ax.barh(range(len(features)), importances_sorted, color=colors, edgecolor='white')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features_sorted, fontsize=9)
    ax.set_xlabel('Importance / Magnitude', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, importances_sorted)):
        ax.text(val + 0.01 * max(importances_sorted), i, f'{val:.3f}',
                va='center', fontsize=8)

    save_and_close(fig, out_path)


def plot_heatmap_with_dendrogram(df, features, row_labels, title, out_path,
                                  cmap='RdBu_r', figsize=(12, 10)):
    """带聚类树状图的热图"""
    subset = df[features].dropna()
    if len(subset) < 3:
        return

    # 归一化
    subset_norm = (subset - subset.min()) / (subset.max() - subset.min() + 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={'width_ratios': [1, 0.15]})

    im = axes[0].imshow(subset_norm.values, cmap=cmap, aspect='auto',
                        interpolation='nearest')
    axes[0].set_xticks(range(len(features)))
    axes[0].set_xticklabels(features, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticks(range(len(subset)))
    axes[0].set_yticklabels([row_labels[i] if i < len(row_labels) else ''
                              for i in range(len(subset))], fontsize=6)
    axes[0].set_title(title, fontsize=13, fontweight='bold')

    plt.colorbar(im, cax=axes[1])

    save_and_close(fig, out_path)
