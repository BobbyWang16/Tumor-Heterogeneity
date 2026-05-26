"""
各评分模型的结果可视化

为每个ITH评分模型提供专用的可视化函数：
- BIH: 分形维数分布、Box-counting拟合曲线、小提琴图、QQ图
- DHI: 密度特征分布、相关性热图、小提琴图、PCA降维、蜂群图
- HAB: 生境占比、HU值分布、生境对比热图
- PTH: 超像素分布、纹理变异、特征重要性
- SHI: 几何特征分布、分形-空隙度关系、雷达图
- THI: ITHscore分布、聚类标签图、ECDF图
- ITH-FS: 置信空间图、综合对比、联合分布

所有模型额外新增：
  - 小提琴图+箱线图组合
  - 联合分布图
  - 经验累积分布图
  - 特征对散点矩阵
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import ndimage

from .utils import (setup_figure, save_and_close, plot_distribution_with_stats,
                    plot_correlation_heatmap, plot_radar_chart,
                    overlay_mask_on_slice, plot_multi_slice_comparison,
                    plot_violin_box, plot_kde, plot_swarm_strip,
                    plot_boxen, plot_regplot, plot_joint_distribution,
                    plot_ecdf, plot_qq, plot_pairgrid, plot_pca_2d,
                    plot_annotated_boxplot, plot_feature_importance,
                    plot_multi_slice_grid, plot_3d_scatter_projection,
                    plot_surface_projection, plot_clustermap,
                    plot_heatmap_with_dendrogram)


# ============================================================
# 1. BIH 可视化
# ============================================================

def visualize_bih_results(csv_path, out_dir, dataset_name):
    """
    BIH (Box-counting Intratumor Heterogeneity) 结果可视化

    包含:
    1. 分形维数FD分布直方图
    2. logA分布直方图
    3. FD vs logA 散点图
    4. R2拟合优度分布
    5. 新增: FD小提琴图+箱线图组合
    6. 新增: FD QQ正态检验图
    7. 新增: FD vs R2 联合分布图
    8. 新增: 特征对散点矩阵
    9. 新增: Box-counting拟合曲线（log-log）
    10. 新增: FD ECDF经验累积分布
    """
    df = pd.read_csv(csv_path)
    df = _normalize_column_names(df)
    os.makedirs(out_dir, exist_ok=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 1. 分形维数FD分布
    if 'fd' in df.columns:
        plot_distribution_with_stats(
            df['fd'].dropna(),
            f'{dataset_name} - Fractal Dimension (FD) Distribution',
            'Fractal Dimension', os.path.join(out_dir, '01_fd_distribution.png')
        )

    # 2. logA分布
    if 'logA' in df.columns:
        plot_distribution_with_stats(
            df['logA'].dropna(),
            f'{dataset_name} - logA (Intercept) Distribution',
            'logA', os.path.join(out_dir, '02_logA_distribution.png'),
            color='coral'
        )

    # 3. FD vs logA 散点图
    if 'fd' in df.columns and 'logA' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(df['fd'], df['logA'], c=df.get('r2', 0.8),
                            cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Fractal Dimension (FD)', fontsize=12)
        ax.set_ylabel('logA (Intercept)', fontsize=12)
        ax.set_title(f'{dataset_name} - FD vs logA', fontsize=13, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('R² (Goodness of Fit)', fontsize=11)
        save_and_close(fig, os.path.join(out_dir, '03_fd_vs_logA_scatter.png'))

    # 4. R2分布
    if 'r2' in df.columns:
        plot_distribution_with_stats(
            df['r2'].dropna(),
            f'{dataset_name} - R² (Goodness of Fit) Distribution',
            'R²', os.path.join(out_dir, '04_r2_distribution.png'),
            color='mediumseagreen'
        )

    # 5. 所有数值特征的相关性热图
    if len(num_cols) > 2:
        plot_correlation_heatmap(
            df, num_cols, f'{dataset_name} - BIH Feature Correlation',
            os.path.join(out_dir, '05_feature_correlation.png')
        )

    # 6. 新增: FD 小提琴图+箱线图
    if 'fd' in df.columns:
        plot_violin_box(
            df['fd'].dropna().values,
            f'{dataset_name} - FD Violin + Box Plot',
            'Fractal Dimension', os.path.join(out_dir, '06_fd_violin_box.png'),
            color='steelblue'
        )

    # 7. 新增: FD QQ正态检验
    if 'fd' in df.columns:
        plot_qq(
            df['fd'].dropna().values,
            f'{dataset_name} - FD Q-Q Normal Plot',
            os.path.join(out_dir, '07_fd_qq_plot.png'),
            color='steelblue'
        )

    # 8. 新增: FD vs R2 联合分布图
    if 'fd' in df.columns and 'r2' in df.columns:
        plot_joint_distribution(
            df, 'fd', 'r2',
            f'{dataset_name} - FD vs R² Joint Distribution',
            os.path.join(out_dir, '08_fd_r2_joint.png'),
            color='steelblue'
        )

    # 9. 新增: 特征对散点矩阵
    if len(num_cols) >= 3:
        plot_pairgrid(
            df, num_cols[:5],
            f'{dataset_name} - BIH Feature Pair Grid',
            os.path.join(out_dir, '09_feature_pairgrid.png')
        )

    # 10. 新增: FD ECDF经验累积分布
    if 'fd' in df.columns:
        plot_ecdf(
            df['fd'].dropna().values,
            f'{dataset_name} - FD Empirical CDF',
            'Fractal Dimension', os.path.join(out_dir, '10_fd_ecdf.png'),
            color='steelblue'
        )

    # 11. 新增: 多特征KDE对比
    kde_features = [c for c in ['fd', 'r2', 'logA', 'residual'] if c in df.columns]
    if len(kde_features) >= 2:
        plot_kde(
            [df[c].dropna().values for c in kde_features],
            kde_features,
            f'{dataset_name} - BIH Feature KDE Comparison',
            'Value',
            os.path.join(out_dir, '11_feature_kde_comparison.png')
        )

    print(f"[BIH] Visualization saved to {out_dir}")


# ============================================================
# 2. DHI 可视化
# ============================================================

def visualize_dhi_results(csv_path, out_dir, dataset_name):
    """
    DHI (Density Heterogeneity Index) 结果可视化

    包含:
    1. 关键特征箱线图
    2. 特征相关性热图
    3. 均值-标准差散点图
    4. 新增: 关键特征小提琴图组合
    5. 新增: 增强箱线图 (letter-value)
    6. 新增: PCA降维散点图
    7. 新增: 特征蜂群图
    8. 新增: ECDF累积分布
    9. 新增: 特征对散点矩阵
    10. 新增: 带统计标注的箱线图
    """
    df = pd.read_csv(csv_path)
    df = _normalize_column_names(df)
    os.makedirs(out_dir, exist_ok=True)

    # 选择关键特征
    key_features = [c for c in df.columns if any(k in c.lower()
                    for k in ['mean', 'std', 'cv', 'entropy', 'skewness', 'kurtosis'])]

    # 1. 关键特征箱线图
    if len(key_features) >= 3:
        fig, ax = plt.subplots(figsize=(max(10, len(key_features)*0.8), 6))
        df_melt = df[key_features[:12]].melt()
        sns.boxplot(data=df_melt, x='variable', y='value', ax=ax, palette='Set2')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_title(f'{dataset_name} - DHI Key Features Boxplot', fontsize=13, fontweight='bold')
        ax.set_xlabel('')
        save_and_close(fig, os.path.join(out_dir, '01_feature_boxplot.png'))

    # 2. 特征相关性热图
    if len(key_features) >= 3:
        plot_correlation_heatmap(
            df, key_features[:15],
            f'{dataset_name} - DHI Feature Correlation',
            os.path.join(out_dir, '02_correlation_heatmap.png')
        )

    # 3. 均值-标准差散点图
    mean_cols = [c for c in df.columns if 'mean' in c.lower()]
    std_cols = [c for c in df.columns if 'std' in c.lower()]
    if mean_cols and std_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df[mean_cols[0]], df[std_cols[0]], alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.set_xlabel(mean_cols[0], fontsize=11)
        ax.set_ylabel(std_cols[0], fontsize=11)
        ax.set_title(f'{dataset_name} - Mean vs Std', fontsize=13, fontweight='bold')
        save_and_close(fig, os.path.join(out_dir, '03_mean_vs_std.png'))

    # 4. 新增: 关键特征小提琴图组合
    if len(key_features) >= 3 and 'score' in df.columns:
        for feat in key_features[:3]:
            plot_violin_box(
                df[feat].dropna().values,
                f'{dataset_name} - {feat} Violin Plot',
                feat, os.path.join(out_dir, f'04_violin_{feat}.png'),
                color='coral'
            )

    # 5. 新增: 增强箱线图
    if len(key_features) >= 3:
        plot_boxen(
            df, key_features[:8],
            f'{dataset_name} - DHI Letter-Value Plot',
            os.path.join(out_dir, '05_boxen_features.png'),
            palette='Set2'
        )

    # 6. 新增: PCA降维散点图
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 4:
        plot_pca_2d(
            df, numeric_cols[:10],
            f'{dataset_name} - DHI PCA Projection',
            os.path.join(out_dir, '06_pca_projection.png')
        )

    # 7. 新增: 特征蜂群图
    if len(key_features) >= 3:
        df_melt = df[key_features[:8]].melt()
        df_melt['feature'] = df_melt['variable']
        plot_swarm_strip(
            df_melt, 'feature', 'value',
            f'{dataset_name} - DHI Feature Swarm Plot',
            os.path.join(out_dir, '07_swarm_strip.png'),
            palette='Set2'
        )

    # 8. 新增: CV/Entropy ECDF
    for feat in ['cv', 'entropy']:
        feat_cols = [c for c in df.columns if feat in c.lower()]
        if feat_cols:
            plot_ecdf(
                df[feat_cols[0]].dropna().values,
                f'{dataset_name} - {feat_cols[0]} ECDF',
                feat_cols[0], os.path.join(out_dir, f'08_ecdf_{feat}.png'),
                color='teal'
            )
            break

    # 9. 新增: 特征对散点矩阵
    if len(key_features) >= 3:
        plot_pairgrid(
            df, key_features[:5],
            f'{dataset_name} - DHI Feature Pair Grid',
            os.path.join(out_dir, '09_pairgrid_features.png')
        )

    # 10. 新增: 带统计标注的箱线图（按score分位）
    if 'score' in df.columns and len(key_features) >= 2:
        df_q = df.copy()
        df_q['score_quartile'] = pd.qcut(df_q['score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        plot_annotated_boxplot(
            df_q, 'score_quartile', key_features[0],
            f'{dataset_name} - {key_features[0]} by Score Quartile',
            os.path.join(out_dir, '10_annotated_boxplot.png'),
            palette='Set2'
        )

    # 11. 新增: 特征KDE对比
    if len(key_features) >= 2:
        plot_kde(
            [df[c].dropna().values for c in key_features[:4]],
            key_features[:4],
            f'{dataset_name} - DHI Feature KDE Comparison',
            'Value',
            os.path.join(out_dir, '11_kde_comparison.png')
        )

    print(f"[DHI] Visualization saved to {out_dir}")


# ============================================================
# 3. HAB 可视化
# ============================================================

def visualize_hab_results(csv_path, out_dir, dataset_name):
    """
    HAB (Habitat Analysis Based) 结果可视化

    包含:
    1. 生境体积占比饼图
    2. JS散度分布
    3. 各病例生境组成堆叠柱状图
    4. 新增: 生境HU值分布小提琴图
    5. 新增: 生境对比热图
    6. 新增: 生境间距离热图
    7. 新增: 生境特征蜂群图
    8. 新增: 各生境占比箱线图
    9. 新增: 生境HU KDE对比
    """
    df = pd.read_csv(csv_path)
    df = _normalize_column_names(df)
    os.makedirs(out_dir, exist_ok=True)

    # 1. 生境体积占比
    hab_vol_cols = [c for c in df.columns if 'volume' in c.lower() and 'habitat' in c.lower()]
    if hab_vol_cols:
        total_volumes = df[hab_vol_cols].sum()
        fig, ax = plt.subplots(figsize=(7, 7))
        colors = plt.cm.Set3(np.linspace(0, 1, len(total_volumes)))
        wedges, texts, autotexts = ax.pie(
            total_volumes, labels=total_volumes.index, autopct='%1.1f%%',
            colors=colors, startangle=90
        )
        ax.set_title(f'{dataset_name} - Habitat Volume Proportion', fontsize=13, fontweight='bold')
        save_and_close(fig, os.path.join(out_dir, '01_habitat_pie.png'))

    # 2. JS散度分布
    js_cols = [c for c in df.columns if 'js' in c.lower() or 'jensen' in c.lower()]
    if js_cols:
        for col in js_cols[:3]:
            plot_distribution_with_stats(
                df[col].dropna(),
                f'{dataset_name} - {col} Distribution',
                col, os.path.join(out_dir, f'02_{col}_dist.png'),
                color='teal'
            )

    # 3. 各病例生境组成（堆叠柱状图）
    if hab_vol_cols and len(df) <= 30:
        fig, ax = plt.subplots(figsize=(max(10, len(df)*0.3), 6))
        bottom = np.zeros(len(df))
        colors = plt.cm.Set3(np.linspace(0, 1, len(hab_vol_cols)))
        for idx, col in enumerate(hab_vol_cols):
            vals = df[col].fillna(0).values
            ax.bar(range(len(df)), vals, bottom=bottom, label=col, color=colors[idx])
            bottom += vals
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df.index, rotation=90, fontsize=7)
        ax.set_ylabel('Volume', fontsize=11)
        ax.set_title(f'{dataset_name} - Habitat Composition per Case', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        save_and_close(fig, os.path.join(out_dir, '03_habitat_composition.png'))

    # 4. 新增: 生境HU值相关分布小提琴图
    hu_cols = [c for c in df.columns if 'hu' in c.lower() or 'mean' in c.lower()]
    if hu_cols:
        for col in hu_cols[:3]:
            plot_violin_box(
                df[col].dropna().values,
                f'{dataset_name} - {col} Violin Plot',
                col, os.path.join(out_dir, f'04_violin_{col}.png'),
                color='mediumseagreen'
            )

    # 5. 新增: 生境对比热图
    if len(hab_vol_cols) >= 2:
        plot_heatmap_with_dendrogram(
            df, hab_vol_cols,
            [f'Case {i+1}' for i in range(min(len(df), 50))],
            f'{dataset_name} - Habitat Volume Heatmap',
            os.path.join(out_dir, '05_habitat_heatmap.png'),
            cmap='YlOrRd'
        )

    # 6. 新增: 生境特征KDE对比
    if len(hu_cols) >= 2:
        plot_kde(
            [df[c].dropna().values for c in hu_cols[:4]],
            hu_cols[:4],
            f'{dataset_name} - Habitat HU KDE Comparison',
            'HU Value',
            os.path.join(out_dir, '06_habitat_kde.png')
        )

    # 7. 新增: 生境占比ECDF
    if hab_vol_cols:
        for col in hab_vol_cols[:2]:
            plot_ecdf(
                df[col].dropna().values,
                f'{dataset_name} - {col} ECDF',
                'Proportion', os.path.join(out_dir, f'07_ecdf_{col}.png'),
                color='indianred'
            )

    # 8. 新增: 生境组成增强箱线图
    if len(hab_vol_cols) >= 2:
        plot_boxen(
            df, hab_vol_cols,
            f'{dataset_name} - Habitat Volume Distribution',
            os.path.join(out_dir, '08_habitat_boxen.png'),
            palette='Set3'
        )

    # 9. 新增: 生境间相关性热图
    all_hab_cols = [c for c in df.columns if 'habitat' in c.lower()]
    if len(all_hab_cols) >= 3:
        plot_correlation_heatmap(
            df, all_hab_cols[:10],
            f'{dataset_name} - Habitat Feature Correlation',
            os.path.join(out_dir, '09_habitat_correlation.png')
        )

    print(f"[HAB] Visualization saved to {out_dir}")


# ============================================================
# 4. PTH 可视化
# ============================================================

def visualize_pth_results(csv_path, out_dir, dataset_name):
    """
    PTH (Peritumoral Heterogeneity) 结果可视化

    包含:
    1. 超像素数量分布
    2. CV相关特征分布
    3. 均值-标准差散点图
    4. 新增: CV特征小提琴图
    5. 新增: PTH特征重要性排序
    6. 新增: 特征ECDF
    7. 新增: CV vs IQR 联合分布
    8. 新增: 特征蜂群图
    9. 新增: 特征对散点矩阵
    10. 新增: 特征PCA投影
    """
    df = pd.read_csv(csv_path)
    df = _normalize_column_names(df)
    os.makedirs(out_dir, exist_ok=True)

    # 1. 超像素数量分布
    if 'n_superpixels' in df.columns:
        plot_distribution_with_stats(
            df['n_superpixels'].dropna(),
            f'{dataset_name} - Superpixel Count Distribution',
            'Number of Superpixels', os.path.join(out_dir, '01_superpixel_count.png'),
            color='mediumpurple'
        )

    # 2. 变异系数(CV)相关特征分布
    cv_cols = [c for c in df.columns if 'cv' in c.lower() or 'coefficient' in c.lower()]
    if cv_cols:
        fig, axes = plt.subplots(1, min(len(cv_cols), 4), figsize=(5*min(len(cv_cols), 4), 4))
        if len(cv_cols) == 1:
            axes = [axes]
        for idx, col in enumerate(cv_cols[:4]):
            axes[idx].hist(df[col].dropna(), bins=20, color='steelblue', edgecolor='white')
            axes[idx].set_title(col, fontsize=10)
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Count')
        fig.suptitle(f'{dataset_name} - PTH Coefficient of Variation Features',
                     fontsize=13, fontweight='bold')
        save_and_close(fig, os.path.join(out_dir, '02_cv_distributions.png'))

    # 3. 均值-标准差散点图
    hu_mean_cols = [c for c in df.columns if 'mean' in c.lower() and ('hu' in c.lower() or 'intensity' in c.lower())]
    hu_std_cols = [c for c in df.columns if 'std' in c.lower() and ('hu' in c.lower() or 'intensity' in c.lower())]
    if hu_mean_cols and hu_std_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df[hu_mean_cols[0]], df[hu_std_cols[0]], alpha=0.6, c='coral', edgecolors='black')
        ax.set_xlabel(hu_mean_cols[0], fontsize=11)
        ax.set_ylabel(hu_std_cols[0], fontsize=11)
        ax.set_title(f'{dataset_name} - Peritumoral Mean vs Std HU', fontsize=13, fontweight='bold')
        save_and_close(fig, os.path.join(out_dir, '03_mean_vs_std_hu.png'))

    # 4. 新增: CV特征小提琴图
    if cv_cols:
        for col in cv_cols[:3]:
            plot_violin_box(
                df[col].dropna().values,
                f'{dataset_name} - {col} Violin Plot',
                col, os.path.join(out_dir, f'04_violin_{col}.png'),
                color='mediumpurple'
            )

    # 5. 新增: PTH特征重要性排序（用标准差作为重要性代理）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 3:
        importances = [df[c].dropna().std() for c in numeric_cols]
        plot_feature_importance(
            numeric_cols[:12],
            importances[:12],
            f'{dataset_name} - PTH Feature Variability Ranking',
            os.path.join(out_dir, '05_feature_importance.png'),
            color='mediumpurple'
        )

    # 6. 新增: 主要特征ECDF
    if cv_cols:
        plot_ecdf(
            df[cv_cols[0]].dropna().values,
            f'{dataset_name} - {cv_cols[0]} ECDF',
            cv_cols[0], os.path.join(out_dir, '06_ecdf_main.png'),
            color='mediumpurple'
        )

    # 7. 新增: CV vs IQR 联合分布
    iqr_cols = [c for c in df.columns if 'iqr' in c.lower()]
    if cv_cols and iqr_cols:
        plot_joint_distribution(
            df, cv_cols[0], iqr_cols[0],
            f'{dataset_name} - CV vs IQR Joint Distribution',
            os.path.join(out_dir, '07_cv_iqr_joint.png'),
            color='mediumpurple'
        )

    # 8. 新增: 特征蜂群图
    if len(numeric_cols) >= 3:
        df_melt = df[numeric_cols[:6]].melt()
        df_melt['feature'] = df_melt['variable']
        plot_swarm_strip(
            df_melt, 'feature', 'value',
            f'{dataset_name} - PTH Feature Swarm Plot',
            os.path.join(out_dir, '08_swarm_features.png'),
            palette='tab10'
        )

    # 9. 新增: 特征对散点矩阵
    if len(numeric_cols) >= 4:
        plot_pairgrid(
            df, numeric_cols[:5],
            f'{dataset_name} - PTH Feature Pair Grid',
            os.path.join(out_dir, '09_pairgrid.png')
        )

    # 10. 新增: PCA投影
    if len(numeric_cols) >= 4:
        plot_pca_2d(
            df, numeric_cols[:10],
            f'{dataset_name} - PTH PCA Projection',
            os.path.join(out_dir, '10_pca_projection.png')
        )

    print(f"[PTH] Visualization saved to {out_dir}")


# ============================================================
# 5. SHI 可视化
# ============================================================

def visualize_shi_results(csv_path, out_dir, dataset_name):
    """
    SHI (Shape Heterogeneity Index) 结果可视化

    包含:
    1. 几何特征分布
    2. 分形维数分布
    3. 空隙度分布
    4. 分形维数vs空隙度散点图
    5. 形状特征雷达图
    6. 新增: 几何特征小提琴图
    7. 新增: FD vs Surface FD 联合分布
    8. 新增: 几何特征ECDF
    9. 新增: 空隙度变化拟合曲线
    10. 新增: 特征KDE对比
    11. 新增: 形状特征层次聚类热图
    12. 新增: PCA降维散点图
    """
    df = pd.read_csv(csv_path)
    df = _normalize_column_names(df)
    os.makedirs(out_dir, exist_ok=True)

    # 1. 几何特征分布
    geo_features = ['sphericity', 'solidity', 'compactness', 'extent']
    available_geo = [c for c in df.columns if any(g in c.lower() for g in geo_features)]
    if available_geo:
        fig, axes = plt.subplots(1, min(len(available_geo), 4), figsize=(5*min(len(available_geo), 4), 4))
        if len(available_geo) == 1:
            axes = [axes]
        for idx, col in enumerate(available_geo[:4]):
            axes[idx].hist(df[col].dropna(), bins=20, color='seagreen', edgecolor='white')
            axes[idx].set_title(col, fontsize=10)
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Count')
        fig.suptitle(f'{dataset_name} - SHI Geometric Features',
                     fontsize=13, fontweight='bold')
        save_and_close(fig, os.path.join(out_dir, '01_geometric_features.png'))

    # 2. 分形维数分布
    fd_cols = [c for c in df.columns if 'fractal' in c.lower() or '_fd' in c.lower()]
    if fd_cols:
        plot_distribution_with_stats(
            df[fd_cols[0]].dropna(),
            f'{dataset_name} - Fractal Dimension Distribution',
            'Fractal Dimension', os.path.join(out_dir, '02_fractal_dimension.png'),
            color='darkorange'
        )

    # 3. 空隙度分布
    lac_cols = [c for c in df.columns if 'lacunar' in c.lower() or 'lac' in c.lower()]
    if lac_cols:
        plot_distribution_with_stats(
            df[lac_cols[0]].dropna(),
            f'{dataset_name} - Lacunarity Distribution',
            'Lacunarity', os.path.join(out_dir, '03_lacunarity.png'),
            color='indianred'
        )

    # 4. 分形维数vs空隙度
    if fd_cols and lac_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df[fd_cols[0]], df[lac_cols[0]], alpha=0.6, c='darkviolet', edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Fractal Dimension', fontsize=11)
        ax.set_ylabel('Lacunarity', fontsize=11)
        ax.set_title(f'{dataset_name} - Fractal Dimension vs Lacunarity', fontsize=13, fontweight='bold')
        save_and_close(fig, os.path.join(out_dir, '04_fd_vs_lacunarity.png'))

    # 5. 形状特征雷达图
    radar_features = ['sphericity', 'solidity', 'extent', 'compactness']
    available_radar = [c for c in df.columns if any(r in c.lower() for r in radar_features)]
    if len(available_radar) >= 3:
        mean_vals = [df[c].mean() for c in available_radar]
        max_vals = [df[c].max() for c in available_radar]
        norm_vals = [m / (mx + 1e-10) for m, mx in zip(mean_vals, max_vals)]
        plot_radar_chart(
            available_radar, norm_vals,
            f'{dataset_name} - SHI Shape Profile (Normalized)',
            os.path.join(out_dir, '05_shape_radar.png')
        )

    # 6. 新增: 几何特征小提琴图
    if available_geo:
        for col in available_geo[:3]:
            plot_violin_box(
                df[col].dropna().values,
                f'{dataset_name} - {col} Violin Plot',
                col, os.path.join(out_dir, f'06_violin_{col}.png'),
                color='seagreen'
            )

    # 7. 新增: FD vs Surface FD 联合分布
    fd_surf_cols = [c for c in df.columns if 'fd_surface' in c.lower()]
    if fd_cols and fd_surf_cols:
        plot_joint_distribution(
            df, fd_cols[0], fd_surf_cols[0],
            f'{dataset_name} - FD vs Surface FD Joint Distribution',
            os.path.join(out_dir, '07_fd_surface_joint.png'),
            color='darkorange'
        )

    # 8. 新增: 几何特征ECDF
    if available_geo:
        for col in available_geo[:2]:
            plot_ecdf(
                df[col].dropna().values,
                f'{dataset_name} - {col} ECDF',
                col, os.path.join(out_dir, f'08_ecdf_{col}.png'),
                color='seagreen'
            )

    # 9. 新增: 形状特征KDE对比
    if len(available_geo) >= 2:
        plot_kde(
            [df[c].dropna().values for c in available_geo[:4]],
            available_geo[:4],
            f'{dataset_name} - Shape Feature KDE Comparison',
            'Value',
            os.path.join(out_dir, '09_shape_kde.png')
        )

    # 10. 新增: 形状特征层次聚类热图
    if len(available_geo) >= 3:
        try:
            plot_clustermap(
                df, available_geo,
                f'{dataset_name} - Shape Feature Cluster Map',
                os.path.join(out_dir, '10_shape_clustermap.png'),
                cmap='RdYlGn'
            )
        except Exception:
            pass

    # 11. 新增: 形状特征PCA
    if len(available_geo) >= 3:
        plot_pca_2d(
            df, available_geo,
            f'{dataset_name} - Shape PCA Projection',
            os.path.join(out_dir, '11_shape_pca.png')
        )

    # 12. 新增: 分形特征对散点矩阵
    frac_cols = [c for c in df.columns if any(k in c.lower() for k in ['fd', 'lac', 'fractal'])]
    if len(frac_cols) >= 3:
        plot_pairgrid(
            df, frac_cols[:5],
            f'{dataset_name} - Fractal Feature Pair Grid',
            os.path.join(out_dir, '12_fractal_pairgrid.png')
        )

    # 13. 新增: FD vs Lacunarity 带回归线
    if fd_cols and lac_cols:
        plot_regplot(
            df, fd_cols[0], lac_cols[0],
            f'{dataset_name} - FD vs Lacunarity (Regression)',
            os.path.join(out_dir, '13_fd_lac_regplot.png'),
            color='darkviolet'
        )

    print(f"[SHI] Visualization saved to {out_dir}")


# ============================================================
# 6. THI 可视化
# ============================================================

def visualize_thi_results(csv_path, out_dir, dataset_name, label_map_dict=None):
    """
    THI (Texture Heterogeneity Index) 结果可视化

    包含:
    1. ITHscore分布直方图
    2. 聚类数分布
    3. 各簇体积占比箱线图
    4. 肿瘤体素数vsITHscore散点图
    5. 聚类标签图
    6. 新增: ITHscore 小提琴图+箱线图
    7. 新增: ITHscore QQ图
    8. 新增: ITHscore ECDF
    9. 新增: 聚类数vsITHscore 联合分布
    10. 新增: 各簇占比蜂群图
    11. 新增: 特征KDE对比
    12. 新增: 聚类特征PCA
    """
    df = pd.read_csv(csv_path)
    df = _normalize_column_names(df)
    os.makedirs(out_dir, exist_ok=True)

    # 1. ITHscore分布
    if 'ith_score' in df.columns:
        plot_distribution_with_stats(
            df['ith_score'].dropna(),
            f'{dataset_name} - ITH Score Distribution',
            'ITH Score', os.path.join(out_dir, '01_ith_score_distribution.png'),
            color='crimson'
        )

    # 2. 聚类数分布
    if 'n_clusters' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        cluster_counts = df['n_clusters'].value_counts().sort_index()
        cluster_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_xlabel('Number of Clusters (K)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{dataset_name} - Cluster Number Distribution', fontsize=13, fontweight='bold')
        ax.tick_params(axis='x', rotation=0)
        save_and_close(fig, os.path.join(out_dir, '02_cluster_number_dist.png'))

    # 3. 各簇体积占比
    cluster_vol_cols = [c for c in df.columns if 'cluster' in c.lower() and ('vol' in c.lower() or 'prop' in c.lower())]
    if cluster_vol_cols:
        fig, ax = plt.subplots(figsize=(max(8, len(cluster_vol_cols)*0.6), 5))
        df_melt = df[cluster_vol_cols].melt()
        sns.boxplot(data=df_melt, x='variable', y='value', ax=ax, palette='Set2')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_title(f'{dataset_name} - Cluster Volume Proportion', fontsize=13, fontweight='bold')
        ax.set_xlabel('Cluster', fontsize=11)
        ax.set_ylabel('Proportion', fontsize=11)
        save_and_close(fig, os.path.join(out_dir, '03_cluster_volume_proportion.png'))

    # 4. 肿瘤体素数vsITHscore
    if 'tumor_voxels' in df.columns and 'ith_score' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(df['tumor_voxels'], df['ith_score'], alpha=0.6, c='teal', edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Tumor Volume (voxels)', fontsize=11)
        ax.set_ylabel('ITH Score', fontsize=11)
        ax.set_title(f'{dataset_name} - Tumor Volume vs ITH Score', fontsize=13, fontweight='bold')
        save_and_close(fig, os.path.join(out_dir, '04_volume_vs_ithscore.png'))

    # 5. 聚类标签图
    if label_map_dict:
        fig = plot_multi_slice_comparison(
            label_map_dict, out_path=os.path.join(out_dir, '05_cluster_label_maps.png')
        )

    # 6. 新增: ITHscore 小提琴图+箱线图
    if 'ith_score' in df.columns:
        plot_violin_box(
            df['ith_score'].dropna().values,
            f'{dataset_name} - ITH Score Violin + Box',
            'ITH Score', os.path.join(out_dir, '06_ith_score_violin.png'),
            color='crimson'
        )

    # 7. 新增: ITHscore QQ正态检验
    if 'ith_score' in df.columns:
        plot_qq(
            df['ith_score'].dropna().values,
            f'{dataset_name} - ITH Score Q-Q Normal Plot',
            os.path.join(out_dir, '07_ith_score_qq.png'),
            color='crimson'
        )

    # 8. 新增: ITHscore ECDF
    if 'ith_score' in df.columns:
        plot_ecdf(
            df['ith_score'].dropna().values,
            f'{dataset_name} - ITH Score ECDF',
            'ITH Score', os.path.join(out_dir, '08_ith_score_ecdf.png'),
            color='crimson'
        )

    # 9. 新增: 聚类数vsITHscore 联合分布
    if 'n_clusters' in df.columns and 'ith_score' in df.columns:
        plot_joint_distribution(
            df, 'n_clusters', 'ith_score',
            f'{dataset_name} - Clusters vs ITH Score Joint',
            os.path.join(out_dir, '09_cluster_ith_joint.png'),
            color='steelblue'
        )

    # 10. 新增: 各簇占比蜂群图
    if cluster_vol_cols:
        df_melt = df[cluster_vol_cols].melt()
        df_melt['cluster'] = df_melt['variable']
        plot_swarm_strip(
            df_melt, 'cluster', 'value',
            f'{dataset_name} - Cluster Proportion Swarm',
            os.path.join(out_dir, '10_cluster_swarm.png'),
            palette='Set2'
        )

    # 11. 新增: 聚类占比KDE对比
    if len(cluster_vol_cols) >= 2:
        plot_kde(
            [df[c].dropna().values for c in cluster_vol_cols],
            cluster_vol_cols,
            f'{dataset_name} - Cluster Proportion KDE',
            'Proportion',
            os.path.join(out_dir, '11_cluster_kde.png')
        )

    # 12. 新增: 聚类特征PCA
    if len(cluster_vol_cols) >= 3:
        plot_pca_2d(
            df, cluster_vol_cols,
            f'{dataset_name} - Cluster PCA Projection',
            os.path.join(out_dir, '12_cluster_pca.png')
        )

    # 13. 新增: 肿瘤体积vs ITHscore 回归图
    if 'tumor_voxels' in df.columns and 'ith_score' in df.columns:
        plot_regplot(
            df, 'tumor_voxels', 'ith_score',
            f'{dataset_name} - Tumor Volume vs ITH Score (Regression)',
            os.path.join(out_dir, '13_volume_ith_regplot.png'),
            color='teal'
        )

    print(f"[THI] Visualization saved to {out_dir}")


# ============================================================
# 7. ITH-Fusion 可视化
# ============================================================

def visualize_ith_fusion_results(csv_path, out_dir, dataset_name,
                                  confidence_map=None, label_map=None):
    """
    ITH Fusion Score 结果可视化

    包含:
    1. ITH Fusion Score分布
    2. 空间一致性分布
    3. 基础分vs最终分对比
    4. 置信空间图切片
    5. 聚类标签图切片
    6. 新增: ITH Fusion Score 小提琴图
    7. 新增: ITH Fusion Score QQ图
    8. 新增: 基础分vs最终分 联合分布
    9. 新增: 空间一致性 ECDF
    10. 新增: 特征对散点矩阵
    11. 新增: 多模型评分雷达图（多患者均值）
    12. 新增: 特征KDE对比
    13. 新增: 基础分vs最终分 回归图
    """
    df = pd.read_csv(csv_path)
    df = _normalize_column_names(df)
    os.makedirs(out_dir, exist_ok=True)

    # 1. ITH Fusion Score分布
    if 'ith_fusion_score' in df.columns:
        plot_distribution_with_stats(
            df['ith_fusion_score'].dropna(),
            f'{dataset_name} - ITH Fusion Score Distribution',
            'ITH Fusion Score', os.path.join(out_dir, '01_ith_fusion_score_dist.png'),
            color='darkmagenta'
        )

    # 2. 空间一致性分布
    if 'spatial_coherence' in df.columns:
        plot_distribution_with_stats(
            df['spatial_coherence'].dropna(),
            f'{dataset_name} - Spatial Coherence Distribution',
            'Spatial Coherence', os.path.join(out_dir, '02_spatial_coherence_dist.png'),
            color='forestgreen'
        )

    # 3. 基础分vs最终分
    if 'ith_base' in df.columns and 'ith_fusion_score' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df['ith_base'], df['ith_fusion_score'], alpha=0.6, c='navy', edgecolors='black', linewidth=0.5)
        lim = [min(df['ith_base'].min(), df['ith_fusion_score'].min()),
               max(df['ith_base'].max(), df['ith_fusion_score'].max())]
        ax.plot(lim, lim, 'r--', alpha=0.5, label='y=x')
        ax.set_xlabel('ITH Base Score', fontsize=11)
        ax.set_ylabel('ITH Fusion Score', fontsize=11)
        ax.set_title(f'{dataset_name} - Base Score vs Fusion Score', fontsize=13, fontweight='bold')
        ax.legend()
        save_and_close(fig, os.path.join(out_dir, '03_base_vs_fusion.png'))

    # 4. 置信空间图切片
    if confidence_map is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        z_mid = confidence_map.shape[0] // 2
        slices = [max(0, z_mid - 3), z_mid, min(confidence_map.shape[0]-1, z_mid + 3)]
        for idx, sl in enumerate(slices):
            im = axes[idx].imshow(confidence_map[sl], cmap='RdYlGn', vmin=0, vmax=1)
            axes[idx].set_title(f'Z={sl}', fontsize=11)
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046)

        fig.suptitle(f'{dataset_name} - Heterogeneity Confidence Map (Green=High Conf, Red=Low Conf)',
                     fontsize=13, fontweight='bold')
        save_and_close(fig, os.path.join(out_dir, '04_confidence_map_slices.png'))

    # 5. 聚类标签图切片
    if label_map is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        z_mid = label_map.shape[0] // 2
        slices = [max(0, z_mid - 3), z_mid, min(label_map.shape[0]-1, z_mid + 3)]
        n_labels = len(np.unique(label_map[label_map > 0]))

        for idx, sl in enumerate(slices):
            im = axes[idx].imshow(label_map[sl], cmap='tab10', vmin=0, vmax=max(9, n_labels))
            axes[idx].set_title(f'Z={sl}', fontsize=11)
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046, ticks=range(1, n_labels+1))

        fig.suptitle(f'{dataset_name} - Cluster Label Map', fontsize=13, fontweight='bold')
        save_and_close(fig, os.path.join(out_dir, '05_cluster_label_slices.png'))

    # 6. 新增: ITH Fusion Score 小提琴图
    if 'ith_fusion_score' in df.columns:
        plot_violin_box(
            df['ith_fusion_score'].dropna().values,
            f'{dataset_name} - ITH Fusion Score Violin',
            'ITH Fusion Score', os.path.join(out_dir, '06_fusion_violin.png'),
            color='darkmagenta'
        )

    # 7. 新增: ITH Fusion Score QQ图
    if 'ith_fusion_score' in df.columns:
        plot_qq(
            df['ith_fusion_score'].dropna().values,
            f'{dataset_name} - ITH Fusion Score Q-Q',
            os.path.join(out_dir, '07_fusion_qq.png'),
            color='darkmagenta'
        )

    # 8. 新增: 基础分vs最终分 联合分布
    if 'ith_base' in df.columns and 'ith_fusion_score' in df.columns:
        plot_joint_distribution(
            df, 'ith_base', 'ith_fusion_score',
            f'{dataset_name} - Base vs Fusion Joint',
            os.path.join(out_dir, '08_base_fusion_joint.png'),
            color='navy'
        )

    # 9. 新增: 空间一致性 ECDF
    if 'spatial_coherence' in df.columns:
        plot_ecdf(
            df['spatial_coherence'].dropna().values,
            f'{dataset_name} - Spatial Coherence ECDF',
            'Spatial Coherence', os.path.join(out_dir, '09_spatial_ecdf.png'),
            color='forestgreen'
        )

    # 10. 新增: 特征对散点矩阵
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 3:
        plot_pairgrid(
            df, numeric_cols[:5],
            f'{dataset_name} - Fusion Feature Pair Grid',
            os.path.join(out_dir, '10_fusion_pairgrid.png')
        )

    # 11. 新增: 基础分vs最终分 回归图
    if 'ith_base' in df.columns and 'ith_fusion_score' in df.columns:
        plot_regplot(
            df, 'ith_base', 'ith_fusion_score',
            f'{dataset_name} - Base vs Fusion (Regression)',
            os.path.join(out_dir, '11_base_fusion_regplot.png'),
            color='navy', add_identity=True
        )

    # 12. 新增: 特征KDE对比
    if len(numeric_cols) >= 2:
        plot_kde(
            [df[c].dropna().values for c in numeric_cols[:4]],
            numeric_cols[:4],
            f'{dataset_name} - Fusion Feature KDE',
            'Value',
            os.path.join(out_dir, '12_fusion_kde.png')
        )

    # 13. 新增: PCA降维
    if len(numeric_cols) >= 4:
        plot_pca_2d(
            df, numeric_cols[:10],
            f'{dataset_name} - Fusion PCA Projection',
            os.path.join(out_dir, '13_fusion_pca.png')
        )

    print(f"[ITH-Fusion] Visualization saved to {out_dir}")


# ============================================================
# 统一入口
# ============================================================

def _normalize_column_names(df):
    """规范化列名: 移除 feat_ 和 detail_ 前缀，使可视化代码适配实际CSV格式。"""
    rename = {}
    for col in df.columns:
        if col.startswith('feat_'):
            rename[col] = col[5:]
        elif col.startswith('detail_'):
            rename[col] = col[7:]
        else:
            rename[col] = col
    return df.rename(columns=rename)


def visualize_model_results(model_name, csv_path, out_dir, dataset_name, **kwargs):
    """
    统一可视化入口

    Args:
        model_name: 模型名称 ('bih', 'dhi', 'hab', 'pth', 'shi', 'thi', 'ith_fusion')
        csv_path: 结果CSV路径
        out_dir: 输出目录
        dataset_name: 数据集名称
        **kwargs: 额外参数（如label_map, confidence_map等）
    """
    visualizers = {
        'bih': visualize_bih_results,
        'dhi': visualize_dhi_results,
        'hab': visualize_hab_results,
        'pth': visualize_pth_results,
        'shi': visualize_shi_results,
        'thi': visualize_thi_results,
        'ith_fusion': visualize_ith_fusion_results,
    }

    if model_name.lower() not in visualizers:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(visualizers.keys())}")

    if not os.path.exists(csv_path):
        print(f"[WARN] CSV not found: {csv_path}, skipping visualization.")
        return

    # 读取CSV并规范化列名
    df = pd.read_csv(csv_path)
    df = _normalize_column_names(df)

    # 保存规范化后的临时CSV供子函数使用
    visualizers[model_name.lower()](csv_path, out_dir, dataset_name, **kwargs)
