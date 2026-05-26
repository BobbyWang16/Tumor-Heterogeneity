"""
生成跨模型汇总对比报告

包含:
1. 各模型评分分布对比
2. 模型间相关性热图
3. 综合评分排行榜
4. 多模型结果对比散点图矩阵
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import save_and_close


def generate_summary_report(dataset_name, results_dir, out_dir):
    """
    生成数据集级别的汇总报告

    Args:
        dataset_name: 数据集名称
        results_dir: 各模型结果所在的根目录 (如 ./results/NSCLC-Radiomics)
        out_dir: 报告输出目录
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Generating summary report for {dataset_name}")
    print(f"{'='*60}")

    # 收集各模型的评分
    model_scores = {}

    # BIH: fractal dimension
    bih_csv = os.path.join(results_dir, 'bih', f'{dataset_name}_fractal_features.csv')
    if os.path.exists(bih_csv):
        df = pd.read_csv(bih_csv)
        if 'fd' in df.columns:
            model_scores['BIH_FD'] = df['fd']

    # DHI: 用某个关键特征代表
    dhi_csv = os.path.join(results_dir, 'dhi', f'dhi_features_{dataset_name}.csv')
    if os.path.exists(dhi_csv):
        df = pd.read_csv(dhi_csv)
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            # 用第一个数值列的均值作为代表（或CV相关列）
            model_scores['DHI'] = df[num_cols[0]]

    # HAB: JS散度
    hab_csv = os.path.join(results_dir, 'hab', f'HAB_{dataset_name}_features.csv')
    if os.path.exists(hab_csv):
        df = pd.read_csv(hab_csv)
        js_cols = [c for c in df.columns if 'js' in c.lower()]
        if js_cols:
            model_scores['HAB_JS'] = df[js_cols[0]]

    # PTH: 超像素数量或CV
    pth_csv = os.path.join(results_dir, 'pth', f'PTH_{dataset_name}_features.csv')
    if os.path.exists(pth_csv):
        df = pd.read_csv(pth_csv)
        cv_cols = [c for c in df.columns if 'cv' in c.lower()]
        if cv_cols:
            model_scores['PTH_CV'] = df[cv_cols[0]]

    # SHI: 分形维数
    shi_csv = os.path.join(results_dir, 'shi', f'shi_features_{dataset_name}.csv')
    if os.path.exists(shi_csv):
        df = pd.read_csv(shi_csv)
        fd_cols = [c for c in df.columns if 'fractal' in c.lower() or '_fd' in c.lower()]
        if fd_cols:
            model_scores['SHI_FD'] = df[fd_cols[0]]

    # THI: ITHscore
    thi_csv = os.path.join(results_dir, 'thi', f'thi_features_{dataset_name}.csv')
    if os.path.exists(thi_csv):
        df = pd.read_csv(thi_csv)
        if 'ith_score' in df.columns:
            model_scores['THI'] = df['ith_score']

    # ITH-Fusion
    fusion_csv = os.path.join(results_dir, 'ith_fusion', f'ITH_Fusion_{dataset_name}.csv')
    if os.path.exists(fusion_csv):
        df = pd.read_csv(fusion_csv)
        if 'ith_fusion_score' in df.columns:
            model_scores['ITH-FS'] = df['ith_fusion_score']

    if not model_scores:
        print(f"[WARN] No score data found for summary report.")
        return

    # 统一长度（取最小长度）
    min_len = min(len(v) for v in model_scores.values())
    score_df = pd.DataFrame({k: v.values[:min_len] for k, v in model_scores.items()})

    # 1. 各模型评分分布对比
    fig, axes = plt.subplots(1, len(score_df.columns), figsize=(5*len(score_df.columns), 5))
    if len(score_df.columns) == 1:
        axes = [axes]
    for idx, col in enumerate(score_df.columns):
        axes[idx].hist(score_df[col].dropna(), bins=20, color='steelblue', edgecolor='white', alpha=0.8)
        axes[idx].axvline(score_df[col].mean(), color='red', linestyle='--', label=f'Mean={score_df[col].mean():.3f}')
        axes[idx].set_title(col, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Score')
        axes[idx].set_ylabel('Count')
        axes[idx].legend(fontsize=9)
    fig.suptitle(f'{dataset_name} - Score Distribution Comparison', fontsize=14, fontweight='bold')
    save_and_close(fig, os.path.join(out_dir, '01_score_distribution_comparison.png'))

    # 2. 模型间相关性热图
    if len(score_df.columns) >= 2:
        fig, ax = plt.subplots(figsize=(max(8, len(score_df.columns)), max(6, len(score_df.columns)*0.6)))
        corr = score_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, square=True, ax=ax,
                    annot_kws={'size': 10})
        ax.set_title(f'{dataset_name} - Cross-Model Score Correlation', fontsize=13, fontweight='bold')
        save_and_close(fig, os.path.join(out_dir, '02_cross_model_correlation.png'))

    # 3. 配对散点图矩阵
    if len(score_df.columns) >= 2 and len(score_df.columns) <= 6:
        fig = plt.figure(figsize=(3*len(score_df.columns), 3*len(score_df.columns)))
        cols = score_df.columns
        n = len(cols)
        for i in range(n):
            for j in range(n):
                ax = fig.add_subplot(n, n, i*n + j + 1)
                if i == j:
                    ax.hist(score_df[cols[i]].dropna(), bins=15, color='steelblue', edgecolor='white')
                    ax.set_title(cols[i], fontsize=10)
                else:
                    ax.scatter(score_df[cols[j]], score_df[cols[i]], alpha=0.5, s=10, edgecolors='black', linewidth=0.3)
                    # 添加回归线
                    valid = score_df[[cols[j], cols[i]]].dropna()
                    if len(valid) > 2:
                        z = np.polyfit(valid[cols[j]], valid[cols[i]], 1)
                        p = np.poly1d(z)
                        ax.plot(valid[cols[j]].sort_values(), p(valid[cols[j]].sort_values()), 'r--', alpha=0.7)
                if i < n-1:
                    ax.set_xticklabels([])
                if j > 0:
                    ax.set_yticklabels([])
        fig.suptitle(f'{dataset_name} - Pairwise Score Comparison', fontsize=14, fontweight='bold')
        save_and_close(fig, os.path.join(out_dir, '03_pairwise_comparison.png'))

    # 4. 箱线图对比
    fig, ax = plt.subplots(figsize=(max(10, len(score_df.columns)*1.2), 6))
    score_df_melt = score_df.melt()
    sns.boxplot(data=score_df_melt, x='variable', y='value', ax=ax, palette='Set2')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=10)
    ax.set_title(f'{dataset_name} - Score Boxplot Comparison', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    save_and_close(fig, os.path.join(out_dir, '04_score_boxplot_comparison.png'))

    # 5. 统计摘要表
    summary = score_df.describe().round(4)
    summary_path = os.path.join(out_dir, '05_statistical_summary.csv')
    summary.to_csv(summary_path)
    print(f"  Statistical summary saved: {summary_path}")

    print(f"[Summary] Report saved to {out_dir}")
