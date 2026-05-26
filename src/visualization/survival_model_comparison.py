import pandas as pd
import matplotlib.pyplot as plt

# 读取结果表格，确保路径正确
metrics_df = pd.read_excel('./model_evaluation_results.xlsx')

# 确保数据正确读取
print(metrics_df.head())

# 设置数据集名称
datasets = metrics_df['Dataset'].unique()

# 定义颜色和标签
colors = ['blue', 'orange', 'green']
labels = ['C-index', '1 - Brier Score', 'Cumulative AUC']

# 为每个数据集绘制图形
for dataset in datasets:
    # 筛选当前的数据集的结果
    df_subset = metrics_df[metrics_df['Dataset'] == dataset]

    # 创建一个图形
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#F0F0F0')  # 设置背景色

    # 定义X轴位置
    indices = range(len(df_subset))

    # 根据指标值设置圆点的大小
    c_index_sizes = df_subset['C-index']**2 * 800  # 使用平方来放大差异
    brier_sizes = (1 - df_subset['Integrated Brier Score'])**2 * 800  # 1 - B-score
    auc_sizes = df_subset['Cumulative Dynamic AUC']**2 * 800  # Cumulative AUC

    # 设置Y轴位置
    y_positions = [0.75, 0.5, 0.25]  # 确保点更靠中间

    # 绘制 C-index
    ax.scatter(indices, [y_positions[0]] * len(df_subset),   
               s=c_index_sizes, alpha=0.7, color=colors[0])  

    # 绘制 1 - Brier Score
    ax.scatter(indices, [y_positions[1]] * len(df_subset),   
               s=brier_sizes, alpha=0.7, color=colors[1])  

    # 绘制 Cumulative AUC
    ax.scatter(indices, [y_positions[2]] * len(df_subset),   
               s=auc_sizes, alpha=0.7, color=colors[2])  

    # 设置图标题和标签
    ax.set_title(f'Model Evaluation Metrics for {dataset}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')

    # 设置X轴标签
    ax.set_xticks(indices)
    ax.set_xticklabels(df_subset['Model'], rotation=45, ha='right', fontweight='bold')

    # 隐藏Y轴
    ax.yaxis.set_visible(False)  # 隐藏Y轴

    # 添加指标名称到Y轴
    for i, label in enumerate(labels):
        ax.text(-0.1, y_positions[i], label, fontsize=16, color=colors[i], va='center', ha='right', transform=ax.transAxes)

    # 可选：添加带透明度的虚线框，确保空间足够
    # margin = 0.2  # 增加边距
    # rectangle = plt.Rectangle((-0.5 + margin, 0), len(df_subset) - margin * 2, 1, 
    #                          color='gray', alpha=0.3, linestyle='--', fill=False)
    # ax.add_patch(rectangle)

    # 保存图形
    plt.savefig(f"{dataset.replace(' ', '_')}_evaluation_metrics.pdf", format='pdf', bbox_inches='tight', dpi=1200)

    # 显示图形
    plt.tight_layout()
    plt.show()