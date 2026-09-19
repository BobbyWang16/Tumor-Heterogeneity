"""
ITH Visualization Module

为各评分模型提供结果可视化功能。
"""

from .model_visualizers import (
    visualize_bih_results,
    visualize_dhi_results,
    visualize_hab_results,
    visualize_pth_results,
    visualize_shi_results,
    visualize_thi_results,
    visualize_ith_fusion_results,
    visualize_model_results,
)

from .utils import (
    setup_figure,
    save_and_close,
    plot_distribution_with_stats,
    plot_correlation_heatmap,
    plot_radar_chart,
    overlay_mask_on_slice,
    plot_multi_slice_comparison,
    plot_violin_box,
    plot_kde,
    plot_swarm_strip,
    plot_boxen,
    plot_regplot,
    plot_joint_distribution,
    plot_ecdf,
    plot_qq,
    plot_pairgrid,
    plot_pca_2d,
    plot_annotated_boxplot,
    plot_feature_importance,
    plot_multi_slice_grid,
    plot_3d_scatter_projection,
    plot_surface_projection,
    plot_clustermap,
    plot_heatmap_with_dendrogram,
)

from .summary_report import generate_summary_report
from .algorithm_diagnostics import (
    collect_score_tables,
    build_score_matrix,
    summarize_score_quality,
    robust_zscore,
    plot_score_quality_dashboard,
    plot_score_distributions,
    plot_model_agreement,
    plot_robust_score_heatmap,
    plot_rank_profiles,
    plot_spatial_model_comparison,
    generate_algorithm_diagnostics,
)

__all__ = [
    'visualize_bih_results',
    'visualize_dhi_results',
    'visualize_hab_results',
    'visualize_pth_results',
    'visualize_shi_results',
    'visualize_thi_results',
    'visualize_ith_fusion_results',
    'visualize_model_results',
    'generate_summary_report',
    'collect_score_tables', 'build_score_matrix',
    'summarize_score_quality', 'robust_zscore',
    'plot_score_quality_dashboard', 'plot_score_distributions',
    'plot_model_agreement', 'plot_robust_score_heatmap',
    'plot_rank_profiles', 'plot_spatial_model_comparison',
    'generate_algorithm_diagnostics',
    # utils
    'setup_figure', 'save_and_close',
    'plot_distribution_with_stats', 'plot_correlation_heatmap',
    'plot_radar_chart', 'overlay_mask_on_slice',
    'plot_multi_slice_comparison',
    'plot_violin_box', 'plot_kde', 'plot_swarm_strip',
    'plot_boxen', 'plot_regplot', 'plot_joint_distribution',
    'plot_ecdf', 'plot_qq', 'plot_pairgrid', 'plot_pca_2d',
    'plot_annotated_boxplot', 'plot_feature_importance',
    'plot_multi_slice_grid', 'plot_3d_scatter_projection',
    'plot_surface_projection', 'plot_clustermap',
    'plot_heatmap_with_dendrogram',
]
