"""
Intraclass Correlation Coefficient (ICC) Analysis for Inter-Observer Variability

用于评估异质性特征是否受不同勾画者（inter-observer）分割差异的影响。

支持的ICC类型：
- ICC(1,1): 单因素随机，单次测量
- ICC(2,1): 双因素随机，单次测量（推荐用于inter-observer）
- ICC(3,1): 双因素混合，单次测量
- ICC(2,k): 双因素随机，k次平均
- ICC(3,k): 双因素混合，k次平均

参考：
Shrout & Fleiss (1979). Intraclass correlations: uses in assessing rater reliability.
Psychological Bulletin, 86(2), 420-428.

数据格式要求：
    每个病例有多个勾画者的掩膜（如 mask_observer1, mask_observer2, mask_observer3）
    ICC分析对每个特征分别计算，判断该特征在不同勾画者间的一致性。
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
#  ICC 核心计算
# ---------------------------------------------------------------------------

@dataclass
class ICCResult:
    """ICC分析结果"""
    feature_name: str
    icc_value: float
    icc_type: str
    ci_lower: float
    ci_upper: float
    f_statistic: float
    p_value: float
    ms_between: float
    ms_within: float
    ms_error: float
    n_subjects: int
    n_observers: int
    reliability: str  # 'Excellent', 'Good', 'Moderate', 'Poor'


def _calculate_icc_from_anova_table(
    ms_between: float,
    ms_within: float,
    ms_error: float,
    n_subjects: int,
    n_observers: int,
    icc_type: str = "ICC(2,1)",
    alpha: float = 0.05,
) -> Dict:
    """
    基于方差分析表计算ICC。

    Args:
        ms_between: 被试间均方 (MS_subject)
        ms_within: 被试内均方 (MS_observer * MS_error)
        ms_error: 误差均方
        n_subjects: 被试（病例）数量
        n_observers: 观察者（勾画者）数量
        icc_type: ICC类型
        alpha: 置信水平

    Returns:
        dict: icc, ci_lower, ci_upper, f, p, ms_between, ms_within, ms_error
    """
    if icc_type == "ICC(1,1)":
        # 单因素随机，单次测量
        # ICC = (MSb - MSw) / (MSb + (k-1)*MSw)
        icc = (ms_between - ms_within) / (ms_between + (n_observers - 1) * ms_within) if (ms_between + (n_observers - 1) * ms_within) > 0 else 0.0
        # F检验
        f = ms_between / ms_within if ms_within > 0 else np.inf
        df1, df2 = n_subjects - 1, n_subjects * (n_observers - 1)

    elif icc_type == "ICC(2,1)":
        # 双因素随机，单次测量（推荐用于inter-observer）
        # ICC = (MSb - MSe) / (MSb + (k-1)*MSe + k*(MSo-MSe)/n)
        # 简化: (MSb - MSe) / (MSb + (k-1)*MSe)
        denom = ms_between + (n_observers - 1) * ms_error
        icc = (ms_between - ms_error) / denom if denom > 0 else 0.0
        f = ms_between / ms_error if ms_error > 0 else np.inf
        df1, df2 = n_subjects - 1, (n_subjects - 1) * (n_observers - 1)

    elif icc_type == "ICC(3,1)":
        # 双因素混合，单次测量
        denom = ms_between + (n_observers - 1) * ms_error
        icc = (ms_between - ms_error) / denom if denom > 0 else 0.0
        f = ms_between / ms_error if ms_error > 0 else np.inf
        df1, df2 = n_subjects - 1, (n_subjects - 1) * (n_observers - 1)

    elif icc_type == "ICC(2,k)":
        # 双因素随机，k次平均
        denom = ms_between + (ms_error - ms_error) / n_observers  # 简化为 MSb
        icc = (ms_between - ms_error) / ms_between if ms_between > 0 else 0.0
        f = ms_between / ms_error if ms_error > 0 else np.inf
        df1, df2 = n_subjects - 1, (n_subjects - 1) * (n_observers - 1)

    elif icc_type == "ICC(3,k)":
        # 双因素混合，k次平均
        icc = (ms_between - ms_error) / ms_between if ms_between > 0 else 0.0
        f = ms_between / ms_error if ms_error > 0 else np.inf
        df1, df2 = n_subjects - 1, (n_subjects - 1) * (n_observers - 1)

    else:
        raise ValueError(f"Unknown ICC type: {icc_type}")

    # 限制ICC在[0,1]范围内
    icc = max(0.0, min(1.0, icc))

    # P值
    p_value = 1.0 - stats.f.cdf(f, df1, df2) if np.isfinite(f) else 0.0

    # 置信区间 (近似方法)
    # 使用Fisher Z变换
    if icc > 0 and icc < 1:
        z = 0.5 * np.log((1 + icc) / (1 - icc))
        se_z = np.sqrt(2.0 / n_subjects)
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        ci_lower_z = z - z_alpha * se_z
        ci_upper_z = z + z_alpha * se_z
        ci_lower = (np.exp(2 * ci_lower_z) - 1) / (np.exp(2 * ci_lower_z) + 1)
        ci_upper = (np.exp(2 * ci_upper_z) - 1) / (np.exp(2 * ci_upper_z) + 1)
    else:
        ci_lower, ci_upper = icc, icc

    return {
        "icc": float(icc),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "f": float(f),
        "p": float(p_value),
        "ms_between": float(ms_between),
        "ms_within": float(ms_within),
        "ms_error": float(ms_error),
        "df1": int(df1),
        "df2": int(df2),
    }


def compute_icc(
    data: pd.DataFrame,
    subject_col: str = "case_id",
    observer_col: str = "observer",
    feature_col: str = "value",
    icc_type: str = "ICC(2,1)",
    alpha: float = 0.05,
) -> ICCResult:
    """
    从长格式DataFrame计算单个特征的ICC。

    Args:
        data: 长格式DataFrame，每行是一个 (subject, observer, value) 观测
        subject_col: 被试列名
        observer_col: 观察者列名
        feature_col: 特征值列名
        icc_type: ICC类型
        alpha: 置信水平

    Returns:
        ICCResult对象
    """
    # 去除缺失值
    df = data[[subject_col, observer_col, feature_col]].dropna()
    if len(df) == 0:
        raise ValueError("No valid data for ICC calculation")

    # 构建观测矩阵 (subjects x observers)
    pivot = df.pivot_table(
        index=subject_col,
        columns=observer_col,
        values=feature_col,
        aggfunc="mean",
    )

    n_subjects = pivot.shape[0]
    n_observers = pivot.shape[1]

    if n_subjects < 2 or n_observers < 2:
        raise ValueError(f"Need at least 2 subjects and 2 observers, got {n_subjects} subjects, {n_observers} observers")

    # 转换为numpy数组
    Y = pivot.values

    # 计算各均值
    grand_mean = np.nanmean(Y)
    subject_means = np.nanmean(Y, axis=1)
    observer_means = np.nanmean(Y, axis=0)

    # 计算平方和
    ss_total = np.nansum((Y - grand_mean) ** 2)
    ss_between_subjects = n_observers * np.nansum((subject_means - grand_mean) ** 2)
    ss_between_observers = n_subjects * np.nansum((observer_means - grand_mean) ** 2)
    ss_residual = ss_total - ss_between_subjects - ss_between_observers

    # 自由度
    df_between_subjects = n_subjects - 1
    df_between_observers = n_observers - 1
    df_residual = (n_subjects - 1) * (n_observers - 1)

    # 均方
    ms_between_subjects = ss_between_subjects / df_between_subjects if df_between_subjects > 0 else 0
    ms_between_observers = ss_between_observers / df_between_observers if df_between_observers > 0 else 0
    ms_residual = ss_residual / df_residual if df_residual > 0 else 0

    # 计算ICC
    result = _calculate_icc_from_anova_table(
        ms_between=ms_between_subjects,
        ms_within=ms_between_observers + ms_residual,  # 近似
        ms_error=ms_residual,
        n_subjects=n_subjects,
        n_observers=n_observers,
        icc_type=icc_type,
        alpha=alpha,
    )

    # 可靠性分级
    icc_val = result["icc"]
    if icc_val >= 0.9:
        reliability = "Excellent"
    elif icc_val >= 0.75:
        reliability = "Good"
    elif icc_val >= 0.5:
        reliability = "Moderate"
    else:
        reliability = "Poor"

    return ICCResult(
        feature_name=feature_col,
        icc_value=icc_val,
        icc_type=icc_type,
        ci_lower=result["ci_lower"],
        ci_upper=result["ci_upper"],
        f_statistic=result["f"],
        p_value=result["p"],
        ms_between=result["ms_between"],
        ms_within=result["ms_within"],
        ms_error=result["ms_error"],
        n_subjects=n_subjects,
        n_observers=n_observers,
        reliability=reliability,
    )


# ---------------------------------------------------------------------------
#  批量ICC分析
# ---------------------------------------------------------------------------

def batch_icc_analysis(
    data: pd.DataFrame,
    feature_cols: List[str],
    subject_col: str = "case_id",
    observer_col: str = "observer",
    icc_type: str = "ICC(2,1)",
    alpha: float = 0.05,
    icc_threshold: float = 0.75,
) -> pd.DataFrame:
    """
    对多个特征批量计算ICC。

    Args:
        data: 宽格式DataFrame，包含 subject_col, observer_col, 和多个特征列
        feature_cols: 要分析的特征列名列表
        subject_col: 被试列名
        observer_col: 观察者列名
        icc_type: ICC类型
        alpha: 置信水平
        icc_threshold: 判定为"可靠"的ICC阈值

    Returns:
        DataFrame，每行一个特征的ICC结果
    """
    results = []

    for feature in feature_cols:
        if feature not in data.columns:
            warnings.warn(f"Feature '{feature}' not found in data, skipping.")
            continue

        try:
            # 构造长格式数据
            long_data = data[[subject_col, observer_col, feature]].copy()
            long_data = long_data.rename(columns={feature: "value"})
            long_data = long_data.dropna()

            if len(long_data) == 0:
                continue

            icc_result = compute_icc(
                long_data,
                subject_col=subject_col,
                observer_col=observer_col,
                feature_col=feature,
                icc_type=icc_type,
                alpha=alpha,
            )

            results.append({
                "feature": feature,
                "icc": icc_result.icc_value,
                "ci_lower": icc_result.ci_lower,
                "ci_upper": icc_result.ci_upper,
                "f_stat": icc_result.f_statistic,
                "p_value": icc_result.p_value,
                "n_subjects": icc_result.n_subjects,
                "n_observers": icc_result.n_observers,
                "reliability": icc_result.reliability,
                "pass": icc_result.icc_value >= icc_threshold,
            })

        except Exception as e:
            warnings.warn(f"ICC calculation failed for '{feature}': {e}")
            results.append({
                "feature": feature,
                "icc": np.nan,
                "error": str(e),
                "pass": False,
            })

    return pd.DataFrame(results)


def summarize_icc_results(results_df: pd.DataFrame, icc_threshold: float = 0.75) -> Dict:
    """
    汇总ICC分析结果。

    Returns:
        dict: 包含各类统计信息
    """
    valid = results_df[results_df["icc"].notna()]

    summary = {
        "total_features": len(results_df),
        "valid_features": len(valid),
        "excellent": int((valid["icc"] >= 0.9).sum()),
        "good": int(((valid["icc"] >= 0.75) & (valid["icc"] < 0.9)).sum()),
        "moderate": int(((valid["icc"] >= 0.5) & (valid["icc"] < 0.75)).sum()),
        "poor": int((valid["icc"] < 0.5).sum()),
        "pass_threshold": int((valid["icc"] >= icc_threshold).sum()),
        "pass_rate": float((valid["icc"] >= icc_threshold).mean()) if len(valid) > 0 else 0.0,
        "mean_icc": float(valid["icc"].mean()) if len(valid) > 0 else np.nan,
        "median_icc": float(valid["icc"].median()) if len(valid) > 0 else np.nan,
        "min_icc": float(valid["icc"].min()) if len(valid) > 0 else np.nan,
        "max_icc": float(valid["icc"].max()) if len(valid) > 0 else np.nan,
    }

    return summary


# ---------------------------------------------------------------------------
#  报告生成
# ---------------------------------------------------------------------------

def generate_icc_report(
    results_df: pd.DataFrame,
    output_path: str,
    icc_threshold: float = 0.75,
    dataset_name: str = "Dataset",
) -> None:
    """
    生成ICC分析报告（CSV + 可选可视化）。

    Args:
        results_df: batch_icc_analysis的输出
        output_path: 输出CSV路径
        icc_threshold: 可靠阈值
        dataset_name: 数据集名称
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存CSV
    results_df.to_csv(out_path, index=False)
    print(f"[ICC] Results saved to {out_path}")

    # 生成可视化
    valid = results_df[results_df["icc"].notna()].copy()
    if len(valid) == 0:
        return

    valid = valid.sort_values("icc", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 图1: ICC柱状图
    colors = ["#2ecc71" if r else "#e74c3c" for r in valid["pass"]]
    axes[0].barh(range(len(valid)), valid["icc"], color=colors, edgecolor="black", linewidth=0.5)
    axes[0].axvline(icc_threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold={icc_threshold}")
    axes[0].axvline(0.5, color="orange", linestyle=":", linewidth=1, alpha=0.7, label="Moderate=0.5")
    axes[0].axvline(0.75, color="blue", linestyle=":", linewidth=1, alpha=0.7, label="Good=0.75")
    axes[0].axvline(0.9, color="green", linestyle=":", linewidth=1, alpha=0.7, label="Excellent=0.9")
    axes[0].set_yticks(range(len(valid)))
    axes[0].set_yticklabels(valid["feature"], fontsize=8)
    axes[0].set_xlabel("ICC Value", fontsize=11)
    axes[0].set_title(f"{dataset_name} - ICC by Feature", fontsize=12, fontweight="bold")
    axes[0].set_xlim(0, 1)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(axis="x", alpha=0.3)

    # 图2: 可靠性分布饼图
    reliability_counts = {
        "Excellent (>=0.9)": (valid["icc"] >= 0.9).sum(),
        "Good (0.75-0.9)": ((valid["icc"] >= 0.75) & (valid["icc"] < 0.9)).sum(),
        "Moderate (0.5-0.75)": ((valid["icc"] >= 0.5) & (valid["icc"] < 0.75)).sum(),
        "Poor (<0.5)": (valid["icc"] < 0.5).sum(),
    }
    colors_pie = ["#27ae60", "#2980b9", "#f39c12", "#c0392b"]
    axes[1].pie(
        reliability_counts.values(),
        labels=reliability_counts.keys(),
        autopct="%1.1f%%",
        colors=colors_pie,
        startangle=90,
    )
    axes[1].set_title(f"{dataset_name} - ICC Reliability Distribution", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig_path = out_path.with_suffix(".png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ICC] Figure saved to {fig_path}")


# ---------------------------------------------------------------------------
#  完整工作流
# ---------------------------------------------------------------------------

def run_icc_workflow(
    data: pd.DataFrame,
    feature_cols: List[str],
    output_dir: str,
    subject_col: str = "case_id",
    observer_col: str = "observer",
    icc_type: str = "ICC(2,1)",
    icc_threshold: float = 0.75,
    dataset_name: str = "Dataset",
) -> Tuple[pd.DataFrame, Dict]:
    """
    完整的ICC分析工作流。

    Args:
        data: 宽格式DataFrame
        feature_cols: 特征列名
        output_dir: 输出目录
        subject_col: 被试列名
        observer_col: 观察者列名
        icc_type: ICC类型
        icc_threshold: 可靠阈值
        dataset_name: 数据集名称

    Returns:
        (results_df, summary_dict)
    """
    print(f"\n{'='*60}")
    print(f"ICC Analysis: {dataset_name}")
    print(f"{'='*60}")
    print(f"  ICC Type: {icc_type}")
    print(f"  Threshold: {icc_threshold}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Subjects: {data[subject_col].nunique()}")
    print(f"  Observers: {data[observer_col].nunique()}")
    print(f"{'='*60}")

    # 批量计算ICC
    results_df = batch_icc_analysis(
        data=data,
        feature_cols=feature_cols,
        subject_col=subject_col,
        observer_col=observer_col,
        icc_type=icc_type,
        icc_threshold=icc_threshold,
    )

    # 汇总
    summary = summarize_icc_results(results_df, icc_threshold)

    # 输出汇总
    print(f"\n--- Summary ---")
    print(f"  Total features: {summary['total_features']}")
    print(f"  Valid features: {summary['valid_features']}")
    print(f"  Excellent (>=0.9): {summary['excellent']}")
    print(f"  Good (0.75-0.9): {summary['good']}")
    print(f"  Moderate (0.5-0.75): {summary['moderate']}")
    print(f"  Poor (<0.5): {summary['poor']}")
    print(f"  Pass threshold (>{icc_threshold}): {summary['pass_threshold']}/{summary['valid_features']} ({summary['pass_rate']*100:.1f}%)")
    print(f"  Mean ICC: {summary['mean_icc']:.3f}")
    print(f"  Median ICC: {summary['median_icc']:.3f}")
    print(f"  Min/Max ICC: {summary['min_icc']:.3f} / {summary['max_icc']:.3f}")

    # 失败的特征
    failed = results_df[results_df["icc"] < icc_threshold]
    if len(failed) > 0:
        print(f"\n--- Failed features (ICC < {icc_threshold}) ---")
        for _, row in failed.iterrows():
            print(f"  {row['feature']}: ICC={row['icc']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")

    # 生成报告
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"ICC_{dataset_name}_{icc_type.replace('(', '').replace(')', '').replace(',', '_')}.csv"
    generate_icc_report(results_df, str(out_csv), icc_threshold, dataset_name)

    # 保存汇总
    summary_df = pd.DataFrame([summary])
    summary_csv = out_dir / f"ICC_{dataset_name}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[ICC] Summary saved to {summary_csv}")

    return results_df, summary
