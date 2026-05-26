"""
ITH Risk Score Construction using LASSO-Cox.

Builds prognostic risk scores from significant univariate Cox features.
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression


def build_risk_score(
    significant_features_path: str,
    feature_type: str,
    data_dir: str = "./feature",
    output_dir: str = "./score",
    p_cutoff: float = 0.1,
    event_col: str = "event",
    time_col: str = "time",
    dataset_name: str = "NSCLC-Radiomics",
) -> Optional[pd.DataFrame]:
    """
    Build a risk score from significant univariate Cox features using LASSO.

    Args:
        significant_features_path: Path to single Cox results Excel file
        feature_type: Feature category (e.g., "hab", "fd", "pth")
        data_dir: Directory containing feature Excel files
        output_dir: Directory to save risk scores
        p_cutoff: P-value threshold for feature selection
        event_col: Event column name in clinical data
        time_col: Time column name in clinical data
        dataset_name: Dataset name prefix for feature file

    Returns:
        DataFrame with risk scores, or None if no significant features found
    """
    # Load significant features
    sig_df = pd.read_excel(significant_features_path)
    sig_filtered = sig_df[sig_df["Feature Type"] == feature_type]
    significant_features = sig_filtered[sig_filtered["p-value"] < p_cutoff]["Feature"].tolist()

    if not significant_features:
        print(f"[{feature_type}] No significant features found (p < {p_cutoff})")
        return None

    # Load dataset features
    data_path = Path(data_dir) / f"{dataset_name}_{feature_type}.xlsx"
    if not data_path.exists():
        print(f"[{feature_type}] Feature file not found: {data_path}")
        return None

    data = pd.read_excel(data_path, index_col=0)

    # Extract survival data
    survival_data = data[[time_col, event_col]]
    feature_data = data[significant_features].join(survival_data)

    # Prepare data
    X = feature_data[significant_features]
    y = survival_data[event_col]

    # Train LASSO model
    lasso_model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)
    lasso_model.fit(X, y)

    # Extract non-zero coefficients
    coef = lasso_model.coef_[0]
    feature_scores = {
        feature: coef[i]
        for i, feature in enumerate(significant_features)
        if abs(coef[i]) > 1e-10
    }

    if not feature_scores:
        print(f"[{feature_type}] All LASSO coefficients are zero")
        return None

    # Print selected features
    for feature, score in feature_scores.items():
        print(f"  {feature}: {score:.4f}")

    # Compute risk scores
    feature_data["risk_score"] = X.dot(coef)
    final_scores = feature_data[[time_col, event_col, "risk_score"]].copy()

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    risk_score_path = out_dir / f"{feature_type}_risk_scores.xlsx"
    final_scores.to_excel(risk_score_path, index=False)
    print(f"[{feature_type}] Risk scores saved to {risk_score_path}")

    # Save formula
    formula_dir = out_dir / "formulas"
    formula_dir.mkdir(parents=True, exist_ok=True)
    formula_path = formula_dir / f"{feature_type}_risk_score_formula.txt"
    with open(formula_path, "w") as f:
        f.write("Risk Score Formula:\n")
        f.write("risk_score = ")
        terms = [
            f"{round(coef[i], 4)} * {feature}"
            for i, feature in enumerate(significant_features)
        ]
        f.write(" + ".join(terms))
        f.write("\n")
    print(f"[{feature_type}] Formula saved to {formula_path}")

    return final_scores


def main():
    """CLI entry point for backward compatibility."""
    significant_features_path = "./single_cox/single_cox_all_results.xlsx"
    data_dir = "./feature"
    output_dir = "./score"
    p_cutoff = 0.1

    for feature_type in ["hab", "fd", "pth"]:
        print(f"\n{'='*50}")
        print(f"Building risk score for: {feature_type}")
        print(f"{'='*50}")
        build_risk_score(
            significant_features_path=significant_features_path,
            feature_type=feature_type,
            data_dir=data_dir,
            output_dir=output_dir,
            p_cutoff=p_cutoff,
        )


if __name__ == "__main__":
    main()
