import pandas as pd
from sklearn.linear_model import LogisticRegression
import os

# 设置系数绝对值的阈值
coef_cutoff = 0.0001 

# 读取有价值的显著特征
significant_features_df = pd.read_excel('./single_cox/single_cox_all_results.xlsx')
# 设置特征门槛
p_cutoff = 0.1

# 1. 使用 NSCLC-Radiomics 数据集构建风险评分公式
for feature_type in ["hab", "fd", "pth"]:
    dataset = "NSCLC-Radiomics"

    # 读取 NSCLC-Radiomics 数据
    file_path = f'./feature/{dataset}_{feature_type}.xlsx'
    data = pd.read_excel(file_path, index_col=0)

    # 过滤显著特征
    significant_features_df_filtered = significant_features_df[significant_features_df['Feature Type'] == feature_type]
    significant_features = significant_features_df_filtered[significant_features_df_filtered['p-value'] < p_cutoff]['Feature'].tolist()

    # 提取生存数据
    event_col = 'event'  # 根据实际情况修改事件列名
    time_col = 'time'    # 根据实际情况修改时间列名
    survival_data = data[[time_col, event_col]]

    # 过滤数据，仅保留显著特征
    feature_data = data[significant_features].join(survival_data)

    # 拆分特征和生存数据
    X = feature_data[significant_features]
    y = survival_data[event_col]

    # 训练LASSO模型
    lasso_model = LogisticRegression(penalty='l1', solver='liblinear')
    lasso_model.fit(X, y)

    # 获取特征的系数
    coef = lasso_model.coef_[0]

    # 根据系数绝对值过滤特征
    feature_scores = {feature: coef[i] for i, feature in enumerate(significant_features) if abs(coef[i]) > coef_cutoff}

    # 打印公式
    print(f"Formula for {dataset} using feature type '{feature_type}':")
    formula_terms = [f"{round(coef[i], 4)} * {feature}" for i, feature in enumerate(significant_features) if abs(coef[i]) > coef_cutoff]
    print("risk_score = " + " + ".join(formula_terms))

    # 把打印的公式保存为txt文件
    formula_path = f'./score/formulas/{dataset}_{feature_type}_formula.txt'
    with open(formula_path, 'w') as f:
        f.write("risk_score = " + " + ".join(formula_terms))

    # 2. 将评分应用到其他数据集
    for other_dataset in ["NSCLC-Radiomics", "NSCLC-Radiogenomics", "LungCT-Diagnosis", "CPTAC-TCGA"]:
        # 读取其他数据集
        other_file_path = f'./feature/{other_dataset}_{feature_type}.xlsx'
        other_data = pd.read_excel(other_file_path, index_col=0)

        # 提取生存数据
        other_survival_data = other_data[[time_col, event_col]]
        
        # 过滤数据，仅保留显著特征
        other_feature_data = other_data[significant_features].join(other_survival_data)

        # 生成评分
        other_feature_data['risk_score'] = other_feature_data[significant_features].dot(coef)

        # 将评分和生存数据结合
        final_other_scores = other_feature_data[[time_col, event_col]].copy()
        final_other_scores['risk_score'] = other_feature_data['risk_score']

        # 保存结果
        risk_score_path = f'./score/{other_dataset}_{feature_type}_risk_scores.xlsx'
        final_other_scores.to_excel(risk_score_path, index=True)
        print(f"Risk scores for {other_dataset} saved to {risk_score_path}.")