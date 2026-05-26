import pandas as pd
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.linear_model import IPCRidge
from sksurv.svm import NaiveSurvivalSVM
from sksurv.tree import SurvivalTree, ExtraSurvivalTree
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

from sksurv.metrics import concordance_index_censored, brier_score, integrated_brier_score, cumulative_dynamic_auc
from sklearn.model_selection import train_test_split
from sksurv.nonparametric import kaplan_meier_estimator  
from sksurv.functions import StepFunction  

def get_test_survival_functions(train_label, test_risks):  
    """  
    使用训练集标签估计基线生存函数，结合模型预测风险分数计算测试集样本的生存函数（StepFunction格式）。  
    
    参数:  
    - train_label: 训练集标签，结构化数组，包含生存事件bool和时间。  
    - test_feature: 测试集特征，模型输入。  
    - model: 训练好的生存模型，必须有 predict 方法返回风险分数。  
    
    返回:  
    - test_surv_funcs: 测试集中每个样本对应的生存函数列表（StepFunction对象）  
    """  
    # 训练集事件与时间  
    event_observed = train_label['event']  
    event_time = train_label['time']  
    
    # 估计基线生存函数  
    times, surv_probs = kaplan_meier_estimator(event_observed, event_time)  
    
    # 构造测试集每个样本的StepFunction生存函数  
    test_surv_funcs = []  
    for rs in test_risks:  
        indiv_surv = surv_probs ** np.exp(rs)  
        sf = StepFunction(times, indiv_surv)  
        test_surv_funcs.append(sf)  
    
    return test_surv_funcs

# 数据集名称
datasets = ["NSCLC-Radiomics", "NSCLC-Radiogenomics", "LungCT-Diagnosis", "CPTAC-TCGA"]

# 初始化结果列表
results = []

# 加载训练集
train_data = pd.read_excel(f'./data/{datasets[0]}_risk_scores.xlsx')  
train_data = train_data.astype({"event": 'bool'})  # 确保 event 列为布尔类型
train_data = train_data[['time', 'event', 'hab', 'fd', 'pth']]  # 选择相关列

# 划分训练集和验证集
X_train = train_data[['hab', 'fd', 'pth']]
y_train = np.array(list(zip(train_data['event'], train_data['time'])), dtype=[('event', bool), ('time', float)])

X_train_set, X_val_set, y_train_set, y_val_set = train_test_split(X_train, y_train, test_size=0.3, random_state = 27)

# 定义时间预测点
times = np.arange(365, 1096, 10)
time_pred = 1095
# print("Selected time points for AUC calculation:", times)

# 模型字典
models = {
    "Cox Proportional Hazards": CoxPHSurvivalAnalysis(),
    "Random Survival Forest": RandomSurvivalForest(n_estimators=100),
    "CoxNet": CoxnetSurvivalAnalysis(),
    "SSVM": NaiveSurvivalSVM(),
    # "IPCR": IPCRidge(),
    "Survival Tree": SurvivalTree(),
    "Extra Survival Tree": ExtraSurvivalTree(),
    "Gradient Boosting": GradientBoostingSurvivalAnalysis(),
}

# 对每个模型进行训练和评估
for model_name, model in models.items():
    print(f"Training and evaluating {model_name}...")

    # 训练模型
    model.fit(X_train_set, y_train_set)

    # 使用训练集和验证集进行预测
    y_train_pred = model.predict(X_train_set)
    y_val_pred = model.predict(X_val_set)

    if model_name == "CoxNet" or model_name == "SSVM" or model_name == "IPCR":
        train_surv_funcs = get_test_survival_functions(y_train_set, y_train_pred)
        val_surv_funcs = get_test_survival_functions(y_train_set, y_val_pred)
    else:
        train_surv_funcs = model.predict_survival_function(X_train_set)
        val_surv_funcs = model.predict_survival_function(X_val_set)

    # if model_name != "CoxNet":
    #         exter_surv_funcs = model.predict_survival_function(X_external)
    # elif model_name == "SSVM":
    #     exter_surv_funcs = model._predict_survival_function(X_external)
    # else:
    #     exter_surv_funcs = get_test_survival_functions(y_train_set, y_external_pred)

    # 计算 C-index 和 Brier Score
    c_index_train = concordance_index_censored(y_train_set['event'], y_train_set['time'], y_train_pred)
    c_index_val = concordance_index_censored(y_val_set['event'], y_val_set['time'], y_val_pred)
    brier_score_train = brier_score(y_train_set, y_train_set, y_train_pred, times=[time_pred])
    brier_score_val = brier_score(y_train_set, y_val_set, y_val_pred, times=[time_pred])

    # 计算 Integrated Brier Score
    integrated_brier_train = integrated_brier_score(y_train_set, y_train_set, [fn(times) for fn in train_surv_funcs], times)
    integrated_brier_val = integrated_brier_score(y_train_set, y_val_set, [fn(times) for fn in val_surv_funcs], times)

    # 计算 Cumulative Dynamic AUC
    cumulative_dynamic_auc_train, mean_auc_train = cumulative_dynamic_auc(y_train_set, y_train_set, [-fn(times) for fn in train_surv_funcs], times)
    cumulative_dynamic_auc_val, mean_auc_val = cumulative_dynamic_auc(y_train_set, y_val_set, [-fn(times) for fn in val_surv_funcs], times)

    # 保存结果
    results.append({
        "Model": model_name,
        "Dataset": f"{datasets[0]} (Training)",
        "C-index": c_index_train[0],
        # "Brier Score": brier_score_train,
        "Integrated Brier Score": integrated_brier_train,
        "Cumulative Dynamic AUC": mean_auc_train,
    })
    results.append({
        "Model": model_name,
        "Dataset": f"{datasets[0]} (Internal Validation)",
        "C-index": c_index_val[0],
        # "Brier Score": brier_score_val,
        "Integrated Brier Score": integrated_brier_val,
        "Cumulative Dynamic AUC": mean_auc_val,
    })

    for dataset in datasets[1:]:
        # 加载外部验证集
        external_data = pd.read_excel(f'./data/{dataset}_risk_scores.xlsx') 
        external_data = external_data.astype({"event": 'bool'})  
        external_data = external_data[['time', 'event', 'hab', 'fd', 'pth']]  

        X_external = external_data[['hab', 'fd', 'pth']]
        y_external = np.array(list(zip(external_data['event'], external_data['time'])), dtype=[('event', bool), ('time', float)])

        # 使用训练好的模型进行预测
        y_external_pred = model.predict(X_external)

        if model_name == "SSVM" or model_name == "CoxNet" or model_name == "IPCR":
            exter_surv_funcs = get_test_survival_functions(y_train_set, y_external_pred)   
        else:
            exter_surv_funcs = model.predict_survival_function(X_external)

        # 计算外部验证集的 C-index 和 Brier Score
        c_index_ext = concordance_index_censored(y_external['event'], y_external['time'], y_external_pred)
        brier_score_ext = brier_score(y_external, y_external, y_external_pred, times=[time_pred])
        integrated_brier_ext = integrated_brier_score(y_external, y_external, [fn(times) for fn in exter_surv_funcs], times)
        cumulative_dynamic_auc_ext, mean_auc_ext = cumulative_dynamic_auc(y_external, y_external, [-fn(times) for fn in exter_surv_funcs], times)

        # 保存外部验证集结果
        results.append({
            "Model": model_name,
            "Dataset": f"{dataset} (External Validation)",
            "C-index": c_index_ext[0],
            # "Brier Score": brier_score_ext,
            "Integrated Brier Score": integrated_brier_ext,
            "Cumulative Dynamic AUC": mean_auc_ext
        })

# 将结果保存在 DataFrame 中
results_df = pd.DataFrame(results)

# 保存到 CSV 文件
results_df.to_excel('model_evaluation_results.xlsx', index=True)

print("Results have been saved to 'model_evaluation_results.xlsx'")