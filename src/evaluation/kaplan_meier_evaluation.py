import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import seaborn as sns  # 确保你安装了 seaborn

# 设置绘图样式
plt.rcParams['pdf.fonttype'] = 42      # TrueType，文本可编辑
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")  # 设置 seaborn 风格

for dataset in ["NSCLC-Radiomics", "NSCLC-Radiogenomics", "LungCT-Diagnosis", "CPTAC-TCGA"]:
    for feature_type in ["hab", "fd", "pth"]:
        # 文件路径
        risk_score_file_path = f'./score/{dataset}_{feature_type}_risk_scores.xlsx'  # 根据您的实际文件名修改
        pdf_file_path = f'./figure/{dataset}_{feature_type}_km_curve_results.pdf'  # 保存 KM 曲线的 PDF 文件

        # 1. 读取评分结果表格
        risk_scores_df = pd.read_excel(risk_score_file_path)  # 或使用 pd.read_csv()
        print(risk_scores_df.head())  # 打印前几行检查数据

        # 2. 计算中位数并分组
        median_risk_score = risk_scores_df['risk_score'].median()
        print(f"Median Risk Score: {median_risk_score}")

        risk_scores_df['group'] = np.where(risk_scores_df['risk_score'] > median_risk_score, 'High', 'Low')

        # 提取生存数据
        event_col = 'event'  # 根据实际情况修改事件列名
        time_col = 'time'    # 根据实际情况修改时间列名
        T = risk_scores_df[time_col]  # 生存时间
        E = risk_scores_df[event_col]  # 事件状态

        # 3. 计算组的平均风险评分和标准差
        high_scores = risk_scores_df[risk_scores_df['group'] == 'High']['risk_score']
        low_scores = risk_scores_df[risk_scores_df['group'] == 'Low']['risk_score']
        n_high = len(high_scores)
        n_low = len(low_scores)
        high_mean = high_scores.mean()
        high_std = high_scores.std()
        low_mean = low_scores.mean()
        low_std = low_scores.std()

        # 4. 拟合Cox比例风险模型
        cph = CoxPHFitter()
        cph.fit(risk_scores_df[[event_col, time_col, 'risk_score']], duration_col=time_col, event_col=event_col)

        # 提取HR和p值
        hr = cph.summary['exp(coef)'].values[0]  # HR for risk_score
        p_value = cph.summary['p'].values[0]  # p-value for risk_score

        # 计算95%可信区间
        ci_lower = np.exp(cph.summary['coef'].values[0] - 1.96 * cph.summary['se(coef)'].values[0])
        ci_upper = np.exp(cph.summary['coef'].values[0] + 1.96 * cph.summary['se(coef)'].values[0])

        # 使用Log-rank检验
        results = logrank_test(T[risk_scores_df['group'] == 'High'], T[risk_scores_df['group'] == 'Low'], 
                                E[risk_scores_df['group'] == 'High'], E[risk_scores_df['group'] == 'Low'])
        logrank_p_value = results.p_value

        # 5. 生成 KM 曲线
        kmf = KaplanMeierFitter()

        # 创建 PDF 文件
        with PdfPages(pdf_file_path) as pdf:

            # 创建图形
            plt.figure(figsize=(8, 6))

            # 按组绘制 KM 曲线
            kmf.fit(T[risk_scores_df['group'] == 'Low'], E[risk_scores_df['group'] == 'Low'])
            kmf.plot_survival_function(label='Low Risk', color='blue', ci_show=False, show_censors=True)

            kmf.fit(T[risk_scores_df['group'] == 'High'], E[risk_scores_df['group'] == 'High'])
            kmf.plot_survival_function(label='High Risk', color='red', ci_show=False, show_censors=True)

            # 图形设置
            plt.title('Kaplan-Meier Survival Curve', fontsize=18, fontweight='bold')
            plt.xlabel('Time', fontsize=16, fontweight='bold')
            plt.ylabel('Survival Probability', fontsize=16, fontweight='bold')
            plt.grid(False)
            plt.legend()
            # 加粗外框
            for spine in plt.gca().spines.values():
                spine.set_linewidth(1.5)  # 设置边框宽度
                spine.set_color('black')  # 设置边框颜色

            # 添加文本内容到右上角
            plt.text(0.95, 0.95, f'Low Risk (n={n_low}, score={low_mean:.2f}±{low_std:.2f})',
                    fontsize=12, ha='right', va='top',
                    transform=plt.gca().transAxes)
            plt.text(0.95, 0.90, f'High Risk (n={n_high}, score={high_mean:.2f}±{high_std:.2f})',
                    fontsize=12, ha='right', va='top',
                    transform=plt.gca().transAxes)
            plt.text(0.95, 0.85, f'HR: {hr:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f}) | logrank p: {logrank_p_value:.4f}',
                    fontsize=12, ha='right', va='top',
                    transform=plt.gca().transAxes)

            # 保存 KM 曲线到 PDF
            pdf.savefig()  # 保存当前图形到 PDF 页
            plt.close()   # 关闭图形

        print(f"KM curve saved as editable PDF: {pdf_file_path}.")