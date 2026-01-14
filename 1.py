
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ==================== 2. 特征预筛选（减少维度） ====================
from sklearn.feature_selection import VarianceThreshold, f_regression, SelectKBest
from sklearn.preprocessing import StandardScaler

# 2.1 移除低方差基因
var_threshold = VarianceThreshold(threshold=0.1)
X_var_filtered = var_threshold.fit_transform(X)
selected_genes_var = X.columns[var_threshold.get_support()]
print(f"After variance filtering: {X_var_filtered.shape[1]} genes")

# 2.2 选择与产量相关性最高的前200个基因
selector_kbest = SelectKBest(score_func=f_regression, k=min(200, X_var_filtered.shape[1]))
X_kbest = selector_kbest.fit_transform(X_var_filtered, y)
selected_indices_kbest = selector_kbest.get_support(indices=True)
selected_genes_kbest = selected_genes_var[selected_indices_kbest]

# 2.3 标准化
scaler = StandardScaler()
X_processed = scaler.fit_transform(X_kbest)
gene_names = selected_genes_kbest.tolist()

print(f"最终特征维度: {X_processed.shape}")

# ==================== 3. 稳定性选择（关键步骤） ====================
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import RepeatedKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 设置重复交叉验证
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

def stability_selection(X, y, gene_names, n_bootstraps=100):
    """稳定性选择：多次抽样评估特征重要性"""
    coef_matrix = np.zeros((n_bootstraps, X.shape[1]))
    
    for i in range(n_bootstraps):
        # 自助采样
        indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
        X_boot = X[indices]
        y_boot = y.iloc[indices]
        
        # 使用弹性网络（结合L1和L2正则化）
        enet = ElasticNetCV(
            l1_ratio=0.5,  # 平衡L1和L2
            cv=5,
            max_iter=10000,
            random_state=i
        )
        
        try:
            enet.fit(X_boot, y_boot)
            coef_matrix[i, :] = enet.coef_
        except:
            continue
    
    # 计算稳定性指标
    stability_scores = (coef_matrix != 0).mean(axis=0)
    mean_coefs = coef_matrix.mean(axis=0)
    std_coefs = coef_matrix.std(axis=0)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'Gene': gene_names,
        'Stability_Score': stability_scores,
        'Mean_Coefficient': mean_coefs,
        'Coefficient_Std': std_coefs,
        'Abs_Mean_Coefficient': np.abs(mean_coefs)
    })
    
    return results_df.sort_values('Stability_Score', ascending=False), coef_matrix

# 执行稳定性选择
stability_results, coef_matrix = stability_selection(X_processed, y, gene_names)

# ==================== 4. 多种机器学习模型比较 ====================
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, RegressorMixin

class EnsembleWeighted(BaseEstimator, RegressorMixin):
    """加权集成模型"""
    def __init__(self):
        self.models = {
            'ElasticNet': ElasticNetCV(cv=5, max_iter=10000, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=42)
        }
        self.weights = None
        
    def fit(self, X, y):
        # 训练所有模型并计算权重
        cv_scores = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            cv_scores[name] = scores.mean()
            self.models[name].fit(X, y)
        
        # 基于CV分数设置权重
        total = sum(cv_scores.values())
        self.weights = {name: score/total for name, score in cv_scores.items()}
        return self
    
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for name, model in self.models.items():
            predictions += self.weights[name] * model.predict(X)
        return predictions

# 训练集成模型
ensemble = EnsembleWeighted()
ensemble.fit(X_processed, y)

# 获取特征重要性（从不同模型）
feature_importance = pd.DataFrame({'Gene': gene_names})

# ElasticNet特征重要性
enet = ensemble.models['ElasticNet']
feature_importance['ElasticNet_Coef'] = np.abs(enet.coef_)

# 随机森林特征重要性
rf = ensemble.models['RandomForest']
feature_importance['RF_Importance'] = rf.feature_importances_

# 梯度提升特征重要性
gbr = ensemble.models['GradientBoosting']
feature_importance['GBR_Importance'] = gbr.feature_importances_

# 组合特征重要性分数
feature_importance['Combined_Importance'] = (
    feature_importance['ElasticNet_Coef'].rank() * 0.4 +
    feature_importance['RF_Importance'].rank() * 0.3 +
    feature_importance['GBR_Importance'].rank() * 0.3
)

# ==================== 5. 识别提高产量的基因 ====================
def identify_yield_enhancing_genes(X, y, gene_names, stability_results, feature_importance):
    """识别可能提高产量的基因"""
    
    # 合并所有信息
    results = pd.merge(
        stability_results,
        feature_importance[['Gene', 'Combined_Importance']],
        on='Gene'
    )
    
    # 筛选条件：稳定性高、系数为正、重要性高
    stable_genes = results[
        (results['Stability_Score'] > 0.7) &  # 稳定性阈值
        (results['Mean_Coefficient'] > 0) &    # 正系数（提高产量）
        (results['Combined_Importance'] > results['Combined_Importance'].quantile(0.75))  # 重要性前25%
    ]
    
    # 排序
    stable_genes = stable_genes.sort_values(
        by=['Stability_Score', 'Mean_Coefficient', 'Combined_Importance'],
        ascending=[False, False, False]
    )
    
    return stable_genes

# 执行识别
enhancing_genes = identify_yield_enhancing_genes(
    X_processed, y, gene_names, 
    stability_results, feature_importance
)

print(f"\n识别出的可能提高产量的基因数量: {len(enhancing_genes)}")
print("\nTop 10 可能提高产量的基因:")
print(enhancing_genes[['Gene', 'Stability_Score', 'Mean_Coefficient', 'Combined_Importance']].head(10))

# ==================== 6. 可视化和验证 ====================
def plot_results(enhancing_genes, coef_matrix, gene_names):
    """绘制分析结果"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 稳定性分数分布
    ax1 = axes[0, 0]
    ax1.hist(stability_results['Stability_Score'], bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(x=0.7, color='r', linestyle='--', label='Threshold (0.7)')
    ax1.set_xlabel('Stability Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Stability Scores')
    ax1.legend()
    
    # 2. 系数分布
    ax2 = axes[0, 1]
    ax2.hist(stability_results['Mean_Coefficient'], bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_xlabel('Mean Coefficient')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Coefficients')
    
    # 3. 重要基因的系数稳定性
    ax3 = axes[1, 0]
    top_genes = enhancing_genes.head(10)['Gene'].tolist()
    top_indices = [list(gene_names).index(gene) for gene in top_genes]
    
    for idx in top_indices[:5]:  # 只显示前5个避免混乱
        ax3.plot(coef_matrix[:, idx], alpha=0.5, label=gene_names[idx])
    ax3.set_xlabel('Bootstrap Iteration')
    ax3.set_ylabel('Coefficient Value')
    ax3.set_title('Coefficient Stability Across Bootstraps')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. 特征重要性对比
    ax4 = axes[1, 1]
    top_genes_info = enhancing_genes.head(10)
    x_pos = np.arange(len(top_genes_info))
    ax4.bar(x_pos - 0.2, top_genes_info['Stability_Score'], width=0.4, label='Stability Score')
    ax4.bar(x_pos + 0.2, top_genes_info['Mean_Coefficient'] / top_genes_info['Mean_Coefficient'].max(), 
            width=0.4, label='Norm. Coefficient')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(top_genes_info['Gene'], rotation=45, ha='right')
    ax4.set_ylabel('Score')
    ax4.set_title('Top Genes: Stability vs Coefficient')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

# 绘制图形
plot_results(enhancing_genes, coef_matrix, gene_names)

# ==================== 7. 生成最终报告 ====================
def generate_report(enhancing_genes, ensemble, X_processed, y):
    """生成最终分析报告"""
    
    print("=" * 80)
    print("麦角硫因产量基因分析报告")
    print("=" * 80)
    
    # 模型性能评估
    y_pred = ensemble.predict(X_processed)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"\n模型性能:")
    print(f"- R²分数: {r2:.4f}")
    print(f"- RMSE: {rmse:.4f}")
    
    print(f"\n筛选出的候选基因总数: {len(enhancing_genes)}")
    
    print("\n建议的基因优先级排序:")
    for i, (_, row) in enumerate(enhancing_genes.head(15).iterrows(), 1):
        print(f"{i}. {row['Gene']}: "
              f"稳定性={row['Stability_Score']:.3f}, "
              f"系数={row['Mean_Coefficient']:.4f}±{row['Coefficient_Std']:.4f}")
    
    print("\n实验验证建议:")
    print("1. 优先验证稳定性分数>0.8且系数较大的基因")
    print("2. 考虑基因之间的相互作用（通路分析）")
    print("3. 建议使用CRISPRi/a进行单基因验证")
    print("4. 考虑组合过表达Top 3-5个基因")
    
    # 保存结果
    enhancing_genes.to_csv('yield_enhancing_genes.csv', index=False)
    print("\n结果已保存到: yield_enhancing_genes.csv")
    
    return enhancing_genes

# 生成报告
final_results = generate_report(enhancing_genes, ensemble, X_processed, y)

# ==================== 8. 可选：通路富集分析（需要额外包） ====================
# 如果安装有gseapy，可以进行通路富集分析
try:
    import gseapy as gp
    
    # 准备基因列表（按重要性排序）
    gene_rank = final_results.set_index('Gene')['Combined_Importance'].to_dict()
    
    # 进行富集分析（示例使用KEGG通路）
    # enr = gp.enrichr(
    #     gene_list=list(gene_rank.keys())[:100],  # 使用前100个重要基因
    #     gene_sets=['KEGG_2019_Human'],
    #     organism='human',  # 根据实际生物修改
    #     cutoff=0.05
    # )
    # print("\n通路富集分析结果:")
    # print(enr.results.head())
    
except ImportError:
    print("\n注意: 如需通路富集分析，请安装 gseapy: pip install gseapy")