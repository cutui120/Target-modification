
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from scipy import stats
import joblib

class EfficientDataProcessor:
    """数据处理器"""
    
    def __init__(self):
        self.gene_scaler = StandardScaler()
        self.metabo_scaler = StandardScaler()
        
    def load_and_process(self, transcript_path, metabo_path):
        """加载并预处理数据"""
        print("加载和预处理数据...")
        
        transcriptomics = pd.read_csv(transcript_path, index_col=0)
        metabolomics = pd.read_csv(metabo_path, index_col=0)
        
        # 识别产量数据
        yield_data, metabolomics = self._identify_yield_data(metabolomics)
        
        # 对齐样本
        common_samples = transcriptomics.index.intersection(metabolomics.index)
        if len(common_samples) == 0:
            common_samples = transcriptomics.index[:min(len(transcriptomics), len(metabolomics))]
            metabolomics.index = common_samples
            yield_data.index = common_samples
        
        transcriptomics = transcriptomics.loc[common_samples]
        metabolomics = metabolomics.loc[common_samples]
        yield_data = yield_data.loc[common_samples]
        
        # 预处理
        X_genes = self._process_transcriptomics(transcriptomics)
        X_metabo = self._process_metabolomics(metabolomics)
        
        # 合并特征
        X_combined = pd.concat([X_genes, X_metabo], axis=1)
        
        # 处理NaN
        X_combined = X_combined.fillna(0)
        
        print(f"  样本数: {len(X_combined)}, 特征数: {X_combined.shape[1]}")
        
        return {
            'X': X_combined,
            'y': yield_data,
            'feature_names': X_combined.columns.tolist(),
            'gene_names': X_genes.columns.tolist(),
            'metabo_names': X_metabo.columns.tolist()
        }
    
    def _identify_yield_data(self, metabolomics):
        """识别产量数据"""
        ergo_keywords = ['ergothioneine', 'egt', 'ergothion', 'thioneine']
        
        for col in metabolomics.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ergo_keywords):
                print(f"  找到麦角硫因数据列: {col}")
                yield_data = metabolomics[col].copy()
                metabolomics = metabolomics.drop(col, axis=1)
                return yield_data, metabolomics
        
        # 使用第一主成分作为代理
        print("  未找到麦角硫因数据，使用代谢组第一主成分作为代理...")
        from sklearn.decomposition import PCA
        
        data_clean = metabolomics.fillna(metabolomics.median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data_clean.values)
        
        pca = PCA(n_components=1, random_state=42)
        proxy_yield = pca.fit_transform(X_scaled).flatten()
        
        return pd.Series(proxy_yield, index=metabolomics.index, name='yield_proxy'), metabolomics
    
    def _process_transcriptomics(self, data):
        """预处理转录组数据"""
        # 对数变换
        data = np.log1p(data.clip(lower=0))
        
        # 过滤低表达基因
        if data.shape[1] > 100:
            expr_threshold = np.percentile(data.values.flatten(), 40)
            mask = (data > expr_threshold).sum(axis=0) > data.shape[0] * 0.3
            if mask.sum() > 50:
                data = data.loc[:, mask]
        
        # 标准化
        data_scaled = self.gene_scaler.fit_transform(data)
        
        return pd.DataFrame(data_scaled, columns=data.columns, index=data.index)
    
    def _process_metabolomics(self, data):
        """预处理代谢组数据"""
        data_filled = data.fillna(data.median())
        data_scaled = self.metabo_scaler.fit_transform(data_filled)
        
        return pd.DataFrame(data_scaled, columns=data.columns, index=data.index)


# ==================== 2. 特征选择器 ====================

class FeatureSelector:
    """基于模型的特征选择器"""
    
    def __init__(self, n_features=500):
        self.n_features = n_features
        self.selected_features = None
        self.feature_importance = None
        
    def select_features(self, X, y):
        """选择重要特征"""
        n_to_select = min(self.n_features, X.shape[1])
        print(f"从 {X.shape[1]} 个特征中选择 {n_to_select} 个最重要特征...")
        
        # 使用梯度提升树进行特征选择
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X, y)
        
        # 获取特征重要性
        importances = model.feature_importances_
        
        # 选择最重要的特征
        top_indices = np.argsort(importances)[::-1][:n_to_select]
        self.selected_features = X.columns[top_indices].tolist()
        
        self.feature_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        
        print(f"  选择了 {len(self.selected_features)} 个特征")
        
        return X[self.selected_features]


# ==================== 3. 模型训练器 ====================

class RobustModelTrainer:
    """稳健的模型训练器"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.shap_values = None
        
    def train(self, X, y, test_size=0.2):
        """训练模型"""
        print("训练模型...")
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 简化的网格搜索
        param_grid = {
            'n_estimators': [100, 150],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 4],
            'min_samples_split': [2, 5]
        }
        
        gbr = GradientBoostingRegressor(random_state=42, subsample=0.8)
        grid_search = GridSearchCV(gbr, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        self.model = grid_search.best_estimator_
        print(f"  最佳参数: {grid_search.best_params_}")
        
        # 交叉验证
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=3, scoring='r2')
        print(f"  交叉验证 R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # 测试集评估
        y_pred = self.model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"  测试集 R²: {test_r2:.4f}")
        print(f"  测试集 RMSE: {test_rmse:.4f}")
        
        # 特征重要性
        self.feature_importance = pd.Series(
            self.model.feature_importances_, index=X.columns
        ).sort_values(ascending=False)
        
        # SHAP分析（可选）
        if HAS_SHAP and X_train_scaled.shape[0] < 500:
            try:
                explainer = shap.TreeExplainer(self.model)
                self.shap_values = explainer.shap_values(X_train_scaled[:100])
            except Exception as e:
                print(f"  SHAP计算跳过: {e}")
        
        # 全数据重新训练
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        return {
            'model': self.model,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'cv_scores': cv_scores,
            'predictions': y_pred,
            'y_test': y_test,
            'feature_importance': self.feature_importance,
            'shap_values': self.shap_values
        }


# ==================== 4. 因果效应估计器 ====================

class CausalEffectEstimator:
    """基于残差法的因果效应估计"""
    
    def __init__(self):
        self.causal_effects = None
        
    def estimate(self, X, y, feature_names, n_bootstrap=50):
        """估计因果效应"""
        print("估计因果效应...")
        
        n_features = X.shape[1]
        causal_effects = np.zeros(n_features)
        
        # 简化的因果效应估计
        for i in range(n_features):
            other_features = [j for j in range(n_features) if j != i]
            
            effects = []
            for _ in range(n_bootstrap):
                indices = np.random.choice(len(y), size=len(y), replace=True)
                X_boot = X[indices]
                y_boot = y[indices]
                
                # 拟合控制模型
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_boot[:, other_features], y_boot)
                y_pred_control = ridge.predict(X_boot[:, other_features])
                
                # 计算残差相关性
                residuals = y_boot - y_pred_control
                corr = np.corrcoef(X_boot[:, i], residuals)[0, 1]
                if not np.isnan(corr):
                    effects.append(corr)
            
            if effects:
                causal_effects[i] = np.median(effects)
        
        # 归一化
        max_abs = np.max(np.abs(causal_effects))
        if max_abs > 0:
            causal_effects = causal_effects / max_abs
        
        self.causal_effects = dict(zip(feature_names, causal_effects))
        
        return self.causal_effects


# ==================== 5. 靶点排序器 ====================

class TargetRanker:
    """综合靶点排序器"""
    
    def __init__(self):
        self.ranking_results = None
        
    def rank_targets(self, feature_importance, causal_effects, feature_names):
        """排序靶点"""
        print("排序靶点...")
        
        importance_scores = np.array([feature_importance.get(f, 0) for f in feature_names])
        causal_scores = np.array([abs(causal_effects.get(f, 0)) for f in feature_names])
        
        # 归一化
        if np.max(importance_scores) > 0:
            importance_scores = importance_scores / np.max(importance_scores)
        if np.max(causal_scores) > 0:
            causal_scores = causal_scores / np.max(causal_scores)
        
        # 综合评分
        combined_scores = 0.6 * importance_scores + 0.4 * causal_scores
        
        # 排序
        sorted_indices = np.argsort(combined_scores)[::-1]
        ranked_features = [feature_names[i] for i in sorted_indices]
        
        results = pd.DataFrame({
            'feature': ranked_features,
            'combined_score': combined_scores[sorted_indices],
            'importance_score': importance_scores[sorted_indices],
            'causal_score': causal_scores[sorted_indices]
        })
        
        results['percentile'] = 1 - (np.arange(len(results)) / len(results))
        
        self.ranking_results = results
        
        return results


# ==================== 6. 主预测系统 ====================

class ErgothioneinePredictor:
    """麦角硫因产量预测系统"""
    
    def __init__(self, output_dir='wangluo/results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 60)
        print("麦角硫因产量优化靶点预测系统 v5.0")
        print("=" * 60)
    
    def run_analysis(self, transcriptomics_path, metabolomics_path, n_features=300):
        """运行完整分析"""
        
        # 1. 数据预处理
        print("\n[1/4] 数据加载与预处理")
        processor = EfficientDataProcessor()
        data = processor.load_and_process(transcriptomics_path, metabolomics_path)
        
        # 2. 特征选择
        print("\n[2/4] 特征选择")
        selector = FeatureSelector(n_features=n_features)
        X_selected = selector.select_features(data['X'], data['y'])
        
        # 3. 模型训练
        print("\n[3/4] 模型训练与评估")
        trainer = RobustModelTrainer()
        results = trainer.train(X_selected, data['y'])
        
        # 4. 因果效应估计
        print("\n[4/4] 因果效应分析与靶点排序")
        causal_estimator = CausalEffectEstimator()
        causal_effects = causal_estimator.estimate(
            X_selected.values, data['y'].values, X_selected.columns.tolist()
        )
        
        # 靶点排序
        ranker = TargetRanker()
        ranking_results = ranker.rank_targets(
            results['feature_importance'].to_dict(),
            causal_effects,
            X_selected.columns.tolist()
        )
        
        # 生成报告
        self._generate_report(data, results, ranking_results)
        
        # 保存
        joblib.dump(trainer.model, f'{self.output_dir}/model.joblib')
        ranking_results.to_csv(f'{self.output_dir}/target_ranking.csv', index=False)
        
        return {
            'top_targets': ranking_results['feature'].head(50).tolist(),
            'ranking_results': ranking_results,
            'model_performance': {
                'test_r2': results['test_r2'],
                'test_rmse': results['test_rmse']
            },
            'data': data
        }
    
    def _generate_report(self, data, model_results, ranking_results):
        """生成分析报告"""
        print("\n生成分析报告...")
        
        # 特征重要性图
        top_features = ranking_results.head(20)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(20), top_features['combined_score'].values[::-1], color='steelblue')
        plt.yticks(range(20), top_features['feature'].values[::-1])
        plt.xlabel('Combined Score')
        plt.title('TOP 20 Target Features')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/top_targets.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 预测性能图
        if 'predictions' in model_results and model_results['predictions'] is not None:
            plt.figure(figsize=(10, 6))
            plt.scatter(model_results['y_test'], model_results['predictions'], alpha=0.7)
            y_min = min(model_results['y_test'].min(), model_results['predictions'].min())
            y_max = max(model_results['y_test'].max(), model_results['predictions'].max())
            plt.plot([y_min, y_max], [y_min, y_max], 'r--', alpha=0.5)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'Prediction Performance (R² = {model_results["test_r2"]:.4f})')
            plt.savefig(f'{self.output_dir}/prediction_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 文本报告
        with open(f'{self.output_dir}/report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("麦角硫因产量优化靶点预测报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("分析概况:\n")
            f.write(f"- 总样本数: {len(data['y'])}\n")
            f.write(f"- 总特征数: {data['X'].shape[1]}\n")
            f.write(f"- 选择特征数: {len(ranking_results)}\n")
            f.write(f"- 模型 R² (测试集): {model_results['test_r2']:.4f}\n")
            f.write(f"- 模型 RMSE (测试集): {model_results['test_rmse']:.4f}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("TOP 10 推荐靶点\n")
            f.write("=" * 60 + "\n\n")
            
            for i, row in ranking_results.head(10).iterrows():
                f.write(f"{i+1:2d}. {row['feature']:25s} ")
                f.write(f"综合得分: {row['combined_score']:.4f}\n")
        
        print(f"\n结果已保存到 {self.output_dir}/")



def main():
    """主函数"""
    
    transcript_path = 'wangluo/data/transcriptomics.csv'
    metabo_path = 'wangluo/data/metabolomics.csv'
    
    # 检查数据文件
    if not os.path.exists(transcript_path):
        print("创建示例数据...")
        create_sample_data(transcript_path, metabo_path)
    
    # 运行分析
    print("\n开始分析...")
    predictor = ErgothioneinePredictor(output_dir='wangluo/results')
    results = predictor.run_analysis(transcript_path, metabo_path, n_features=200)
    
    # 打印结果
    if results:
        print("\n" + "=" * 60)
        print("结果摘要")
        print("=" * 60)
        
        print(f"\n模型性能:")
        print(f"  R²: {results['model_performance']['test_r2']:.4f}")
        print(f"  RMSE: {results['model_performance']['test_rmse']:.4f}")
        
        print("\nTOP 10 推荐靶点:")
        for i, target in enumerate(results['top_targets'][:10], 1):
            row = results['ranking_results'][results['ranking_results']['feature'] == target].iloc[0]
            print(f"  {i:2d}. {target:25s} (得分: {row['combined_score']:.4f})")
        
        print("\n" + "=" * 60)
    
    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()

