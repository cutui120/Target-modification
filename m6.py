
"""
高效麦角硫因产量优化靶点预测系统 v4.0
基于正交机器学习、动态先验整合与不确定性量化
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')
import os
from typing import Dict, List, Tuple
import json
from difflib import SequenceMatcher

np.random.seed(42)



# ==================== 1. 数据处理器 ====================

class RobustDataProcessor:
    """稳健的多组学数据处理器"""
    
    def __init__(self):
        self.gene_scaler = RobustScaler(quantile_range=(10, 90))
        self.metabo_scaler = StandardScaler()
        
    def load_and_process(self, transcript_path: str, metabo_path: str, 
                        yield_col: str = None) -> Dict:
        """加载并预处理数据"""
        print("加载和预处理数据...")
        
        transcriptomics = pd.read_csv(transcript_path, index_col=0)
        metabolomics = pd.read_csv(metabo_path, index_col=0)
        
        # 识别产量数据
        yield_data, yield_info, metabolomics = self._identify_yield_data(
            transcriptomics, metabolomics, yield_col
        )
        
        # 对齐样本
        common_samples = transcriptomics.index.intersection(metabolomics.index)
        if len(common_samples) < 10:
            common_samples = transcriptomics.index[:min(len(transcriptomics), len(metabolomics))]
            metabolomics.index = transcriptomics.index[:len(metabolomics)]
            yield_data.index = transcriptomics.index[:len(yield_data)]
        
        transcriptomics = transcriptomics.loc[common_samples]
        metabolomics = metabolomics.loc[common_samples]
        yield_data = yield_data.loc[common_samples]
        
        # 预处理
        X_genes = self._process_transcriptomics(transcriptomics)
        X_metabo = self._process_metabolomics(metabolomics)
        
        # 基因筛选
        selected_genes = self._gene_selection(X_genes, yield_data, n_genes=800)
        
        return {
            'X_genes': X_genes[selected_genes],
            'X_metabo': X_metabo,
            'y': yield_data,
            'yield_info': yield_info,
            'gene_names': selected_genes,
            'metabo_names': X_metabo.columns.tolist()
        }
    
    def _identify_yield_data(self, transcriptomics, metabolomics, yield_col=None):
        """识别产量数据"""
        yield_info = {'method': 'unknown', 'confidence': 0.0}
        
        if yield_col and yield_col in metabolomics.columns:
            yield_data = metabolomics[yield_col].copy()
            metabolomics = metabolomics.drop(yield_col, axis=1)
            yield_info.update({'method': 'specified_column', 'confidence': 0.9})
            print(f"  使用指定产量列: {yield_col}")
            return yield_data, yield_info, metabolomics
        
        # 关键词匹配
        ergo_keywords = ['ergothioneine', 'ergothion', 'egt', 'thioneine', 'hercynine']
        
        for col in metabolomics.columns:
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in ergo_keywords):
                print(f"  通过关键词找到产量数据: {col}")
                yield_data = metabolomics[col].copy()
                metabolomics = metabolomics.drop(col, axis=1)
                yield_info.update({'method': 'keyword_match', 'confidence': 0.7})
                return yield_data, yield_info, metabolomics
        
        # 使用PCA代理
        print("  未找到麦角硫因数据，使用代谢组第一主成分作为代理...")
        data_clean = metabolomics.fillna(metabolomics.median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data_clean.values)
        
        pca = PCA(n_components=1, random_state=42)
        proxy_yield = pca.fit_transform(X_scaled).flatten()
        
        yield_info.update({'method': 'pca_proxy', 'confidence': 0.5})
        return pd.Series(proxy_yield, index=metabolomics.index, name='yield_proxy'), yield_info, metabolomics
    
    def _process_transcriptomics(self, data):
        """预处理转录组数据"""
        # 对数变换
        min_val = data[data > 0].min().min()
        epsilon = min_val * 0.1 if min_val > 0 else 1e-6
        data_log = np.log1p(data.clip(lower=0) + epsilon)
        
        # 过滤低表达基因
        expr_threshold = np.percentile(data_log.values.flatten(), 25)
        mask_expr = (data_log > expr_threshold).sum(axis=0) > data_log.shape[0] * 0.3
        
        var_threshold = np.percentile(data_log.var(), 25)
        mask_var = data_log.var() > var_threshold
        
        mask = mask_expr & mask_var
        if mask.sum() < 100:
            mask = data_log.var() > np.percentile(data_log.var(), 10)
        
        data_filtered = data_log.loc[:, mask]
        print(f"  过滤后保留基因: {data_filtered.shape[1]}/{data.shape[1]}")
        
        # 标准化
        data_scaled = self.gene_scaler.fit_transform(data_filtered)
        
        return pd.DataFrame(data_scaled, columns=data_filtered.columns, index=data_filtered.index)
    
    def _process_metabolomics(self, data):
        """预处理代谢组数据"""
        # KNN插补
        imputer = KNNImputer(n_neighbors=5)
        data_filled = pd.DataFrame(
            imputer.fit_transform(data.fillna(data.median())),
            columns=data.columns, index=data.index
        )
        
        # 离群值处理
        for col in data_filled.columns:
            median = np.median(data_filled[col])
            mad = np.median(np.abs(data_filled[col] - median)) * 1.4826
            if mad > 0:
                lower, upper = median - 3 * mad, median + 3 * mad
                data_filled[col] = data_filled[col].clip(lower, upper)
        
        data_scaled = self.metabo_scaler.fit_transform(data_filled)
        return pd.DataFrame(data_scaled, columns=data_filled.columns, index=data_filled.index)
    
    def _gene_selection(self, X_genes, y, n_genes=800):
        """多层次基因筛选"""
        print(f"  进行基因筛选，目标保留 {n_genes} 个基因...")
        
        # 高变异基因
        variances = X_genes.var()
        high_var_genes = variances.nlargest(int(n_genes * 1.5)).index.tolist()
        X_temp = X_genes[high_var_genes]
        
        # 相关性筛选
        correlations = [(col, abs(np.corrcoef(X_temp[col], y)[0, 1])) for col in X_temp.columns]
        correlations.sort(key=lambda x: x[1] if not np.isnan(x[1]) else 0, reverse=True)
        corr_genes = [g for g, _ in correlations[:int(n_genes * 1.2)]]
        
        # 随机森林重要性
        X_temp2 = X_genes[corr_genes]
        rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(X_temp2.values, y.values)
        
        importances = pd.Series(rf.feature_importances_, index=corr_genes)
        selected_genes = importances.nlargest(n_genes).index.tolist()
        
        print(f"  最终选择 {len(selected_genes)} 个基因")
        return selected_genes


# ==================== 2. 正交机器学习特征选择器 ====================

class OrthogonalFeatureSelector:
    """基于正交机器学习的特征选择器"""
    
    def __init__(self, n_folds=5, n_bootstrap=50):
        self.n_folds = n_folds
        self.n_bootstrap = n_bootstrap
        
    def select_features(self, X, y, feature_names):
        """使用正交机器学习进行特征选择"""
        print("使用正交机器学习选择特征...")
        
        n_features = X.shape[1]
        causal_effects = np.zeros(n_features)
        p_values = np.ones(n_features)
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            for j in range(n_features):
                effect, p_val = self._orthogonal_effect(X_train, y_train, X_test, y_test, j)
                if not np.isnan(effect):
                    causal_effects[j] += effect
                    p_values[j] = min(p_values[j], p_val)
        
        causal_effects /= self.n_folds
        
        # FDR校正
        _, pvals_fdr = fdrcorrection(p_values, alpha=0.1)
        
        # 稳定性分数
        stability_scores = self._compute_stability(X, y, feature_names)
        
        # 综合评分
        combined_scores = self._compute_combined_scores(causal_effects, pvals_fdr, stability_scores)
        
        # 选择特征
        threshold = np.percentile(combined_scores, 70)
        selected_mask = combined_scores > threshold
        
        if selected_mask.sum() < 20:
            selected_mask = combined_scores > np.percentile(combined_scores, 50)
        
        selected_features = [feature_names[i] for i in np.where(selected_mask)[0]]
        
        print(f"  选择了 {len(selected_features)} 个特征")
        
        return {
            'selected_features': selected_features,
            'causal_effects': dict(zip(feature_names, causal_effects)),
            'p_values': dict(zip(feature_names, pvals_fdr)),
            'stability_scores': dict(zip(feature_names, stability_scores)),
            'combined_scores': dict(zip(feature_names, combined_scores))
        }
    
    def _orthogonal_effect(self, X_train, y_train, X_test, y_test, feature_idx):
        """正交机器学习估计单个特征效应"""
        try:
            D = X_train[:, feature_idx]
            W = np.delete(X_train, feature_idx, axis=1)
            D_test = X_test[:, feature_idx]
            W_test = np.delete(X_test, feature_idx, axis=1)
            
            # 预测处理变量
            model_D = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            model_D.fit(W, D)
            D_residual = D_test - model_D.predict(W_test)
            
            # 预测结果变量
            model_Y = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            model_Y.fit(W, y_train)
            Y_residual = y_test - model_Y.predict(W_test)
            
            # 正交回归
            if np.var(D_residual) > 1e-10:
                model = BayesianRidge()
                model.fit(D_residual.reshape(-1, 1), Y_residual)
                effect = model.coef_[0]
                
                std_error = np.sqrt(model.sigma_[0, 0]) if hasattr(model, 'sigma_') else 0.1
                if std_error > 0:
                    z_score = abs(effect) / std_error
                    p_value = 2 * (1 - stats.norm.cdf(z_score))
                else:
                    p_value = 1.0
                
                return effect, p_value
            
            return 0.0, 1.0
        except:
            return 0.0, 1.0
    
    def _compute_stability(self, X, y, feature_names):
        """计算稳定性分数"""
        n_features = X.shape[1]
        selection_counts = np.zeros(n_features)
        
        for _ in range(self.n_bootstrap):
            indices = np.random.choice(len(y), len(y), replace=True)
            X_boot, y_boot = X[indices], y[indices]
            
            model = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=None, n_jobs=-1)
            model.fit(X_boot, y_boot)
            
            threshold = np.percentile(model.feature_importances_, 75)
            selection_counts[model.feature_importances_ > threshold] += 1
        
        return selection_counts / self.n_bootstrap
    
    def _compute_combined_scores(self, causal_effects, p_values, stability_scores):
        """计算综合评分"""
        # 归一化
        abs_effects = np.abs(causal_effects)
        if abs_effects.max() > 0:
            norm_effects = abs_effects / abs_effects.max()
        else:
            norm_effects = abs_effects
        
        significance = 1 - p_values
        
        # 加权组合
        combined = 0.4 * norm_effects + 0.3 * significance + 0.3 * stability_scores
        
        return combined


# ==================== 3. 堆叠集成预测器 ====================

class StackedEnsemblePredictor:
    """堆叠集成预测器"""
    
    def __init__(self, cv_folds=5):
        self.cv_folds = cv_folds
        self.base_models = [
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)),
            ('gbr', GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)),
            ('elastic', ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=5000)),
            ('extra', ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)),
            ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance'))
        ]
        self.meta_model = BayesianRidge()
        
    def predict_and_explain(self, X_genes, X_metabo, y, gene_names, metabo_names):
        """预测和解释"""
        print("训练堆叠集成模型...")
        
        X_combined = np.column_stack([X_genes, X_metabo])
        feature_names = gene_names + metabo_names
        n_genes = len(gene_names)
        
        # 训练堆叠集成
        meta_features, model_scores = self._train_ensemble(X_combined, y)
        self.meta_model.fit(meta_features, y)
        
        # 特征重要性
        feature_importance = self._compute_importance(X_combined, y, feature_names)
        
        # 模型贡献度
        model_contributions = {name: max(0, score) for name, score in model_scores.items()}
        total = sum(model_contributions.values())
        if total > 0:
            model_contributions = {k: v/total for k, v in model_contributions.items()}
        
        return {
            'feature_importance': feature_importance,
            'gene_importance': {k: v for k, v in feature_importance.items() if k in gene_names},
            'metabo_importance': {k: v for k, v in feature_importance.items() if k in metabo_names},
            'model_contributions': model_contributions
        }
    
    def _train_ensemble(self, X, y):
        """训练集成模型"""
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))
        model_scores = {}
        
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            for i, (name, model) in enumerate(self.base_models):
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                y_pred = model_clone.predict(X_val)
                meta_features[val_idx, i] = y_pred
                
                if name not in model_scores:
                    model_scores[name] = []
                model_scores[name].append(r2_score(y_val, y_pred))
        
        model_scores = {k: np.mean(v) for k, v in model_scores.items()}
        return meta_features, model_scores
    
    def _compute_importance(self, X, y, feature_names):
        """计算特征重要性"""
        n_features = X.shape[1]
        importance_sum = np.zeros(n_features)
        weight_sum = 0
        
        for name, model in self.base_models:
            model_clone = clone(model)
            model_clone.fit(X, y)
            
            if hasattr(model_clone, 'feature_importances_'):
                imp = model_clone.feature_importances_
            elif hasattr(model_clone, 'coef_'):
                imp = np.abs(model_clone.coef_)
            else:
                continue
            
            if imp.sum() > 0:
                imp = imp / imp.sum()
            
            importance_sum += imp
            weight_sum += 1
        
        if weight_sum > 0:
            importance_avg = importance_sum / weight_sum
        else:
            importance_avg = np.ones(n_features) / n_features
        
        return dict(zip(feature_names, importance_avg))


# ==================== 4. 生物学先验整合器 ====================

class BiologicalPriorIntegrator:
    """整合生物学先验知识"""
    
    def __init__(self):
        self.ergo_pathways = {
            'sulfur_metabolism': ['cysteine', 'methionine', 'homocysteine', 'glutathione', 'sulfate'],
            'histidine_metabolism': ['histidine', 'hercynine', 'ergothioneine', 'histamine'],
            'redox_regulation': ['thioredoxin', 'peroxiredoxin', 'glutaredoxin', 'oxidase', 'reductase'],
            'methylation': ['methyltransferase', 'sam', 'sah', 'adenosyl']
        }
        
        self.known_ergo_genes = ['egtA', 'egtB', 'egtC', 'egtD', 'egtE', 'egt1', 'egt2']
    
    def integrate_prior(self, gene_scores, gene_names):
        """整合生物学先验"""
        print("整合生物学先验知识...")
        
        enhanced_scores = {}
        
        for gene in gene_names:
            base_score = gene_scores.get(gene, 0)
            
            pathway_bonus = self._pathway_score(gene)
            similarity_bonus = self._similarity_score(gene)
            keyword_bonus = self._keyword_score(gene)
            
            # 动态权重
            prior_weight = 0.4 * (1 - min(base_score, 1))
            prior_contribution = (pathway_bonus + similarity_bonus + keyword_bonus) / 3
            
            enhanced_score = base_score * (1 + prior_weight * prior_contribution)
            enhanced_scores[gene] = enhanced_score
        
        return {'enhanced_scores': enhanced_scores}
    
    def _pathway_score(self, gene):
        """通路关联分数"""
        gene_lower = gene.lower()
        for pathway, keywords in self.ergo_pathways.items():
            for kw in keywords:
                if kw in gene_lower or gene_lower in kw:
                    return 0.4
        return 0.0
    
    def _similarity_score(self, gene):
        """与已知基因相似性"""
        gene_lower = gene.lower()
        max_sim = 0.0
        
        for known in self.known_ergo_genes:
            if known.lower() in gene_lower or gene_lower in known.lower():
                return 0.5
            sim = SequenceMatcher(None, gene_lower, known.lower()).ratio()
            max_sim = max(max_sim, sim)
        
        return max_sim * 0.3 if max_sim > 0.6 else 0.0
    
    def _keyword_score(self, gene):
        """关键词分数"""
        gene_lower = gene.lower()
        
        functional_kw = ['synthase', 'transferase', 'ligase', 'reductase', 'oxidase', 'transporter']
        substrate_kw = ['cysteine', 'methionine', 'glutathione', 'histidine', 'hercynine', 'sulfur', 'thiol']
        
        bonus = 0.0
        for kw in functional_kw:
            if kw in gene_lower:
                bonus += 0.2
                break
        
        for kw in substrate_kw:
            if kw in gene_lower:
                bonus += 0.3
                break
        
        return min(bonus, 0.5)


# ==================== 5. 贝叶斯综合排序器 ====================

class BayesianRanker:
    """贝叶斯综合排序"""
    
    def __init__(self, n_bootstrap=500):
        self.n_bootstrap = n_bootstrap
    
    def rank_features(self, scores_dict, method='bayesian'):
        """综合排名"""
        print(f"使用 {method} 方法进行综合排名...")
        
        all_features = set()
        for scores in scores_dict.values():
            all_features.update(scores.keys())
        
        feature_list = list(all_features)
        n_features = len(feature_list)
        n_methods = len(scores_dict)
        
        score_matrix = np.zeros((n_features, n_methods))
        for i, feature in enumerate(feature_list):
            for j, scores in enumerate(scores_dict.values()):
                score_matrix[i, j] = scores.get(feature, 0)
        
        # 贝叶斯组合
        combined_scores = self._bayesian_combine(score_matrix)
        
        # 不确定性
        uncertainties = self._compute_uncertainties(score_matrix)
        
        # 排名置信区间
        rank_ci = self._compute_rank_ci(score_matrix, feature_list)
        
        # 排序
        sorted_idx = np.argsort(combined_scores)[::-1]
        ranked_features = [feature_list[i] for i in sorted_idx]
        
        return {
            'ranked_features': ranked_features,
            'combined_scores': dict(zip(feature_list, combined_scores)),
            'uncertainties': dict(zip(feature_list, uncertainties)),
            'rank_confidence_intervals': rank_ci
        }
    
    def _bayesian_combine(self, score_matrix):
        """贝叶斯组合"""
        n_features, n_methods = score_matrix.shape
        
        # 标准化
        score_norm = np.zeros_like(score_matrix)
        for j in range(n_methods):
            col = score_matrix[:, j]
            if col.std() > 0:
                score_norm[:, j] = (col - col.mean()) / col.std()
        
        # 加权平均
        weights = 1 / (1 + np.var(score_norm, axis=0))
        weights = weights / weights.sum()
        
        combined = np.average(score_norm, axis=1, weights=weights)
        
        # 归一化到 [0, 1]
        if combined.max() > combined.min():
            combined = (combined - combined.min()) / (combined.max() - combined.min())
        
        return combined
    
    def _compute_uncertainties(self, score_matrix):
        """计算不确定性"""
        uncertainties = np.std(score_matrix, axis=1)
        if uncertainties.max() > 0:
            uncertainties = uncertainties / uncertainties.max()
        return uncertainties
    
    def _compute_rank_ci(self, score_matrix, feature_list):
        """计算排名置信区间"""
        n_features, n_methods = score_matrix.shape
        bootstrap_ranks = np.zeros((self.n_bootstrap, n_features))
        
        for b in range(self.n_bootstrap):
            method_idx = np.random.choice(n_methods, n_methods, replace=True)
            boot_scores = score_matrix[:, method_idx].mean(axis=1)
            bootstrap_ranks[b] = stats.rankdata(-boot_scores)
        
        rank_ci = {}
        for i, feature in enumerate(feature_list):
            ranks = bootstrap_ranks[:, i]
            rank_ci[feature] = {
                'median_rank': float(np.median(ranks)),
                'ci_lower': float(np.percentile(ranks, 2.5)),
                'ci_upper': float(np.percentile(ranks, 97.5))
            }
        
        return rank_ci


# ==================== 6. 主分析系统 ====================

class ErgothioneinePredictor:
    """麦角硫因产量优化靶点预测系统"""
    
    def __init__(self, output_dir='wangluo/advanced_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 60)
        print("麦角硫因产量优化靶点预测系统 v4.0")
        print("=" * 60)
    
    def run_analysis(self, transcriptomics_path, metabolomics_path, yield_column=None):
        """运行完整分析"""
        
        # 1. 数据预处理
        print("\n[1/5] 数据加载与预处理")
        processor = RobustDataProcessor()
        data = processor.load_and_process(transcriptomics_path, metabolomics_path, yield_column)
        
        print(f"  样本数: {len(data['y'])}")
        print(f"  基因数: {len(data['gene_names'])}")
        print(f"  代谢物数: {len(data['metabo_names'])}")
        
        # 2. 正交机器学习特征选择
        print("\n[2/5] 正交机器学习特征选择")
        selector = OrthogonalFeatureSelector(n_folds=3, n_bootstrap=30)
        selection_results = selector.select_features(
            data['X_genes'].values, data['y'].values, data['gene_names']
        )
        
        selected_genes = selection_results['selected_features']
        X_selected = data['X_genes'][selected_genes].values
        
        # 3. 堆叠集成预测
        print("\n[3/5] 堆叠集成预测")
        predictor = StackedEnsemblePredictor(cv_folds=3)
        prediction_results = predictor.predict_and_explain(
            X_selected, data['X_metabo'].values, data['y'].values,
            selected_genes, data['metabo_names']
        )
        
        # 4. 生物学先验整合
        print("\n[4/5] 生物学先验整合")
        bio_integrator = BiologicalPriorIntegrator()
        biological_results = bio_integrator.integrate_prior(
            prediction_results['gene_importance'], selected_genes
        )
        
        # 5. 贝叶斯综合排名
        print("\n[5/5] 贝叶斯综合排名")
        ranker = BayesianRanker(n_bootstrap=200)
        
        all_scores = {
            'causal_effects': selection_results['causal_effects'],
            'prediction_importance': prediction_results['gene_importance'],
            'biological_enhanced': biological_results['enhanced_scores'],
            'stability': selection_results['stability_scores']
        }
        
        ranking_results = ranker.rank_features(all_scores)
        
        # 生成报告
        self._generate_report(ranking_results, data, all_scores)
        
        return {
            'top_targets': ranking_results['ranked_features'][:50],
            'all_scores': ranking_results['combined_scores'],
            'uncertainties': ranking_results['uncertainties'],
            'rank_ci': ranking_results['rank_confidence_intervals'],
            'data': data
        }
    
    def _generate_report(self, ranking_results, data, all_scores):
        """生成报告"""
        print("\n生成报告...")
        
        # 保存详细结果
        detailed_data = []
        for gene in ranking_results['ranked_features']:
            detailed_data.append({
                'Gene': gene,
                'Combined_Score': ranking_results['combined_scores'].get(gene, 0),
                'Causal_Effect': all_scores['causal_effects'].get(gene, 0),
                'Prediction_Importance': all_scores['prediction_importance'].get(gene, 0),
                'Biological_Score': all_scores['biological_enhanced'].get(gene, 0),
                'Stability': all_scores['stability'].get(gene, 0),
                'Uncertainty': ranking_results['uncertainties'].get(gene, 0)
            })
        
        results_df = pd.DataFrame(detailed_data)
        results_df.to_csv(f'{self.output_dir}/gene_ranking.csv', index=False)
        
        # 文本报告
        with open(f'{self.output_dir}/summary.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("麦角硫因产量优化靶点预测报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"样本数: {len(data['y'])}\n")
            f.write(f"分析基因数: {len(ranking_results['ranked_features'])}\n")
            f.write(f"产量识别方法: {data['yield_info']['method']}\n\n")
            
            f.write("TOP 20 推荐靶点:\n")
            f.write("-" * 60 + "\n")
            
            for i, gene in enumerate(ranking_results['ranked_features'][:20], 1):
                score = ranking_results['combined_scores'].get(gene, 0)
                unc = ranking_results['uncertainties'].get(gene, 0)
                f.write(f"{i:2d}. {gene:30s} 得分: {score:.4f} (±{unc:.3f})\n")
        
        print(f"\n结果已保存到 {self.output_dir}/")



    # 转录组数据
    base_expr = np.random.lognormal(mean=3, sigma=1.2, size=(n_samples, n_genes))
    
    # 添加调控模式
    for i, idx in enumerate(regulatory_indices):
        strength = np.random.uniform(0.5, 2.0)
        sign = 1 if i < n_regulatory // 2 else -1
        sample_idx = np.random.choice(n_samples, int(n_samples * 0.6), replace=False)
        base_expr[sample_idx, idx] *= (1 + strength * sign)
    
    # 基因名称
    gene_names = [f'Gene_{i:05d}' for i in range(n_genes)]
    
    ergo_genes = ['egtA', 'egtB', 'egtC', 'egtD', 'egtE', 'egt1', 'egt2',
                 'hercynine_synthase', 'cysteine_ligase', 'glutathione_synthase',
                 'methionine_adenosyltransferase', 'ergothioneine_transporter',
                 'sulfur_regulator', 'redox_sensor', 'SAM_synthase']
    
    for i, idx in enumerate(regulatory_indices[:len(ergo_genes)]):
        gene_names[idx] = ergo_genes[i]
    
    transcriptomics = pd.DataFrame(
        base_expr, columns=gene_names,
        index=[f'Sample_{i:04d}' for i in range(n_samples)]
    )
    
    # 代谢组数据
    metabo_base = np.random.lognormal(mean=0, sigma=1.0, size=(n_samples, n_metabolites))
    
    metabo_names = [f'Met_{i:04d}' for i in range(n_metabolites)]
    known_metabo = ['Hercynine', 'Glutathione', 'Cysteine', 'Methionine',
                   'S_Adenosyl_Methionine', 'Homocysteine', 'Histidine']
    
    for i, name in enumerate(known_metabo):
        if i < len(metabo_names):
            metabo_names[i] = name
    
    # 产量数据（与调控基因相关）
    yield_data = np.zeros(n_samples)
    for i in range(min(10, n_regulatory)):
        idx = regulatory_indices[i]
        yield_data += 0.2 * base_expr[:, idx]
    
    for i in range(3):
        yield_data += 0.25 * metabo_base[:, i]
    
    yield_data += np.random.normal(0, 0.1, n_samples)
    yield_data = (yield_data - yield_data.mean()) / yield_data.std()
    
    metabolomics = pd.DataFrame(
        metabo_base, columns=metabo_names,
        index=[f'Sample_{i:04d}' for i in range(n_samples)]
    )
    metabolomics['Ergothioneine'] = yield_data
    
    # 添加缺失值
    missing_mask = np.random.random(metabolomics.shape) < 0.03
    metabolomics.values[missing_mask] = np.nan
    
    transcriptomics.to_csv(transcript_path)
    metabolomics.to_csv(metabo_path)
    
    print(f"示例数据已创建: {transcript_path}, {metabo_path}")


# ==================== 8. 主函数 ====================

def main():
    """主函数"""
    
    transcript_path = 'wangluo/data/transcriptomics.csv'
    metabo_path = 'wangluo/data/metabolomics.csv'
    
    if not os.path.exists(transcript_path):
        print("创建示例数据...")
        create_sample_data(transcript_path, metabo_path)
    
    print("\n开始分析...")
    predictor = ErgothioneinePredictor(output_dir='wangluo/advanced_results')
    
    try:
        results = predictor.run_analysis(transcript_path, metabo_path)
        
        if results and 'top_targets' in results:
            print("\n" + "=" * 60)
            print("分析完成！主要发现:")
            print("=" * 60)
            
            print("\nTOP 10 推荐靶点:")
            for i, gene in enumerate(results['top_targets'][:10], 1):
                score = results['all_scores'].get(gene, 0)
                unc = results['uncertainties'].get(gene, 0)
                print(f"  {i:2d}. {gene:30s} 得分: {score:.4f} (±{unc:.3f})")
            
            print(f"\n详细结果已保存到 wangluo/advanced_results/")
            print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"分析出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()

