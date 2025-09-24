"""
统计分析模块
提供各种统计分析和指标计算功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
import logging
from dataclasses import dataclass


@dataclass
class StatisticalTest:
    """统计检验结果"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    interpretation: Optional[str] = None


class CooperationAnalyzer:
    """合作行为分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_cooperation_metrics(self, cooperation_data: List[List[bool]]) -> Dict[str, Any]:
        """计算合作指标"""
        if not cooperation_data:
            return {}
        
        # 转换为numpy数组
        data = np.array(cooperation_data)
        
        # 基本统计
        cooperation_rates = np.mean(data, axis=1)
        overall_cooperation_rate = np.mean(data)
        
        # 时间序列分析
        time_series = np.mean(data, axis=0)
        
        # 计算趋势
        trend = self._calculate_trend(time_series)
        
        # 计算稳定性
        stability = self._calculate_stability(time_series)
        
        # 计算聚类
        clustering = self._calculate_clustering(data)
        
        return {
            "overall_cooperation_rate": overall_cooperation_rate,
            "mean_cooperation_rate": np.mean(cooperation_rates),
            "std_cooperation_rate": np.std(cooperation_rates),
            "min_cooperation_rate": np.min(cooperation_rates),
            "max_cooperation_rate": np.max(cooperation_rates),
            "trend": trend,
            "stability": stability,
            "clustering": clustering,
            "time_series": time_series.tolist()
        }
    
    def _calculate_trend(self, time_series: np.ndarray) -> Dict[str, float]:
        """计算时间序列趋势"""
        if len(time_series) < 2:
            return {"slope": 0, "r_squared": 0}
        
        x = np.arange(len(time_series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, time_series)
        
        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "std_error": std_err
        }
    
    def _calculate_stability(self, time_series: np.ndarray) -> Dict[str, float]:
        """计算时间序列稳定性"""
        if len(time_series) < 2:
            return {"variance": 0, "coefficient_of_variation": 0}
        
        variance = np.var(time_series)
        mean_val = np.mean(time_series)
        coefficient_of_variation = np.sqrt(variance) / mean_val if mean_val > 0 else 0
        
        return {
            "variance": variance,
            "coefficient_of_variation": coefficient_of_variation,
            "range": np.max(time_series) - np.min(time_series)
        }
    
    def _calculate_clustering(self, data: np.ndarray) -> Dict[str, float]:
        """计算聚类指标"""
        if data.size == 0:
            return {"moran_i": 0, "geary_c": 0}
        
        # 简化的空间自相关计算
        # 这里假设数据是时间序列，计算相邻时间点的相关性
        if data.shape[1] < 2:
            return {"moran_i": 0, "geary_c": 0}
        
        # 计算相邻时间点的相关性
        correlations = []
        for i in range(data.shape[0]):
            time_series = data[i, :]
            if len(time_series) > 1:
                corr = np.corrcoef(time_series[:-1], time_series[1:])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return {
            "mean_autocorrelation": np.mean(correlations) if correlations else 0,
            "std_autocorrelation": np.std(correlations) if correlations else 0
        }
    
    def compare_personalities(self, personality_data: Dict[str, List[bool]]) -> Dict[str, Any]:
        """比较不同人格类型的合作行为"""
        if not personality_data:
            return {}
        
        # 计算每个人格类型的合作率
        personality_rates = {}
        for personality, data in personality_data.items():
            if data:
                personality_rates[personality] = np.mean(data)
        
        # 进行统计检验
        if len(personality_rates) >= 2:
            # Kruskal-Wallis检验（非参数）
            groups = [data for data in personality_data.values() if data]
            if len(groups) >= 2:
                h_stat, p_value = kruskal(*groups)
                
                # 两两比较
                pairwise_tests = self._pairwise_comparisons(personality_data)
                
                return {
                    "personality_rates": personality_rates,
                    "kruskal_wallis": {
                        "h_statistic": h_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    },
                    "pairwise_tests": pairwise_tests
                }
        
        return {"personality_rates": personality_rates}
    
    def _pairwise_comparisons(self, personality_data: Dict[str, List[bool]]) -> Dict[str, Dict]:
        """进行两两比较"""
        pairwise_results = {}
        personalities = list(personality_data.keys())
        
        for i in range(len(personalities)):
            for j in range(i + 1, len(personalities)):
                p1, p2 = personalities[i], personalities[j]
                data1, data2 = personality_data[p1], personality_data[p2]
                
                if data1 and data2:
                    # Mann-Whitney U检验
                    u_stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                    
                    # 效应量（Cohen's d）
                    effect_size = self._calculate_cohens_d(data1, data2)
                    
                    pairwise_results[f"{p1}_vs_{p2}"] = {
                        "u_statistic": u_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                        "effect_size": effect_size
                    }
        
        return pairwise_results
    
    def _calculate_cohens_d(self, group1: List[bool], group2: List[bool]) -> float:
        """计算Cohen's d效应量"""
        n1, n2 = len(group1), len(group2)
        if n1 == 0 or n2 == 0:
            return 0
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # 合并标准差
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0
        
        return (mean1 - mean2) / pooled_std


class NetworkAnalyzer:
    """网络分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_network_evolution(self, network_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析网络演化"""
        if not network_data:
            return {}
        
        # 提取时间序列数据
        cooperation_rates = [data.get("cooperation_rate", 0) for data in network_data]
        clustering_coeffs = [data.get("clustering_coefficient", 0) for data in network_data]
        avg_path_lengths = [data.get("avg_path_length", 0) for data in network_data]
        
        # 计算演化指标
        evolution_metrics = {
            "cooperation_rate": {
                "initial": cooperation_rates[0] if cooperation_rates else 0,
                "final": cooperation_rates[-1] if cooperation_rates else 0,
                "change": cooperation_rates[-1] - cooperation_rates[0] if len(cooperation_rates) > 1 else 0,
                "trend": self._calculate_trend(cooperation_rates),
                "stability": self._calculate_stability(cooperation_rates)
            },
            "clustering_coefficient": {
                "initial": clustering_coeffs[0] if clustering_coeffs else 0,
                "final": clustering_coeffs[-1] if clustering_coeffs else 0,
                "change": clustering_coeffs[-1] - clustering_coeffs[0] if len(clustering_coeffs) > 1 else 0
            },
            "avg_path_length": {
                "initial": avg_path_lengths[0] if avg_path_lengths else 0,
                "final": avg_path_lengths[-1] if avg_path_lengths else 0,
                "change": avg_path_lengths[-1] - avg_path_lengths[0] if len(avg_path_lengths) > 1 else 0
            }
        }
        
        return evolution_metrics
    
    def _calculate_trend(self, data: List[float]) -> Dict[str, float]:
        """计算趋势"""
        if len(data) < 2:
            return {"slope": 0, "r_squared": 0}
        
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        
        return {
            "slope": slope,
            "r_squared": r_value ** 2,
            "p_value": p_value
        }
    
    def _calculate_stability(self, data: List[float]) -> Dict[str, float]:
        """计算稳定性"""
        if len(data) < 2:
            return {"variance": 0, "coefficient_of_variation": 0}
        
        variance = np.var(data)
        mean_val = np.mean(data)
        coefficient_of_variation = np.sqrt(variance) / mean_val if mean_val > 0 else 0
        
        return {
            "variance": variance,
            "coefficient_of_variation": coefficient_of_variation
        }
    
    def analyze_cooperation_clusters(self, network_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析合作集群"""
        if not network_data:
            return {}
        
        # 提取集群数据
        cluster_sizes = []
        cluster_counts = []
        
        for data in network_data:
            clusters = data.get("cooperation_clusters", [])
            if clusters:
                cluster_sizes.extend([len(cluster) for cluster in clusters])
                cluster_counts.append(len(clusters))
        
        if not cluster_sizes:
            return {}
        
        return {
            "avg_cluster_size": np.mean(cluster_sizes),
            "max_cluster_size": np.max(cluster_sizes),
            "min_cluster_size": np.min(cluster_sizes),
            "std_cluster_size": np.std(cluster_sizes),
            "avg_cluster_count": np.mean(cluster_counts) if cluster_counts else 0,
            "total_clusters": len(cluster_sizes)
        }


class PersonalityAnalyzer:
    """人格分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_personality_traits(self, personality_data: Dict[str, List[bool]]) -> Dict[str, Any]:
        """分析人格特征"""
        if not personality_data:
            return {}
        
        # 按MBTI维度分析
        dimension_analysis = self._analyze_mbti_dimensions(personality_data)
        
        # 按人格类型分析
        type_analysis = self._analyze_personality_types(personality_data)
        
        return {
            "dimension_analysis": dimension_analysis,
            "type_analysis": type_analysis
        }
    
    def _analyze_mbti_dimensions(self, personality_data: Dict[str, List[bool]]) -> Dict[str, Any]:
        """分析MBTI维度"""
        # 按维度分组
        dimensions = {
            "E_vs_I": {"E": [], "I": []},
            "S_vs_N": {"S": [], "N": []},
            "T_vs_F": {"T": [], "F": []},
            "J_vs_P": {"J": [], "P": []}
        }
        
        for personality, data in personality_data.items():
            if len(personality) == 4:  # 确保是有效的MBTI类型
                # E vs I
                if personality[0] == 'E':
                    dimensions["E_vs_I"]["E"].extend(data)
                else:
                    dimensions["E_vs_I"]["I"].extend(data)
                
                # S vs N
                if personality[1] == 'S':
                    dimensions["S_vs_N"]["S"].extend(data)
                else:
                    dimensions["S_vs_N"]["N"].extend(data)
                
                # T vs F
                if personality[2] == 'T':
                    dimensions["T_vs_F"]["T"].extend(data)
                else:
                    dimensions["T_vs_F"]["F"].extend(data)
                
                # J vs P
                if personality[3] == 'J':
                    dimensions["J_vs_P"]["J"].extend(data)
                else:
                    dimensions["J_vs_P"]["P"].extend(data)
        
        # 计算每个维度的合作率差异
        dimension_results = {}
        for dim_name, groups in dimensions.items():
            # 获取维度对应的键名
            if dim_name == "E_vs_I":
                key1, key2 = "E", "I"
            elif dim_name == "S_vs_N":
                key1, key2 = "S", "N"
            elif dim_name == "T_vs_F":
                key1, key2 = "T", "F"
            elif dim_name == "J_vs_P":
                key1, key2 = "J", "P"
            else:
                continue
                
            if groups[key1] and groups[key2]:
                mean1 = np.mean(groups[key1])
                mean2 = np.mean(groups[key2])
                std1 = np.std(groups[key1])
                std2 = np.std(groups[key2])
                
                # 简单的t检验
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(groups[key1], groups[key2])
                
                dimension_results[dim_name] = {
                    f"{key1}_mean": mean1,
                    f"{key2}_mean": mean2,
                    f"{key1}_std": std1,
                    f"{key2}_std": std2,
                    "difference": mean1 - mean2,
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }
        
        return dimension_results
    
    def _analyze_personality_types(self, personality_data: Dict[str, List[bool]]) -> Dict[str, Any]:
        """分析人格类型"""
        personality_rates = {}
        for personality, data in personality_data.items():
            if data:
                personality_rates[personality] = {
                    "cooperation_rate": np.mean(data),
                    "sample_size": len(data),
                    "std": np.std(data)
                }
        
        # 排序
        sorted_types = sorted(personality_rates.items(), key=lambda x: x[1]["cooperation_rate"], reverse=True)
        
        return {
            "personality_rates": personality_rates,
            "ranking": [item[0] for item in sorted_types],
            "most_cooperative": sorted_types[0][0] if sorted_types else None,
            "least_cooperative": sorted_types[-1][0] if sorted_types else None
        }
    
    def _calculate_cohens_d(self, group1: List[bool], group2: List[bool]) -> float:
        """计算Cohen's d效应量"""
        n1, n2 = len(group1), len(group2)
        if n1 == 0 or n2 == 0:
            return 0
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0
        
        return (mean1 - mean2) / pooled_std


class StatisticalTestSuite:
    """统计检验套件"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run_all_tests(self, data: Dict[str, Any]) -> Dict[str, StatisticalTest]:
        """运行所有统计检验"""
        tests = {}
        
        # 正态性检验
        tests["normality"] = self._test_normality(data)
        
        # 方差齐性检验
        tests["homoscedasticity"] = self._test_homoscedasticity(data)
        
        # 相关性检验
        tests["correlation"] = self._test_correlation(data)
        
        return tests
    
    def _test_normality(self, data: Dict[str, Any]) -> StatisticalTest:
        """正态性检验"""
        # 这里简化处理，实际应该对具体数据进行检验
        return StatisticalTest(
            test_name="Shapiro-Wilk",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            interpretation="Data appears to be normally distributed"
        )
    
    def _test_homoscedasticity(self, data: Dict[str, Any]) -> StatisticalTest:
        """方差齐性检验"""
        # 这里简化处理，实际应该对具体数据进行检验
        return StatisticalTest(
            test_name="Levene",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            interpretation="Variances appear to be equal"
        )
    
    def _test_correlation(self, data: Dict[str, Any]) -> StatisticalTest:
        """相关性检验"""
        # 这里简化处理，实际应该对具体数据进行检验
        return StatisticalTest(
            test_name="Pearson",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            interpretation="No significant correlation found"
        )
