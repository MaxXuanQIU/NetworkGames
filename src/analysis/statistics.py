"""
Statistical Analysis Module
Provides various statistical analysis and metric calculation functions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
import logging


class CooperationAnalyzer:
    """Cooperation Behavior Analyzer"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_cooperation_metrics(self, cooperation_data: List[List[bool]]) -> Dict[str, Any]:
        """Calculate cooperation metrics"""
        if not cooperation_data:
            return {}
        
        # Convert to numpy array
        data = np.array(cooperation_data)
        
        # Basic statistics
        cooperation_rates = np.mean(data, axis=1)
        overall_cooperation_rate = np.mean(data)
        
        # Time series analysis
        time_series = np.mean(data, axis=0)
        
        # Calculate trend
        trend = self._calculate_trend(time_series)
        
        # Calculate stability
        stability = self._calculate_stability(time_series)
        
        # Calculate clustering
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
        """Calculate time series trend"""
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
        """Calculate time series stability"""
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
        """Calculate clustering metrics"""
        if data.size == 0:
            return {"moran_i": 0, "geary_c": 0}
        
        # Simplified spatial autocorrelation calculation
        # Here we assume the data is a time series and calculate the correlation between adjacent time points
        if data.shape[1] < 2:
            return {"moran_i": 0, "geary_c": 0}
        
        # Calculate correlation between adjacent time points
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
        """Compare cooperation behavior across different personality types"""
        if not personality_data:
            return {}
        
        # Calculate cooperation rate for each personality type
        personality_cooperation_rates = {}
        for personality, data in personality_data.items():
            if data:
                personality_cooperation_rates[personality] = np.mean(data)
        
        # Statistical tests
        if len(personality_cooperation_rates) >= 2:
            # Kruskal-Wallis test (non-parametric)
            groups = [data for data in personality_data.values() if data]
            if len(groups) >= 2:
                h_stat, p_value = kruskal(*groups)
                
                # Pairwise comparisons
                pairwise_tests = self._pairwise_comparisons(personality_data)
                
                return {
                    "cooperation_rates": personality_cooperation_rates,
                    "kruskal_wallis": {
                        "h_statistic": h_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    },
                    "pairwise_tests": pairwise_tests
                }
        
        return {"cooperation_rates": personality_cooperation_rates}
    
    def _pairwise_comparisons(self, personality_data: Dict[str, List[bool]]) -> Dict[str, Dict]:
        """Perform pairwise comparisons"""
        pairwise_results = {}
        personalities = list(personality_data.keys())
        
        for i in range(len(personalities)):
            for j in range(i + 1, len(personalities)):
                p1, p2 = personalities[i], personalities[j]
                data1, data2 = personality_data[p1], personality_data[p2]
                
                if data1 and data2:
                    # Mann-Whitney U test
                    u_stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                    
                    # Effect size (Cohen's d)
                    effect_size = self._calculate_cohens_d(data1, data2)
                    
                    pairwise_results[f"{p1}_vs_{p2}"] = {
                        "u_statistic": u_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                        "effect_size": effect_size
                    }
        
        return pairwise_results
    
    def _calculate_cohens_d(self, group1: List[bool], group2: List[bool]) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        if n1 == 0 or n2 == 0:
            return 0
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0
        
        return (mean1 - mean2) / pooled_std


class NetworkAnalyzer:
    """Network Analyzer"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_network_evolution(self, network_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze network evolution"""
        if not network_data:
            return {}
        
        # Extract time series data
        cooperation_rates = [data.get("cooperation_rate", 0) for data in network_data]
        clustering_coeffs = [data.get("clustering_coefficient", 0) for data in network_data]
        avg_path_lengths = [data.get("avg_path_length", 0) for data in network_data]
        
        # Calculate evolution metrics
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
        """Calculate trend"""
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
        """Calculate stability"""
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
        """Analyze cooperation clusters"""
        if not network_data:
            return {}
        
        # Extract cluster data
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
    """Personality Analyzer"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_personality_traits(self, personality_data: Dict[str, List[bool]]) -> Dict[str, Any]:
        """Analyze personality traits"""
        if not personality_data:
            return {}
        
        # Note: This function was originally intended to be an external calling interface, 
        # but the current code directly calls _analyze_mbti_dimensions and _analyze_personality_types 
        # in pair_game_experiment.
        dimension_analysis = self._analyze_mbti_dimensions(personality_data)
        
        type_analysis = self._analyze_personality_types(personality_data, {}) # Placeholder if payoff data isn't available in this scope
        
        return {
            "dimension_analysis": dimension_analysis,
            "type_analysis": type_analysis
        }
    
    def _analyze_mbti_dimensions(self, personality_data: Dict[str, List[bool]], payoff_data: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
        """Analyze MBTI dimensions for both cooperation and payoff"""
        # Group by dimension for Cooperation
        dimensions_coop = {
            "E_vs_I": {"E": [], "I": []},
            "S_vs_N": {"S": [], "N": []},
            "T_vs_F": {"T": [], "F": []},
            "J_vs_P": {"J": [], "P": []}
        }
        
        # Group by dimension for Payoff
        dimensions_payoff = {
            "E_vs_I": {"E": [], "I": []},
            "S_vs_N": {"S": [], "N": []},
            "T_vs_F": {"T": [], "F": []},
            "J_vs_P": {"J": [], "P": []}
        }
        
        # Process Cooperation Data
        for personality, data in personality_data.items():
            if len(personality) == 4:
                # E vs I
                key = 'E' if personality[0] == 'E' else 'I'
                dimensions_coop["E_vs_I"][key].extend(data)
                
                # S vs N
                key = 'S' if personality[1] == 'S' else 'N'
                dimensions_coop["S_vs_N"][key].extend(data)
                
                # T vs F
                key = 'T' if personality[2] == 'T' else 'F'
                dimensions_coop["T_vs_F"][key].extend(data)
                
                # J vs P
                key = 'J' if personality[3] == 'J' else 'P'
                dimensions_coop["J_vs_P"][key].extend(data)

        # Process Payoff Data
        for personality, data in payoff_data.items():
            if len(personality) == 4:
                # E vs I
                key = 'E' if personality[0] == 'E' else 'I'
                dimensions_payoff["E_vs_I"][key].extend(data)
                
                # S vs N
                key = 'S' if personality[1] == 'S' else 'N'
                dimensions_payoff["S_vs_N"][key].extend(data)
                
                # T vs F
                key = 'T' if personality[2] == 'T' else 'F'
                dimensions_payoff["T_vs_F"][key].extend(data)
                
                # J vs P
                key = 'J' if personality[3] == 'J' else 'P'
                dimensions_payoff["J_vs_P"][key].extend(data)
        
        # Helper to calculate stats
        def calculate_dimension_stats(dimensions_dict, metric_name):
            results = {}
            for dim_name, groups in dimensions_dict.items():
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
                    std1 = np.std(groups[key1], ddof=1)
                    std2 = np.std(groups[key2], ddof=1)
                    
                    # Use Welch's t-test (equal_var=False) which does not assume equal population variance
                    from scipy import stats
                    t_stat, p_value = stats.ttest_ind(groups[key1], groups[key2], equal_var=False)
                    
                    results[dim_name] = {
                        f"{key1}_mean": mean1,
                        f"{key2}_mean": mean2,
                        f"{key1}_std": std1,
                        f"{key2}_std": std2,
                        "difference": mean1 - mean2,
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    }
            return results

        coop_results = calculate_dimension_stats(dimensions_coop, metric_name="rate")
        payoff_results = calculate_dimension_stats(dimensions_payoff, metric_name="payoff")

        return {
            "cooperation": coop_results,
            "payoff": payoff_results
        }
    
    def _analyze_personality_types(self, cooperation_data: Dict[str, List[bool]], payoff_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze personality types, support both cooperation and payoff ranking"""
        # Cooperation ranking
        cooperation_rates = {}
        for personality, data in cooperation_data.items():
            if data:
                cooperation_rates[personality] = {
                    "cooperation_rate": np.mean(data),
                    "sample_size": len(data),
                    "std": np.std(data, ddof=1)
                }
        sorted_types = sorted(cooperation_rates.items(), key=lambda x: x[1]["cooperation_rate"], reverse=True)
        
        result = {
            "cooperation_rates": cooperation_rates,
            "cooperation_ranking": [item[0] for item in sorted_types],
            "most_cooperative": sorted_types[0][0] if sorted_types else None,
            "least_cooperative": sorted_types[-1][0] if sorted_types else None
        }
        
        # Payoff ranking
        payoff_rates = {}
        for personality, vals in payoff_data.items():
            if vals:
                payoff_rates[personality] = {
                    "payoff": float(np.mean(vals)),
                    "sample_size": len(vals),
                    "std": float(np.std(vals))
                }
        sorted_payoff = sorted(payoff_rates.items(), key=lambda x: x[1]["payoff"], reverse=True)
        result["payoff_rates"] = payoff_rates
        result["payoff_ranking"] = [item[0] for item in sorted_payoff]
        result["highest_payoff"] = sorted_payoff[0][0] if sorted_payoff else None
        result["lowest_payoff"] = sorted_payoff[-1][0] if sorted_payoff else None
        
        # Statistical significance calculation (Kruskal-Wallis H-test)
        # 1. Cooperation Rate Significance
        # Filter out empty groups to avoid errors
        coop_groups = [data for data in cooperation_data.values() if len(data) > 0]
        if len(coop_groups) >= 2:
            try:
                h_stat, p_val = kruskal(*coop_groups)
                result["cooperation_significance"] = {
                    "test": "Kruskal-Wallis",
                    "statistic": h_stat,
                    "p_value": p_val,
                    "significant": p_val < 0.05
                }
            except ValueError:
                result["cooperation_significance"] = {"error": "Insufficient data"}

        # 2. Payoff Significance
        payoff_groups = [vals for vals in payoff_data.values() if len(vals) > 0]
        if len(payoff_groups) >= 2:
            try:
                h_stat, p_val = kruskal(*payoff_groups)
                result["payoff_significance"] = {
                    "test": "Kruskal-Wallis",
                    "statistic": h_stat,
                    "p_value": p_val,
                    "significant": p_val < 0.05
                }
            except ValueError:
                result["payoff_significance"] = {"error": "Insufficient data"}
        
        return result
    
    def _calculate_cohens_d(self, group1: List[bool], group2: List[bool]) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        if n1 == 0 or n2 == 0:
            return 0
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0
        
        return (mean1 - mean2) / pooled_std

