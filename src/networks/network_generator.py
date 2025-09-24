"""
网络拓扑生成器
支持生成规则网络、小世界网络、随机网络等不同类型的网络拓扑
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import random
import logging
from dataclasses import dataclass


class NetworkType(Enum):
    """网络类型"""
    REGULAR = "regular"           # 规则网络
    SMALL_WORLD = "small_world"   # 小世界网络
    RANDOM = "random"             # 随机网络
    SCALE_FREE = "scale_free"     # 无标度网络
    COMPLETE = "complete"         # 完全网络
    STAR = "star"                 # 星形网络
    RING = "ring"                 # 环形网络


@dataclass
class NetworkConfig:
    """网络配置"""
    network_type: NetworkType
    num_nodes: int
    # 规则网络参数
    k: int = 4  # 每个节点的邻居数
    # 小世界网络参数
    p: float = 0.1  # 重连概率
    # 随机网络参数
    edge_probability: float = 0.1  # 边存在概率
    # 无标度网络参数
    m: int = 2  # 新节点连接的边数
    # 其他参数
    seed: Optional[int] = None
    directed: bool = False


class NetworkGenerator:
    """网络生成器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_network(self, config: NetworkConfig) -> nx.Graph:
        """根据配置生成网络"""
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
        
        if config.network_type == NetworkType.REGULAR:
            return self._generate_regular_network(config)
        elif config.network_type == NetworkType.SMALL_WORLD:
            return self._generate_small_world_network(config)
        elif config.network_type == NetworkType.RANDOM:
            return self._generate_random_network(config)
        elif config.network_type == NetworkType.SCALE_FREE:
            return self._generate_scale_free_network(config)
        elif config.network_type == NetworkType.COMPLETE:
            return self._generate_complete_network(config)
        elif config.network_type == NetworkType.STAR:
            return self._generate_star_network(config)
        elif config.network_type == NetworkType.RING:
            return self._generate_ring_network(config)
        else:
            raise ValueError(f"Unsupported network type: {config.network_type}")
    
    def _generate_regular_network(self, config: NetworkConfig) -> nx.Graph:
        """生成规则网络（k-regular graph）"""
        if config.k >= config.num_nodes:
            raise ValueError(f"k ({config.k}) must be less than num_nodes ({config.num_nodes})")
        
        if config.k % 2 == 1 and config.num_nodes % 2 == 1:
            raise ValueError("Cannot create k-regular graph with odd k and odd num_nodes")
        
        G = nx.random_regular_graph(config.k, config.num_nodes, seed=config.seed)
        return G
    
    def _generate_small_world_network(self, config: NetworkConfig) -> nx.Graph:
        """生成小世界网络（Watts-Strogatz模型）"""
        if config.k >= config.num_nodes:
            raise ValueError(f"k ({config.k}) must be less than num_nodes ({config.num_nodes})")
        
        G = nx.watts_strogatz_graph(
            config.num_nodes, 
            config.k, 
            config.p, 
            seed=config.seed
        )
        return G
    
    def _generate_random_network(self, config: NetworkConfig) -> nx.Graph:
        """生成随机网络（Erdős-Rényi模型）"""
        G = nx.erdos_renyi_graph(
            config.num_nodes, 
            config.edge_probability, 
            seed=config.seed,
            directed=config.directed
        )
        return G
    
    def _generate_scale_free_network(self, config: NetworkConfig) -> nx.Graph:
        """生成无标度网络（Barabási-Albert模型）"""
        if config.m >= config.num_nodes:
            raise ValueError(f"m ({config.m}) must be less than num_nodes ({config.num_nodes})")
        
        G = nx.barabasi_albert_graph(
            config.num_nodes, 
            config.m, 
            seed=config.seed
        )
        return G
    
    def _generate_complete_network(self, config: NetworkConfig) -> nx.Graph:
        """生成完全网络"""
        G = nx.complete_graph(config.num_nodes)
        return G
    
    def _generate_star_network(self, config: NetworkConfig) -> nx.Graph:
        """生成星形网络"""
        G = nx.star_graph(config.num_nodes - 1)
        return G
    
    def _generate_ring_network(self, config: NetworkConfig) -> nx.Graph:
        """生成环形网络"""
        G = nx.cycle_graph(config.num_nodes)
        return G
    
    def generate_multiple_networks(self, configs: List[NetworkConfig]) -> Dict[str, nx.Graph]:
        """生成多个网络"""
        networks = {}
        for i, config in enumerate(configs):
            name = f"{config.network_type.value}_{i}"
            networks[name] = self.generate_network(config)
        return networks


class NetworkAnalyzer:
    """网络分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_network(self, G: nx.Graph) -> Dict[str, Any]:
        """分析网络的基本属性"""
        if G.number_of_nodes() == 0:
            return {}
        
        # 基本统计
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G)
        
        # 连通性
        is_connected = nx.is_connected(G) if not G.is_directed() else nx.is_weakly_connected(G)
        num_components = nx.number_connected_components(G) if not G.is_directed() else nx.number_weakly_connected_components(G)
        
        # 度分布
        degrees = [d for n, d in G.degree()]
        avg_degree = np.mean(degrees)
        degree_std = np.std(degrees)
        max_degree = max(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0
        
        # 聚类系数
        clustering_coeff = nx.average_clustering(G)
        
        # 路径长度（仅对连通图）
        if is_connected:
            avg_path_length = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
        else:
        # 对于非连通图，只计算最大连通分量
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            avg_path_length = nx.average_shortest_path_length(subgraph)
            diameter = nx.diameter(subgraph)
        
        # 中心性
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        # 度中心性统计
        degree_centrality_values = list(degree_centrality.values())
        betweenness_centrality_values = list(betweenness_centrality.values())
        closeness_centrality_values = list(closeness_centrality.values())
        
        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": density,
            "is_connected": is_connected,
            "num_components": num_components,
            "avg_degree": avg_degree,
            "degree_std": degree_std,
            "max_degree": max_degree,
            "min_degree": min_degree,
            "clustering_coefficient": clustering_coeff,
            "avg_path_length": avg_path_length,
            "diameter": diameter,
            "avg_degree_centrality": np.mean(degree_centrality_values),
            "avg_betweenness_centrality": np.mean(betweenness_centrality_values),
            "avg_closeness_centrality": np.mean(closeness_centrality_values),
            "max_degree_centrality": max(degree_centrality_values) if degree_centrality_values else 0,
            "max_betweenness_centrality": max(betweenness_centrality_values) if betweenness_centrality_values else 0,
            "max_closeness_centrality": max(closeness_centrality_values) if closeness_centrality_values else 0
        }
    
    def compare_networks(self, networks: Dict[str, nx.Graph]) -> Dict[str, Any]:
        """比较多个网络"""
        analysis_results = {}
        for name, G in networks.items():
            analysis_results[name] = self.analyze_network(G)
        
        # 计算比较统计
        comparison = {}
        for metric in ["density", "avg_degree", "clustering_coefficient", "avg_path_length"]:
            values = [analysis[metric] for analysis in analysis_results.values() 
                     if analysis[metric] is not None]
            if values:
                comparison[f"{metric}_mean"] = np.mean(values)
                comparison[f"{metric}_std"] = np.std(values)
                comparison[f"{metric}_min"] = min(values)
                comparison[f"{metric}_max"] = max(values)
        
        return {
            "individual_analysis": analysis_results,
            "comparison": comparison
        }
    
    def get_network_characteristics(self, G: nx.Graph) -> Dict[str, Any]:
        """获取网络特征"""
        analysis = self.analyze_network(G)
        
        # 判断网络类型特征
        characteristics = {
            "is_regular": self._is_regular_network(G),
            "is_small_world": self._is_small_world_network(G),
            "is_scale_free": self._is_scale_free_network(G),
            "is_star": self._is_star_network(G),
            "is_ring": self._is_ring_network(G)
        }
        
        return {**analysis, **characteristics}
    
    def _is_regular_network(self, G: nx.Graph) -> bool:
        """判断是否为规则网络"""
        degrees = [d for n, d in G.degree()]
        return len(set(degrees)) == 1 and degrees[0] > 0
    
    def _is_small_world_network(self, G: nx.Graph) -> bool:
        """判断是否为小世界网络（高聚类系数，短路径长度）"""
        if not nx.is_connected(G):
            return False
        
        clustering = nx.average_clustering(G)
        path_length = nx.average_shortest_path_length(G)
        
        # 与随机网络比较
        n = G.number_of_nodes()
        p = nx.density(G)
        random_clustering = p
        random_path_length = np.log(n) / np.log(n * p) if n * p > 1 else float('inf')
        
        return clustering > random_clustering and path_length < random_path_length
    
    def _is_scale_free_network(self, G: nx.Graph) -> bool:
        """判断是否为无标度网络（度分布遵循幂律）"""
        degrees = [d for n, d in G.degree()]
        if len(degrees) < 10:  # 节点太少，无法判断
            return False
        
        # 简单的幂律检验
        degree_counts = {}
        for d in degrees:
            degree_counts[d] = degree_counts.get(d, 0) + 1
        
        # 检查是否存在高度节点
        max_degree = max(degrees)
        avg_degree = np.mean(degrees)
        
        return max_degree > 2 * avg_degree  # 简化的判断条件
    
    def _is_star_network(self, G: nx.Graph) -> bool:
        """判断是否为星形网络"""
        degrees = [d for n, d in G.degree()]
        return len(degrees) > 1 and max(degrees) == len(degrees) - 1 and min(degrees) == 1
    
    def _is_ring_network(self, G: nx.Graph) -> bool:
        """判断是否为环形网络"""
        degrees = [d for n, d in G.degree()]
        return len(set(degrees)) == 1 and degrees[0] == 2


class NetworkVisualizer:
    """网络可视化器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def visualize_network(self, G: nx.Graph, title: str = "Network", 
                         layout: str = "spring", figsize: Tuple[int, int] = (10, 8),
                         node_color: str = "lightblue", edge_color: str = "gray",
                         node_size: int = 300, font_size: int = 10) -> None:
        """可视化网络"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=figsize)
        
        # 选择布局
        if layout == "spring":
            pos = nx.spring_layout(G, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "random":
            pos = nx.random_layout(G, seed=42)
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # 绘制网络
        nx.draw(G, pos, 
                node_color=node_color,
                edge_color=edge_color,
                node_size=node_size,
                font_size=font_size,
                with_labels=True)
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def visualize_degree_distribution(self, G: nx.Graph, title: str = "Degree Distribution") -> None:
        """可视化度分布"""
        import matplotlib.pyplot as plt
        
        degrees = [d for n, d in G.degree()]
        
        plt.figure(figsize=(10, 6))
        plt.hist(degrees, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def visualize_centrality(self, G: nx.Graph, centrality_type: str = "degree",
                           title: str = "Centrality") -> None:
        """可视化中心性"""
        import matplotlib.pyplot as plt
        
        if centrality_type == "degree":
            centrality = nx.degree_centrality(G)
        elif centrality_type == "betweenness":
            centrality = nx.betweenness_centrality(G)
        elif centrality_type == "closeness":
            centrality = nx.closeness_centrality(G)
        else:
            raise ValueError(f"Unsupported centrality type: {centrality_type}")
        
        values = list(centrality.values())
        
        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel(f'{centrality_type.title()} Centrality')
        plt.ylabel('Frequency')
        plt.title(f'{centrality_type.title()} Centrality Distribution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# 预定义的网络配置
PREDEFINED_NETWORKS = {
    "regular_4": NetworkConfig(NetworkType.REGULAR, 50, k=4),
    "small_world_0.1": NetworkConfig(NetworkType.SMALL_WORLD, 50, k=4, p=0.1),
    "small_world_0.5": NetworkConfig(NetworkType.SMALL_WORLD, 50, k=4, p=0.5),
    "random_0.1": NetworkConfig(NetworkType.RANDOM, 50, edge_probability=0.1),
    "scale_free": NetworkConfig(NetworkType.SCALE_FREE, 50, m=2),
    "complete": NetworkConfig(NetworkType.COMPLETE, 20),  # 完全网络节点数不宜太多
    "star": NetworkConfig(NetworkType.STAR, 20),
    "ring": NetworkConfig(NetworkType.RING, 50)
}
