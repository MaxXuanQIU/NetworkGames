"""
Network topology generator
Supports generating different types of network topologies such as regular networks, small-world networks, random networks, etc.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import random
import logging
from dataclasses import dataclass


class NetworkType(Enum):
    """Network type"""
    REGULAR = "regular"           # Regular network
    SMALL_WORLD = "small_world"   # Small-world network
    RANDOM = "random"             # Random network
    SCALE_FREE = "scale_free"     # Scale-free network
    COMPLETE = "complete"         # Complete network
    STAR = "star"                 # Star network
    RING = "ring"                 # Ring network


@dataclass
class NetworkConfig:
    """Network configuration"""
    network_type: NetworkType
    num_nodes: int
    # Regular network parameter
    k: int = 4  # Number of neighbors per node
    # Small-world network parameter
    p: float = 0.1  # Rewiring probability
    # Random network parameter
    edge_probability: float = 0.1  # Edge existence probability
    # Scale-free network parameter
    m: int = 2  # Number of edges to attach from a new node
    # Other parameters
    seed: Optional[int] = None
    directed: bool = False


class NetworkGenerator:
    """Network generator"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_network(self, config: NetworkConfig) -> nx.Graph:
        """Generate network according to configuration"""
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
        """Generate regular network (k-regular graph)"""
        if config.k >= config.num_nodes:
            raise ValueError(f"k ({config.k}) must be less than num_nodes ({config.num_nodes})")
        
        if config.k % 2 == 1 and config.num_nodes % 2 == 1:
            raise ValueError("Cannot create k-regular graph with odd k and odd num_nodes")
        
        G = nx.random_regular_graph(config.k, config.num_nodes, seed=config.seed)
        return G
    
    def _generate_small_world_network(self, config: NetworkConfig) -> nx.Graph:
        """Generate small-world network (Watts-Strogatz model)"""
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
        """Generate random network (Erdős-Rényi model)"""
        G = nx.erdos_renyi_graph(
            config.num_nodes, 
            config.edge_probability, 
            seed=config.seed,
            directed=config.directed
        )
        return G
    
    def _generate_scale_free_network(self, config: NetworkConfig) -> nx.Graph:
        """Generate scale-free network (Barabási-Albert model)"""
        if config.m >= config.num_nodes:
            raise ValueError(f"m ({config.m}) must be less than num_nodes ({config.num_nodes})")
        
        G = nx.barabasi_albert_graph(
            config.num_nodes, 
            config.m, 
            seed=config.seed
        )
        return G
    
    def _generate_complete_network(self, config: NetworkConfig) -> nx.Graph:
        """Generate complete network"""
        G = nx.complete_graph(config.num_nodes)
        return G
    
    def _generate_star_network(self, config: NetworkConfig) -> nx.Graph:
        """Generate star network"""
        G = nx.star_graph(config.num_nodes - 1)
        return G
    
    def _generate_ring_network(self, config: NetworkConfig) -> nx.Graph:
        """Generate ring network"""
        G = nx.cycle_graph(config.num_nodes)
        return G
    
    def generate_multiple_networks(self, configs: List[NetworkConfig]) -> Dict[str, nx.Graph]:
        """Generate multiple networks"""
        networks = {}
        for i, config in enumerate(configs):
            name = f"{config.network_type.value}_{i}"
            networks[name] = self.generate_network(config)
        return networks


class NetworkAnalyzer:
    """Network analyzer"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_network(self, G: nx.Graph) -> Dict[str, Any]:
        """Analyze basic properties of the network"""
        if G.number_of_nodes() == 0:
            return {}
        
        # Basic statistics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G)
        
        # Connectivity
        is_connected = nx.is_connected(G) if not G.is_directed() else nx.is_weakly_connected(G)
        num_components = nx.number_connected_components(G) if not G.is_directed() else nx.number_weakly_connected_components(G)
        
        # Degree distribution
        degrees = [d for n, d in G.degree()]
        avg_degree = np.mean(degrees)
        degree_std = np.std(degrees)
        max_degree = max(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0
        
        # Clustering coefficient
        clustering_coeff = nx.average_clustering(G)
        
        # Path length (only for connected graphs)
        if is_connected:
            avg_path_length = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
        else:
        # For disconnected graphs, only compute the largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            avg_path_length = nx.average_shortest_path_length(subgraph)
            diameter = nx.diameter(subgraph)
        
        # Centrality
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        # Centrality statistics
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
        """Compare multiple networks"""
        analysis_results = {}
        for name, G in networks.items():
            analysis_results[name] = self.analyze_network(G)
        
        # Compute comparison statistics
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
        """Get network characteristics"""
        analysis = self.analyze_network(G)
        
        # Determine network type characteristics
        characteristics = {
            "is_regular": self._is_regular_network(G),
            "is_small_world": self._is_small_world_network(G),
            "is_scale_free": self._is_scale_free_network(G),
            "is_star": self._is_star_network(G),
            "is_ring": self._is_ring_network(G)
        }
        
        return {**analysis, **characteristics}
    
    def _is_regular_network(self, G: nx.Graph) -> bool:
        """Determine if it is a regular network"""
        degrees = [d for n, d in G.degree()]
        return len(set(degrees)) == 1 and degrees[0] > 0
    
    def _is_small_world_network(self, G: nx.Graph) -> bool:
        """Determine if it is a small-world network (high clustering coefficient, short path length)"""
        if not nx.is_connected(G):
            return False
        
        clustering = nx.average_clustering(G)
        path_length = nx.average_shortest_path_length(G)
        
        # Compare with random network
        n = G.number_of_nodes()
        p = nx.density(G)
        random_clustering = p
        random_path_length = np.log(n) / np.log(n * p) if n * p > 1 else float('inf')
        
        return clustering > random_clustering and path_length < random_path_length
    
    def _is_scale_free_network(self, G: nx.Graph) -> bool:
        """Determine if it is a scale-free network (degree distribution follows power law)"""
        degrees = [d for n, d in G.degree()]
        if len(degrees) < 10:  # Too few nodes to judge
            return False
        
        # Simple power law test
        degree_counts = {}
        for d in degrees:
            degree_counts[d] = degree_counts.get(d, 0) + 1
        
        # Check for high-degree nodes
        max_degree = max(degrees)
        avg_degree = np.mean(degrees)
        
        return max_degree > 2 * avg_degree  # Simplified criterion
    
    def _is_star_network(self, G: nx.Graph) -> bool:
        """Determine if it is a star network"""
        degrees = [d for n, d in G.degree()]
        return len(degrees) > 1 and max(degrees) == len(degrees) - 1 and min(degrees) == 1
    
    def _is_ring_network(self, G: nx.Graph) -> bool:
        """Determine if it is a ring network"""
        degrees = [d for n, d in G.degree()]
        return len(set(degrees)) == 1 and degrees[0] == 2


class NetworkVisualizer:
    """Network visualizer"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def visualize_network(self, G: nx.Graph, title: str = "Network", 
                         layout: str = "spring", figsize: Tuple[int, int] = (10, 8),
                         node_color: str = "lightblue", edge_color: str = "gray",
                         node_size: int = 300, font_size: int = 10) -> None:
        """Visualize network"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=figsize)
        
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(G, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "random":
            pos = nx.random_layout(G, seed=42)
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Draw network
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
        """Visualize degree distribution"""
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
        """Visualize centrality"""
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


# Predefined network configurations
PREDEFINED_NETWORKS = {
    "regular_4": NetworkConfig(NetworkType.REGULAR, num_nodes=50, k=4),
    "small_world_0.1": NetworkConfig(NetworkType.SMALL_WORLD, num_nodes=50, k=4, p=0.1),
    "small_world_0.5": NetworkConfig(NetworkType.SMALL_WORLD, num_nodes=50, k=4, p=0.5),
    "random_0.1": NetworkConfig(NetworkType.RANDOM, num_nodes=50, edge_probability=0.1),
    "scale_free": NetworkConfig(NetworkType.SCALE_FREE, num_nodes=50, m=2),
    "complete": NetworkConfig(NetworkType.COMPLETE, num_nodes=20),  # Too many nodes for complete network is not recommended
    "star": NetworkConfig(NetworkType.STAR, num_nodes=20),
    "ring": NetworkConfig(NetworkType.RING, num_nodes=50)
}
