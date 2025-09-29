"""
Visualization module
Provides various charts and visualization functions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from pathlib import Path
import warnings
from matplotlib.patches import Patch
warnings.filterwarnings('ignore')


class BasePlotter:
    """Base plotting class"""
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8), 
                 dpi: int = 300, color_palette: str = "Set2"):
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        self.color_palette = color_palette
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set matplotlib style
        plt.style.use(style)
        sns.set_palette(color_palette)
    
    def save_plot(self, filename: str, output_dir: str = "results"):
        """Save plot"""
        output_path = Path(output_dir) / "visualizations"
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / f"{filename}.png"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        self.logger.info(f"Plot saved to: {filepath}")
        
        # Also save as PDF
        pdf_path = output_path / f"{filename}.pdf"
        plt.savefig(pdf_path, bbox_inches='tight')
        
        return filepath


class PairGamePlotter(BasePlotter):
    """Two-player game visualization"""
    
    def plot_cooperation_heatmap(self, cooperation_matrix: np.ndarray, 
                                personality_types: List[str], 
                                title: str = "MBTI Cooperation Rate Matrix",
                                filename: str = "cooperation_heatmap") -> str:
        """Plot cooperation rate heatmap"""
        plt.figure(figsize=self.figsize)
        
        # Create heatmap, set annotation font size
        sns.heatmap(cooperation_matrix, 
                   xticklabels=personality_types,
                   yticklabels=personality_types,
                   annot=True, 
                   fmt='.3f',
                   annot_kws={"size": 8},
                   cmap='RdYlBu_r',
                   center=0.5,
                   square=True,
                   cbar_kws={'label': 'Cooperation Rate'})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Opponent Personality Type', fontsize=12)
        plt.ylabel('Player Personality Type', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return str(self.save_plot(filename))
    
    def plot_payoff_heatmap(self, payoff_matrix: np.ndarray,
                            personality_types: List[str],
                            title: str = "MBTI Payoff Matrix",
                            filename: str = "payoff_heatmap") -> str:
        """Plot payoff heatmap"""
        plt.figure(figsize=self.figsize)

        sns.heatmap(payoff_matrix,
                    xticklabels=personality_types,
                    yticklabels=personality_types,
                    annot=True,
                    fmt='.2f',
                    annot_kws={"size": 8},
                    cmap='YlGnBu',
                    center=np.mean(payoff_matrix),
                    square=True,
                    cbar_kws={'label': 'Average Payoff'})

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Opponent Personality Type', fontsize=12)
        plt.ylabel('Player Personality Type', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        return str(self.save_plot(filename))
    
    def plot_cooperation_distribution(self, cooperation_rates: List[float],
                                    title: str = "Cooperation Rate Distribution",
                                    filename: str = "cooperation_distribution") -> str:
        """Plot cooperation rate distribution"""
        plt.figure(figsize=self.figsize)
        
        plt.hist(cooperation_rates, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(cooperation_rates), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(cooperation_rates):.3f}')
        plt.axvline(np.median(cooperation_rates), color='green', linestyle='--', 
                   label=f'Median: {np.median(cooperation_rates):.3f}')
        
        plt.xlabel('Cooperation Rate')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return str(self.save_plot(filename))
    
    def plot_personality_ranking(self, personality_rates: Dict[str, Dict[str, float]],
                               title: str = "Personality Cooperation Ranking",
                               filename: str = "personality_ranking") -> str:
        """Plot personality cooperation rate ranking (colored by MBTI type)"""
        # MBTI type to color mapping (adjust as needed)
        mbti_color_map = {
            "NT": "#8e44ad",   # Purple
            "NF": "#27ae60",   # Green
            "SJ": "#2980b9",   # Blue
            "SP": "#e67e22",   # Orange
        }
        # MBTI type to four groups mapping
        def get_mbti_group(mbti: str) -> str:
            if mbti[1:3] == "NT":
                return "NT"
            elif mbti[1:3] == "NF":
                return "NF"
            elif mbti[1] == "S" and mbti[3] == "J":
                return "SJ"
            elif mbti[1] == "S" and mbti[3] == "P":
                return "SP"
            else:
                return "Other"

        # Extract cooperation rate, sample size, and std
        data = [
            (ptype, pdata.get('cooperation_rate', 0), pdata.get('std', 0))
            for ptype, pdata in personality_rates.items()
        ]
        # Sort by cooperation rate
        data_sorted = sorted(data, key=lambda x: x[1], reverse=True)
        personalities, rates, stds = zip(*data_sorted)

        # Color by MBTI group
        bar_colors = []
        for p in personalities:
            group = get_mbti_group(p)
            bar_colors.append(mbti_color_map.get(group, "#95a5a6"))  # Default gray

        plt.figure(figsize=self.figsize)

        bars = plt.bar(range(len(personalities)), rates, 
                       yerr=stds, capsize=8, ecolor='black',
                       error_kw=dict(lw=1, capthick=1),  # Error bar length and thickness
                       color=bar_colors)

        # Add value labels and sample size
        for i, (bar, rate) in enumerate(zip(bars, rates)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{rate:.3f}', ha='center', va='bottom', fontsize=9)

        plt.xlabel('Personality Type')
        plt.ylabel('Cooperation Rate')
        plt.title(title)
        plt.xticks(range(len(personalities)), personalities, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        # Add legend to the side (right)
        legend_elements = [
            Patch(facecolor=color, label=group)
            for group, color in mbti_color_map.items()
        ]
        plt.legend(handles=legend_elements, title="MBTI Group", loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)

        plt.tight_layout()
        return str(self.save_plot(filename))

    def plot_payoff_ranking(self, payoff_rates: Dict[str, Dict[str, float]],
                            title: str = "Personality Payoff Ranking",
                            filename: str = "personality_payoff_ranking") -> str:
        """Plot personality payoff ranking"""
        # MBTI type to color mapping (as same as cooperation ranking)
        mbti_color_map = {
            "NT": "#8e44ad",
            "NF": "#27ae60",
            "SJ": "#2980b9",
            "SP": "#e67e22",
        }
        def get_mbti_group(mbti: str) -> str:
            if mbti[1:3] == "NT":
                return "NT"
            elif mbti[1:3] == "NF":
                return "NF"
            elif mbti[1] == "S" and mbti[3] == "J":
                return "SJ"
            elif mbti[1] == "S" and mbti[3] == "P":
                return "SP"
            else:
                return "Other"

        # Extract payoff, sample size, std
        data = [
            (ptype, pdata.get('payoff', 0), pdata.get('std', 0))
            for ptype, pdata in payoff_rates.items()
        ]
        data_sorted = sorted(data, key=lambda x: x[1], reverse=True)
        personalities, payoffs, stds = zip(*data_sorted)

        bar_colors = []
        for p in personalities:
            group = get_mbti_group(p)
            bar_colors.append(mbti_color_map.get(group, "#95a5a6"))

        plt.figure(figsize=self.figsize)
        bars = plt.bar(range(len(personalities)), payoffs,
                       yerr=stds, capsize=8, ecolor='black',
                       error_kw=dict(lw=1, capthick=1),
                       color=bar_colors)
        for i, (bar, payoff) in enumerate(zip(bars, payoffs)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{payoff:.2f}', ha='center', va='bottom', fontsize=9)
        plt.xlabel('Personality Type')
        plt.ylabel('Average Payoff')
        plt.title(title)
        plt.xticks(range(len(personalities)), personalities, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        legend_elements = [
            Patch(facecolor=color, label=group)
            for group, color in mbti_color_map.items()
        ]
        plt.legend(handles=legend_elements, title="MBTI Group", loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
        plt.tight_layout()
        return str(self.save_plot(filename))

    def plot_mbti_dimension_analysis(self, dimension_data: Dict[str, Dict[str, float]],
                                   title: str = "MBTI Dimension Analysis",
                                   filename: str = "mbti_dimension_analysis") -> str:
        """Plot MBTI dimension analysis"""
        # Assign unique color for each dimension group
        group_colors = {
            'E': '#1f77b4',  # Blue
            'I': '#ff7f0e',  # Orange
            'S': '#2ca02c',  # Green
            'N': '#d62728',  # Red
            'T': '#9467bd',  # Purple
            'F': '#8c564b',  # Brown
            'J': '#e377c2',  # Pink
            'P': '#7f7f7f',  # Gray
        }

        fig, axes = plt.subplots(2, 2, figsize=(5, 10))
        axes = axes.flatten()

        for i, (dim, data) in enumerate(dimension_data.items()):
            if i >= 4:
                break

            ax = axes[i]

            # Extract mean and std
            means = [(k.replace('_mean', ''), v) for k, v in data.items() if k.endswith('_mean')]
            stds = [(k.replace('_std', ''), v) for k, v in data.items() if k.endswith('_std')]
            if len(means) == 2 and len(stds) == 2:
                labels, rates = zip(*means)
                _, errors = zip(*stds)
                x = np.arange(len(labels))

                # Assign color for each group
                bar_colors = [group_colors.get(label, "#cccccc") for label in labels]

                bars = ax.bar(x, rates, yerr=errors, capsize=8, ecolor='black',
                              error_kw=dict(lw=1.2, capthick=1.2), color=bar_colors)

                # Add value labels
                for bar, rate in zip(bars, rates):
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,
                        f'{rate:.3f}',
                        ha='center',
                        va='bottom',
                        zorder=10  # Set higher zorder so label is above other elements
                    )

                # Set x-axis labels
                ax.set_xticks(x)
                ax.set_xticklabels(labels)

                # Add statistics
                diff = data.get('difference', None)
                t_stat = data.get('t_statistic', None)
                p_val = data.get('p_value', None)
                stat_note = (
                    f"Δ={diff:.3f}, t={t_stat:.2f}, p={p_val:.3f}"
                )
                dim_title = dim.replace('_', ' ')
                ax.set_title(f'{dim_title} Dimension\n{stat_note}', fontsize=12)
                ax.set_ylabel('Cooperation Rate')
                ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        return str(self.save_plot(filename))
    
    def plot_statistical_significance(self, test_results: Dict[str, Any],
                                    title: str = "Statistical Significance",
                                    filename: str = "statistical_significance") -> str:
        """Plot statistical significance"""
        plt.figure(figsize=self.figsize)
        
        # Extract p-values
        test_names = list(test_results.keys())
        p_values = [test_results[name].get('p_value', 1.0) for name in test_names]
        
        # Create bar chart
        bars = plt.bar(test_names, p_values, color=['red' if p < 0.05 else 'green' for p in p_values])
        
        # Add significance lines
        plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
        plt.axhline(y=0.01, color='darkred', linestyle='--', alpha=0.7, label='α = 0.01')
        
        # Add value labels
        for bar, p_val in zip(bars, p_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{p_val:.3f}', ha='center', va='bottom')
        
        plt.xlabel('Statistical Test')
        plt.ylabel('p-value')
        plt.title(title)
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        return str(self.save_plot(filename))


class NetworkGamePlotter(BasePlotter):
    """Network game visualization"""
    
    def plot_network_evolution(self, evolution_data: List[Dict[str, Any]],
                          title: str = "Network Evolution",
                          filename: str = "network_evolution") -> str:
        """Plot network evolution"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Extract time series data
        rounds = list(range(len(evolution_data)))
        cooperation_rates = [data.get('cooperation_rate', 0) for data in evolution_data]
        avg_payoffs = [data.get('avg_payoff', 0) for data in evolution_data]
        std_payoffs = [data.get('std_payoff', 0) for data in evolution_data]
        cooperation_cluster_sizes = [
            max([len(cluster) for cluster in data.get('cooperation_clusters', [])], default=0)
            for data in evolution_data
        ]
        single_cooperation_rates = [data.get('single_cooperation_rate', 0) for data in evolution_data]
        both_defect_rates = [data.get('both_defect_rate', 0) for data in evolution_data]

        # 1. Cooperation rate evolution
        axes[0, 0].plot(rounds, cooperation_rates, marker='o', color='tab:blue')
        axes[0, 0].set_title("Cooperation Rate Evolution")
        axes[0, 0].set_xlabel("Round")
        axes[0, 0].set_ylabel("Cooperation Rate")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Average payoff evolution
        axes[0, 1].plot(rounds, avg_payoffs, marker='o', color='tab:green')
        axes[0, 1].set_title("Average Payoff Evolution")
        axes[0, 1].set_xlabel("Round")
        axes[0, 1].set_ylabel("Average Payoff")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Payoff std evolution
        axes[0, 2].plot(rounds, std_payoffs, marker='o', color='tab:purple')
        axes[0, 2].set_title("Payoff Std Evolution")
        axes[0, 2].set_xlabel("Round")
        axes[0, 2].set_ylabel("Payoff Std")
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Max cooperation cluster size evolution
        axes[1, 0].plot(rounds, cooperation_cluster_sizes, marker='o', color='tab:orange')
        axes[1, 0].set_title("Max Cooperation Cluster Size Evolution")
        axes[1, 0].set_xlabel("Round")
        axes[1, 0].set_ylabel("Max Cluster Size")
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Single cooperation rate evolution
        axes[1, 1].plot(rounds, single_cooperation_rates, marker='o', color='tab:red')
        axes[1, 1].set_title("Single Cooperation Rate Evolution")
        axes[1, 1].set_xlabel("Round")
        axes[1, 1].set_ylabel("Single Cooperation Rate")
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Both-defect rates evolution
        axes[1, 2].plot(rounds, both_defect_rates, marker='o', color='tab:brown')
        axes[1, 2].set_title("Both-Defect Rates Evolution")
        axes[1, 2].set_xlabel("Round")
        axes[1, 2].set_ylabel("Both-Defect Rates")
        axes[1, 2].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return str(self.save_plot(filename))
    
    def plot_network_snapshot(self, G, personality_assignment, title="", filename="", legend_labels=None, edge_actions=None):
        """Plot network snapshot, node color by personality type, edge color by action"""
        # MBTI type color scheme
        color_palette = [
            '#9467bd',  # INTJ
            '#8c6bb1',  # INTP
            '#7b4173',  # ENTJ
            '#c5b0d5',  # ENTP
            '#2ca02c',  # INFJ
            '#98df8a',  # INFP
            '#17becf',  # ENFJ
            '#bcbd22',  # ENFP
            '#1f77b4',  # ISTJ
            '#aec7e8',  # ISFJ
            "#3556a1",  # ESTJ
            '#5dade2',  # ESFJ
            '#ffbb78',  # ISTP
            '#ffd700',  # ISFP
            '#ffeb99',  # ESTP
            '#ffe066',  # ESFP
        ]
        from src.agents.mbti_personalities import get_all_mbti_types
        mbti_types = list(get_all_mbti_types())
        mbti_color_map = {mbti_type.value: color_palette[i % len(color_palette)] for i, mbti_type in enumerate(mbti_types)}
        node_colors = [
            mbti_color_map[personality_assignment[node].value]
            for node in G.nodes()
        ]
        legend_labels = {mbti_type.value: mbti_color_map[mbti_type.value] for mbti_type in mbti_types}

        plt.figure(figsize=self.figsize)
        pos = nx.spring_layout(G, seed=42)

        # Edge color handling
        edge_color_list = []
        if edge_actions:
            for u, v in G.edges():
                key1 = f"{u}_{v}"
                key2 = f"{v}_{u}"
                action_tuple = edge_actions.get(key1) or edge_actions.get(key2)
                if action_tuple:
                    a1, a2 = action_tuple
                    # Both cooperate: green; both defect: red; one cooperate one defect: orange
                    if a1 == "COOPERATE" and a2 == "COOPERATE":
                        edge_color_list.append("green")
                    elif a1 == "DEFECT" and a2 == "DEFECT":
                        edge_color_list.append("red")
                    else:
                        edge_color_list.append("orange")
                else:
                    edge_color_list.append("gray")
        else:
            edge_color_list = ["gray"] * G.number_of_edges()

        nx.draw(G, pos,
                node_color=node_colors,
                node_size=300,
                edge_color=edge_color_list,
                width=2,
                alpha=0.8,
                with_labels=True,
                font_size=8)

        plt.title(title)
        plt.axis('off')
        plt.tight_layout()

        # Add legend
        if legend_labels:
            legend_elements = [
                Patch(facecolor=color, label=label)
                for label, color in legend_labels.items()
            ]
            plt.legend(handles=legend_elements, title="Personality Type", loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)

        # Edge action legend
        edge_legend = [
            Patch(facecolor="green", edgecolor="green", label="Both Cooperate"),
            Patch(facecolor="orange", edgecolor="orange", label="One Cooperate, One Defect"),
            Patch(facecolor="red", edgecolor="red", label="Both Defect"),
            Patch(facecolor="gray", edgecolor="gray", label="No Data"),
        ]
        plt.legend(handles=legend_elements + edge_legend, title="Legend", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)

        return str(self.save_plot(filename))
    
    def plot_cooperation_clusters(self, cluster_data: List[Dict[str, Any]],
                                title: str = "Cooperation Clusters",
                                filename: str = "cooperation_clusters") -> str:
        """Plot cooperation cluster analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Cluster size distribution
        cluster_sizes = []
        for data in cluster_data:
            clusters = data.get('cooperation_clusters', [])
            cluster_sizes.extend([len(cluster) for cluster in clusters])
        
        if cluster_sizes:
            axes[0].hist(cluster_sizes, bins=20, alpha=0.7, edgecolor='black')
            axes[0].set_title('Cluster Size Distribution')
            axes[0].set_xlabel('Cluster Size')
            axes[0].set_ylabel('Frequency')
            axes[0].grid(True, alpha=0.3)
        
        # Cluster count evolution
        cluster_counts = [len(data.get('cooperation_clusters', [])) for data in cluster_data]
        rounds = list(range(len(cluster_data)))
        
        axes[1].plot(rounds, cluster_counts, 'b-', linewidth=2, marker='o', markersize=4)
        axes[1].set_title('Number of Clusters Over Time')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Number of Clusters')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return str(self.save_plot(filename))
    
    def plot_network_comparison(self, network_comparison: Dict[str, Dict[str, Any]],
                              title: str = "Network Comparison",
                              filename: str = "network_comparison") -> str:
        """Plot network comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(9, 12))
        
        # Extract data
        network_names = list(network_comparison.keys())
        cooperation_rates = [network_comparison[name].get('avg_final_cooperation_rate', 0) for name in network_names]
        clustering_coeffs = [network_comparison[name].get('clustering_coefficient', 0) for name in network_names]
        avg_path_lengths = [network_comparison[name].get('avg_path_length', 0) for name in network_names]
        densities = [network_comparison[name].get('density', 0) for name in network_names]
        num_nodes = [network_comparison[name].get('num_nodes', 0) for name in network_names]
        num_edges = [network_comparison[name].get('num_edges', 0) for name in network_names]
        
        # Cooperation rate comparison
        axes[0, 0].bar(network_names, cooperation_rates, color='skyblue')
        axes[0, 0].set_title('Final Cooperation Rate by Network Type')
        axes[0, 0].set_ylabel('Cooperation Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Clustering coefficient comparison
        axes[0, 1].bar(network_names, clustering_coeffs, color='lightcoral')
        axes[0, 1].set_title('Clustering Coefficient by Network Type')
        axes[0, 1].set_ylabel('Clustering Coefficient')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Average path length comparison
        axes[0, 2].bar(network_names, avg_path_lengths, color='lightgreen')
        axes[0, 2].set_title('Average Path Length by Network Type')
        axes[0, 2].set_ylabel('Average Path Length')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # Density comparison
        axes[1, 0].bar(network_names, densities, color='gold')
        axes[1, 0].set_title('Network Density by Network Type')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Number of nodes comparison
        axes[1, 1].bar(network_names, num_nodes, color='deepskyblue')
        axes[1, 1].set_title('Number of Nodes by Network Type')
        axes[1, 1].set_ylabel('Num Nodes')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Number of edges comparison
        axes[1, 2].bar(network_names, num_edges, color='salmon')
        axes[1, 2].set_title('Number of Edges by Network Type')
        axes[1, 2].set_ylabel('Num Edges')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return str(self.save_plot(filename))


class InteractivePlotter:
    """Interactive visualization (using Plotly)"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_interactive_heatmap(self, cooperation_matrix: np.ndarray, 
                                 personality_types: List[str],
                                 title: str = "Interactive MBTI Cooperation Matrix") -> go.Figure:
        """Create interactive heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=cooperation_matrix,
            x=personality_types,
            y=personality_types,
            colorscale='RdYlBu_r',
            hoverongaps=False,
            text=np.round(cooperation_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Opponent Personality Type",
            yaxis_title="Player Personality Type",
            width=800,
            height=800
        )
        
        return fig
    
    def create_interactive_network(self, G: nx.Graph, node_colors: List[str],
                                 title: str = "Interactive Network") -> go.Figure:
        """Create interactive network graph"""
        pos = nx.spring_layout(G, seed=42)
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=[f'Node {node}' for node in G.nodes()],
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                color=node_colors,
                size=10,
                colorbar=dict(
                    thickness=15,
                    xanchor="left",
                    titleside="right"
                )
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title,
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Interactive network visualization",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="black", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        return fig
    
    def save_interactive_plot(self, fig: go.Figure, filename: str, output_dir: str = "results"):
        """Save interactive plot"""
        output_path = Path(output_dir) / "visualizations"
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / f"{filename}.html"
        fig.write_html(str(filepath))
        self.logger.info(f"Interactive plot saved to: {filepath}")
        
        return filepath


class RadarPlotter:
    """Radar chart plotter"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def plot_personality_radar(self, personality_data: Dict[str, Dict[str, float]],
                             title: str = "Personality Radar Chart",
                             filename: str = "personality_radar") -> str:
        """Plot personality radar chart"""
        # Define radar chart dimensions
        dimensions = ['Cooperation Rate', 'Stability', 'Responsiveness', 'Clustering', 'Efficiency']
        
        # Create radar chart for each personality type
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw=dict(projection='polar'))
        axes = axes.flatten()
        
        personalities = list(personality_data.keys())[:4]  # Show only first 4
        
        for i, personality in enumerate(personalities):
            if i >= 4:
                break
                
            ax = axes[i]
            data = personality_data[personality]
            
            # Extract data (adjust according to actual data structure)
            values = [
                data.get('cooperation_rate', 0),
                data.get('stability', 0),
                data.get('responsiveness', 0),
                data.get('clustering', 0),
                data.get('efficiency', 0)
            ]
            
            # Ensure data length matches
            values = values[:len(dimensions)]
            while len(values) < len(dimensions):
                values.append(0)
            
            # Close the data
            values += values[:1]
            angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=personality)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(dimensions)
            ax.set_ylim(0, 1)
            ax.set_title(personality, fontweight='bold')
            ax.grid(True)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return self.save_plot(filename)
    
    def save_plot(self, filename: str, output_dir: str = "results") -> str:
        """Save plot"""
        output_path = Path(output_dir) / "visualizations"
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / f"{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        self.logger.info(f"Plot saved to: {filepath}")
        
        return str(filepath)
