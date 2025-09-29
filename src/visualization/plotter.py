"""
可视化模块
提供各种图表和可视化功能
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
    """基础绘图类"""
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8), 
                 dpi: int = 300, color_palette: str = "Set2"):
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        self.color_palette = color_palette
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 设置matplotlib样式
        plt.style.use(style)
        sns.set_palette(color_palette)
    
    def save_plot(self, filename: str, output_dir: str = "results"):
        """保存图表"""
        output_path = Path(output_dir) / "visualizations"
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / f"{filename}.png"
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        self.logger.info(f"Plot saved to: {filepath}")
        
        # 同时保存为PDF
        pdf_path = output_path / f"{filename}.pdf"
        plt.savefig(pdf_path, bbox_inches='tight')
        
        return filepath


class PairGamePlotter(BasePlotter):
    """两人博弈可视化"""
    
    def plot_cooperation_heatmap(self, cooperation_matrix: np.ndarray, 
                                personality_types: List[str], 
                                title: str = "MBTI Cooperation Rate Matrix",
                                filename: str = "cooperation_heatmap") -> str:
        """绘制合作率热力图"""
        plt.figure(figsize=self.figsize)
        
        # 创建热力图，设置注释字体大小
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
        """绘制收益热力图"""
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
        """绘制合作率分布"""
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
        """绘制人格合作率排名（根据MBTI类型着色）"""
        # MBTI类型到颜色的映射（可根据需要调整）
        mbti_color_map = {
            "NT": "#8e44ad",   # 紫色
            "NF": "#27ae60",   # 绿色
            "SJ": "#2980b9",   # 蓝色
            "SP": "#e67e22",   # 橙色
        }
        # MBTI类型到四组的映射
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

        # 提取合作率、样本量和标准差
        data = [
            (ptype, pdata.get('cooperation_rate', 0), pdata.get('std', 0))
            for ptype, pdata in personality_rates.items()
        ]
        # 按合作率排序
        data_sorted = sorted(data, key=lambda x: x[1], reverse=True)
        personalities, rates, stds = zip(*data_sorted)

        # 根据MBTI类型分组着色
        bar_colors = []
        for p in personalities:
            group = get_mbti_group(p)
            bar_colors.append(mbti_color_map.get(group, "#95a5a6"))  # 默认灰色

        plt.figure(figsize=self.figsize)

        bars = plt.bar(range(len(personalities)), rates, 
                       yerr=stds, capsize=8, ecolor='black',
                       error_kw=dict(lw=1, capthick=1),  # 横线长度和粗细
                       color=bar_colors)

        # 添加数值标签和样本量
        for i, (bar, rate) in enumerate(zip(bars, rates)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{rate:.3f}', ha='center', va='bottom', fontsize=9)

        plt.xlabel('Personality Type')
        plt.ylabel('Cooperation Rate')
        plt.title(title)
        plt.xticks(range(len(personalities)), personalities, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        # 添加图例到图的旁边（右侧）
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
        """绘制MBTI维度分析（含更多统计信息，每个分组独特颜色）"""
        # 为每个维度分组指定独特颜色
        group_colors = {
            'E': '#1f77b4',  # 蓝色
            'I': '#ff7f0e',  # 橙色
            'S': '#2ca02c',  # 绿色
            'N': '#d62728',  # 红色
            'T': '#9467bd',  # 紫色
            'F': '#8c564b',  # 棕色
            'J': '#e377c2',  # 粉色
            'P': '#7f7f7f',  # 灰色
        }

        fig, axes = plt.subplots(2, 2, figsize=(5, 10))
        axes = axes.flatten()

        for i, (dim, data) in enumerate(dimension_data.items()):
            if i >= 4:
                break

            ax = axes[i]

            # 提取均值和标准差
            means = [(k.replace('_mean', ''), v) for k, v in data.items() if k.endswith('_mean')]
            stds = [(k.replace('_std', ''), v) for k, v in data.items() if k.endswith('_std')]
            if len(means) == 2 and len(stds) == 2:
                labels, rates = zip(*means)
                _, errors = zip(*stds)
                x = np.arange(len(labels))

                # 为每个分组分配颜色
                bar_colors = [group_colors.get(label, "#cccccc") for label in labels]

                bars = ax.bar(x, rates, yerr=errors, capsize=8, ecolor='black',
                              error_kw=dict(lw=1.2, capthick=1.2), color=bar_colors)

                # 添加数值标签
                for bar, rate in zip(bars, rates):
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,
                        f'{rate:.3f}',
                        ha='center',
                        va='bottom',
                        zorder=10  # 设置较高的zorder使标签浮在其他元素之上
                    )

                # 设置x轴标签
                ax.set_xticks(x)
                ax.set_xticklabels(labels)

                # 添加统计信息
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
        """绘制统计显著性"""
        plt.figure(figsize=self.figsize)
        
        # 提取p值
        test_names = list(test_results.keys())
        p_values = [test_results[name].get('p_value', 1.0) for name in test_names]
        
        # 创建条形图
        bars = plt.bar(test_names, p_values, color=['red' if p < 0.05 else 'green' for p in p_values])
        
        # 添加显著性线
        plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
        plt.axhline(y=0.01, color='darkred', linestyle='--', alpha=0.7, label='α = 0.01')
        
        # 添加数值标签
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
    """网络博弈可视化"""
    
    def plot_network_evolution(self, evolution_data: List[Dict[str, Any]],
                          title: str = "Network Evolution",
                          filename: str = "network_evolution") -> str:
        """绘制网络演化"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 提取时间序列数据
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

        # 1. 合作率演化
        axes[0, 0].plot(rounds, cooperation_rates, marker='o', color='tab:blue')
        axes[0, 0].set_title("Cooperation Rate Evolution")
        axes[0, 0].set_xlabel("Round")
        axes[0, 0].set_ylabel("Cooperation Rate")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 平均收益演化
        axes[0, 1].plot(rounds, avg_payoffs, marker='o', color='tab:green')
        axes[0, 1].set_title("Average Payoff Evolution")
        axes[0, 1].set_xlabel("Round")
        axes[0, 1].set_ylabel("Average Payoff")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 收益标准差演化
        axes[0, 2].plot(rounds, std_payoffs, marker='o', color='tab:purple')
        axes[0, 2].set_title("Payoff Std Evolution")
        axes[0, 2].set_xlabel("Round")
        axes[0, 2].set_ylabel("Payoff Std")
        axes[0, 2].grid(True, alpha=0.3)

        # 4. 最大合作集群规模演化
        axes[1, 0].plot(rounds, cooperation_cluster_sizes, marker='o', color='tab:orange')
        axes[1, 0].set_title("Max Cooperation Cluster Size Evolution")
        axes[1, 0].set_xlabel("Round")
        axes[1, 0].set_ylabel("Max Cluster Size")
        axes[1, 0].grid(True, alpha=0.3)

        # 5. 单方合作率演化
        axes[1, 1].plot(rounds, single_cooperation_rates, marker='o', color='tab:red')
        axes[1, 1].set_title("Single Cooperation Rate Evolution")
        axes[1, 1].set_xlabel("Round")
        axes[1, 1].set_ylabel("Single Cooperation Rate")
        axes[1, 1].grid(True, alpha=0.3)

        # 6. 双方背叛边数演化
        axes[1, 2].plot(rounds, both_defect_rates, marker='o', color='tab:brown')
        axes[1, 2].set_title("Both-Defect Rates Evolution")
        axes[1, 2].set_xlabel("Round")
        axes[1, 2].set_ylabel("Both-Defect Rates")
        axes[1, 2].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return str(self.save_plot(filename))
    
    def plot_network_snapshot(self, G, personality_assignment, title="", filename="", legend_labels=None, edge_actions=None):
        """绘制网络快照，节点颜色根据人格类型着色，边颜色根据动作染色"""
        # MBTI类型配色方案
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

        # 边颜色处理
        edge_color_list = []
        if edge_actions:
            for u, v in G.edges():
                key1 = f"{u}_{v}"
                key2 = f"{v}_{u}"
                action_tuple = edge_actions.get(key1) or edge_actions.get(key2)
                if action_tuple:
                    a1, a2 = action_tuple
                    # 如果双方都合作，绿色；双方都背叛，红色；一方合作一方背叛，橙色
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

        # 添加图例
        if legend_labels:
            legend_elements = [
                Patch(facecolor=color, label=label)
                for label, color in legend_labels.items()
            ]
            plt.legend(handles=legend_elements, title="Personality Type", loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)

        # 边动作图例
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
        """绘制合作集群分析"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 集群大小分布
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
        
        # 集群数量演化
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
        """绘制网络比较"""
        fig, axes = plt.subplots(2, 3, figsize=(9, 12))
        
        # 提取数据
        network_names = list(network_comparison.keys())
        cooperation_rates = [network_comparison[name].get('avg_final_cooperation_rate', 0) for name in network_names]
        clustering_coeffs = [network_comparison[name].get('clustering_coefficient', 0) for name in network_names]
        avg_path_lengths = [network_comparison[name].get('avg_path_length', 0) for name in network_names]
        densities = [network_comparison[name].get('density', 0) for name in network_names]
        num_nodes = [network_comparison[name].get('num_nodes', 0) for name in network_names]
        num_edges = [network_comparison[name].get('num_edges', 0) for name in network_names]
        
        # 合作率比较
        axes[0, 0].bar(network_names, cooperation_rates, color='skyblue')
        axes[0, 0].set_title('Final Cooperation Rate by Network Type')
        axes[0, 0].set_ylabel('Cooperation Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 聚类系数比较
        axes[0, 1].bar(network_names, clustering_coeffs, color='lightcoral')
        axes[0, 1].set_title('Clustering Coefficient by Network Type')
        axes[0, 1].set_ylabel('Clustering Coefficient')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 平均路径长度比较
        axes[0, 2].bar(network_names, avg_path_lengths, color='lightgreen')
        axes[0, 2].set_title('Average Path Length by Network Type')
        axes[0, 2].set_ylabel('Average Path Length')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # 密度比较
        axes[1, 0].bar(network_names, densities, color='gold')
        axes[1, 0].set_title('Network Density by Network Type')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 节点数比较
        axes[1, 1].bar(network_names, num_nodes, color='deepskyblue')
        axes[1, 1].set_title('Number of Nodes by Network Type')
        axes[1, 1].set_ylabel('Num Nodes')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 边数比较
        axes[1, 2].bar(network_names, num_edges, color='salmon')
        axes[1, 2].set_title('Number of Edges by Network Type')
        axes[1, 2].set_ylabel('Num Edges')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return str(self.save_plot(filename))


class InteractivePlotter:
    """交互式可视化（使用Plotly）"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_interactive_heatmap(self, cooperation_matrix: np.ndarray, 
                                 personality_types: List[str],
                                 title: str = "Interactive MBTI Cooperation Matrix") -> go.Figure:
        """创建交互式热力图"""
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
        """创建交互式网络图"""
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
        """保存交互式图表"""
        output_path = Path(output_dir) / "visualizations"
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / f"{filename}.html"
        fig.write_html(str(filepath))
        self.logger.info(f"Interactive plot saved to: {filepath}")
        
        return filepath


class RadarPlotter:
    """雷达图绘制器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def plot_personality_radar(self, personality_data: Dict[str, Dict[str, float]],
                             title: str = "Personality Radar Chart",
                             filename: str = "personality_radar") -> str:
        """绘制人格雷达图"""
        # 定义雷达图的维度
        dimensions = ['Cooperation Rate', 'Stability', 'Responsiveness', 'Clustering', 'Efficiency']
        
        # 为每个人格类型创建雷达图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw=dict(projection='polar'))
        axes = axes.flatten()
        
        personalities = list(personality_data.keys())[:4]  # 只显示前4个
        
        for i, personality in enumerate(personalities):
            if i >= 4:
                break
                
            ax = axes[i]
            data = personality_data[personality]
            
            # 提取数据（这里需要根据实际数据结构调整）
            values = [
                data.get('cooperation_rate', 0),
                data.get('stability', 0),
                data.get('responsiveness', 0),
                data.get('clustering', 0),
                data.get('efficiency', 0)
            ]
            
            # 确保数据长度匹配
            values = values[:len(dimensions)]
            while len(values) < len(dimensions):
                values.append(0)
            
            # 闭合数据
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
        """保存图表"""
        output_path = Path(output_dir) / "visualizations"
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / f"{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        self.logger.info(f"Plot saved to: {filepath}")
        
        return str(filepath)
