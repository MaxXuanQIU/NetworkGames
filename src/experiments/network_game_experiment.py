"""
实验2：网络博弈实验
不同网络拓扑和人格分布下的网络博弈行为分析
"""

import asyncio
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import json
from tqdm import tqdm
import random
from collections import defaultdict
from scipy.stats import linregress

from src.agents.mbti_personalities import MBTIType, MBTIPersonality, get_all_mbti_types
from src.llm.llm_interface import LLMManager, LLMFactory
from src.games.prisoners_dilemma import PrisonersDilemma, Action, GameHistory
from src.networks.network_generator import NetworkGenerator, NetworkConfig, NetworkType, NetworkAnalyzer
from src.analysis.statistics import CooperationAnalyzer, NetworkAnalyzer as NetworkStatsAnalyzer
from src.visualization.plotter import NetworkGamePlotter, InteractivePlotter
from src.config.config_manager import ExperimentConfig


class NetworkGameExperiment:
    """网络博弈实验类"""
    
    def __init__(self, config: ExperimentConfig, llm_manager: LLMManager):
        self.config = config
        self.llm_manager = llm_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化组件
        self.game = PrisonersDilemma()
        self.network_generator = NetworkGenerator()
        self.network_analyzer = NetworkAnalyzer()
        self.cooperation_analyzer = CooperationAnalyzer()
        self.network_stats_analyzer = NetworkStatsAnalyzer()
        self.plotter = NetworkGamePlotter(
            style=config.visualization_config.get("style", "seaborn-v0_8"),
            figsize=tuple(config.visualization_config.get("figsize", [12, 8])),
            dpi=config.visualization_config.get("dpi", 300),
            color_palette=config.visualization_config.get("color_palette", "Set2")
        )
        self.interactive_plotter = InteractivePlotter()
        
        # 获取所有MBTI类型
        self.mbti_types = get_all_mbti_types()
        self.personalities = {mbti_type: MBTIPersonality(mbti_type) for mbti_type in self.mbti_types}
        
        # 设置随机种子
        if config.game.random_seed is not None:
            random.seed(config.game.random_seed)
            np.random.seed(config.game.random_seed)
    
    async def run_experiment(self) -> Dict[str, Any]:
        """运行实验"""
        self.logger.info("Starting network game experiment...")
        
        # 创建结果目录
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 获取实验配置
        network_config = self.config.network_game_config
        network_types = network_config.get("network_types", ["small_world"])
        personality_scenarios = network_config.get("personality_scenarios", ["uniform"])
        
        # 运行不同网络类型的实验
        all_results = {}
        
        for network_type in network_types:
            self.logger.info(f"Running experiments for network type: {network_type}")
            network_results = {}
            
            for scenario in personality_scenarios:
                self.logger.info(f"Running scenario: {scenario}")
                scenario_results = await self._run_network_scenario(network_type, scenario)
                network_results[scenario] = scenario_results
            
            all_results[network_type] = network_results
        
        # 分析结果
        analysis_results = self._analyze_network_results(all_results)
        
        # 生成可视化
        visualization_results = self._generate_network_visualizations(all_results, analysis_results)
        
        # 保存结果
        self._save_network_results(all_results, analysis_results, visualization_results)
        
        self.logger.info("Network game experiment completed!")
        
        return {
            "network_results": all_results,
            "analysis_results": analysis_results,
            "visualization_results": visualization_results
        }
    
    async def _run_network_scenario(self, network_type: str, scenario: str) -> Dict[str, Any]:
        """运行单个网络场景"""
        # 生成网络
        network_config = self._get_network_config(network_type)
        G = self.network_generator.generate_network(network_config)
        
        # 分配人格类型
        personality_assignment = self._assign_personalities(G, scenario)
        
        # 运行网络博弈
        evolution_data = await self._run_network_game(G, personality_assignment)
        
        # 分析网络演化
        network_analysis = self.network_analyzer.analyze_network(G)
        evolution_analysis = self.network_stats_analyzer.analyze_network_evolution(evolution_data)
        
        return {
            "network": G,
            "personality_assignment": personality_assignment,
            "evolution_data": evolution_data,
            "network_analysis": network_analysis,
            "evolution_analysis": evolution_analysis
        }
    
    def _get_network_config(self, network_type: str) -> NetworkConfig:
        """获取网络配置"""
        base_config = NetworkConfig(
            network_type=NetworkType.SMALL_WORLD,
            num_nodes=self.config.network.num_nodes,
            k=self.config.network.k,
            p=self.config.network.p,
            seed=self.config.network.seed
        )
        
        if network_type == "regular":
            base_config.network_type = NetworkType.REGULAR
        elif network_type == "small_world_0.1":
            base_config.network_type = NetworkType.SMALL_WORLD
            base_config.p = 0.1
        elif network_type == "small_world_0.5":
            base_config.network_type = NetworkType.SMALL_WORLD
            base_config.p = 0.5
        elif network_type == "random":
            base_config.network_type = NetworkType.RANDOM
            base_config.edge_probability = 0.1
        elif network_type == "scale_free":
            base_config.network_type = NetworkType.SCALE_FREE
            base_config.m = 2
        
        return base_config
    
    def _assign_personalities(self, G: nx.Graph, scenario: str) -> Dict[int, MBTIType]:
        """分配人格类型"""
        nodes = list(G.nodes())
        personality_assignment = {}
        
        if scenario == "uniform":
            # 均匀随机分布
            for node in nodes:
                personality_assignment[node] = random.choice(self.mbti_types)
        
        elif scenario == "single_ENTJ":
            # 所有人都是ENTJ
            for node in nodes:
                personality_assignment[node] = MBTIType.ENTJ
        
        elif scenario == "clustered":
            # 聚类分布
            personality_assignment = self._assign_clustered_personalities(G)
        
        else:
            # 默认均匀分布
            for node in nodes:
                personality_assignment[node] = random.choice(self.mbti_types)
        
        return personality_assignment
    
    def _assign_clustered_personalities(self, G: nx.Graph) -> Dict[int, MBTIType]:
        """分配聚类人格分布"""
        # 使用社区检测算法
        try:
            communities = nx.community.greedy_modularity_communities(G)
        except:
            # 如果社区检测失败，使用简单的聚类
            communities = [list(G.nodes())]
        
        personality_assignment = {}
        
        # 为每个社区分配主导人格类型
        dominant_personalities = random.sample(list(self.mbti_types), len(communities))
        
        for i, community in enumerate(communities):
            dominant_personality = dominant_personalities[i % len(dominant_personalities)]
            
            for node in community:
                # 80%概率分配主导人格，20%概率分配其他人格
                if random.random() < 0.8:
                    personality_assignment[node] = dominant_personality
                else:
                    personality_assignment[node] = random.choice(self.mbti_types)
        
        return personality_assignment
    
    async def _run_network_game(self, G: nx.Graph, personality_assignment: Dict[int, MBTIType]) -> List[Dict[str, Any]]:
        """运行网络博弈"""
        evolution_data = []
        
        # 初始化节点状态
        node_actions = {node: Action.COOPERATE for node in G.nodes()}
        node_payoffs = {node: 0.0 for node in G.nodes()}
        
        # 运行多轮博弈
        for round_num in range(1, self.config.game.num_rounds + 1):
            self.logger.info(f"Running round {round_num}/{self.config.game.num_rounds}")
            
            # 为每个节点生成决策
            new_actions = {}
            
            for node in G.nodes():
                # 获取邻居信息
                neighbors = list(G.neighbors(node))
                if not neighbors:
                    new_actions[node] = Action.COOPERATE
                    continue
                
                # 生成决策prompt
                personality = self.personalities[personality_assignment[node]]
                prompt = self._generate_network_prompt(personality, node, neighbors, 
                                                     node_actions, round_num)
                
                # 获取LLM决策
                response = await self.llm_manager.generate_response("default", prompt)
                new_actions[node] = self._parse_action(response.content)
            
            # 更新节点动作
            node_actions = new_actions
            
            # 计算收益
            node_payoffs = self._calculate_network_payoffs(G, node_actions)
            
            # 记录演化数据
            round_data = self._record_round_data(G, node_actions, node_payoffs, 
                                               personality_assignment, round_num)
            evolution_data.append(round_data)
        
        return evolution_data
    
    def _parse_action(self, response: str) -> Action:
        """解析LLM响应为博弈动作（与两人博弈保持一致）"""
        import random
        if not isinstance(response, str):
            return Action.COOPERATE if random.random() < 0.5 else Action.DEFECT
        text = response.strip().upper()
        if "COOPERATE" in text or "合作" in text:
            return Action.COOPERATE
        if "DEFECT" in text or "背叛" in text:
            return Action.DEFECT
        # 无法解析时，报错
        raise ValueError(f"无法解析LLM响应: {response}")

    def _generate_network_prompt(self, personality: MBTIPersonality, node: int, 
                               neighbors: List[int], node_actions: Dict[int, Action], 
                               round_num: int) -> str:
        """生成网络博弈的决策prompt"""
        # 获取邻居的历史行为
        neighbor_actions = [node_actions.get(neighbor, Action.COOPERATE) for neighbor in neighbors]
        cooperation_count = sum(1 for action in neighbor_actions if action == Action.COOPERATE)
        total_neighbors = len(neighbors)
        
        # 生成基础prompt
        base_prompt = personality.prompt_template
        
        # 添加网络信息
        network_info = f"""
网络博弈信息：
- 当前节点: {node}
- 邻居数量: {total_neighbors}
- 邻居合作数量: {cooperation_count}
- 邻居背叛数量: {total_neighbors - cooperation_count}
- 当前轮次: {round_num}

请基于以上信息做出你的决策：合作(COOPERATE)还是背叛(DEFECT)？
请只回答COOPERATE或DEFECT，不要解释原因。"""
        
        return base_prompt + network_info
    
    def _calculate_network_payoffs(self, G: nx.Graph, node_actions: Dict[int, Action]) -> Dict[int, float]:
        """计算网络收益"""
        node_payoffs = {node: 0.0 for node in G.nodes()}
        
        for node in G.nodes():
            total_payoff = 0.0
            neighbors = list(G.neighbors(node))
            
            for neighbor in neighbors:
                # 与每个邻居进行囚徒困境博弈
                payoff1, payoff2 = self.game.calculate_payoff(
                    node_actions[node], node_actions[neighbor]
                )
                total_payoff += payoff1
            
            node_payoffs[node] = total_payoff
        
        return node_payoffs
    
    def _record_round_data(self, G: nx.Graph, node_actions: Dict[int, Action], 
                          node_payoffs: Dict[int, float], personality_assignment: Dict[int, MBTIType],
                          round_num: int) -> Dict[str, Any]:
        """记录轮次数据"""
        # 计算合作率
        cooperation_count = sum(1 for action in node_actions.values() if action == Action.COOPERATE)
        cooperation_rate = cooperation_count / len(node_actions)
        
        # 计算网络指标
        network_analysis = self.network_analyzer.analyze_network(G)
        
        # 按人格类型统计
        personality_stats = defaultdict(lambda: {"cooperation_count": 0, "total_count": 0})
        for node, action in node_actions.items():
            personality = personality_assignment[node]
            personality_stats[personality.value]["total_count"] += 1
            if action == Action.COOPERATE:
                personality_stats[personality.value]["cooperation_count"] += 1
        
        # 计算合作集群
        cooperation_clusters = self._find_cooperation_clusters(G, node_actions)
        
        return {
            "round": round_num,
            "cooperation_rate": cooperation_rate,
            "cooperation_count": cooperation_count,
            "total_nodes": len(node_actions),
            "avg_payoff": np.mean(list(node_payoffs.values())),
            "std_payoff": np.std(list(node_payoffs.values())),
            "clustering_coefficient": network_analysis.get("clustering_coefficient", 0),
            "avg_path_length": network_analysis.get("avg_path_length", 0),
            "num_components": network_analysis.get("num_components", 1),
            "personality_stats": dict(personality_stats),
            "cooperation_clusters": cooperation_clusters
        }
    
    def _find_cooperation_clusters(self, G: nx.Graph, node_actions: Dict[int, Action]) -> List[List[int]]:
        """查找合作集群"""
        # 创建只包含合作节点的子图
        cooperation_nodes = [node for node, action in node_actions.items() 
                           if action == Action.COOPERATE]
        
        if not cooperation_nodes:
            return []
        
        # 创建子图
        subgraph = G.subgraph(cooperation_nodes)
        
        # 找到连通分量
        clusters = []
        for component in nx.connected_components(subgraph):
            clusters.append(list(component))
        
        return clusters
    
    def _analyze_network_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析网络实验结果"""
        self.logger.info("Analyzing network results...")
        
        analysis_results = {}
        
        # 比较不同网络类型
        network_comparison = self._compare_networks(all_results)
        
        # 比较不同人格场景
        scenario_comparison = self._compare_scenarios(all_results)
        
        # 分析演化模式
        evolution_patterns = self._analyze_evolution_patterns(all_results)
        
        analysis_results = {
            "network_comparison": network_comparison,
            "scenario_comparison": scenario_comparison,
            "evolution_patterns": evolution_patterns
        }
        
        return analysis_results
    
    def _compare_networks(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """比较不同网络类型"""
        network_metrics = {}
        
        for network_type, scenarios in all_results.items():
            # 计算该网络类型下的平均指标
            final_cooperation_rates = []
            final_clustering_coeffs = []
            final_avg_path_lengths = []
            densities = []
            
            for scenario, results in scenarios.items():
                evolution_data = results["evolution_data"]
                G = results["network"]
                if evolution_data:
                    final_data = evolution_data[-1]
                    final_cooperation_rates.append(final_data["cooperation_rate"])
                    final_clustering_coeffs.append(final_data["clustering_coefficient"])
                    final_avg_path_lengths.append(final_data["avg_path_length"])
                # 计算网络密度
                if G is not None:
                    densities.append(nx.density(G))
            
            network_metrics[network_type] = {
                "avg_final_cooperation_rate": np.mean(final_cooperation_rates) if final_cooperation_rates else 0,
                "avg_clustering_coefficient": np.mean(final_clustering_coeffs) if final_clustering_coeffs else 0,
                "avg_path_length": np.mean(final_avg_path_lengths) if final_avg_path_lengths else 0,
                "avg_density": np.mean(densities) if densities else 0
            }
        
        return network_metrics
    
    def _compare_scenarios(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """比较不同人格场景"""
        scenario_metrics = {}
        
        # 收集所有场景的数据
        all_scenarios = set()
        for scenarios in all_results.values():
            all_scenarios.update(scenarios.keys())
        
        for scenario in all_scenarios:
            final_cooperation_rates = []
            
            for network_type, scenarios in all_results.items():
                if scenario in scenarios:
                    evolution_data = scenarios[scenario]["evolution_data"]
                    if evolution_data:
                        final_data = evolution_data[-1]
                        final_cooperation_rates.append(final_data["cooperation_rate"])
            
            scenario_metrics[scenario] = {
                "avg_final_cooperation_rate": np.mean(final_cooperation_rates) if final_cooperation_rates else 0,
                "std_final_cooperation_rate": np.std(final_cooperation_rates) if final_cooperation_rates else 0
            }
        
        return scenario_metrics
    
    def _analyze_evolution_patterns(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析演化模式"""
        patterns = {}
        
        for network_type, scenarios in all_results.items():
            for scenario, results in scenarios.items():
                evolution_data = results["evolution_data"]
                if evolution_data:
                    # 提取时间序列
                    cooperation_rates = [data["cooperation_rate"] for data in evolution_data]
                    
                    # 计算趋势
                    x = np.arange(len(cooperation_rates))
                    slope, intercept, r_value, p_value, std_err = linregress(x, cooperation_rates)
                    
                    patterns[f"{network_type}_{scenario}"] = {
                        "initial_cooperation_rate": cooperation_rates[0],
                        "final_cooperation_rate": cooperation_rates[-1],
                        "trend_slope": slope,
                        "trend_r_squared": r_value ** 2,
                        "stability": np.std(cooperation_rates)
                    }
        
        return patterns
    
    def _generate_network_visualizations(self, all_results: Dict[str, Any], 
                                       analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """生成网络可视化"""
        self.logger.info("Generating network visualizations...")
        
        visualization_files = {}
        
        # 网络演化图
        for network_type, scenarios in all_results.items():
            for scenario, results in scenarios.items():
                evolution_data = results["evolution_data"]
                if evolution_data:
                    evolution_file = self.plotter.plot_network_evolution(
                        evolution_data,
                        title=f"Network Evolution: {network_type} - {scenario}",
                        filename=f"network_evolution_{network_type}_{scenario}"
                    )
                    visualization_files[f"evolution_{network_type}_{scenario}"] = evolution_file
        
        # 网络比较图
        network_comparison = analysis_results["network_comparison"]
        comparison_file = self.plotter.plot_network_comparison(
            network_comparison,
            title="Network Type Comparison",
            filename="network_comparison"
        )
        visualization_files["network_comparison"] = comparison_file
        
        # 网络快照
        for network_type, scenarios in all_results.items():
            for scenario, results in scenarios.items():
                G = results["network"]
                personality_assignment = results["personality_assignment"]
                
                # 生成节点颜色（基于人格类型）
                node_colors = [hash(personality_assignment[node].value) % 16 
                             for node in G.nodes()]
                
                snapshot_file = self.plotter.plot_network_snapshot(
                    G, node_colors,
                    title=f"Network Snapshot: {network_type} - {scenario}",
                    filename=f"network_snapshot_{network_type}_{scenario}"
                )
                visualization_files[f"snapshot_{network_type}_{scenario}"] = snapshot_file
        
        return visualization_files
    
    def _save_network_results(self, all_results: Dict[str, Any], 
                            analysis_results: Dict[str, Any],
                            visualization_results: Dict[str, str]):
        """保存网络实验结果"""
        self.logger.info("Saving network results...")
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 保存详细结果
        with open(output_dir / "network_results.json", 'w', encoding='utf-8') as f:
            # 转换networkx图对象为可序列化的格式
            serializable_results = {}
            for network_type, scenarios in all_results.items():
                serializable_results[network_type] = {}
                for scenario, results in scenarios.items():
                    serializable_results[network_type][scenario] = {
                        "evolution_data": results["evolution_data"],
                        "network_analysis": results["network_analysis"],
                        "evolution_analysis": results["evolution_analysis"],
                        "personality_assignment": {str(k): v.value for k, v in results["personality_assignment"].items()}
                    }
            
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # 保存分析结果
        with open(output_dir / "network_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        # 保存CSV格式的汇总数据
        self._save_network_summary_csv(all_results, output_dir)
        
        self.logger.info(f"Network results saved to: {output_dir}")
    
    def _save_network_summary_csv(self, all_results: Dict[str, Any], output_dir: Path):
        """保存网络实验汇总CSV"""
        summary_data = []
        
        for network_type, scenarios in all_results.items():
            for scenario, results in scenarios.items():
                evolution_data = results["evolution_data"]
                if evolution_data:
                    final_data = evolution_data[-1]
                    summary_data.append({
                        "network_type": network_type,
                        "scenario": scenario,
                        "final_cooperation_rate": final_data["cooperation_rate"],
                        "final_clustering_coefficient": final_data["clustering_coefficient"],
                        "final_avg_path_length": final_data["avg_path_length"],
                        "final_avg_payoff": final_data["avg_payoff"],
                        "final_std_payoff": final_data["std_payoff"]
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / "network_summary.csv", index=False)


async def run_network_game_experiment(config_file: str = "configs/network_game.yaml"):
    """运行网络博弈实验的主函数"""
    from src.config.config_manager import ConfigManager
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load_config(config_file)
    
    # 创建LLM管理器
    llm_manager = LLMManager()
    
    # 添加LLM实例
    llm = LLMFactory.create_from_config({
        "provider": config.llm.provider,
        "model_name": config.llm.model_name,
        "api_key": config.llm.api_key,
        "kwargs": config.llm.kwargs
    })
    llm_manager.add_llm("default", llm)
    
    # 运行实验
    experiment = NetworkGameExperiment(config, llm_manager)
    results = await experiment.run_experiment()
    
    return results


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行实验
    asyncio.run(run_network_game_experiment())
