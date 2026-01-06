"""
Experiment 2: Network Game Experiment
Analysis of network game behavior under different network topologies and personality distributions
"""

import asyncio
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import json
import tqdm
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
    """Network Game Experiment Class"""
    
    def __init__(self, config: ExperimentConfig, llm_manager: LLMManager):
        self.config = config
        self.llm_manager = llm_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        payoff_matrix = self._parse_payoff_matrix(config.game.payoff_matrix)
        self.game = PrisonersDilemma(payoff_matrix=payoff_matrix)
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
        
        # Get all MBTI types
        self.mbti_types = get_all_mbti_types()
        self.personalities = {mbti_type: MBTIPersonality(mbti_type) for mbti_type in self.mbti_types}
        
        # Set random seed
        if config.game.random_seed is not None:
            random.seed(config.game.random_seed)
            np.random.seed(config.game.random_seed)
    
    async def run_experiment(self) -> Dict[str, Any]:
        """Run experiment"""
        self.logger.info("Starting network game experiment...")
        
        # Create results directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Get experiment config
        network_game_config = self.config.network_game_config
        network_types = network_game_config.get("network_types")
        personality_scenarios = network_game_config.get("personality_scenarios")
        
        # Run experiments for different network types
        all_results = {}
        
        for network_type in network_types:
            self.logger.info(f"Running experiments for network type: {network_type}")
            network_results = {}
            
            for scenario in personality_scenarios:
                self.logger.info(f"Running scenario: {scenario}")
                scenario_results = await self._run_network_scenario(network_type, scenario)
                network_results[scenario] = scenario_results
            
            all_results[network_type] = network_results
        
        # Analyze results
        analysis_results = self._analyze_network_results(all_results)
        
        # Generate visualizations
        visualization_results = self._generate_network_visualizations(all_results, analysis_results)
        
        # Save results
        self._save_network_results(all_results, analysis_results, visualization_results)
        
        self.logger.info("Network game experiment completed!")
        
        return {
            "network_results": all_results,
            "analysis_results": analysis_results,
            "visualization_results": visualization_results
        }
    
    async def _run_network_scenario(self, network_type: str, scenario: str) -> Dict[str, Any]:
        """Run a single network scenario"""
        # Generate network
        network_config = self._get_network_config(network_type)
        G = self.network_generator.generate_network(network_config)
        
        # Assign personality types
        personality_assignment = self._assign_personalities(G, scenario)
        
        # Run network game
        evolution_data = await self._run_network_game(G, personality_assignment)
        
        # Analyze network evolution
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
        """Get network config"""
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
        else:
            raise ValueError(f"Unknown network type: {network_type}")

        return base_config
    
    def _assign_personalities(self, G: nx.Graph, scenario: str) -> Dict[int, MBTIType]:
        """Assign personality types"""
        nodes = list(G.nodes())
        personality_assignment = {}
        
        if scenario == "uniform":
            # Uniform random distribution
            for node in nodes:
                personality_assignment[node] = random.choice(self.mbti_types)
        
        elif scenario == "single_ENTJ":
            # All are ENTJ
            for node in nodes:
                personality_assignment[node] = MBTIType.ENTJ
        
        elif scenario == "clustered":
            # Clustered distribution
            personality_assignment = self._assign_clustered_personalities(G)

        elif scenario == "dominant_high_degree_ENTJ":
            # High-degree nodes are ENTJ, others random
            personality_assignment = self._assign_high_degree_personality(G, MBTIType.ENTJ, top_n=5)
        
        elif scenario == "dominant_high_degree_ESFJ":
            # High-degree nodes are ESFJ, others random
            personality_assignment = self._assign_high_degree_personality(G, MBTIType.ESFJ, top_n=5)

        else:
            # Default to uniform distribution
            for node in nodes:
                personality_assignment[node] = random.choice(self.mbti_types)
        
        return personality_assignment
    
    def _assign_clustered_personalities(self, G: nx.Graph) -> Dict[int, MBTIType]:
        """Assign clustered personality distribution"""
        # Use community detection algorithm
        communities = nx.community.greedy_modularity_communities(G)
        
        personality_assignment = {}
        
        # Assign dominant personality type to each community
        dominant_personalities = random.sample(list(self.mbti_types), len(communities))
        
        for i, community in enumerate(communities):
            dominant_personality = dominant_personalities[i % len(dominant_personalities)]
            
            for node in community:
                # 80% probability assign dominant personality, 20% assign other personality
                if random.random() < 0.8:
                    personality_assignment[node] = dominant_personality
                else:
                    personality_assignment[node] = random.choice(self.mbti_types)
        
        return personality_assignment
    
    def _assign_high_degree_personality(self, G: nx.Graph, target_type: MBTIType, top_n: int = 5) -> Dict[int, MBTIType]:
        """
        Assign target_type to top_n high-degree nodes, others random.
        """
        personality_assignment = {}
        degrees = sorted(list(G.degree()), key=lambda x: x[1], reverse=True)
        top_nodes = [node for node, _ in degrees[:top_n]]
        for node in top_nodes:
            personality_assignment[node] = target_type
        for node in set(G.nodes()) - set(top_nodes):
            personality_assignment[node] = random.choice(self.mbti_types)
        return personality_assignment
    
    async def _run_network_game(self, G: nx.Graph, personality_assignment: Dict[int, MBTIType]) -> List[Dict[str, Any]]:
        """Run network game (each node makes decisions with each neighbor, considering network context)"""
        evolution_data = []
        # Store history for each node pair: (node, neighbor) -> List[GameResult]
        pair_histories = defaultdict(list)
        node_payoffs = {node: 0.0 for node in G.nodes()}
        num_rounds = self.config.game.num_rounds
        pbar = tqdm.tqdm(range(1, num_rounds + 1), desc="Network game rounds")

        # Create semaphore to limit max concurrency
        semaphore = asyncio.Semaphore(10)

        # Store history of actions of neighbors for context
        node_neighbor_actions = {node: [] for node in G.nodes()}

        for round_num in pbar:
            self.logger.info(f"Running round {round_num}/{num_rounds}")
            round_payoffs = {node: 0.0 for node in G.nodes()}
            round_actions = {node: {} for node in G.nodes()}
            edge_actions = {}

            # Collect all node decision tasks
            node_tasks = []
            node_neighbors = []
            for node in G.nodes():
                neighbors = list(G.neighbors(node))
                personality = self.personalities[personality_assignment[node]]
                node_neighbors.append((node, neighbors, personality))
            
            async def get_actions_for_node(node, neighbors, personality):
                actions = {}
                # Calculate neighbor statistics
                neighbor_stats = {"cooperation_rate": None, "majority_action": None}
                neighbor_actions = []
                for neighbor in neighbors:
                    # Get the neighbor's last action (from the previous round)
                    if node_neighbor_actions[neighbor]:
                        neighbor_actions.append(node_neighbor_actions[neighbor][-1])
                if neighbor_actions:
                    coop_count = sum(1 for a in neighbor_actions if a == Action.COOPERATE)
                    neighbor_stats["cooperation_rate"] = coop_count / len(neighbor_actions)
                    majority_action = Action.COOPERATE if coop_count >= len(neighbor_actions)/2 else Action.DEFECT
                    neighbor_stats["majority_action"] = majority_action.value

                for neighbor in neighbors:
                    pair_key = tuple(sorted((node, neighbor)))
                    history = pair_histories[pair_key]
                    prompt = personality.get_decision_prompt(
                        history,
                        personality_assignment[neighbor].value,
                        is_player1=(node <= neighbor),
                        neighbor_stats=neighbor_stats
                    )
                    async with semaphore:
                        action = await self._get_llm_action(prompt, f"node_{node}")
                    actions[neighbor] = action
                return node, actions

            # Run all node decision tasks concurrently
            for node, neighbors, personality in node_neighbors:
                node_tasks.append(get_actions_for_node(node, neighbors, personality))
            node_results = await asyncio.gather(*node_tasks)

            # Fill round_actions
            for node, actions in node_results:
                round_actions[node] = actions

            # Play all games and accumulate payoffs
            for node in G.nodes():
                for neighbor in G.neighbors(node):
                    if node < neighbor:
                        action1 = round_actions[node][neighbor]
                        action2 = round_actions[neighbor][node]
                        result = self.game.play_round(action1, action2, round_num)
                        round_payoffs[node] += result.player1_payoff
                        round_payoffs[neighbor] += result.player2_payoff
                        pair_key = (node, neighbor)
                        pair_histories[tuple(sorted(pair_key))].append(result)
                        edge_actions[(node, neighbor)] = (action1, action2)
                        # Update neighbor action history for next round context
                        node_neighbor_actions[node].append(action1)
                        node_neighbor_actions[neighbor].append(action2)

            # Record round data
            round_data = self._record_round_data(
                G, 
                round_payoffs,
                personality_assignment,
                round_num,
                edge_actions
            )
            evolution_data.append(round_data)

        return evolution_data

    async def _get_llm_action(self, prompt: str, player_name: str, max_retries: int = 10) -> Action:
        """Get LLM action with automatic retry on parse failure"""
        for attempt in range(max_retries):
            response = await self.llm_manager.generate_response("default", prompt, **self.config.llm.kwargs)
            action = self._parse_action(response.content)
            if action is not None:
                return action
            self.logger.warning(
                f"Parse {player_name} action failed (attempt {attempt+1}): Unable to parse '{response.content}'. Retrying..."
            )
            sleep_time = 2 ** attempt
            await asyncio.sleep(sleep_time)
        raise ValueError(f"Failed to parse LLM response for {player_name} after {max_retries} attempts.")

    def _parse_action(self, response: str) -> Action:
        """Parse LLM response to game action"""
        response = response.strip().upper()
        if "COOPERATE" in response:
            return Action.COOPERATE
        elif "DEFECT" in response:
            return Action.DEFECT
        else:
            return None

    def _parse_payoff_matrix(self, matrix_cfg: dict) -> dict:
        """Convert payoff_matrix from yaml config to internal format"""
        result = {}
        for a1_str, row in matrix_cfg.items():
            for a2_str, payoff in row.items():
                result[(Action[a1_str], Action[a2_str])] = tuple(payoff)
        return result
    
    def _record_round_data(self, G: nx.Graph, 
                          node_payoffs: Dict[int, float], personality_assignment: Dict[int, MBTIType],
                          round_num: int, edge_actions: Optional[Dict[Tuple[int, int], Tuple[Action, Action]]] = None
                          ) -> Dict[str, Any]:
        """Record round data, statistics based on edge_actions"""
        # Statistics for cooperation on edges
        edge_actions = edge_actions or {}
        total_edges = len(edge_actions)
        cooperation_edges = 0
        single_cooperation_edges = 0
        both_defect_edges = 0
        personality_stats = defaultdict(lambda: {"cooperation_count": 0, "total_count": 0})
        for (u, v), (a1, a2) in edge_actions.items():
            # Only count as cooperation edge if both cooperate
            if a1 == Action.COOPERATE and a2 == Action.COOPERATE:
                cooperation_edges += 1
                # Count cooperation edges for each personality (each node +1)
                personality_stats[personality_assignment[u].value]["cooperation_count"] += 1
                personality_stats[personality_assignment[v].value]["cooperation_count"] += 1
            elif (a1 == Action.COOPERATE and a2 == Action.DEFECT) or (a1 == Action.DEFECT and a2 == Action.COOPERATE):
                single_cooperation_edges += 1
            elif a1 == Action.DEFECT and a2 == Action.DEFECT:
                both_defect_edges += 1
            # For every edge, increment total count for each personality
            personality_stats[personality_assignment[u].value]["total_count"] += 1
            personality_stats[personality_assignment[v].value]["total_count"] += 1
        cooperation_rate = cooperation_edges / total_edges if total_edges > 0 else 0
        single_cooperation_rate = single_cooperation_edges / total_edges if total_edges > 0 else 0
        both_defect_rate = both_defect_edges / total_edges if total_edges > 0 else 0
        # Calculate cooperation clusters (only count edges where both cooperate, generate subgraph)
        cooperation_nodes = set()
        for (u, v), (a1, a2) in edge_actions.items():
            if a1 == Action.COOPERATE and a2 == Action.COOPERATE:
                cooperation_nodes.add(u)
                cooperation_nodes.add(v)
        if cooperation_nodes:
            subgraph = G.subgraph(cooperation_nodes)
            cooperation_clusters = [list(comp) for comp in nx.connected_components(subgraph)]
        else:
            cooperation_clusters = []
        return {
            "round": round_num,
            "cooperation_rate": cooperation_rate,
            "cooperation_count": cooperation_edges,
            "single_cooperation_rate": single_cooperation_rate,
            "single_cooperation_count": single_cooperation_edges,
            "both_defect_rate": both_defect_rate,
            "both_defect_count": both_defect_edges,
            "total_edges": total_edges,
            "avg_payoff": np.mean(list(node_payoffs.values())),
            "std_payoff": np.std(list(node_payoffs.values())),
            "personality_stats": dict(personality_stats),
            "cooperation_clusters": cooperation_clusters,
            "edge_actions": {f"{k[0]}_{k[1]}": (v[0].value, v[1].value) for k, v in edge_actions.items()}
        }
    
    def _find_cooperation_clusters(self, G: nx.Graph, node_actions: Dict[int, Action]) -> List[List[int]]:
        """Find cooperation clusters"""
        # Create subgraph containing only cooperation nodes
        cooperation_nodes = [node for node, action in node_actions.items() 
                           if action == Action.COOPERATE]
        
        if not cooperation_nodes:
            return []
        
        # Create subgraph
        subgraph = G.subgraph(cooperation_nodes)
        
        # Find connected components
        clusters = []
        for component in nx.connected_components(subgraph):
            clusters.append(list(component))
        
        return clusters
    
    def _analyze_network_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze network experiment results"""
        self.logger.info("Analyzing network results...")
        
        analysis_results = {}
        
        # Compare different network types
        network_comparison = self._compare_networks(all_results)
        
        # Compare different personality scenarios
        scenario_comparison = self._compare_scenarios(all_results)
        
        # Analyze evolution patterns
        evolution_patterns = self._analyze_evolution_patterns(all_results)
        
        analysis_results = {
            "network_comparison": network_comparison,
            "scenario_comparison": scenario_comparison,
            "evolution_patterns": evolution_patterns
        }
        
        return analysis_results
    
    def _compare_networks(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different network types"""
        network_metrics = {}
        
        for network_type, scenarios in all_results.items():
            # Calculate metrics for this network type
            final_cooperation_rates = []
            all_cooperation_rates = []
            all_avg_payoffs = []
            
            first_scenario = next(iter(scenarios.values()))
            G = first_scenario["network"]
            network_analysis = self.network_analyzer.analyze_network(G) if G is not None else {}
            clustering_coefficient = network_analysis.get("clustering_coefficient", 0)
            avg_path_length = network_analysis.get("avg_path_length", 0)
            num_components = network_analysis.get("num_components", 0)
            density = nx.density(G) if G is not None else 0
            num_nodes = G.number_of_nodes() if G is not None else 0
            num_edges = G.number_of_edges() if G is not None else 0
            avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0
            for scenario, results in scenarios.items():
                evolution_data = results["evolution_data"]
                if evolution_data:
                    final_data = evolution_data[-1]
                    final_cooperation_rates.append(final_data["cooperation_rate"])
                    # Collect all rounds' cooperation rates and avg_payoff
                    all_cooperation_rates.extend([rd["cooperation_rate"] for rd in evolution_data])
                    all_avg_payoffs.extend([rd["avg_payoff"] for rd in evolution_data])
            
            network_metrics[network_type] = {
                "final_cooperation_rate": np.mean(final_cooperation_rates) if final_cooperation_rates else 0,
                "overall_avg_cooperation_rate": np.mean(all_cooperation_rates) if all_cooperation_rates else 0,
                "overall_avg_payoff": np.mean(all_avg_payoffs) if all_avg_payoffs else 0,
                "clustering_coefficient": clustering_coefficient,
                "avg_path_length": avg_path_length,
                "num_components": num_components,
                "density": density,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "avg_degree": avg_degree
            }
        
        return network_metrics
    
    def _compare_scenarios(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different personality scenarios"""
        scenario_metrics = {}
        
        # Collect all scenario data
        all_scenarios = set()
        for scenarios in all_results.values():
            all_scenarios.update(scenarios.keys())
        
        for scenario in all_scenarios:
            final_cooperation_rates = []
            all_cooperation_rates = []
            all_avg_payoffs = []
            
            for network_type, scenarios in all_results.items():
                if scenario in scenarios:
                    evolution_data = scenarios[scenario]["evolution_data"]
                    if evolution_data:
                        final_data = evolution_data[-1]
                        final_cooperation_rates.append(final_data["cooperation_rate"])
                        # Collect all rounds' cooperation rates and avg_payoff
                        all_cooperation_rates.extend([rd["cooperation_rate"] for rd in evolution_data])
                        all_avg_payoffs.extend([rd["avg_payoff"] for rd in evolution_data])
            
            scenario_metrics[scenario] = {
                "avg_final_cooperation_rate": np.mean(final_cooperation_rates) if final_cooperation_rates else 0,
                "overall_avg_cooperation_rate": np.mean(all_cooperation_rates) if all_cooperation_rates else 0,
                "overall_avg_payoff": np.mean(all_avg_payoffs) if all_avg_payoffs else 0
            }
        
        return scenario_metrics
    
    def _analyze_evolution_patterns(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze evolution patterns"""
        patterns = {}
        
        for network_type, scenarios in all_results.items():
            for scenario, results in scenarios.items():
                evolution_data = results["evolution_data"]
                if evolution_data:
                    # Extract time series
                    cooperation_rates = [data["cooperation_rate"] for data in evolution_data]
                    
                    # Calculate trend
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
        """Generate network visualizations"""
        self.logger.info("Generating network visualizations...")
        
        visualization_files = {}
        
        # Network evolution plots
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

        # Edge action evolution plots
        for network_type, scenarios in all_results.items():
            for scenario, results in scenarios.items():
                evolution_data = results["evolution_data"]
                if evolution_data:
                    edge_action_file = self.plotter.plot_edge_action_evolution(
                        evolution_data,
                        title=f"Edge Action Evolution: {network_type} - {scenario}",
                        filename=f"edge_action_evolution_{network_type}_{scenario}"
                    )
                    visualization_files[f"edge_action_evolution_{network_type}_{scenario}"] = edge_action_file
        
        # Network comparison plot
        network_comparison = analysis_results["network_comparison"]
        comparison_file = self.plotter.plot_network_comparison(
            network_comparison,
            title="Network Type Comparison",
            filename="network_comparison"
        )
        visualization_files["network_comparison"] = comparison_file
        
        # Scenarios comparison plot
        scenario_comparison = analysis_results["scenario_comparison"]
        scenario_comparison_file = self.plotter.plot_scenario_comparison(
            scenario_comparison,
            title="Personality Scenario Comparison",
            filename="scenario_comparison"
        )
        visualization_files["scenario_comparison"] = scenario_comparison_file
        
        # Network snapshots
        for network_type, scenarios in all_results.items():
            for scenario, results in scenarios.items():
                G = results["network"]
                personality_assignment = results["personality_assignment"]
                
                # Get edge actions from last round
                evolution_data = results["evolution_data"]
                edge_actions = None
                if evolution_data and "edge_actions" in evolution_data[-1]:
                    edge_actions = evolution_data[-1]["edge_actions"]
                snapshot_file = self.plotter.plot_network_snapshot(
                    G, personality_assignment,
                    edge_actions=edge_actions,
                    title=f"Network Snapshot: {network_type} - {scenario}",
                    filename=f"network_snapshot_{network_type}_{scenario}"
                )
                visualization_files[f"snapshot_{network_type}_{scenario}"] = snapshot_file
        
        return visualization_files
    
    def _save_network_results(self, all_results: Dict[str, Any], 
                            analysis_results: Dict[str, Any],
                            visualization_results: Dict[str, str]):
        """Save network experiment results"""
        self.logger.info("Saving network results...")
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        with open(output_dir / "network_results.json", 'w', encoding='utf-8') as f:
            # Convert networkx graph objects to serializable format
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
        
        # Save analysis results
        with open(output_dir / "network_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        # Save summary data in CSV format
        self._save_network_summary_csv(all_results, output_dir)
        
        self.logger.info(f"Network results saved to: {output_dir}")
    
    def _save_network_summary_csv(self, all_results: Dict[str, Any], output_dir: Path):
        """Save network experiment summary CSV"""
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
                        "final_cooperation_count": final_data["cooperation_count"],
                        "final_single_cooperation_rate": final_data["single_cooperation_rate"],
                        "final_single_cooperation_count": final_data["single_cooperation_count"],
                        "final_both_defect_rate": final_data["both_defect_rate"],
                        "final_both_defect_count": final_data["both_defect_count"],
                        "final_avg_payoff": final_data["avg_payoff"],
                        "final_std_payoff": final_data["std_payoff"]
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / "network_summary.csv", index=False)


async def run_network_game_experiment(config_file: str = "configs/network_game.yaml"):
    """Main function to run the network game experiment"""
    from src.config.config_manager import ConfigManager

    # Load config
    config_manager = ConfigManager()
    config = config_manager.load_config(config_file)
    
    # Create LLM manager
    llm_manager = LLMManager()
    
    # Add LLM instance
    llm = LLMFactory.create_from_config({
        "provider": config.llm.provider,
        "model_name": config.llm.model_name,
        "api_key": config.llm.api_key,
        "kwargs": config.llm.kwargs
    })
    llm_manager.add_llm("default", llm)
    
    # Run experiment
    experiment = NetworkGameExperiment(config, llm_manager)
    results = await experiment.run_experiment()
    
    return results


if __name__ == "__main__":
    # Set logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING) # Reduce httpx logging noise
    
    # Run experiment
    asyncio.run(run_network_game_experiment())

