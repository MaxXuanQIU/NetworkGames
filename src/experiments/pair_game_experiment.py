"""
Experiment 1: Two-player game experiment
16x16 MBTI personality matrix behavior analysis in repeated Prisoner's Dilemma
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import json
from tqdm import tqdm
import random

from src.agents.mbti_personalities import MBTIType, MBTIPersonality, get_all_mbti_types
from src.agents.bfi_personalities import (
    BFIPersonality,
    get_all_bfi_profiles,
    get_bfi_profile_by_name,
)
from src.llm.llm_interface import LLMManager, LLMFactory, LLMProvider
from src.games.two_player_game import GameFactory, Action, GameHistory
from src.analysis.statistics import CooperationAnalyzer, PersonalityAnalyzer
from src.visualization.plotter import PairGamePlotter
from src.config.config_manager import ExperimentConfig


class PairGameExperiment:
    """Two-player game experiment class"""
    
    def __init__(self, config: ExperimentConfig, llm_manager: LLMManager):
        self.config = config
        self.llm_manager = llm_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        # Use the actual model name from config as the key (e.g., "qwen2.5", "llama3")
        self.model_name = config.llm.model_name
        
        # Initialize components
        payoff_matrix = self._parse_payoff_matrix(config.game.payoff_matrix) if config.game.payoff_matrix else None
        self.game = GameFactory.create_game(config.game.game_name, payoff_matrix)
        self.cooperation_analyzer = CooperationAnalyzer()
        self.personality_analyzer = PersonalityAnalyzer()
        self.plotter = PairGamePlotter(
            style=config.visualization_config.get("style", "seaborn-v0_8"),
            figsize=tuple(config.visualization_config.get("figsize", [12, 8])),
            dpi=config.visualization_config.get("dpi", 300),
            color_palette=config.visualization_config.get("color_palette", "Set2")
        )
        
        # Get all MBTI types or BFI profiles
        pair_config = config.pair_game_config
        agent_framework = pair_config.agent_framework.lower() if pair_config else "mbti"
        requested_types = pair_config.participant_types if pair_config else None

        if agent_framework == "bfi":
            available_profiles = get_all_bfi_profiles()
            self.mbti_types = (
                [get_bfi_profile_by_name(name) for name in requested_types]
                if requested_types
                else available_profiles
            )
            self.personalities = {
                profile: BFIPersonality(profile)
                for profile in self.mbti_types
            }
        elif agent_framework == "mbti":
            self.mbti_types = get_all_mbti_types()
            self.personalities = {
                mbti_type: MBTIPersonality(mbti_type)
                for mbti_type in self.mbti_types
            }
        else:
            raise ValueError(
                f"Unsupported pair_game_config.agent_framework: {agent_framework}. "
                "Expected 'mbti' or 'bfi'."
            )

        # Set random seed
        if config.game.random_seed is not None:
            random.seed(config.game.random_seed)
            np.random.seed(config.game.random_seed)
    
    async def run_experiment(self) -> Dict[str, Any]:
        """Run experiment"""
        self.logger.info("Starting pair game experiment...")
        
        # Create result directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Run 16x16 matrix experiment
        matrix_results = await self._run_matrix_experiment()
        
        # Analyze results
        analysis_results = self._analyze_results(matrix_results)
        
        # Generate visualizations
        visualization_results = self._generate_visualizations(matrix_results, analysis_results)
        
        # Save results
        self._save_results(matrix_results, analysis_results, visualization_results)
        
        self.logger.info("Pair game experiment completed!")
        
        return {
            "matrix_results": matrix_results,
            "analysis_results": analysis_results,
            "visualization_results": visualization_results
        }
    
    async def _run_matrix_experiment(self) -> Dict[str, Any]:
        """Run a participant-by-participant matrix experiment."""
        num_types = len(self.mbti_types)
        self.logger.info("Running %d x %d matrix experiment...", num_types, num_types)

        # Initialize result matrices
        cooperation_matrix = np.zeros((num_types, num_types))
        payoff_matrix = np.zeros((num_types, num_types))
        std_matrix = np.zeros((num_types, num_types))

        # Store detailed results
        detailed_results = {}

        # Includes self-play: n(n + 1)/2 unique unordered matchups.
        pbar = tqdm(
            total=num_types * (num_types + 1) // 2,
            desc="Running matrix experiment",
        )

        # Define semaphore for limiting concurrency
        semaphore = asyncio.Semaphore(self.config.llm.concurrency)

        # Construct all game tasks
        tasks = []
        pair_indices = []
        for i, player1_type in enumerate(self.mbti_types):
            for j, player2_type in enumerate(self.mbti_types):
                if j < i:
                    continue  # Only run each unique pair once
                pair_indices.append((i, j, player1_type, player2_type))

        async def run_and_collect(i, j, player1_type, player2_type):
            cooperation_rates_1, cooperation_rates_2 = [], []
            total_payoffs_1, total_payoffs_2 = [], []
            for rep in range(self.config.game.num_repetitions):
                async with semaphore:
                    history = await self._run_single_game(player1_type, player2_type)
                cooperation_rates_1.append(history.player1_cooperation_rate)
                cooperation_rates_2.append(history.player2_cooperation_rate)
                total_payoffs_1.append(history.player1_total_payoff)
                total_payoffs_2.append(history.player2_total_payoff)
            # Statistics
            mean_coop_1 = np.mean(cooperation_rates_1)
            std_coop_1 = np.std(cooperation_rates_1)
            mean_payoff_1 = np.mean(total_payoffs_1)
            mean_coop_2 = np.mean(cooperation_rates_2)
            std_coop_2 = np.std(cooperation_rates_2)
            mean_payoff_2 = np.mean(total_payoffs_2)
            # Return results
            return (i, j, player1_type, player2_type,
                    mean_coop_1, std_coop_1, mean_payoff_1, cooperation_rates_1, total_payoffs_1,
                    mean_coop_2, std_coop_2, mean_payoff_2, cooperation_rates_2, total_payoffs_2)

        # Launch all pair games concurrently
        for i, j, player1_type, player2_type in pair_indices:
            tasks.append(run_and_collect(i, j, player1_type, player2_type))

        # Collect results in order
        for fut in asyncio.as_completed(tasks):
            res = await fut
            (i, j, player1_type, player2_type,
             mean_coop_1, std_coop_1, mean_payoff_1, cooperation_rates_1, total_payoffs_1,
             mean_coop_2, std_coop_2, mean_payoff_2, cooperation_rates_2, total_payoffs_2) = res

            cooperation_matrix[i, j] = mean_coop_1
            payoff_matrix[i, j] = mean_payoff_1
            std_matrix[i, j] = std_coop_1
            cooperation_matrix[j, i] = mean_coop_2
            payoff_matrix[j, i] = mean_payoff_2
            std_matrix[j, i] = std_coop_2

            # Store detailed results for both directions
            detailed_results[f"{player1_type.value}_{player2_type.value}"] = {
                "player1_type": player1_type.value,
                "player2_type": player2_type.value,
                "cooperation_rates": cooperation_rates_1,
                "payoffs": total_payoffs_1,
                "mean_cooperation": mean_coop_1,
                "std_cooperation": std_coop_1,
                "mean_payoff": mean_payoff_1,
                "std_payoff": np.std(total_payoffs_1)
            }
            detailed_results[f"{player2_type.value}_{player1_type.value}"] = {
                "player1_type": player2_type.value,
                "player2_type": player1_type.value,
                "cooperation_rates": cooperation_rates_2,
                "payoffs": total_payoffs_2,
                "mean_cooperation": mean_coop_2,
                "std_cooperation": std_coop_2,
                "mean_payoff": mean_payoff_2,
                "std_payoff": np.std(total_payoffs_2)
            }
            pbar.update(1)
        pbar.close()
        return {
            "cooperation_matrix": cooperation_matrix,
            "payoff_matrix": payoff_matrix,
            "std_matrix": std_matrix,
            "detailed_results": detailed_results,
            "personality_types": [t.value for t in self.mbti_types]
        }
    
    async def _run_single_game(self, player1_type: MBTIType, player2_type: MBTIType) -> GameHistory:
        """Run a single game"""
        # Get personality objects
        personality1 = self.personalities[player1_type]
        personality2 = self.personalities[player2_type]
        
        # Initialize game history
        history = GameHistory(
            results=[],
            total_rounds=0,
            player1_total_payoff=0.0,
            player2_total_payoff=0.0,
            player1_cooperation_rate=0.0,
            player2_cooperation_rate=0.0
        )
        
        game_rules = self.game.get_rules_description()
        
        # Run multiple rounds
        for round_num in range(1, self.config.game.num_rounds + 1):
            # Generate decision prompt
            # pass prompt composition options from config.prompt_config
            pc = self.config.prompt_config
            prompt1 = personality1.get_decision_prompt(
                history.results,
                player2_type.value,
                is_player1=True,
                neighbor_stats=None,
                game_rules=game_rules, # Pass rules
                # Ablation settings
                include_history=pc.include_history,
                include_opponent_type=pc.include_opponent_type,
                include_neighbor_stats=pc.include_neighbor_stats,
                personality_strength=pc.personality_injection,
            )
            prompt2 = personality2.get_decision_prompt(
                history.results,
                player1_type.value,
                is_player1=False,
                neighbor_stats=None,
                game_rules=game_rules, # Pass rules
                # Ablation settings
                include_history=pc.include_history,
                include_opponent_type=pc.include_opponent_type,
                include_neighbor_stats=pc.include_neighbor_stats,
                personality_strength=pc.personality_injection
            )
            # Get actions concurrently
            action1, action2 = await asyncio.gather(
                self._get_llm_action(prompt1, "player1"),
                self._get_llm_action(prompt2, "player2")
            )
            result = self.game.play_round(action1, action2, round_num)
            history.add_result(result)

        return history
    
    async def _get_llm_action(self, prompt: str, player_name: str, max_retries: int = 10) -> Action:
        """Get LLM action with automatic retry on parse failure"""
        for attempt in range(max_retries):
            response = await self.llm_manager.generate_response(self.model_name, prompt, **self.config.llm.kwargs)

            if response.is_cooperate():
                return Action.COOPERATE
            if response.is_defect():
                return Action.DEFECT
            
            content = response.content[:12] + "..." if len(response.content) > 12 else response.content
            self.logger.warning(
                f"Parse {player_name} action failed (attempt {attempt+1}): Unable to parse '{content}'. Retrying..."
            )
            sleep_time = 2 * attempt
            await asyncio.sleep(sleep_time)
        raise ValueError(f"Failed to parse LLM response for {player_name} after {max_retries} attempts.")
 
    def _parse_payoff_matrix(self, matrix_cfg: dict) -> dict:
        """Convert yaml-configured payoff_matrix to internal format"""
        result = {}
        for a1_str, row in matrix_cfg.items():
            for a2_str, payoff in row.items():
                result[(Action[a1_str], Action[a2_str])] = tuple(payoff)
        return result
    
    def _analyze_results(self, matrix_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experiment results."""
        self.logger.info("Analyzing results...")

        cooperation_matrix = matrix_results["cooperation_matrix"]
        detailed_results = matrix_results["detailed_results"]

        results = {
            "basic_statistics": self._calculate_basic_statistics(cooperation_matrix),
            "personality_analysis": self._analyze_personalities(detailed_results),
        }

        if self.config.pair_game_config.agent_framework.lower() == "mbti":
            results["dimension_analysis"] = self._analyze_mbti_dimensions(
                detailed_results
            )

        return results

    def _calculate_basic_statistics(self, cooperation_matrix: np.ndarray) -> Dict[str, Any]:
        """Calculate basic statistics"""
        return {
            "overall_mean": np.mean(cooperation_matrix),
            "overall_std": np.std(cooperation_matrix),
            "min_cooperation": np.min(cooperation_matrix),
            "max_cooperation": np.max(cooperation_matrix),
            "median_cooperation": np.median(cooperation_matrix),
            "q25": np.percentile(cooperation_matrix, 25),
            "q75": np.percentile(cooperation_matrix, 75)
        }
    
    def _analyze_personalities(self, detailed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze personality traits (calls statistics analyzer)"""
        # Construct data containers
        cooperation_data = {ptype.value: [] for ptype in self.mbti_types}
        payoff_data = {ptype.value: [] for ptype in self.mbti_types}

        for result in detailed_results.values():
            ptype = result["player1_type"]
            cooperation_data[ptype].extend(result.get("cooperation_rates", []))
            payoff_data[ptype].extend(result.get("payoffs", []))

        # Use analyzer for both cooperation and payoff ranking
        analysis = self.personality_analyzer._analyze_personality_types(cooperation_data, payoff_data)
        return analysis

    def _analyze_mbti_dimensions(self, detailed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze MBTI dimensions (calls statistics analyzer)"""
        # Initialize containers
        personality_data = {ptype.value: [] for ptype in self.mbti_types}
        payoff_data = {ptype.value: [] for ptype in self.mbti_types}
        
        for result in detailed_results.values():
            p1_type = result["player1_type"]
            
            # detailed_results contains both directions (A vs B, and B vs A separately)
            # So we only need to collect player1's data from each entry
            # Extend with the list of rates from all repetitions
            personality_data[p1_type].extend(result["cooperation_rates"])
            
            # Extend with the list of total payoffs from all repetitions
            # Note: This uses total payoff per game.
            payoff_data[p1_type].extend(result["payoffs"])
            
        return self.personality_analyzer._analyze_mbti_dimensions(personality_data, payoff_data)
    
    def _generate_visualizations(self, matrix_results: Dict[str, Any], 
                               analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate visualizations"""
        self.logger.info("Generating visualizations...")
        
        visualization_files = {}
        
        # Cooperation rate heatmap
        heatmap_file = self.plotter.plot_cooperation_heatmap(
            matrix_results["cooperation_matrix"],
            matrix_results["personality_types"],
            title="MBTI Cooperation Rate Matrix",
            filename="cooperation_heatmap"
        )
        visualization_files["heatmap"] = heatmap_file

        # Payoff heatmap
        payoff_heatmap_file = self.plotter.plot_payoff_heatmap(
            matrix_results["payoff_matrix"],
            matrix_results["personality_types"],
            title="MBTI Payoff Matrix",
            filename="payoff_heatmap"
        )
        visualization_files["payoff_heatmap"] = payoff_heatmap_file
        
        # Cooperation rate distribution
        all_rates = matrix_results["cooperation_matrix"].flatten()
        distribution_file = self.plotter.plot_cooperation_distribution(
            all_rates,
            title="Cooperation Rate Distribution",
            filename="cooperation_distribution"
        )
        visualization_files["distribution"] = distribution_file
        
        # Payoff distribution
        all_payoffs = matrix_results["payoff_matrix"].flatten()
        payoff_distribution_file = self.plotter.plot_payoff_distribution(
            all_payoffs,
            title="Payoff Distribution",
            filename="payoff_distribution"
        )
        visualization_files["payoff_distribution"] = payoff_distribution_file
        
        # Personality cooperation ranking
        personality_cooperation_rates = analysis_results["personality_analysis"]["cooperation_rates"]
        ranking_file = self.plotter.plot_cooperation_ranking(
            personality_cooperation_rates,
            title="Personality Cooperation Ranking",
            filename="personality_cooperation_ranking"
        )
        visualization_files["cooperation_ranking"] = ranking_file
        
        # Payoff ranking
        payoff_rates = analysis_results["personality_analysis"]["payoff_rates"]
        payoff_ranking_file = self.plotter.plot_payoff_ranking(
            payoff_rates,
            title="Personality Payoff Ranking",
            filename="personality_payoff_ranking"
        )
        visualization_files["payoff_ranking"] = payoff_ranking_file

        # MBTI dimension analysis (Cooperation & Payoff)
        if "dimension_analysis" in analysis_results:
            dimension_data_all = analysis_results["dimension_analysis"]

            # 1. Cooperation Dimension Analysis
            dimension_file_coop = self.plotter.plot_mbti_dimension_analysis(
                dimension_data_all["cooperation"],
                title="MBTI Dimension Analysis - Cooperation",
                filename="mbti_dimension_analysis_coop",
                ylabel="Cooperation Rate",
            )
            visualization_files["dimensions_coop"] = dimension_file_coop

            # 2. Payoff Dimension Analysis
            dimension_file_payoff = self.plotter.plot_mbti_dimension_analysis(
                dimension_data_all["payoff"],
                title="MBTI Dimension Analysis - Payoff",
                filename="mbti_dimension_analysis_payoff",
                ylabel="Average Payoff",
            )
            visualization_files["dimensions_payoff"] = dimension_file_payoff
        
        return visualization_files
    
    def _save_results(self, matrix_results: Dict[str, Any], 
                     analysis_results: Dict[str, Any],
                     visualization_results: Dict[str, str]):
        """Save results"""
        self.logger.info("Saving results...")
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save cooperation rate matrix
        matrix_df = pd.DataFrame(
            matrix_results["cooperation_matrix"],
            index=matrix_results["personality_types"],
            columns=matrix_results["personality_types"]
        )
        matrix_df.to_csv(output_dir / "cooperation_matrix.csv")

        # Save payoff matrix
        payoff_df = pd.DataFrame(
            matrix_results["payoff_matrix"],
            index=matrix_results["personality_types"],
            columns=matrix_results["personality_types"]
        )
        payoff_df.to_csv(output_dir / "payoff_matrix.csv")
        
        # Save detailed results
        detailed_df = pd.DataFrame([
            {
                "player1_type": result["player1_type"],
                "player2_type": result["player2_type"],
                "mean_cooperation": result["mean_cooperation"],
                "std_cooperation": result["std_cooperation"],
                "mean_payoff": result["mean_payoff"],
                "std_payoff": result["std_payoff"]
            }
            for result in matrix_results["detailed_results"].values()
        ])
        detailed_df.to_csv(output_dir / "detailed_results.csv", index=False)
        
        # Save analysis results
        with open(output_dir / "analysis_results.json", 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save experiment config
        config_dict = {
            "experiment_type": self.config.experiment_type.value,
            "num_rounds": self.config.game.num_rounds,
            "num_repetitions": self.config.game.num_repetitions,
            "random_seed": self.config.game.random_seed,
            "llm_provider": self.config.llm.provider,
            "llm_model": self.config.llm.model_name
        }
        with open(output_dir / "experiment_config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to: {output_dir}")


async def run_pair_game_experiment(config_file: str = "configs/pair_game.yaml"):
    """Main function to run two-player game experiment"""
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
    llm_manager.add_llm(config.llm.model_name, llm)
    
    # Run experiment
    experiment = PairGameExperiment(config, llm_manager)
    results = await experiment.run_experiment()
    
    return results


if __name__ == "__main__":
    # Set logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING) # Reduce httpx logging noise
    
    # Run experiment
    asyncio.run(run_pair_game_experiment())
