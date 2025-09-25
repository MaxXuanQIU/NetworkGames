"""
实验1：两人博弈实验
16x16 MBTI人格矩阵在重复囚徒困境中的行为分析
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
from src.llm.llm_interface import LLMManager, LLMFactory, LLMProvider
from src.games.prisoners_dilemma import PrisonersDilemma, Action, GameHistory
from src.analysis.statistics import CooperationAnalyzer, PersonalityAnalyzer
from src.visualization.plotter import PairGamePlotter
from src.config.config_manager import ExperimentConfig


class PairGameExperiment:
    """两人博弈实验类"""
    
    def __init__(self, config: ExperimentConfig, llm_manager: LLMManager):
        self.config = config
        self.llm_manager = llm_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化组件
        self.game = PrisonersDilemma()
        self.cooperation_analyzer = CooperationAnalyzer()
        self.personality_analyzer = PersonalityAnalyzer()
        self.plotter = PairGamePlotter(
            style=config.visualization_config.get("style", "seaborn-v0_8"),
            figsize=tuple(config.visualization_config.get("figsize", [12, 8])),
            dpi=config.visualization_config.get("dpi", 300),
            color_palette=config.visualization_config.get("color_palette", "Set2")
        )
        
        # 获取所有MBTI类型
        self.mbti_types = get_all_mbti_types()
        self.personalities = {mbti_type: MBTIPersonality(mbti_type) for mbti_type in self.mbti_types}
        
        # 设置随机种子
        if config.game.random_seed is not None:
            random.seed(config.game.random_seed)
            np.random.seed(config.game.random_seed)
    
    async def run_experiment(self) -> Dict[str, Any]:
        """运行实验"""
        self.logger.info("Starting pair game experiment...")
        
        # 创建结果目录
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 运行16x16矩阵实验
        matrix_results = await self._run_matrix_experiment()
        
        # 分析结果
        analysis_results = self._analyze_results(matrix_results)
        
        # 生成可视化
        visualization_results = self._generate_visualizations(matrix_results, analysis_results)
        
        # 保存结果
        self._save_results(matrix_results, analysis_results, visualization_results)
        
        self.logger.info("Pair game experiment completed!")
        
        return {
            "matrix_results": matrix_results,
            "analysis_results": analysis_results,
            "visualization_results": visualization_results
        }
    
    async def _run_matrix_experiment(self) -> Dict[str, Any]:
        """运行16x16矩阵实验"""
        self.logger.info("Running 16x16 matrix experiment...")
        
        # 初始化结果矩阵
        cooperation_matrix = np.zeros((16, 16))
        payoff_matrix = np.zeros((16, 16))
        std_matrix = np.zeros((16, 16))
        
        # 存储详细结果
        detailed_results = {}
        
        # 创建进度条
        total_combinations = 16 * 16
        pbar = tqdm(total=total_combinations, desc="Running matrix experiment")
        
        for i, player1_type in enumerate(self.mbti_types):
            for j, player2_type in enumerate(self.mbti_types):
                # 运行重复实验
                cooperation_rates = []
                total_payoffs = []
                
                for rep in range(self.config.game.num_repetitions):
                    # 运行单次实验
                    history = await self._run_single_game(player1_type, player2_type)
                    
                    cooperation_rates.append(history.player1_cooperation_rate)
                    total_payoffs.append(history.player1_total_payoff)
                
                # 计算统计量
                mean_cooperation = np.mean(cooperation_rates)
                std_cooperation = np.std(cooperation_rates)
                mean_payoff = np.mean(total_payoffs)
                
                # 更新矩阵
                cooperation_matrix[i, j] = mean_cooperation
                payoff_matrix[i, j] = mean_payoff
                std_matrix[i, j] = std_cooperation
                
                # 存储详细结果
                detailed_results[f"{player1_type.value}_{player2_type.value}"] = {
                    "player1_type": player1_type.value,
                    "player2_type": player2_type.value,
                    "cooperation_rates": cooperation_rates,
                    "payoffs": total_payoffs,
                    "mean_cooperation": mean_cooperation,
                    "std_cooperation": std_cooperation,
                    "mean_payoff": mean_payoff,
                    "std_payoff": np.std(total_payoffs)
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
        """运行单次博弈"""
        # 获取人格对象
        personality1 = self.personalities[player1_type]
        personality2 = self.personalities[player2_type]
        
        # 初始化博弈历史
        history = GameHistory(
            results=[],
            total_rounds=0,
            player1_total_payoff=0.0,
            player2_total_payoff=0.0,
            player1_cooperation_rate=0.0,
            player2_cooperation_rate=0.0
        )
        
        # 运行多轮博弈
        for round_num in range(1, self.config.game.num_rounds + 1):
            # 生成决策prompt
            prompt1 = personality1.get_decision_prompt(
                history.results, 
                player2_type.value
            )
            prompt2 = personality2.get_decision_prompt(
                history.results, 
                player1_type.value
            )
            
            # 获取LLM决策
            response1 = await self.llm_manager.generate_response(
                "default", prompt1
            )
            response2 = await self.llm_manager.generate_response(
                "default", prompt2
            )
            
            # 解析决策
            action1 = self._parse_action(response1.content)
            action2 = self._parse_action(response2.content)
            
            # 进行博弈
            result = self.game.play_round(action1, action2, round_num)
            history.add_result(result)
        
        return history
    
    def _parse_action(self, response: str) -> Action:
        """解析LLM响应为博弈动作"""
        response = response.strip().upper()
        
        if "COOPERATE" in response:
            return Action.COOPERATE
        elif "DEFECT" in response:
            return Action.DEFECT
        else:
            # 默认策略：基于响应的随机性
            return Action.COOPERATE if random.random() < 0.5 else Action.DEFECT
    
    def _analyze_results(self, matrix_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析实验结果"""
        self.logger.info("Analyzing results...")
        
        cooperation_matrix = matrix_results["cooperation_matrix"]
        detailed_results = matrix_results["detailed_results"]
        
        # 计算基本统计
        basic_stats = self._calculate_basic_statistics(cooperation_matrix)
        
        # 人格分析
        personality_analysis = self._analyze_personalities(detailed_results)
        
        # MBTI维度分析
        dimension_analysis = self._analyze_mbti_dimensions(detailed_results)
        
        # 统计检验
        statistical_tests = self._perform_statistical_tests(detailed_results)
        
        return {
            "basic_statistics": basic_stats,
            "personality_analysis": personality_analysis,
            "dimension_analysis": dimension_analysis,
            "statistical_tests": statistical_tests
        }
    
    def _calculate_basic_statistics(self, cooperation_matrix: np.ndarray) -> Dict[str, Any]:
        """计算基本统计量"""
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
        """分析人格特征（调用统计分析器）"""
        # 构造 personality_data: Dict[str, List[bool]]
        personality_data = {ptype.value: [] for ptype in self.mbti_types}
        for result in detailed_results.values():
            personality_data[result["player1_type"]].extend(result["cooperation_rates"])
        # 调用统一分析器
        return self.personality_analyzer._analyze_personality_types(personality_data)

    def _analyze_mbti_dimensions(self, detailed_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析MBTI维度（调用统计分析器）"""
        personality_data = {ptype.value: [] for ptype in self.mbti_types}
        for result in detailed_results.values():
            personality_data[result["player1_type"]].extend(result["cooperation_rates"])
        return self.personality_analyzer._analyze_mbti_dimensions(personality_data)
    
    def _perform_statistical_tests(self, detailed_results: Dict[str, Any]) -> Dict[str, Any]:
        """执行统计检验"""
        # 这里可以添加更多的统计检验
        return {
            "note": "Statistical tests would be implemented here",
            "tests_performed": []
        }
    
    def _generate_visualizations(self, matrix_results: Dict[str, Any], 
                               analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """生成可视化"""
        self.logger.info("Generating visualizations...")
        
        visualization_files = {}
        
        # 合作率热力图
        heatmap_file = self.plotter.plot_cooperation_heatmap(
            matrix_results["cooperation_matrix"],
            matrix_results["personality_types"],
            title="MBTI Cooperation Rate Matrix",
            filename="cooperation_heatmap"
        )
        visualization_files["heatmap"] = heatmap_file

        # 收益热力图
        payoff_heatmap_file = self.plotter.plot_payoff_heatmap(
            matrix_results["payoff_matrix"],
            matrix_results["personality_types"],
            title="MBTI Payoff Matrix",
            filename="payoff_heatmap"
        )
        visualization_files["payoff_heatmap"] = payoff_heatmap_file
        
        # 合作率分布
        all_rates = matrix_results["cooperation_matrix"].flatten()
        distribution_file = self.plotter.plot_cooperation_distribution(
            all_rates,
            title="Cooperation Rate Distribution",
            filename="cooperation_distribution"
        )
        visualization_files["distribution"] = distribution_file
        
        # 人格排名
        personality_rates = analysis_results["personality_analysis"]["personality_rates"]
        ranking_file = self.plotter.plot_personality_ranking(
            personality_rates,
            title="Personality Cooperation Ranking",
            filename="personality_ranking"
        )
        visualization_files["ranking"] = ranking_file
        
        # MBTI维度分析
        dimension_data = analysis_results["dimension_analysis"]
        dimension_file = self.plotter.plot_mbti_dimension_analysis(
            dimension_data,
            title="MBTI Dimension Analysis",
            filename="mbti_dimension_analysis"
        )
        visualization_files["dimensions"] = dimension_file
        
        return visualization_files
    
    def _save_results(self, matrix_results: Dict[str, Any], 
                     analysis_results: Dict[str, Any],
                     visualization_results: Dict[str, str]):
        """保存结果"""
        self.logger.info("Saving results...")
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 保存合作率矩阵
        matrix_df = pd.DataFrame(
            matrix_results["cooperation_matrix"],
            index=matrix_results["personality_types"],
            columns=matrix_results["personality_types"]
        )
        matrix_df.to_csv(output_dir / "cooperation_matrix.csv")

        # 保存收益矩阵
        payoff_df = pd.DataFrame(
            matrix_results["payoff_matrix"],
            index=matrix_results["personality_types"],
            columns=matrix_results["personality_types"]
        )
        payoff_df.to_csv(output_dir / "payoff_matrix.csv")
        
        # 保存详细结果
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
        
        # 保存分析结果
        with open(output_dir / "analysis_results.json", 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存实验配置
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
    """运行两人博弈实验的主函数"""
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
    experiment = PairGameExperiment(config, llm_manager)
    results = await experiment.run_experiment()
    
    return results


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行实验
    asyncio.run(run_pair_game_experiment())
