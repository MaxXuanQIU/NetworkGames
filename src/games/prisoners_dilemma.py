"""
囚徒困境博弈实现
包含博弈逻辑、收益计算、策略分析等功能
"""

from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging


class Action(Enum):
    """博弈动作"""
    COOPERATE = "COOPERATE"
    DEFECT = "DEFECT"


@dataclass
class GameResult:
    """单轮博弈结果"""
    player1_action: Action
    player2_action: Action
    player1_payoff: float
    player2_payoff: float
    round_number: int


@dataclass
class GameHistory:
    """博弈历史记录"""
    results: List[GameResult]
    total_rounds: int
    player1_total_payoff: float
    player2_total_payoff: float
    player1_cooperation_rate: float
    player2_cooperation_rate: float
    
    def add_result(self, result: GameResult):
        """添加一轮结果"""
        self.results.append(result)
        self.total_rounds += 1
        self.player1_total_payoff += result.player1_payoff
        self.player2_total_payoff += result.player2_payoff
        
        # 更新合作率
        player1_cooperations = sum(1 for r in self.results if r.player1_action == Action.COOPERATE)
        player2_cooperations = sum(1 for r in self.results if r.player2_action == Action.COOPERATE)
        
        self.player1_cooperation_rate = player1_cooperations / self.total_rounds
        self.player2_cooperation_rate = player2_cooperations / self.total_rounds


class PrisonersDilemma:
    """囚徒困境博弈类"""
    
    # 标准囚徒困境收益矩阵
    PAYOFF_MATRIX = {
        (Action.COOPERATE, Action.COOPERATE): (3, 3),  # 双方合作
        (Action.COOPERATE, Action.DEFECT): (0, 5),      # 我合作，对手背叛
        (Action.DEFECT, Action.COOPERATE): (5, 0),      # 我背叛，对手合作
        (Action.DEFECT, Action.DEFECT): (1, 1)          # 双方背叛
    }
    
    def __init__(self, payoff_matrix: Optional[Dict[Tuple[Action, Action], Tuple[float, float]]] = None):
        """
        初始化囚徒困境博弈
        
        Args:
            payoff_matrix: 自定义收益矩阵，格式为 {(action1, action2): (payoff1, payoff2)}
        """
        self.payoff_matrix = payoff_matrix or self.PAYOFF_MATRIX
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 验证收益矩阵
        self._validate_payoff_matrix()
    
    def _validate_payoff_matrix(self):
        """验证收益矩阵的有效性"""
        required_combinations = [
            (Action.COOPERATE, Action.COOPERATE),
            (Action.COOPERATE, Action.DEFECT),
            (Action.DEFECT, Action.COOPERATE),
            (Action.DEFECT, Action.DEFECT)
        ]
        
        for combo in required_combinations:
            if combo not in self.payoff_matrix:
                raise ValueError(f"Missing payoff for action combination: {combo}")
    
    def calculate_payoff(self, action1: Action, action2: Action) -> Tuple[float, float]:
        """计算给定动作组合的收益"""
        return self.payoff_matrix[(action1, action2)]
    
    def play_round(self, action1: Action, action2: Action, round_number: int = 1) -> GameResult:
        """进行一轮博弈"""
        payoff1, payoff2 = self.calculate_payoff(action1, action2)
        
        return GameResult(
            player1_action=action1,
            player2_action=action2,
            player1_payoff=payoff1,
            player2_payoff=payoff2,
            round_number=round_number
        )
    
    def play_game(self, player1_actions: List[Action], player2_actions: List[Action]) -> GameHistory:
        """进行多轮博弈"""
        if len(player1_actions) != len(player2_actions):
            raise ValueError("Players must have the same number of actions")
        
        history = GameHistory(
            results=[],
            total_rounds=0,
            player1_total_payoff=0.0,
            player2_total_payoff=0.0,
            player1_cooperation_rate=0.0,
            player2_cooperation_rate=0.0
        )
        
        for i, (action1, action2) in enumerate(zip(player1_actions, player2_actions), 1):
            result = self.play_round(action1, action2, i)
            history.add_result(result)
        
        return history
    
    def get_nash_equilibrium(self) -> Tuple[Action, Action]:
        """获取纳什均衡（背叛，背叛）"""
        return Action.DEFECT, Action.DEFECT
    
    def get_pareto_optimal(self) -> Tuple[Action, Action]:
        """获取帕累托最优（合作，合作）"""
        return Action.COOPERATE, Action.COOPERATE
    
    def is_dominant_strategy(self, action: Action) -> bool:
        """检查是否为占优策略（背叛是占优策略）"""
        return action == Action.DEFECT
    
    def get_cooperation_incentive(self) -> float:
        """获取合作激励（CC - CD的收益差）"""
        cc_payoff = self.payoff_matrix[(Action.COOPERATE, Action.COOPERATE)][0]
        cd_payoff = self.payoff_matrix[(Action.COOPERATE, Action.DEFECT)][0]
        return cc_payoff - cd_payoff
    
    def get_defection_temptation(self) -> float:
        """获取背叛诱惑（DC - CC的收益差）"""
        dc_payoff = self.payoff_matrix[(Action.DEFECT, Action.COOPERATE)][0]
        cc_payoff = self.payoff_matrix[(Action.COOPERATE, Action.COOPERATE)][0]
        return dc_payoff - cc_payoff
    
    def get_sucker_payoff(self) -> float:
        """获取被欺骗收益（CD收益）"""
        return self.payoff_matrix[(Action.COOPERATE, Action.DEFECT)][0]
    
    def get_punishment_payoff(self) -> float:
        """获取惩罚收益（DD收益）"""
        return self.payoff_matrix[(Action.DEFECT, Action.DEFECT)][0]
    
    def is_valid_prisoners_dilemma(self) -> bool:
        """检查是否为有效的囚徒困境（满足T > R > P > S）"""
        cc = self.payoff_matrix[(Action.COOPERATE, Action.COOPERATE)][0]
        cd = self.payoff_matrix[(Action.COOPERATE, Action.DEFECT)][0]
        dc = self.payoff_matrix[(Action.DEFECT, Action.COOPERATE)][0]
        dd = self.payoff_matrix[(Action.DEFECT, Action.DEFECT)][0]
        
        # T > R > P > S (Temptation > Reward > Punishment > Sucker)
        return dc > cc > dd > cd
    
    def get_game_parameters(self) -> Dict[str, float]:
        """获取博弈参数"""
        return {
            "reward": self.payoff_matrix[(Action.COOPERATE, Action.COOPERATE)][0],
            "temptation": self.payoff_matrix[(Action.DEFECT, Action.COOPERATE)][0],
            "punishment": self.payoff_matrix[(Action.DEFECT, Action.DEFECT)][0],
            "sucker": self.payoff_matrix[(Action.COOPERATE, Action.DEFECT)][0],
            "cooperation_incentive": self.get_cooperation_incentive(),
            "defection_temptation": self.get_defection_temptation(),
            "is_valid_pd": self.is_valid_prisoners_dilemma()
        }


class StrategyAnalyzer:
    """策略分析器"""
    
    def __init__(self, game: PrisonersDilemma):
        self.game = game
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_strategy(self, actions: List[Action]) -> Dict[str, Any]:
        """分析单个玩家的策略"""
        if not actions:
            return {}
        
        cooperation_rate = sum(1 for a in actions if a == Action.COOPERATE) / len(actions)
        defect_rate = 1 - cooperation_rate
        
        # 分析策略模式
        patterns = self._analyze_patterns(actions)
        
        # 分析响应性
        responsiveness = self._analyze_responsiveness(actions)
        
        return {
            "cooperation_rate": cooperation_rate,
            "defect_rate": defect_rate,
            "total_actions": len(actions),
            "cooperation_count": sum(1 for a in actions if a == Action.COOPERATE),
            "defect_count": sum(1 for a in actions if a == Action.DEFECT),
            "patterns": patterns,
            "responsiveness": responsiveness
        }
    
    def _analyze_patterns(self, actions: List[Action]) -> Dict[str, Any]:
        """分析策略模式"""
        if len(actions) < 2:
            return {}
        
        # 计算连续合作/背叛的长度
        cooperation_streaks = []
        defect_streaks = []
        
        current_streak = 1
        current_action = actions[0]
        
        for i in range(1, len(actions)):
            if actions[i] == current_action:
                current_streak += 1
            else:
                if current_action == Action.COOPERATE:
                    cooperation_streaks.append(current_streak)
                else:
                    defect_streaks.append(current_streak)
                current_streak = 1
                current_action = actions[i]
        
        # 添加最后一个streak
        if current_action == Action.COOPERATE:
            cooperation_streaks.append(current_streak)
        else:
            defect_streaks.append(current_streak)
        
        return {
            "max_cooperation_streak": max(cooperation_streaks) if cooperation_streaks else 0,
            "max_defect_streak": max(defect_streaks) if defect_streaks else 0,
            "avg_cooperation_streak": np.mean(cooperation_streaks) if cooperation_streaks else 0,
            "avg_defect_streak": np.mean(defect_streaks) if defect_streaks else 0,
            "cooperation_streaks": cooperation_streaks,
            "defect_streaks": defect_streaks
        }
    
    def _analyze_responsiveness(self, actions: List[Action]) -> Dict[str, float]:
        """分析响应性（对对手行为的反应）"""
        if len(actions) < 2:
            return {}
        
        # 计算动作变化率
        changes = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
        change_rate = changes / (len(actions) - 1)
        
        return {
            "change_rate": change_rate,
            "total_changes": changes
        }
    
    def compare_strategies(self, strategy1: List[Action], strategy2: List[Action]) -> Dict[str, Any]:
        """比较两个策略"""
        analysis1 = self.analyze_strategy(strategy1)
        analysis2 = self.analyze_strategy(strategy2)
        
        return {
            "strategy1": analysis1,
            "strategy2": analysis2,
            "cooperation_rate_diff": analysis1["cooperation_rate"] - analysis2["cooperation_rate"],
            "similarity": self._calculate_similarity(strategy1, strategy2)
        }
    
    def _calculate_similarity(self, strategy1: List[Action], strategy2: List[Action]) -> float:
        """计算两个策略的相似性"""
        if len(strategy1) != len(strategy2):
            return 0.0
        
        matches = sum(1 for a1, a2 in zip(strategy1, strategy2) if a1 == a2)
        return matches / len(strategy1)


class GameStatistics:
    """博弈统计类"""
    
    def __init__(self, game: PrisonersDilemma):
        self.game = game
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_expected_payoff(self, strategy1: List[Action], strategy2: List[Action]) -> Tuple[float, float]:
        """计算期望收益"""
        history = self.game.play_game(strategy1, strategy2)
        return history.player1_total_payoff, history.player2_total_payoff
    
    def calculate_efficiency(self, strategy1: List[Action], strategy2: List[Action]) -> float:
        """计算效率（相对于帕累托最优的收益比例）"""
        payoff1, payoff2 = self.calculate_expected_payoff(strategy1, strategy2)
        optimal_payoff = self.game.payoff_matrix[(Action.COOPERATE, Action.COOPERATE)][0]
        total_optimal = optimal_payoff * 2
        total_actual = payoff1 + payoff2
        return total_actual / total_optimal if total_optimal > 0 else 0
    
    def calculate_fairness(self, strategy1: List[Action], strategy2: List[Action]) -> float:
        """计算公平性（收益差异的倒数）"""
        payoff1, payoff2 = self.calculate_expected_payoff(strategy1, strategy2)
        if payoff1 + payoff2 == 0:
            return 1.0
        return 1 - abs(payoff1 - payoff2) / (payoff1 + payoff2)
    
    def generate_summary_statistics(self, histories: List[GameHistory]) -> Dict[str, Any]:
        """生成汇总统计"""
        if not histories:
            return {}
        
        cooperation_rates_1 = [h.player1_cooperation_rate for h in histories]
        cooperation_rates_2 = [h.player2_cooperation_rate for h in histories]
        payoffs_1 = [h.player1_total_payoff for h in histories]
        payoffs_2 = [h.player2_total_payoff for h in histories]
        
        return {
            "num_games": len(histories),
            "avg_cooperation_rate_1": np.mean(cooperation_rates_1),
            "avg_cooperation_rate_2": np.mean(cooperation_rates_2),
            "std_cooperation_rate_1": np.std(cooperation_rates_1),
            "std_cooperation_rate_2": np.std(cooperation_rates_2),
            "avg_payoff_1": np.mean(payoffs_1),
            "avg_payoff_2": np.mean(payoffs_2),
            "std_payoff_1": np.std(payoffs_1),
            "std_payoff_2": np.std(payoffs_2),
            "total_cooperation_rate": np.mean(cooperation_rates_1 + cooperation_rates_2),
            "total_std_cooperation_rate": np.std(cooperation_rates_1 + cooperation_rates_2)
        }
