"""
Prisoner's Dilemma Game Implementation
Includes game logic, payoff calculation, strategy analysis, and other features
"""

from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging


class Action(Enum):
    """Game actions"""
    COOPERATE = "COOPERATE"
    DEFECT = "DEFECT"
    # Extend with more actions if needed


@dataclass
class GameResult:
    """Single round game result"""
    player1_action: Action
    player2_action: Action
    player1_payoff: float
    player2_payoff: float
    round_number: int


@dataclass
class GameHistory:
    """Game history record"""
    results: List[GameResult]
    total_rounds: int
    player1_total_payoff: float
    player2_total_payoff: float
    player1_cooperation_rate: float
    player2_cooperation_rate: float
    
    def add_result(self, result: GameResult):
        """Add a round result"""
        self.results.append(result)
        self.total_rounds += 1
        self.player1_total_payoff += result.player1_payoff
        self.player2_total_payoff += result.player2_payoff
        
        # Update cooperation rate
        player1_cooperations = sum(1 for r in self.results if r.player1_action == Action.COOPERATE)
        player2_cooperations = sum(1 for r in self.results if r.player2_action == Action.COOPERATE)
        
        self.player1_cooperation_rate = player1_cooperations / self.total_rounds
        self.player2_cooperation_rate = player2_cooperations / self.total_rounds


class TwoPlayerGame:
    """Generic two-player game"""
    def __init__(self, payoff_matrix: Dict[Tuple[Action, Action], Tuple[float, float]]):
        self.payoff_matrix = payoff_matrix
        self._validate_payoff_matrix()

    def _validate_payoff_matrix(self):
        # Check all combinations are defined
        for a1 in Action:
            for a2 in Action:
                if (a1, a2) not in self.payoff_matrix:
                    raise ValueError(f"Missing payoff for: {(a1, a2)}")

    def calculate_payoff(self, action1: Action, action2: Action):
        return self.payoff_matrix[(action1, action2)]


class PrisonersDilemma(TwoPlayerGame):
    """Prisoner's Dilemma game class"""
    
    # Standard Prisoner's Dilemma payoff matrix
    PAYOFF_MATRIX = {
        (Action.COOPERATE, Action.COOPERATE): (3, 3),  # Both cooperate
        (Action.COOPERATE, Action.DEFECT): (0, 5),      # I cooperate, opponent defects
        (Action.DEFECT, Action.COOPERATE): (5, 0),      # I defect, opponent cooperates
        (Action.DEFECT, Action.DEFECT): (1, 1)          # Both defect
    }
    
    def __init__(self, payoff_matrix: Optional[Dict[Tuple[Action, Action], Tuple[float, float]]] = None):
        """
        Initialize Prisoner's Dilemma game
        
        Args:
            payoff_matrix: Custom payoff matrix, format {(action1, action2): (payoff1, payoff2)}
        """
        super().__init__(payoff_matrix or self.PAYOFF_MATRIX)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def play_round(self, action1: Action, action2: Action, round_number: int = 1) -> GameResult:
        """Play one round"""
        payoff1, payoff2 = self.calculate_payoff(action1, action2)
        
        return GameResult(
            player1_action=action1,
            player2_action=action2,
            player1_payoff=payoff1,
            player2_payoff=payoff2,
            round_number=round_number
        )
    
    def play_game(self, player1_actions: List[Action], player2_actions: List[Action]) -> GameHistory:
        """Play multiple rounds"""
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
        """Get Nash equilibrium (Defect, Defect)"""
        return Action.DEFECT, Action.DEFECT
    
    def get_pareto_optimal(self) -> Tuple[Action, Action]:
        """Get Pareto optimal (Cooperate, Cooperate)"""
        return Action.COOPERATE, Action.COOPERATE
    
    def is_dominant_strategy(self, action: Action) -> bool:
        """Check if action is a dominant strategy (Defect is dominant)"""
        return action == Action.DEFECT
    
    def get_cooperation_incentive(self) -> float:
        """Get cooperation incentive (CC - CD payoff difference)"""
        cc_payoff = self.payoff_matrix[(Action.COOPERATE, Action.COOPERATE)][0]
        cd_payoff = self.payoff_matrix[(Action.COOPERATE, Action.DEFECT)][0]
        return cc_payoff - cd_payoff
    
    def get_defection_temptation(self) -> float:
        """Get defection temptation (DC - CC payoff difference)"""
        dc_payoff = self.payoff_matrix[(Action.DEFECT, Action.COOPERATE)][0]
        cc_payoff = self.payoff_matrix[(Action.COOPERATE, Action.COOPERATE)][0]
        return dc_payoff - cc_payoff
    
    def get_sucker_payoff(self) -> float:
        """Get sucker payoff (CD payoff)"""
        return self.payoff_matrix[(Action.COOPERATE, Action.DEFECT)][0]
    
    def get_punishment_payoff(self) -> float:
        """Get punishment payoff (DD payoff)"""
        return self.payoff_matrix[(Action.DEFECT, Action.DEFECT)][0]
    
    def is_valid_prisoners_dilemma(self) -> bool:
        """Check if valid Prisoner's Dilemma (T > R > P > S)"""
        cc = self.payoff_matrix[(Action.COOPERATE, Action.COOPERATE)][0]
        cd = self.payoff_matrix[(Action.COOPERATE, Action.DEFECT)][0]
        dc = self.payoff_matrix[(Action.DEFECT, Action.COOPERATE)][0]
        dd = self.payoff_matrix[(Action.DEFECT, Action.DEFECT)][0]
        
        # T > R > P > S (Temptation > Reward > Punishment > Sucker)
        return dc > cc > dd > cd
    
    def get_game_parameters(self) -> Dict[str, float]:
        """Get game parameters"""
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
    """Strategy analyzer"""
    
    def __init__(self, game: PrisonersDilemma):
        self.game = game
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_strategy(self, actions: List[Action]) -> Dict[str, Any]:
        """Analyze a single player's strategy"""
        if not actions:
            return {}
        
        cooperation_rate = sum(1 for a in actions if a == Action.COOPERATE) / len(actions)
        defect_rate = 1 - cooperation_rate
        
        # Analyze strategy patterns
        patterns = self._analyze_patterns(actions)
        
        # Analyze responsiveness
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
        """Analyze strategy patterns"""
        if len(actions) < 2:
            return {}
        
        # Calculate lengths of consecutive cooperation/defection
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
        
        # Add last streak
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
        """Analyze responsiveness (reaction to opponent's actions)"""
        if len(actions) < 2:
            return {}
        
        # Calculate action change rate
        changes = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
        change_rate = changes / (len(actions) - 1)
        
        return {
            "change_rate": change_rate,
            "total_changes": changes
        }
    
    def compare_strategies(self, strategy1: List[Action], strategy2: List[Action]) -> Dict[str, Any]:
        """Compare two strategies"""
        analysis1 = self.analyze_strategy(strategy1)
        analysis2 = self.analyze_strategy(strategy2)
        
        return {
            "strategy1": analysis1,
            "strategy2": analysis2,
            "cooperation_rate_diff": analysis1["cooperation_rate"] - analysis2["cooperation_rate"],
            "similarity": self._calculate_similarity(strategy1, strategy2)
        }
    
    def _calculate_similarity(self, strategy1: List[Action], strategy2: List[Action]) -> float:
        """Calculate similarity between two strategies"""
        if len(strategy1) != len(strategy2):
            return 0.0
        
        matches = sum(1 for a1, a2 in zip(strategy1, strategy2) if a1 == a2)
        return matches / len(strategy1)


class GameStatistics:
    """Game statistics class"""
    
    def __init__(self, game: PrisonersDilemma):
        self.game = game
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_expected_payoff(self, strategy1: List[Action], strategy2: List[Action]) -> Tuple[float, float]:
        """Calculate expected payoff"""
        history = self.game.play_game(strategy1, strategy2)
        return history.player1_total_payoff, history.player2_total_payoff
    
    def calculate_efficiency(self, strategy1: List[Action], strategy2: List[Action]) -> float:
        """Calculate efficiency (relative to Pareto optimal payoff ratio)"""
        payoff1, payoff2 = self.calculate_expected_payoff(strategy1, strategy2)
        optimal_payoff = self.game.payoff_matrix[(Action.COOPERATE, Action.COOPERATE)][0]
        total_optimal = optimal_payoff * 2
        total_actual = payoff1 + payoff2
        return total_actual / total_optimal if total_optimal > 0 else 0
    
    def calculate_fairness(self, strategy1: List[Action], strategy2: List[Action]) -> float:
        """Calculate fairness (inverse of payoff difference)"""
        payoff1, payoff2 = self.calculate_expected_payoff(strategy1, strategy2)
        if payoff1 + payoff2 == 0:
            return 1.0
        return 1 - abs(payoff1 - payoff2) / (payoff1 + payoff2)
    
    def generate_summary_statistics(self, histories: List[GameHistory]) -> Dict[str, Any]:
        """Generate summary statistics"""
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
