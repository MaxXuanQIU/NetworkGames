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
        self.name = "Game"

    def _validate_payoff_matrix(self):
        # Check all combinations are defined
        for a1 in Action:
            for a2 in Action:
                if (a1, a2) not in self.payoff_matrix:
                    raise ValueError(f"Missing payoff for: {(a1, a2)}")

    def calculate_payoff(self, action1: Action, action2: Action):
        return self.payoff_matrix[(action1, action2)]

    def get_rules_description(self) -> str:
        """Generate text description of game rules and payoffs"""
        cc = self.payoff_matrix[(Action.COOPERATE, Action.COOPERATE)]
        cd = self.payoff_matrix[(Action.COOPERATE, Action.DEFECT)]
        dc = self.payoff_matrix[(Action.DEFECT, Action.COOPERATE)]
        dd = self.payoff_matrix[(Action.DEFECT, Action.DEFECT)]

        return (
            f"{self.name} payoff matrix:\n"
            f"- If both choose COOPERATE, each gets {cc[0]} points\n"
            f"- If you COOPERATE and opponent DEFECTS, you get {cd[0]} points, opponent gets {cd[1]} points\n"
            f"- If you DEFECT and opponent COOPERATES, you get {dc[0]} points, opponent gets {dc[1]} points\n"
            f"- If both DEFECT, each gets {dd[0]} points\n"
        )
    
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


class PrisonersDilemma(TwoPlayerGame):
    """Prisoner's Dilemma game class"""
    
    # Standard Prisoner's Dilemma payoff matrix
    PAYOFF_MATRIX = {
        (Action.COOPERATE, Action.COOPERATE): (3, 3),  # Both cooperate (Reward)
        (Action.COOPERATE, Action.DEFECT): (0, 5),      # I cooperate, opponent defects (Sucker, Temptation)
        (Action.DEFECT, Action.COOPERATE): (5, 0),      # I defect, opponent cooperates (Temptation, Sucker)
        (Action.DEFECT, Action.DEFECT): (1, 1)          # Both defect (Punishment)
    }
    
    def __init__(self, payoff_matrix: Optional[Dict[Tuple[Action, Action], Tuple[float, float]]] = None):
        """
        Initialize Prisoner's Dilemma game
        
        Args:
            payoff_matrix: Custom payoff matrix, format {(action1, action2): (payoff1, payoff2)}
        """
        super().__init__(payoff_matrix or self.PAYOFF_MATRIX)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.name = "Prisoner's Dilemma"
    
    def get_nash_equilibrium(self) -> Tuple[Action, Action]:
        """Get Nash equilibrium (Defect, Defect)"""
        return Action.DEFECT, Action.DEFECT
    
    def get_pareto_optimal(self) -> Tuple[Action, Action]:
        """Get Pareto optimal (Cooperate, Cooperate)"""
        return Action.COOPERATE, Action.COOPERATE
    
    def get_cooperation_incentive(self) -> float:
        """Get cooperation incentive (CC - CD or S - P payoff difference)"""
        cc_payoff = self.payoff_matrix[(Action.COOPERATE, Action.COOPERATE)][0]
        cd_payoff = self.payoff_matrix[(Action.COOPERATE, Action.DEFECT)][0]
        return cc_payoff - cd_payoff
    
    def get_defection_temptation(self) -> float:
        """Get defection temptation (DC - CC or T - R payoff difference)"""
        dc_payoff = self.payoff_matrix[(Action.DEFECT, Action.COOPERATE)][0]
        cc_payoff = self.payoff_matrix[(Action.COOPERATE, Action.COOPERATE)][0]
        return dc_payoff - cc_payoff
    
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
            # "is_valid_pd": self.is_valid_prisoners_dilemma() # Removed or kept optional
        }


class StagHunt(TwoPlayerGame):
    """
    Stag Hunt game class.
    Cooperation (Stag) is risky but rewarding. Defection (Hare) is safe.
    Payoff ordering: Reward > Temptation >= Punishment > Sucker
    """
    PAYOFF_MATRIX = {
        (Action.COOPERATE, Action.COOPERATE): (5, 5),  # Both Cooperate (Stag)
        (Action.COOPERATE, Action.DEFECT): (0, 3),    # I Cooperate (Stag), Opponent Defects (Hare)
        (Action.DEFECT, Action.COOPERATE): (3, 0),    # I Defect (Hare), Opponent Cooperates (Stag)
        (Action.DEFECT, Action.DEFECT): (1, 1)        # Both Defect (Hare)
    }

    def __init__(self, payoff_matrix: Optional[Dict[Tuple[Action, Action], Tuple[float, float]]] = None):
        super().__init__(payoff_matrix or self.PAYOFF_MATRIX)
        self.name = "Stag Hunt"


class Snowdrift(TwoPlayerGame):
    """
    Snowdrift (Chicken) game class.
    It is better to cooperate if the other defects, but better to defect if the other cooperates.
    Payoff ordering: Temptation > Reward > Sucker > Punishment
    """
    PAYOFF_MATRIX = {
        (Action.COOPERATE, Action.COOPERATE): (3, 3),  # Both Shovel (Reward)
        (Action.COOPERATE, Action.DEFECT): (1, 5),    # I Shovel, Opponent Waits (Sucker, Temptation)
        (Action.DEFECT, Action.COOPERATE): (5, 1),    # I Wait, Opponent Shovels (Temptation, Sucker)
        (Action.DEFECT, Action.DEFECT): (0, 0)        # Both Wait (Punishment/Crash)
    }

    def __init__(self, payoff_matrix: Optional[Dict[Tuple[Action, Action], Tuple[float, float]]] = None):
        super().__init__(payoff_matrix or self.PAYOFF_MATRIX)
        self.name = "Snowdrift"


class GameFactory:
    """Factory for creating games"""
    
    @staticmethod
    def create_game(game_name: str, payoff_matrix: Optional[Dict] = None) -> TwoPlayerGame:
        game_name = game_name.lower()
        if game_name == "prisoners_dilemma" or game_name == "pd":
            return PrisonersDilemma(payoff_matrix)
        elif game_name == "stag_hunt" or game_name == "sh":
            return StagHunt(payoff_matrix)
        elif game_name == "snowdrift" or game_name == "sd" or game_name == "chicken":
            return Snowdrift(payoff_matrix)
        else:
            raise ValueError(f"Unknown game type: {game_name}")
