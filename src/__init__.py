"""
LLM Network Games Framework
A reproducible and modifiable framework for studying LLM behavior in network games
"""

__version__ = "1.0.0"
__author__ = "Xuan Qiu"
__email__ = "maxxuanqiu@gmail.com"
__description__ = "A framework for studying LLM behavior in network games"

# Import main modules
from .agents.mbti_personalities import MBTIType, MBTIPersonality, get_all_mbti_types
from .games.prisoners_dilemma import PrisonersDilemma, Action, GameHistory
from .llm.llm_interface import LLMManager, LLMFactory, LLMProvider
from .networks.network_generator import NetworkGenerator, NetworkConfig, NetworkType
from .config.config_manager import ConfigManager, ExperimentConfig

__all__ = [
    "MBTIType",
    "MBTIPersonality", 
    "get_all_mbti_types",
    "PrisonersDilemma",
    "Action",
    "GameHistory",
    "LLMManager",
    "LLMFactory",
    "LLMProvider",
    "NetworkGenerator",
    "NetworkConfig",
    "NetworkType",
    "ConfigManager",
    "ExperimentConfig"
]
