"""
LLM Network Games Framework
一个用于研究LLM在网络博弈中行为的可复现、可修改的框架
"""

__version__ = "1.0.0"
__author__ = "Xuan Qiu"
__email__ = "maxxuanqiu@hkust-gz.edu.cn"
__description__ = "A framework for studying LLM behavior in network games"

# 导入主要模块
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
