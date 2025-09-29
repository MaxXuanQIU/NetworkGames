"""
Configuration Management System
Supports loading, validating, and managing YAML configuration files
"""

import yaml
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import logging
from enum import Enum


class ExperimentType(Enum):
    """Experiment type"""
    PAIR_GAME = "pair_game"
    NETWORK_GAME = "network_game"


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: str
    model_name: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 50
    timeout: int = 30
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameConfig:
    """Game configuration"""
    payoff_matrix: Dict[str, Dict[str, List[float]]] = field(default_factory=lambda: {
        "COOPERATE": {"COOPERATE": [3, 3], "DEFECT": [0, 5]},
        "DEFECT": {"COOPERATE": [5, 0], "DEFECT": [1, 1]}
    })
    num_rounds: int = 100
    num_repetitions: int = 20
    random_seed: Optional[int] = None


@dataclass
class NetworkConfig:
    """Network configuration"""
    network_type: str
    num_nodes: int
    k: int = 4
    p: float = 0.1
    edge_probability: float = 0.1
    m: int = 2
    seed: Optional[int] = None
    directed: bool = False


@dataclass
class PersonalityDistributionConfig:
    """Personality distribution configuration"""
    distribution_type: str  # "uniform", "clustered", "single", "custom"
    single_type: Optional[str] = None  # Used when distribution_type is "single"
    cluster_config: Optional[Dict[str, Any]] = None  # Clustered distribution configuration
    custom_distribution: Optional[Dict[str, float]] = None  # Custom distribution


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    experiment_type: ExperimentType
    name: str
    description: str
    
    # Basic configuration
    llm: LLMConfig
    game: GameConfig
    network: NetworkConfig
    personality_distribution: PersonalityDistributionConfig
    
    # Experiment-specific configuration
    pair_game_config: Optional[Dict[str, Any]] = None
    network_game_config: Optional[Dict[str, Any]] = None
    
    # Output configuration
    output_dir: str = "results"
    save_detailed_results: bool = True
    save_visualizations: bool = True
    save_network_snapshots: bool = True
    
    # Visualization configuration
    visualization_config: Dict[str, Any] = field(default_factory=lambda: {
        "figsize": [12, 8],
        "dpi": 300,
        "style": "seaborn-v0_8",
        "color_palette": "Set2"
    })


class ConfigManager:
    """Configuration manager"""
    
    def __init__(self, config_dir: str = "configs"):
        # Convention: command line --config only supports paths like
        # "configs/xxx.yaml" (or absolute paths).
        # Save project root and config directory for later use.
        project_root = Path(__file__).parent.parent.parent
        self.project_root = project_root
        # Config directory is fixed as <project_root>/configs
        self.config_dir = project_root / config_dir if not Path(config_dir).is_absolute() else Path(config_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._ensure_config_dir()
    
    def _ensure_config_dir(self):
        """Ensure config directory exists"""
        self.config_dir.mkdir(exist_ok=True)
    
    def load_config(self, config_file: str) -> ExperimentConfig:
        """Load configuration file
        Rules:
        - Recommended and only supported: relative path like "configs/xxx.yaml" or absolute path
        - No longer supports only file name (like "pair_game.yaml")
        """
        cfg_path = Path(config_file)
        if cfg_path.is_absolute():
            config_path = cfg_path
        else:
            # Only accept relative paths starting with configs/
            # Compatible with Windows backslash
            cfg_str = str(cfg_path).replace("\\", "/")
            if cfg_str.startswith("configs/"):
                config_path = self.project_root / cfg_str
            else:
                raise FileNotFoundError(
                    "Config path must start with 'configs/' or be an absolute path. "
                    f"Got: {config_file}"
                )
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return self._parse_config(config_data)
    
    def save_config(self, config: ExperimentConfig, config_file: str):
        """Save configuration file
        Requires path like "configs/xxx.yaml" or absolute path
        """
        cfg_path = Path(config_file)
        if cfg_path.is_absolute():
            config_path = cfg_path
        else:
            cfg_str = str(cfg_path).replace("\\", "/")
            if cfg_str.startswith("configs/"):
                config_path = self.project_root / cfg_str
            else:
                # Default save to standard directory
                config_path = self.config_dir / cfg_path.name
        
        config_data = self._serialize_config(config)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"Config saved to: {config_path}")
    
    def _parse_config(self, data: Dict[str, Any]) -> ExperimentConfig:
        """Parse configuration data"""
        # Parse LLM configuration
        llm_data = data.get("llm", {})
        llm_config = LLMConfig(
            provider=llm_data.get("provider", "mock"),
            model_name=llm_data.get("model_name", "mock-model"),
            api_key=llm_data.get("api_key"),
            temperature=llm_data.get("temperature", 0.7),
            max_tokens=llm_data.get("max_tokens", 50),
            timeout=llm_data.get("timeout", 30),
            kwargs=llm_data.get("kwargs", {})
        )
        
        # Parse game configuration
        game_data = data.get("game", {})
        game_config = GameConfig(
            payoff_matrix=game_data.get("payoff_matrix", {
                "COOPERATE": {"COOPERATE": [3, 3], "DEFECT": [0, 5]},
                "DEFECT": {"COOPERATE": [5, 0], "DEFECT": [1, 1]}
            }),
            num_rounds=game_data.get("num_rounds", 100),
            num_repetitions=game_data.get("num_repetitions", 20),
            random_seed=game_data.get("random_seed")
        )
        
        # Parse network configuration
        network_data = data.get("network", {})
        network_config = NetworkConfig(
            network_type=network_data.get("network_type", "small_world"),
            num_nodes=network_data.get("num_nodes", 50),
            k=network_data.get("k", 4),
            p=network_data.get("p", 0.1),
            edge_probability=network_data.get("edge_probability", 0.1),
            m=network_data.get("m", 2),
            seed=network_data.get("seed"),
            directed=network_data.get("directed", False)
        )
        
        # Parse personality distribution configuration
        personality_data = data.get("personality_distribution", {})
        personality_config = PersonalityDistributionConfig(
            distribution_type=personality_data.get("distribution_type", "uniform"),
            single_type=personality_data.get("single_type"),
            cluster_config=personality_data.get("cluster_config"),
            custom_distribution=personality_data.get("custom_distribution")
        )
        
        # Parse experiment configuration
        experiment_type = ExperimentType(data.get("experiment_type", "pair_game"))
        
        experiment_config = ExperimentConfig(
            experiment_type=experiment_type,
            name=data.get("name", "unnamed_experiment"),
            description=data.get("description", ""),
            llm=llm_config,
            game=game_config,
            network=network_config,
            personality_distribution=personality_config,
            pair_game_config=data.get("pair_game_config"),
            network_game_config=data.get("network_game_config"),
            output_dir=data.get("output_dir", "results"),
            save_detailed_results=data.get("save_detailed_results", True),
            save_visualizations=data.get("save_visualizations", True),
            save_network_snapshots=data.get("save_network_snapshots", True),
            visualization_config=data.get("visualization_config", {
                "figsize": [12, 8],
                "dpi": 300,
                "style": "seaborn-v0_8",
                "color_palette": "Set2"
            })
        )
        
        return experiment_config
    
    def _serialize_config(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Serialize configuration object"""
        return {
            "experiment_type": config.experiment_type.value,
            "name": config.name,
            "description": config.description,
            "llm": {
                "provider": config.llm.provider,
                "model_name": config.llm.model_name,
                "api_key": config.llm.api_key,
                "temperature": config.llm.temperature,
                "max_tokens": config.llm.max_tokens,
                "timeout": config.llm.timeout,
                "kwargs": config.llm.kwargs
            },
            "game": {
                "payoff_matrix": config.game.payoff_matrix,
                "num_rounds": config.game.num_rounds,
                "num_repetitions": config.game.num_repetitions,
                "random_seed": config.game.random_seed
            },
            "network": {
                "network_type": config.network.network_type,
                "num_nodes": config.network.num_nodes,
                "k": config.network.k,
                "p": config.network.p,
                "edge_probability": config.network.edge_probability,
                "m": config.network.m,
                "seed": config.network.seed,
                "directed": config.network.directed
            },
            "personality_distribution": {
                "distribution_type": config.personality_distribution.distribution_type,
                "single_type": config.personality_distribution.single_type,
                "cluster_config": config.personality_distribution.cluster_config,
                "custom_distribution": config.personality_distribution.custom_distribution
            },
            "pair_game_config": config.pair_game_config,
            "network_game_config": config.network_game_config,
            "output_dir": config.output_dir,
            "save_detailed_results": config.save_detailed_results,
            "save_visualizations": config.save_visualizations,
            "save_network_snapshots": config.save_network_snapshots,
            "visualization_config": config.visualization_config
        }
    
    def create_default_configs(self):
        """Create default configuration files"""
        # Pair game experiment configuration
        pair_game_config = ExperimentConfig(
            experiment_type=ExperimentType.PAIR_GAME,
            name="MBTI Pair Game Experiment",
            description="16x16 MBTI personality matrix in repeated prisoner's dilemma",
            llm=LLMConfig(
                provider="mock",
                model_name="mock-model",
                temperature=0.7
            ),
            game=GameConfig(
                num_rounds=100,
                num_repetitions=20,
                random_seed=42
            ),
            network=NetworkConfig(
                network_type="small_world",
                num_nodes=50
            ),
            personality_distribution=PersonalityDistributionConfig(
                distribution_type="uniform"
            ),
            pair_game_config={
                "matrix_size": 16,
                "save_heatmap": True,
                "save_statistics": True
            }
        )
        
        # Network game experiment configuration
        network_game_config = ExperimentConfig(
            experiment_type=ExperimentType.NETWORK_GAME,
            name="MBTI Network Game Experiment",
            description="Network-based prisoner's dilemma with different personality distributions",
            llm=LLMConfig(
                provider="mock",
                model_name="mock-model",
                temperature=0.7
            ),
            game=GameConfig(
                num_rounds=100,
                num_repetitions=5,
                random_seed=42
            ),
            network=NetworkConfig(
                network_type="small_world",
                num_nodes=50,
                k=4,
                p=0.1
            ),
            personality_distribution=PersonalityDistributionConfig(
                distribution_type="uniform"
            ),
            network_game_config={
                "network_types": ["regular", "small_world_0.1", "small_world_0.5", "random"],
                "personality_scenarios": ["uniform", "single_ENTJ", "clustered"],
                "save_network_evolution": True,
                "save_cooperation_metrics": True
            }
        )
        
        # Save configuration files
        self.save_config(pair_game_config, "pair_game.yaml")
        self.save_config(network_game_config, "network_game.yaml")
        
        self.logger.info("Default config files created")
    
    def validate_config(self, config: ExperimentConfig) -> List[str]:
        """Validate configuration"""
        errors = []
        
        # Validate experiment type
        if config.experiment_type not in ExperimentType:
            errors.append(f"Invalid experiment type: {config.experiment_type}")
        
        # Validate LLM configuration
        if not config.llm.provider:
            errors.append("LLM provider is required")
        
        if not config.llm.model_name:
            errors.append("LLM model name is required")
        
        # Validate game configuration
        if config.game.num_rounds <= 0:
            errors.append("Number of rounds must be positive")
        
        if config.game.num_repetitions <= 0:
            errors.append("Number of repetitions must be positive")
        
        # Validate network configuration
        if config.network.num_nodes <= 0:
            errors.append("Number of nodes must be positive")
        
        if config.network.k >= config.network.num_nodes:
            errors.append("k must be less than number of nodes")
        
        # Validate personality distribution configuration
        if config.personality_distribution.distribution_type == "single":
            if not config.personality_distribution.single_type:
                errors.append("Single type must be specified for single distribution")
        
        return errors
    
    def list_configs(self) -> List[str]:
        """List all configuration files"""
        config_files = []
        for file_path in self.config_dir.glob("*.yaml"):
            config_files.append(file_path.name)
        return sorted(config_files)
    
    def get_config_info(self, config_file: str) -> Dict[str, Any]:
        """Get configuration file information"""
        try:
            config = self.load_config(config_file)
            return {
                "name": config.name,
                "description": config.description,
                "experiment_type": config.experiment_type.value,
                "llm_provider": config.llm.provider,
                "llm_model": config.llm.model_name,
                "num_rounds": config.game.num_rounds,
                "num_repetitions": config.game.num_repetitions,
                "network_type": config.network.network_type,
                "num_nodes": config.network.num_nodes,
                "personality_distribution": config.personality_distribution.distribution_type
            }
        except Exception as e:
            return {"error": str(e)}


# Function to create default configuration files
def create_default_configs():
    """Create default configuration files"""
    config_manager = ConfigManager()
    config_manager.create_default_configs()
    return config_manager

