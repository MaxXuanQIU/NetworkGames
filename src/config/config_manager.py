"""
Configuration Management System
Supports loading, validating, and managing YAML configuration files
"""

import yaml
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
class PairGameConfig:
    """Pair game specific configuration"""
    matrix_size: int = 16
    save_heatmap: bool = True
    save_statistics: bool = True

@dataclass
class NetworkGameConfig:
    """Network game specific configuration"""
    network_types: List[str] = field(default_factory=lambda: ["regular", "small_world_0.1", "random"])
    personality_scenarios: List[str] = field(default_factory=lambda: ["uniform"])
    seed: Optional[int] = None
    save_network_evolution: bool = True
    save_cooperation_metrics: bool = True

@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    experiment_type: ExperimentType
    name: str
    description: str
    
    # Basic configuration
    llm: LLMConfig
    game: GameConfig
    network: Optional[NetworkConfig] = None

    # Experiment-specific configuration (now typed)
    pair_game_config: Optional[PairGameConfig] = None
    network_game_config: Optional[NetworkGameConfig] = None
    
    # Output configuration
    output_dir: str = "results"
    
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
        
        # parse first
        config = self._parse_config(config_data)
        # validate and raise if invalid (validate_config enforces experiment-type-specific rules)
        errors = self.validate_config(config)
        if errors:
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
        return config
    
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
        """Parse configuration data

        Strict mode: no sensible defaults are used. Required fields must be present
        in the provided YAML; otherwise a ValueError listing missing fields is raised.
        """
        missing = []

        # Helper to mark missing nested keys
        def _require_section(section_name: str):
            if section_name not in data or data.get(section_name) is None:
                missing.append(section_name)
                return None
            return data[section_name]

        def _require_key(container: Dict[str, Any], key_path: str, container_name: str):
            if container is None:
                # section already missing, don't duplicate errors for inner keys
                return None
            if key_path not in container or container.get(key_path) is None:
                missing.append(f"{container_name}.{key_path}")
                return None
            return container.get(key_path)

        # LLM section (required)
        llm_data = _require_section("llm")
        llm_provider = _require_key(llm_data, "provider", "llm")
        llm_model = _require_key(llm_data, "model_name", "llm")
        # optional fields (will be taken as-is if present)
        llm_api_key = llm_data.get("api_key") if llm_data else None
        llm_temperature = llm_data.get("temperature") if llm_data else None
        llm_max_tokens = llm_data.get("max_tokens") if llm_data else None
        llm_timeout = llm_data.get("timeout") if llm_data else None
        llm_kwargs = llm_data.get("kwargs") if llm_data else None

        # Game section (required)
        game_data = _require_section("game")
        game_payoff = _require_key(game_data, "payoff_matrix", "game")
        game_num_rounds = _require_key(game_data, "num_rounds", "game")
        game_num_repetitions = _require_key(game_data, "num_repetitions", "game")
        game_random_seed = game_data.get("random_seed") if game_data else None

        # Network section (parsed permissively; validated later only for NETWORK_GAME)
        network_data = data.get("network")
        if network_data:
            network_type = network_data.get("network_type")
            network_num_nodes = network_data.get("num_nodes")
            network_k = network_data.get("k")
            network_p = network_data.get("p")
            network_edge_probability = network_data.get("edge_probability")
            network_m = network_data.get("m")
            network_seed = network_data.get("seed")
            network_directed = network_data.get("directed")
            network_config = NetworkConfig(
                network_type=network_type,
                num_nodes=network_num_nodes,
                k=network_k if network_k is not None else 4,
                p=network_p if network_p is not None else 0.1,
                edge_probability=network_edge_probability if network_edge_probability is not None else 0.1,
                m=network_m if network_m is not None else 2,
                seed=network_seed,
                directed=network_directed if network_directed is not None else False
            )
        else:
            network_config = None

        # Experiment-specific configs (optional sections)
        pg_cfg = None
        if "pair_game_config" in data and data.get("pair_game_config") is not None:
            pg = data.get("pair_game_config")
            # require explicit fields inside pair_game_config (no defaults)
            if "matrix_size" not in pg or pg.get("matrix_size") is None:
                missing.append("pair_game_config.matrix_size")
            if "save_heatmap" not in pg or pg.get("save_heatmap") is None:
                missing.append("pair_game_config.save_heatmap")
            if "save_statistics" not in pg or pg.get("save_statistics") is None:
                missing.append("pair_game_config.save_statistics")
            if not any(m.startswith("pair_game_config") for m in missing):
                pg_cfg = PairGameConfig(
                    matrix_size=pg["matrix_size"],
                    save_heatmap=pg["save_heatmap"],
                    save_statistics=pg["save_statistics"]
                )

        ng_cfg = None
        if "network_game_config" in data and data.get("network_game_config") is not None:
            ng = data.get("network_game_config")
            # require explicit fields inside network_game_config (no defaults)
            if "network_types" not in ng or ng.get("network_types") is None:
                missing.append("network_game_config.network_types")
            if "personality_scenarios" not in ng or ng.get("personality_scenarios") is None:
                missing.append("network_game_config.personality_scenarios")
            if "save_network_evolution" not in ng or ng.get("save_network_evolution") is None:
                missing.append("network_game_config.save_network_evolution")
            if "save_cooperation_metrics" not in ng or ng.get("save_cooperation_metrics") is None:
                missing.append("network_game_config.save_cooperation_metrics")
            if not any(m.startswith("network_game_config") for m in missing):
                ng_cfg = NetworkGameConfig(
                    network_types=ng["network_types"],
                    personality_scenarios=ng["personality_scenarios"],
                    seed=ng.get("seed"),
                    save_network_evolution=ng["save_network_evolution"],
                    save_cooperation_metrics=ng["save_cooperation_metrics"]
                )

        # Other top-level required fields
        if "experiment_type" not in data or data.get("experiment_type") is None:
            missing.append("experiment_type")
        if "name" not in data or data.get("name") is None:
            missing.append("name")
        if "description" not in data or data.get("description") is None:
            missing.append("description")
        if "output_dir" not in data or data.get("output_dir") is None:
            missing.append("output_dir")

        if "visualization_config" not in data or data.get("visualization_config") is None:
            missing.append("visualization_config")

        # parsing-level missing fields are fatal for parsing
        if missing:
            raise ValueError(f"Missing required configuration fields: {', '.join(sorted(set(missing)))}")

        # All required values present -> construct dataclasses (use values from config)
        llm_config = LLMConfig(
            provider=llm_provider,
            model_name=llm_model,
            api_key=llm_api_key,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            timeout=llm_timeout,
            kwargs=llm_kwargs or {}
        )

        game_config = GameConfig(
            payoff_matrix=game_payoff,
            num_rounds=game_num_rounds,
            num_repetitions=game_num_repetitions,
            random_seed=game_random_seed
        )

        experiment_type = ExperimentType(data["experiment_type"])

        experiment_config = ExperimentConfig(
            experiment_type=experiment_type,
            name=data["name"],
            description=data["description"],
            llm=llm_config,
            game=game_config,
            network=network_config,
            pair_game_config=pg_cfg,
            network_game_config=ng_cfg,
            output_dir=data["output_dir"],
            visualization_config=data["visualization_config"]
        )

        return experiment_config
    
    def _serialize_config(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Serialize configuration object"""
        data = {
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
            "network": None if config.network is None else {
                "network_type": config.network.network_type,
                "num_nodes": config.network.num_nodes,
                "k": config.network.k,
                "p": config.network.p,
                "edge_probability": config.network.edge_probability,
                "m": config.network.m,
                "seed": config.network.seed,
                "directed": config.network.directed
            },
        }
        # serialize pair_game_config if present
        if config.pair_game_config:
            data["pair_game_config"] = {
                "matrix_size": config.pair_game_config.matrix_size,
                "save_heatmap": config.pair_game_config.save_heatmap,
                "save_statistics": config.pair_game_config.save_statistics,
            }
        else:
            data["pair_game_config"] = None
        # serialize network_game_config if present
        if config.network_game_config:
            data["network_game_config"] = {
                "network_types": config.network_game_config.network_types,
                "personality_scenarios": config.network_game_config.personality_scenarios,
                "seed": config.network_game_config.seed,
                "save_network_evolution": config.network_game_config.save_network_evolution,
                "save_cooperation_metrics": config.network_game_config.save_cooperation_metrics,
            }
        else:
            data["network_game_config"] = None

        # remaining fields
        data.update({
            "output_dir": config.output_dir,
            "visualization_config": config.visualization_config
        })
        return data
    
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
            pair_game_config=PairGameConfig(
                matrix_size=16,
                save_heatmap=True,
                save_statistics=True
            )
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
            network_game_config=NetworkGameConfig(
                network_types=["regular", "small_world_0.1", "small_world_0.5", "random", "scale_free"],
                personality_scenarios=["uniform", "single_ENTJ", "clustered"],
                save_network_evolution=True,
                save_cooperation_metrics=True
            )
        )
        
        # Save configuration files
        self.save_config(pair_game_config, "pair_game.yaml")
        self.save_config(network_game_config, "network_game.yaml")
        
        self.logger.info("Default config files created")
    
    def validate_config(self, config: ExperimentConfig) -> List[str]:
        """Validate configuration"""
        errors = []
        
        # Only support the two known experiment types
        if config.experiment_type not in (ExperimentType.PAIR_GAME, ExperimentType.NETWORK_GAME):
            errors.append(f"Unsupported experiment type: {config.experiment_type}")
            return errors

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
        
        # Experiment-type specific requirements
        if config.experiment_type == ExperimentType.PAIR_GAME:
            # pair_game_config is mandatory
            if not config.pair_game_config:
                errors.append("pair_game_config must be provided for PAIR_GAME experiments")
            else:
                if config.pair_game_config.matrix_size <= 0:
                    errors.append("pair_game_config.matrix_size must be positive")
        else:  # NETWORK_GAME
            # network_game_config and network are mandatory
            if not config.network_game_config:
                errors.append("network_game_config must be provided for NETWORK_GAME experiments")
            else:
                if not config.network_game_config.network_types:
                    errors.append("network_game_config.network_types must be specified")
                if not config.network_game_config.personality_scenarios:
                    errors.append("network_game_config.personality_scenarios must be specified")
            # network config presence and basic validation
            if not config.network:
                errors.append("network configuration must be provided for NETWORK_GAME experiments")
            else:
                if config.network.num_nodes <= 0:
                    errors.append("Number of nodes must be positive")
                if config.network.k >= config.network.num_nodes:
                    errors.append("k must be less than number of nodes")
        
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
            }
        except Exception as e:
            return {"error": str(e)}


# Function to create default configuration files
def create_default_configs():
    """Create default configuration files"""
    config_manager = ConfigManager()
    config_manager.create_default_configs()
    return config_manager

