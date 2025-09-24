"""
配置管理系统
支持YAML配置文件的加载、验证和管理
"""

import yaml
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import logging
from enum import Enum


class ExperimentType(Enum):
    """实验类型"""
    PAIR_GAME = "pair_game"
    NETWORK_GAME = "network_game"


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: str
    model_name: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 50
    timeout: int = 30
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameConfig:
    """博弈配置"""
    payoff_matrix: Dict[str, Dict[str, List[float]]] = field(default_factory=lambda: {
        "COOPERATE": {"COOPERATE": [3, 3], "DEFECT": [0, 5]},
        "DEFECT": {"COOPERATE": [5, 0], "DEFECT": [1, 1]}
    })
    num_rounds: int = 100
    num_repetitions: int = 20
    random_seed: Optional[int] = None


@dataclass
class NetworkConfig:
    """网络配置"""
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
    """人格分布配置"""
    distribution_type: str  # "uniform", "clustered", "single", "custom"
    single_type: Optional[str] = None  # 当distribution_type为"single"时使用
    cluster_config: Optional[Dict[str, Any]] = None  # 聚类分布配置
    custom_distribution: Optional[Dict[str, float]] = None  # 自定义分布


@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_type: ExperimentType
    name: str
    description: str
    
    # 基础配置
    llm: LLMConfig
    game: GameConfig
    network: NetworkConfig
    personality_distribution: PersonalityDistributionConfig
    
    # 实验特定配置
    pair_game_config: Optional[Dict[str, Any]] = None
    network_game_config: Optional[Dict[str, Any]] = None
    
    # 输出配置
    output_dir: str = "results"
    save_detailed_results: bool = True
    save_visualizations: bool = True
    save_network_snapshots: bool = True
    
    # 可视化配置
    visualization_config: Dict[str, Any] = field(default_factory=lambda: {
        "figsize": [12, 8],
        "dpi": 300,
        "style": "seaborn-v0_8",
        "color_palette": "Set2"
    })


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "configs"):
        # 统一约定：命令行 --config 仅支持传入形如
        # "configs/xxx.yaml" 的路径（或绝对路径）。
        # 这里保存项目根目录与配置目录，供后续拼接使用。
        project_root = Path(__file__).parent.parent.parent
        self.project_root = project_root
        # 配置目录固定为 <project_root>/configs
        self.config_dir = project_root / config_dir if not Path(config_dir).is_absolute() else Path(config_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._ensure_config_dir()
    
    def _ensure_config_dir(self):
        """确保配置目录存在"""
        self.config_dir.mkdir(exist_ok=True)
    
    def load_config(self, config_file: str) -> ExperimentConfig:
        """加载配置文件
        统一规则：
        - 推荐与仅支持：传入形如 "configs/xxx.yaml" 的相对路径，或绝对路径
        - 不再支持仅文件名（如 "pair_game.yaml"）
        """
        cfg_path = Path(config_file)
        if cfg_path.is_absolute():
            config_path = cfg_path
        else:
            # 仅接受以 configs/ 开头的相对路径
            # 兼容 Windows 反斜杠写法
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
        """保存配置文件
        要求传入形如 "configs/xxx.yaml" 或绝对路径
        """
        cfg_path = Path(config_file)
        if cfg_path.is_absolute():
            config_path = cfg_path
        else:
            cfg_str = str(cfg_path).replace("\\", "/")
            if cfg_str.startswith("configs/"):
                config_path = self.project_root / cfg_str
            else:
                # 默认保存到标准目录
                config_path = self.config_dir / cfg_path.name
        
        config_data = self._serialize_config(config)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"Config saved to: {config_path}")
    
    def _parse_config(self, data: Dict[str, Any]) -> ExperimentConfig:
        """解析配置数据"""
        # 解析LLM配置
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
        
        # 解析博弈配置
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
        
        # 解析网络配置
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
        
        # 解析人格分布配置
        personality_data = data.get("personality_distribution", {})
        personality_config = PersonalityDistributionConfig(
            distribution_type=personality_data.get("distribution_type", "uniform"),
            single_type=personality_data.get("single_type"),
            cluster_config=personality_data.get("cluster_config"),
            custom_distribution=personality_data.get("custom_distribution")
        )
        
        # 解析实验配置
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
        """序列化配置对象"""
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
        """创建默认配置文件"""
        # 两人博弈实验配置
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
        
        # 网络博弈实验配置
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
        
        # 保存配置文件
        self.save_config(pair_game_config, "pair_game.yaml")
        self.save_config(network_game_config, "network_game.yaml")
        
        self.logger.info("Default config files created")
    
    def validate_config(self, config: ExperimentConfig) -> List[str]:
        """验证配置的有效性"""
        errors = []
        
        # 验证实验类型
        if config.experiment_type not in ExperimentType:
            errors.append(f"Invalid experiment type: {config.experiment_type}")
        
        # 验证LLM配置
        if not config.llm.provider:
            errors.append("LLM provider is required")
        
        if not config.llm.model_name:
            errors.append("LLM model name is required")
        
        # 验证博弈配置
        if config.game.num_rounds <= 0:
            errors.append("Number of rounds must be positive")
        
        if config.game.num_repetitions <= 0:
            errors.append("Number of repetitions must be positive")
        
        # 验证网络配置
        if config.network.num_nodes <= 0:
            errors.append("Number of nodes must be positive")
        
        if config.network.k >= config.network.num_nodes:
            errors.append("k must be less than number of nodes")
        
        # 验证人格分布配置
        if config.personality_distribution.distribution_type == "single":
            if not config.personality_distribution.single_type:
                errors.append("Single type must be specified for single distribution")
        
        return errors
    
    def list_configs(self) -> List[str]:
        """列出所有配置文件"""
        config_files = []
        for file_path in self.config_dir.glob("*.yaml"):
            config_files.append(file_path.name)
        return sorted(config_files)
    
    def get_config_info(self, config_file: str) -> Dict[str, Any]:
        """获取配置文件信息"""
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


# 创建默认配置文件的函数
def create_default_configs():
    """创建默认配置文件"""
    config_manager = ConfigManager()
    config_manager.create_default_configs()
    return config_manager
