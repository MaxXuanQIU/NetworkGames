# API Reference

## 核心模块

### MBTI人格系统

#### `MBTIType`
```python
from src.agents.mbti_personalities import MBTIType

# 16种MBTI人格类型
MBTIType.INTJ  # 建筑师
MBTIType.INTP  # 思想家
MBTIType.ENTJ  # 指挥官
MBTIType.ENTP  # 辩论家
# ... 等等
```

#### `MBTIPersonality`
```python
from src.agents.mbti_personalities import MBTIPersonality

# 创建人格对象
personality = MBTIPersonality(MBTIType.INTJ)

# 获取决策prompt
prompt = personality.get_decision_prompt(game_history, opponent_type)
```

### LLM接口

#### `LLMManager`
```python
from src.llm.llm_interface import LLMManager, LLMFactory

# 创建LLM管理器
llm_manager = LLMManager()

# 添加LLM实例
llm = LLMFactory.create_llm(
    provider=LLMProvider.OPENAI,
    model_name="gpt-5-nano",
    api_key="your-api-key"
)
llm_manager.add_llm("default", llm)

# 生成响应
response = await llm_manager.generate_response("default", prompt)
```

### 博弈系统

#### `PrisonersDilemma`
```python
from src.games.prisoners_dilemma import PrisonersDilemma, Action

# 创建博弈实例
game = PrisonersDilemma()

# 进行一轮博弈
result = game.play_round(Action.COOPERATE, Action.DEFECT)

# 进行多轮博弈
history = game.play_game(player1_actions, player2_actions)
```

### 网络生成器

#### `NetworkGenerator`
```python
from src.networks.network_generator import NetworkGenerator, NetworkConfig, NetworkType

# 创建网络生成器
generator = NetworkGenerator()

# 配置网络
config = NetworkConfig(
    network_type=NetworkType.SMALL_WORLD,
    num_nodes=50,
    k=4,
    p=0.1
)

# 生成网络
G = generator.generate_network(config)
```

### 配置管理

#### `ConfigManager`
```python
from src.config.config_manager import ConfigManager

# 创建配置管理器
config_manager = ConfigManager()

# 加载配置
config = config_manager.load_config("configs/pair_game.yaml")

# 保存配置
config_manager.save_config(config, "my_config.yaml")

# 验证配置
errors = config_manager.validate_config(config)
```

## 实验模块

### 两人博弈实验

```python
from src.experiments.pair_game_experiment import PairGameExperiment

# 创建实验
experiment = PairGameExperiment(config, llm_manager)

# 运行实验
results = await experiment.run_experiment()
```

### 网络博弈实验

```python
from src.experiments.network_game_experiment import NetworkGameExperiment

# 创建实验
experiment = NetworkGameExperiment(config, llm_manager)

# 运行实验
results = await experiment.run_experiment()
```

## 分析模块

### 合作行为分析

```python
from src.analysis.statistics import CooperationAnalyzer

# 创建分析器
analyzer = CooperationAnalyzer()

# 分析合作指标
metrics = analyzer.calculate_cooperation_metrics(cooperation_data)

# 比较人格类型
comparison = analyzer.compare_personalities(personality_data)
```

### 网络分析

```python
from src.networks.network_generator import NetworkAnalyzer

# 创建网络分析器
analyzer = NetworkAnalyzer()

# 分析网络
analysis = analyzer.analyze_network(G)

# 比较网络
comparison = analyzer.compare_networks(networks)
```

## 可视化模块

### 两人博弈可视化

```python
from src.visualization.plotter import PairGamePlotter

# 创建绘图器
plotter = PairGamePlotter()

# 绘制热力图
plotter.plot_cooperation_heatmap(matrix, types)

# 绘制分布图
plotter.plot_cooperation_distribution(rates)
```

### 网络博弈可视化

```python
from src.visualization.plotter import NetworkGamePlotter

# 创建绘图器
plotter = NetworkGamePlotter()

# 绘制网络演化
plotter.plot_network_evolution(evolution_data)

# 绘制网络快照
plotter.plot_network_snapshot(G, node_colors)
```

## 结果管理

### `ResultManager`

```python
from src.utils.result_manager import ResultManager

# 创建结果管理器
result_manager = ResultManager("results")

# 保存结果
experiment_dir = result_manager.save_experiment_results(
    "pair_game", results, config
)

# 加载结果
loaded_results = result_manager.load_experiment_results(experiment_id)

# 列出实验
experiments = result_manager.list_experiments()
```

## 命令行接口

### 运行实验

```bash
# 使用Mock LLM进行快速测试
python main.py --experiment pair_game --config configs/quick_test_pair_game.yaml

# 运行两人博弈实验
python main.py --experiment pair_game --config configs/pair_game.yaml

# 运行网络博弈实验
python main.py --experiment network_game --config configs/network_game.yaml

# 使用不同的LLM，请在configs/下进行配置
```

### 配置管理

```bash
# 创建默认配置
python main.py --create-configs

# 列出配置
python main.py --list-configs

# 验证配置
python main.py --validate-config configs/pair_game.yaml
```


## 配置格式

### 实验配置

```yaml
experiment_type: pair_game
name: "MBTI Pair Game Experiment"
description: "16x16 MBTI personality matrix in repeated prisoner's dilemma"

llm:
  provider: "openai"
  model_name: "gpt-5-nano"
  api_key: "your-api-key"
  temperature: 0.7

game:
  num_rounds: 100
  num_repetitions: 20
  random_seed: 42

network:
  network_type: "small_world"
  num_nodes: 50
  k: 4
  p: 0.1

personality_distribution:
  distribution_type: "uniform"
```

## 数据格式

### 实验结果

```json
{
  "matrix_results": {
    "cooperation_matrix": [[0.5, 0.6, ...], ...],
    "payoff_matrix": [[3.2, 2.8, ...], ...],
    "personality_types": ["INTJ", "INTP", ...]
  },
  "analysis_results": {
    "basic_statistics": {...},
    "personality_analysis": {...},
    "dimension_analysis": {...}
  },
  "visualization_results": {
    "heatmap": "path/to/heatmap.png",
    "distribution": "path/to/distribution.png"
  }
}
```

### 网络结果

```json
{
  "network_results": {
    "small_world": {
      "uniform": {
        "evolution_data": [...],
        "network_analysis": {...},
        "personality_assignment": {...}
      }
    }
  },
  "analysis_results": {
    "network_comparison": {...},
    "scenario_comparison": {...}
  }
}
```
