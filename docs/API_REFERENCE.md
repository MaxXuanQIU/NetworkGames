# API Reference

## Core Modules

### MBTI Personality System

#### `MBTIType`
```python
from src.agents.mbti_personalities import MBTIType

# 16 MBTI personality types
MBTIType.INTJ  # Architect
MBTIType.INTP  # Thinker
MBTIType.ENTJ  # Commander
MBTIType.ENTP  # Debater
# ... etc.
```

#### `MBTIPersonality`
```python
from src.agents.mbti_personalities import MBTIPersonality

# Create a personality object
personality = MBTIPersonality(MBTIType.INTJ)

# Get decision prompt
prompt = personality.get_decision_prompt(game_history, opponent_type)
```

### LLM Interface

#### `LLMManager`
```python
from src.llm.llm_interface import LLMManager, LLMFactory

# Create LLM manager
llm_manager = LLMManager()

# Add LLM instance
llm = LLMFactory.create_llm(
  provider=LLMProvider.OPENAI,
  model_name="gpt-5-nano",
  api_key="your-api-key"
)
llm_manager.add_llm("default", llm)

# Generate response
response = await llm_manager.generate_response("default", prompt)
```

### Game System

#### `PrisonersDilemma`
```python
from src.games.prisoners_dilemma import PrisonersDilemma, Action

# Create game instance
game = PrisonersDilemma()

# Play one round
result = game.play_round(Action.COOPERATE, Action.DEFECT)

# Play multiple rounds
history = game.play_game(player1_actions, player2_actions)
```

### Network Generator

#### `NetworkGenerator`
```python
from src.networks.network_generator import NetworkGenerator, NetworkConfig, NetworkType

# Create network generator
generator = NetworkGenerator()

# Configure network
config = NetworkConfig(
  network_type=NetworkType.SMALL_WORLD,
  num_nodes=50,
  k=4,
  p=0.1
)

# Generate network
G = generator.generate_network(config)
```

### Configuration Management

#### `ConfigManager`
```python
from src.config.config_manager import ConfigManager

# Create configuration manager
config_manager = ConfigManager()

# Load configuration
config = config_manager.load_config("configs/pair_game.yaml")

# Save configuration
config_manager.save_config(config, "my_config.yaml")

# Validate configuration
errors = config_manager.validate_config(config)
```

## Experiment Modules

### Two-Player Game Experiment

```python
from src.experiments.pair_game_experiment import PairGameExperiment

# Create experiment
experiment = PairGameExperiment(config, llm_manager)

# Run experiment
results = await experiment.run_experiment()
```

### Network Game Experiment

```python
from src.experiments.network_game_experiment import NetworkGameExperiment

# Create experiment
experiment = NetworkGameExperiment(config, llm_manager)

# Run experiment
results = await experiment.run_experiment()
```

## Analysis Modules

### Cooperation Behavior Analysis

```python
from src.analysis.statistics import CooperationAnalyzer

# Create analyzer
analyzer = CooperationAnalyzer()

# Analyze cooperation metrics
metrics = analyzer.calculate_cooperation_metrics(cooperation_data)

# Compare personality types
comparison = analyzer.compare_personalities(personality_data)
```

### Network Analysis

```python
from src.networks.network_generator import NetworkAnalyzer

# Create network analyzer
analyzer = NetworkAnalyzer()

# Analyze network
analysis = analyzer.analyze_network(G)

# Compare networks
comparison = analyzer.compare_networks(networks)
```

## Visualization Modules

### Two-Player Game Visualization

```python
from src.visualization.plotter import PairGamePlotter

# Create plotter
plotter = PairGamePlotter()

# Plot heatmap
plotter.plot_cooperation_heatmap(matrix, types)

# Plot distribution
plotter.plot_cooperation_distribution(rates)
```

### Network Game Visualization

```python
from src.visualization.plotter import NetworkGamePlotter

# Create plotter
plotter = NetworkGamePlotter()

# Plot network evolution
plotter.plot_network_evolution(evolution_data)

# Plot network snapshot
plotter.plot_network_snapshot(G, node_colors)
```

## Result Management

### `ResultManager`

```python
from src.utils.result_manager import ResultManager

# Create result manager
result_manager = ResultManager("results")

# Save results
experiment_dir = result_manager.save_experiment_results(
  "pair_game", results, config
)

# Load results
loaded_results = result_manager.load_experiment_results(experiment_id)

# List experiments
experiments = result_manager.list_experiments()
```

## Command Line Interface

### Run Experiment

```bash
# Use Mock LLM for quick testing
python main.py --experiment pair_game --config configs/quick_test_pair_game.yaml

# Run two-player game experiment
python main.py --experiment pair_game --config configs/pair_game.yaml

# Run network game experiment
python main.py --experiment network_game --config configs/network_game.yaml

# To use different LLMs, configure in configs/
```

### Configuration Management

```bash
# Create default configs
python main.py --create-configs

# List configs
python main.py --list-configs

# Validate config
python main.py --validate-config configs/pair_game.yaml
```

## Configuration Format

### Experiment Configuration

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

## Data Format

### Experiment Results

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

### Network Results

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
