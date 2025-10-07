# LLM Network Games Framework

A reproducible and customizable framework for studying LLM behavior in network games.

## Project Overview

This project aims to explore behavioral patterns of different LLM models in network games, focusing on:
- Behavioral differences based on MBTI 16 personality types
- Cooperation and betrayal strategies in repeated Prisoner's Dilemma
- Impact of network topology on game behavior
- Influence of personality distribution on network evolution

## Main Features

- **Multi-LLM Support**: Supports various representative LLM models (OpenAI, Anthropic, Google, Mock)
- **MBTI Personality System**: Implements typical personality traits for all 16 MBTI types
- **Network Games**: Supports multiple network topologies (regular, small-world, random, scale-free)
- **Configurable Experiments**: Manage experiment parameters via YAML config files
- **Rich Visualization**: Heatmaps, network graphs, time series, and other statistical charts
- **Reproducibility**: Complete random seed control and result output
- **Statistical Analysis**: Comprehensive statistical tests and effect size analysis

## Experiment Design

### Experiment 1: Pairwise Game
- **Goal**: Baseline behavior of 16x16 personality combinations in repeated Prisoner's Dilemma
- **Design**: All MBTI personality pairs play repeated Prisoner's Dilemma
- **Parameters**: Adjustable game rounds (default 100) and repetitions (default 20)
- **Output**: 16x16 cooperation rate heatmap, statistical significance analysis, personality ranking

### Experiment 2: Network Game
- **Goal**: Network game evolution under different topologies and personality distributions
- **Variable 1**: Network topology (regular, small-world, random)
- **Variable 2**: Personality distribution (uniform, single type, clustered)
- **Variable 3**: Interaction rounds (default 100)
- **Output**: Network evolution animation, cooperation rate time series, network snapshots

## Quick Start

### 1. Install Dependencies

```bash
# Clone the project
git clone https://github.com/MaxXuanQIU/NetworkGames.git
cd LLM-Network-Games

# Install dependencies
pip install -r requirements.txt
```

### 2. Create Default Configs

```bash
python main.py --create-configs
```

### 3. Run Quick Test

```bash
# Quick test with Mock LLM
python main.py --experiment pair_game --config configs/quick_test_pair_game.yaml
```

### 4. Run Full Experiments

```bash
# Run pairwise game experiment
python main.py --experiment pair_game

# Run network game experiment
python main.py --experiment network_game

# Note: Real LLMs require API keys, configure them in --config configs/pair_game.yaml and --config configs/network_game.yaml
```

## Project Structure

```
NetworkGames/
├── src/                           # Source code
│   ├── agents/                   # Agents
│   │   └── mbti_personalities.py # MBTI personality system
│   ├── games/                    # Game logic
│   │   └── prisoners_dilemma.py  # Prisoner's Dilemma implementation
│   ├── networks/                 # Network topology
│   │   └── network_generator.py  # Network generator
│   ├── llm/                     # LLM interface
│   │   └── llm_interface.py     # LLM abstraction layer
│   ├── analysis/                # Data analysis
│   │   └── statistics.py        # Statistical analysis
│   ├── visualization/           # Visualization
│   │   └── plotter.py          # Plotting tools
│   ├── config/                  # Config management
│   │   └── config_manager.py    # Config manager
│   ├── experiments/             # Experiment implementations
│   │   ├── pair_game_experiment.py      # Pairwise game experiment
│   │   └── network_game_experiment.py   # Network game experiment
│   └── utils/                   # Utility functions
│       └── result_manager.py    # Result manager
├── configs/                     # Config files
│   ├── pair_game.yaml          # Pairwise game config
│   └── network_game.yaml       # Network game config
├── results/                    # Output results
├── docs/                      # Documentation
│   ├── API_REFERENCE.md       # API reference
│   └── EXPERIMENT_GUIDE.md    # Experiment guide
├── main.py                    # Main entry point
├── requirements.txt           # Dependency list
└── README.md                 # Project description
```

## Usage Examples

### Basic Usage

```python
import asyncio
from src.experiments.pair_game_experiment import run_pair_game_experiment

# Run pairwise game experiment
results = await run_pair_game_experiment("configs/pair_game.yaml")
print("Experiment completed! Results saved in results/ directory")
```

### Custom Config

```python
from src.config.config_manager import ConfigManager, ExperimentConfig

# Create custom config
config = ExperimentConfig(
  experiment_type="pair_game",
  name="My Custom Experiment",
  llm_provider="openai",
  llm_model="gpt-4",
  num_rounds=200,
  num_repetitions=50
)

# Save config
config_manager = ConfigManager()
config_manager.save_config(config, "my_config.yaml")
```

### Result Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
results = pd.read_csv("results/cooperation_matrix.csv", index_col=0)

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(results, annot=True, cmap='RdYlBu_r')
plt.title("MBTI Cooperation Rate Matrix")
plt.show()

# Analyze personality ranking
personality_rates = results.mean(axis=1).sort_values(ascending=False)
print("Most cooperative personality type:", personality_rates.index[0])
print("Least cooperative personality type:", personality_rates.index[-1])
```

## Config Explanation

### LLM Config

Supports multiple LLM providers:

```yaml
# OpenAI
llm:
  provider: "openai"
  model_name: "gpt-4"
  api_key: "your-openai-api-key"

# Anthropic
llm:
  provider: "anthropic"
  model_name: "claude-3-sonnet-20240229"
  api_key: "your-anthropic-api-key"

# Google
llm:
  provider: "google"
  model_name: "gemini-pro"
  api_key: "your-google-api-key"

# Mock (for testing)
llm:
  provider: "mock"
  model_name: "mock-model"
```

### Network Config

```yaml
# Small-world network
network:
  network_type: "small_world"
  num_nodes: 50
  k: 4
  p: 0.1

# Regular network
network:
  network_type: "regular"
  num_nodes: 50
  k: 4

# Random network
network:
  network_type: "random"
  num_nodes: 50
  edge_probability: 0.1
```

## Output Results

### Pairwise Game Experiment Output

- `cooperation_matrix.csv`: 16x16 cooperation rate matrix
- `payoff_matrix.csv`: 16x16 payoff matrix
- `detailed_results.csv`: Detailed result data
- `analysis_results.json`: Statistical analysis results
- `experiment_config.json`: Experiment config info
- `cooperation_heatmap.png`: Cooperation rate heatmap
- `cooperation_distribution.png`: Cooperation rate distribution
- `personality_ranking.png`: Personality cooperation ranking
- `mbti_dimension_analysis.png`: MBTI dimension analysis

### Network Game Experiment Output

- `network_results.json`: Detailed network game results
- `network_analysis.json`: Network analysis results
- `network_evolution_*.png`: Network evolution plots
- `network_comparison.png`: Network type comparison
- `network_snapshot_*.png`: Network snapshots
- `cooperation_clusters.png`: Cooperation cluster analysis

## Advanced Features

### Custom Personality Types

```python
from src.agents.mbti_personalities import MBTIPersonality, MBTIType

class CustomPersonality(MBTIPersonality):
  def _get_prompt_template(self) -> str:
    return "Your custom prompt template here..."
```

### Custom Network Topology

```python
from src.networks.network_generator import NetworkGenerator

def create_custom_network(num_nodes):
  # Implement custom network generation logic
  pass
```

### Batch Experiments

```python
# Run experiments for multiple configs
configs = ["config1.yaml", "config2.yaml", "config3.yaml"]
for config in configs:
  results = await run_experiment(config)
  # Process results
```

## Performance Optimization

- Supports asynchronous parallel processing
- Configurable batch size
- Result caching mechanism
- Memory usage optimization

## Troubleshooting

### Common Issues

1. **API key error**: Check if LLM API key is set correctly
2. **Out of memory**: Reduce network node count or repetitions
3. **Inconsistent results**: Set a fixed random seed
4. **Network connection issues**: Check network and firewall settings

### Debug Mode

```bash
# Enable detailed logs
python main.py --experiment pair_game --log-level DEBUG

# Quick test with Mock LLM
python main.py --experiment pair_game --config configs/quick_test_pair_game.yaml
```

## Contribution Guide

Contributions, bug reports, and suggestions are welcome!

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/
```

### Commit Guidelines

- Use clear commit messages
- Add appropriate tests
- Update relevant documentation

## License

Apache 2.0 License

## Citation

If you use this project in your research, please cite:

```bibtex
@software{llm_network_games,
  title={LLM Network Games Framework},
  author={Xuan Qiu},
  year={2025},
  url={https://github.com/MaxXuanQIU/NetworkGames}
}
```

## Contact

- Project homepage: https://github.com/MaxXuanQIU/NetworkGames
- Issue tracker: https://github.com/MaxXuanQIU/NetworkGames/issues
- Email: maxxuanqiu@gmail.com