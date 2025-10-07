# Experiment Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Create Default Configurations

```bash
python main.py --create-configs
```

### 3. Run Quick Test

```bash
# Use Mock LLM for quick testing
python main.py --experiment pair_game --config configs/quick_test_pair_game.yaml
```

## Experiment Types

### Experiment 1: Pair Game

#### Objective
Explore the behavioral patterns of 16 MBTI personality types in repeated Prisoner's Dilemma, generating 16x16 cooperation rate and payoff matrices.

#### Configuration
```yaml
# configs/pair_game.yaml
experiment_type: pair_game
name: "MBTI Pair Game Experiment"

llm:
  provider: "mock"  # or "openai", "anthropic", "google"
  model_name: "mock-model"
  api_key: null
  kwargs:
    temperature: 0.7
    max_tokens: 50

game:
  num_rounds: 100      # Number of rounds per pair
  num_repetitions: 20  # Number of repetitions
  random_seed: 42      # Random seed

pair_game_config:
  matrix_size: 16
  save_heatmap: true
  save_statistics: true
```

#### Run
```bash
python main.py --experiment pair_game --config configs/pair_game.yaml
```

#### Output Files
- `cooperation_matrix.csv`: 16x16 cooperation rate matrix
- `payoff_matrix.csv`: 16x16 payoff matrix  
- `detailed_results.csv`: Detailed results (includes mean and std)
- `analysis_results.json`: Full analysis results
- `experiment_config.json`: Experiment configuration record
- **Visualization files:**
  - `cooperation_heatmap.png`: Cooperation rate heatmap
  - `payoff_heatmap.png`: Payoff heatmap
  - `cooperation_distribution.png`: Cooperation rate distribution
  - `personality_cooperation_ranking.png`: Personality cooperation ranking
  - `personality_payoff_ranking.png`: Personality payoff ranking
  - `mbti_dimension_analysis.png`: MBTI dimension analysis

#### Features
- **Concurrent Execution:** Uses asynchronous concurrency for efficiency, supports semaphore control (default 10)
- **Smart Retry:** Automatically retries LLM parsing failures (up to 10 times) with exponential backoff
- **Symmetric Matrix:** Each personality pair runs once, symmetric results auto-filled
- **Detailed Statistics:** Includes mean, std, and other metrics for cooperation and payoff

### Experiment 2: Network Game

#### Objective
Study the impact of different network topologies and personality distributions on network game evolution.

#### Configuration
```yaml
# configs/network_game.yaml
experiment_type: network_game
name: "MBTI Network Game Experiment"

network:
  network_type: "small_world"
  num_nodes: 50
  k: 4
  p: 0.1

network_game_config:
  network_types: ["regular", "small_world_0.1", "small_world_0.5", "random"]
  personality_scenarios: ["uniform", "single_ENTJ", "clustered"]
  save_network_evolution: true
  snapshot_rounds: [1, 25, 50, 75, 100]
```

#### Run
```bash
python main.py --experiment network_game --config configs/network_game.yaml
```

#### Output
- `network_results.json`: Detailed network game results
- `network_analysis.json`: Network analysis results
- `network_evolution_*.png`: Network evolution plots
- `network_comparison.png`: Network type comparison
- `network_snapshot_*.png`: Network snapshots

## Advanced Configuration

### LLM Configuration

#### OpenAI
```yaml
llm:
  provider: "openai"
  model_name: "gpt-4"
  api_key: "your-openai-api-key"
  temperature: 0.7
  max_tokens: 50
```

#### Anthropic
```yaml
llm:
  provider: "anthropic"
  model_name: "claude-3-sonnet-20240229"
  api_key: "your-anthropic-api-key"
  temperature: 0.7
```

#### Google
```yaml
llm:
  provider: "google"
  model_name: "gemini-pro"
  api_key: "your-google-api-key"
  temperature: 0.7
```

### Network Configuration

#### Regular Network
```yaml
network:
  network_type: "regular"
  num_nodes: 50
  k: 4  # Number of neighbors per node
```

#### Small World Network
```yaml
network:
  network_type: "small_world"
  num_nodes: 50
  k: 4
  p: 0.1  # Rewiring probability
```

#### Random Network
```yaml
network:
  network_type: "random"
  num_nodes: 50
  edge_probability: 0.1
```

#### Scale-Free Network
```yaml
network:
  network_type: "scale_free"
  num_nodes: 50
  m: 2  # Number of edges for new node
```

### Personality Distribution Configuration

#### Uniform Distribution
```yaml
personality_distribution:
  distribution_type: "uniform"
```

#### Single Type
```yaml
personality_distribution:
  distribution_type: "single"
  single_type: "ENTJ"
```

#### Clustered Distribution
```yaml
personality_distribution:
  distribution_type: "clustered"
  cluster_config:
    num_clusters: 4
    cluster_size_variance: 0.2
```

## Custom Experiments

### Create Custom Configuration

```python
from src.config.config_manager import ConfigManager, ExperimentConfig, LLMConfig, GameConfig

# Create custom config
config = ExperimentConfig(
    experiment_type=ExperimentType.PAIR_GAME,
    name="Custom Experiment",
    description="My custom experiment",
    llm=LLMConfig(
        provider="openai",
        model_name="gpt-4",
        api_key="your-key"
    ),
    game=GameConfig(
        num_rounds=200,
        num_repetitions=50
    )
)

# Save config
config_manager = ConfigManager()
config_manager.save_config(config, "my_custom_config.yaml")
```

### Run Experiment Directly

```python
import asyncio
from src.experiments.pair_game_experiment import run_pair_game_experiment

# Run experiment
results = await run_pair_game_experiment("configs/my_config.yaml")
```

### Custom Personality Types

```python
from src.agents.mbti_personalities import MBTIPersonality, MBTIType

# Create custom personality
class CustomPersonality(MBTIPersonality):
    def _get_prompt_template(self) -> str:
        return "Your custom prompt template here..."

# Use custom personality
personality = CustomPersonality(MBTIType.INTJ)
```

### Custom Network Topology

```python
from src.networks.network_generator import NetworkGenerator, NetworkConfig, NetworkType

# Create custom network
def create_custom_network(num_nodes):
    G = nx.Graph()
    # Add nodes and edges logic
    return G

# Register custom network type
NetworkType.CUSTOM = "custom"
```

## Result Analysis

### Load and Analyze Structure

```python
import pandas as pd
import numpy as np
import json

# Load results
cooperation_matrix = pd.read_csv("results/cooperation_matrix.csv", index_col=0)

# Load detailed results
detailed_results = pd.read_csv("results/pair_game/detailed_results.csv")

# Load analysis results
with open("results/pair_game/analysis_results.json", 'r', encoding='utf-8') as f:
    analysis = json.load(f)

# Basic statistics
print("Average cooperation rate:", cooperation_matrix.values.mean())
print("Std deviation:", cooperation_matrix.values.std())
print("Max cooperation rate:", cooperation_matrix.values.max())
print("Min cooperation rate:", cooperation_matrix.values.min())
```

### Personality Analysis

```python
# Analyze by personality type
personality_rates = results.mean(axis=1)
print("Most cooperative personality:", personality_rates.idxmax())
print("Least cooperative personality:", personality_rates.idxmin())

# Analyze by MBTI dimension
E_types = [t for t in results.index if t.startswith('E')]
I_types = [t for t in results.index if t.startswith('I')]
E_rate = results.loc[E_types].values.mean()
I_rate = results.loc[I_types].values.mean()
print(f"E-type average cooperation rate: {E_rate:.3f}")
print(f"I-type average cooperation rate: {I_rate:.3f}")
```

### Network Analysis

```python
import json

# Load network results
with open("results/network_results.json", 'r') as f:
    network_results = json.load(f)

# Analyze different network types
for network_type, scenarios in network_results.items():
    for scenario, results in scenarios.items():
        evolution_data = results["evolution_data"]
        final_cooperation = evolution_data[-1]["cooperation_rate"]
        print(f"{network_type} - {scenario}: {final_cooperation:.3f}")
```

## Performance Optimization

### Concurrency Control

```python
# Adjust concurrency in PairGameExperiment
semaphore = asyncio.Semaphore(5)  # Lower concurrency to reduce resource usage
```

### Memory Optimization

```python
# Batch process large datasets
def process_large_dataset(data, batch_size=1000):
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        yield process_batch(batch)
```

### Cache LLM Responses

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_llm_response(prompt_hash):
    return llm.generate_response(prompt)

def get_prompt_hash(prompt):
    return hashlib.md5(prompt.encode()).hexdigest()
```

## Troubleshooting

### Common Issues

1. **LLM API Error**
   - Check API key
   - Ensure network connection
   - Check API quota

2. **Out of Memory**
   - Reduce network node count
   - Reduce experiment repetitions
   - Use batch processing

3. **Inconsistent Results**
   - Set fixed random seed
   - Check config parameters
   - Validate data input

### Debug Mode

```bash
# Enable detailed logging
python main.py --experiment pair_game --log-level DEBUG

# Use Mock LLM for quick testing
python main.py --experiment pair_game --config configs/quick_test_pair_game.yaml
```

### Performance Monitoring

```python
import time
import psutil
import logging

# Set logging to monitor performance
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Reduce HTTP request logs

# Monitor resource usage
def monitor_performance():
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    # Run experiment
    results = run_experiment()
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss
    
    print(f"Run time: {end_time - start_time:.2f}s")
    print(f"Memory usage: {(end_memory - start_memory) / 1024 / 1024:.2f}MB")
    
    return results
```

### Error Recovery

```python
# Resume experiment after failure
def resume_experiment(checkpoint_file):
    """Resume experiment from checkpoint"""
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        # Restore experiment state
        return checkpoint
    except FileNotFoundError:
        print("Checkpoint file not found, starting experiment from scratch")
        return None
```

## Extensions

### Add New MBTI Personality

```python
from src.agents.mbti_personalities import MBTIType, MBTIPersonality

# Extend MBTIType enum (if needed)
class ExtendedMBTIType(MBTIType):
    CUSTOM_TYPE = "CUST"

class CustomMBTIPersonality(MBTIPersonality):
    def get_decision_prompt(self, history, opponent_type, is_player1=True):
        # Implement custom logic
        return custom_prompt
```

### Custom Analyzer

```python
from src.analysis.statistics import CooperationAnalyzer

class CustomAnalyzer(CooperationAnalyzer):
    def analyze_custom_metrics(self, data):
        # Implement custom analysis logic
        return custom_analysis_results
```

### Custom Visualization

```python
from src.visualization.plotter import PairGamePlotter

class CustomPlotter(PairGamePlotter):
    def plot_custom_visualization(self, data, **kwargs):
        # Implement custom visualization
        return plot_file_path
```
