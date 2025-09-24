# 实验指南

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 创建默认配置

```bash
python main.py --create-configs
```

### 3. 运行快速测试

```bash
# 使用Mock LLM进行快速测试
python main.py --experiment pair_game --config configs/quick_test.yaml
```

## 实验类型

### 实验1：两人博弈

#### 目标
探究16种MBTI人格类型在重复囚徒困境中的行为模式，生成16x16合作率矩阵。

#### 配置
```yaml
# configs/pair_game.yaml
experiment_type: pair_game
name: "MBTI Pair Game Experiment"

llm:
  provider: "mock"  # 或 "openai", "anthropic", "google"
  model_name: "mock-model"

game:
  num_rounds: 100      # 每对组合的博弈轮数
  num_repetitions: 20  # 重复实验次数
  random_seed: 42      # 随机种子

pair_game_config:
  matrix_size: 16
  save_heatmap: true
  save_statistics: true
```

#### 运行
```bash
python main.py --experiment pair_game --config configs/pair_game.yaml
```

#### 输出
- `cooperation_matrix.csv`: 16x16合作率矩阵
- `payoff_matrix.csv`: 16x16收益矩阵
- `detailed_results.csv`: 详细结果数据
- `cooperation_heatmap.png`: 合作率热力图
- `personality_ranking.png`: 人格合作率排名
- `mbti_dimension_analysis.png`: MBTI维度分析

### 实验2：网络博弈

#### 目标
研究不同网络拓扑和人格分布对网络博弈演化的影响。

#### 配置
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

#### 运行
```bash
python main.py --experiment network_game --config configs/network_game.yaml
```

#### 输出
- `network_results.json`: 网络博弈详细结果
- `network_analysis.json`: 网络分析结果
- `network_evolution_*.png`: 网络演化图
- `network_comparison.png`: 网络类型比较
- `network_snapshot_*.png`: 网络快照

## 高级配置

### LLM配置

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

### 网络配置

#### 规则网络
```yaml
network:
  network_type: "regular"
  num_nodes: 50
  k: 4  # 每个节点的邻居数
```

#### 小世界网络
```yaml
network:
  network_type: "small_world"
  num_nodes: 50
  k: 4
  p: 0.1  # 重连概率
```

#### 随机网络
```yaml
network:
  network_type: "random"
  num_nodes: 50
  edge_probability: 0.1
```

#### 无标度网络
```yaml
network:
  network_type: "scale_free"
  num_nodes: 50
  m: 2  # 新节点连接的边数
```

### 人格分布配置

#### 均匀分布
```yaml
personality_distribution:
  distribution_type: "uniform"
```

#### 单一类型
```yaml
personality_distribution:
  distribution_type: "single"
  single_type: "ENTJ"
```

#### 聚类分布
```yaml
personality_distribution:
  distribution_type: "clustered"
  cluster_config:
    num_clusters: 4
    cluster_size_variance: 0.2
```

## 自定义实验

### 创建自定义配置

```python
from src.config.config_manager import ConfigManager, ExperimentConfig, LLMConfig, GameConfig

# 创建自定义配置
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

# 保存配置
config_manager = ConfigManager()
config_manager.save_config(config, "my_custom_config.yaml")
```

### 自定义人格类型

```python
from src.agents.mbti_personalities import MBTIPersonality, MBTIType

# 创建自定义人格
class CustomPersonality(MBTIPersonality):
    def _get_prompt_template(self) -> str:
        return "Your custom prompt template here..."

# 使用自定义人格
personality = CustomPersonality(MBTIType.INTJ)
```

### 自定义网络拓扑

```python
from src.networks.network_generator import NetworkGenerator, NetworkConfig, NetworkType

# 创建自定义网络
def create_custom_network(num_nodes):
    G = nx.Graph()
    # 添加节点和边的逻辑
    return G

# 注册自定义网络类型
NetworkType.CUSTOM = "custom"
```

## 结果分析

### 基本统计

```python
import pandas as pd
import numpy as np

# 加载结果
results = pd.read_csv("results/cooperation_matrix.csv", index_col=0)

# 计算统计量
print("平均合作率:", results.values.mean())
print("标准差:", results.values.std())
print("最高合作率:", results.values.max())
print("最低合作率:", results.values.min())
```

### 人格分析

```python
# 按人格类型分析
personality_rates = results.mean(axis=1)
print("最合作的人格:", personality_rates.idxmax())
print("最不合作的人格:", personality_rates.idxmin())

# 按MBTI维度分析
E_types = [t for t in results.index if t.startswith('E')]
I_types = [t for t in results.index if t.startswith('I')]
E_rate = results.loc[E_types].values.mean()
I_rate = results.loc[I_types].values.mean()
print(f"E型平均合作率: {E_rate:.3f}")
print(f"I型平均合作率: {I_rate:.3f}")
```

### 网络分析

```python
import json

# 加载网络结果
with open("results/network_results.json", 'r') as f:
    network_results = json.load(f)

# 分析不同网络类型
for network_type, scenarios in network_results.items():
    for scenario, results in scenarios.items():
        evolution_data = results["evolution_data"]
        final_cooperation = evolution_data[-1]["cooperation_rate"]
        print(f"{network_type} - {scenario}: {final_cooperation:.3f}")
```

## 性能优化

### 并行处理

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 并行运行多个实验
async def run_parallel_experiments():
    tasks = []
    for config_file in config_files:
        task = run_experiment(config_file)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

### 内存优化

```python
# 分批处理大数据集
def process_large_dataset(data, batch_size=1000):
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        yield process_batch(batch)
```

### 缓存结果

```python
import pickle
from functools import lru_cache

# 缓存LLM响应
@lru_cache(maxsize=1000)
def cached_llm_response(prompt_hash):
    return llm.generate_response(prompt)
```

## 故障排除

### 常见问题

1. **LLM API错误**
   - 检查API密钥是否正确
   - 确认网络连接
   - 检查API配额

2. **内存不足**
   - 减少网络节点数
   - 减少重复实验次数
   - 使用批处理

3. **结果不一致**
   - 设置固定随机种子
   - 检查配置参数
   - 验证数据输入

### 调试模式

```bash
# 启用详细日志
python main.py --experiment pair_game --log-level DEBUG

# 使用Mock LLM进行快速测试
python main.py --experiment pair_game --config configs/quick_test.yaml
```

### 性能监控

```python
import time
import psutil

# 监控资源使用
def monitor_performance():
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    # 运行实验
    results = run_experiment()
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss
    
    print(f"运行时间: {end_time - start_time:.2f}秒")
    print(f"内存使用: {(end_memory - start_memory) / 1024 / 1024:.2f}MB")
    
    return results
```
