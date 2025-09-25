# LLM Network Games Framework

一个用于研究LLM在网络博弈中行为的可复现、可修改的框架。

## 项目概述

本项目旨在探究不同LLM模型在网络博弈中的行为模式，特别是：
- 基于MBTI 16人格分类的LLM行为差异
- 重复囚徒困境中的合作与背叛策略
- 网络拓扑对博弈行为的影响
- 人格分布对网络演化的影响

## 主要特性

- **多LLM支持**: 支持调用多种具有代表性的LLM模型（OpenAI、Anthropic、Google、Mock）
- **MBTI人格系统**: 16种人格类型的夸张刻板印象prompt
- **网络博弈**: 支持多种网络拓扑（规则、小世界、随机网络、无标度网络）
- **可配置实验**: 通过YAML配置文件管理实验参数
- **丰富可视化**: 热力图、网络图、时间序列等多种统计图表
- **可复现性**: 完整的随机种子控制和结果输出
- **统计分析**: 完整的统计检验和效应量分析

## 实验设计

### 实验1: 两人博弈
- **目标**: 16x16人格组合矩阵在重复囚徒困境中的行为基线
- **设计**: 所有MBTI人格两两组合进行重复囚徒困境博弈
- **参数**: 可调节的博弈轮数（默认100轮）和重复次数（默认20次）
- **输出**: 16x16合作率热力图、统计显著性分析、人格排名

### 实验2: 网络博弈
- **目标**: 不同网络拓扑和人格分布下的网络博弈演化
- **变量1**: 网络拓扑（规则、小世界、随机网络）
- **变量2**: 人格分布（均匀、单一类型、聚类分布）
- **变量3**: 交互轮次（默认100轮）
- **输出**: 网络演化动图、合作率时间序列、网络快照

## 快速开始

### 1. 安装依赖

```bash
# 克隆项目
git clone https://github.com/MaxXuanQIU/NetworkGames.git
cd LLM-Network-Games

# 安装依赖
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

### 4. 运行完整实验

```bash
# 运行两人博弈实验
python main.py --experiment pair_game

# 运行网络博弈实验
python main.py --experiment network_game

# 注意，使用真实LLM需要API密钥，请在 --config configs/pair_game.yaml 和 --config configs/network_game.yaml" 中配置
```

## 项目结构

```
NetworkGames/
├── src/                           # 源代码
│   ├── agents/                   # 智能体相关
│   │   └── mbti_personalities.py # MBTI人格系统
│   ├── games/                    # 博弈逻辑
│   │   └── prisoners_dilemma.py  # 囚徒困境实现
│   ├── networks/                 # 网络拓扑
│   │   └── network_generator.py  # 网络生成器
│   ├── llm/                     # LLM接口
│   │   └── llm_interface.py     # LLM抽象层
│   ├── analysis/                # 数据分析
│   │   └── statistics.py        # 统计分析
│   ├── visualization/           # 可视化
│   │   └── plotter.py          # 绘图工具
│   ├── config/                  # 配置管理
│   │   └── config_manager.py    # 配置管理器
│   ├── experiments/             # 实验实现
│   │   ├── pair_game_experiment.py      # 两人博弈实验
│   │   └── network_game_experiment.py   # 网络博弈实验
│   └── utils/                   # 工具函数
│       └── result_manager.py    # 结果管理
├── configs/                     # 配置文件
│   ├── pair_game.yaml          # 两人博弈配置
│   └── network_game.yaml       # 网络博弈配置
├── experiments/                 # 实验脚本
│   └── run_experiments.py      # 实验运行脚本
├── results/                    # 结果输出
├── docs/                      # 文档
│   ├── API_REFERENCE.md       # API参考
│   └── EXPERIMENT_GUIDE.md    # 实验指南
├── main.py                    # 主程序入口
├── requirements.txt           # 依赖列表
└── README.md                 # 项目说明
```

## 使用示例

### 基本使用

```python
import asyncio
from src.experiments.pair_game_experiment import run_pair_game_experiment

# 运行两人博弈实验
results = await run_pair_game_experiment("configs/pair_game.yaml")
print("实验完成！结果保存在 results/ 目录")
```

### 自定义配置

```python
from src.config.config_manager import ConfigManager, ExperimentConfig

# 创建自定义配置
config = ExperimentConfig(
    experiment_type="pair_game",
    name="My Custom Experiment",
    llm_provider="openai",
    llm_model="gpt-4",
    num_rounds=200,
    num_repetitions=50
)

# 保存配置
config_manager = ConfigManager()
config_manager.save_config(config, "my_config.yaml")
```

### 结果分析

```python
import pandas as pd
import matplotlib.pyplot as plt

# 加载结果
results = pd.read_csv("results/cooperation_matrix.csv", index_col=0)

# 绘制热力图
plt.figure(figsize=(12, 10))
sns.heatmap(results, annot=True, cmap='RdYlBu_r')
plt.title("MBTI Cooperation Rate Matrix")
plt.show()

# 分析人格排名
personality_rates = results.mean(axis=1).sort_values(ascending=False)
print("最合作的人格类型:", personality_rates.index[0])
print("最不合作的人格类型:", personality_rates.index[-1])
```

## 配置说明

### LLM配置

支持多种LLM提供商：

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

# Mock（用于测试）
llm:
  provider: "mock"
  model_name: "mock-model"
```

### 网络配置

```yaml
# 小世界网络
network:
  network_type: "small_world"
  num_nodes: 50
  k: 4
  p: 0.1

# 规则网络
network:
  network_type: "regular"
  num_nodes: 50
  k: 4

# 随机网络
network:
  network_type: "random"
  num_nodes: 50
  edge_probability: 0.1
```

## 输出结果

### 两人博弈实验输出

- `cooperation_matrix.csv`: 16x16合作率矩阵
- `payoff_matrix.csv`: 16x16收益矩阵
- `detailed_results.csv`: 详细结果数据
- `analysis_results.json`: 统计分析结果
- `experiment_config.json`: 实验配置信息
- `cooperation_heatmap.png`: 合作率热力图
- `cooperation_distribution.png`: 合作率分布图
- `personality_ranking.png`: 人格合作率排名
- `mbti_dimension_analysis.png`: MBTI维度分析

### 网络博弈实验输出

- `network_results.json`: 网络博弈详细结果
- `network_analysis.json`: 网络分析结果
- `network_evolution_*.png`: 网络演化图
- `network_comparison.png`: 网络类型比较
- `network_snapshot_*.png`: 网络快照
- `cooperation_clusters.png`: 合作集群分析

## 高级功能

### 自定义人格类型

```python
from src.agents.mbti_personalities import MBTIPersonality, MBTIType

class CustomPersonality(MBTIPersonality):
    def _get_prompt_template(self) -> str:
        return "Your custom prompt template here..."
```

### 自定义网络拓扑

```python
from src.networks.network_generator import NetworkGenerator

def create_custom_network(num_nodes):
    # 实现自定义网络生成逻辑
    pass
```

### 批量实验

```python
# 运行多个配置的实验
configs = ["config1.yaml", "config2.yaml", "config3.yaml"]
for config in configs:
    results = await run_experiment(config)
    # 处理结果
```

## 性能优化

- 支持异步并行处理
- 可配置的批处理大小
- 结果缓存机制
- 内存使用优化

## 故障排除

### 常见问题

1. **API密钥错误**: 检查LLM API密钥是否正确设置
2. **内存不足**: 减少网络节点数或重复次数
3. **结果不一致**: 设置固定随机种子
4. **网络连接问题**: 检查网络连接和防火墙设置

### 调试模式

```bash
# 启用详细日志
python main.py --experiment pair_game --log-level DEBUG

# 使用Mock LLM进行快速测试
python main.py --experiment pair_game --config configs/quick_test.yaml
```

## 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

### 开发环境设置

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest tests/

# 代码格式化
black src/
flake8 src/
```

### 提交规范

- 使用清晰的提交信息
- 添加适当的测试
- 更新相关文档

## 许可证

Apache 2.0 License

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@software{llm_network_games,
  title={LLM Network Games Framework},
  author={Xuan Qiu},
  year={2025},
  url={https://github.com/MaxXuanQIU/NetworkGames}
}
```

## 联系方式

- 项目主页: https://github.com/MaxXuanQIU/NetworkGames
- 问题反馈: https://github.com/MaxXuanQIU/NetworkGames/issues
- 邮箱: maxxuanqiu@hkust-gz.edu.cn