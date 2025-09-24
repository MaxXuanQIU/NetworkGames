"""
LLM Network Games Framework 演示脚本
展示框架的主要功能
"""

import asyncio
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.agents.mbti_personalities import MBTIType, MBTIPersonality, get_all_mbti_types
from src.config.config_manager import ConfigManager
from src.games.prisoners_dilemma import PrisonersDilemma, Action
from src.llm.llm_interface import LLMManager, LLMFactory, LLMProvider
from src.networks.network_generator import NetworkGenerator, NetworkConfig, NetworkType


def demo_mbti_system():
    """演示MBTI人格系统"""
    print("=" * 60)
    print("MBTI人格系统演示")
    print("=" * 60)
    
    # 显示所有MBTI类型
    mbti_types = get_all_mbti_types()
    print(f"支持的MBTI人格类型 ({len(mbti_types)}种):")
    for i, mbti_type in enumerate(mbti_types, 1):
        personality = MBTIPersonality(mbti_type)
        print(f"{i:2d}. {personality.name}")
    
    print("\n" + "-" * 40)
    
    # 演示特定人格类型
    intj = MBTIPersonality(MBTIType.INTJ)
    print(f"INTJ人格特征: {intj.description}")
    print(f"INTJ决策prompt示例:")
    print(intj.get_decision_prompt([], "ENTP")[:200] + "...")


def demo_game_system():
    """演示博弈系统"""
    print("\n" + "=" * 60)
    print("囚徒困境博弈系统演示")
    print("=" * 60)
    
    # 创建博弈实例
    game = PrisonersDilemma()
    
    # 显示收益矩阵
    print("标准囚徒困境收益矩阵:")
    print("        合作    背叛")
    print("合作   (3,3)   (0,5)")
    print("背叛   (5,0)   (1,1)")
    
    # 演示博弈
    print("\n博弈演示:")
    scenarios = [
        ("双方合作", Action.COOPERATE, Action.COOPERATE),
        ("我合作，对手背叛", Action.COOPERATE, Action.DEFECT),
        ("我背叛，对手合作", Action.DEFECT, Action.COOPERATE),
        ("双方背叛", Action.DEFECT, Action.DEFECT)
    ]
    
    for desc, action1, action2 in scenarios:
        result = game.play_round(action1, action2)
        print(f"{desc:12}: 我{action1.value}, 对手{action2.value} -> 收益: 我{result.player1_payoff}, 对手{result.player2_payoff}")
    
    # 演示多轮博弈
    print("\n多轮博弈演示 (5轮):")
    actions1 = [Action.COOPERATE, Action.DEFECT, Action.COOPERATE, Action.DEFECT, Action.COOPERATE]
    actions2 = [Action.DEFECT, Action.COOPERATE, Action.COOPERATE, Action.DEFECT, Action.DEFECT]
    
    history = game.play_game(actions1, actions2)
    print(f"我的合作率: {history.player1_cooperation_rate:.2f}")
    print(f"对手合作率: {history.player2_cooperation_rate:.2f}")
    print(f"我的总收益: {history.player1_total_payoff}")
    print(f"对手总收益: {history.player2_total_payoff}")


def demo_network_system():
    """演示网络系统"""
    print("\n" + "=" * 60)
    print("网络拓扑生成系统演示")
    print("=" * 60)
    
    generator = NetworkGenerator()
    
    # 生成不同类型的网络
    network_types = [
        ("规则网络", NetworkType.REGULAR),
        ("小世界网络", NetworkType.SMALL_WORLD),
        ("随机网络", NetworkType.RANDOM),
        ("无标度网络", NetworkType.SCALE_FREE)
    ]
    
    for name, network_type in network_types:
        config = NetworkConfig(
            network_type=network_type,
            num_nodes=20,
            k=4,
            p=0.1,
            edge_probability=0.1,
            m=2
        )
        
        try:
            G = generator.generate_network(config)
            print(f"{name:8}: 节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}, 密度={G.number_of_edges()/(G.number_of_nodes()*(G.number_of_nodes()-1)/2):.3f}")
        except Exception as e:
            print(f"{name:8}: 生成失败 - {e}")


async def demo_llm_system():
    """演示LLM系统"""
    print("\n" + "=" * 60)
    print("LLM接口系统演示")
    print("=" * 60)
    
    # 创建LLM管理器
    llm_manager = LLMManager()
    
    # 添加Mock LLM
    mock_llm = LLMFactory.create_llm(
        provider=LLMProvider.MOCK,
        model_name="demo-model",
        cooperation_rate=0.7
    )
    llm_manager.add_llm("demo", mock_llm)
    
    print("支持的LLM提供商:")
    for provider in LLMProvider:
        print(f"  - {provider.value}")
    
    print(f"\n当前使用的LLM: {mock_llm.get_provider().value}/{mock_llm.model_name}")
    
    # 演示LLM响应
    print("\nLLM响应演示:")
    prompts = [
        "请选择合作或背叛",
        "在囚徒困境中，你会如何选择？",
        "基于你的性格，做出决策"
    ]
    
    for prompt in prompts:
        response = await llm_manager.generate_response("demo", prompt)
        print(f"输入: {prompt}")
        print(f"输出: {response.content} (响应时间: {response.response_time:.3f}s)")
        print()


def demo_config_system():
    """演示配置系统"""
    print("\n" + "=" * 60)
    print("配置管理系统演示")
    print("=" * 60)
    
    config_manager = ConfigManager()
    
    # 列出配置文件
    config_files = config_manager.list_configs()
    print("可用的配置文件:")
    for config_file in config_files:
        info = config_manager.get_config_info(config_file)
        if "error" not in info:
            print(f"  - {config_file}: {info['name']} ({info['experiment_type']})")
    
    # 演示配置验证
    print("\n配置验证演示:")
    for config_file in config_files:
        try:
            config = config_manager.load_config(config_file)
            errors = config_manager.validate_config(config)
            if errors:
                print(f"  {config_file}: 验证失败 - {len(errors)}个错误")
            else:
                print(f"  {config_file}: 验证通过 ✓")
        except Exception as e:
            print(f"  {config_file}: 加载失败 - {e}")


async def main():
    """主演示函数"""
    print("🎮 LLM Network Games Framework 演示")
    print("这是一个用于研究LLM在网络博弈中行为的框架")
    
    # 运行各个模块的演示
    demo_mbti_system()
    demo_game_system()
    demo_network_system()
    await demo_llm_system()
    demo_config_system()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n要运行完整实验，请使用:")
    print("  python main.py --experiment pair_game --config configs/pair_game.yaml")
    print("  python main.py --experiment network_game --config configs/network_game.yaml")


if __name__ == "__main__":
    asyncio.run(main())
