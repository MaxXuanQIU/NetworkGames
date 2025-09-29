"""
LLM Network Games Framework Demo Script
Showcase of main framework features
"""

import asyncio
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.agents.mbti_personalities import MBTIType, MBTIPersonality, get_all_mbti_types
from src.config.config_manager import ConfigManager
from src.games.prisoners_dilemma import PrisonersDilemma, Action
from src.llm.llm_interface import LLMManager, LLMFactory, LLMProvider
from src.networks.network_generator import NetworkGenerator, NetworkConfig, NetworkType


def demo_mbti_system():
    """Demo MBTI personality system"""
    print("=" * 60)
    print("MBTI Personality System Demo")
    print("=" * 60)
    
    # Show all MBTI types
    mbti_types = get_all_mbti_types()
    print(f"Supported MBTI personality types ({len(mbti_types)} types):")
    for i, mbti_type in enumerate(mbti_types, 1):
        personality = MBTIPersonality(mbti_type)
        print(f"{i:2d}. {personality.name}")
    
    print("\n" + "-" * 40)
    
    # Demo specific personality type
    intj = MBTIPersonality(MBTIType.INTJ)
    print(f"INTJ Personality Traits: {intj.description}")
    print(f"INTJ Decision Prompt Example:")
    print(intj.get_decision_prompt([], "ENTP")[:200] + "...")


def demo_game_system():
    """Demo game system"""
    print("\n" + "=" * 60)
    print("Prisoner's Dilemma Game System Demo")
    print("=" * 60)
    
    # Create game instance
    game = PrisonersDilemma()
    
    # Show payoff matrix
    print("Standard Prisoner's Dilemma Payoff Matrix:")
    print("        Cooperate    Defect")
    print("Cooperate   (3,3)   (0,5)")
    print("Defect      (5,0)   (1,1)")
    
    # Demo game
    print("\nGame Demo:")
    scenarios = [
        ("Both Cooperate", Action.COOPERATE, Action.COOPERATE),
        ("I Cooperate, Opponent Defects", Action.COOPERATE, Action.DEFECT),
        ("I Defect, Opponent Cooperates", Action.DEFECT, Action.COOPERATE),
        ("Both Defect", Action.DEFECT, Action.DEFECT)
    ]
    
    for desc, action1, action2 in scenarios:
        result = game.play_round(action1, action2)
        print(f"{desc:28}: I {action1.value}, Opponent {action2.value} -> Payoff: I {result.player1_payoff}, Opponent {result.player2_payoff}")
    
    # Demo multi-round game
    print("\nMulti-round Game Demo (5 rounds):")
    actions1 = [Action.COOPERATE, Action.DEFECT, Action.COOPERATE, Action.DEFECT, Action.COOPERATE]
    actions2 = [Action.DEFECT, Action.COOPERATE, Action.COOPERATE, Action.DEFECT, Action.DEFECT]
    
    history = game.play_game(actions1, actions2)
    print(f"My cooperation rate: {history.player1_cooperation_rate:.2f}")
    print(f"Opponent cooperation rate: {history.player2_cooperation_rate:.2f}")
    print(f"My total payoff: {history.player1_total_payoff}")
    print(f"Opponent total payoff: {history.player2_total_payoff}")


def demo_network_system():
    """Demo network system"""
    print("\n" + "=" * 60)
    print("Network Topology Generation System Demo")
    print("=" * 60)
    
    generator = NetworkGenerator()
    
    # Generate different types of networks
    network_types = [
        ("Regular Network", NetworkType.REGULAR),
        ("Small-world Network", NetworkType.SMALL_WORLD),
        ("Random Network", NetworkType.RANDOM),
        ("Scale-free Network", NetworkType.SCALE_FREE)
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
            print(f"{name:18}: Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}, Density={G.number_of_edges()/(G.number_of_nodes()*(G.number_of_nodes()-1)/2):.3f}")
        except Exception as e:
            print(f"{name:18}: Generation failed - {e}")


async def demo_llm_system():
    """Demo LLM system"""
    print("\n" + "=" * 60)
    print("LLM Interface System Demo")
    print("=" * 60)
    
    # Create LLM manager
    llm_manager = LLMManager()
    
    # Add Mock LLM
    mock_llm = LLMFactory.create_llm(
        provider=LLMProvider.MOCK,
        model_name="demo-model",
        cooperation_rate=0.7
    )
    llm_manager.add_llm("demo", mock_llm)
    
    print("Supported LLM Providers:")
    for provider in LLMProvider:
        print(f"  - {provider.value}")
    
    print(f"\nCurrent LLM: {mock_llm.get_provider().value}/{mock_llm.model_name}")
    
    # Demo LLM response
    print("\nLLM Response Demo:")
    prompts = [
        "Please choose cooperate or defect",
        "In the prisoner's dilemma, what would you choose?",
        "Make a decision based on your personality"
    ]
    
    for prompt in prompts:
        response = await llm_manager.generate_response("demo", prompt)
        print(f"Input: {prompt}")
        print(f"Output: {response.content} (Response time: {response.response_time:.3f}s)")
        print()


def demo_config_system():
    """Demo config system"""
    print("\n" + "=" * 60)
    print("Config Management System Demo")
    print("=" * 60)
    
    config_manager = ConfigManager()
    
    # List config files
    config_files = config_manager.list_configs()
    print("Available config files:")
    for config_file in config_files:
        info = config_manager.get_config_info(f"configs/{config_file}")
        print(f"  - {config_file}: {info['name']} ({info['experiment_type']})")
    
    # Demo config validation
    print("\nConfig Validation Demo:")
    for config_file in config_files:
        config_path = f"configs/{config_file}" if not config_file.startswith("configs/") else config_file
        try:
            config = config_manager.load_config(config_path)
            errors = config_manager.validate_config(config)
            if errors:
                print(f"  {config_file}: Validation failed - {len(errors)} errors")
            else:
                print(f"  {config_file}: Validation passed ✓")
        except Exception as e:
            print(f"  {config_file}: Load failed - {e}")


async def main():
    """Main demo function"""
    print("🎮 LLM Network Games Framework Demo")
    print("This is a framework for studying LLM behavior in network games")
    
    # Run demos for each module
    demo_mbti_system()
    demo_game_system()
    demo_network_system()
    await demo_llm_system()
    demo_config_system()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nTo run a full experiment, use:")
    print("  python main.py --experiment pair_game --config configs/pair_game.yaml")
    print("  python main.py --experiment network_game --config configs/network_game.yaml")


if __name__ == "__main__":
    asyncio.run(main())
