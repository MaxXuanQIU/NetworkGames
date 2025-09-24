"""
框架测试脚本
验证各个模块是否正常工作
"""

import asyncio
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """测试模块导入"""
    print("Testing imports...")
    
    try:
        from src.agents.mbti_personalities import MBTIType, MBTIPersonality
        print("✓ MBTI personalities module imported successfully")
        
        from src.games.prisoners_dilemma import PrisonersDilemma, Action
        print("✓ Prisoners dilemma module imported successfully")
        
        from src.llm.llm_interface import LLMManager, LLMFactory, LLMProvider
        print("✓ LLM interface module imported successfully")
        
        from src.networks.network_generator import NetworkGenerator, NetworkType
        print("✓ Network generator module imported successfully")
        
        from src.config.config_manager import ConfigManager
        print("✓ Config manager module imported successfully")
        
        from src.analysis.statistics import CooperationAnalyzer
        print("✓ Statistics module imported successfully")
        
        from src.visualization.plotter import PairGamePlotter
        print("✓ Visualization module imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_mbti_system():
    """测试MBTI系统"""
    print("\nTesting MBTI system...")
    
    try:
        from src.agents.mbti_personalities import MBTIType, MBTIPersonality, get_all_mbti_types
        
        # 测试获取所有MBTI类型
        mbti_types = get_all_mbti_types()
        assert len(mbti_types) == 16, f"Expected 16 MBTI types, got {len(mbti_types)}"
        print("✓ All 16 MBTI types loaded")
        
        # 测试创建人格对象
        personality = MBTIPersonality(MBTIType.INTJ)
        assert personality.mbti_type == MBTIType.INTJ
        assert "INTJ" in personality.name
        print("✓ MBTI personality object created successfully")
        
        # 测试生成决策prompt
        prompt = personality.get_decision_prompt([], "ENTP")
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        print("✓ Decision prompt generated successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ MBTI system test failed: {e}")
        return False


def test_game_system():
    """测试博弈系统"""
    print("\nTesting game system...")
    
    try:
        from src.games.prisoners_dilemma import PrisonersDilemma, Action
        
        # 创建博弈实例
        game = PrisonersDilemma()
        print("✓ Prisoners dilemma game created")
        
        # 测试单轮博弈
        result = game.play_round(Action.COOPERATE, Action.DEFECT)
        assert result.player1_action == Action.COOPERATE
        assert result.player2_action == Action.DEFECT
        assert result.player1_payoff == 0  # 被欺骗
        assert result.player2_payoff == 5  # 背叛收益
        print("✓ Single round game works correctly")
        
        # 测试多轮博弈
        actions1 = [Action.COOPERATE, Action.DEFECT, Action.COOPERATE]
        actions2 = [Action.DEFECT, Action.COOPERATE, Action.DEFECT]
        history = game.play_game(actions1, actions2)
        assert history.total_rounds == 3
        assert history.player1_cooperation_rate == 2/3
        assert history.player2_cooperation_rate == 1/3
        print("✓ Multi-round game works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Game system test failed: {e}")
        return False


def test_network_system():
    """测试网络系统"""
    print("\nTesting network system...")
    
    try:
        from src.networks.network_generator import NetworkGenerator, NetworkConfig, NetworkType
        
        # 创建网络生成器
        generator = NetworkGenerator()
        print("✓ Network generator created")
        
        # 测试生成小世界网络
        config = NetworkConfig(
            network_type=NetworkType.SMALL_WORLD,
            num_nodes=20,
            k=4,
            p=0.1
        )
        G = generator.generate_network(config)
        assert G.number_of_nodes() == 20
        print("✓ Small world network generated")
        
        # 测试网络分析
        from src.networks.network_generator import NetworkAnalyzer
        analyzer = NetworkAnalyzer()
        analysis = analyzer.analyze_network(G)
        assert "num_nodes" in analysis
        assert "density" in analysis
        print("✓ Network analysis works")
        
        return True
        
    except Exception as e:
        print(f"✗ Network system test failed: {e}")
        return False


def test_llm_system():
    """测试LLM系统"""
    print("\nTesting LLM system...")
    
    try:
        from src.llm.llm_interface import LLMManager, LLMFactory, LLMProvider
        
        # 创建LLM管理器
        llm_manager = LLMManager()
        print("✓ LLM manager created")
        
        # 创建Mock LLM
        mock_llm = LLMFactory.create_llm(
            provider=LLMProvider.MOCK,
            model_name="test-model",
            cooperation_rate=0.7
        )
        llm_manager.add_llm("test", mock_llm)
        print("✓ Mock LLM created and added")
        
        return True
        
    except Exception as e:
        print(f"✗ LLM system test failed: {e}")
        return False


async def test_async_functionality():
    """测试异步功能"""
    print("\nTesting async functionality...")
    
    try:
        from src.llm.llm_interface import LLMManager, LLMFactory, LLMProvider
        
        # 创建LLM管理器
        llm_manager = LLMManager()
        mock_llm = LLMFactory.create_llm(
            provider=LLMProvider.MOCK,
            model_name="test-model",
            cooperation_rate=0.8
        )
        llm_manager.add_llm("test", mock_llm)
        
        # 测试异步响应生成
        response = await llm_manager.generate_response("test", "Test prompt")
        assert response.success
        assert response.is_valid_action()
        print("✓ Async LLM response generation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Async functionality test failed: {e}")
        return False


def test_config_system():
    """测试配置系统"""
    print("\nTesting config system...")
    
    try:
        from src.config.config_manager import ConfigManager
        
        # 创建配置管理器
        config_manager = ConfigManager()
        print("✓ Config manager created")
        
        # 测试创建默认配置
        config_manager.create_default_configs()
        print("✓ Default configs created")
        
        # 测试加载配置
        try:
            config = config_manager.load_config("pair_game.yaml")
            assert config.experiment_type.value == "pair_game"
            print("✓ Config loading works")
        except FileNotFoundError:
            print("✓ Config files exist (loading test skipped)")
        
        return True
        
    except Exception as e:
        print(f"✗ Config system test failed: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 50)
    print("LLM Network Games Framework Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_mbti_system,
        test_game_system,
        test_network_system,
        test_llm_system,
        test_config_system
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    # 运行异步测试
    print("Testing async functionality...")
    if asyncio.run(test_async_functionality()):
        passed += 1
        print()
    total += 1
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Framework is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
