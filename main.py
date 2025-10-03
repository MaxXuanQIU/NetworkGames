"""
LLM Network Games Framework main program
Supports running different experiments and configuration management
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config.config_manager import ConfigManager, ExperimentType
from src.llm.llm_interface import LLMManager, LLMFactory
from src.experiments.pair_game_experiment import run_pair_game_experiment
from src.experiments.network_game_experiment import run_network_game_experiment


def setup_logging(level: str = "INFO"):
    """Set up logging"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('experiment.log')
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Suppress httpx INFO logs


def create_default_configs():
    """Create default configuration files"""
    config_manager = ConfigManager()
    config_manager.create_default_configs()
    print("Default configuration files created in configs/ directory")


def list_configs():
    """List all configuration files"""
    config_manager = ConfigManager()
    config_files = config_manager.list_configs()
    
    print("Available configuration files:")
    for config_file in config_files:
        info = config_manager.get_config_info(config_file)
        if "error" not in info:
            print(f"  {config_file}:")
            print(f"    Name: {info['name']}")
            print(f"    Type: {info['experiment_type']}")
            print(f"    Description: {info['description']}")
            print(f"    LLM: {info['llm_provider']}/{info['llm_model']}")
            print()


def validate_config(config_file: str):
    """Validate configuration file"""
    config_manager = ConfigManager()
    try:
        config = config_manager.load_config(config_file)
        errors = config_manager.validate_config(config)
        
        if errors:
            print(f"Configuration validation failed for {config_file}:")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"Configuration {config_file} is valid!")
    except Exception as e:
        print(f"Error loading configuration {config_file}: {e}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="LLM Network Games Framework")
    parser.add_argument("--experiment", choices=["pair_game", "network_game"], 
                       help="Type of experiment to run")
    parser.add_argument("--config", default=None,
                       help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--create-configs", action="store_true",
                       help="Create default configuration files")
    parser.add_argument("--list-configs", action="store_true",
                       help="List available configuration files")
    parser.add_argument("--validate-config", help="Validate a configuration file")
    
    args = parser.parse_args()
    
    # Set default configuration file based on experiment type
    if args.config is None:
        if args.experiment == "pair_game":
            args.config = "configs/pair_game.yaml"
        elif args.experiment == "network_game":
            args.config = "configs/network_game.yaml"
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Handle special commands
    if args.create_configs:
        create_default_configs()
        return
    
    if args.list_configs:
        list_configs()
        return
    
    if args.validate_config:
        validate_config(args.validate_config)
        return
    
    # Check experiment type
    if not args.experiment:
        print("Error: Please specify an experiment type using --experiment")
        print("Available experiments: pair_game, network_game")
        return
    
    # Run experiment
    try:
        if args.experiment == "pair_game":
            logger.info("Starting pair game experiment...")
            results = await run_pair_game_experiment(args.config)
            logger.info("Pair game experiment completed successfully!")
            
        elif args.experiment == "network_game":
            logger.info("Starting network game experiment...")
            results = await run_network_game_experiment(args.config)
            logger.info("Network game experiment completed successfully!")
        
        print(f"\nExperiment completed! Results saved to: {results.get('output_dir', 'results')}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
