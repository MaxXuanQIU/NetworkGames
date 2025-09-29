"""
Result Manager
Responsible for saving, loading, and managing experiment results
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import pickle
import networkx as nx


class ResultManager:
    """Result Manager"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def save_experiment_results(self, experiment_type: str, results: Dict[str, Any], 
                              config: Dict[str, Any], experiment_id: Optional[str] = None):
        """Save experiment results"""
        if experiment_id is None:
            experiment_id = f"{experiment_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create experiment directory
        experiment_dir = self.output_dir / experiment_id
        experiment_dir.mkdir(exist_ok=True)
        
        # Save config
        self._save_config(experiment_dir, config)
        
        # Save results
        if experiment_type == "pair_game":
            self._save_pair_game_results(experiment_dir, results)
        elif experiment_type == "network_game":
            self._save_network_game_results(experiment_dir, results)
        
        # Save metadata
        self._save_metadata(experiment_dir, experiment_type, results)
        
        self.logger.info(f"Results saved to: {experiment_dir}")
        return experiment_dir
    
    def _save_config(self, experiment_dir: Path, config: Dict[str, Any]):
        """Save experiment config"""
        config_file = experiment_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)
    
    def _save_pair_game_results(self, experiment_dir: Path, results: Dict[str, Any]):
        """Save pair game results"""
        # Save matrix data
        if "matrix_results" in results:
            matrix_results = results["matrix_results"]
            
            # Cooperation rate matrix
            if "cooperation_matrix" in matrix_results:
                matrix_df = pd.DataFrame(
                    matrix_results["cooperation_matrix"],
                    index=matrix_results["personality_types"],
                    columns=matrix_results["personality_types"]
                )
                matrix_df.to_csv(experiment_dir / "cooperation_matrix.csv")
            
            # Payoff matrix
            if "payoff_matrix" in matrix_results:
                payoff_df = pd.DataFrame(
                    matrix_results["payoff_matrix"],
                    index=matrix_results["personality_types"],
                    columns=matrix_results["personality_types"]
                )
                payoff_df.to_csv(experiment_dir / "payoff_matrix.csv")
            
            # Standard deviation matrix
            if "std_matrix" in matrix_results:
                std_df = pd.DataFrame(
                    matrix_results["std_matrix"],
                    index=matrix_results["personality_types"],
                    columns=matrix_results["personality_types"]
                )
                std_df.to_csv(experiment_dir / "std_matrix.csv")
            
            # Detailed results
            if "detailed_results" in matrix_results:
                detailed_data = []
                for key, result in matrix_results["detailed_results"].items():
                    detailed_data.append({
                        "player1_type": result["player1_type"],
                        "player2_type": result["player2_type"],
                        "mean_cooperation": result["mean_cooperation"],
                        "std_cooperation": result["std_cooperation"],
                        "mean_payoff": result["mean_payoff"],
                        "std_payoff": result["std_payoff"],
                        "cooperation_rates": result["cooperation_rates"],
                        "payoffs": result["payoffs"]
                    })
                
                detailed_df = pd.DataFrame(detailed_data)
                detailed_df.to_csv(experiment_dir / "detailed_results.csv", index=False)
        
        # Save analysis results
        if "analysis_results" in results:
            analysis_file = experiment_dir / "analysis_results.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(results["analysis_results"], f, indent=2, ensure_ascii=False, default=str)
        
        # Save visualization file paths
        if "visualization_results" in results:
            viz_file = experiment_dir / "visualization_files.json"
            with open(viz_file, 'w', encoding='utf-8') as f:
                json.dump(results["visualization_results"], f, indent=2, ensure_ascii=False)
    
    def _save_network_game_results(self, experiment_dir: Path, results: Dict[str, Any]):
        """Save network game results"""
        # Save network results
        if "network_results" in results:
            network_results = results["network_results"]
            
            # Create network results directory
            network_dir = experiment_dir / "networks"
            network_dir.mkdir(exist_ok=True)
            
            for network_type, scenarios in network_results.items():
                for scenario, scenario_results in scenarios.items():
                    # Save evolution data
                    evolution_data = scenario_results.get("evolution_data", [])
                    if evolution_data:
                        evolution_df = pd.DataFrame(evolution_data)
                        evolution_file = network_dir / f"evolution_{network_type}_{scenario}.csv"
                        evolution_df.to_csv(evolution_file, index=False)
                    
                    # Save network analysis
                    network_analysis = scenario_results.get("network_analysis", {})
                    analysis_file = network_dir / f"analysis_{network_type}_{scenario}.json"
                    with open(analysis_file, 'w', encoding='utf-8') as f:
                        json.dump(network_analysis, f, indent=2, ensure_ascii=False, default=str)
                    
                    # Save personality assignment
                    personality_assignment = scenario_results.get("personality_assignment", {})
                    personality_file = network_dir / f"personalities_{network_type}_{scenario}.json"
                    with open(personality_file, 'w', encoding='utf-8') as f:
                        json.dump(personality_assignment, f, indent=2, ensure_ascii=False)
        
        # Save analysis results
        if "analysis_results" in results:
            analysis_file = experiment_dir / "network_analysis.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(results["analysis_results"], f, indent=2, ensure_ascii=False, default=str)
        
        # Save visualization file paths
        if "visualization_results" in results:
            viz_file = experiment_dir / "visualization_files.json"
            with open(viz_file, 'w', encoding='utf-8') as f:
                json.dump(results["visualization_results"], f, indent=2, ensure_ascii=False)
    
    def _save_metadata(self, experiment_dir: Path, experiment_type: str, results: Dict[str, Any]):
        """Save metadata"""
        metadata = {
            "experiment_type": experiment_type,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "files": self._list_experiment_files(experiment_dir)
        }
        
        metadata_file = experiment_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _list_experiment_files(self, experiment_dir: Path) -> List[str]:
        """List experiment files"""
        files = []
        for file_path in experiment_dir.rglob("*"):
            if file_path.is_file():
                files.append(str(file_path.relative_to(experiment_dir)))
        return sorted(files)
    
    def load_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Load experiment results"""
        experiment_dir = self.output_dir / experiment_id
        
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
        
        # Load metadata
        metadata_file = experiment_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Load config
        config_file = experiment_dir / "config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Load results according to experiment type
        experiment_type = metadata.get("experiment_type", "unknown")
        
        if experiment_type == "pair_game":
            results = self._load_pair_game_results(experiment_dir)
        elif experiment_type == "network_game":
            results = self._load_network_game_results(experiment_dir)
        else:
            results = {}
        
        return {
            "metadata": metadata,
            "config": config,
            "results": results
        }
    
    def _load_pair_game_results(self, experiment_dir: Path) -> Dict[str, Any]:
        """Load pair game results"""
        results = {}
        
        # Load matrix data
        matrix_files = {
            "cooperation_matrix": "cooperation_matrix.csv",
            "payoff_matrix": "payoff_matrix.csv",
            "std_matrix": "std_matrix.csv"
        }
        
        for key, filename in matrix_files.items():
            file_path = experiment_dir / filename
            if file_path.exists():
                df = pd.read_csv(file_path, index_col=0)
                results[key] = df.values
                results[f"{key}_index"] = df.index.tolist()
                results[f"{key}_columns"] = df.columns.tolist()
        
        # Load detailed results
        detailed_file = experiment_dir / "detailed_results.csv"
        if detailed_file.exists():
            results["detailed_results"] = pd.read_csv(detailed_file)
        
        # Load analysis results
        analysis_file = experiment_dir / "analysis_results.json"
        if analysis_file.exists():
            with open(analysis_file, 'r', encoding='utf-8') as f:
                results["analysis_results"] = json.load(f)
        
        return results
    
    def _load_network_game_results(self, experiment_dir: Path) -> Dict[str, Any]:
        """Load network game results"""
        results = {}
        
        # Load network results
        network_dir = experiment_dir / "networks"
        if network_dir.exists():
            network_results = {}
            
            for file_path in network_dir.glob("evolution_*.csv"):
                filename = file_path.stem
                parts = filename.split("_", 2)
                if len(parts) >= 3:
                    network_type = parts[1]
                    scenario = parts[2]
                    
                    if network_type not in network_results:
                        network_results[network_type] = {}
                    
                    df = pd.read_csv(file_path)
                    network_results[network_type][scenario] = {
                        "evolution_data": df.to_dict('records')
                    }
            
            results["network_results"] = network_results
        
        # Load analysis results
        analysis_file = experiment_dir / "network_analysis.json"
        if analysis_file.exists():
            with open(analysis_file, 'r', encoding='utf-8') as f:
                results["analysis_results"] = json.load(f)
        
        return results
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments"""
        experiments = []
        
        for experiment_dir in self.output_dir.iterdir():
            if experiment_dir.is_dir():
                metadata_file = experiment_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    experiments.append({
                        "id": experiment_dir.name,
                        "type": metadata.get("experiment_type", "unknown"),
                        "timestamp": metadata.get("timestamp", ""),
                        "files": metadata.get("files", [])
                    })
        
        return sorted(experiments, key=lambda x: x["timestamp"], reverse=True)

    
    def cleanup_old_experiments(self, days: int = 30):
        """Clean up old experiments"""
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (days * 24 * 60 * 60)
        
        cleaned_count = 0
        for experiment_dir in self.output_dir.iterdir():
            if experiment_dir.is_dir():
                if experiment_dir.stat().st_mtime < cutoff_time:
                    import shutil
                    shutil.rmtree(experiment_dir)
                    cleaned_count += 1
        
        self.logger.info(f"Cleaned up {cleaned_count} old experiments")
        return cleaned_count
