"""
Experiment 3: Personality Probe Experiment
Validates if LLM personality injection aligns with MBTI dimensions using specific probe questions.
"""

import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

from src.config.config_manager import ExperimentConfig
from src.llm.llm_interface import LLMManager, LLMFactory
from src.agents.mbti_personalities import MBTIType, MBTIPersonality, get_all_mbti_types

class PersonalityProbeExperiment:
    """Personality Probe Experiment Class"""

    PROBE_QUESTIONS = {
        "E_vs_I": {
            "question": "How do you usually recharge your energy? A. By spending time with a group of friends or engaging in active social events. B. By spending quiet time alone reading, thinking, or relaxing.",
            "A": "E", "B": "I"
        },
        "S_vs_N": {
            "question": "When analyzing a situation, what draws your attention first? A. The concrete facts, specific details, and practical realities. B. The big picture, underlying patterns, and future possibilities.",
            "A": "S", "B": "N"
        },
        "T_vs_F": {
            "question": "When resolving a disagreement, which approach do you prioritize? A. Logical consistency and objective criteria. B. Considering people's feelings and values.",
            "A": "T", "B": "F"
        },
        "J_vs_P": {
            "question": "How do you prefer your daily life to be structured? A. Organized, scheduled, and settled in advance. B. Spontaneous, flexible, and open to last-minute changes.",
            "A": "J", "B": "P"
        }
    }

    def __init__(self, config: ExperimentConfig, llm_manager: LLMManager):
        self.config = config
        self.llm_manager = llm_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = config.llm.model_name
        self.mbti_types = get_all_mbti_types()
        self.personalities = {mbti_type: MBTIPersonality(mbti_type) for mbti_type in self.mbti_types}

    async def run_experiment(self) -> Dict[str, Any]:
        """Run the personality probe experiment"""
        self.logger.info("Starting personality probe experiment...")
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        repetitions = self.config.personality_probe_config.num_repetitions
        
        total_steps = len(self.mbti_types) * repetitions * 4 # 4 dimensions
        pbar = tqdm(total=total_steps, desc="Probing personalities")
        
        score_stats = {dim: {"correct": 0, "total": 0} for dim in self.PROBE_QUESTIONS}
        trait_stats = {t: {"correct": 0, "total": 0} for t in "EISNTFJP"}

        for rep in range(repetitions):
            for mbti_type in self.mbti_types:
                personality = self.personalities[mbti_type]
                type_str = mbti_type.value # e.g., "INTJ"
                
                # Check each dimension
                dimensions = [
                    ("E_vs_I", type_str[0]),
                    ("S_vs_N", type_str[1]),
                    ("T_vs_F", type_str[2]),
                    ("J_vs_P", type_str[3])
                ]
                
                for dim_name, target_trait in dimensions:
                    probe = self.PROBE_QUESTIONS[dim_name]
                    
                    # Determine expected answer (A or B) depending on which trait matches the option map
                    expected_answer = "A" if probe["A"] == target_trait else "B"
                    
                    response_choice = await self._ask_probe_question(personality, probe["question"])
                    
                    is_correct = (response_choice == expected_answer)
                    
                    # Update stats
                    score_stats[dim_name]["total"] += 1
                    trait_stats[target_trait]["total"] += 1
                    
                    if is_correct:
                        score_stats[dim_name]["correct"] += 1
                        trait_stats[target_trait]["correct"] += 1

                    results.append({
                        "repetition": rep + 1,
                        "mbti_type": type_str,
                        "dimension": dim_name,
                        "target_trait": target_trait,
                        "question": probe["question"],
                        "expected": expected_answer,
                        "actual": response_choice,
                        "matched": is_correct
                    })
                    pbar.update(1)
        
        pbar.close()
        
        # Calculate summary statistics
        summary = {}
        # Dimension stats
        for dim, stats in score_stats.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            summary[dim] = accuracy

        # Individual trait stats
        for trait, stats in trait_stats.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            summary[trait] = accuracy
        
        self.logger.info(f"Probe Accuracy Summary: {summary}")
        
        # Save results
        if self.config.personality_probe_config.save_results:
            results_df = pd.DataFrame(results)
            results_path = output_dir / "personality_probe_details.csv"
            results_df.to_csv(results_path, index=False)
            
            with open(output_dir / "personality_probe_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
                
        return {"results": results, "summary": summary}
    
    async def _ask_probe_question(self, personality: MBTIPersonality, question: str) -> str:
        """Inject personality and ask the probe question"""
        
        # Customize prompt injection
        # We assume 'strong' injection for specific probes
        personality_prompt = personality._get_personality_prompt("probe")
        
        full_prompt = f"""{personality_prompt}
Please answer the following question based on your personality.
{question}

Please reply with ONLY the letter 'A' or 'B'. Do not add any explanation.
"""
        try:
            response = await self.llm_manager.generate_response(
                self.model_name, 
                full_prompt, 
                **self.config.llm.kwargs
            )
            
            content = response.content.strip().upper()
            
            # Simple parsing
            if "A" in content and "B" not in content:
                return "A"
            if "B" in content and "A" not in content:
                return "B"
            
            # If strictly A or B is expected
            if content == "A" or content.startswith("A."):
                return "A"
            if content == "B" or content.startswith("B."):
                return "B"
                
            return "UNKNOWN"
            
        except Exception as e:
            self.logger.error(f"Error probing {personality.name}: {e}")
            return "ERROR"

async def run_personality_probe_experiment(config_file: str):
    """Main function to run personality probe experiment"""
    from src.config.config_manager import ConfigManager

    # Load config
    config_manager = ConfigManager()
    config = config_manager.load_config(config_file)
    
    # Create LLM manager
    llm_manager = LLMManager()
    
    # Add LLM instance
    llm = LLMFactory.create_from_config({
        "provider": config.llm.provider,
        "model_name": config.llm.model_name,
        "api_key": config.llm.api_key,
        "kwargs": config.llm.kwargs
    })
    llm_manager.add_llm(config.llm.model_name, llm)
    
    experiment = PersonalityProbeExperiment(config, llm_manager)
    results = await experiment.run_experiment()
    
    return results