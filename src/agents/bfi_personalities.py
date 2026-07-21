"""
Operational Big Five Inventory (BFI) profiles for pair-game experiments.

These are deliberately profile labels, not personality diagnoses. They represent
contrasting, reproducible trait configurations for experimental prompting.
"""

from enum import Enum
from typing import List

from src.agents.mbti_personalities import MBTIPersonality


class BFIProfile(Enum):
    COOPERATIVE = "Cooperative (high agreeableness/conscientiousness)"
    COMPETITIVE = "Competitive (low agreeableness/high extraversion)"


class BFIPersonality(MBTIPersonality):
    """
    BFI-profile agent that reuses MBTIPersonality.get_decision_prompt().

    The inherited prompt-composition logic is profile-agnostic: it only needs
    prompt_template, official_prompt, and _get_personality_name().
    """

    def __init__(self, profile: BFIProfile):
        self.profile = profile
        self.name = self._get_personality_name()
        self.description = self._get_personality_description()
        self.prompt_template = self._get_prompt_template()
        self.official_prompt = self.prompt_template

    def _get_personality_name(self) -> str:
        return self.profile.value

    def _get_personality_description(self) -> str:
        descriptions = {
            BFIProfile.COOPERATIVE: (
                "High agreeableness and conscientiousness; inclined toward "
                "reciprocity, reliability, and maintaining beneficial relationships."
            ),
            BFIProfile.COMPETITIVE: (
                "Low agreeableness and high extraversion; inclined toward "
                "assertiveness, opportunism, and immediate competitive advantage."
            ),
        }
        return descriptions[self.profile]

    def _get_prompt_template(self) -> str:
        templates = {
            BFIProfile.COOPERATIVE: """You are a BFI-profiled AI agent.

Your personality dimensions are:
- Extraversion: 0.60 (moderately high)
- Agreeableness: 0.85 (very high)
- Conscientiousness: 0.75 (high)
- Neuroticism: 0.30 (low)
- Openness: 0.65 (moderately high)

In repeated games, interpret these traits as behavioral tendencies.""",

            BFIProfile.COMPETITIVE: """You are a BFI-profiled AI agent.

Your personality dimensions are:
- Extraversion: 0.85 (very high)
- Agreeableness: 0.25 (low)
- Conscientiousness: 0.45 (moderate)
- Neuroticism: 0.40 (moderately low)
- Openness: 0.60 (moderately high)

In repeated games, interpret these traits as behavioral tendencies.""",
        }
        return templates[self.profile]


def get_all_bfi_profiles() -> List[BFIProfile]:
    """Return the profiles available to BFI pair-game experiments."""
    return list(BFIProfile)


def get_bfi_profile_by_name(name: str) -> BFIProfile:
    """Resolve either an enum name or the display value."""
    normalized_name = name.strip().upper()
    for profile in BFIProfile:
        if normalized_name == profile.name or name.strip() == profile.value:
            return profile
    raise ValueError(f"Unknown BFI profile: {name}")