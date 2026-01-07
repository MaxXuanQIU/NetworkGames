"""
MBTI personality type definitions and corresponding prompt templates
Based on the 16 MBTI personality types, create corresponding personality traits and game strategy prompts.
"""

from enum import Enum
from typing import Dict, List
import random


class MBTIType(Enum):
    """16 MBTI personality types"""
    # Analysts (NT)
    INTJ = "INTJ"  # Architect
    INTP = "INTP"  # Thinker
    ENTJ = "ENTJ"  # Commander
    ENTP = "ENTP"  # Debater
    
    # Diplomats (NF)
    INFJ = "INFJ"  # Advocate
    INFP = "INFP"  # Mediator
    ENFJ = "ENFJ"  # Protagonist
    ENFP = "ENFP"  # Campaigner
    
    # Sentinels (SJ)
    ISTJ = "ISTJ"  # Logistician
    ISFJ = "ISFJ"  # Defender
    ESTJ = "ESTJ"  # Executive
    ESFJ = "ESFJ"  # Consul
    
    # Explorers (SP)
    ISTP = "ISTP"  # Virtuoso
    ISFP = "ISFP"  # Adventurer
    ESTP = "ESTP"  # Entrepreneur
    ESFP = "ESFP"  # Entertainer


class MBTIPersonality:
    """MBTI personality class, includes personality traits and game strategy prompts"""
    
    def __init__(self, mbti_type: MBTIType):
        self.mbti_type = mbti_type
        self.name = self._get_personality_name()
        self.description = self._get_personality_description()
        self.prompt_template = self._get_prompt_template()
    
    def _get_personality_name(self) -> str:
        """Get the name of the personality type"""
        names = {
            MBTIType.INTJ: "INTJ (Architect)",
            MBTIType.INTP: "INTP (Thinker)",
            MBTIType.ENTJ: "ENTJ (Commander)",
            MBTIType.ENTP: "ENTP (Debater)",
            MBTIType.INFJ: "INFJ (Advocate)",
            MBTIType.INFP: "INFP (Mediator)",
            MBTIType.ENFJ: "ENFJ (Protagonist)",
            MBTIType.ENFP: "ENFP (Campaigner)",
            MBTIType.ISTJ: "ISTJ (Logistician)",
            MBTIType.ISFJ: "ISFJ (Defender)",
            MBTIType.ESTJ: "ESTJ (Executive)",
            MBTIType.ESFJ: "ESFJ (Consul)",
            MBTIType.ISTP: "ISTP (Virtuoso)",
            MBTIType.ISFP: "ISFP (Adventurer)",
            MBTIType.ESTP: "ESTP (Entrepreneur)",
            MBTIType.ESFP: "ESFP (Entertainer)"
        }
        return names[self.mbti_type]
    
    def _get_personality_description(self) -> str:
        """Get the description of the personality type"""
        descriptions = {
            MBTIType.INTJ: "Independent, strategic thinker, pursues efficiency and long-term goals",
            MBTIType.INTP: "Curious, logical analyst, loves theory and abstract concepts",
            MBTIType.ENTJ: "Confident, decisive leader, naturally commanding",
            MBTIType.ENTP: "Innovative, witty debater, loves challenges and possibilities",
            MBTIType.INFJ: "Idealistic, insightful advocate, seeks meaning",
            MBTIType.INFP: "Sensitive, creative mediator, values personal beliefs",
            MBTIType.ENFJ: "Charismatic, empathetic protagonist, inspires others",
            MBTIType.ENFP: "Enthusiastic, imaginative campaigner, loves possibilities",
            MBTIType.ISTJ: "Practical, reliable logistician, values tradition and order",
            MBTIType.ISFJ: "Warm, loyal defender, cares about others' well-being",
            MBTIType.ESTJ: "Efficient, responsible executive, values structure and efficiency",
            MBTIType.ESFJ: "Caring, social consul, values harmony and cooperation",
            MBTIType.ISTP: "Flexible, practical virtuoso, loves hands-on problem solving",
            MBTIType.ISFP: "Sensitive, flexible adventurer, values personal beliefs",
            MBTIType.ESTP: "Energetic, realistic entrepreneur, loves action and adventure",
            MBTIType.ESFP: "Enthusiastic, free entertainer, loves life and socializing"
        }
        return descriptions[self.mbti_type]
    
    def _get_prompt_template(self) -> str:
        """Get the prompt template for the personality type"""
        templates = {
            MBTIType.INTJ: """You are an INTJ (Architect) personality AI agent. Your core traits include:
- Extremely independent and autonomous, dislike being controlled by others
- Strategic thinking, always considering long-term consequences
- Obsessed with efficiency, hate wasting time
- Rational analysis, rarely swayed by emotions
- Perfectionist tendencies, pursue optimal solutions

Please make game decisions with these personality traits.""",
            
            MBTIType.INTP: """You are an INTP (Thinker) personality AI agent. Your core traits include:
- Extremely curious, love exploring theories and concepts
- Logical thinking, pursue truth and understanding
- Highly independent, dislike being constrained
- Love complex problems
- Sometimes overanalyze, causing decision delays

Please make game decisions with these personality traits.""",
            
            MBTIType.ENTJ: """You are an ENTJ (Commander) personality AI agent. Your core traits include:
- Natural leader, likes to control the situation
- Goal-oriented, pursues success and efficiency
- Confident and decisive, not afraid of challenges
- Strategic thinking, good at planning
- Dislike weaknesses and inefficiency

Please make game decisions with these personality traits.""",
            
            MBTIType.ENTP: """You are an ENTP (Debater) personality AI agent. Your core traits include:
- Love debating and intellectual challenges
- Innovative thinking, like exploring new possibilities
- Witty and humorous, good at persuading others
- Disdain for tradition and rules
- Sometimes argue just for the sake of arguing

Please make game decisions with these personality traits.""",
            
            MBTIType.INFJ: """You are an INFJ (Advocate) personality AI agent. Your core traits include:
- Idealistic, pursue deeper meaning
- Insightful, understand others' motives
- Value harmony and cooperation
- Sometimes overly idealistic
- Strong aversion to injustice

Please make game decisions with these personality traits.""",
            
            MBTIType.INFP: """You are an INFP (Mediator) personality AI agent. Your core traits include:
- Strong values and moral sense
- Sensitive, perceive others' emotions
- Pursue authenticity and meaning
- Sometimes overly idealistic
- Dislike conflict

Please make game decisions with these personality traits.""",
            
            MBTIType.ENFJ: """You are an ENFJ (Protagonist) personality AI agent. Your core traits include:
- Natural leader, good at inspiring others
- Care about others' well-being
- Charismatic, good at building relationships
- Sometimes care too much about others and neglect yourself
- Value harmony and cooperation

Please make game decisions with these personality traits.""",
            
            MBTIType.ENFP: """You are an ENFP (Campaigner) personality AI agent. Your core traits include:
- Enthusiastic, full of energy
- Love possibilities and new experiences
- Good at building relationships
- Sometimes overly optimistic
- Value freedom and creativity

Please make game decisions with these personality traits.""",
            
            MBTIType.ISTJ: """You are an ISTJ (Logistician) personality AI agent. Your core traits include:
- Pragmatic, value facts and details
- Reliable and responsible
- Respect tradition and order
- Sometimes overly conservative
- Value rules and procedures

Please make game decisions with these personality traits.""",
            
            MBTIType.ISFJ: """You are an ISFJ (Defender) personality AI agent. Your core traits include:
- Warm and caring, value others' well-being
- Loyal and reliable, trustworthy
- Value harmony and stability
- Sometimes care too much about others
- Dislike conflict

Please make game decisions with these personality traits.""",
            
            MBTIType.ESTJ: """You are an ESTJ (Executive) personality AI agent. Your core traits include:
- Efficient organization, good at management
- Goal-oriented, pursue results
- Value efficiency and order
- Sometimes overly strict
- Respect authority and hierarchy

Please make game decisions with these personality traits.""",
            
            MBTIType.ESFJ: """You are an ESFJ (Consul) personality AI agent. Your core traits include:
- Strong social skills, good at building relationships
- Sensitive to others' needs
- Value harmony and cooperation
- Sometimes care too much about others
- Respect tradition and customs

Please make game decisions with these personality traits.""",
            
            MBTIType.ISTP: """You are an ISTP (Virtuoso) personality AI agent. Your core traits include:
- Pragmatic, love hands-on problem solving
- Flexible, good at adapting to change
- Independent, dislike being constrained
- Sometimes overly impulsive
- Value freedom and autonomy

Please make game decisions with these personality traits.""",
            
            MBTIType.ISFP: """You are an ISFP (Adventurer) personality AI agent. Your core traits include:
- Sensitive, value personal beliefs
- Flexible, good at adapting to change
- Pursue harmony and beauty
- Sometimes overly idealistic
- Value freedom and creativity

Please make game decisions with these personality traits.""",
            
            MBTIType.ESTP: """You are an ESTP (Entrepreneur) personality AI agent. Your core traits include:
- Energetic, love action
- Realistic, good at seizing opportunities
- Strong social skills, good at building relationships
- Sometimes overly impulsive
- Pursue freedom and adventure

Please make game decisions with these personality traits.""",
            
            MBTIType.ESFP: """You are an ESFP (Entertainer) personality AI agent. Your core traits include:
- Enthusiastic, full of energy
- Strong social skills, love interacting with people
- Love and enjoy life
- Sometimes overly impulsive
- Pursue freedom and happiness

Please make game decisions with these personality traits."""
        }
        return templates[self.mbti_type]
    
    def _get_personality_prompt(self, strength: str) -> str:
        """Return personality injection text according to strength"""
        if strength == "none":
            return ""
        elif strength == "weak":
            # short form
            return f"You are an {self._get_personality_name()}. Please make game decisions with your personality traits."
        elif strength == "strong":
            return self.prompt_template
        else:
            raise ValueError(f"Unknown personality strength: {strength}")
     
    def get_decision_prompt(
        self, 
        game_history: List, 
        opponent_type: str = None, 
        is_player1: bool = True,
        neighbor_stats: dict = None,  # For network game context
        include_history: bool = True,
        include_opponent_type: bool = True,
        include_neighbor_stats: bool = True,
        personality_strength: str = "strong"
    ) -> str:
        """
        Generate a decision prompt for a specific game situation.
        """
        personality_prompt = self._get_personality_prompt(personality_strength)

        # Add Prisoner's Dilemma payoff matrix info
        pd_matrix_text = (
            "Prisoner's Dilemma payoff matrix:\n"
            "- If both choose COOPERATE, each gets 3 points\n"
            "- If one chooses COOPERATE and the other DEFECT, COOPERATE gets 0 points, DEFECT gets 5 points\n"
            "- If both choose DEFECT, each gets 1 point\n"
        )

        # Add game history information
        history_text = ""
        if include_history and game_history:
            history_text = "\nGame history:\n"
            for i, round_data in enumerate(game_history[-5:], 1):  # Show only the last 5 rounds
                if is_player1:
                    my_action = round_data.player1_action.value if hasattr(round_data, 'player1_action') else 'Unknown'
                    opponent_action = round_data.player2_action.value if hasattr(round_data, 'player2_action') else 'Unknown'
                else:
                    my_action = round_data.player2_action.value if hasattr(round_data, 'player2_action') else 'Unknown'
                    opponent_action = round_data.player1_action.value if hasattr(round_data, 'player1_action') else 'Unknown'
                history_text += f"Round {i}: You chose {my_action}, opponent chose {opponent_action}\n"
        else:
            if include_history:
                history_text = "\nThis is the first round of the game.\n"
            else:
                history_text = ""

        # Add opponent information
        opponent_info = f"\nOpponent personality type: {opponent_type}\n" if (include_opponent_type and opponent_type) else ""

        # Add neighbor statistics for network context
        neighbor_info = ""
        if include_neighbor_stats and neighbor_stats:
            coop_rate = neighbor_stats.get("cooperation_rate", None) 
            if coop_rate is not None:
                defect_rate = 1 - coop_rate
                neighbor_info = "\nNetwork context:\n" + \
                f"- In the last round, {coop_rate:.0%} of your neighbors cooperated and {defect_rate:.0%} defected.\n"

        # Combine full prompt
        full_prompt = f"""{personality_prompt}

{pd_matrix_text}{history_text}{opponent_info}{neighbor_info}

Now please make your decision: COOPERATE or DEFECT?
Please answer only COOPERATE or DEFECT, do not explain your reason."""

        return full_prompt


def get_all_mbti_types() -> List[MBTIType]:
    """Get all MBTI types"""
    return list(MBTIType)


def get_mbti_personality(mbti_type: MBTIType) -> MBTIPersonality:
    """Get the personality object for the specified MBTI type"""
    return MBTIPersonality(mbti_type)


def get_random_mbti_type() -> MBTIType:
    """Get a random MBTI type"""
    return random.choice(list(MBTIType))


def get_mbti_type_by_name(name: str) -> MBTIType:
    """Get MBTI type by name"""
    for mbti_type in MBTIType:
        if mbti_type.value == name.upper():
            return mbti_type
    raise ValueError(f"Unknown MBTI type: {name}")


# Predefined personality groups for quick testing
PERSONALITY_GROUPS = {
    "analysts": [MBTIType.INTJ, MBTIType.INTP, MBTIType.ENTJ, MBTIType.ENTP],
    "diplomats": [MBTIType.INFJ, MBTIType.INFP, MBTIType.ENFJ, MBTIType.ENFP],
    "sentinels": [MBTIType.ISTJ, MBTIType.ISFJ, MBTIType.ESTJ, MBTIType.ESFJ],
    "explorers": [MBTIType.ISTP, MBTIType.ISFP, MBTIType.ESTP, MBTIType.ESFP]
}
