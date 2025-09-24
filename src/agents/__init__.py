"""
智能体模块
包含MBTI人格系统和智能体相关功能
"""

from .mbti_personalities import (
    MBTIType,
    MBTIPersonality,
    get_all_mbti_types,
    get_mbti_personality,
    get_random_mbti_type,
    get_mbti_type_by_name,
    PERSONALITY_GROUPS
)

__all__ = [
    "MBTIType",
    "MBTIPersonality",
    "get_all_mbti_types",
    "get_mbti_personality",
    "get_random_mbti_type",
    "get_mbti_type_by_name",
    "PERSONALITY_GROUPS"
]
