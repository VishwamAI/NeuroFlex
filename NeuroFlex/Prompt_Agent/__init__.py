"""
NeuroFlex Prompt Agent Module

This module provides various prompt agents for AI tasks, including zero-shot, few-shot, and chain-of-thought agents.

Recent updates:
- Enhanced support for meta-prompting and self-consistency
- Improved agent creation and initialization functions
- Added support for knowledge prompting agents
"""

from .agentic_behavior import (
    ZeroShotAgent,
    FewShotAgent,
    ChainOfThoughtAgent,
    MetaPromptingAgent,
    BaseAgent,
    SelfConsistencyAgent,
    GenerateKnowledgePromptingAgent
)

__all__ = [
    'ZeroShotAgent',
    'FewShotAgent',
    'ChainOfThoughtAgent',
    'MetaPromptingAgent',
    'BaseAgent',
    'SelfConsistencyAgent',
    'GenerateKnowledgePromptingAgent',
    'get_prompt_agent_version',
    'SUPPORTED_AGENT_TYPES',
    'initialize_prompt_agent',
    'create_agent'
]

def get_prompt_agent_version():
    return "1.0.0"

SUPPORTED_AGENT_TYPES = [
    "ZeroShot",
    "FewShot",
    "ChainOfThought",
    "MetaPrompting",
    "Base",
    "SelfConsistency",
    "GenerateKnowledgePrompting"
]

def initialize_prompt_agent():
    print("Initializing Prompt Agent Module...")
    # Add any necessary initialization code here

def create_agent(agent_type, *args, **kwargs):
    if agent_type == "ZeroShot":
        return ZeroShotAgent(*args, **kwargs)
    elif agent_type == "FewShot":
        return FewShotAgent(*args, **kwargs)
    elif agent_type == "ChainOfThought":
        return ChainOfThoughtAgent(*args, **kwargs)
    elif agent_type == "MetaPrompting":
        return MetaPromptingAgent(*args, **kwargs)
    elif agent_type == "Base":
        return BaseAgent(*args, **kwargs)
    elif agent_type == "SelfConsistency":
        return SelfConsistencyAgent(*args, **kwargs)
    elif agent_type == "GenerateKnowledgePrompting":
        return GenerateKnowledgePromptingAgent(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

# Add any other Prompt Agent-specific utility functions or constants as needed
