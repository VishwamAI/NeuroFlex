# MIT License
# 
# Copyright (c) 2024 VishwamAI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
