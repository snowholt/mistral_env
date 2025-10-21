"""
Agent delegation system for BeautyAI Inference Framework.

This module provides a simple agent delegation pattern for routing tasks
to appropriate handlers based on task type.
"""

from .agent_delegate import AgentDelegate, BaseAgent

__all__ = ['AgentDelegate', 'BaseAgent']
