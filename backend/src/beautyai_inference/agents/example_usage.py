"""
Example usage of the agent delegation system.

This file demonstrates how to use the AgentDelegate for task routing.
"""

from beautyai_inference.agents import AgentDelegate, BaseAgent
from typing import Dict, Any


# Example 1: Simple task-specific agents
class ChatAgent(BaseAgent):
    """Agent for handling chat/conversation tasks."""
    
    def __init__(self):
        super().__init__("ChatAgent")
    
    def handle(self, task: Dict[str, Any]) -> Any:
        prompt = task.get("prompt", "")
        return f"Chat response to: {prompt}"
    
    def can_handle(self, task: Dict[str, Any]) -> bool:
        return "prompt" in task and task.get("task_type") == "chat"


class TranscriptionAgent(BaseAgent):
    """Agent for handling transcription tasks."""
    
    def __init__(self):
        super().__init__("TranscriptionAgent")
    
    def handle(self, task: Dict[str, Any]) -> Any:
        audio_data = task.get("audio_data", b"")
        return f"Transcribed {len(audio_data)} bytes of audio"
    
    def can_handle(self, task: Dict[str, Any]) -> bool:
        return "audio_data" in task


class DefaultAgent(BaseAgent):
    """Default fallback agent."""
    
    def __init__(self):
        super().__init__("DefaultAgent")
    
    def handle(self, task: Dict[str, Any]) -> Any:
        return f"Default handling for task: {task.get('task_type', 'unknown')}"


def example_basic_usage():
    """Demonstrate basic agent delegation."""
    print("=== Basic Agent Delegation Example ===\n")
    
    # Create delegate and register agents
    delegate = AgentDelegate()
    delegate.register_agent("chat", ChatAgent())
    delegate.register_agent("transcription", TranscriptionAgent())
    delegate.register_agent("default", DefaultAgent(), is_default=True)
    
    # List registered agents
    print("Registered agents:")
    for agent_type, agent_name in delegate.list_agents().items():
        print(f"  - {agent_type}: {agent_name}")
    print()
    
    # Delegate various tasks
    print("Task 1 - Chat:")
    result1 = delegate.delegate("chat", {"prompt": "Hello, how are you?"})
    print(f"  Result: {result1}\n")
    
    print("Task 2 - Transcription:")
    result2 = delegate.delegate("transcription", {"audio_data": b"audio bytes here"})
    print(f"  Result: {result2}\n")
    
    print("Task 3 - Unknown type (fallback to default):")
    result3 = delegate.delegate("unknown", {"task_type": "unknown", "data": "test"})
    print(f"  Result: {result3}\n")


def example_can_handle():
    """Demonstrate can_handle method for dynamic routing."""
    print("=== Dynamic Routing with can_handle() ===\n")
    
    delegate = AgentDelegate()
    delegate.register_agent("chat", ChatAgent())
    delegate.register_agent("transcription", TranscriptionAgent())
    
    # Task without explicit type - agents will check can_handle()
    print("Task with audio_data (no explicit type):")
    result = delegate.delegate("auto", {"audio_data": b"some audio data"})
    print(f"  Result: {result}\n")


if __name__ == "__main__":
    example_basic_usage()
    example_can_handle()
