"""
Unit tests for agent delegation system.
"""

import pytest
from beautyai_inference.agents import AgentDelegate, BaseAgent
from typing import Dict, Any


class MockChatAgent(BaseAgent):
    """Mock agent for testing chat functionality."""
    
    def __init__(self):
        super().__init__("MockChatAgent")
    
    def handle(self, task: Dict[str, Any]) -> str:
        return f"Handled chat: {task.get('message', '')}"
    
    def can_handle(self, task: Dict[str, Any]) -> bool:
        return "message" in task


class MockTranscriptionAgent(BaseAgent):
    """Mock agent for testing transcription functionality."""
    
    def __init__(self):
        super().__init__("MockTranscriptionAgent")
    
    def handle(self, task: Dict[str, Any]) -> str:
        return f"Transcribed: {len(task.get('audio', b''))} bytes"
    
    def can_handle(self, task: Dict[str, Any]) -> bool:
        return "audio" in task


class MockDefaultAgent(BaseAgent):
    """Mock default agent for testing fallback behavior."""
    
    def __init__(self):
        super().__init__("MockDefaultAgent")
    
    def handle(self, task: Dict[str, Any]) -> str:
        return "Default handler"


class TestAgentDelegate:
    """Test suite for AgentDelegate."""
    
    def test_register_agent(self):
        """Test agent registration."""
        delegate = AgentDelegate()
        agent = MockChatAgent()
        
        delegate.register_agent("chat", agent)
        
        agents = delegate.list_agents()
        assert "chat" in agents
        assert agents["chat"] == "MockChatAgent"
    
    def test_register_default_agent(self):
        """Test default agent registration."""
        delegate = AgentDelegate()
        default_agent = MockDefaultAgent()
        
        delegate.register_agent("default", default_agent, is_default=True)
        
        # Verify default agent is registered
        agents = delegate.list_agents()
        assert "default" in agents
    
    def test_delegate_by_type(self):
        """Test delegation by exact type match."""
        delegate = AgentDelegate()
        delegate.register_agent("chat", MockChatAgent())
        
        result = delegate.delegate("chat", {"message": "Hello"})
        
        assert result == "Handled chat: Hello"
    
    def test_delegate_by_can_handle(self):
        """Test delegation using can_handle method."""
        delegate = AgentDelegate()
        delegate.register_agent("chat", MockChatAgent())
        delegate.register_agent("transcription", MockTranscriptionAgent())
        
        # Task doesn't match "chat" or "transcription" type exactly,
        # but MockTranscriptionAgent.can_handle() should return True
        result = delegate.delegate("unknown", {"audio": b"test audio data"})
        
        assert "Transcribed: 15 bytes" in result
    
    def test_delegate_to_default(self):
        """Test fallback to default agent."""
        delegate = AgentDelegate()
        delegate.register_agent("chat", MockChatAgent())
        delegate.register_agent("default", MockDefaultAgent(), is_default=True)
        
        # Task that doesn't match any specific agent
        result = delegate.delegate("unknown", {"some_data": "value"})
        
        assert result == "Default handler"
    
    def test_delegate_no_match_raises_error(self):
        """Test that delegation raises error when no agent matches."""
        delegate = AgentDelegate()
        delegate.register_agent("chat", MockChatAgent())
        
        with pytest.raises(ValueError, match="No agent found to handle task type"):
            delegate.delegate("unknown", {"some_data": "value"})
    
    def test_list_agents(self):
        """Test listing registered agents."""
        delegate = AgentDelegate()
        delegate.register_agent("chat", MockChatAgent())
        delegate.register_agent("transcription", MockTranscriptionAgent())
        
        agents = delegate.list_agents()
        
        assert len(agents) == 2
        assert agents["chat"] == "MockChatAgent"
        assert agents["transcription"] == "MockTranscriptionAgent"


class TestBaseAgent:
    """Test suite for BaseAgent."""
    
    def test_base_agent_initialization(self):
        """Test BaseAgent initialization."""
        agent = BaseAgent("TestAgent")
        
        assert agent.name == "TestAgent"
    
    def test_handle_not_implemented(self):
        """Test that handle method raises NotImplementedError."""
        agent = BaseAgent("TestAgent")
        
        with pytest.raises(NotImplementedError):
            agent.handle({})
    
    def test_can_handle_default_false(self):
        """Test that can_handle returns False by default."""
        agent = BaseAgent("TestAgent")
        
        assert agent.can_handle({}) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
