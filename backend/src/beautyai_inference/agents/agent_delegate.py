"""
Agent delegation implementation.

Provides a simple pattern for delegating tasks to specialized agents.
"""

from typing import Any, Dict, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for all agents in the system."""
    
    def __init__(self, name: str):
        """
        Initialize the agent.
        
        Args:
            name: The name of the agent
        """
        self.name = name
        
    def handle(self, task: Dict[str, Any]) -> Any:
        """
        Handle a delegated task.
        
        Args:
            task: Task data dictionary
            
        Returns:
            Task result
            
        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement handle method")
    
    def can_handle(self, task: Dict[str, Any]) -> bool:
        """
        Check if this agent can handle the given task.
        
        Args:
            task: Task data dictionary
            
        Returns:
            True if agent can handle the task, False otherwise
        """
        return False


class AgentDelegate:
    """
    Central delegation system for routing tasks to appropriate agents.
    
    This class implements a simple delegation pattern where tasks are routed
    to registered agents based on their capabilities.
    """
    
    def __init__(self):
        """Initialize the agent delegate."""
        self._agents: Dict[str, BaseAgent] = {}
        self._default_agent: Optional[BaseAgent] = None
        
    def register_agent(self, agent_type: str, agent: BaseAgent, is_default: bool = False) -> None:
        """
        Register an agent with the delegate.
        
        Args:
            agent_type: The type identifier for this agent
            agent: The agent instance
            is_default: Whether this agent should be the default handler
        """
        self._agents[agent_type] = agent
        if is_default:
            self._default_agent = agent
        logger.info(f"Registered agent '{agent.name}' for type '{agent_type}'")
        
    def delegate(self, task_type: str, task: Dict[str, Any]) -> Any:
        """
        Delegate a task to the appropriate agent.
        
        Args:
            task_type: The type of task to delegate
            task: The task data
            
        Returns:
            The result from the agent that handled the task
            
        Raises:
            ValueError: If no suitable agent is found
        """
        # First try to find agent by exact type match
        if task_type in self._agents:
            agent = self._agents[task_type]
            logger.debug(f"Delegating task to agent '{agent.name}' (type: {task_type})")
            return agent.handle(task)
        
        # Try to find any agent that can handle this task
        for agent in self._agents.values():
            if agent.can_handle(task):
                logger.debug(f"Delegating task to agent '{agent.name}' (can_handle)")
                return agent.handle(task)
        
        # Fall back to default agent if available
        if self._default_agent:
            logger.debug(f"Delegating task to default agent '{self._default_agent.name}'")
            return self._default_agent.handle(task)
        
        raise ValueError(f"No agent found to handle task type: {task_type}")
    
    def list_agents(self) -> Dict[str, str]:
        """
        List all registered agents.
        
        Returns:
            Dictionary mapping agent types to agent names
        """
        return {agent_type: agent.name for agent_type, agent in self._agents.items()}
