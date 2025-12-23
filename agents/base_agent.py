"""
Base Agent - Abstract base class for all agents.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime

from core.llm_client import LLMClient
from core.knowledge_graph import KnowledgeGraph
from core.error_handling import logger, LogCategory


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    
    Provides common functionality:
    - LLM client access
    - Knowledge graph integration
    - Logging utilities
    - Performance tracking
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        name: str = "BaseAgent"
    ):
        self.llm = llm_client
        self.name = name
        self._operation_count = 0
        self._total_duration_ms = 0
    
    @property
    def agent_type(self) -> str:
        """Return the agent type identifier."""
        return self.__class__.__name__
    
    def log_info(self, message: str, **kwargs):
        """Log an info message."""
        logger.info(message, category=LogCategory.AGENT, agent=self.name, **kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """Log a warning message."""
        logger.warning(message, category=LogCategory.AGENT, agent=self.name, **kwargs)
    
    def log_error(self, message: str, **kwargs):
        """Log an error message."""
        logger.error(message, category=LogCategory.AGENT, agent=self.name, **kwargs)
    
    def log_debug(self, message: str, **kwargs):
        """Log a debug message."""
        logger.debug(message, category=LogCategory.AGENT, agent=self.name, **kwargs)
    
    async def _timed_operation(self, operation_name: str, coro):
        """Execute a coroutine and track its duration."""
        start = datetime.now()
        self.log_debug(f"Starting operation: {operation_name}")
        
        try:
            result = await coro
            duration_ms = int((datetime.now() - start).total_seconds() * 1000)
            
            self._operation_count += 1
            self._total_duration_ms += duration_ms
            
            self.log_debug(f"Completed {operation_name} in {duration_ms}ms")
            return result
        except Exception as e:
            duration_ms = int((datetime.now() - start).total_seconds() * 1000)
            self.log_error(f"{operation_name} failed after {duration_ms}ms: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        return {
            "agent": self.name,
            "type": self.agent_type,
            "operations": self._operation_count,
            "total_duration_ms": self._total_duration_ms,
            "avg_duration_ms": self._total_duration_ms / max(1, self._operation_count)
        }
    
    @abstractmethod
    async def process(self, *args, **kwargs) -> Any:
        """
        Main processing method to be implemented by subclasses.
        """
        pass