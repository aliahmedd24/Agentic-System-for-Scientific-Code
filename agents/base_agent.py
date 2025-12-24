"""
Base Agent - Abstract base class for all agents.

Provides:
- Unified logging interface
- Operation timing and statistics
- LLM client management
- Standard property accessors (agent_id, agent_type)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from core.llm_client import LLMClient
from core.knowledge_graph import KnowledgeGraph
from core.error_handling import logger, LogCategory
from core.metrics import get_metrics_collector


@dataclass
class AgentStats:
    """Statistics for agent operations."""
    operations: int = 0
    total_duration_ms: float = 0
    errors: int = 0
    last_operation: Optional[datetime] = None

    @property
    def avg_duration_ms(self) -> float:
        """Average duration per operation in milliseconds."""
        return self.total_duration_ms / self.operations if self.operations > 0 else 0


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.

    Provides common functionality:
    - LLM client access
    - Knowledge graph integration
    - Logging utilities
    - Performance tracking
    - Standard properties (agent_id, agent_type)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        name: str = "BaseAgent"
    ):
        self.llm = llm_client
        self.name = name
        self._stats = AgentStats()
        self._logger = logger
        self._metrics = get_metrics_collector()

    # ========================================================================
    # Required Properties (for protocol compliance)
    # ========================================================================

    @property
    def agent_id(self) -> str:
        """Unique identifier for this agent instance."""
        return self.name.lower().replace(' ', '_').replace('-', '_')

    @property
    def agent_type(self) -> str:
        """Type name of this agent class."""
        return self.__class__.__name__

    # ========================================================================
    # Logging Helpers
    # ========================================================================

    def log_info(self, message: str, **kwargs):
        """Log an info message."""
        self._logger.info(message, category=LogCategory.AGENT, agent=self.name, **kwargs)

    def log_warning(self, message: str, **kwargs):
        """Log a warning message."""
        self._logger.warning(message, category=LogCategory.AGENT, agent=self.name, **kwargs)

    def log_error(self, message: str, **kwargs):
        """Log an error message."""
        self._logger.error(message, category=LogCategory.AGENT, agent=self.name, **kwargs)
        self._stats.errors += 1

    def log_debug(self, message: str, **kwargs):
        """Log a debug message."""
        self._logger.debug(message, category=LogCategory.AGENT, agent=self.name, **kwargs)

    # ========================================================================
    # Timed Operations
    # ========================================================================

    async def _timed_operation(self, operation_name: str, coro):
        """Execute a coroutine and track its duration."""
        start = datetime.now()
        self.log_debug(f"Starting operation: {operation_name}")

        try:
            result = await coro
            duration_ms = (datetime.now() - start).total_seconds() * 1000

            self._stats.operations += 1
            self._stats.total_duration_ms += duration_ms
            self._stats.last_operation = datetime.now()

            self.log_debug(f"Completed {operation_name} in {duration_ms:.1f}ms")

            # Record metrics for successful operation
            self._metrics.record_agent_operation(
                agent_name=self.name,
                operation=operation_name,
                duration_ms=duration_ms,
                success=True
            )

            return result
        except Exception as e:
            duration_ms = (datetime.now() - start).total_seconds() * 1000
            self.log_error(f"{operation_name} failed after {duration_ms:.1f}ms: {str(e)}")

            # Record metrics for failed operation
            self._metrics.record_agent_operation(
                agent_name=self.name,
                operation=operation_name,
                duration_ms=duration_ms,
                success=False
            )

            raise

    # ========================================================================
    # Statistics (with correct key names for compatibility)
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        return {
            "agent": self.name,
            "agent_id": self.agent_id,
            "type": self.agent_type,
            "operation_count": self._stats.operations,
            "operations": self._stats.operations,  # Alias for compatibility
            "total_duration_ms": self._stats.total_duration_ms,
            "avg_duration_ms": self._stats.avg_duration_ms,
            "errors": self._stats.errors,
            "last_operation": self._stats.last_operation.isoformat() if self._stats.last_operation else None
        }

    # ========================================================================
    # Abstract Method (subclasses must implement)
    # ========================================================================

    @abstractmethod
    async def process(self, **kwargs) -> Dict[str, Any]:
        """
        Process input and produce output.

        Subclasses should define specific keyword arguments
        matching their input contract from agents/protocols.py.

        Returns:
            Dictionary with agent-specific output
        """
        pass