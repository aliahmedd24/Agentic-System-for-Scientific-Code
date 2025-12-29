"""
Structured error handling and retry logic for the scientific agent system.
"""

import logging
import traceback
import random
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any, Callable, TypeVar, Awaitable
from functools import wraps
import asyncio

from pydantic import BaseModel, Field, ConfigDict, field_serializer

# Type variable for generic function return types
T = TypeVar('T')


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogCategory(Enum):
    SYSTEM = "system"
    AGENT = "agent"
    LLM = "llm"
    PIPELINE = "pipeline"
    EXECUTION = "execution"
    NETWORK = "network"
    VALIDATION = "validation"
    PARSING = "parsing"
    USER = "user"


class ErrorSeverity(Enum):
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ErrorCategory(Enum):
    NETWORK = "network"
    PARSING = "parsing"
    VALIDATION = "validation"
    EXECUTION = "execution"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    LLM = "llm"
    CONFIGURATION = "configuration"


class StructuredLog(BaseModel):
    """Structured log entry with full context."""
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    timestamp: datetime = Field(..., description="Log timestamp")
    level: LogLevel = Field(..., description="Log level")
    category: LogCategory = Field(..., description="Log category")
    agent: Optional[str] = Field(None, description="Agent name")
    stage: Optional[str] = Field(None, description="Pipeline stage")
    message: str = Field("", description="Log message")
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional data")
    error: Optional[Exception] = Field(None, description="Exception if any")
    duration_ms: Optional[int] = Field(None, description="Duration in milliseconds")

    @field_serializer('error')
    def serialize_exception(self, exc: Optional[Exception]) -> Optional[str]:
        """Serialize Exception to string for JSON compatibility."""
        if exc is None:
            return None
        return f"{type(exc).__name__}: {str(exc)}"


class StructuredError(BaseModel):
    """Structured error with context and recovery information."""
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    category: ErrorCategory = Field(..., description="Error category")
    severity: ErrorSeverity = Field(..., description="Error severity")
    message: str = Field(..., description="Error message")
    original_error: Optional[Exception] = Field(None, description="Original exception")
    context: Dict[str, Any] = Field(default_factory=dict, description="Error context")
    suggestion: str = Field("", description="Suggested fix")
    recoverable: bool = Field(True, description="Whether error is recoverable")
    retry_allowed: bool = Field(True, description="Whether retry is allowed")

    @field_serializer('original_error')
    def serialize_exception(self, exc: Optional[Exception]) -> Optional[str]:
        """Serialize Exception to string for JSON compatibility."""
        if exc is None:
            return None
        return f"{type(exc).__name__}: {str(exc)}"

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.category.value}: {self.message}"


class AgentError(Exception):
    """Custom exception for agent-related errors."""
    def __init__(self, structured_error: StructuredError):
        self.structured_error = structured_error
        super().__init__(str(structured_error))


class RetryStrategy:
    """
    Configurable retry logic with exponential backoff and jitter.
    """
    DEFAULT_CONFIG = {
        ErrorCategory.NETWORK: {"max_attempts": 3, "base_delay": 1.0, "max_delay": 30.0},
        ErrorCategory.TIMEOUT: {"max_attempts": 2, "base_delay": 2.0, "max_delay": 20.0},
        ErrorCategory.LLM: {"max_attempts": 3, "base_delay": 1.5, "max_delay": 30.0},
        ErrorCategory.PARSING: {"max_attempts": 1, "base_delay": 0, "max_delay": 0},
        ErrorCategory.VALIDATION: {"max_attempts": 1, "base_delay": 0, "max_delay": 0},
        ErrorCategory.EXECUTION: {"max_attempts": 2, "base_delay": 1.0, "max_delay": 10.0},
        ErrorCategory.RESOURCE: {"max_attempts": 2, "base_delay": 1.0, "max_delay": 10.0},
        ErrorCategory.CONFIGURATION: {"max_attempts": 1, "base_delay": 0, "max_delay": 0},
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self.DEFAULT_CONFIG

    def get_config(self, category: ErrorCategory) -> Dict:
        return self.config.get(category, {"max_attempts": 1, "base_delay": 0, "max_delay": 0})

    def calculate_delay(self, category: ErrorCategory, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        config = self.get_config(category)
        base_delay = config["base_delay"]
        max_delay = config["max_delay"]
        
        if base_delay == 0:
            return 0
        
        # Exponential backoff: base_delay * 2^attempt
        delay = base_delay * (2 ** attempt)
        # Add jitter (±25%)
        jitter = delay * 0.25 * (random.random() * 2 - 1)
        delay = delay + jitter
        # Cap at max_delay
        return min(delay, max_delay)


class SystemLogger:
    """
    Structured logging system with multiple outputs.
    """
    _instance = None
    _callbacks: list = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.logs: list[StructuredLog] = []
        self._callbacks = []
        
        # Configure Python logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger("scientific-agent")

    def add_callback(self, callback: Callable[[StructuredLog], None]):
        """Add a callback for log events (e.g., WebSocket broadcast)."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[StructuredLog], None]):
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def log(
        self,
        level: LogLevel,
        category: LogCategory,
        message: str,
        agent: Optional[str] = None,
        stage: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
        duration_ms: Optional[int] = None
    ) -> StructuredLog:
        """Create and broadcast a structured log entry."""
        log_entry = StructuredLog(
            timestamp=datetime.now(),
            level=level,
            category=category,
            agent=agent,
            stage=stage,
            message=message,
            data=data or {},
            error=error,
            duration_ms=duration_ms
        )
        
        self.logs.append(log_entry)
        
        # Python logging
        log_func = getattr(self.logger, level.value)
        prefix = f"[{category.value}]"
        if agent:
            prefix += f"[{agent}]"
        if stage:
            prefix += f"[{stage}]"
        log_func(f"{prefix} {message}")
        
        if error:
            self.logger.error(f"Error details: {error}")
            self.logger.debug(traceback.format_exc())
        
        # Broadcast to callbacks
        for callback in self._callbacks:
            try:
                callback(log_entry)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
        
        return log_entry

    def debug(self, message: str, category: LogCategory = LogCategory.SYSTEM, context: Optional[Dict[str, Any]] = None, **kwargs):
        kwargs['data'] = context or kwargs.get('data')
        return self.log(LogLevel.DEBUG, category, message, **kwargs)

    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM, context: Optional[Dict[str, Any]] = None, **kwargs):
        kwargs['data'] = context or kwargs.get('data')
        return self.log(LogLevel.INFO, category, message, **kwargs)

    def warning(self, message: str, category: LogCategory = LogCategory.SYSTEM, context: Optional[Dict[str, Any]] = None, **kwargs):
        kwargs['data'] = context or kwargs.get('data')
        return self.log(LogLevel.WARNING, category, message, **kwargs)

    def error(self, message: str, category: LogCategory = LogCategory.SYSTEM, context: Optional[Dict[str, Any]] = None, **kwargs):
        kwargs['data'] = context or kwargs.get('data')
        return self.log(LogLevel.ERROR, category, message, **kwargs)

    def critical(self, message: str, category: LogCategory = LogCategory.SYSTEM, context: Optional[Dict[str, Any]] = None, **kwargs):
        kwargs['data'] = context or kwargs.get('data')
        return self.log(LogLevel.CRITICAL, category, message, **kwargs)

    def get_recent_logs(self, count: int = 100) -> list[Dict]:
        """Get recent logs as dictionaries."""
        return [log.model_dump() for log in self.logs[-count:]]


# Global logger instance
logger = SystemLogger()


def with_retry(
    category: ErrorCategory,
    operation_name: str = "operation"
):
    """
    Decorator for async functions that adds retry logic with exponential backoff.
    """
    strategy = RetryStrategy()

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            config = strategy.get_config(category)
            max_attempts = config["max_attempts"]
            last_error = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    
                    if attempt < max_attempts - 1:
                        delay = strategy.calculate_delay(category, attempt)
                        logger.warning(
                            f"{operation_name} failed (attempt {attempt + 1}/{max_attempts}), "
                            f"retrying in {delay:.1f}s: {str(e)}",
                            category=LogCategory.AGENT
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"{operation_name} failed after {max_attempts} attempts: {str(e)}",
                            category=LogCategory.AGENT
                        )

            raise AgentError(StructuredError(
                category=category,
                severity=ErrorSeverity.ERROR,
                message=f"{operation_name} failed after {max_attempts} attempts",
                original_error=last_error,
                suggestion="Check the error details and try again",
                recoverable=False,
                retry_allowed=False
            ))

        return wrapper
    return decorator


def create_error(
    category: ErrorCategory,
    message: str,
    original_error: Optional[Exception] = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    suggestion: str = "",
    context: Optional[Dict[str, Any]] = None,
    recoverable: bool = True
) -> StructuredError:
    """Helper function to create structured errors."""
    return StructuredError(
        category=category,
        severity=severity,
        message=message,
        original_error=original_error,
        context=context or {},
        suggestion=suggestion or f"Check the {category.value} configuration and try again",
        recoverable=recoverable,
        retry_allowed=recoverable
    )