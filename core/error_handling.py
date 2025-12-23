"""
Structured error handling and retry logic for the scientific agent system.
"""

import logging
import traceback
import random
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, Callable, TypeVar, Awaitable
from functools import wraps
import asyncio

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


@dataclass
class StructuredLog:
    """Structured log entry with full context."""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    agent: Optional[str] = None
    stage: Optional[str] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None
    duration_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "category": self.category.value,
            "agent": self.agent,
            "stage": self.stage,
            "message": self.message,
            "data": self.data,
            "error": str(self.error) if self.error else None,
            "duration_ms": self.duration_ms
        }


@dataclass
class StructuredError:
    """Structured error with context and recovery information."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    original_error: Optional[Exception] = None
    context: Dict[str, Any] = field(default_factory=dict)
    suggestion: str = ""
    recoverable: bool = True
    retry_allowed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "original_error": str(self.original_error) if self.original_error else None,
            "context": self.context,
            "suggestion": self.suggestion,
            "recoverable": self.recoverable,
            "retry_allowed": self.retry_allowed,
            "traceback": traceback.format_exc() if self.original_error else None
        }

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
        return [log.to_dict() for log in self.logs[-count:]]


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