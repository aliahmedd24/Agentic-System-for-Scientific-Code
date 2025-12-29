"""
Tests for the Error Handling module.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

from core.error_handling import (
    LogLevel,
    LogCategory,
    StructuredLog,
    ErrorCategory,
    ErrorSeverity,
    StructuredError,
    AgentError,
    RetryStrategy,
    SystemLogger,
    with_retry,
)


class TestLogLevel:
    """Tests for LogLevel enum."""
    
    def test_log_levels_ordered(self):
        """Test log levels have correct ordering."""
        assert LogLevel.DEBUG.value < LogLevel.INFO.value
        assert LogLevel.INFO.value < LogLevel.WARNING.value
        assert LogLevel.WARNING.value < LogLevel.ERROR.value
        assert LogLevel.ERROR.value < LogLevel.CRITICAL.value


class TestLogCategory:
    """Tests for LogCategory enum."""
    
    def test_all_categories_exist(self):
        """Verify all expected categories exist."""
        expected = ["SYSTEM", "NETWORK", "PARSING", "LLM", "AGENT", "EXECUTION", "USER"]
        for cat in expected:
            assert hasattr(LogCategory, cat)


class TestStructuredLog:
    """Tests for StructuredLog dataclass."""
    
    def test_create_log(self):
        """Test creating a structured log entry."""
        log = StructuredLog(
            level=LogLevel.INFO,
            category=LogCategory.SYSTEM,
            message="Test message"
        )
        
        assert log.level == LogLevel.INFO
        assert log.category == LogCategory.SYSTEM
        assert log.message == "Test message"
        assert log.timestamp is not None
    
    def test_log_with_context(self):
        """Test log with additional context."""
        log = StructuredLog(
            level=LogLevel.ERROR,
            category=LogCategory.LLM,
            message="API error",
            agent_id="paper_parser",
            context={"model": "gpt-4", "error_code": 500},
            duration_ms=1500
        )
        
        assert log.agent_id == "paper_parser"
        assert log.context["model"] == "gpt-4"
        assert log.duration_ms == 1500
    
    def test_log_to_dict(self):
        """Test converting log to dictionary."""
        log = StructuredLog(
            level=LogLevel.INFO,
            category=LogCategory.AGENT,
            message="Processing"
        )

        log_dict = log.model_dump()
        
        assert "level" in log_dict
        assert "category" in log_dict
        assert "message" in log_dict
        assert "timestamp" in log_dict


class TestErrorCategory:
    """Tests for ErrorCategory enum."""
    
    def test_all_error_categories_exist(self):
        """Verify all expected error categories exist."""
        expected = [
            "NETWORK", "PARSING", "VALIDATION", "EXECUTION",
            "RESOURCE", "TIMEOUT", "LLM", "CONFIGURATION"
        ]
        for cat in expected:
            assert hasattr(ErrorCategory, cat)


class TestStructuredError:
    """Tests for StructuredError dataclass."""
    
    def test_create_error(self):
        """Test creating a structured error."""
        error = StructuredError(
            category=ErrorCategory.NETWORK,
            message="Connection failed",
            severity=ErrorSeverity.HIGH
        )
        
        assert error.category == ErrorCategory.NETWORK
        assert error.message == "Connection failed"
        assert error.severity == ErrorSeverity.HIGH
        assert error.recoverable == True  # default
    
    def test_error_with_suggestions(self):
        """Test error with recovery suggestions."""
        error = StructuredError(
            category=ErrorCategory.LLM,
            message="Rate limit exceeded",
            severity=ErrorSeverity.MEDIUM,
            suggestions=["Wait 60 seconds", "Use a different API key"],
            recoverable=True
        )
        
        assert len(error.suggestions) == 2
        assert "Wait 60 seconds" in error.suggestions


class TestAgentError:
    """Tests for AgentError exception."""
    
    def test_create_agent_error(self):
        """Test creating an agent error."""
        structured_error = StructuredError(
            category=ErrorCategory.PARSING,
            message="Failed to parse PDF",
            severity=ErrorSeverity.HIGH
        )
        
        error = AgentError("paper_parser", structured_error)
        
        assert error.agent_id == "paper_parser"
        assert error.error.category == ErrorCategory.PARSING
        assert "paper_parser" in str(error)
    
    def test_agent_error_inheritance(self):
        """Test AgentError is an Exception."""
        structured_error = StructuredError(
            category=ErrorCategory.EXECUTION,
            message="Test error",
            severity=ErrorSeverity.LOW
        )
        
        error = AgentError("test_agent", structured_error)
        
        assert isinstance(error, Exception)
        
        with pytest.raises(AgentError):
            raise error


class TestRetryStrategy:
    """Tests for RetryStrategy class."""
    
    def test_default_strategy(self):
        """Test default retry strategy."""
        strategy = RetryStrategy()
        
        assert strategy.max_retries == 3
        assert strategy.base_delay == 1.0
        assert strategy.max_delay == 60.0
        assert strategy.exponential_base == 2.0
    
    def test_custom_strategy(self):
        """Test custom retry strategy."""
        strategy = RetryStrategy(
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
            exponential_base=3.0
        )
        
        assert strategy.max_retries == 5
        assert strategy.base_delay == 2.0
    
    def test_get_delay(self):
        """Test delay calculation."""
        strategy = RetryStrategy(base_delay=1.0, exponential_base=2.0, max_delay=60.0)
        
        # First retry
        delay1 = strategy.get_delay(0)
        assert 1.0 <= delay1 <= 2.0  # Base delay with jitter
        
        # Second retry (exponential)
        delay2 = strategy.get_delay(1)
        assert delay2 >= delay1
        
        # Should not exceed max_delay
        delay_large = strategy.get_delay(100)
        assert delay_large <= 60.0 * 1.5  # max_delay with jitter


class TestSystemLogger:
    """Tests for SystemLogger class."""
    
    def test_singleton(self):
        """Test SystemLogger is a singleton."""
        logger1 = SystemLogger()
        logger2 = SystemLogger()
        
        assert logger1 is logger2
    
    def test_log_info(self):
        """Test logging info message."""
        logger = SystemLogger()
        
        # Should not raise
        logger.info("Test info message", category=LogCategory.SYSTEM)
    
    def test_log_error(self):
        """Test logging error message."""
        logger = SystemLogger()
        
        logger.error(
            "Test error",
            category=LogCategory.LLM,
            context={"model": "test"}
        )
    
    def test_log_with_callback(self):
        """Test logging triggers callbacks."""
        logger = SystemLogger()
        callback_called = []
        
        def test_callback(log: StructuredLog):
            callback_called.append(log)
        
        logger.add_callback(test_callback)
        logger.info("Test callback", category=LogCategory.SYSTEM)
        
        assert len(callback_called) > 0
        assert callback_called[-1].message == "Test callback"
        
        # Cleanup
        logger.remove_callback(test_callback)
    
    def test_get_logs(self):
        """Test retrieving logs."""
        logger = SystemLogger()
        
        logger.info("Test log 1", category=LogCategory.SYSTEM)
        logger.info("Test log 2", category=LogCategory.AGENT)
        
        all_logs = logger.get_logs()
        assert len(all_logs) >= 2
        
        # Filter by category
        system_logs = logger.get_logs(category=LogCategory.SYSTEM)
        assert all(log.category == LogCategory.SYSTEM for log in system_logs)


class TestWithRetry:
    """Tests for with_retry decorator."""
    
    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Test successful function doesn't retry."""
        call_count = 0
        
        @with_retry(max_retries=3)
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await successful_func()
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test function retries on failure."""
        call_count = 0
        
        @with_retry(max_retries=3, base_delay=0.01)
        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success"
        
        result = await failing_then_success()
        
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test exception raised after max retries."""
        call_count = 0
        
        @with_retry(max_retries=2, base_delay=0.01)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            await always_fails()
        
        assert call_count == 3  # Initial + 2 retries
    
    @pytest.mark.asyncio
    async def test_retry_with_category(self):
        """Test retry with specific error category."""
        @with_retry(max_retries=2, category=ErrorCategory.NETWORK, base_delay=0.01)
        async def network_func():
            raise ConnectionError("Network error")
        
        with pytest.raises(ConnectionError):
            await network_func()


class TestErrorSeverity:
    """Tests for ErrorSeverity enum."""
    
    def test_severity_levels(self):
        """Test severity levels exist."""
        assert hasattr(ErrorSeverity, "LOW")
        assert hasattr(ErrorSeverity, "MEDIUM")
        assert hasattr(ErrorSeverity, "HIGH")
        assert hasattr(ErrorSeverity, "CRITICAL")
    
    def test_severity_ordering(self):
        """Test severity levels are ordered."""
        assert ErrorSeverity.LOW.value < ErrorSeverity.MEDIUM.value
        assert ErrorSeverity.MEDIUM.value < ErrorSeverity.HIGH.value
        assert ErrorSeverity.HIGH.value < ErrorSeverity.CRITICAL.value
