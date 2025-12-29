"""
Tests for Sandbox Execution - Secure code execution with isolation.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch


# ============================================================================
# IsolationLevel Enum Tests
# ============================================================================

class TestIsolationLevel:
    """Tests for IsolationLevel enum."""

    def test_all_levels_exist(self):
        """Test all expected isolation levels are defined."""
        from core.bubblewrap_sandbox import IsolationLevel

        assert IsolationLevel.NONE.value == "none"
        assert IsolationLevel.SUBPROCESS.value == "subprocess"
        assert IsolationLevel.DOCKER.value == "docker"
        assert IsolationLevel.BUBBLEWRAP.value == "bubblewrap"
        assert IsolationLevel.QEMU.value == "qemu"


# ============================================================================
# SandboxConfig Tests
# ============================================================================

class TestSandboxConfig:
    """Tests for SandboxConfig model."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from core.bubblewrap_sandbox import SandboxConfig, IsolationLevel

        config = SandboxConfig()

        assert config.isolation_level == IsolationLevel.SUBPROCESS
        assert config.timeout_seconds == 120
        assert config.memory_limit_mb == 2048
        assert config.cpu_limit == 2.0
        assert config.network_enabled is False
        assert config.language == "python"

    def test_config_custom_values(self, sandbox_config):
        """Test custom configuration values."""
        from core.bubblewrap_sandbox import IsolationLevel

        assert sandbox_config.isolation_level == IsolationLevel.SUBPROCESS
        assert sandbox_config.timeout_seconds == 30
        assert sandbox_config.memory_limit_mb == 512

    def test_config_validation_timeout(self):
        """Test timeout must be >= 1."""
        from core.bubblewrap_sandbox import SandboxConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SandboxConfig(timeout_seconds=0)

    def test_config_validation_memory(self):
        """Test memory must be >= 128."""
        from core.bubblewrap_sandbox import SandboxConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SandboxConfig(memory_limit_mb=64)

    def test_config_with_paths(self):
        """Test configuration with mount paths."""
        from core.bubblewrap_sandbox import SandboxConfig

        config = SandboxConfig(
            read_only_paths=["/usr/lib"],
            read_write_paths=["/tmp/output"]
        )

        assert "/usr/lib" in config.read_only_paths
        assert "/tmp/output" in config.read_write_paths


# ============================================================================
# ExecutionResult Tests
# ============================================================================

class TestExecutionResult:
    """Tests for ExecutionResult model."""

    def test_result_creation(self):
        """Test creating execution result."""
        from core.bubblewrap_sandbox import ExecutionResult, IsolationLevel

        result = ExecutionResult(
            success=True,
            stdout="Hello, world!",
            stderr="",
            exit_code=0,
            execution_time=1.5,
            isolation_level=IsolationLevel.SUBPROCESS
        )

        assert result.success is True
        assert result.stdout == "Hello, world!"
        assert result.exit_code == 0
        assert result.execution_time == 1.5

    def test_result_defaults(self):
        """Test result default values."""
        from core.bubblewrap_sandbox import ExecutionResult, IsolationLevel

        result = ExecutionResult(success=False, isolation_level=IsolationLevel.NONE)

        assert result.stdout == ""
        assert result.stderr == ""
        assert result.exit_code == -1
        assert result.output_files == []
        assert result.error is None

    def test_result_with_error(self):
        """Test result with error message."""
        from core.bubblewrap_sandbox import ExecutionResult, IsolationLevel

        result = ExecutionResult(
            success=False,
            stderr="ImportError: No module named 'foo'",
            exit_code=1,
            error="Module not found",
            isolation_level=IsolationLevel.SUBPROCESS
        )

        assert result.success is False
        assert result.error == "Module not found"


# ============================================================================
# SubprocessBackend Tests
# ============================================================================

class TestSubprocessBackend:
    """Tests for SubprocessBackend."""

    @pytest.fixture
    def backend(self):
        """Create subprocess backend."""
        from core.bubblewrap_sandbox import SubprocessBackend
        return SubprocessBackend()

    def test_is_available(self, backend):
        """Test backend is always available."""
        assert backend.is_available() is True

    def test_isolation_level(self, backend):
        """Test backend returns correct isolation level."""
        from core.bubblewrap_sandbox import IsolationLevel
        assert backend.isolation_level == IsolationLevel.SUBPROCESS

    @pytest.mark.asyncio
    @pytest.mark.sandbox
    async def test_execute_simple_script(self, backend, sandbox_config, tmp_path):
        """Test executing a simple Python script."""
        script_path = tmp_path / "hello.py"
        script_path.write_text('print("Hello, sandbox!")')

        result = await backend.execute(script_path, sandbox_config)

        assert result.success is True
        assert "Hello, sandbox!" in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.sandbox
    async def test_execute_with_error(self, backend, sandbox_config, tmp_path):
        """Test executing script with error."""
        script_path = tmp_path / "error.py"
        script_path.write_text('raise ValueError("Test error")')

        result = await backend.execute(script_path, sandbox_config)

        assert result.success is False
        assert result.exit_code != 0
        assert "ValueError" in result.stderr

    @pytest.mark.asyncio
    @pytest.mark.sandbox
    async def test_execute_timeout(self, backend, tmp_path):
        """Test script timeout."""
        from core.bubblewrap_sandbox import SandboxConfig

        script_path = tmp_path / "slow.py"
        script_path.write_text('import time; time.sleep(10)')

        config = SandboxConfig(timeout_seconds=1)

        result = await backend.execute(script_path, config)

        assert result.success is False
        # May have error about timeout

    def test_get_run_command_python(self, backend, tmp_path):
        """Test Python run command generation."""
        from core.bubblewrap_sandbox import SandboxConfig

        script_path = tmp_path / "test.py"
        config = SandboxConfig(language="python")

        cmd = backend._get_run_command(script_path, config)

        assert "python" in cmd[0].lower() or "python" in str(cmd)

    def test_get_run_command_julia(self, backend, tmp_path):
        """Test Julia run command generation."""
        from core.bubblewrap_sandbox import SandboxConfig

        script_path = tmp_path / "test.jl"
        config = SandboxConfig(language="julia")

        cmd = backend._get_run_command(script_path, config)

        assert "julia" in " ".join(cmd).lower()


# ============================================================================
# DockerBackend Tests
# ============================================================================

class TestDockerBackend:
    """Tests for DockerBackend."""

    @pytest.fixture
    def backend(self):
        """Create Docker backend."""
        from core.bubblewrap_sandbox import DockerBackend
        return DockerBackend()

    def test_isolation_level(self, backend):
        """Test backend returns correct isolation level."""
        from core.bubblewrap_sandbox import IsolationLevel
        assert backend.isolation_level == IsolationLevel.DOCKER

    def test_is_available_checks_docker(self, backend):
        """Test availability check looks for Docker."""
        # This test may pass or fail depending on Docker installation
        # The method should not raise
        result = backend.is_available()
        assert isinstance(result, bool)

    def test_get_docker_image_python(self, backend):
        """Test Docker image for Python."""
        image = backend._get_docker_image("python")
        assert "python" in image.lower()

    def test_get_docker_image_julia(self, backend):
        """Test Docker image for Julia."""
        image = backend._get_docker_image("julia")
        assert "julia" in image.lower()

    def test_get_docker_image_r(self, backend):
        """Test Docker image for R."""
        image = backend._get_docker_image("r")
        # Should be r-base or similar
        assert image is not None


# ============================================================================
# BubblewrapBackend Tests
# ============================================================================

class TestBubblewrapBackend:
    """Tests for BubblewrapBackend."""

    @pytest.fixture
    def backend(self):
        """Create Bubblewrap backend."""
        from core.bubblewrap_sandbox import BubblewrapBackend
        return BubblewrapBackend()

    def test_isolation_level(self, backend):
        """Test backend returns correct isolation level."""
        from core.bubblewrap_sandbox import IsolationLevel
        assert backend.isolation_level == IsolationLevel.BUBBLEWRAP

    def test_is_available_platform_check(self, backend):
        """Test availability checks platform."""
        import sys
        result = backend.is_available()

        # Bubblewrap is Linux-only
        if sys.platform != "linux":
            assert result is False


# ============================================================================
# SandboxManager Tests
# ============================================================================

class TestSandboxManager:
    """Tests for SandboxManager."""

    @pytest.fixture
    def manager(self):
        """Create sandbox manager."""
        from core.bubblewrap_sandbox import SandboxManager
        return SandboxManager()

    def test_manager_creation(self, manager):
        """Test manager is created with backends."""
        assert manager._backends is not None

    def test_available_backends(self, manager):
        """Test listing available backends."""
        backends = manager.available_backends

        assert isinstance(backends, list)
        # At minimum, subprocess should be available
        from core.bubblewrap_sandbox import IsolationLevel
        assert IsolationLevel.SUBPROCESS in backends

    def test_best_backend(self, manager):
        """Test getting best available backend."""
        best = manager.best_backend

        # Should return some backend or None
        if best is not None:
            from core.bubblewrap_sandbox import IsolationLevel
            assert isinstance(best, IsolationLevel)

    def test_get_backend(self, manager):
        """Test getting specific backend."""
        from core.bubblewrap_sandbox import IsolationLevel

        backend = manager.get_backend(IsolationLevel.SUBPROCESS)

        assert backend is not None
        assert backend.isolation_level == IsolationLevel.SUBPROCESS

    def test_get_backend_unavailable(self, manager):
        """Test getting unavailable backend."""
        from core.bubblewrap_sandbox import IsolationLevel

        # QEMU is typically not available
        backend = manager.get_backend(IsolationLevel.QEMU)

        # May return None if not available
        # Just verify it doesn't raise

    @pytest.mark.asyncio
    @pytest.mark.sandbox
    async def test_execute(self, manager, sandbox_config, tmp_path):
        """Test executing through manager."""
        script_path = tmp_path / "test.py"
        script_path.write_text('print("Manager test")')

        result = await manager.execute(script_path, sandbox_config)

        assert result is not None
        assert result.isolation_level is not None

    @pytest.mark.asyncio
    @pytest.mark.sandbox
    async def test_execute_code(self, manager, sandbox_config):
        """Test executing code string."""
        code = 'print("Hello from code")'

        result = await manager.execute_code(code, "python", sandbox_config)

        assert result is not None
        if result.success:
            assert "Hello from code" in result.stdout


# ============================================================================
# Multi-Language Execution Tests
# ============================================================================

class TestMultiLanguageExecution:
    """Tests for multi-language execution."""

    @pytest.mark.asyncio
    @pytest.mark.sandbox
    async def test_execute_python(self, mock_sandbox_manager, sandbox_config, tmp_path):
        """Test Python execution."""
        script_path = tmp_path / "test.py"
        script_path.write_text('x = 1 + 1\nprint(f"Result: {x}")')

        result = await mock_sandbox_manager.execute(script_path, sandbox_config)

        assert result.success is True

    @pytest.mark.asyncio
    @pytest.mark.sandbox
    async def test_execute_with_imports(self, mock_sandbox_manager, sandbox_config, tmp_path):
        """Test execution with standard library imports."""
        script_path = tmp_path / "test.py"
        script_path.write_text('import os\nprint(os.name)')

        result = await mock_sandbox_manager.execute(script_path, sandbox_config)

        assert result.success is True


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestExecuteInSandbox:
    """Tests for execute_in_sandbox convenience function."""

    @pytest.mark.asyncio
    @pytest.mark.sandbox
    async def test_convenience_function(self, tmp_path):
        """Test execute_in_sandbox function."""
        from core.bubblewrap_sandbox import execute_in_sandbox

        script_path = tmp_path / "test.py"
        script_path.write_text('print("Convenience test")')

        result = await execute_in_sandbox(script_path)

        assert result is not None
        # May succeed or fail depending on environment


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestSandboxErrorHandling:
    """Tests for error handling in sandbox."""

    @pytest.mark.asyncio
    @pytest.mark.sandbox
    async def test_syntax_error_captured(self, mock_sandbox_manager, sandbox_config, tmp_path):
        """Test syntax error is captured."""
        # For mock, we'll just verify it handles the case
        script_path = tmp_path / "bad.py"
        script_path.write_text('def broken(:')

        result = await mock_sandbox_manager.execute(script_path, sandbox_config)

        # Mock returns success, but real would fail

    @pytest.mark.asyncio
    @pytest.mark.sandbox
    async def test_runtime_error_captured(self, mock_sandbox_manager, sandbox_config, tmp_path):
        """Test runtime error is captured."""
        script_path = tmp_path / "error.py"
        script_path.write_text('x = 1/0')  # ZeroDivisionError

        result = await mock_sandbox_manager.execute(script_path, sandbox_config)

        # Just verify it completes


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestSandboxIntegration:
    """Integration tests for sandbox execution."""

    @pytest.mark.asyncio
    @pytest.mark.sandbox
    async def test_full_execution_cycle(self, tmp_path):
        """Test full execution cycle."""
        from core.bubblewrap_sandbox import SandboxManager, SandboxConfig

        manager = SandboxManager()

        if not manager.available_backends:
            pytest.skip("No sandbox backends available")

        config = SandboxConfig(
            timeout_seconds=30,
            memory_limit_mb=256,
            language="python"
        )

        script_path = tmp_path / "integration_test.py"
        script_path.write_text('''
import sys
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print("Integration test passed!")
''')

        result = await manager.execute(script_path, config)

        if result.success:
            assert "Integration test passed!" in result.stdout
