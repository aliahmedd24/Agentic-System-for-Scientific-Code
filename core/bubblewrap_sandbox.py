"""
Bubblewrap Sandbox - Secure code execution with isolation.

Provides multiple isolation backends:
1. Bubblewrap (bwrap) - Linux namespace isolation
2. QEMU - Full system virtualization
3. Docker - Container isolation (fallback)
4. Subprocess - Minimal isolation (development only)

Implements KPI 7: Security isolation for code validation.
"""

import os
import sys
import json
import asyncio
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from .error_handling import logger, LogCategory


class IsolationLevel(Enum):
    """Levels of sandbox isolation."""
    NONE = "none"              # No isolation (development only)
    SUBPROCESS = "subprocess"   # Basic process isolation
    DOCKER = "docker"          # Container isolation
    BUBBLEWRAP = "bubblewrap"  # Linux namespace isolation
    QEMU = "qemu"              # Full system virtualization


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""
    isolation_level: IsolationLevel = IsolationLevel.SUBPROCESS
    timeout_seconds: int = 120
    memory_limit_mb: int = 2048
    cpu_limit: float = 2.0
    network_enabled: bool = False
    read_only_paths: List[str] = field(default_factory=list)
    read_write_paths: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    working_dir: str = "/workspace"

    # Language-specific settings
    language: str = "python"
    python_path: Optional[str] = None

    # QEMU-specific settings
    qemu_image: Optional[str] = None
    qemu_memory: str = "2G"
    qemu_cpus: int = 2


@dataclass
class ExecutionResult:
    """Result of sandboxed execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    execution_time: float = 0.0
    output_files: List[str] = field(default_factory=list)
    error: Optional[str] = None
    isolation_level: IsolationLevel = IsolationLevel.NONE


class SandboxBackend(ABC):
    """Abstract base class for sandbox backends."""

    @abstractmethod
    async def execute(
        self,
        script_path: Path,
        config: SandboxConfig
    ) -> ExecutionResult:
        """Execute a script in the sandbox."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass

    @property
    @abstractmethod
    def isolation_level(self) -> IsolationLevel:
        """Return the isolation level of this backend."""
        pass


class SubprocessBackend(SandboxBackend):
    """Basic subprocess execution with minimal isolation."""

    def is_available(self) -> bool:
        return True

    @property
    def isolation_level(self) -> IsolationLevel:
        return IsolationLevel.SUBPROCESS

    async def execute(
        self,
        script_path: Path,
        config: SandboxConfig
    ) -> ExecutionResult:
        result = ExecutionResult(success=False, isolation_level=self.isolation_level)

        import time
        start_time = time.time()

        # Get run command for language
        run_cmd = self._get_run_command(config.language)

        # Build environment
        env = os.environ.copy()
        env.update(config.env_vars)
        if config.python_path:
            env["PYTHONPATH"] = config.python_path

        try:
            process = await asyncio.create_subprocess_exec(
                *run_cmd, str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=config.working_dir if Path(config.working_dir).exists() else None,
                env=env
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout_seconds
            )

            result.stdout = stdout.decode('utf-8', errors='ignore')
            result.stderr = stderr.decode('utf-8', errors='ignore')
            result.exit_code = process.returncode
            result.success = process.returncode == 0

        except asyncio.TimeoutError:
            result.error = f"Execution timed out ({config.timeout_seconds}s)"
            result.stderr = "Timeout"
        except Exception as e:
            result.error = str(e)
            result.stderr = str(e)

        result.execution_time = time.time() - start_time
        return result

    def _get_run_command(self, language: str) -> List[str]:
        """Get the run command for a language."""
        commands = {
            "python": [sys.executable],
            "julia": ["julia"],
            "r": ["Rscript"],
            "matlab": ["octave", "--no-gui"]
        }
        return commands.get(language, [sys.executable])


class DockerBackend(SandboxBackend):
    """Docker container isolation."""

    def __init__(self):
        self._available = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                result = subprocess.run(
                    ["docker", "info"],
                    capture_output=True,
                    timeout=5
                )
                self._available = result.returncode == 0
            except Exception:
                self._available = False
        return self._available

    @property
    def isolation_level(self) -> IsolationLevel:
        return IsolationLevel.DOCKER

    async def execute(
        self,
        script_path: Path,
        config: SandboxConfig
    ) -> ExecutionResult:
        result = ExecutionResult(success=False, isolation_level=self.isolation_level)

        if not self.is_available():
            result.error = "Docker not available"
            return result

        import time
        start_time = time.time()

        # Get Docker image for language
        image = self._get_docker_image(config.language)
        run_cmd = self._get_run_command(config.language)

        # Build docker command
        cmd = [
            "docker", "run",
            "--rm",
            "--cpus", str(config.cpu_limit),
            "--memory", f"{config.memory_limit_mb}m",
            "-v", f"{script_path.parent}:/scripts:ro",
            "-w", config.working_dir
        ]

        # Add network restrictions
        if not config.network_enabled:
            cmd.extend(["--network", "none"])

        # Add read-only mounts
        for path in config.read_only_paths:
            if Path(path).exists():
                cmd.extend(["-v", f"{path}:{path}:ro"])

        # Add read-write mounts
        for path in config.read_write_paths:
            if Path(path).exists():
                cmd.extend(["-v", f"{path}:{path}"])

        # Add environment variables
        for key, value in config.env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Add image and command
        cmd.append(image)
        cmd.extend(run_cmd)
        cmd.append(f"/scripts/{script_path.name}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout_seconds
            )

            result.stdout = stdout.decode('utf-8', errors='ignore')
            result.stderr = stderr.decode('utf-8', errors='ignore')
            result.exit_code = process.returncode
            result.success = process.returncode == 0

        except asyncio.TimeoutError:
            result.error = f"Execution timed out ({config.timeout_seconds}s)"
            # Kill the container
            subprocess.run(["docker", "kill", f"sandbox_{script_path.stem}"], capture_output=True)
        except Exception as e:
            result.error = str(e)

        result.execution_time = time.time() - start_time
        return result

    def _get_docker_image(self, language: str) -> str:
        """Get Docker image for language."""
        images = {
            "python": "python:3.11-slim",
            "julia": "julia:1.10",
            "r": "r-base:4.3.0",
            "matlab": "gnuoctave/octave:8.4.0"
        }
        return images.get(language, "python:3.11-slim")

    def _get_run_command(self, language: str) -> List[str]:
        """Get run command for language."""
        commands = {
            "python": ["python"],
            "julia": ["julia"],
            "r": ["Rscript"],
            "matlab": ["octave", "--no-gui"]
        }
        return commands.get(language, ["python"])


class BubblewrapBackend(SandboxBackend):
    """Bubblewrap (bwrap) namespace isolation for Linux."""

    def __init__(self):
        self._available = None

    def is_available(self) -> bool:
        if self._available is None:
            # Bubblewrap only works on Linux
            if sys.platform != "linux":
                self._available = False
            else:
                try:
                    result = subprocess.run(
                        ["bwrap", "--version"],
                        capture_output=True,
                        timeout=5
                    )
                    self._available = result.returncode == 0
                except Exception:
                    self._available = False
        return self._available

    @property
    def isolation_level(self) -> IsolationLevel:
        return IsolationLevel.BUBBLEWRAP

    async def execute(
        self,
        script_path: Path,
        config: SandboxConfig
    ) -> ExecutionResult:
        result = ExecutionResult(success=False, isolation_level=self.isolation_level)

        if not self.is_available():
            result.error = "Bubblewrap not available"
            return result

        import time
        start_time = time.time()

        # Build bwrap command
        cmd = [
            "bwrap",
            # Basic isolation
            "--unshare-pid",
            "--unshare-uts",
            "--unshare-ipc",
            # No network if disabled
            *(["--unshare-net"] if not config.network_enabled else []),
            # Read-only root
            "--ro-bind", "/", "/",
            # Writable tmpfs for working directory
            "--tmpfs", "/tmp",
            "--tmpfs", "/workspace",
            # Working directory
            "--chdir", config.working_dir,
            # Proc filesystem
            "--proc", "/proc",
            # Dev filesystem (minimal)
            "--dev", "/dev",
        ]

        # Mount script directory read-only
        cmd.extend(["--ro-bind", str(script_path.parent), "/scripts"])

        # Add read-only paths
        for path in config.read_only_paths:
            if Path(path).exists():
                cmd.extend(["--ro-bind", path, path])

        # Add read-write paths
        for path in config.read_write_paths:
            if Path(path).exists():
                cmd.extend(["--bind", path, path])

        # Add environment variables
        for key, value in config.env_vars.items():
            cmd.extend(["--setenv", key, value])

        # Add Python path
        if config.python_path:
            cmd.extend(["--setenv", "PYTHONPATH", config.python_path])

        # Add the actual command
        run_cmd = self._get_run_command(config.language)
        cmd.extend(run_cmd)
        cmd.append(f"/scripts/{script_path.name}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout_seconds
            )

            result.stdout = stdout.decode('utf-8', errors='ignore')
            result.stderr = stderr.decode('utf-8', errors='ignore')
            result.exit_code = process.returncode
            result.success = process.returncode == 0

        except asyncio.TimeoutError:
            result.error = f"Execution timed out ({config.timeout_seconds}s)"
        except Exception as e:
            result.error = str(e)

        result.execution_time = time.time() - start_time
        return result

    def _get_run_command(self, language: str) -> List[str]:
        """Get run command for language."""
        commands = {
            "python": [sys.executable],
            "julia": ["julia"],
            "r": ["Rscript"],
            "matlab": ["octave", "--no-gui"]
        }
        return commands.get(language, [sys.executable])


class QEMUBackend(SandboxBackend):
    """QEMU full system virtualization (strongest isolation)."""

    def __init__(self):
        self._available = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                result = subprocess.run(
                    ["qemu-system-x86_64", "--version"],
                    capture_output=True,
                    timeout=5
                )
                self._available = result.returncode == 0
            except Exception:
                self._available = False
        return self._available

    @property
    def isolation_level(self) -> IsolationLevel:
        return IsolationLevel.QEMU

    async def execute(
        self,
        script_path: Path,
        config: SandboxConfig
    ) -> ExecutionResult:
        result = ExecutionResult(success=False, isolation_level=self.isolation_level)

        if not self.is_available():
            result.error = "QEMU not available"
            return result

        if not config.qemu_image:
            result.error = "QEMU image not specified"
            return result

        import time
        start_time = time.time()

        # Create shared directory for script
        shared_dir = tempfile.mkdtemp(prefix="qemu_share_")
        script_dest = Path(shared_dir) / script_path.name
        shutil.copy(script_path, script_dest)

        # Build QEMU command
        cmd = [
            "qemu-system-x86_64",
            "-m", config.qemu_memory,
            "-smp", str(config.qemu_cpus),
            "-drive", f"file={config.qemu_image},format=qcow2",
            "-virtfs", f"local,path={shared_dir},mount_tag=share,security_model=none",
            "-nographic",
            "-enable-kvm" if self._kvm_available() else "-machine", "accel=tcg",
        ]

        # Disable network if needed
        if not config.network_enabled:
            cmd.extend(["-net", "none"])
        else:
            cmd.extend(["-net", "user", "-net", "nic"])

        result.error = "QEMU execution requires VM image setup (not implemented)"
        result.execution_time = time.time() - start_time

        # Clean up
        try:
            shutil.rmtree(shared_dir)
        except Exception:
            pass

        return result

    def _kvm_available(self) -> bool:
        """Check if KVM acceleration is available."""
        return Path("/dev/kvm").exists()


class SandboxManager:
    """
    Manager for sandbox execution with automatic backend selection.

    Provides a unified interface for secure code execution across
    multiple isolation backends.
    """

    def __init__(self, preferred_level: IsolationLevel = None):
        """
        Initialize the sandbox manager.

        Args:
            preferred_level: Preferred isolation level. If not available,
                           will fall back to available backends.
        """
        self._backends: Dict[IsolationLevel, SandboxBackend] = {}
        self._preferred_level = preferred_level
        self._init_backends()

        logger.info(
            "SandboxManager initialized",
            category=LogCategory.SYSTEM,
            available_backends=list(self._backends.keys())
        )

    def _init_backends(self):
        """Initialize available backends."""
        # Try each backend in order of isolation strength
        backends = [
            QEMUBackend(),
            BubblewrapBackend(),
            DockerBackend(),
            SubprocessBackend(),
        ]

        for backend in backends:
            if backend.is_available():
                self._backends[backend.isolation_level] = backend

    @property
    def available_backends(self) -> List[IsolationLevel]:
        """Get list of available isolation levels."""
        return list(self._backends.keys())

    @property
    def best_backend(self) -> Optional[IsolationLevel]:
        """Get the strongest available isolation level."""
        # Priority order
        priority = [
            IsolationLevel.QEMU,
            IsolationLevel.BUBBLEWRAP,
            IsolationLevel.DOCKER,
            IsolationLevel.SUBPROCESS,
        ]

        for level in priority:
            if level in self._backends:
                return level
        return None

    def get_backend(
        self,
        level: IsolationLevel = None
    ) -> Optional[SandboxBackend]:
        """
        Get a backend at the specified isolation level.

        Falls back to available backends if requested level is unavailable.
        """
        if level and level in self._backends:
            return self._backends[level]

        if self._preferred_level and self._preferred_level in self._backends:
            return self._backends[self._preferred_level]

        if self.best_backend:
            return self._backends[self.best_backend]

        return None

    async def execute(
        self,
        script_path: Path,
        config: SandboxConfig = None,
        isolation_level: IsolationLevel = None
    ) -> ExecutionResult:
        """
        Execute a script in a sandbox.

        Args:
            script_path: Path to the script to execute
            config: Sandbox configuration (optional)
            isolation_level: Specific isolation level to use (optional)

        Returns:
            ExecutionResult with output and status
        """
        config = config or SandboxConfig()

        # Get backend
        backend = self.get_backend(isolation_level)

        if not backend:
            logger.warning(
                "No sandbox backend available, using subprocess",
                category=LogCategory.SYSTEM
            )
            backend = SubprocessBackend()

        logger.info(
            f"Executing script in {backend.isolation_level.value} sandbox",
            category=LogCategory.SYSTEM,
            script=str(script_path)
        )

        result = await backend.execute(script_path, config)

        logger.info(
            f"Sandbox execution completed",
            category=LogCategory.SYSTEM,
            success=result.success,
            execution_time=result.execution_time,
            isolation_level=result.isolation_level.value
        )

        return result

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        config: SandboxConfig = None,
        isolation_level: IsolationLevel = None
    ) -> ExecutionResult:
        """
        Execute code string in a sandbox.

        Args:
            code: Code to execute
            language: Programming language
            config: Sandbox configuration (optional)
            isolation_level: Specific isolation level (optional)

        Returns:
            ExecutionResult with output and status
        """
        config = config or SandboxConfig()
        config.language = language

        # Get file extension
        extensions = {
            "python": ".py",
            "julia": ".jl",
            "r": ".R",
            "matlab": ".m"
        }
        ext = extensions.get(language, ".py")

        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=ext,
            delete=False
        ) as f:
            f.write(code)
            script_path = Path(f.name)

        try:
            result = await self.execute(script_path, config, isolation_level)
        finally:
            # Clean up
            try:
                script_path.unlink()
            except Exception:
                pass

        return result


# Global sandbox manager instance
_sandbox_manager: Optional[SandboxManager] = None


def get_sandbox_manager(
    preferred_level: IsolationLevel = None
) -> SandboxManager:
    """Get or create the global sandbox manager."""
    global _sandbox_manager

    if _sandbox_manager is None:
        _sandbox_manager = SandboxManager(preferred_level)

    return _sandbox_manager


async def execute_in_sandbox(
    code: str,
    language: str = "python",
    timeout: int = 120,
    memory_mb: int = 2048,
    network: bool = False
) -> ExecutionResult:
    """
    Convenience function to execute code in a sandbox.

    Args:
        code: Code to execute
        language: Programming language
        timeout: Timeout in seconds
        memory_mb: Memory limit in MB
        network: Enable network access

    Returns:
        ExecutionResult with output and status
    """
    manager = get_sandbox_manager()

    config = SandboxConfig(
        timeout_seconds=timeout,
        memory_limit_mb=memory_mb,
        network_enabled=network,
        language=language
    )

    return await manager.execute_code(code, language, config)


__all__ = [
    'IsolationLevel',
    'SandboxConfig',
    'ExecutionResult',
    'SandboxBackend',
    'SubprocessBackend',
    'DockerBackend',
    'BubblewrapBackend',
    'QEMUBackend',
    'SandboxManager',
    'get_sandbox_manager',
    'execute_in_sandbox',
]
