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
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from .error_handling import logger, LogCategory
from .qemu_backend import (
    QEMUBackendImpl,
    QEMUVMConfig,
    QEMUImageManager,
    QEMUExecutionResult,
    ExecutionMode,
    create_qemu_backend,
)


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

    def _create_preexec_fn(self, config: SandboxConfig):
        """Create a preexec function to set resource limits on Unix."""
        def set_limits():
            if sys.platform != "win32":
                import resource
                # Memory limit (soft, hard) in bytes
                memory_bytes = config.memory_limit_mb * 1024 * 1024
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
                except (ValueError, resource.error):
                    pass  # May fail if limit is too low

                # CPU time limit in seconds
                cpu_seconds = int(config.timeout_seconds * config.cpu_limit)
                try:
                    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
                except (ValueError, resource.error):
                    pass

        return set_limits if sys.platform != "win32" else None

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

        # Create preexec function for resource limits (Unix only)
        preexec_fn = self._create_preexec_fn(config)

        try:
            process = await asyncio.create_subprocess_exec(
                *run_cmd, str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=config.working_dir if Path(config.working_dir).exists() else None,
                env=env,
                preexec_fn=preexec_fn
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

        # Wrap with systemd-run for resource limits if available
        if self._has_systemd_run():
            cmd = [
                "systemd-run",
                "--scope",
                "--user",
                f"--property=MemoryMax={config.memory_limit_mb}M",
                f"--property=CPUQuota={int(config.cpu_limit * 100)}%",
                "--quiet",
                "--"
            ] + cmd

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

    def _has_systemd_run(self) -> bool:
        """Check if systemd-run is available for resource limiting."""
        try:
            result = subprocess.run(
                ["systemd-run", "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False


class QEMUBackend(SandboxBackend):
    """
    QEMU full system virtualization backend (strongest isolation).

    Provides complete system-level isolation by running code inside
    fully virtualized QEMU instances. This is the most secure option,
    offering hardware-level isolation from the host system.

    Features:
    - Full system virtualization with hardware isolation
    - Support for KVM (Linux), WHPX (Windows), HVF (macOS) acceleration
    - Automatic fallback to TCG software emulation
    - VM snapshot support for fast reset between executions
    - VirtFS (9p) for efficient host-guest file sharing
    - QMP (QEMU Machine Protocol) for VM control
    - Optional VM pooling for reduced startup latency
    """

    def __init__(
        self,
        use_pool: bool = False,
        pool_size: int = 2,
        images_dir: Optional[Path] = None
    ):
        """
        Initialize QEMU backend.

        Args:
            use_pool: Enable VM pooling for faster execution (pre-warms VMs)
            pool_size: Number of VMs to keep in the pool
            images_dir: Directory for VM images (default: ~/.scientific-agent/qemu-images)
        """
        self._available: Optional[bool] = None
        self._backend: Optional[QEMUBackendImpl] = None
        self._use_pool = use_pool
        self._pool_size = pool_size
        self._images_dir = images_dir
        self._initialized = False

    def is_available(self) -> bool:
        """Check if QEMU is available on this system."""
        if self._available is None:
            # Check for qemu-system-x86_64
            candidates = [
                "qemu-system-x86_64",
                "/usr/bin/qemu-system-x86_64",
                "/usr/local/bin/qemu-system-x86_64",
                "C:\\Program Files\\qemu\\qemu-system-x86_64.exe",
                "C:\\Program Files (x86)\\qemu\\qemu-system-x86_64.exe",
            ]

            for candidate in candidates:
                try:
                    result = subprocess.run(
                        [candidate, "--version"],
                        capture_output=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        self._available = True
                        logger.debug(
                            f"QEMU found: {candidate}",
                            category=LogCategory.SYSTEM
                        )
                        break
                except Exception:
                    continue
            else:
                self._available = False

        return self._available

    @property
    def isolation_level(self) -> IsolationLevel:
        """Return the isolation level (QEMU = strongest)."""
        return IsolationLevel.QEMU

    async def _ensure_initialized(self):
        """Ensure the backend is initialized."""
        if not self._initialized:
            self._backend = create_qemu_backend(
                use_pool=self._use_pool,
                pool_size=self._pool_size,
                images_dir=self._images_dir
            )
            self._initialized = True

    async def execute(
        self,
        script_path: Path,
        config: SandboxConfig
    ) -> ExecutionResult:
        """
        Execute a script in a QEMU virtual machine.

        Args:
            script_path: Path to the script to execute
            config: Sandbox configuration

        Returns:
            ExecutionResult with output, exit code, and execution metadata
        """
        result = ExecutionResult(success=False, isolation_level=self.isolation_level)
        start_time = time.time()

        # Check availability
        if not self.is_available():
            result.error = "QEMU not available on this system"
            return result

        # Ensure backend is initialized
        await self._ensure_initialized()

        try:
            # Build QEMU VM configuration from sandbox config
            vm_config = QEMUVMConfig(
                name=f"sandbox_{script_path.stem}_{int(time.time())}",
                memory=config.qemu_memory,
                cpus=config.qemu_cpus,
                disk_image=config.qemu_image,
                network_enabled=config.network_enabled,
                timeout_seconds=config.timeout_seconds,
                execution_mode=ExecutionMode.VIRTFS,
                use_snapshot=True,  # Always use snapshots for safety
            )

            # If no image specified, try to find a base image for the language
            if not vm_config.disk_image:
                base_image = self._backend.get_base_image_path(config.language)
                if base_image:
                    vm_config.disk_image = str(base_image)
                else:
                    result.error = (
                        f"No QEMU base image available for {config.language}. "
                        f"Please create one at: {self._backend.image_manager.images_dir}"
                    )
                    result.execution_time = time.time() - start_time
                    return result

            # Execute in QEMU VM
            qemu_result: QEMUExecutionResult = await self._backend.execute(
                script_path=script_path,
                config=vm_config,
                language=config.language
            )

            # Convert QEMU result to standard ExecutionResult
            result.success = qemu_result.success
            result.stdout = qemu_result.stdout
            result.stderr = qemu_result.stderr
            result.exit_code = qemu_result.exit_code
            result.execution_time = qemu_result.execution_time
            result.output_files = qemu_result.output_files
            result.error = qemu_result.error

            logger.info(
                f"QEMU execution completed",
                category=LogCategory.EXECUTION,
                context={
                    "success": result.success,
                    "exit_code": result.exit_code,
                    "execution_time": f"{result.execution_time:.2f}s",
                    "vm_boot_time": f"{qemu_result.vm_boot_time:.2f}s" if qemu_result.vm_boot_time else None
                }
            )

        except Exception as e:
            result.error = f"QEMU execution failed: {str(e)}"
            result.stderr = str(e)
            logger.error(
                f"QEMU execution error: {e}",
                category=LogCategory.EXECUTION
            )

        result.execution_time = time.time() - start_time
        return result

    async def shutdown(self):
        """Shutdown the QEMU backend and clean up resources."""
        if self._backend:
            await self._backend.shutdown()
            self._initialized = False
            logger.info("QEMU backend shutdown complete", category=LogCategory.SYSTEM)

    def _kvm_available(self) -> bool:
        """Check if KVM acceleration is available (Linux only)."""
        return Path("/dev/kvm").exists()

    def _get_accelerator_info(self) -> Dict[str, Any]:
        """Get information about available hardware acceleration."""
        import platform
        system = platform.system().lower()

        info = {
            "platform": system,
            "kvm": False,
            "whpx": False,
            "hvf": False,
            "tcg": True,  # Always available
        }

        if system == "linux":
            info["kvm"] = self._kvm_available()
        elif system == "windows":
            # Check WHPX via QEMU
            try:
                result = subprocess.run(
                    ["qemu-system-x86_64", "-accel", "help"],
                    capture_output=True,
                    timeout=5
                )
                info["whpx"] = b"whpx" in result.stdout.lower()
            except Exception:
                pass
        elif system == "darwin":
            # Check HVF
            try:
                result = subprocess.run(
                    ["qemu-system-x86_64", "-accel", "help"],
                    capture_output=True,
                    timeout=5
                )
                info["hvf"] = b"hvf" in result.stdout.lower()
            except Exception:
                pass

        return info

    @classmethod
    def create_base_image(
        cls,
        language: str,
        output_path: Optional[Path] = None,
        size: str = "10G"
    ) -> Path:
        """
        Create a base VM image for a specific language.

        This is a helper method to create base images that can be used
        for sandbox execution. The image should be set up with:
        - Minimal Linux OS (e.g., Alpine, Debian minimal)
        - Required language runtime (Python, Julia, R, etc.)
        - VirtFS (9p) support for file sharing
        - Optional: QEMU guest agent for enhanced control

        Args:
            language: Programming language (python, julia, r, matlab)
            output_path: Output path for the image (optional)
            size: Disk size (default: 10G)

        Returns:
            Path to the created image
        """
        image_manager = QEMUImageManager()

        image_name = f"{language}-sandbox"
        if output_path:
            # Create at specified path
            image_path = output_path
        else:
            image_path = image_manager.create_base_image(image_name, size)

        logger.info(
            f"Created base image template at: {image_path}",
            category=LogCategory.SYSTEM,
            context={
                "language": language,
                "size": size,
                "note": "Image needs OS installation and language runtime setup"
            }
        )

        return image_path


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
