"""
QEMU System Virtualization Backend - Full system VM isolation for secure code execution.

This module provides the strongest isolation level by running code inside
fully virtualized QEMU instances. Features include:

1. Full system virtualization with hardware-level isolation
2. VM lifecycle management (create, start, stop, destroy)
3. Snapshot support for fast VM resets
4. Host-guest file sharing via VirtFS (9p)
5. QMP (QEMU Machine Protocol) for VM control
6. Multiple execution modes (SSH, serial console, cloud-init)
7. Support for KVM (Linux), WHPX (Windows), and TCG (fallback) acceleration
8. Automatic VM image management and caching

Implements KPI 7: Maximum security isolation for code validation.
"""

import os
import sys
import json
import asyncio
import socket
import tempfile
import subprocess
import shutil
import hashlib
import uuid
import platform
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict
import time

from .error_handling import logger, LogCategory, create_error, ErrorCategory


class QEMUAccelerator(Enum):
    """QEMU hardware acceleration options."""
    KVM = "kvm"           # Linux KVM
    WHPX = "whpx"         # Windows Hypervisor Platform
    HVF = "hvf"           # macOS Hypervisor.framework
    HAXM = "haxm"         # Intel HAXM (cross-platform)
    TCG = "tcg"           # Software emulation (slowest, always available)


class VMState(Enum):
    """Virtual machine state."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class ExecutionMode(Enum):
    """How to execute commands in the guest VM."""
    SERIAL = "serial"     # Via serial console (requires guest setup)
    SSH = "ssh"           # Via SSH (requires network and SSH server)
    VIRTFS = "virtfs"     # Via shared filesystem (requires 9p mount)
    CLOUDINIT = "cloudinit"  # Via cloud-init userdata


class QEMUVMConfig(BaseModel):
    """Extended configuration for QEMU virtual machines."""
    model_config = ConfigDict(extra="forbid")

    # Basic VM settings
    name: str = Field("sandbox-vm", description="VM name")
    memory: str = Field("2G", description="Memory allocation")
    cpus: int = Field(2, ge=1, description="Number of CPUs")
    cpu_model: str = Field("host", description="CPU model (or 'qemu64' for TCG)")

    # Disk configuration
    disk_image: Optional[str] = Field(None, description="Disk image path")
    disk_size: str = Field("10G", description="Disk size")
    disk_format: str = Field("qcow2", description="Disk format")
    use_snapshot: bool = Field(True, description="Use snapshots for fast reset")

    # Network configuration
    network_enabled: bool = Field(False, description="Enable networking")
    network_type: str = Field("user", description="Network type (user, tap, bridge)")
    ssh_port: int = Field(0, ge=0, description="SSH port (0 for auto-assign)")

    # Display/console
    display: str = Field("none", description="Display type (none, vnc, sdl)")
    serial_console: bool = Field(True, description="Enable serial console")

    # File sharing
    shared_dir: Optional[str] = Field(None, description="Shared directory path")
    shared_mount_tag: str = Field("hostshare", description="Shared mount tag")

    # Execution
    execution_mode: ExecutionMode = Field(ExecutionMode.VIRTFS, description="Execution mode")
    timeout_seconds: int = Field(120, ge=1, description="Timeout in seconds")

    # Guest OS settings
    guest_user: str = Field("sandbox", description="Guest username")
    guest_password: str = Field("sandbox", description="Guest password")
    guest_workdir: str = Field("/sandbox", description="Guest working directory")

    # Acceleration
    accelerator: Optional[QEMUAccelerator] = Field(None, description="Hardware accelerator")

    # QEMU paths (auto-detected if not specified)
    qemu_binary: Optional[str] = Field(None, description="QEMU binary path")
    qemu_img_binary: Optional[str] = Field(None, description="QEMU-img binary path")


class QEMUExecutionResult(BaseModel):
    """Result of execution within QEMU VM."""
    model_config = ConfigDict(extra="forbid")

    success: bool = Field(..., description="Whether execution succeeded")
    stdout: str = Field("", description="Standard output")
    stderr: str = Field("", description="Standard error")
    exit_code: int = Field(-1, description="Exit code")
    execution_time: float = Field(0.0, ge=0, description="Execution time in seconds")
    vm_boot_time: float = Field(0.0, ge=0, description="VM boot time in seconds")
    output_files: List[str] = Field(default_factory=list, description="Output files")
    error: Optional[str] = Field(None, description="Error message")
    vm_state: VMState = Field(VMState.STOPPED, description="Final VM state")


class QEMUImageManager:
    """
    Manages QEMU disk images including base images and snapshots.

    Supports creating, caching, and managing qcow2 images with
    backing file support for efficient snapshot operations.
    """

    DEFAULT_IMAGES_DIR = Path.home() / ".scientific-agent" / "qemu-images"

    def __init__(self, images_dir: Optional[Path] = None):
        """Initialize the image manager."""
        self.images_dir = Path(images_dir) if images_dir else self.DEFAULT_IMAGES_DIR
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self._qemu_img = self._find_qemu_img()

        logger.info(
            "QEMUImageManager initialized",
            category=LogCategory.SYSTEM,
            context={"images_dir": str(self.images_dir)}
        )

    def _find_qemu_img(self) -> str:
        """Find qemu-img binary."""
        candidates = [
            "qemu-img",
            "/usr/bin/qemu-img",
            "/usr/local/bin/qemu-img",
            "C:\\Program Files\\qemu\\qemu-img.exe",
            "C:\\Program Files (x86)\\qemu\\qemu-img.exe",
        ]

        for candidate in candidates:
            try:
                result = subprocess.run(
                    [candidate, "--version"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return candidate
            except Exception:
                continue

        raise RuntimeError("qemu-img not found. Please install QEMU.")

    def create_base_image(
        self,
        name: str,
        size: str = "10G",
        fmt: str = "qcow2"
    ) -> Path:
        """Create a new base disk image."""
        image_path = self.images_dir / f"{name}.{fmt}"

        if image_path.exists():
            logger.info(f"Base image already exists: {image_path}", category=LogCategory.SYSTEM)
            return image_path

        cmd = [
            self._qemu_img, "create",
            "-f", fmt,
            str(image_path),
            size
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create image: {result.stderr}")

        logger.info(
            f"Created base image: {image_path}",
            category=LogCategory.SYSTEM,
            context={"size": size, "format": fmt}
        )

        return image_path

    def create_snapshot(
        self,
        base_image: Path,
        snapshot_name: Optional[str] = None
    ) -> Path:
        """Create a copy-on-write snapshot from a base image."""
        if not base_image.exists():
            raise FileNotFoundError(f"Base image not found: {base_image}")

        snapshot_name = snapshot_name or f"snapshot_{uuid.uuid4().hex[:8]}"
        snapshot_path = self.images_dir / "snapshots" / f"{snapshot_name}.qcow2"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self._qemu_img, "create",
            "-f", "qcow2",
            "-F", "qcow2",
            "-b", str(base_image.absolute()),
            str(snapshot_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create snapshot: {result.stderr}")

        logger.debug(
            f"Created snapshot: {snapshot_path}",
            category=LogCategory.SYSTEM,
            context={"base": str(base_image)}
        )

        return snapshot_path

    def delete_snapshot(self, snapshot_path: Path) -> bool:
        """Delete a snapshot image."""
        try:
            if snapshot_path.exists():
                snapshot_path.unlink()
                logger.debug(f"Deleted snapshot: {snapshot_path}", category=LogCategory.SYSTEM)
                return True
        except Exception as e:
            logger.warning(f"Failed to delete snapshot: {e}", category=LogCategory.SYSTEM)
        return False

    def get_image_info(self, image_path: Path) -> Dict[str, Any]:
        """Get information about a disk image."""
        cmd = [self._qemu_img, "info", "--output=json", str(image_path)]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to get image info: {result.stderr}")

        return json.loads(result.stdout)

    def list_images(self) -> List[Dict[str, Any]]:
        """List all available images."""
        images = []

        for image_file in self.images_dir.glob("**/*.qcow2"):
            try:
                info = self.get_image_info(image_file)
                info["path"] = str(image_file)
                info["name"] = image_file.stem
                images.append(info)
            except Exception as e:
                logger.warning(f"Failed to get info for {image_file}: {e}", category=LogCategory.SYSTEM)

        return images

    def cleanup_old_snapshots(self, max_age_hours: int = 24) -> int:
        """Clean up snapshots older than max_age_hours."""
        snapshots_dir = self.images_dir / "snapshots"
        if not snapshots_dir.exists():
            return 0

        deleted = 0
        cutoff_time = time.time() - (max_age_hours * 3600)

        for snapshot in snapshots_dir.glob("*.qcow2"):
            if snapshot.stat().st_mtime < cutoff_time:
                if self.delete_snapshot(snapshot):
                    deleted += 1

        if deleted > 0:
            logger.info(
                f"Cleaned up {deleted} old snapshots",
                category=LogCategory.SYSTEM
            )

        return deleted


class QEMUMonitor:
    """
    QMP (QEMU Machine Protocol) client for VM control.

    Provides programmatic control over QEMU instances including:
    - Power management (start, stop, reset)
    - Snapshot operations
    - Guest agent communication
    - Status queries
    """

    def __init__(self, socket_path: str):
        """Initialize QMP client."""
        self.socket_path = socket_path
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False

    async def connect(self, timeout: float = 30.0) -> bool:
        """Connect to QMP socket."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                if sys.platform == "win32":
                    # Windows: use TCP sockets (Unix sockets not supported)
                    if self.socket_path.startswith("tcp:"):
                        # Explicit TCP format: tcp:host:port
                        host, port = self.socket_path[4:].split(":")
                        self._reader, self._writer = await asyncio.open_connection(
                            host, int(port)
                        )
                    else:
                        # Convert Unix socket path to TCP localhost connection
                        # Use a hash of the socket path to derive a consistent port
                        import hashlib
                        path_hash = int(hashlib.md5(self.socket_path.encode()).hexdigest()[:4], 16)
                        tcp_port = 4444 + (path_hash % 1000)  # Port range 4444-5443
                        logger.debug(f"Windows: Converting socket {self.socket_path} to TCP localhost:{tcp_port}",
                                    category=LogCategory.SYSTEM)
                        self._reader, self._writer = await asyncio.open_connection(
                            "127.0.0.1", tcp_port
                        )
                else:
                    # Unix socket
                    self._reader, self._writer = await asyncio.open_unix_connection(
                        self.socket_path
                    )

                # Read greeting
                greeting = await asyncio.wait_for(
                    self._reader.readline(),
                    timeout=5.0
                )

                # Enter command mode
                await self._execute({"execute": "qmp_capabilities"})

                self._connected = True
                logger.debug("QMP connected", category=LogCategory.SYSTEM)
                return True

            except (ConnectionRefusedError, FileNotFoundError):
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(f"QMP connection error: {e}", category=LogCategory.SYSTEM)
                await asyncio.sleep(0.5)

        return False

    async def disconnect(self):
        """Disconnect from QMP socket."""
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        self._connected = False

    async def _execute(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a QMP command."""
        if not self._writer or not self._reader:
            raise RuntimeError("Not connected to QMP")

        # Send command
        cmd_bytes = json.dumps(command).encode() + b"\n"
        self._writer.write(cmd_bytes)
        await self._writer.drain()

        # Read response
        response = await asyncio.wait_for(
            self._reader.readline(),
            timeout=10.0
        )

        return json.loads(response.decode())

    async def query_status(self) -> Dict[str, Any]:
        """Query VM status."""
        return await self._execute({"execute": "query-status"})

    async def stop(self):
        """Stop (pause) the VM."""
        return await self._execute({"execute": "stop"})

    async def cont(self):
        """Continue (resume) the VM."""
        return await self._execute({"execute": "cont"})

    async def quit(self):
        """Quit QEMU."""
        try:
            return await self._execute({"execute": "quit"})
        except Exception:
            pass  # Connection will be closed

    async def system_powerdown(self):
        """Send ACPI power down signal."""
        return await self._execute({"execute": "system_powerdown"})

    async def system_reset(self):
        """Reset the VM."""
        return await self._execute({"execute": "system_reset"})

    async def savevm(self, name: str):
        """Save VM snapshot."""
        return await self._execute({
            "execute": "savevm",
            "arguments": {"name": name}
        })

    async def loadvm(self, name: str):
        """Load VM snapshot."""
        return await self._execute({
            "execute": "loadvm",
            "arguments": {"name": name}
        })

    async def guest_exec(
        self,
        path: str,
        args: List[str] = None,
        capture_output: bool = True,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Execute command via QEMU Guest Agent.

        Requires qemu-guest-agent running in the VM.
        """
        command = {
            "execute": "guest-exec",
            "arguments": {
                "path": path,
                "arg": args or [],
                "capture-output": capture_output
            }
        }

        result = await self._execute(command)

        if "return" not in result:
            return result

        pid = result["return"]["pid"]

        # Wait for completion
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = await self._execute({
                "execute": "guest-exec-status",
                "arguments": {"pid": pid}
            })

            if status.get("return", {}).get("exited", False):
                return status["return"]

            await asyncio.sleep(0.1)

        raise TimeoutError(f"Command execution timed out after {timeout}s")


class QEMUVirtualMachine:
    """
    Manages a single QEMU virtual machine instance.

    Handles the complete VM lifecycle including:
    - Process management
    - Network configuration
    - File sharing setup
    - Command execution
    - Resource cleanup
    """

    def __init__(self, config: QEMUVMConfig, image_manager: QEMUImageManager):
        """Initialize VM instance."""
        self.config = config
        self.image_manager = image_manager

        self._process: Optional[asyncio.subprocess.Process] = None
        self._state = VMState.STOPPED
        self._qmp: Optional[QEMUMonitor] = None

        # Runtime paths
        self._runtime_dir: Optional[Path] = None
        self._qmp_socket: Optional[str] = None
        self._serial_socket: Optional[str] = None
        self._snapshot_image: Optional[Path] = None
        self._ssh_port: int = 0

        # Find QEMU binary
        self._qemu_binary = config.qemu_binary or self._find_qemu_binary()
        self._accelerator = config.accelerator or self._detect_accelerator()

        logger.debug(
            f"VM instance created: {config.name}",
            category=LogCategory.SYSTEM,
            context={"accelerator": self._accelerator.value}
        )

    def _find_qemu_binary(self) -> str:
        """Find QEMU system binary."""
        arch = "x86_64"

        candidates = [
            f"qemu-system-{arch}",
            f"/usr/bin/qemu-system-{arch}",
            f"/usr/local/bin/qemu-system-{arch}",
            f"C:\\Program Files\\qemu\\qemu-system-{arch}.exe",
            f"C:\\Program Files (x86)\\qemu\\qemu-system-{arch}.exe",
        ]

        for candidate in candidates:
            try:
                result = subprocess.run(
                    [candidate, "--version"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return candidate
            except Exception:
                continue

        raise RuntimeError("QEMU not found. Please install QEMU.")

    def _detect_accelerator(self) -> QEMUAccelerator:
        """Detect available hardware acceleration."""
        system = platform.system().lower()

        if system == "linux":
            if Path("/dev/kvm").exists():
                # Check if accessible
                try:
                    result = subprocess.run(
                        [self._qemu_binary, "-accel", "kvm", "-accel", "help"],
                        capture_output=True,
                        timeout=5
                    )
                    if b"kvm" in result.stdout.lower():
                        return QEMUAccelerator.KVM
                except Exception:
                    pass

        elif system == "windows":
            # Check for WHPX
            try:
                result = subprocess.run(
                    [self._qemu_binary, "-accel", "help"],
                    capture_output=True,
                    timeout=5
                )
                if b"whpx" in result.stdout.lower():
                    return QEMUAccelerator.WHPX
            except Exception:
                pass

        elif system == "darwin":
            # Check for HVF
            try:
                result = subprocess.run(
                    [self._qemu_binary, "-accel", "help"],
                    capture_output=True,
                    timeout=5
                )
                if b"hvf" in result.stdout.lower():
                    return QEMUAccelerator.HVF
            except Exception:
                pass

        # Fallback to TCG
        return QEMUAccelerator.TCG

    @property
    def state(self) -> VMState:
        """Get current VM state."""
        return self._state

    def _setup_runtime_dir(self):
        """Create runtime directory for sockets and temp files."""
        self._runtime_dir = Path(tempfile.mkdtemp(prefix=f"qemu_{self.config.name}_"))

        # Set up socket paths
        if sys.platform == "win32":
            # Use TCP sockets on Windows
            self._qmp_socket = f"tcp:127.0.0.1:{self._find_free_port()}"
            self._serial_socket = f"tcp:127.0.0.1:{self._find_free_port()}"
        else:
            self._qmp_socket = str(self._runtime_dir / "qmp.sock")
            self._serial_socket = str(self._runtime_dir / "serial.sock")

    def _find_free_port(self) -> int:
        """Find a free TCP port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    def _build_qemu_command(self) -> List[str]:
        """Build QEMU command line arguments."""
        cmd = [self._qemu_binary]

        # Machine and acceleration
        if self._accelerator == QEMUAccelerator.KVM:
            cmd.extend(["-enable-kvm"])
            cmd.extend(["-cpu", self.config.cpu_model])
        elif self._accelerator == QEMUAccelerator.WHPX:
            cmd.extend(["-accel", "whpx"])
            cmd.extend(["-cpu", "qemu64"])
        elif self._accelerator == QEMUAccelerator.HVF:
            cmd.extend(["-accel", "hvf"])
            cmd.extend(["-cpu", "host"])
        else:
            cmd.extend(["-accel", "tcg"])
            cmd.extend(["-cpu", "qemu64"])

        # Memory and CPUs
        cmd.extend(["-m", self.config.memory])
        cmd.extend(["-smp", str(self.config.cpus)])

        # Disk
        disk_image = self._snapshot_image or self.config.disk_image
        if disk_image:
            cmd.extend([
                "-drive",
                f"file={disk_image},format={self.config.disk_format},if=virtio"
            ])

        # Display
        cmd.extend(["-display", self.config.display])

        # QMP socket for control
        if sys.platform == "win32" and self._qmp_socket.startswith("tcp:"):
            host_port = self._qmp_socket[4:]
            cmd.extend(["-qmp", f"tcp:{host_port},server,nowait"])
        else:
            cmd.extend(["-qmp", f"unix:{self._qmp_socket},server,nowait"])

        # Serial console
        if self.config.serial_console:
            if sys.platform == "win32" and self._serial_socket.startswith("tcp:"):
                host_port = self._serial_socket[4:]
                cmd.extend(["-serial", f"tcp:{host_port},server,nowait"])
            else:
                cmd.extend(["-serial", f"unix:{self._serial_socket},server,nowait"])

        # Network
        if self.config.network_enabled:
            if self._ssh_port == 0:
                self._ssh_port = self._find_free_port()

            if self.config.network_type == "user":
                cmd.extend([
                    "-netdev", f"user,id=net0,hostfwd=tcp::{self._ssh_port}-:22",
                    "-device", "virtio-net-pci,netdev=net0"
                ])
            elif self.config.network_type == "tap":
                cmd.extend([
                    "-netdev", "tap,id=net0",
                    "-device", "virtio-net-pci,netdev=net0"
                ])
        else:
            cmd.extend(["-net", "none"])

        # Shared directory via VirtFS (9p)
        if self.config.shared_dir:
            cmd.extend([
                "-virtfs",
                f"local,path={self.config.shared_dir},"
                f"mount_tag={self.config.shared_mount_tag},"
                f"security_model=mapped-xattr,readonly=off"
            ])

        # Guest agent
        guest_agent_sock = self._runtime_dir / "guest-agent.sock" if self._runtime_dir else None
        if guest_agent_sock:
            if sys.platform != "win32":
                cmd.extend([
                    "-chardev", f"socket,path={guest_agent_sock},server=on,wait=off,id=qga0",
                    "-device", "virtio-serial",
                    "-device", "virtserialport,chardev=qga0,name=org.qemu.guest_agent.0"
                ])

        # Random number generator for faster boot
        cmd.extend(["-device", "virtio-rng-pci"])

        # Daemonize (don't daemonize - we manage the process)
        cmd.extend(["-daemonize"])

        return cmd

    async def start(self) -> bool:
        """Start the virtual machine."""
        if self._state != VMState.STOPPED:
            logger.warning(f"Cannot start VM in state: {self._state}", category=LogCategory.SYSTEM)
            return False

        self._state = VMState.STARTING

        try:
            # Setup runtime directory
            self._setup_runtime_dir()

            # Create snapshot if using base image
            if self.config.use_snapshot and self.config.disk_image:
                base_image = Path(self.config.disk_image)
                if base_image.exists():
                    self._snapshot_image = self.image_manager.create_snapshot(
                        base_image,
                        f"{self.config.name}_{uuid.uuid4().hex[:8]}"
                    )

            # Build and execute QEMU command
            cmd = self._build_qemu_command()

            logger.debug(
                "Starting QEMU",
                category=LogCategory.SYSTEM,
                context={"command": " ".join(cmd)}
            )

            # Start QEMU process
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Wait a moment for QEMU to start
            await asyncio.sleep(1.0)

            # Connect to QMP
            self._qmp = QEMUMonitor(self._qmp_socket)
            if await self._qmp.connect(timeout=30.0):
                self._state = VMState.RUNNING
                logger.info(
                    f"VM started: {self.config.name}",
                    category=LogCategory.SYSTEM,
                    context={"ssh_port": self._ssh_port if self.config.network_enabled else None}
                )
                return True
            else:
                raise RuntimeError("Failed to connect to QMP")

        except Exception as e:
            self._state = VMState.ERROR
            logger.error(
                f"Failed to start VM: {e}",
                category=LogCategory.SYSTEM
            )
            await self.cleanup()
            return False

    async def stop(self, force: bool = False, timeout: float = 30.0) -> bool:
        """Stop the virtual machine."""
        if self._state not in (VMState.RUNNING, VMState.PAUSED, VMState.ERROR):
            return True

        self._state = VMState.STOPPING

        try:
            if self._qmp and self._qmp._connected:
                if force:
                    await self._qmp.quit()
                else:
                    # Try graceful shutdown first
                    await self._qmp.system_powerdown()

                    # Wait for VM to stop
                    start_time = time.time()
                    while time.time() - start_time < timeout:
                        try:
                            status = await self._qmp.query_status()
                            if status.get("return", {}).get("status") == "shutdown":
                                break
                        except Exception:
                            break
                        await asyncio.sleep(0.5)
                    else:
                        # Force quit if graceful shutdown failed
                        await self._qmp.quit()

            # Wait for process to terminate
            if self._process:
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()

            self._state = VMState.STOPPED
            logger.info(f"VM stopped: {self.config.name}", category=LogCategory.SYSTEM)
            return True

        except Exception as e:
            logger.error(f"Error stopping VM: {e}", category=LogCategory.SYSTEM)
            self._state = VMState.ERROR
            return False
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up VM resources."""
        # Disconnect QMP
        if self._qmp:
            await self._qmp.disconnect()
            self._qmp = None

        # Clean up snapshot
        if self._snapshot_image:
            self.image_manager.delete_snapshot(self._snapshot_image)
            self._snapshot_image = None

        # Clean up runtime directory
        if self._runtime_dir and self._runtime_dir.exists():
            try:
                shutil.rmtree(self._runtime_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up runtime dir: {e}", category=LogCategory.SYSTEM)
            self._runtime_dir = None

    async def reset(self) -> bool:
        """
        Reset VM state for reuse in a pool.

        This restores the VM to a clean state by:
        1. Loading a saved snapshot (if available)
        2. Clearing the shared directory
        3. Falling back to system reset if no snapshot

        Returns:
            True if reset successful, False otherwise
        """
        if self._state != VMState.RUNNING:
            logger.warning(f"Cannot reset VM in state: {self._state}", category=LogCategory.SYSTEM)
            return False

        try:
            # Clear shared directory if configured
            if self.config.shared_dir:
                shared_path = Path(self.config.shared_dir)
                if shared_path.exists():
                    for item in shared_path.iterdir():
                        try:
                            if item.is_file():
                                item.unlink()
                            elif item.is_dir():
                                shutil.rmtree(item)
                        except Exception as e:
                            logger.warning(f"Failed to clear shared item {item}: {e}",
                                         category=LogCategory.SYSTEM)

            # Try to load a clean snapshot if the VM has one saved
            # Note: This requires the VM to have been set up with a saved snapshot
            if self._qmp and self._qmp._connected:
                try:
                    # Try to load the "clean" snapshot if it exists
                    await self._qmp.loadvm("clean")
                    logger.debug("VM reset via snapshot restore", category=LogCategory.SYSTEM)
                    return True
                except Exception:
                    # No clean snapshot, fall back to system reset
                    pass

                # Fall back to system reset
                await self._qmp.system_reset()
                logger.debug("VM reset via system reset", category=LogCategory.SYSTEM)

                # Wait a moment for the reset to complete
                await asyncio.sleep(2.0)
                return True

            return False

        except Exception as e:
            logger.error(f"VM reset failed: {e}", category=LogCategory.SYSTEM)
            return False

    async def save_clean_snapshot(self) -> bool:
        """
        Save a clean snapshot for later restoration.

        Call this after VM is fully booted and ready to create a restore point.
        """
        if self._state != VMState.RUNNING or not self._qmp:
            return False

        try:
            await self._qmp.savevm("clean")
            logger.info("Saved clean VM snapshot", category=LogCategory.SYSTEM)
            return True
        except Exception as e:
            logger.warning(f"Failed to save clean snapshot: {e}", category=LogCategory.SYSTEM)
            return False

    async def execute_via_serial(
        self,
        command: str,
        timeout: float = 60.0
    ) -> Tuple[str, str, int]:
        """
        Execute a command via serial console.

        This requires the guest to have a getty or similar on the serial port.
        """
        if not self._serial_socket:
            raise RuntimeError("Serial console not configured")

        stdout_data = []
        stderr_data = []
        exit_code = -1

        try:
            if sys.platform == "win32" and self._serial_socket.startswith("tcp:"):
                host, port = self._serial_socket[4:].split(":")
                reader, writer = await asyncio.open_connection(host, int(port))
            else:
                reader, writer = await asyncio.open_unix_connection(self._serial_socket)

            # Send command with exit code capture
            full_command = f"{command}; echo EXIT_CODE:$?\n"
            writer.write(full_command.encode())
            await writer.drain()

            # Read output until we see the exit code marker
            start_time = time.time()
            output_buffer = ""

            while time.time() - start_time < timeout:
                try:
                    data = await asyncio.wait_for(reader.read(4096), timeout=1.0)
                    if not data:
                        break
                    output_buffer += data.decode('utf-8', errors='ignore')

                    # Check for exit code marker
                    if "EXIT_CODE:" in output_buffer:
                        lines = output_buffer.split("\n")
                        for line in lines:
                            if line.startswith("EXIT_CODE:"):
                                try:
                                    exit_code = int(line.split(":")[1].strip())
                                except ValueError:
                                    pass
                            elif not line.startswith(command):
                                stdout_data.append(line)
                        break
                except asyncio.TimeoutError:
                    continue

            writer.close()
            await writer.wait_closed()

        except Exception as e:
            stderr_data.append(str(e))

        return "\n".join(stdout_data), "\n".join(stderr_data), exit_code

    async def execute_via_ssh(
        self,
        command: str,
        timeout: float = 60.0
    ) -> Tuple[str, str, int]:
        """Execute a command via SSH."""
        if not self.config.network_enabled or self._ssh_port == 0:
            raise RuntimeError("SSH not available - network not enabled")

        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            "-p", str(self._ssh_port),
            f"{self.config.guest_user}@127.0.0.1",
            command
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            return (
                stdout.decode('utf-8', errors='ignore'),
                stderr.decode('utf-8', errors='ignore'),
                process.returncode or 0
            )

        except asyncio.TimeoutError:
            raise TimeoutError(f"SSH command timed out after {timeout}s")

    async def execute_via_virtfs(
        self,
        script_path: Path,
        language: str = "python",
        timeout: float = 60.0
    ) -> Tuple[str, str, int]:
        """
        Execute a script via VirtFS shared directory.

        The script is copied to the shared directory, executed via serial/SSH,
        and output is captured from result files.
        """
        if not self.config.shared_dir:
            raise RuntimeError("VirtFS not configured")

        shared_dir = Path(self.config.shared_dir)

        # Copy script to shared directory
        script_name = script_path.name
        shared_script = shared_dir / script_name
        shutil.copy(script_path, shared_script)

        # Create output capture script
        output_file = shared_dir / "output.txt"
        error_file = shared_dir / "error.txt"
        exitcode_file = shared_dir / "exitcode.txt"

        # Build run command based on language
        run_commands = {
            "python": f"python3 /mnt/hostshare/{script_name}",
            "julia": f"julia /mnt/hostshare/{script_name}",
            "r": f"Rscript /mnt/hostshare/{script_name}",
            "matlab": f"octave --no-gui /mnt/hostshare/{script_name}",
        }

        run_cmd = run_commands.get(language, f"python3 /mnt/hostshare/{script_name}")

        # Create wrapper script
        wrapper_script = f"""#!/bin/bash
cd /mnt/hostshare
{run_cmd} > /mnt/hostshare/output.txt 2> /mnt/hostshare/error.txt
echo $? > /mnt/hostshare/exitcode.txt
"""

        wrapper_path = shared_dir / "run_script.sh"
        wrapper_path.write_text(wrapper_script)

        # Execute wrapper script
        execute_cmd = f"chmod +x /mnt/hostshare/run_script.sh && /mnt/hostshare/run_script.sh"

        try:
            if self.config.network_enabled and self._ssh_port > 0:
                await self.execute_via_ssh(execute_cmd, timeout)
            else:
                await self.execute_via_serial(execute_cmd, timeout)
        except Exception as e:
            return "", str(e), -1

        # Read results
        stdout = ""
        stderr = ""
        exit_code = -1

        # Wait for output files
        await asyncio.sleep(1.0)

        if output_file.exists():
            stdout = output_file.read_text()
        if error_file.exists():
            stderr = error_file.read_text()
        if exitcode_file.exists():
            try:
                exit_code = int(exitcode_file.read_text().strip())
            except ValueError:
                pass

        # Cleanup
        for f in [shared_script, output_file, error_file, exitcode_file, wrapper_path]:
            try:
                f.unlink()
            except Exception:
                pass

        return stdout, stderr, exit_code


class QEMUPool:
    """
    Pool of pre-warmed QEMU VMs for fast execution.

    Maintains a pool of ready-to-use VMs that can be quickly
    allocated for script execution, then reset and returned.
    """

    def __init__(
        self,
        config: QEMUVMConfig,
        pool_size: int = 2,
        image_manager: Optional[QEMUImageManager] = None
    ):
        """Initialize VM pool."""
        self.config = config
        self.pool_size = pool_size
        self.image_manager = image_manager or QEMUImageManager()

        self._available: List[QEMUVirtualMachine] = []
        self._in_use: Dict[str, QEMUVirtualMachine] = {}
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        """Initialize the VM pool with pre-warmed instances."""
        if self._initialized:
            return

        logger.info(
            f"Initializing QEMU pool with {self.pool_size} VMs",
            category=LogCategory.SYSTEM
        )

        for i in range(self.pool_size):
            vm_config = QEMUVMConfig(**{
                **self.config.model_dump(),
                "name": f"{self.config.name}_{i}"
            })

            vm = QEMUVirtualMachine(vm_config, self.image_manager)

            if await vm.start():
                # Save a clean snapshot for efficient resets
                await vm.save_clean_snapshot()
                self._available.append(vm)
            else:
                logger.warning(f"Failed to start pool VM {i}", category=LogCategory.SYSTEM)

        self._initialized = True
        logger.info(
            f"QEMU pool initialized with {len(self._available)} VMs",
            category=LogCategory.SYSTEM
        )

    async def acquire(self) -> Optional[QEMUVirtualMachine]:
        """Acquire a VM from the pool."""
        async with self._lock:
            if self._available:
                vm = self._available.pop(0)
                vm_id = str(uuid.uuid4())
                self._in_use[vm_id] = vm
                return vm
        return None

    async def release(self, vm: QEMUVirtualMachine):
        """Release a VM back to the pool."""
        async with self._lock:
            # Find and remove from in_use
            vm_id = None
            for vid, v in self._in_use.items():
                if v is vm:
                    vm_id = vid
                    break

            if vm_id:
                del self._in_use[vm_id]

            # Reset and return to pool if still healthy
            if vm.state == VMState.RUNNING:
                # Reset VM state (restore snapshot, clear shared dir, etc.)
                if await vm.reset():
                    self._available.append(vm)
                    logger.debug(f"VM reset and returned to pool", category=LogCategory.SYSTEM)
                else:
                    # Reset failed, stop and recreate
                    logger.warning("VM reset failed, recreating", category=LogCategory.SYSTEM)
                    await vm.stop(force=True)
                    new_vm = QEMUVirtualMachine(vm.config, self.image_manager)
                    if await new_vm.start():
                        # Save a clean snapshot for future resets
                        await new_vm.save_clean_snapshot()
                        self._available.append(new_vm)
            else:
                # VM is unhealthy, stop it and create a new one
                await vm.stop(force=True)

                new_vm = QEMUVirtualMachine(vm.config, self.image_manager)
                if await new_vm.start():
                    self._available.append(new_vm)

    async def shutdown(self):
        """Shutdown all VMs in the pool."""
        async with self._lock:
            for vm in self._available + list(self._in_use.values()):
                await vm.stop(force=True)

            self._available.clear()
            self._in_use.clear()
            self._initialized = False

        logger.info("QEMU pool shutdown complete", category=LogCategory.SYSTEM)


class QEMUBackendImpl:
    """
    Full QEMU backend implementation for the sandbox system.

    This class implements the actual QEMU virtualization logic,
    separate from the SandboxBackend interface implementation.
    """

    # Default base image locations
    DEFAULT_BASE_IMAGES = {
        "python": "python-sandbox.qcow2",
        "julia": "julia-sandbox.qcow2",
        "r": "r-sandbox.qcow2",
        "matlab": "octave-sandbox.qcow2",
    }

    def __init__(
        self,
        image_manager: Optional[QEMUImageManager] = None,
        use_pool: bool = False,
        pool_size: int = 2
    ):
        """Initialize QEMU backend."""
        self.image_manager = image_manager or QEMUImageManager()
        self.use_pool = use_pool
        self._pool: Optional[QEMUPool] = None
        self._pool_size = pool_size
        self._initialized = False

    async def initialize(self, base_config: Optional[QEMUVMConfig] = None):
        """Initialize the backend and optionally the VM pool."""
        if self._initialized:
            return

        # Initialize pool if enabled
        if self.use_pool and base_config:
            self._pool = QEMUPool(
                base_config,
                self._pool_size,
                self.image_manager
            )
            await self._pool.initialize()

        self._initialized = True
        logger.info("QEMU backend initialized", category=LogCategory.SYSTEM)

    async def shutdown(self):
        """Shutdown the backend."""
        if self._pool:
            await self._pool.shutdown()

        # Cleanup old snapshots
        self.image_manager.cleanup_old_snapshots()

        self._initialized = False

    def is_available(self) -> bool:
        """Check if QEMU is available on this system."""
        try:
            candidates = [
                "qemu-system-x86_64",
                "/usr/bin/qemu-system-x86_64",
                "C:\\Program Files\\qemu\\qemu-system-x86_64.exe",
            ]

            for candidate in candidates:
                try:
                    result = subprocess.run(
                        [candidate, "--version"],
                        capture_output=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        return True
                except Exception:
                    continue

            return False
        except Exception:
            return False

    def get_base_image_path(self, language: str) -> Optional[Path]:
        """Get path to base image for a language."""
        image_name = self.DEFAULT_BASE_IMAGES.get(language)
        if not image_name:
            return None

        image_path = self.image_manager.images_dir / image_name
        if image_path.exists():
            return image_path

        return None

    async def execute(
        self,
        script_path: Path,
        config: QEMUVMConfig,
        language: str = "python"
    ) -> QEMUExecutionResult:
        """
        Execute a script in a QEMU VM.

        Args:
            script_path: Path to the script to execute
            config: VM configuration
            language: Programming language

        Returns:
            QEMUExecutionResult with output and status
        """
        result = QEMUExecutionResult(success=False)
        start_time = time.time()

        # Check for base image
        if not config.disk_image:
            base_image = self.get_base_image_path(language)
            if base_image:
                config.disk_image = str(base_image)
            else:
                result.error = f"No base image available for {language}"
                return result

        # Setup shared directory
        shared_dir = None
        if config.execution_mode == ExecutionMode.VIRTFS:
            shared_dir = tempfile.mkdtemp(prefix="qemu_share_")
            config.shared_dir = shared_dir

        vm: Optional[QEMUVirtualMachine] = None

        try:
            # Get VM from pool or create new one
            if self._pool:
                vm = await self._pool.acquire()

            if not vm:
                vm = QEMUVirtualMachine(config, self.image_manager)
                boot_start = time.time()

                if not await vm.start():
                    result.error = "Failed to start VM"
                    return result

                result.vm_boot_time = time.time() - boot_start

            # Execute script
            if config.execution_mode == ExecutionMode.VIRTFS:
                stdout, stderr, exit_code = await vm.execute_via_virtfs(
                    script_path,
                    language,
                    config.timeout_seconds
                )
            elif config.execution_mode == ExecutionMode.SSH:
                # Copy script content and execute
                script_content = script_path.read_text()
                run_commands = {
                    "python": f"python3 -c {repr(script_content)}",
                    "julia": f"julia -e {repr(script_content)}",
                    "r": f"Rscript -e {repr(script_content)}",
                }
                cmd = run_commands.get(language, f"python3 -c {repr(script_content)}")
                stdout, stderr, exit_code = await vm.execute_via_ssh(
                    cmd,
                    config.timeout_seconds
                )
            else:
                # Serial console
                run_commands = {
                    "python": f"python3 {script_path}",
                    "julia": f"julia {script_path}",
                    "r": f"Rscript {script_path}",
                }
                cmd = run_commands.get(language, f"python3 {script_path}")
                stdout, stderr, exit_code = await vm.execute_via_serial(
                    cmd,
                    config.timeout_seconds
                )

            result.stdout = stdout
            result.stderr = stderr
            result.exit_code = exit_code
            result.success = exit_code == 0
            result.vm_state = vm.state

        except TimeoutError as e:
            result.error = str(e)
            result.stderr = f"Execution timed out after {config.timeout_seconds}s"
        except Exception as e:
            result.error = str(e)
            logger.error(f"QEMU execution error: {e}", category=LogCategory.EXECUTION)
        finally:
            # Return VM to pool or stop it
            if vm:
                if self._pool:
                    await self._pool.release(vm)
                else:
                    await vm.stop()

            # Cleanup shared directory
            if shared_dir and Path(shared_dir).exists():
                try:
                    shutil.rmtree(shared_dir)
                except Exception:
                    pass

        result.execution_time = time.time() - start_time
        return result


# Convenience function to create pre-configured backend
def create_qemu_backend(
    use_pool: bool = False,
    pool_size: int = 2,
    images_dir: Optional[Path] = None
) -> QEMUBackendImpl:
    """
    Create a configured QEMU backend instance.

    Args:
        use_pool: Enable VM pooling for faster execution
        pool_size: Number of VMs to keep in the pool
        images_dir: Directory for VM images

    Returns:
        Configured QEMUBackendImpl instance
    """
    image_manager = QEMUImageManager(images_dir) if images_dir else QEMUImageManager()

    return QEMUBackendImpl(
        image_manager=image_manager,
        use_pool=use_pool,
        pool_size=pool_size
    )


__all__ = [
    'QEMUAccelerator',
    'VMState',
    'ExecutionMode',
    'QEMUVMConfig',
    'QEMUExecutionResult',
    'QEMUImageManager',
    'QEMUMonitor',
    'QEMUVirtualMachine',
    'QEMUPool',
    'QEMUBackendImpl',
    'create_qemu_backend',
]
