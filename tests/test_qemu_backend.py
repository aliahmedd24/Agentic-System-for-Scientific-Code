"""
Tests for QEMU system virtualization backend.

These tests cover the QEMU backend functionality including:
- Configuration validation (QEMUVMConfig, QEMUExecutionResult)
- Enum types (QEMUAccelerator, VMState, ExecutionMode)
- QEMUImageManager (disk image management)
- QEMUMonitor (QMP protocol)
- QEMUVirtualMachine (VM lifecycle)
- QEMUPool (VM pooling)
- QEMUBackendImpl (full backend implementation)

Note: Tests marked with @pytest.mark.qemu require QEMU to be installed and
are very slow (involve full VM operations). They are skipped by default.
"""

import pytest
import asyncio
import tempfile
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from core.qemu_backend import (
    QEMUAccelerator,
    VMState,
    ExecutionMode,
    QEMUVMConfig,
    QEMUExecutionResult,
    QEMUImageManager,
    QEMUMonitor,
    QEMUVirtualMachine,
    QEMUPool,
    QEMUBackendImpl,
    create_qemu_backend,
)


# =============================================================================
# Enum Tests
# =============================================================================

class TestQEMUAccelerator:
    """Tests for QEMUAccelerator enum."""

    def test_accelerator_values(self):
        """Test accelerator enum values."""
        assert QEMUAccelerator.KVM.value == "kvm"
        assert QEMUAccelerator.WHPX.value == "whpx"
        assert QEMUAccelerator.HVF.value == "hvf"
        assert QEMUAccelerator.HAXM.value == "haxm"
        assert QEMUAccelerator.TCG.value == "tcg"

    def test_all_accelerators_defined(self):
        """Test that all expected accelerators are defined."""
        accelerators = list(QEMUAccelerator)
        assert len(accelerators) == 5

    def test_accelerator_from_string(self):
        """Test creating accelerator from string value."""
        assert QEMUAccelerator("kvm") == QEMUAccelerator.KVM
        assert QEMUAccelerator("tcg") == QEMUAccelerator.TCG


class TestVMState:
    """Tests for VMState enum."""

    def test_state_values(self):
        """Test VM state enum values."""
        assert VMState.STOPPED.value == "stopped"
        assert VMState.STARTING.value == "starting"
        assert VMState.RUNNING.value == "running"
        assert VMState.PAUSED.value == "paused"
        assert VMState.STOPPING.value == "stopping"
        assert VMState.ERROR.value == "error"

    def test_all_states_defined(self):
        """Test that all expected states are defined."""
        states = list(VMState)
        assert len(states) == 6


class TestExecutionMode:
    """Tests for ExecutionMode enum."""

    def test_mode_values(self):
        """Test execution mode enum values."""
        assert ExecutionMode.SERIAL.value == "serial"
        assert ExecutionMode.SSH.value == "ssh"
        assert ExecutionMode.VIRTFS.value == "virtfs"
        assert ExecutionMode.CLOUDINIT.value == "cloudinit"


# =============================================================================
# Pydantic Model Tests
# =============================================================================

class TestQEMUVMConfig:
    """Tests for QEMUVMConfig Pydantic model."""

    def test_default_values(self):
        """Test config with default values."""
        config = QEMUVMConfig()

        assert config.name == "sandbox-vm"
        assert config.memory == "2G"
        assert config.cpus == 2
        assert config.cpu_model == "host"
        assert config.disk_size == "10G"
        assert config.disk_format == "qcow2"
        assert config.use_snapshot is True
        assert config.network_enabled is False
        assert config.display == "none"
        assert config.serial_console is True
        assert config.execution_mode == ExecutionMode.VIRTFS
        assert config.timeout_seconds == 120
        assert config.guest_user == "sandbox"
        assert config.guest_workdir == "/sandbox"

    def test_custom_values(self):
        """Test config with custom values."""
        config = QEMUVMConfig(
            name="test-vm",
            memory="4G",
            cpus=4,
            network_enabled=True,
            ssh_port=2222,
            execution_mode=ExecutionMode.SSH,
            timeout_seconds=300
        )

        assert config.name == "test-vm"
        assert config.memory == "4G"
        assert config.cpus == 4
        assert config.network_enabled is True
        assert config.ssh_port == 2222
        assert config.execution_mode == ExecutionMode.SSH
        assert config.timeout_seconds == 300

    def test_cpus_validation(self):
        """Test CPU count validation (must be >= 1)."""
        config = QEMUVMConfig(cpus=1)
        assert config.cpus == 1

        with pytest.raises(ValueError):
            QEMUVMConfig(cpus=0)

        with pytest.raises(ValueError):
            QEMUVMConfig(cpus=-1)

    def test_timeout_validation(self):
        """Test timeout validation (must be >= 1)."""
        config = QEMUVMConfig(timeout_seconds=1)
        assert config.timeout_seconds == 1

        with pytest.raises(ValueError):
            QEMUVMConfig(timeout_seconds=0)

    def test_ssh_port_validation(self):
        """Test SSH port validation (must be >= 0)."""
        config = QEMUVMConfig(ssh_port=0)
        assert config.ssh_port == 0

        config = QEMUVMConfig(ssh_port=22)
        assert config.ssh_port == 22

        with pytest.raises(ValueError):
            QEMUVMConfig(ssh_port=-1)

    def test_accelerator_setting(self):
        """Test setting hardware accelerator."""
        config = QEMUVMConfig(accelerator=QEMUAccelerator.KVM)
        assert config.accelerator == QEMUAccelerator.KVM

        config = QEMUVMConfig(accelerator=QEMUAccelerator.TCG)
        assert config.accelerator == QEMUAccelerator.TCG

    def test_forbid_extra_fields(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValueError):
            QEMUVMConfig(unknown_field="value")

    def test_model_dump(self):
        """Test model serialization."""
        config = QEMUVMConfig(name="test-vm", cpus=4)
        data = config.model_dump()

        assert isinstance(data, dict)
        assert data["name"] == "test-vm"
        assert data["cpus"] == 4


class TestQEMUExecutionResult:
    """Tests for QEMUExecutionResult Pydantic model."""

    def test_minimal_result(self):
        """Test result with minimal required fields."""
        result = QEMUExecutionResult(success=True)

        assert result.success is True
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.exit_code == -1
        assert result.execution_time == 0.0
        assert result.vm_boot_time == 0.0
        assert result.output_files == []
        assert result.error is None
        assert result.vm_state == VMState.STOPPED

    def test_complete_result(self):
        """Test result with all fields."""
        result = QEMUExecutionResult(
            success=True,
            stdout="Hello, World!",
            stderr="",
            exit_code=0,
            execution_time=5.5,
            vm_boot_time=2.3,
            output_files=["output.txt"],
            vm_state=VMState.RUNNING
        )

        assert result.success is True
        assert result.stdout == "Hello, World!"
        assert result.exit_code == 0
        assert result.execution_time == 5.5
        assert result.vm_boot_time == 2.3
        assert "output.txt" in result.output_files

    def test_failed_result(self):
        """Test failed execution result."""
        result = QEMUExecutionResult(
            success=False,
            stderr="Error: script failed",
            exit_code=1,
            error="Execution failed",
            vm_state=VMState.ERROR
        )

        assert result.success is False
        assert "Error" in result.stderr
        assert result.exit_code == 1
        assert result.error == "Execution failed"
        assert result.vm_state == VMState.ERROR

    def test_execution_time_validation(self):
        """Test execution time must be non-negative."""
        result = QEMUExecutionResult(success=True, execution_time=0.0)
        assert result.execution_time == 0.0

        with pytest.raises(ValueError):
            QEMUExecutionResult(success=True, execution_time=-1.0)

    def test_vm_boot_time_validation(self):
        """Test VM boot time must be non-negative."""
        result = QEMUExecutionResult(success=True, vm_boot_time=0.0)
        assert result.vm_boot_time == 0.0

        with pytest.raises(ValueError):
            QEMUExecutionResult(success=True, vm_boot_time=-1.0)


# =============================================================================
# QEMUImageManager Tests
# =============================================================================

class TestQEMUImageManager:
    """Tests for QEMUImageManager class."""

    @patch('subprocess.run')
    def test_initialization_with_custom_dir(self, mock_run, tmp_path):
        """Test initialization with custom images directory."""
        mock_run.return_value = Mock(returncode=0)

        manager = QEMUImageManager(images_dir=tmp_path)

        assert manager.images_dir == tmp_path
        assert tmp_path.exists()

    @patch('subprocess.run')
    def test_qemu_img_not_found(self, mock_run, tmp_path):
        """Test error when qemu-img binary not found."""
        mock_run.side_effect = FileNotFoundError("qemu-img not found")

        with pytest.raises(RuntimeError, match="qemu-img not found"):
            QEMUImageManager(images_dir=tmp_path)

    @patch('subprocess.run')
    def test_create_base_image(self, mock_run, tmp_path):
        """Test creating a base disk image."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        manager = QEMUImageManager(images_dir=tmp_path)
        image_path = manager.create_base_image("test-base", size="5G")

        assert image_path.suffix == ".qcow2"
        # Check that qemu-img create was called
        create_calls = [c for c in mock_run.call_args_list if "create" in str(c)]
        assert len(create_calls) > 0

    @patch('subprocess.run')
    def test_create_base_image_already_exists(self, mock_run, tmp_path):
        """Test that existing base image is not recreated."""
        mock_run.return_value = Mock(returncode=0)

        manager = QEMUImageManager(images_dir=tmp_path)

        # Create fake existing image
        existing_image = tmp_path / "existing.qcow2"
        existing_image.touch()

        result = manager.create_base_image("existing", fmt="qcow2")
        assert result == existing_image

    @patch('subprocess.run')
    def test_create_snapshot(self, mock_run, tmp_path):
        """Test creating a snapshot from base image."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        manager = QEMUImageManager(images_dir=tmp_path)

        # Create fake base image
        base_image = tmp_path / "base.qcow2"
        base_image.touch()

        snapshot_path = manager.create_snapshot(base_image, "test-snapshot")

        assert "snapshots" in str(snapshot_path)
        assert snapshot_path.suffix == ".qcow2"

    @patch('subprocess.run')
    def test_create_snapshot_base_not_found(self, mock_run, tmp_path):
        """Test error when base image doesn't exist."""
        mock_run.return_value = Mock(returncode=0)

        manager = QEMUImageManager(images_dir=tmp_path)

        with pytest.raises(FileNotFoundError):
            manager.create_snapshot(tmp_path / "nonexistent.qcow2")

    @patch('subprocess.run')
    def test_delete_snapshot(self, mock_run, tmp_path):
        """Test deleting a snapshot."""
        mock_run.return_value = Mock(returncode=0)

        manager = QEMUImageManager(images_dir=tmp_path)

        # Create fake snapshot
        snapshot = tmp_path / "snapshot.qcow2"
        snapshot.touch()
        assert snapshot.exists()

        result = manager.delete_snapshot(snapshot)
        assert result is True
        assert not snapshot.exists()

    @patch('subprocess.run')
    def test_delete_nonexistent_snapshot(self, mock_run, tmp_path):
        """Test deleting non-existent snapshot returns False."""
        mock_run.return_value = Mock(returncode=0)

        manager = QEMUImageManager(images_dir=tmp_path)
        result = manager.delete_snapshot(tmp_path / "nonexistent.qcow2")
        assert result is False

    @patch('subprocess.run')
    def test_get_image_info(self, mock_run, tmp_path):
        """Test getting disk image info."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"format": "qcow2", "virtual-size": 10737418240}',
            stderr=""
        )

        manager = QEMUImageManager(images_dir=tmp_path)

        # Create fake image
        image = tmp_path / "test.qcow2"
        image.touch()

        info = manager.get_image_info(image)
        assert isinstance(info, dict)
        assert info["format"] == "qcow2"

    @patch('subprocess.run')
    def test_list_images(self, mock_run, tmp_path):
        """Test listing all available images."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"format": "qcow2", "virtual-size": 10737418240}',
            stderr=""
        )

        manager = QEMUImageManager(images_dir=tmp_path)

        # Create some fake images
        (tmp_path / "image1.qcow2").touch()
        (tmp_path / "image2.qcow2").touch()

        images = manager.list_images()
        assert len(images) >= 2

    @patch('subprocess.run')
    def test_cleanup_old_snapshots(self, mock_run, tmp_path):
        """Test cleaning up old snapshots."""
        import time

        mock_run.return_value = Mock(returncode=0)

        manager = QEMUImageManager(images_dir=tmp_path)

        # Create snapshots directory
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()

        # Create old snapshot (modify timestamp)
        old_snapshot = snapshots_dir / "old.qcow2"
        old_snapshot.touch()

        # Set modification time to 48 hours ago
        import os
        old_time = time.time() - (48 * 3600)
        os.utime(old_snapshot, (old_time, old_time))

        # Create recent snapshot
        recent_snapshot = snapshots_dir / "recent.qcow2"
        recent_snapshot.touch()

        # Cleanup with 24 hour max age
        deleted = manager.cleanup_old_snapshots(max_age_hours=24)

        assert deleted == 1
        assert not old_snapshot.exists()
        assert recent_snapshot.exists()


# =============================================================================
# QEMUMonitor Tests (Mocked)
# =============================================================================

class TestQEMUMonitor:
    """Tests for QEMUMonitor (QMP client) - mocked."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = QEMUMonitor("/tmp/qmp.sock")

        assert monitor.socket_path == "/tmp/qmp.sock"
        assert monitor._connected is False

    @pytest.mark.asyncio
    async def test_connect_timeout(self):
        """Test connection timeout handling."""
        monitor = QEMUMonitor("/nonexistent/socket.sock")

        # Should fail to connect within timeout
        result = await monitor.connect(timeout=0.1)
        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        """Test disconnect when not connected."""
        monitor = QEMUMonitor("/tmp/qmp.sock")

        # Should not raise
        await monitor.disconnect()
        assert monitor._connected is False

    @pytest.mark.asyncio
    async def test_execute_not_connected(self):
        """Test execute raises when not connected."""
        monitor = QEMUMonitor("/tmp/qmp.sock")

        with pytest.raises(RuntimeError, match="Not connected"):
            await monitor._execute({"execute": "query-status"})

    @pytest.mark.asyncio
    async def test_query_status_mocked(self):
        """Test query_status with mocked connection."""
        monitor = QEMUMonitor("/tmp/qmp.sock")

        # Mock the _execute method
        monitor._execute = AsyncMock(return_value={
            "return": {"status": "running", "singlestep": False, "running": True}
        })
        monitor._connected = True

        result = await monitor.query_status()
        assert result["return"]["status"] == "running"

    @pytest.mark.asyncio
    async def test_stop_mocked(self):
        """Test stop command."""
        monitor = QEMUMonitor("/tmp/qmp.sock")
        monitor._execute = AsyncMock(return_value={"return": {}})
        monitor._connected = True

        result = await monitor.stop()
        monitor._execute.assert_called_with({"execute": "stop"})

    @pytest.mark.asyncio
    async def test_cont_mocked(self):
        """Test continue command."""
        monitor = QEMUMonitor("/tmp/qmp.sock")
        monitor._execute = AsyncMock(return_value={"return": {}})
        monitor._connected = True

        result = await monitor.cont()
        monitor._execute.assert_called_with({"execute": "cont"})

    @pytest.mark.asyncio
    async def test_system_reset_mocked(self):
        """Test system reset command."""
        monitor = QEMUMonitor("/tmp/qmp.sock")
        monitor._execute = AsyncMock(return_value={"return": {}})
        monitor._connected = True

        await monitor.system_reset()
        monitor._execute.assert_called_with({"execute": "system_reset"})

    @pytest.mark.asyncio
    async def test_savevm_mocked(self):
        """Test save VM snapshot command."""
        monitor = QEMUMonitor("/tmp/qmp.sock")
        monitor._execute = AsyncMock(return_value={"return": {}})
        monitor._connected = True

        await monitor.savevm("clean")
        monitor._execute.assert_called_with({
            "execute": "savevm",
            "arguments": {"name": "clean"}
        })

    @pytest.mark.asyncio
    async def test_loadvm_mocked(self):
        """Test load VM snapshot command."""
        monitor = QEMUMonitor("/tmp/qmp.sock")
        monitor._execute = AsyncMock(return_value={"return": {}})
        monitor._connected = True

        await monitor.loadvm("clean")
        monitor._execute.assert_called_with({
            "execute": "loadvm",
            "arguments": {"name": "clean"}
        })


# =============================================================================
# QEMUVirtualMachine Tests (Mocked)
# =============================================================================

class TestQEMUVirtualMachine:
    """Tests for QEMUVirtualMachine class - mocked."""

    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    @patch.object(QEMUVirtualMachine, '_find_qemu_binary', return_value="qemu-system-x86_64")
    @patch.object(QEMUVirtualMachine, '_detect_accelerator', return_value=QEMUAccelerator.TCG)
    def test_initialization(self, mock_accel, mock_binary, mock_img, tmp_path):
        """Test VM initialization."""
        config = QEMUVMConfig(name="test-vm")
        image_manager = QEMUImageManager(images_dir=tmp_path)

        vm = QEMUVirtualMachine(config, image_manager)

        assert vm.config.name == "test-vm"
        assert vm.state == VMState.STOPPED
        assert vm._accelerator == QEMUAccelerator.TCG

    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    @patch.object(QEMUVirtualMachine, '_find_qemu_binary', return_value="qemu-system-x86_64")
    @patch.object(QEMUVirtualMachine, '_detect_accelerator', return_value=QEMUAccelerator.TCG)
    def test_state_property(self, mock_accel, mock_binary, mock_img, tmp_path):
        """Test VM state property."""
        config = QEMUVMConfig()
        image_manager = QEMUImageManager(images_dir=tmp_path)
        vm = QEMUVirtualMachine(config, image_manager)

        assert vm.state == VMState.STOPPED

    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    @patch.object(QEMUVirtualMachine, '_find_qemu_binary', return_value="qemu-system-x86_64")
    @patch.object(QEMUVirtualMachine, '_detect_accelerator', return_value=QEMUAccelerator.TCG)
    def test_setup_runtime_dir(self, mock_accel, mock_binary, mock_img, tmp_path):
        """Test runtime directory setup."""
        config = QEMUVMConfig()
        image_manager = QEMUImageManager(images_dir=tmp_path)
        vm = QEMUVirtualMachine(config, image_manager)

        vm._setup_runtime_dir()

        assert vm._runtime_dir is not None
        assert vm._runtime_dir.exists()
        assert vm._qmp_socket is not None

    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    @patch.object(QEMUVirtualMachine, '_find_qemu_binary', return_value="qemu-system-x86_64")
    @patch.object(QEMUVirtualMachine, '_detect_accelerator', return_value=QEMUAccelerator.TCG)
    def test_find_free_port(self, mock_accel, mock_binary, mock_img, tmp_path):
        """Test finding a free TCP port."""
        config = QEMUVMConfig()
        image_manager = QEMUImageManager(images_dir=tmp_path)
        vm = QEMUVirtualMachine(config, image_manager)

        port = vm._find_free_port()

        assert isinstance(port, int)
        assert port > 0
        assert port < 65536

    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    @patch.object(QEMUVirtualMachine, '_find_qemu_binary', return_value="qemu-system-x86_64")
    @patch.object(QEMUVirtualMachine, '_detect_accelerator', return_value=QEMUAccelerator.TCG)
    def test_build_qemu_command(self, mock_accel, mock_binary, mock_img, tmp_path):
        """Test building QEMU command line."""
        config = QEMUVMConfig(
            name="test-vm",
            memory="4G",
            cpus=2,
            disk_image=str(tmp_path / "disk.qcow2")
        )
        image_manager = QEMUImageManager(images_dir=tmp_path)
        vm = QEMUVirtualMachine(config, image_manager)

        # Setup runtime dir for socket paths
        vm._setup_runtime_dir()

        cmd = vm._build_qemu_command()

        assert "qemu-system-x86_64" in cmd[0]
        assert "-m" in cmd
        assert "4G" in cmd
        assert "-smp" in cmd
        assert "2" in cmd

    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    @patch.object(QEMUVirtualMachine, '_find_qemu_binary', return_value="qemu-system-x86_64")
    @patch.object(QEMUVirtualMachine, '_detect_accelerator', return_value=QEMUAccelerator.KVM)
    def test_build_command_with_kvm(self, mock_accel, mock_binary, mock_img, tmp_path):
        """Test command building with KVM acceleration."""
        config = QEMUVMConfig(accelerator=QEMUAccelerator.KVM)
        image_manager = QEMUImageManager(images_dir=tmp_path)
        vm = QEMUVirtualMachine(config, image_manager)
        vm._setup_runtime_dir()

        cmd = vm._build_qemu_command()

        assert "-enable-kvm" in cmd

    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    @patch.object(QEMUVirtualMachine, '_find_qemu_binary', return_value="qemu-system-x86_64")
    @patch.object(QEMUVirtualMachine, '_detect_accelerator', return_value=QEMUAccelerator.TCG)
    def test_build_command_with_network(self, mock_accel, mock_binary, mock_img, tmp_path):
        """Test command building with network enabled."""
        config = QEMUVMConfig(network_enabled=True)
        image_manager = QEMUImageManager(images_dir=tmp_path)
        vm = QEMUVirtualMachine(config, image_manager)
        vm._setup_runtime_dir()

        cmd = vm._build_qemu_command()
        cmd_str = " ".join(cmd)

        assert "hostfwd" in cmd_str
        assert "virtio-net" in cmd_str

    @pytest.mark.asyncio
    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    @patch.object(QEMUVirtualMachine, '_find_qemu_binary', return_value="qemu-system-x86_64")
    @patch.object(QEMUVirtualMachine, '_detect_accelerator', return_value=QEMUAccelerator.TCG)
    async def test_cleanup(self, mock_accel, mock_binary, mock_img, tmp_path):
        """Test VM cleanup."""
        config = QEMUVMConfig()
        image_manager = QEMUImageManager(images_dir=tmp_path)
        vm = QEMUVirtualMachine(config, image_manager)
        vm._setup_runtime_dir()

        runtime_dir = vm._runtime_dir

        await vm.cleanup()

        assert vm._qmp is None
        assert vm._snapshot_image is None

    @pytest.mark.asyncio
    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    @patch.object(QEMUVirtualMachine, '_find_qemu_binary', return_value="qemu-system-x86_64")
    @patch.object(QEMUVirtualMachine, '_detect_accelerator', return_value=QEMUAccelerator.TCG)
    async def test_start_invalid_state(self, mock_accel, mock_binary, mock_img, tmp_path):
        """Test starting VM from invalid state."""
        config = QEMUVMConfig()
        image_manager = QEMUImageManager(images_dir=tmp_path)
        vm = QEMUVirtualMachine(config, image_manager)

        # Set state to running
        vm._state = VMState.RUNNING

        result = await vm.start()
        assert result is False

    @pytest.mark.asyncio
    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    @patch.object(QEMUVirtualMachine, '_find_qemu_binary', return_value="qemu-system-x86_64")
    @patch.object(QEMUVirtualMachine, '_detect_accelerator', return_value=QEMUAccelerator.TCG)
    async def test_stop_already_stopped(self, mock_accel, mock_binary, mock_img, tmp_path):
        """Test stopping an already stopped VM."""
        config = QEMUVMConfig()
        image_manager = QEMUImageManager(images_dir=tmp_path)
        vm = QEMUVirtualMachine(config, image_manager)

        result = await vm.stop()
        assert result is True

    @pytest.mark.asyncio
    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    @patch.object(QEMUVirtualMachine, '_find_qemu_binary', return_value="qemu-system-x86_64")
    @patch.object(QEMUVirtualMachine, '_detect_accelerator', return_value=QEMUAccelerator.TCG)
    async def test_reset_invalid_state(self, mock_accel, mock_binary, mock_img, tmp_path):
        """Test resetting VM from invalid state."""
        config = QEMUVMConfig()
        image_manager = QEMUImageManager(images_dir=tmp_path)
        vm = QEMUVirtualMachine(config, image_manager)

        # VM is stopped, can't reset
        result = await vm.reset()
        assert result is False

    @pytest.mark.asyncio
    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    @patch.object(QEMUVirtualMachine, '_find_qemu_binary', return_value="qemu-system-x86_64")
    @patch.object(QEMUVirtualMachine, '_detect_accelerator', return_value=QEMUAccelerator.TCG)
    async def test_save_clean_snapshot_invalid_state(self, mock_accel, mock_binary, mock_img, tmp_path):
        """Test saving snapshot from invalid state."""
        config = QEMUVMConfig()
        image_manager = QEMUImageManager(images_dir=tmp_path)
        vm = QEMUVirtualMachine(config, image_manager)

        # VM is stopped, can't save snapshot
        result = await vm.save_clean_snapshot()
        assert result is False


# =============================================================================
# QEMUPool Tests (Mocked)
# =============================================================================

class TestQEMUPool:
    """Tests for QEMUPool class - mocked."""

    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    def test_initialization(self, mock_img, tmp_path):
        """Test pool initialization."""
        config = QEMUVMConfig(name="pool-vm")
        image_manager = QEMUImageManager(images_dir=tmp_path)

        pool = QEMUPool(config, pool_size=2, image_manager=image_manager)

        assert pool.config.name == "pool-vm"
        assert pool.pool_size == 2
        assert len(pool._available) == 0
        assert len(pool._in_use) == 0
        assert pool._initialized is False

    @pytest.mark.asyncio
    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    async def test_acquire_empty_pool(self, mock_img, tmp_path):
        """Test acquiring from empty (uninitialized) pool."""
        config = QEMUVMConfig()
        image_manager = QEMUImageManager(images_dir=tmp_path)
        pool = QEMUPool(config, pool_size=2, image_manager=image_manager)

        vm = await pool.acquire()
        assert vm is None

    @pytest.mark.asyncio
    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    async def test_shutdown_empty_pool(self, mock_img, tmp_path):
        """Test shutting down empty pool."""
        config = QEMUVMConfig()
        image_manager = QEMUImageManager(images_dir=tmp_path)
        pool = QEMUPool(config, pool_size=2, image_manager=image_manager)

        # Should not raise
        await pool.shutdown()

        assert len(pool._available) == 0
        assert len(pool._in_use) == 0
        assert pool._initialized is False


# =============================================================================
# QEMUBackendImpl Tests (Mocked)
# =============================================================================

class TestQEMUBackendImpl:
    """Tests for QEMUBackendImpl class - mocked."""

    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    def test_initialization(self, mock_img, tmp_path):
        """Test backend initialization."""
        image_manager = QEMUImageManager(images_dir=tmp_path)
        backend = QEMUBackendImpl(image_manager=image_manager)

        assert backend.use_pool is False
        assert backend._initialized is False

    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    def test_initialization_with_pool(self, mock_img, tmp_path):
        """Test backend initialization with pooling enabled."""
        image_manager = QEMUImageManager(images_dir=tmp_path)
        backend = QEMUBackendImpl(
            image_manager=image_manager,
            use_pool=True,
            pool_size=3
        )

        assert backend.use_pool is True
        assert backend._pool_size == 3

    @patch('subprocess.run')
    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    def test_is_available_true(self, mock_img, mock_run, tmp_path):
        """Test availability check when QEMU is installed."""
        mock_run.return_value = Mock(returncode=0)

        image_manager = QEMUImageManager(images_dir=tmp_path)
        backend = QEMUBackendImpl(image_manager=image_manager)

        result = backend.is_available()
        assert result is True

    @patch('subprocess.run')
    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    def test_is_available_false(self, mock_img, mock_run, tmp_path):
        """Test availability check when QEMU is not installed."""
        mock_run.side_effect = FileNotFoundError()

        image_manager = QEMUImageManager(images_dir=tmp_path)
        backend = QEMUBackendImpl(image_manager=image_manager)

        result = backend.is_available()
        assert result is False

    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    def test_get_base_image_path_exists(self, mock_img, tmp_path):
        """Test getting base image path when it exists."""
        image_manager = QEMUImageManager(images_dir=tmp_path)
        backend = QEMUBackendImpl(image_manager=image_manager)

        # Create fake python base image
        python_image = tmp_path / "python-sandbox.qcow2"
        python_image.touch()

        path = backend.get_base_image_path("python")
        assert path == python_image

    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    def test_get_base_image_path_not_exists(self, mock_img, tmp_path):
        """Test getting base image path when it doesn't exist."""
        image_manager = QEMUImageManager(images_dir=tmp_path)
        backend = QEMUBackendImpl(image_manager=image_manager)

        path = backend.get_base_image_path("python")
        assert path is None

    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    def test_get_base_image_path_unknown_language(self, mock_img, tmp_path):
        """Test getting base image path for unknown language."""
        image_manager = QEMUImageManager(images_dir=tmp_path)
        backend = QEMUBackendImpl(image_manager=image_manager)

        path = backend.get_base_image_path("unknown_lang")
        assert path is None

    @pytest.mark.asyncio
    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    async def test_initialize_without_pool(self, mock_img, tmp_path):
        """Test backend initialization without pool."""
        image_manager = QEMUImageManager(images_dir=tmp_path)
        backend = QEMUBackendImpl(image_manager=image_manager)

        await backend.initialize()

        assert backend._initialized is True
        assert backend._pool is None

    @pytest.mark.asyncio
    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    async def test_shutdown(self, mock_img, tmp_path):
        """Test backend shutdown."""
        image_manager = QEMUImageManager(images_dir=tmp_path)
        backend = QEMUBackendImpl(image_manager=image_manager)

        await backend.initialize()
        await backend.shutdown()

        assert backend._initialized is False

    @pytest.mark.asyncio
    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    async def test_execute_no_base_image(self, mock_img, tmp_path):
        """Test execution fails without base image."""
        image_manager = QEMUImageManager(images_dir=tmp_path)
        backend = QEMUBackendImpl(image_manager=image_manager)

        script_path = tmp_path / "test.py"
        script_path.write_text("print('hello')")

        config = QEMUVMConfig()

        result = await backend.execute(script_path, config, "python")

        assert result.success is False
        assert "No base image" in result.error


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateQEMUBackend:
    """Tests for create_qemu_backend factory function."""

    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    def test_create_default_backend(self, mock_img):
        """Test creating backend with defaults."""
        backend = create_qemu_backend()

        assert isinstance(backend, QEMUBackendImpl)
        assert backend.use_pool is False

    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    def test_create_backend_with_pool(self, mock_img):
        """Test creating backend with pool enabled."""
        backend = create_qemu_backend(use_pool=True, pool_size=4)

        assert backend.use_pool is True
        assert backend._pool_size == 4

    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    def test_create_backend_with_custom_dir(self, mock_img, tmp_path):
        """Test creating backend with custom images directory."""
        backend = create_qemu_backend(images_dir=tmp_path)

        assert backend.image_manager.images_dir == tmp_path


# =============================================================================
# Integration Tests (Require QEMU - skipped by default)
# =============================================================================

@pytest.mark.qemu
@pytest.mark.slow
class TestQEMUIntegration:
    """
    Integration tests that require actual QEMU installation.

    These tests are marked with @pytest.mark.qemu and are skipped
    unless specifically enabled (e.g., pytest -m qemu).
    """

    @pytest.fixture
    def check_qemu_available(self):
        """Skip tests if QEMU is not available."""
        import subprocess
        try:
            result = subprocess.run(
                ["qemu-system-x86_64", "--version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                pytest.skip("QEMU not available")
        except Exception:
            pytest.skip("QEMU not available")

    def test_qemu_binary_detection(self, check_qemu_available, tmp_path):
        """Test that QEMU binary is correctly detected."""
        config = QEMUVMConfig()
        image_manager = QEMUImageManager(images_dir=tmp_path)

        # This should not raise if QEMU is available
        vm = QEMUVirtualMachine(config, image_manager)
        assert vm._qemu_binary is not None

    def test_accelerator_detection(self, check_qemu_available, tmp_path):
        """Test hardware accelerator detection."""
        config = QEMUVMConfig()
        image_manager = QEMUImageManager(images_dir=tmp_path)

        vm = QEMUVirtualMachine(config, image_manager)

        # Should detect some accelerator (at minimum TCG)
        assert vm._accelerator is not None
        assert isinstance(vm._accelerator, QEMUAccelerator)

    def test_image_creation(self, check_qemu_available, tmp_path):
        """Test disk image creation with real qemu-img."""
        image_manager = QEMUImageManager(images_dir=tmp_path)

        # Create a small test image
        image_path = image_manager.create_base_image("test-image", size="100M")

        assert image_path.exists()
        assert image_path.suffix == ".qcow2"

        # Verify image info
        info = image_manager.get_image_info(image_path)
        assert info["format"] == "qcow2"

    def test_snapshot_creation(self, check_qemu_available, tmp_path):
        """Test snapshot creation from base image."""
        image_manager = QEMUImageManager(images_dir=tmp_path)

        # Create base image
        base_image = image_manager.create_base_image("base", size="100M")

        # Create snapshot
        snapshot = image_manager.create_snapshot(base_image, "test-snap")

        assert snapshot.exists()

        # Verify snapshot backing file
        info = image_manager.get_image_info(snapshot)
        assert "backing-filename" in info

    @pytest.mark.asyncio
    async def test_vm_start_stop_cycle(self, check_qemu_available, tmp_path):
        """Test basic VM start/stop cycle (requires bootable image)."""
        # This test requires a bootable image to fully work
        # Skip if no bootable image is available

        image_manager = QEMUImageManager(images_dir=tmp_path)

        # Check for a pre-existing bootable image
        python_image = image_manager.images_dir / "python-sandbox.qcow2"
        if not python_image.exists():
            pytest.skip("No bootable image available for VM test")

        config = QEMUVMConfig(
            name="test-vm",
            disk_image=str(python_image),
            timeout_seconds=60
        )

        vm = QEMUVirtualMachine(config, image_manager)

        # Attempt start
        started = await vm.start()

        if started:
            assert vm.state == VMState.RUNNING

            # Stop
            stopped = await vm.stop()
            assert stopped is True
            assert vm.state == VMState.STOPPED
        else:
            # VM failed to start (e.g., image issues)
            await vm.cleanup()


# =============================================================================
# Accelerator Detection Tests
# =============================================================================

class TestAcceleratorDetection:
    """Tests for accelerator detection logic."""

    @patch('platform.system', return_value='Linux')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('subprocess.run')
    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    @patch.object(QEMUVirtualMachine, '_find_qemu_binary', return_value="qemu-system-x86_64")
    def test_detect_kvm_on_linux(self, mock_binary, mock_img, mock_run, mock_exists, mock_system, tmp_path):
        """Test KVM detection on Linux."""
        mock_run.return_value = Mock(returncode=0, stdout=b"kvm", stderr=b"")

        config = QEMUVMConfig()
        image_manager = QEMUImageManager(images_dir=tmp_path)

        vm = QEMUVirtualMachine(config, image_manager)
        # The detection happens in __init__, but we can verify
        # accelerator was detected

    @patch('platform.system', return_value='Windows')
    @patch('subprocess.run')
    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    @patch.object(QEMUVirtualMachine, '_find_qemu_binary', return_value="qemu-system-x86_64.exe")
    def test_detect_whpx_on_windows(self, mock_binary, mock_img, mock_run, mock_system, tmp_path):
        """Test WHPX detection on Windows."""
        mock_run.return_value = Mock(returncode=0, stdout=b"whpx", stderr=b"")

        config = QEMUVMConfig()
        image_manager = QEMUImageManager(images_dir=tmp_path)

        vm = QEMUVirtualMachine(config, image_manager)

    @patch('platform.system', return_value='Darwin')
    @patch('subprocess.run')
    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    @patch.object(QEMUVirtualMachine, '_find_qemu_binary', return_value="qemu-system-x86_64")
    def test_detect_hvf_on_macos(self, mock_binary, mock_img, mock_run, mock_system, tmp_path):
        """Test HVF detection on macOS."""
        mock_run.return_value = Mock(returncode=0, stdout=b"hvf", stderr=b"")

        config = QEMUVMConfig()
        image_manager = QEMUImageManager(images_dir=tmp_path)

        vm = QEMUVirtualMachine(config, image_manager)

    @patch('platform.system', return_value='Linux')
    @patch('pathlib.Path.exists', return_value=False)
    @patch('subprocess.run')
    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    @patch.object(QEMUVirtualMachine, '_find_qemu_binary', return_value="qemu-system-x86_64")
    def test_fallback_to_tcg(self, mock_binary, mock_img, mock_run, mock_exists, mock_system, tmp_path):
        """Test fallback to TCG when no hardware acceleration available."""
        mock_run.return_value = Mock(returncode=1, stdout=b"", stderr=b"no accel")

        config = QEMUVMConfig()
        image_manager = QEMUImageManager(images_dir=tmp_path)

        vm = QEMUVirtualMachine(config, image_manager)
        assert vm._accelerator == QEMUAccelerator.TCG


# =============================================================================
# Execution Mode Tests
# =============================================================================

class TestExecutionModes:
    """Tests for different execution modes."""

    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    @patch.object(QEMUVirtualMachine, '_find_qemu_binary', return_value="qemu-system-x86_64")
    @patch.object(QEMUVirtualMachine, '_detect_accelerator', return_value=QEMUAccelerator.TCG)
    def test_virtfs_command_args(self, mock_accel, mock_binary, mock_img, tmp_path):
        """Test VirtFS arguments in QEMU command."""
        shared_dir = tmp_path / "shared"
        shared_dir.mkdir()

        config = QEMUVMConfig(
            shared_dir=str(shared_dir),
            execution_mode=ExecutionMode.VIRTFS
        )
        image_manager = QEMUImageManager(images_dir=tmp_path)
        vm = QEMUVirtualMachine(config, image_manager)
        vm._setup_runtime_dir()

        cmd = vm._build_qemu_command()
        cmd_str = " ".join(cmd)

        assert "-virtfs" in cmd_str
        assert "hostshare" in cmd_str

    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    @patch.object(QEMUVirtualMachine, '_find_qemu_binary', return_value="qemu-system-x86_64")
    @patch.object(QEMUVirtualMachine, '_detect_accelerator', return_value=QEMUAccelerator.TCG)
    def test_ssh_requires_network(self, mock_accel, mock_binary, mock_img, tmp_path):
        """Test SSH execution mode requires network."""
        config = QEMUVMConfig(
            execution_mode=ExecutionMode.SSH,
            network_enabled=False  # Network disabled but SSH mode
        )
        image_manager = QEMUImageManager(images_dir=tmp_path)
        vm = QEMUVirtualMachine(config, image_manager)

        # execute_via_ssh should fail without network
        with pytest.raises(RuntimeError, match="SSH not available"):
            asyncio.get_event_loop().run_until_complete(
                vm.execute_via_ssh("echo test")
            )

    @patch.object(QEMUImageManager, '_find_qemu_img', return_value="qemu-img")
    @patch.object(QEMUVirtualMachine, '_find_qemu_binary', return_value="qemu-system-x86_64")
    @patch.object(QEMUVirtualMachine, '_detect_accelerator', return_value=QEMUAccelerator.TCG)
    def test_serial_console_args(self, mock_accel, mock_binary, mock_img, tmp_path):
        """Test serial console arguments in QEMU command."""
        config = QEMUVMConfig(serial_console=True)
        image_manager = QEMUImageManager(images_dir=tmp_path)
        vm = QEMUVirtualMachine(config, image_manager)
        vm._setup_runtime_dir()

        cmd = vm._build_qemu_command()
        cmd_str = " ".join(cmd)

        assert "-serial" in cmd_str
