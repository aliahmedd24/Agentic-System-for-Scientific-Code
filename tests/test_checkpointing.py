"""
Tests for Checkpoint Manager - Pipeline state persistence.
"""

import pytest
from datetime import datetime
from pathlib import Path


# ============================================================================
# CheckpointStage Enum Tests
# ============================================================================

class TestCheckpointStage:
    """Tests for CheckpointStage enum."""

    def test_all_stages_exist(self):
        """Test all expected stages are defined."""
        from core.checkpointing import CheckpointStage

        expected_stages = [
            "INITIALIZED", "PAPER_PARSED", "REPO_ANALYZED",
            "CONCEPTS_MAPPED", "CODE_GENERATED", "TESTS_EXECUTED", "COMPLETED"
        ]

        for stage_name in expected_stages:
            assert hasattr(CheckpointStage, stage_name), f"Missing stage: {stage_name}"

    def test_stage_values(self):
        """Test stage values."""
        from core.checkpointing import CheckpointStage

        assert CheckpointStage.INITIALIZED.value == "initialized"
        assert CheckpointStage.PAPER_PARSED.value == "paper_parsed"
        assert CheckpointStage.COMPLETED.value == "completed"


# ============================================================================
# CheckpointMetadata Tests
# ============================================================================

class TestCheckpointMetadata:
    """Tests for CheckpointMetadata model."""

    def test_metadata_creation(self):
        """Test creating metadata."""
        from core.checkpointing import CheckpointMetadata, CheckpointStage

        metadata = CheckpointMetadata(
            checkpoint_id="ckpt_test_abc123",
            stage=CheckpointStage.PAPER_PARSED,
            created_at=datetime.now(),
            paper_source="2301.00001",
            repo_url="https://github.com/test/repo",
            data_size_bytes=1024,
            is_compressed=True
        )

        assert metadata.checkpoint_id == "ckpt_test_abc123"
        assert metadata.stage == CheckpointStage.PAPER_PARSED
        assert metadata.data_size_bytes == 1024
        assert metadata.version == "1.0"  # Default

    def test_metadata_validation(self):
        """Test metadata validation."""
        from core.checkpointing import CheckpointMetadata, CheckpointStage
        from pydantic import ValidationError

        # Missing required fields
        with pytest.raises(ValidationError):
            CheckpointMetadata(
                checkpoint_id="test"
                # Missing other required fields
            )


# ============================================================================
# Checkpoint Model Tests
# ============================================================================

class TestCheckpoint:
    """Tests for Checkpoint model."""

    def test_checkpoint_creation(self):
        """Test creating a checkpoint."""
        from core.checkpointing import Checkpoint, CheckpointMetadata, CheckpointStage

        metadata = CheckpointMetadata(
            checkpoint_id="ckpt_test",
            stage=CheckpointStage.INITIALIZED,
            created_at=datetime.now(),
            paper_source="test.pdf",
            repo_url="https://github.com/test/repo",
            data_size_bytes=0,
            is_compressed=True
        )

        checkpoint = Checkpoint(metadata=metadata)

        assert checkpoint.metadata.checkpoint_id == "ckpt_test"
        assert checkpoint.paper_data is None
        assert checkpoint.repo_data is None
        assert checkpoint.errors == []

    def test_checkpoint_with_data(self, sample_checkpoint_data):
        """Test checkpoint with data."""
        from core.checkpointing import Checkpoint, CheckpointMetadata, CheckpointStage

        metadata = CheckpointMetadata(
            checkpoint_id="ckpt_test",
            stage=CheckpointStage.PAPER_PARSED,
            created_at=datetime.now(),
            paper_source="test.pdf",
            repo_url="https://github.com/test/repo",
            data_size_bytes=1024,
            is_compressed=True
        )

        checkpoint = Checkpoint(
            metadata=metadata,
            paper_data=sample_checkpoint_data["paper_data"],
            repo_data=sample_checkpoint_data["repo_data"],
            mappings=sample_checkpoint_data["mappings"]
        )

        assert checkpoint.paper_data is not None
        assert checkpoint.paper_data["title"] == "Test Paper on Neural Networks"


# ============================================================================
# CheckpointManager Initialization Tests
# ============================================================================

class TestCheckpointManagerInit:
    """Tests for CheckpointManager initialization."""

    def test_manager_creation(self, tmp_path):
        """Test creating manager."""
        from core.checkpointing import CheckpointManager

        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path / "checkpoints")
        )

        assert manager.max_checkpoints == 10  # Default
        assert manager.compress is True  # Default

    def test_manager_creates_directory(self, tmp_path):
        """Test manager creates directory."""
        from core.checkpointing import CheckpointManager

        checkpoint_dir = tmp_path / "new_checkpoints"
        manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))

        assert checkpoint_dir.exists()

    def test_manager_custom_settings(self, tmp_path):
        """Test manager with custom settings."""
        from core.checkpointing import CheckpointManager

        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            max_checkpoints=5,
            compress=False
        )

        assert manager.max_checkpoints == 5
        assert manager.compress is False


# ============================================================================
# Checkpoint ID Generation Tests
# ============================================================================

class TestCheckpointIDGeneration:
    """Tests for checkpoint ID generation."""

    def test_generate_checkpoint_id_unique(self, checkpoint_manager):
        """Test checkpoint IDs are unique."""
        from core.checkpointing import CheckpointStage
        import time

        id1 = checkpoint_manager._generate_checkpoint_id(
            "paper1.pdf", "https://github.com/test/repo", CheckpointStage.PAPER_PARSED
        )
        time.sleep(0.01)  # Small delay to ensure different timestamp
        id2 = checkpoint_manager._generate_checkpoint_id(
            "paper1.pdf", "https://github.com/test/repo", CheckpointStage.PAPER_PARSED
        )

        assert id1 != id2  # Different timestamps

    def test_get_pipeline_id_consistent(self, checkpoint_manager):
        """Test pipeline ID is consistent for same inputs."""
        id1 = checkpoint_manager._get_pipeline_id("paper.pdf", "https://github.com/test/repo")
        id2 = checkpoint_manager._get_pipeline_id("paper.pdf", "https://github.com/test/repo")

        assert id1 == id2

    def test_get_pipeline_id_different_inputs(self, checkpoint_manager):
        """Test pipeline ID differs for different inputs."""
        id1 = checkpoint_manager._get_pipeline_id("paper1.pdf", "https://github.com/test/repo")
        id2 = checkpoint_manager._get_pipeline_id("paper2.pdf", "https://github.com/test/repo")

        assert id1 != id2


# ============================================================================
# Save/Load Tests
# ============================================================================

class TestCheckpointSaveLoad:
    """Tests for checkpoint save and load."""

    def test_save_checkpoint(self, checkpoint_manager, sample_checkpoint_data):
        """Test saving a checkpoint."""
        from core.checkpointing import CheckpointStage

        checkpoint_id = checkpoint_manager.save(
            stage=CheckpointStage.PAPER_PARSED,
            paper_source="test.pdf",
            repo_url="https://github.com/test/repo",
            paper_data=sample_checkpoint_data["paper_data"]
        )

        assert checkpoint_id is not None
        assert checkpoint_id.startswith("ckpt_paper_parsed")

    def test_load_checkpoint(self, checkpoint_manager, sample_checkpoint_data):
        """Test loading a saved checkpoint."""
        from core.checkpointing import CheckpointStage

        # Save first
        checkpoint_id = checkpoint_manager.save(
            stage=CheckpointStage.PAPER_PARSED,
            paper_source="test.pdf",
            repo_url="https://github.com/test/repo",
            paper_data=sample_checkpoint_data["paper_data"]
        )

        # Load
        checkpoint = checkpoint_manager.load(checkpoint_id)

        assert checkpoint is not None
        assert checkpoint.paper_data is not None
        assert checkpoint.paper_data["title"] == sample_checkpoint_data["paper_data"]["title"]

    def test_load_nonexistent_checkpoint(self, checkpoint_manager):
        """Test loading non-existent checkpoint."""
        checkpoint = checkpoint_manager.load("nonexistent_checkpoint_id")

        assert checkpoint is None

    def test_save_checkpoint_compressed(self, tmp_path, sample_checkpoint_data):
        """Test checkpoint is compressed."""
        from core.checkpointing import CheckpointManager, CheckpointStage

        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            compress=True
        )

        checkpoint_id = manager.save(
            stage=CheckpointStage.PAPER_PARSED,
            paper_source="test.pdf",
            repo_url="https://github.com/test/repo",
            paper_data=sample_checkpoint_data["paper_data"]
        )

        checkpoint = manager.load(checkpoint_id)
        assert checkpoint is not None
        assert checkpoint.metadata.is_compressed is True

    def test_save_checkpoint_uncompressed(self, tmp_path, sample_checkpoint_data):
        """Test saving uncompressed checkpoint."""
        from core.checkpointing import CheckpointManager, CheckpointStage

        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            compress=False
        )

        checkpoint_id = manager.save(
            stage=CheckpointStage.PAPER_PARSED,
            paper_source="test.pdf",
            repo_url="https://github.com/test/repo",
            paper_data=sample_checkpoint_data["paper_data"]
        )

        checkpoint = manager.load(checkpoint_id)
        assert checkpoint is not None
        assert checkpoint.metadata.is_compressed is False


# ============================================================================
# Get Latest/By Stage Tests
# ============================================================================

class TestCheckpointRetrieval:
    """Tests for checkpoint retrieval methods."""

    def test_get_latest(self, checkpoint_manager, sample_checkpoint_data):
        """Test getting latest checkpoint."""
        from core.checkpointing import CheckpointStage
        import time

        # Save multiple checkpoints
        checkpoint_manager.save(
            stage=CheckpointStage.PAPER_PARSED,
            paper_source="test.pdf",
            repo_url="https://github.com/test/repo",
            paper_data=sample_checkpoint_data["paper_data"]
        )

        time.sleep(0.01)

        checkpoint_manager.save(
            stage=CheckpointStage.REPO_ANALYZED,
            paper_source="test.pdf",
            repo_url="https://github.com/test/repo",
            paper_data=sample_checkpoint_data["paper_data"],
            repo_data=sample_checkpoint_data["repo_data"]
        )

        # Get latest
        latest = checkpoint_manager.get_latest("test.pdf", "https://github.com/test/repo")

        assert latest is not None
        assert latest.metadata.stage == CheckpointStage.REPO_ANALYZED

    def test_get_latest_no_checkpoints(self, checkpoint_manager):
        """Test get_latest with no checkpoints."""
        latest = checkpoint_manager.get_latest("nonexistent.pdf", "https://github.com/x/y")

        assert latest is None

    def test_get_checkpoint_at_stage(self, checkpoint_manager, sample_checkpoint_data):
        """Test getting checkpoint at specific stage."""
        from core.checkpointing import CheckpointStage

        # Save at specific stage
        checkpoint_manager.save(
            stage=CheckpointStage.PAPER_PARSED,
            paper_source="test.pdf",
            repo_url="https://github.com/test/repo",
            paper_data=sample_checkpoint_data["paper_data"]
        )

        # Get at stage
        checkpoint = checkpoint_manager.get_checkpoint_at_stage(
            "test.pdf",
            "https://github.com/test/repo",
            CheckpointStage.PAPER_PARSED
        )

        assert checkpoint is not None
        assert checkpoint.metadata.stage == CheckpointStage.PAPER_PARSED

    def test_get_checkpoint_at_stage_not_found(self, checkpoint_manager):
        """Test getting checkpoint at non-existent stage."""
        from core.checkpointing import CheckpointStage

        checkpoint = checkpoint_manager.get_checkpoint_at_stage(
            "test.pdf",
            "https://github.com/test/repo",
            CheckpointStage.COMPLETED
        )

        assert checkpoint is None


# ============================================================================
# List Checkpoints Tests
# ============================================================================

class TestListCheckpoints:
    """Tests for listing checkpoints."""

    def test_list_checkpoints(self, checkpoint_manager, sample_checkpoint_data):
        """Test listing checkpoints."""
        from core.checkpointing import CheckpointStage

        # Save some checkpoints
        checkpoint_manager.save(
            stage=CheckpointStage.PAPER_PARSED,
            paper_source="test.pdf",
            repo_url="https://github.com/test/repo",
            paper_data=sample_checkpoint_data["paper_data"]
        )

        checkpoints = checkpoint_manager.list_checkpoints(
            "test.pdf",
            "https://github.com/test/repo"
        )

        assert len(checkpoints) >= 1

    def test_list_checkpoints_empty(self, checkpoint_manager):
        """Test listing checkpoints with none saved."""
        checkpoints = checkpoint_manager.list_checkpoints(
            "nonexistent.pdf",
            "https://github.com/nonexistent/repo"
        )

        assert checkpoints == []


# ============================================================================
# Delete Tests
# ============================================================================

class TestDeleteCheckpoints:
    """Tests for deleting checkpoints."""

    def test_delete_checkpoint(self, checkpoint_manager, sample_checkpoint_data):
        """Test deleting a checkpoint."""
        from core.checkpointing import CheckpointStage

        # Save first
        checkpoint_id = checkpoint_manager.save(
            stage=CheckpointStage.PAPER_PARSED,
            paper_source="test.pdf",
            repo_url="https://github.com/test/repo",
            paper_data=sample_checkpoint_data["paper_data"]
        )

        # Delete
        result = checkpoint_manager.delete(checkpoint_id)

        assert result is True

        # Verify deleted
        checkpoint = checkpoint_manager.load(checkpoint_id)
        assert checkpoint is None

    def test_delete_nonexistent_checkpoint(self, checkpoint_manager):
        """Test deleting non-existent checkpoint."""
        result = checkpoint_manager.delete("nonexistent_id")

        assert result is False

    def test_delete_pipeline(self, checkpoint_manager, sample_checkpoint_data):
        """Test deleting all checkpoints for a pipeline."""
        from core.checkpointing import CheckpointStage

        # Save multiple checkpoints
        checkpoint_manager.save(
            stage=CheckpointStage.PAPER_PARSED,
            paper_source="test.pdf",
            repo_url="https://github.com/test/repo",
            paper_data=sample_checkpoint_data["paper_data"]
        )

        checkpoint_manager.save(
            stage=CheckpointStage.REPO_ANALYZED,
            paper_source="test.pdf",
            repo_url="https://github.com/test/repo",
            repo_data=sample_checkpoint_data["repo_data"]
        )

        # Delete pipeline
        count = checkpoint_manager.delete_pipeline("test.pdf", "https://github.com/test/repo")

        assert count >= 2

        # Verify deleted
        checkpoints = checkpoint_manager.list_checkpoints(
            "test.pdf",
            "https://github.com/test/repo"
        )
        assert len(checkpoints) == 0


# ============================================================================
# Cleanup Tests
# ============================================================================

class TestCheckpointCleanup:
    """Tests for automatic cleanup."""

    def test_cleanup_old_checkpoints(self, tmp_path, sample_checkpoint_data):
        """Test old checkpoints are cleaned up."""
        from core.checkpointing import CheckpointManager, CheckpointStage
        import time

        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            max_checkpoints=2
        )

        # Save more than max
        for i in range(4):
            manager.save(
                stage=CheckpointStage.PAPER_PARSED,
                paper_source="test.pdf",
                repo_url="https://github.com/test/repo",
                paper_data=sample_checkpoint_data["paper_data"]
            )
            time.sleep(0.01)  # Ensure different timestamps

        # Should only have max_checkpoints left
        checkpoints = manager.list_checkpoints(
            "test.pdf",
            "https://github.com/test/repo"
        )

        assert len(checkpoints) <= 2


# ============================================================================
# Storage Usage Tests
# ============================================================================

class TestStorageUsage:
    """Tests for storage usage statistics."""

    def test_get_storage_usage(self, checkpoint_manager, sample_checkpoint_data):
        """Test getting storage usage."""
        from core.checkpointing import CheckpointStage

        # Save a checkpoint
        checkpoint_manager.save(
            stage=CheckpointStage.PAPER_PARSED,
            paper_source="test.pdf",
            repo_url="https://github.com/test/repo",
            paper_data=sample_checkpoint_data["paper_data"]
        )

        usage = checkpoint_manager.get_storage_usage()

        assert "total_size_bytes" in usage
        assert "checkpoint_count" in usage
        assert "pipeline_count" in usage
        assert usage["checkpoint_count"] >= 1


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_save_with_errors(self, checkpoint_manager):
        """Test saving checkpoint with errors."""
        from core.checkpointing import CheckpointStage

        checkpoint_id = checkpoint_manager.save(
            stage=CheckpointStage.PAPER_PARSED,
            paper_source="test.pdf",
            repo_url="https://github.com/test/repo",
            errors=["Error 1", "Error 2"]
        )

        checkpoint = checkpoint_manager.load(checkpoint_id)
        assert checkpoint is not None
        assert len(checkpoint.errors) == 2
        assert "Error 1" in checkpoint.errors

    def test_save_empty_data(self, checkpoint_manager):
        """Test saving checkpoint with no data."""
        from core.checkpointing import CheckpointStage

        checkpoint_id = checkpoint_manager.save(
            stage=CheckpointStage.INITIALIZED,
            paper_source="test.pdf",
            repo_url="https://github.com/test/repo"
        )

        checkpoint = checkpoint_manager.load(checkpoint_id)
        assert checkpoint is not None
        assert checkpoint.paper_data is None
        assert checkpoint.repo_data is None

    def test_special_characters_in_source(self, checkpoint_manager, sample_checkpoint_data):
        """Test handling special characters in source paths."""
        from core.checkpointing import CheckpointStage

        checkpoint_id = checkpoint_manager.save(
            stage=CheckpointStage.PAPER_PARSED,
            paper_source="path/to/paper with spaces.pdf",
            repo_url="https://github.com/user/repo-name",
            paper_data=sample_checkpoint_data["paper_data"]
        )

        checkpoint = checkpoint_manager.load(checkpoint_id)
        assert checkpoint is not None
