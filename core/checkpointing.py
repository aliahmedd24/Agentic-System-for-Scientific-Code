"""
Checkpoint Manager - Provides pipeline checkpointing and recovery.

This module enables:
- Saving pipeline state at key stages
- Resuming from checkpoints after failures
- Managing checkpoint storage
- Automatic cleanup of old checkpoints
"""

import json
import pickle
import gzip
import shutil
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from enum import Enum
import hashlib

from pydantic import BaseModel, Field, ConfigDict


class CheckpointStage(Enum):
    """Pipeline stages that support checkpointing."""
    INITIALIZED = "initialized"
    PAPER_PARSED = "paper_parsed"
    REPO_ANALYZED = "repo_analyzed"
    CONCEPTS_MAPPED = "concepts_mapped"
    CODE_GENERATED = "code_generated"
    TESTS_EXECUTED = "tests_executed"
    COMPLETED = "completed"


class CheckpointMetadata(BaseModel):
    """Metadata for a checkpoint."""
    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str = Field(..., description="Unique checkpoint ID")
    stage: CheckpointStage = Field(..., description="Pipeline stage")
    created_at: datetime = Field(..., description="Creation timestamp")
    paper_source: str = Field(..., description="Paper source path/URL")
    repo_url: str = Field(..., description="Repository URL")
    data_size_bytes: int = Field(..., description="Data size in bytes")
    is_compressed: bool = Field(True, description="Whether data is compressed")
    version: str = Field("1.0", description="Checkpoint format version")


class Checkpoint(BaseModel):
    """A pipeline checkpoint."""
    model_config = ConfigDict(extra="forbid")

    metadata: CheckpointMetadata = Field(..., description="Checkpoint metadata")
    paper_data: Optional[Dict[str, Any]] = Field(None, description="Parsed paper data")
    repo_data: Optional[Dict[str, Any]] = Field(None, description="Analyzed repo data")
    mappings: Optional[List[Dict[str, Any]]] = Field(None, description="Concept mappings")
    code_results: Optional[List[Dict[str, Any]]] = Field(None, description="Code execution results")
    knowledge_graph_data: Optional[Dict[str, Any]] = Field(None, description="Knowledge graph data")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")


class CheckpointManager:
    """
    Manages pipeline checkpoints for recovery and resumption.

    Features:
    - Automatic checkpointing after each stage
    - Compressed storage (gzip)
    - Checkpoint listing and management
    - Automatic cleanup of old checkpoints
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        max_checkpoints: int = 10,
        compress: bool = True
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum checkpoints to keep per pipeline
            compress: Whether to compress checkpoint data
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.compress = compress

    def _generate_checkpoint_id(
        self,
        paper_source: str,
        repo_url: str,
        stage: CheckpointStage
    ) -> str:
        """Generate unique checkpoint ID."""
        # Create hash from inputs
        content = f"{paper_source}|{repo_url}|{stage.value}|{datetime.now().isoformat()}"
        hash_str = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"ckpt_{stage.value}_{hash_str}"

    def _get_pipeline_id(self, paper_source: str, repo_url: str) -> str:
        """Generate pipeline ID from sources."""
        content = f"{paper_source}|{repo_url}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def save(
        self,
        stage: CheckpointStage,
        paper_source: str,
        repo_url: str,
        paper_data: Dict[str, Any] = None,
        repo_data: Dict[str, Any] = None,
        mappings: List[Dict[str, Any]] = None,
        code_results: List[Dict[str, Any]] = None,
        knowledge_graph_data: Dict[str, Any] = None,
        errors: List[str] = None
    ) -> str:
        """
        Save a checkpoint.

        Args:
            stage: Current pipeline stage
            paper_source: Paper source (arXiv ID, URL, or path)
            repo_url: Repository URL
            paper_data: Parsed paper data
            repo_data: Analyzed repo data
            mappings: Concept-code mappings
            code_results: Test execution results
            knowledge_graph_data: Serialized knowledge graph
            errors: Any errors encountered

        Returns:
            Checkpoint ID
        """
        checkpoint_id = self._generate_checkpoint_id(paper_source, repo_url, stage)
        pipeline_id = self._get_pipeline_id(paper_source, repo_url)

        # Prepare checkpoint data
        checkpoint_data = {
            "paper_data": paper_data,
            "repo_data": repo_data,
            "mappings": mappings,
            "code_results": code_results,
            "knowledge_graph_data": knowledge_graph_data,
            "errors": errors or []
        }

        # Serialize data
        if self.compress:
            data_bytes = gzip.compress(pickle.dumps(checkpoint_data))
        else:
            data_bytes = pickle.dumps(checkpoint_data)

        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            stage=stage,
            created_at=datetime.now(),
            paper_source=paper_source,
            repo_url=repo_url,
            data_size_bytes=len(data_bytes),
            is_compressed=self.compress
        )

        # Create pipeline directory
        pipeline_dir = self.checkpoint_dir / pipeline_id
        pipeline_dir.mkdir(exist_ok=True)

        # Save data file
        data_path = pipeline_dir / f"{checkpoint_id}.data"
        data_path.write_bytes(data_bytes)

        # Save metadata file
        meta_path = pipeline_dir / f"{checkpoint_id}.meta.json"
        meta_path.write_text(json.dumps(metadata.model_dump(mode='json'), indent=2))

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(pipeline_dir)

        return checkpoint_id

    def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """
        Load a checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint ID to load

        Returns:
            Checkpoint object, or None if not found
        """
        # Search for checkpoint in all pipeline directories
        for pipeline_dir in self.checkpoint_dir.iterdir():
            if not pipeline_dir.is_dir():
                continue

            meta_path = pipeline_dir / f"{checkpoint_id}.meta.json"
            data_path = pipeline_dir / f"{checkpoint_id}.data"

            if meta_path.exists() and data_path.exists():
                try:
                    # Load metadata
                    metadata = CheckpointMetadata.model_validate(
                        json.loads(meta_path.read_text())
                    )

                    # Load data
                    data_bytes = data_path.read_bytes()
                    if metadata.is_compressed:
                        data = pickle.loads(gzip.decompress(data_bytes))
                    else:
                        data = pickle.loads(data_bytes)

                    return Checkpoint(
                        metadata=metadata,
                        paper_data=data.get("paper_data"),
                        repo_data=data.get("repo_data"),
                        mappings=data.get("mappings"),
                        code_results=data.get("code_results"),
                        knowledge_graph_data=data.get("knowledge_graph_data"),
                        errors=data.get("errors", [])
                    )
                except Exception as e:
                    print(f"Failed to load checkpoint {checkpoint_id}: {e}")
                    return None

        return None

    def get_latest(
        self,
        paper_source: str,
        repo_url: str
    ) -> Optional[Checkpoint]:
        """
        Get the latest checkpoint for a pipeline.

        Args:
            paper_source: Paper source
            repo_url: Repository URL

        Returns:
            Latest checkpoint, or None if none exists
        """
        pipeline_id = self._get_pipeline_id(paper_source, repo_url)
        pipeline_dir = self.checkpoint_dir / pipeline_id

        if not pipeline_dir.exists():
            return None

        # Find all metadata files
        checkpoints = []
        for meta_path in pipeline_dir.glob("*.meta.json"):
            try:
                metadata = CheckpointMetadata.model_validate(
                    json.loads(meta_path.read_text())
                )
                checkpoints.append(metadata)
            except Exception:
                continue

        if not checkpoints:
            return None

        # Sort by creation time and get latest
        latest = max(checkpoints, key=lambda c: c.created_at)
        return self.load(latest.checkpoint_id)

    def get_checkpoint_at_stage(
        self,
        paper_source: str,
        repo_url: str,
        stage: CheckpointStage
    ) -> Optional[Checkpoint]:
        """
        Get checkpoint at a specific stage.

        Args:
            paper_source: Paper source
            repo_url: Repository URL
            stage: Target stage

        Returns:
            Checkpoint at stage, or None
        """
        pipeline_id = self._get_pipeline_id(paper_source, repo_url)
        pipeline_dir = self.checkpoint_dir / pipeline_id

        if not pipeline_dir.exists():
            return None

        # Find checkpoint at this stage
        for meta_path in pipeline_dir.glob("*.meta.json"):
            try:
                metadata = CheckpointMetadata.model_validate(
                    json.loads(meta_path.read_text())
                )
                if metadata.stage == stage:
                    return self.load(metadata.checkpoint_id)
            except Exception:
                continue

        return None

    def list_checkpoints(
        self,
        paper_source: str = None,
        repo_url: str = None
    ) -> List[CheckpointMetadata]:
        """
        List available checkpoints.

        Args:
            paper_source: Optional filter by paper source
            repo_url: Optional filter by repo URL

        Returns:
            List of checkpoint metadata
        """
        checkpoints = []

        # If both filters provided, look in specific pipeline dir
        if paper_source and repo_url:
            pipeline_id = self._get_pipeline_id(paper_source, repo_url)
            dirs = [self.checkpoint_dir / pipeline_id]
        else:
            dirs = [d for d in self.checkpoint_dir.iterdir() if d.is_dir()]

        for pipeline_dir in dirs:
            if not pipeline_dir.exists():
                continue

            for meta_path in pipeline_dir.glob("*.meta.json"):
                try:
                    metadata = CheckpointMetadata.model_validate(
                        json.loads(meta_path.read_text())
                    )
                    checkpoints.append(metadata)
                except Exception:
                    continue

        # Sort by creation time
        return sorted(checkpoints, key=lambda c: c.created_at, reverse=True)

    def delete(self, checkpoint_id: str) -> bool:
        """
        Delete a specific checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to delete

        Returns:
            True if deleted, False if not found
        """
        for pipeline_dir in self.checkpoint_dir.iterdir():
            if not pipeline_dir.is_dir():
                continue

            meta_path = pipeline_dir / f"{checkpoint_id}.meta.json"
            data_path = pipeline_dir / f"{checkpoint_id}.data"

            if meta_path.exists():
                meta_path.unlink()
                if data_path.exists():
                    data_path.unlink()
                return True

        return False

    def delete_pipeline(self, paper_source: str, repo_url: str) -> int:
        """
        Delete all checkpoints for a pipeline.

        Args:
            paper_source: Paper source
            repo_url: Repository URL

        Returns:
            Number of checkpoints deleted
        """
        pipeline_id = self._get_pipeline_id(paper_source, repo_url)
        pipeline_dir = self.checkpoint_dir / pipeline_id

        if not pipeline_dir.exists():
            return 0

        count = len(list(pipeline_dir.glob("*.meta.json")))
        shutil.rmtree(pipeline_dir)
        return count

    def _cleanup_old_checkpoints(self, pipeline_dir: Path) -> None:
        """Remove old checkpoints beyond max limit."""
        meta_files = list(pipeline_dir.glob("*.meta.json"))

        if len(meta_files) <= self.max_checkpoints:
            return

        # Sort by creation time
        checkpoints = []
        for meta_path in meta_files:
            try:
                metadata = CheckpointMetadata.model_validate(
                    json.loads(meta_path.read_text())
                )
                checkpoints.append((metadata, meta_path))
            except Exception:
                continue

        checkpoints.sort(key=lambda x: x[0].created_at)

        # Delete oldest
        to_delete = len(checkpoints) - self.max_checkpoints
        for metadata, meta_path in checkpoints[:to_delete]:
            self.delete(metadata.checkpoint_id)

    def get_storage_usage(self) -> Dict[str, Any]:
        """Get checkpoint storage statistics."""
        total_size = 0
        checkpoint_count = 0
        pipeline_count = 0

        for pipeline_dir in self.checkpoint_dir.iterdir():
            if not pipeline_dir.is_dir():
                continue

            pipeline_count += 1
            for data_path in pipeline_dir.glob("*.data"):
                total_size += data_path.stat().st_size
                checkpoint_count += 1

        return {
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "checkpoint_count": checkpoint_count,
            "pipeline_count": pipeline_count
        }


# Global checkpoint manager instance
_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager(checkpoint_dir: str = "./checkpoints") -> CheckpointManager:
    """Get or create the global checkpoint manager."""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager(checkpoint_dir)
    return _checkpoint_manager
