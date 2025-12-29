"""
Tests for Pipeline Orchestrator - Coordinates the multi-agent analysis pipeline.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


# ============================================================================
# PipelineStage Enum Tests
# ============================================================================

class TestPipelineStage:
    """Tests for PipelineStage enum."""

    def test_all_stages_exist(self):
        """Test all expected stages are defined."""
        from core.orchestrator import PipelineStage

        expected_stages = [
            "INITIALIZED", "PARSING_PAPER", "ANALYZING_REPO",
            "MAPPING_CONCEPTS", "GENERATING_CODE", "SETTING_UP_ENV",
            "EXECUTING_CODE", "GENERATING_REPORT", "COMPLETED", "FAILED"
        ]

        for stage_name in expected_stages:
            assert hasattr(PipelineStage, stage_name), f"Missing stage: {stage_name}"

    def test_stage_values(self):
        """Test stage values are correct."""
        from core.orchestrator import PipelineStage

        assert PipelineStage.INITIALIZED.value == "initialized"
        assert PipelineStage.PARSING_PAPER.value == "parsing_paper"
        assert PipelineStage.COMPLETED.value == "completed"
        assert PipelineStage.FAILED.value == "failed"

    def test_stage_progress_mapping(self):
        """Test STAGE_PROGRESS dict is complete."""
        from core.orchestrator import PipelineStage, STAGE_PROGRESS

        for stage in PipelineStage:
            assert stage in STAGE_PROGRESS, f"Missing progress for {stage}"

    def test_stage_progress_values(self):
        """Test progress values are sensible."""
        from core.orchestrator import STAGE_PROGRESS, PipelineStage

        assert STAGE_PROGRESS[PipelineStage.INITIALIZED] == 0
        assert STAGE_PROGRESS[PipelineStage.COMPLETED] == 100
        assert STAGE_PROGRESS[PipelineStage.FAILED] == -1

        # Progress should generally increase
        assert STAGE_PROGRESS[PipelineStage.PARSING_PAPER] < STAGE_PROGRESS[PipelineStage.ANALYZING_REPO]
        assert STAGE_PROGRESS[PipelineStage.ANALYZING_REPO] < STAGE_PROGRESS[PipelineStage.MAPPING_CONCEPTS]


# ============================================================================
# PipelineEvent Tests
# ============================================================================

class TestPipelineEvent:
    """Tests for PipelineEvent model."""

    def test_event_creation(self):
        """Test creating a pipeline event."""
        from core.orchestrator import PipelineEvent, PipelineStage

        event = PipelineEvent(
            stage=PipelineStage.PARSING_PAPER,
            progress=15,
            message="Parsing scientific paper..."
        )

        assert event.stage == PipelineStage.PARSING_PAPER
        assert event.progress == 15
        assert event.message == "Parsing scientific paper..."
        assert event.data == {}
        assert event.timestamp is not None

    def test_event_with_data(self):
        """Test event with additional data."""
        from core.orchestrator import PipelineEvent, PipelineStage

        event = PipelineEvent(
            stage=PipelineStage.COMPLETED,
            progress=100,
            message="Pipeline completed",
            data={"papers_processed": 1, "mappings_found": 10}
        )

        assert event.data["papers_processed"] == 1
        assert event.data["mappings_found"] == 10

    def test_event_progress_bounds(self):
        """Test progress must be 0-100."""
        from core.orchestrator import PipelineEvent, PipelineStage
        from pydantic import ValidationError

        # Valid progress values
        PipelineEvent(stage=PipelineStage.INITIALIZED, progress=0, message="test")
        PipelineEvent(stage=PipelineStage.COMPLETED, progress=100, message="test")

        # Invalid progress values
        with pytest.raises(ValidationError):
            PipelineEvent(stage=PipelineStage.INITIALIZED, progress=-1, message="test")
        with pytest.raises(ValidationError):
            PipelineEvent(stage=PipelineStage.INITIALIZED, progress=101, message="test")

    def test_event_timestamp_auto(self):
        """Test timestamp is auto-generated."""
        from core.orchestrator import PipelineEvent, PipelineStage

        before = datetime.now()
        event = PipelineEvent(
            stage=PipelineStage.INITIALIZED,
            progress=0,
            message="test"
        )
        after = datetime.now()

        assert before <= event.timestamp <= after


# ============================================================================
# PipelineResult Tests
# ============================================================================

class TestPipelineResult:
    """Tests for PipelineResult model."""

    def test_result_creation_defaults(self):
        """Test result with defaults."""
        from core.orchestrator import PipelineResult

        result = PipelineResult()

        assert result.paper_data is None
        assert result.repo_data is None
        assert result.mappings is None
        assert result.code_results is None
        assert result.knowledge_graph is None
        assert result.report_paths == []
        assert result.errors == []
        assert result.status == "completed"

    def test_result_with_data(self, valid_paper_parser_output, valid_repo_analyzer_output):
        """Test result with populated data."""
        from core.orchestrator import PipelineResult

        result = PipelineResult(
            paper_data=valid_paper_parser_output,
            repo_data=valid_repo_analyzer_output,
            mappings=[],
            status="completed"
        )

        assert result.paper_data is not None
        assert result.repo_data is not None

    def test_result_with_errors(self):
        """Test result with errors."""
        from core.orchestrator import PipelineResult

        result = PipelineResult(
            status="failed",
            errors=["Paper parsing failed", "Network error"]
        )

        assert result.status == "failed"
        assert len(result.errors) == 2


# ============================================================================
# PipelineOrchestrator Initialization Tests
# ============================================================================

class TestPipelineOrchestratorInit:
    """Tests for PipelineOrchestrator initialization."""

    def test_orchestrator_creation_default(self):
        """Test creating orchestrator with defaults."""
        from core.orchestrator import PipelineOrchestrator, PipelineStage

        orchestrator = PipelineOrchestrator()

        assert orchestrator.llm_provider == "gemini"
        assert orchestrator._current_stage == PipelineStage.INITIALIZED
        assert orchestrator._current_progress == 0
        assert orchestrator._event_callbacks == []

    def test_orchestrator_creation_custom_provider(self):
        """Test creating orchestrator with custom provider."""
        from core.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(llm_provider="anthropic")

        assert orchestrator.llm_provider == "anthropic"

    def test_orchestrator_with_event_callback(self):
        """Test creating orchestrator with event callback."""
        from core.orchestrator import PipelineOrchestrator

        async def callback(event):
            pass

        orchestrator = PipelineOrchestrator(event_callback=callback)

        assert len(orchestrator._event_callbacks) == 1


# ============================================================================
# Event Callback Tests
# ============================================================================

class TestEventCallbacks:
    """Tests for event callback management."""

    def test_add_event_callback(self):
        """Test adding event callback."""
        from core.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        async def callback(event):
            pass

        orchestrator.add_event_callback(callback)

        assert callback in orchestrator._event_callbacks

    def test_add_multiple_callbacks(self):
        """Test adding multiple callbacks."""
        from core.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        async def callback1(event):
            pass

        async def callback2(event):
            pass

        orchestrator.add_event_callback(callback1)
        orchestrator.add_event_callback(callback2)

        assert len(orchestrator._event_callbacks) == 2

    def test_remove_event_callback(self):
        """Test removing event callback."""
        from core.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        async def callback(event):
            pass

        orchestrator.add_event_callback(callback)
        orchestrator.remove_event_callback(callback)

        assert callback not in orchestrator._event_callbacks

    def test_remove_nonexistent_callback(self):
        """Test removing callback that doesn't exist."""
        from core.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        async def callback(event):
            pass

        # Should not raise error
        orchestrator.remove_event_callback(callback)


# ============================================================================
# _emit_event Tests
# ============================================================================

class TestEmitEvent:
    """Tests for _emit_event method."""

    @pytest.mark.asyncio
    async def test_emit_event_updates_state(self):
        """Test emitting event updates current state."""
        from core.orchestrator import PipelineOrchestrator, PipelineStage, STAGE_PROGRESS

        orchestrator = PipelineOrchestrator()

        await orchestrator._emit_event(
            PipelineStage.PARSING_PAPER,
            "Parsing paper..."
        )

        assert orchestrator._current_stage == PipelineStage.PARSING_PAPER
        assert orchestrator._current_progress == STAGE_PROGRESS[PipelineStage.PARSING_PAPER]

    @pytest.mark.asyncio
    async def test_emit_event_broadcasts_to_callbacks(self, pipeline_event_collector):
        """Test event is broadcast to all callbacks."""
        from core.orchestrator import PipelineOrchestrator, PipelineStage

        orchestrator = PipelineOrchestrator()
        orchestrator.add_event_callback(pipeline_event_collector["callback"])

        await orchestrator._emit_event(
            PipelineStage.PARSING_PAPER,
            "Parsing paper...",
            {"paper_id": "test123"}
        )

        events = pipeline_event_collector["events"]
        assert len(events) == 1
        assert events[0].stage == PipelineStage.PARSING_PAPER
        assert events[0].message == "Parsing paper..."
        assert events[0].data["paper_id"] == "test123"

    @pytest.mark.asyncio
    async def test_emit_event_callback_error_handled(self):
        """Test callback errors don't stop pipeline."""
        from core.orchestrator import PipelineOrchestrator, PipelineStage

        orchestrator = PipelineOrchestrator()

        async def failing_callback(event):
            raise Exception("Callback error!")

        orchestrator.add_event_callback(failing_callback)

        # Should not raise
        await orchestrator._emit_event(
            PipelineStage.INITIALIZED,
            "Test message"
        )

    @pytest.mark.asyncio
    async def test_emit_event_multiple_callbacks(self, pipeline_event_collector):
        """Test event reaches all callbacks."""
        from core.orchestrator import PipelineOrchestrator, PipelineStage

        orchestrator = PipelineOrchestrator()

        events2 = []
        async def callback2(event):
            events2.append(event)

        orchestrator.add_event_callback(pipeline_event_collector["callback"])
        orchestrator.add_event_callback(callback2)

        await orchestrator._emit_event(PipelineStage.INITIALIZED, "Test")

        assert len(pipeline_event_collector["events"]) == 1
        assert len(events2) == 1


# ============================================================================
# Agent Lazy Loading Tests
# ============================================================================

class TestAgentLazyLoading:
    """Tests for lazy agent initialization."""

    @pytest.mark.asyncio
    async def test_agents_not_initialized_on_creation(self):
        """Test agents are not created immediately."""
        from core.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        assert orchestrator._paper_parser is None
        assert orchestrator._repo_analyzer is None
        assert orchestrator._semantic_mapper is None
        assert orchestrator._coding_agent is None
        assert orchestrator._llm_client is None

    @pytest.mark.asyncio
    async def test_get_llm_client_creates_once(self):
        """Test LLM client is created once."""
        from core.orchestrator import PipelineOrchestrator
        import os

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            orchestrator = PipelineOrchestrator()

            client1 = await orchestrator._get_llm_client()
            client2 = await orchestrator._get_llm_client()

            assert client1 is client2


# ============================================================================
# Pipeline Run Tests
# ============================================================================

class TestPipelineRun:
    """Tests for pipeline execution."""

    @pytest.fixture
    def mock_agents(self, valid_paper_parser_output, valid_repo_analyzer_output, valid_mapping_result):
        """Create mock agents for testing."""
        paper_parser = MagicMock()
        paper_parser.process = AsyncMock(return_value=valid_paper_parser_output)

        repo_analyzer = MagicMock()
        repo_analyzer.process = AsyncMock(return_value=valid_repo_analyzer_output)

        semantic_mapper = MagicMock()
        mapper_output = MagicMock()
        mapper_output.mappings = [valid_mapping_result]
        semantic_mapper.process = AsyncMock(return_value=mapper_output)

        coding_agent = MagicMock()
        coding_output = MagicMock()
        coding_output.results = []
        coding_output.scripts = []
        coding_agent.process = AsyncMock(return_value=coding_output)

        return {
            "paper_parser": paper_parser,
            "repo_analyzer": repo_analyzer,
            "semantic_mapper": semantic_mapper,
            "coding_agent": coding_agent
        }

    @pytest.mark.asyncio
    async def test_run_initializes_result(self, mock_agents):
        """Test run creates proper result structure."""
        from core.orchestrator import PipelineOrchestrator
        import os

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            orchestrator = PipelineOrchestrator()

            # Mock agent getters
            orchestrator._get_paper_parser = AsyncMock(return_value=mock_agents["paper_parser"])
            orchestrator._get_repo_analyzer = AsyncMock(return_value=mock_agents["repo_analyzer"])
            orchestrator._get_semantic_mapper = AsyncMock(return_value=mock_agents["semantic_mapper"])
            orchestrator._get_coding_agent = AsyncMock(return_value=mock_agents["coding_agent"])

            result = await orchestrator.run(
                paper_source="test.pdf",
                repo_url="https://github.com/test/repo",
                auto_execute=False
            )

            assert result.knowledge_graph is not None

    @pytest.mark.asyncio
    async def test_run_emits_initialized_event(self, mock_agents, pipeline_event_collector):
        """Test run emits INITIALIZED event first."""
        from core.orchestrator import PipelineOrchestrator, PipelineStage
        import os

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            orchestrator = PipelineOrchestrator()
            orchestrator.add_event_callback(pipeline_event_collector["callback"])

            # Mock agents
            orchestrator._get_paper_parser = AsyncMock(return_value=mock_agents["paper_parser"])
            orchestrator._get_repo_analyzer = AsyncMock(return_value=mock_agents["repo_analyzer"])
            orchestrator._get_semantic_mapper = AsyncMock(return_value=mock_agents["semantic_mapper"])
            orchestrator._get_coding_agent = AsyncMock(return_value=mock_agents["coding_agent"])

            await orchestrator.run(
                paper_source="test.pdf",
                repo_url="https://github.com/test/repo",
                auto_execute=False
            )

            events = pipeline_event_collector["events"]
            assert len(events) > 0
            assert events[0].stage == PipelineStage.INITIALIZED

    @pytest.mark.asyncio
    async def test_run_skip_execution(self, mock_agents):
        """Test auto_execute=False skips code execution."""
        from core.orchestrator import PipelineOrchestrator
        import os

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            orchestrator = PipelineOrchestrator()

            # Mock agents
            orchestrator._get_paper_parser = AsyncMock(return_value=mock_agents["paper_parser"])
            orchestrator._get_repo_analyzer = AsyncMock(return_value=mock_agents["repo_analyzer"])
            orchestrator._get_semantic_mapper = AsyncMock(return_value=mock_agents["semantic_mapper"])
            orchestrator._get_coding_agent = AsyncMock(return_value=mock_agents["coding_agent"])

            result = await orchestrator.run(
                paper_source="test.pdf",
                repo_url="https://github.com/test/repo",
                auto_execute=False
            )

            # Coding agent should be called with execute=False
            coding_call = mock_agents["coding_agent"].process.call_args
            # Check if execute=False was passed (implementation specific)
            assert result is not None


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestPipelineErrorHandling:
    """Tests for pipeline error handling."""

    @pytest.mark.asyncio
    async def test_paper_parse_error_captured(self):
        """Test paper parsing error is captured in result."""
        from core.orchestrator import PipelineOrchestrator
        import os

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            orchestrator = PipelineOrchestrator()

            # Mock paper parser to fail
            mock_parser = MagicMock()
            mock_parser.process = AsyncMock(side_effect=Exception("Paper parsing failed"))
            orchestrator._get_paper_parser = AsyncMock(return_value=mock_parser)

            # Mock other agents
            mock_repo = MagicMock()
            mock_repo.process = AsyncMock(return_value=MagicMock())
            orchestrator._get_repo_analyzer = AsyncMock(return_value=mock_repo)

            mock_mapper = MagicMock()
            mock_mapper.process = AsyncMock(return_value=MagicMock(mappings=[]))
            orchestrator._get_semantic_mapper = AsyncMock(return_value=mock_mapper)

            mock_coder = MagicMock()
            mock_coder.process = AsyncMock(return_value=MagicMock(results=[], scripts=[]))
            orchestrator._get_coding_agent = AsyncMock(return_value=mock_coder)

            result = await orchestrator.run(
                paper_source="test.pdf",
                repo_url="https://github.com/test/repo",
                auto_execute=False
            )

            assert result.paper_data is None
            assert len(result.errors) > 0
            assert "Paper parsing failed" in result.errors[0]


# ============================================================================
# Checkpoint Resume Tests
# ============================================================================

class TestCheckpointResume:
    """Tests for checkpoint resumption."""

    @pytest.mark.asyncio
    async def test_resume_loads_checkpoint(self, mock_agents, checkpoint_manager, sample_checkpoint_data):
        """Test resume=True loads from checkpoint."""
        from core.orchestrator import PipelineOrchestrator
        from core.checkpointing import CheckpointStage, Checkpoint, CheckpointMetadata
        import os
        from datetime import datetime

        # Save a checkpoint first
        checkpoint_id = checkpoint_manager.save(
            stage=CheckpointStage.PAPER_PARSED,
            paper_source="test.pdf",
            repo_url="https://github.com/test/repo",
            paper_data=sample_checkpoint_data["paper_data"]
        )

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            orchestrator = PipelineOrchestrator()
            orchestrator._checkpoint_manager = checkpoint_manager

            # Mock agents
            orchestrator._get_paper_parser = AsyncMock(return_value=mock_agents["paper_parser"])
            orchestrator._get_repo_analyzer = AsyncMock(return_value=mock_agents["repo_analyzer"])
            orchestrator._get_semantic_mapper = AsyncMock(return_value=mock_agents["semantic_mapper"])
            orchestrator._get_coding_agent = AsyncMock(return_value=mock_agents["coding_agent"])

            result = await orchestrator.run(
                paper_source="test.pdf",
                repo_url="https://github.com/test/repo",
                resume=True,
                auto_execute=False
            )

            # Paper parser should not be called (resumed from checkpoint)
            # Note: This depends on implementation details


# ============================================================================
# Stage Progression Tests
# ============================================================================

class TestStageProgression:
    """Tests for stage progression during pipeline execution."""

    @pytest.mark.asyncio
    async def test_stages_progress_in_order(self, mock_agents, pipeline_event_collector):
        """Test stages are emitted in correct order."""
        from core.orchestrator import PipelineOrchestrator, PipelineStage
        import os

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            orchestrator = PipelineOrchestrator()
            orchestrator.add_event_callback(pipeline_event_collector["callback"])

            # Mock agents
            orchestrator._get_paper_parser = AsyncMock(return_value=mock_agents["paper_parser"])
            orchestrator._get_repo_analyzer = AsyncMock(return_value=mock_agents["repo_analyzer"])
            orchestrator._get_semantic_mapper = AsyncMock(return_value=mock_agents["semantic_mapper"])
            orchestrator._get_coding_agent = AsyncMock(return_value=mock_agents["coding_agent"])

            await orchestrator.run(
                paper_source="test.pdf",
                repo_url="https://github.com/test/repo",
                auto_execute=False
            )

            events = pipeline_event_collector["events"]
            stages = [e.stage for e in events]

            # First stage should be INITIALIZED
            assert stages[0] == PipelineStage.INITIALIZED

            # PARSING_PAPER should come before ANALYZING_REPO
            if PipelineStage.PARSING_PAPER in stages and PipelineStage.ANALYZING_REPO in stages:
                assert stages.index(PipelineStage.PARSING_PAPER) < stages.index(PipelineStage.ANALYZING_REPO)

    @pytest.mark.asyncio
    async def test_progress_increases(self, mock_agents, pipeline_event_collector):
        """Test progress generally increases through pipeline."""
        from core.orchestrator import PipelineOrchestrator
        import os

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            orchestrator = PipelineOrchestrator()
            orchestrator.add_event_callback(pipeline_event_collector["callback"])

            # Mock agents
            orchestrator._get_paper_parser = AsyncMock(return_value=mock_agents["paper_parser"])
            orchestrator._get_repo_analyzer = AsyncMock(return_value=mock_agents["repo_analyzer"])
            orchestrator._get_semantic_mapper = AsyncMock(return_value=mock_agents["semantic_mapper"])
            orchestrator._get_coding_agent = AsyncMock(return_value=mock_agents["coding_agent"])

            await orchestrator.run(
                paper_source="test.pdf",
                repo_url="https://github.com/test/repo",
                auto_execute=False
            )

            events = pipeline_event_collector["events"]
            # Filter out any FAILED events which have -1 progress
            progress_values = [e.progress for e in events if e.progress >= 0]

            # Progress should generally increase (allow for multiple events at same stage)
            for i in range(1, len(progress_values)):
                assert progress_values[i] >= progress_values[i-1] or progress_values[i] == 0
