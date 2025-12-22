"""
Pipeline Orchestrator - Coordinates the multi-agent analysis pipeline.
"""

import asyncio
import uuid
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable, Awaitable
from pathlib import Path

from .knowledge_graph import KnowledgeGraph
from .error_handling import (
    logger, LogCategory, ErrorCategory, ErrorSeverity,
    create_error, AgentError, StructuredError
)


class PipelineStage(Enum):
    """Pipeline execution stages."""
    INITIALIZED = "initialized"
    PARSING_PAPER = "parsing_paper"
    ANALYZING_REPO = "analyzing_repo"
    MAPPING_CONCEPTS = "mapping_concepts"
    GENERATING_CODE = "generating_code"
    SETTING_UP_ENV = "setting_up_environment"
    EXECUTING_CODE = "executing_code"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"
    FAILED = "failed"


# Stage progress percentages
STAGE_PROGRESS = {
    PipelineStage.INITIALIZED: 0,
    PipelineStage.PARSING_PAPER: 15,
    PipelineStage.ANALYZING_REPO: 35,
    PipelineStage.MAPPING_CONCEPTS: 50,
    PipelineStage.GENERATING_CODE: 65,
    PipelineStage.SETTING_UP_ENV: 75,
    PipelineStage.EXECUTING_CODE: 85,
    PipelineStage.GENERATING_REPORT: 95,
    PipelineStage.COMPLETED: 100,
    PipelineStage.FAILED: -1
}


@dataclass
class PipelineEvent:
    """Event emitted during pipeline execution."""
    timestamp: datetime
    stage: PipelineStage
    event_type: str  # info, warning, error, progress
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "stage": self.stage.value,
            "event_type": self.event_type,
            "message": self.message,
            "data": self.data
        }


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    paper_url: str
    repo_url: str
    auto_fix_errors: bool = True
    use_docker: bool = True
    max_retries: int = 3
    timeout_seconds: int = 600
    output_dir: str = "./outputs"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_url": self.paper_url,
            "repo_url": self.repo_url,
            "auto_fix_errors": self.auto_fix_errors,
            "use_docker": self.use_docker,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "output_dir": self.output_dir
        }


@dataclass
class PipelineResult:
    """Complete result of pipeline execution."""
    run_id: str
    config: PipelineConfig
    status: str  # running, completed, failed
    stage: PipelineStage
    progress: int
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Results from each stage
    paper_data: Optional[Dict[str, Any]] = None
    repo_data: Optional[Dict[str, Any]] = None
    mappings: Optional[List[Dict[str, Any]]] = None
    generated_code: Optional[List[Dict[str, Any]]] = None
    execution_results: Optional[List[Dict[str, Any]]] = None
    visualizations: Optional[List[str]] = None
    
    # Knowledge graph
    knowledge_graph: Optional[KnowledgeGraph] = None
    
    # Events and errors
    events: List[PipelineEvent] = field(default_factory=list)
    errors: List[StructuredError] = field(default_factory=list)
    
    # Report paths
    report_path: Optional[str] = None
    json_export_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "status": self.status,
            "stage": self.stage.value,
            "progress": self.progress,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "paper_data": self.paper_data,
            "repo_data": self.repo_data,
            "mappings": self.mappings,
            "generated_code": self.generated_code,
            "execution_results": self.execution_results,
            "visualizations": self.visualizations,
            "events": [e.to_dict() for e in self.events],
            "errors": [e.to_dict() for e in self.errors],
            "report_path": self.report_path,
            "json_export_path": self.json_export_path,
            "knowledge_graph_stats": self.knowledge_graph.get_statistics() if self.knowledge_graph else None
        }


# Type for event callbacks
EventCallback = Callable[[PipelineEvent], Awaitable[None]]


class PipelineOrchestrator:
    """
    Coordinates the entire multi-agent analysis pipeline.
    """
    
    def __init__(
        self,
        paper_parser_agent,
        repo_analyzer_agent,
        semantic_mapper,
        coding_agent,
        report_engine
    ):
        self.paper_parser = paper_parser_agent
        self.repo_analyzer = repo_analyzer_agent
        self.semantic_mapper = semantic_mapper
        self.coding_agent = coding_agent
        self.report_engine = report_engine
        
        self._event_callbacks: List[EventCallback] = []
        self._active_runs: Dict[str, PipelineResult] = {}
    
    def add_event_callback(self, callback: EventCallback):
        """Add callback for pipeline events (e.g., WebSocket broadcast)."""
        self._event_callbacks.append(callback)
    
    def remove_event_callback(self, callback: EventCallback):
        """Remove event callback."""
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)
    
    async def _emit_event(
        self,
        result: PipelineResult,
        event_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """Emit a pipeline event."""
        event = PipelineEvent(
            timestamp=datetime.now(),
            stage=result.stage,
            event_type=event_type,
            message=message,
            data=data or {}
        )
        
        result.events.append(event)
        
        # Log the event
        log_func = getattr(logger, event_type if event_type in ['info', 'warning', 'error'] else 'info')
        log_func(
            LogCategory.PIPELINE,
            f"[{result.stage.value}] {message}",
            stage=result.stage.value
        )
        
        # Broadcast to callbacks
        for callback in self._event_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(LogCategory.PIPELINE, f"Event callback error: {e}")
    
    async def _update_stage(
        self,
        result: PipelineResult,
        stage: PipelineStage,
        message: Optional[str] = None
    ):
        """Update pipeline stage and emit progress event."""
        result.stage = stage
        result.progress = STAGE_PROGRESS.get(stage, 0)
        
        await self._emit_event(
            result,
            "progress",
            message or f"Entering stage: {stage.value}",
            {"progress": result.progress, "stage": stage.value}
        )
    
    async def run(self, config: PipelineConfig) -> PipelineResult:
        """
        Execute the full analysis pipeline.
        """
        run_id = str(uuid.uuid4())[:8]
        
        # Initialize result
        result = PipelineResult(
            run_id=run_id,
            config=config,
            status="running",
            stage=PipelineStage.INITIALIZED,
            progress=0,
            start_time=datetime.now(),
            knowledge_graph=KnowledgeGraph()
        )
        
        self._active_runs[run_id] = result
        
        # Create output directory
        output_dir = Path(config.output_dir) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            await self._emit_event(
                result, "info",
                f"Starting analysis pipeline (run_id: {run_id})"
            )
            
            # Stage 1: Parse Paper
            await self._update_stage(result, PipelineStage.PARSING_PAPER, "Parsing scientific paper...")
            try:
                result.paper_data = await self.paper_parser.parse(
                    config.paper_url,
                    result.knowledge_graph
                )
                await self._emit_event(
                    result, "info",
                    f"Paper parsed: {result.paper_data.get('title', 'Unknown')}"
                )
            except Exception as e:
                await self._handle_stage_error(result, "Paper parsing", e)
                if not config.auto_fix_errors:
                    raise
            
            # Stage 2: Analyze Repository
            await self._update_stage(result, PipelineStage.ANALYZING_REPO, "Analyzing code repository...")
            try:
                result.repo_data = await self.repo_analyzer.analyze(
                    config.repo_url,
                    result.knowledge_graph
                )
                await self._emit_event(
                    result, "info",
                    f"Repository analyzed: {result.repo_data.get('name', 'Unknown')}"
                )
            except Exception as e:
                await self._handle_stage_error(result, "Repository analysis", e)
                if not config.auto_fix_errors:
                    raise
            
            # Stage 3: Map Concepts to Code
            await self._update_stage(result, PipelineStage.MAPPING_CONCEPTS, "Mapping paper concepts to code...")
            try:
                result.mappings = await self.semantic_mapper.map_concepts(
                    result.paper_data,
                    result.repo_data,
                    result.knowledge_graph
                )
                await self._emit_event(
                    result, "info",
                    f"Found {len(result.mappings or [])} concept-code mappings"
                )
            except Exception as e:
                await self._handle_stage_error(result, "Concept mapping", e)
                if not config.auto_fix_errors:
                    raise
            
            # Stage 4: Generate Test Code
            await self._update_stage(result, PipelineStage.GENERATING_CODE, "Generating test code...")
            try:
                result.generated_code = await self.coding_agent.generate_tests(
                    result.mappings,
                    result.repo_data,
                    result.knowledge_graph
                )
                await self._emit_event(
                    result, "info",
                    f"Generated {len(result.generated_code or [])} test scripts"
                )
            except Exception as e:
                await self._handle_stage_error(result, "Code generation", e)
                if not config.auto_fix_errors:
                    raise
            
            # Stage 5: Execute Code
            await self._update_stage(result, PipelineStage.EXECUTING_CODE, "Executing generated code...")
            try:
                result.execution_results, result.visualizations = await self.coding_agent.execute_tests(
                    result.generated_code,
                    output_dir,
                    use_docker=config.use_docker
                )
                
                success_count = sum(1 for r in (result.execution_results or []) if r.get("success"))
                await self._emit_event(
                    result, "info",
                    f"Execution complete: {success_count}/{len(result.execution_results or [])} tests passed"
                )
            except Exception as e:
                await self._handle_stage_error(result, "Code execution", e)
                if not config.auto_fix_errors:
                    raise
            
            # Stage 6: Generate Report
            await self._update_stage(result, PipelineStage.GENERATING_REPORT, "Generating analysis report...")
            try:
                result.report_path = await self.report_engine.generate_html_report(
                    result,
                    output_dir
                )
                result.json_export_path = await self.report_engine.export_json(
                    result,
                    output_dir
                )
                await self._emit_event(
                    result, "info",
                    f"Report generated: {result.report_path}"
                )
            except Exception as e:
                await self._handle_stage_error(result, "Report generation", e)
            
            # Complete
            await self._update_stage(result, PipelineStage.COMPLETED, "Analysis complete!")
            result.status = "completed"
            
        except Exception as e:
            result.status = "failed"
            result.stage = PipelineStage.FAILED
            result.errors.append(create_error(
                ErrorCategory.EXECUTION,
                f"Pipeline failed: {str(e)}",
                original_error=e,
                severity=ErrorSeverity.CRITICAL
            ))
            await self._emit_event(result, "error", f"Pipeline failed: {str(e)}")
        
        finally:
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            
            await self._emit_event(
                result, "info",
                f"Pipeline finished in {result.duration_seconds:.1f}s",
                {"status": result.status, "duration": result.duration_seconds}
            )
        
        return result
    
    async def _handle_stage_error(
        self,
        result: PipelineResult,
        stage_name: str,
        error: Exception
    ):
        """Handle an error in a pipeline stage."""
        structured_error = create_error(
            ErrorCategory.EXECUTION,
            f"{stage_name} failed: {str(error)}",
            original_error=error,
            severity=ErrorSeverity.ERROR
        )
        result.errors.append(structured_error)
        
        await self._emit_event(
            result, "error",
            f"{stage_name} error: {str(error)}",
            {"error": structured_error.to_dict()}
        )
    
    def get_run(self, run_id: str) -> Optional[PipelineResult]:
        """Get a pipeline run by ID."""
        return self._active_runs.get(run_id)
    
    def get_all_runs(self) -> List[PipelineResult]:
        """Get all pipeline runs."""
        return list(self._active_runs.values())
    
    async def cancel_run(self, run_id: str) -> bool:
        """Cancel a running pipeline (not fully implemented)."""
        result = self._active_runs.get(run_id)
        if result and result.status == "running":
            result.status = "cancelled"
            result.stage = PipelineStage.FAILED
            await self._emit_event(result, "warning", "Pipeline cancelled by user")
            return True
        return False
