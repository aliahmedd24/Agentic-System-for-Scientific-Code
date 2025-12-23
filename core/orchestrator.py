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
from .error_handling import logger, LogCategory


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
    progress: int
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "stage": self.stage.value,
            "progress": self.progress,
            "message": self.message,
            "data": self.data
        }


@dataclass
class PipelineResult:
    """Result returned by the pipeline."""
    paper_data: Optional[Dict[str, Any]] = None
    repo_data: Optional[Dict[str, Any]] = None
    mappings: Optional[List[Dict[str, Any]]] = None
    code_results: Optional[List[Dict[str, Any]]] = None
    knowledge_graph: Optional[KnowledgeGraph] = None
    report_paths: List[str] = field(default_factory=list)
    errors: List[Any] = field(default_factory=list)
    status: str = "completed"


# Type for event callbacks
EventCallback = Callable[[PipelineEvent], Awaitable[None]]


class PipelineOrchestrator:
    """
    Coordinates the entire multi-agent analysis pipeline.
    """
    
    def __init__(
        self,
        llm_provider: str = "gemini",
        event_callback: Optional[EventCallback] = None
    ):
        self.llm_provider = llm_provider
        self._event_callbacks: List[EventCallback] = []
        
        # Add event callback if provided
        if event_callback:
            self._event_callbacks.append(event_callback)
        
        # Agents will be created lazily when needed
        self._paper_parser = None
        self._repo_analyzer = None
        self._semantic_mapper = None
        self._coding_agent = None
        self._llm_client = None
        
        # Current state
        self._current_stage = PipelineStage.INITIALIZED
        self._current_progress = 0
    
    async def _get_llm_client(self):
        """Get or create LLM client."""
        if self._llm_client is None:
            from .llm_client import LLMClient
            self._llm_client = LLMClient(provider=self.llm_provider)
        return self._llm_client
    
    async def _get_paper_parser(self):
        """Get or create paper parser agent."""
        if self._paper_parser is None:
            from agents.paper_parser_agent import PaperParserAgent
            client = await self._get_llm_client()
            self._paper_parser = PaperParserAgent(llm_client=client)
        return self._paper_parser
    
    async def _get_repo_analyzer(self):
        """Get or create repo analyzer agent."""
        if self._repo_analyzer is None:
            from agents.repo_analyzer_agent import RepoAnalyzerAgent
            client = await self._get_llm_client()
            self._repo_analyzer = RepoAnalyzerAgent(llm_client=client)
        return self._repo_analyzer
    
    async def _get_semantic_mapper(self):
        """Get or create semantic mapper."""
        if self._semantic_mapper is None:
            from agents.semantic_mapper import SemanticMapper
            client = await self._get_llm_client()
            self._semantic_mapper = SemanticMapper(llm_client=client)
        return self._semantic_mapper
    
    async def _get_coding_agent(self):
        """Get or create coding agent."""
        if self._coding_agent is None:
            from agents.coding_agent import CodingAgent
            client = await self._get_llm_client()
            self._coding_agent = CodingAgent(llm_client=client)
        return self._coding_agent
    
    def add_event_callback(self, callback: EventCallback):
        """Add callback for pipeline events."""
        self._event_callbacks.append(callback)
    
    def remove_event_callback(self, callback: EventCallback):
        """Remove event callback."""
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)
    
    async def _emit_event(
        self,
        stage: PipelineStage,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """Emit a pipeline event to all callbacks."""
        self._current_stage = stage
        self._current_progress = STAGE_PROGRESS.get(stage, 0)
        
        event = PipelineEvent(
            timestamp=datetime.now(),
            stage=stage,
            progress=self._current_progress,
            message=message,
            data=data or {}
        )
        
        # Log the event
        logger.info(f"[{stage.value}] {message}", category=LogCategory.PIPELINE)
        
        # Broadcast to callbacks
        for callback in self._event_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}", category=LogCategory.PIPELINE)
    
    async def run(
        self,
        paper_source: str,
        repo_url: str,
        auto_execute: bool = True
    ) -> PipelineResult:
        """
        Execute the full analysis pipeline.
        
        Args:
            paper_source: arXiv ID, URL, or path to PDF
            repo_url: GitHub repository URL
            auto_execute: Whether to execute generated code
            
        Returns:
            PipelineResult with all analysis data
        """
        result = PipelineResult(
            knowledge_graph=KnowledgeGraph()
        )
        
        try:
            await self._emit_event(
                PipelineStage.INITIALIZED,
                f"Starting analysis pipeline"
            )
            
            # Stage 1: Parse Paper
            await self._emit_event(
                PipelineStage.PARSING_PAPER,
                "Parsing scientific paper..."
            )
            try:
                paper_parser = await self._get_paper_parser()
                result.paper_data = await paper_parser.process(
                    paper_source=paper_source,
                    knowledge_graph=result.knowledge_graph
                )
                title = result.paper_data.get('title', 'Unknown') if result.paper_data else 'Unknown'
                await self._emit_event(
                    PipelineStage.PARSING_PAPER,
                    f"Paper parsed: {title}"
                )
            except Exception as e:
                logger.error(f"Paper parsing failed: {e}", category=LogCategory.PIPELINE)
                result.paper_data = {"error": str(e), "title": "Parse Failed"}
                result.errors.append(str(e))
            
            # Stage 2: Analyze Repository
            await self._emit_event(
                PipelineStage.ANALYZING_REPO,
                "Analyzing code repository..."
            )
            try:
                repo_analyzer = await self._get_repo_analyzer()
                result.repo_data = await repo_analyzer.process(
                    repo_url=repo_url,
                    knowledge_graph=result.knowledge_graph
                )
                name = result.repo_data.get('name', 'Unknown') if result.repo_data else 'Unknown'
                await self._emit_event(
                    PipelineStage.ANALYZING_REPO,
                    f"Repository analyzed: {name}"
                )
            except Exception as e:
                logger.error(f"Repository analysis failed: {e}", category=LogCategory.PIPELINE)
                result.repo_data = {"error": str(e), "name": "Analysis Failed"}
                result.errors.append(str(e))
            
            # Stage 3: Semantic Mapping
            await self._emit_event(
                PipelineStage.MAPPING_CONCEPTS,
                "Mapping concepts to code..."
            )
            try:
                semantic_mapper = await self._get_semantic_mapper()
                mapping_result = await semantic_mapper.process(
                    paper_data=result.paper_data,
                    repo_data=result.repo_data,
                    knowledge_graph=result.knowledge_graph
                )
                result.mappings = mapping_result.get("mappings", []) if mapping_result else []
                await self._emit_event(
                    PipelineStage.MAPPING_CONCEPTS,
                    f"Found {len(result.mappings)} concept-to-code mappings"
                )
            except Exception as e:
                logger.error(f"Semantic mapping failed: {e}", category=LogCategory.PIPELINE)
                result.mappings = []
                result.errors.append(str(e))
            
            # Stage 4 & 5: Generate and Execute Code
            result.code_results = []
            if auto_execute and result.mappings:
                await self._emit_event(
                    PipelineStage.GENERATING_CODE,
                    "Generating validation code..."
                )
                try:
                    coding_agent = await self._get_coding_agent()
                    code_result = await coding_agent.process(
                        mappings=result.mappings,
                        repo_data=result.repo_data,
                        knowledge_graph=result.knowledge_graph,
                        execute=True
                    )
                    result.code_results = code_result.get("results", []) if code_result else []
                    await self._emit_event(
                        PipelineStage.EXECUTING_CODE,
                        f"Executed {len(result.code_results)} test scripts"
                    )
                except Exception as e:
                    logger.error(f"Code generation/execution failed: {e}", category=LogCategory.PIPELINE)
                    result.errors.append(str(e))
            
            # Stage 6: Complete
            await self._emit_event(
                PipelineStage.COMPLETED,
                "Analysis complete!"
            )
            result.status = "completed"
            
        except Exception as e:
            result.status = "failed"
            result.errors.append(str(e))
            await self._emit_event(
                PipelineStage.FAILED,
                f"Pipeline failed: {e}"
            )
        
        return result