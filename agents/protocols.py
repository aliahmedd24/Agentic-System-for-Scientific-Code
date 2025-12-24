"""
Agent Protocols - Type definitions and contracts for all agents.

This module defines the input/output contracts for each agent,
ensuring type safety and consistent interfaces across the pipeline.
"""

from typing import Protocol, TypedDict, Dict, Any, List, Optional, runtime_checkable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime


# ============================================================================
# Input/Output Type Definitions
# ============================================================================

class PaperParserInput(TypedDict, total=False):
    """Input contract for PaperParserAgent."""
    paper_source: str  # arXiv ID, URL, or file path (REQUIRED)
    knowledge_graph: Any  # Optional KnowledgeGraph to populate


class PaperParserOutput(TypedDict, total=False):
    """Output contract for PaperParserAgent."""
    title: str
    authors: List[str]
    abstract: str
    key_concepts: List[Dict[str, Any]]
    algorithms: List[Dict[str, Any]]
    methodology: Dict[str, Any]
    reproducibility: Dict[str, Any]
    expected_implementations: List[Dict[str, Any]]
    source_metadata: Dict[str, Any]
    _kg_paper_id: Optional[str]


class RepoAnalyzerInput(TypedDict, total=False):
    """Input contract for RepoAnalyzerAgent."""
    repo_url: str  # GitHub URL or local path (REQUIRED)
    knowledge_graph: Any  # Optional KnowledgeGraph to populate


class RepoAnalyzerOutput(TypedDict, total=False):
    """Output contract for RepoAnalyzerAgent."""
    name: str
    url: str
    overview: Dict[str, Any]
    key_components: List[Dict[str, Any]]
    entry_points: List[Dict[str, Any]]
    dependencies: Dict[str, Any]
    setup_complexity: Dict[str, Any]
    compute_requirements: Dict[str, Any]
    stats: Dict[str, int]
    _structure: Dict[str, Any]
    _code_elements: Dict[str, Any]
    _repo_path: str
    _kg_repo_id: Optional[str]


class SemanticMapperInput(TypedDict, total=False):
    """Input contract for SemanticMapper."""
    paper_data: Dict[str, Any]  # REQUIRED
    repo_data: Dict[str, Any]   # REQUIRED
    knowledge_graph: Any  # Optional KnowledgeGraph


class MappingResult(TypedDict, total=False):
    """Single concept-to-code mapping."""
    concept_name: str
    concept_description: str
    code_element: str
    code_file: str
    confidence: float
    match_signals: Dict[str, float]
    evidence: List[str]
    reasoning: str


class SemanticMapperOutput(TypedDict, total=False):
    """Output contract for SemanticMapper."""
    mappings: List[MappingResult]
    unmapped_concepts: List[Dict[str, Any]]
    unmapped_code: List[Dict[str, Any]]


class CodingAgentInput(TypedDict, total=False):
    """Input contract for CodingAgent."""
    mappings: List[Dict[str, Any]]  # REQUIRED
    repo_data: Dict[str, Any]       # REQUIRED
    knowledge_graph: Any  # Optional KnowledgeGraph
    execute: bool  # Default True


class TestResult(TypedDict, total=False):
    """Single test execution result."""
    concept: str
    code_element: str
    success: bool
    stdout: str
    stderr: str
    execution_time: float
    return_code: int
    output_files: List[str]
    error: str


class CodingAgentOutput(TypedDict, total=False):
    """Output contract for CodingAgent."""
    scripts: List[Dict[str, Any]]
    results: List[TestResult]
    language: str
    summary: Dict[str, Any]


# ============================================================================
# Agent Statistics
# ============================================================================

@dataclass
class AgentStats:
    """Statistics for agent operations."""
    operations: int = 0
    total_duration_ms: float = 0
    errors: int = 0
    last_operation: Optional[datetime] = None

    @property
    def avg_duration_ms(self) -> float:
        """Average duration per operation in milliseconds."""
        return self.total_duration_ms / self.operations if self.operations > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operations": self.operations,
            "operation_count": self.operations,  # Alias for compatibility
            "total_duration_ms": self.total_duration_ms,
            "avg_duration_ms": self.avg_duration_ms,
            "errors": self.errors,
            "last_operation": self.last_operation.isoformat() if self.last_operation else None
        }


# ============================================================================
# Agent Protocol (Runtime Checkable)
# ============================================================================

@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol that all agents must implement."""

    @property
    def agent_id(self) -> str:
        """Unique identifier for this agent instance."""
        ...

    @property
    def agent_type(self) -> str:
        """Type name of this agent class."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name for this agent."""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get agent operation statistics."""
        ...

    async def process(self, **kwargs) -> Dict[str, Any]:
        """Process input and return output."""
        ...


# ============================================================================
# Validation Helpers
# ============================================================================

class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def validate_paper_parser_input(kwargs: Dict[str, Any]) -> None:
    """Validate PaperParserAgent input."""
    if not kwargs.get("paper_source"):
        raise ValidationError("paper_source is required for PaperParserAgent")


def validate_repo_analyzer_input(kwargs: Dict[str, Any]) -> None:
    """Validate RepoAnalyzerAgent input."""
    if not kwargs.get("repo_url"):
        raise ValidationError("repo_url is required for RepoAnalyzerAgent")


def validate_semantic_mapper_input(kwargs: Dict[str, Any]) -> None:
    """Validate SemanticMapper input."""
    if not kwargs.get("paper_data"):
        raise ValidationError("paper_data is required for SemanticMapper")
    if not kwargs.get("repo_data"):
        raise ValidationError("repo_data is required for SemanticMapper")


def validate_coding_agent_input(kwargs: Dict[str, Any]) -> None:
    """Validate CodingAgent input."""
    if not kwargs.get("mappings"):
        raise ValidationError("mappings is required for CodingAgent")
    if not kwargs.get("repo_data"):
        raise ValidationError("repo_data is required for CodingAgent")


# ============================================================================
# Code Element Types (for multi-language parsing)
# ============================================================================

@dataclass
class CodeElement:
    """Represents a parsed code element from any language."""
    name: str
    element_type: str  # "class", "function", "struct", "type", etc.
    file_path: str
    line_number: int
    docstring: str = ""
    signature: str = ""
    args: List[str] = field(default_factory=list)
    return_type: str = ""
    decorators: List[str] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)  # For classes
    methods: List[str] = field(default_factory=list)  # For classes
    language: str = "python"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "type": self.element_type,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "docstring": self.docstring,
            "signature": self.signature,
            "args": self.args,
            "return_type": self.return_type,
            "decorators": self.decorators,
            "bases": self.bases,
            "methods": self.methods,
            "language": self.language
        }


@dataclass
class ParsedFile:
    """Result of parsing a single source file."""
    file_path: str
    language: str
    classes: List[CodeElement] = field(default_factory=list)
    functions: List[CodeElement] = field(default_factory=list)
    imports: List[Dict[str, Any]] = field(default_factory=list)
    constants: List[Dict[str, Any]] = field(default_factory=list)
    parse_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "language": self.language,
            "classes": [c.to_dict() for c in self.classes],
            "functions": [f.to_dict() for f in self.functions],
            "imports": self.imports,
            "constants": self.constants,
            "parse_errors": self.parse_errors
        }
