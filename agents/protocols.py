"""
Agent Protocols - Type definitions and contracts for all agents.

This module defines the input/output contracts for each agent using Pydantic
BaseModels for strict typing, validation, and schema generation.
"""

from typing import Protocol, Dict, Any, List, Optional, runtime_checkable
from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, ConfigDict

from core.resource_estimator import ResourceEstimate


# ============================================================================
# Shared Sub-Models
# ============================================================================

class Concept(BaseModel):
    """Represents a key concept extracted from a paper."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="The name of the concept")
    description: str = Field("", description="Detailed description of the concept")
    importance: str = Field("medium", description="Importance level: low, medium, high, critical")
    related_sections: List[str] = Field(default_factory=list, description="Paper sections where this concept appears")
    likely_names: List[str] = Field(default_factory=list, description="Possible names in code implementations")


class Algorithm(BaseModel):
    """Represents an algorithm described in a paper."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Algorithm name")
    description: str = Field("", description="What the algorithm does")
    pseudocode: str = Field("", description="Pseudocode or steps if available")
    complexity: str = Field("", description="Time/space complexity if mentioned")
    inputs: List[str] = Field(default_factory=list, description="Required inputs")
    outputs: List[str] = Field(default_factory=list, description="Expected outputs")


class CodeEntityModel(BaseModel):
    """Represents a code element (class, function, etc.) as a Pydantic model."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Type: class, function, struct, type, etc.")
    file_path: str = Field(..., description="Path to the source file")
    line_number: int = Field(0, description="Line number where entity is defined")
    docstring: str = Field("", description="Documentation string if available")
    signature: str = Field("", description="Function/method signature")
    language: str = Field("python", description="Programming language")


# ============================================================================
# Paper Parser Models
# ============================================================================

class Methodology(BaseModel):
    """Paper methodology details."""
    model_config = ConfigDict(extra="forbid")

    approach: str = Field("", description="Overall methodological approach")
    datasets: List[str] = Field(default_factory=list, description="Datasets used")
    evaluation_metrics: List[str] = Field(default_factory=list, description="Metrics used for evaluation")
    baselines: List[str] = Field(default_factory=list, description="Baseline methods compared against")


class Reproducibility(BaseModel):
    """Reproducibility assessment of the paper."""
    model_config = ConfigDict(extra="forbid")

    code_available: bool = Field(False, description="Whether code is publicly available")
    data_available: bool = Field(False, description="Whether data is publicly available")
    hardware_requirements: str = Field("", description="Hardware requirements mentioned")
    estimated_time: str = Field("", description="Estimated reproduction time")


class ExpectedImplementation(BaseModel):
    """Expected implementation for a paper component."""
    model_config = ConfigDict(extra="forbid")

    component_name: str = Field(..., description="Name of the component to implement")
    description: str = Field("", description="What this component should do")
    priority: str = Field("medium", description="Implementation priority: low, medium, high")
    dependencies: List[str] = Field(default_factory=list, description="Dependencies on other components")


class SourceMetadata(BaseModel):
    """Metadata about the paper source."""
    model_config = ConfigDict(extra="forbid")

    source_type: str = Field("", description="Type: arxiv, url, file")
    arxiv_id: Optional[str] = Field(None, description="arXiv ID if applicable")
    url: Optional[str] = Field(None, description="Source URL if applicable")
    file_path: Optional[str] = Field(None, description="Local file path if applicable")
    extraction_date: Optional[str] = Field(None, description="When the paper was extracted")


class PaperParserInput(BaseModel):
    """Input contract for PaperParserAgent."""
    model_config = ConfigDict(extra="forbid")

    paper_source: str = Field(..., description="arXiv ID, URL, or file path (REQUIRED)")
    knowledge_graph: Optional[Any] = Field(None, description="Optional KnowledgeGraph to populate")


class PaperParserOutput(BaseModel):
    """Output contract for PaperParserAgent."""
    model_config = ConfigDict(extra="ignore")  # Allow private fields

    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    abstract: str = Field(..., description="Paper abstract (minimum 100 characters)")
    key_concepts: List[Concept] = Field(default_factory=list, description="Key concepts from the paper")
    algorithms: List[Algorithm] = Field(default_factory=list, description="Algorithms described in the paper")
    methodology: Methodology = Field(default_factory=Methodology, description="Methodology details")
    reproducibility: Reproducibility = Field(default_factory=Reproducibility, description="Reproducibility assessment")
    expected_implementations: List[ExpectedImplementation] = Field(default_factory=list, description="Expected code implementations")
    source_metadata: SourceMetadata = Field(default_factory=SourceMetadata, description="Source metadata")
    _kg_paper_id: Optional[str] = None  # Private field for knowledge graph

    @field_validator('abstract')
    @classmethod
    def validate_abstract_length(cls, v: str) -> str:
        if len(v) < 100:
            raise ValueError(f"Abstract must be at least 100 characters, got {len(v)}")
        return v


# ============================================================================
# Repo Analyzer Models
# ============================================================================

class FileStats(BaseModel):
    """Repository file statistics."""
    model_config = ConfigDict(extra="forbid")

    total_files: int = Field(0, description="Total number of files")
    code_files: int = Field(0, description="Number of code files")
    classes: int = Field(0, description="Number of classes found")
    functions: int = Field(0, description="Number of functions found")


class DependencyInfo(BaseModel):
    """Dependency information for the repository."""
    model_config = ConfigDict(extra="forbid")

    python: List[str] = Field(default_factory=list, description="Python dependencies")
    julia: List[str] = Field(default_factory=list, description="Julia dependencies")
    r: List[str] = Field(default_factory=list, description="R dependencies")
    javascript: List[str] = Field(default_factory=list, description="JavaScript dependencies")
    system: List[str] = Field(default_factory=list, description="System-level dependencies")


class OverviewInfo(BaseModel):
    """Repository overview information."""
    model_config = ConfigDict(extra="forbid")

    purpose: str = Field("", description="Main purpose of the repository")
    architecture: str = Field("", description="Architecture description")
    key_features: List[str] = Field(default_factory=list, description="Key features")


class KeyComponent(BaseModel):
    """Key component in the repository."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Component name")
    path: str = Field("", description="File or directory path")
    description: str = Field("", description="What this component does")
    importance: str = Field("medium", description="Importance level")


class EntryPoint(BaseModel):
    """Entry point for the repository."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Entry point name")
    path: str = Field(..., description="File path")
    description: str = Field("", description="What this entry point does")
    arguments: List[str] = Field(default_factory=list, description="Expected arguments")


class SetupComplexity(BaseModel):
    """Setup complexity assessment."""
    model_config = ConfigDict(extra="forbid")

    level: str = Field("medium", description="Complexity level: easy, medium, hard, expert")
    steps: List[str] = Field(default_factory=list, description="Setup steps required")
    estimated_time: str = Field("", description="Estimated setup time")


class ComputeRequirements(BaseModel):
    """Compute requirements for the repository."""
    model_config = ConfigDict(extra="forbid")

    cpu_cores: int = Field(1, description="Recommended CPU cores")
    memory_gb: float = Field(4.0, description="Recommended RAM in GB")
    gpu_required: bool = Field(False, description="Whether GPU is required")
    gpu_memory_gb: float = Field(0.0, description="GPU memory if required")


class RepoAnalyzerInput(BaseModel):
    """Input contract for RepoAnalyzerAgent."""
    model_config = ConfigDict(extra="forbid")

    repo_url: str = Field(..., description="GitHub URL or local path (REQUIRED)")
    knowledge_graph: Optional[Any] = Field(None, description="Optional KnowledgeGraph to populate")


class RepoAnalyzerOutput(BaseModel):
    """Output contract for RepoAnalyzerAgent."""
    model_config = ConfigDict(extra="ignore")  # Allow private fields

    name: str = Field(..., description="Repository name")
    url: str = Field("", description="Repository URL")
    overview: OverviewInfo = Field(default_factory=OverviewInfo, description="Repository overview")
    key_components: List[KeyComponent] = Field(default_factory=list, description="Key components")
    entry_points: List[EntryPoint] = Field(default_factory=list, description="Entry points")
    dependencies: DependencyInfo = Field(default_factory=DependencyInfo, description="Dependencies")
    setup_complexity: SetupComplexity = Field(default_factory=SetupComplexity, description="Setup complexity")
    compute_requirements: ComputeRequirements = Field(default_factory=ComputeRequirements, description="Compute requirements")
    stats: FileStats = Field(default_factory=FileStats, description="File statistics")
    _structure: Dict[str, Any] = {}  # Private: raw structure data
    _code_elements: Dict[str, Any] = {}  # Private: raw code elements
    _repo_path: str = ""  # Private: local path
    _kg_repo_id: Optional[str] = None  # Private: knowledge graph ID


# ============================================================================
# Semantic Mapper Models
# ============================================================================

class MatchSignals(BaseModel):
    """Match signals for a mapping."""
    model_config = ConfigDict(extra="forbid")

    lexical: float = Field(0.0, ge=0.0, le=1.0, description="Lexical similarity score")
    semantic: float = Field(0.0, ge=0.0, le=1.0, description="Semantic similarity score")
    documentary: float = Field(0.0, ge=0.0, le=1.0, description="Documentary evidence score")


class MappingResult(BaseModel):
    """Single concept-to-code mapping."""
    model_config = ConfigDict(extra="forbid")

    concept_name: str = Field(..., description="Name of the paper concept")
    concept_description: str = Field("", description="Description of the concept")
    code_element: str = Field(..., description="Name of the matching code element")
    code_file: str = Field("", description="File containing the code element")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    match_signals: MatchSignals = Field(default_factory=MatchSignals, description="Individual match signal scores")
    evidence: List[str] = Field(default_factory=list, description="Evidence supporting the mapping")
    reasoning: str = Field("", description="Reasoning for this mapping")

    @field_validator('confidence')
    @classmethod
    def validate_confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {v}")
        return v


class UnmappedItem(BaseModel):
    """An item that could not be mapped."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the unmapped item")
    description: str = Field("", description="Description of the item")
    reason: str = Field("", description="Why it could not be mapped")


class SemanticMapperInput(BaseModel):
    """Input contract for SemanticMapper."""
    model_config = ConfigDict(extra="forbid")

    paper_data: Any = Field(..., description="Paper parser output (REQUIRED)")
    repo_data: Any = Field(..., description="Repo analyzer output (REQUIRED)")
    knowledge_graph: Optional[Any] = Field(None, description="Optional KnowledgeGraph")


class SemanticMapperOutput(BaseModel):
    """Output contract for SemanticMapper."""
    model_config = ConfigDict(extra="forbid")

    mappings: List[MappingResult] = Field(default_factory=list, description="Concept-to-code mappings")
    unmapped_concepts: List[UnmappedItem] = Field(default_factory=list, description="Paper concepts without code matches")
    unmapped_code: List[UnmappedItem] = Field(default_factory=list, description="Code elements without paper matches")


# ============================================================================
# Coding Agent Models
# ============================================================================

class TestResult(BaseModel):
    """Single test execution result."""
    model_config = ConfigDict(extra="forbid")

    concept: str = Field(..., description="Concept being tested")
    code_element: str = Field("", description="Code element being tested")
    success: bool = Field(False, description="Whether the test passed")
    stdout: str = Field("", description="Standard output from test execution")
    stderr: str = Field("", description="Standard error from test execution")
    execution_time: float = Field(0.0, description="Execution time in seconds")
    return_code: int = Field(-1, description="Process return code")
    output_files: List[str] = Field(default_factory=list, description="Generated output files")
    error: str = Field("", description="Error message if test failed")
    isolation_level: str = Field("none", description="Sandbox isolation level used")


class GeneratedScript(BaseModel):
    """A generated test script."""
    model_config = ConfigDict(extra="forbid")

    concept: str = Field(..., description="Concept this script tests")
    code_element: str = Field("", description="Code element being tested")
    code_file: str = Field("", description="Source file of the code element")
    confidence: float = Field(0.0, description="Confidence of the mapping")
    code: str = Field(..., description="Generated test code")
    file_name: str = Field(..., description="Output file name for the script")
    language: str = Field("python", description="Programming language")
    syntax_valid: bool = Field(True, description="Whether syntax validation passed")
    import_valid: bool = Field(True, description="Whether import validation passed")
    import_path: str = Field("", description="Import statement used")
    validation_error: Optional[str] = Field(None, description="Validation error if any")


class ExecutionSummary(BaseModel):
    """Summary of test execution."""
    model_config = ConfigDict(extra="forbid")

    total_tests: int = Field(0, description="Total number of tests run")
    passed: int = Field(0, description="Number of tests that passed")
    failed: int = Field(0, description="Number of tests that failed")
    skipped: int = Field(0, description="Number of tests skipped")
    total_time: float = Field(0.0, description="Total execution time in seconds")


# ResourceEstimate is imported from core.resource_estimator


class CodingAgentInput(BaseModel):
    """Input contract for CodingAgent."""
    model_config = ConfigDict(extra="forbid")

    mappings: List[MappingResult] = Field(..., description="Concept-to-code mappings (REQUIRED)")
    repo_data: Any = Field(..., description="Repo analyzer output (REQUIRED)")
    knowledge_graph: Optional[Any] = Field(None, description="Optional KnowledgeGraph")
    execute: bool = Field(True, description="Whether to execute generated tests")


class CodingAgentOutput(BaseModel):
    """Output contract for CodingAgent."""
    model_config = ConfigDict(extra="forbid")

    scripts: List[GeneratedScript] = Field(default_factory=list, description="Generated test scripts")
    results: List[TestResult] = Field(default_factory=list, description="Test execution results")
    language: str = Field("python", description="Primary language detected")
    summary: ExecutionSummary = Field(default_factory=ExecutionSummary, description="Execution summary")
    resource_estimate: Optional[ResourceEstimate] = Field(None, description="Resource usage estimate")


# ============================================================================
# Agent Statistics
# ============================================================================

class AgentStats(BaseModel):
    """Statistics for agent operations."""
    model_config = ConfigDict(extra="forbid")

    operations: int = Field(0, description="Number of operations performed")
    total_duration_ms: float = Field(0.0, description="Total duration in milliseconds")
    errors: int = Field(0, description="Number of errors encountered")
    last_operation: Optional[datetime] = Field(None, description="Timestamp of last operation")

    @property
    def avg_duration_ms(self) -> float:
        """Average duration per operation in milliseconds."""
        return self.total_duration_ms / self.operations if self.operations > 0 else 0


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


def validate_paper_parser_input(kwargs: Dict[str, Any]) -> PaperParserInput:
    """Validate and parse PaperParserAgent input."""
    if not kwargs.get("paper_source"):
        raise ValidationError("paper_source is required for PaperParserAgent")
    return PaperParserInput(**kwargs)


def validate_repo_analyzer_input(kwargs: Dict[str, Any]) -> RepoAnalyzerInput:
    """Validate and parse RepoAnalyzerAgent input."""
    if not kwargs.get("repo_url"):
        raise ValidationError("repo_url is required for RepoAnalyzerAgent")
    return RepoAnalyzerInput(**kwargs)


def validate_semantic_mapper_input(kwargs: Dict[str, Any]) -> SemanticMapperInput:
    """Validate and parse SemanticMapper input."""
    if not kwargs.get("paper_data"):
        raise ValidationError("paper_data is required for SemanticMapper")
    if not kwargs.get("repo_data"):
        raise ValidationError("repo_data is required for SemanticMapper")
    return SemanticMapperInput(**kwargs)


def validate_coding_agent_input(kwargs: Dict[str, Any]) -> CodingAgentInput:
    """Validate and parse CodingAgent input."""
    if not kwargs.get("mappings"):
        raise ValidationError("mappings is required for CodingAgent")
    if not kwargs.get("repo_data"):
        raise ValidationError("repo_data is required for CodingAgent")
    return CodingAgentInput(**kwargs)


# ============================================================================
# Code Element Types (for multi-language parsing)
# ============================================================================

class CodeElement(BaseModel):
    """Represents a parsed code element from any language."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Element name")
    element_type: str = Field(..., description="Type: class, function, struct, type, etc.")
    file_path: str = Field(..., description="Path to source file")
    line_number: int = Field(0, description="Line number where defined")
    docstring: str = Field("", description="Documentation string")
    signature: str = Field("", description="Function/method signature")
    args: List[str] = Field(default_factory=list, description="Function arguments")
    return_type: str = Field("", description="Return type annotation")
    decorators: List[str] = Field(default_factory=list, description="Decorators applied")
    bases: List[str] = Field(default_factory=list, description="Base classes (for classes)")
    methods: List[str] = Field(default_factory=list, description="Methods (for classes)")
    language: str = Field("python", description="Programming language")


class ParsedFile(BaseModel):
    """Result of parsing a single source file."""
    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(..., description="Path to the parsed file")
    language: str = Field(..., description="Programming language")
    classes: List[CodeElement] = Field(default_factory=list, description="Parsed classes")
    functions: List[CodeElement] = Field(default_factory=list, description="Parsed functions")
    imports: List[Dict[str, Any]] = Field(default_factory=list, description="Import statements")
    constants: List[Dict[str, Any]] = Field(default_factory=list, description="Constants defined")
    parse_errors: List[str] = Field(default_factory=list, description="Parsing errors encountered")
