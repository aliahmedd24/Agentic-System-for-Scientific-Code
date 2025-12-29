"""
Tests for Agent Protocols - Pydantic model validation and contracts.
"""

import pytest
from pydantic import ValidationError


# ============================================================================
# Concept Model Tests
# ============================================================================

class TestConcept:
    """Tests for Concept model."""

    def test_concept_creation_minimal(self):
        """Test creating a concept with minimal fields."""
        from agents.protocols import Concept
        concept = Concept(name="attention")
        assert concept.name == "attention"
        assert concept.description == ""
        assert concept.importance == "medium"
        assert concept.related_sections == []
        assert concept.likely_names == []

    def test_concept_creation_full(self):
        """Test creating a concept with all fields."""
        from agents.protocols import Concept
        concept = Concept(
            name="attention",
            description="Self-attention mechanism",
            importance="high",
            related_sections=["Section 3", "Section 4"],
            likely_names=["Attention", "SelfAttention", "MultiHeadAttention"]
        )
        assert concept.name == "attention"
        assert concept.description == "Self-attention mechanism"
        assert concept.importance == "high"
        assert len(concept.related_sections) == 2
        assert len(concept.likely_names) == 3

    def test_concept_extra_fields_forbidden(self):
        """Test that extra fields are rejected."""
        from agents.protocols import Concept
        with pytest.raises(ValidationError):
            Concept(name="test", unknown_field="value")


# ============================================================================
# Algorithm Model Tests
# ============================================================================

class TestAlgorithm:
    """Tests for Algorithm model."""

    def test_algorithm_creation_minimal(self):
        """Test creating an algorithm with minimal fields."""
        from agents.protocols import Algorithm
        algo = Algorithm(name="transformer")
        assert algo.name == "transformer"
        assert algo.description == ""
        assert algo.pseudocode == ""
        assert algo.complexity == ""

    def test_algorithm_creation_full(self):
        """Test creating an algorithm with all fields."""
        from agents.protocols import Algorithm
        algo = Algorithm(
            name="transformer",
            description="Self-attention based architecture",
            pseudocode="1. Compute Q, K, V\n2. Apply attention",
            complexity="O(n^2)",
            inputs=["embeddings", "mask"],
            outputs=["hidden_states"]
        )
        assert algo.name == "transformer"
        assert algo.complexity == "O(n^2)"
        assert len(algo.inputs) == 2
        assert len(algo.outputs) == 1


# ============================================================================
# CodeEntityModel Tests
# ============================================================================

class TestCodeEntityModel:
    """Tests for CodeEntityModel."""

    def test_code_entity_creation(self):
        """Test creating a code entity."""
        from agents.protocols import CodeEntityModel
        entity = CodeEntityModel(
            name="MultiHeadAttention",
            entity_type="class",
            file_path="model.py"
        )
        assert entity.name == "MultiHeadAttention"
        assert entity.entity_type == "class"
        assert entity.file_path == "model.py"
        assert entity.line_number == 0
        assert entity.language == "python"

    def test_code_entity_full(self):
        """Test code entity with all fields."""
        from agents.protocols import CodeEntityModel
        entity = CodeEntityModel(
            name="forward",
            entity_type="function",
            file_path="model.py",
            line_number=42,
            docstring="Forward pass of the model.",
            signature="def forward(self, x: Tensor) -> Tensor",
            language="python"
        )
        assert entity.line_number == 42
        assert "Forward pass" in entity.docstring


# ============================================================================
# PaperParserInput/Output Tests
# ============================================================================

class TestPaperParserInput:
    """Tests for PaperParserInput model."""

    def test_input_requires_paper_source(self):
        """Test that paper_source is required."""
        from agents.protocols import PaperParserInput
        with pytest.raises(ValidationError):
            PaperParserInput()

    def test_input_with_paper_source(self):
        """Test valid input with paper_source."""
        from agents.protocols import PaperParserInput
        inp = PaperParserInput(paper_source="2301.00001")
        assert inp.paper_source == "2301.00001"
        assert inp.knowledge_graph is None

    def test_input_with_knowledge_graph(self):
        """Test input with optional knowledge_graph."""
        from agents.protocols import PaperParserInput
        from core.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph()
        inp = PaperParserInput(paper_source="test.pdf", knowledge_graph=kg)
        assert inp.knowledge_graph is not None


class TestPaperParserOutput:
    """Tests for PaperParserOutput model."""

    def test_output_requires_title_and_abstract(self):
        """Test that title and abstract are required."""
        from agents.protocols import PaperParserOutput
        with pytest.raises(ValidationError):
            PaperParserOutput(title="Test")  # Missing abstract

    def test_output_abstract_minimum_length(self):
        """Test that abstract must be at least 100 characters."""
        from agents.protocols import PaperParserOutput
        with pytest.raises(ValidationError) as exc_info:
            PaperParserOutput(
                title="Test Paper",
                abstract="Too short"
            )
        assert "at least 100 characters" in str(exc_info.value)

    def test_output_valid_creation(self):
        """Test valid output creation."""
        from agents.protocols import PaperParserOutput
        output = PaperParserOutput(
            title="Test Paper on Deep Learning",
            abstract="A" * 150  # 150 characters
        )
        assert output.title == "Test Paper on Deep Learning"
        assert len(output.abstract) == 150
        assert output.authors == []
        assert output.key_concepts == []

    def test_output_full_creation(self, valid_paper_parser_output):
        """Test output with all fields using fixture."""
        assert valid_paper_parser_output.title == "Test Paper on Deep Learning"
        assert len(valid_paper_parser_output.authors) == 2
        assert len(valid_paper_parser_output.key_concepts) == 1
        assert len(valid_paper_parser_output.algorithms) == 1


# ============================================================================
# RepoAnalyzerInput/Output Tests
# ============================================================================

class TestRepoAnalyzerInput:
    """Tests for RepoAnalyzerInput model."""

    def test_input_requires_repo_url(self):
        """Test that repo_url is required."""
        from agents.protocols import RepoAnalyzerInput
        with pytest.raises(ValidationError):
            RepoAnalyzerInput()

    def test_input_with_repo_url(self):
        """Test valid input with repo_url."""
        from agents.protocols import RepoAnalyzerInput
        inp = RepoAnalyzerInput(repo_url="https://github.com/test/repo")
        assert inp.repo_url == "https://github.com/test/repo"

    def test_input_with_local_path(self):
        """Test input with local path."""
        from agents.protocols import RepoAnalyzerInput
        inp = RepoAnalyzerInput(repo_url="/path/to/local/repo")
        assert inp.repo_url == "/path/to/local/repo"


class TestRepoAnalyzerOutput:
    """Tests for RepoAnalyzerOutput model."""

    def test_output_requires_name(self):
        """Test that name is required."""
        from agents.protocols import RepoAnalyzerOutput
        with pytest.raises(ValidationError):
            RepoAnalyzerOutput()

    def test_output_minimal_creation(self):
        """Test minimal output creation."""
        from agents.protocols import RepoAnalyzerOutput
        output = RepoAnalyzerOutput(name="test-repo")
        assert output.name == "test-repo"
        assert output.url == ""
        assert output.key_components == []

    def test_output_full_creation(self, valid_repo_analyzer_output):
        """Test output with all fields using fixture."""
        assert valid_repo_analyzer_output.name == "test-repo"
        assert valid_repo_analyzer_output.url == "https://github.com/test/repo"
        assert len(valid_repo_analyzer_output.key_components) == 1
        assert valid_repo_analyzer_output.stats.total_files == 10


# ============================================================================
# MappingResult Tests
# ============================================================================

class TestMappingResult:
    """Tests for MappingResult model."""

    def test_mapping_requires_fields(self):
        """Test that required fields must be provided."""
        from agents.protocols import MappingResult
        with pytest.raises(ValidationError):
            MappingResult()

    def test_mapping_confidence_validation(self):
        """Test confidence must be between 0 and 1."""
        from agents.protocols import MappingResult
        # Valid confidence
        mapping = MappingResult(
            concept_name="attention",
            code_element="Attention",
            confidence=0.85
        )
        assert mapping.confidence == 0.85

    def test_mapping_confidence_out_of_range_high(self):
        """Test confidence > 1 is rejected."""
        from agents.protocols import MappingResult
        with pytest.raises(ValidationError):
            MappingResult(
                concept_name="attention",
                code_element="Attention",
                confidence=1.5
            )

    def test_mapping_confidence_out_of_range_low(self):
        """Test confidence < 0 is rejected."""
        from agents.protocols import MappingResult
        with pytest.raises(ValidationError):
            MappingResult(
                concept_name="attention",
                code_element="Attention",
                confidence=-0.1
            )

    def test_mapping_full_creation(self, valid_mapping_result):
        """Test mapping with all fields using fixture."""
        assert valid_mapping_result.concept_name == "attention"
        assert valid_mapping_result.code_element == "MultiHeadAttention"
        assert valid_mapping_result.confidence == 0.85
        assert len(valid_mapping_result.evidence) == 1


class TestMatchSignals:
    """Tests for MatchSignals model."""

    def test_match_signals_defaults(self):
        """Test match signals default values."""
        from agents.protocols import MatchSignals
        signals = MatchSignals()
        assert signals.lexical == 0.0
        assert signals.semantic == 0.0
        assert signals.documentary == 0.0

    def test_match_signals_validation(self):
        """Test match signals must be 0-1."""
        from agents.protocols import MatchSignals
        with pytest.raises(ValidationError):
            MatchSignals(lexical=1.5)

    def test_match_signals_full(self):
        """Test match signals with all values."""
        from agents.protocols import MatchSignals
        signals = MatchSignals(lexical=0.7, semantic=0.9, documentary=0.5)
        assert signals.lexical == 0.7
        assert signals.semantic == 0.9
        assert signals.documentary == 0.5


# ============================================================================
# SemanticMapperInput/Output Tests
# ============================================================================

class TestSemanticMapperInput:
    """Tests for SemanticMapperInput model."""

    def test_input_requires_paper_data(self):
        """Test that paper_data is required."""
        from agents.protocols import SemanticMapperInput
        with pytest.raises(ValidationError):
            SemanticMapperInput(repo_data={"name": "test"})

    def test_input_requires_repo_data(self):
        """Test that repo_data is required."""
        from agents.protocols import SemanticMapperInput
        with pytest.raises(ValidationError):
            SemanticMapperInput(paper_data={"title": "test"})

    def test_input_valid_creation(self):
        """Test valid input creation."""
        from agents.protocols import SemanticMapperInput
        inp = SemanticMapperInput(
            paper_data={"title": "Test Paper"},
            repo_data={"name": "test-repo"}
        )
        assert inp.paper_data is not None
        assert inp.repo_data is not None


class TestSemanticMapperOutput:
    """Tests for SemanticMapperOutput model."""

    def test_output_defaults(self):
        """Test output default values."""
        from agents.protocols import SemanticMapperOutput
        output = SemanticMapperOutput()
        assert output.mappings == []
        assert output.unmapped_concepts == []
        assert output.unmapped_code == []

    def test_output_with_mappings(self, valid_mapping_result):
        """Test output with mappings."""
        from agents.protocols import SemanticMapperOutput
        output = SemanticMapperOutput(mappings=[valid_mapping_result])
        assert len(output.mappings) == 1
        assert output.mappings[0].concept_name == "attention"


# ============================================================================
# CodingAgentInput/Output Tests
# ============================================================================

class TestCodingAgentInput:
    """Tests for CodingAgentInput model."""

    def test_input_requires_mappings(self, valid_mapping_result):
        """Test that mappings is required."""
        from agents.protocols import CodingAgentInput
        with pytest.raises(ValidationError):
            CodingAgentInput(repo_data={"name": "test"})

    def test_input_requires_repo_data(self, valid_mapping_result):
        """Test that repo_data is required."""
        from agents.protocols import CodingAgentInput
        with pytest.raises(ValidationError):
            CodingAgentInput(mappings=[valid_mapping_result])

    def test_input_valid_creation(self, valid_mapping_result):
        """Test valid input creation."""
        from agents.protocols import CodingAgentInput
        inp = CodingAgentInput(
            mappings=[valid_mapping_result],
            repo_data={"name": "test-repo"}
        )
        assert len(inp.mappings) == 1
        assert inp.execute is True  # Default


class TestCodingAgentOutput:
    """Tests for CodingAgentOutput model."""

    def test_output_defaults(self):
        """Test output default values."""
        from agents.protocols import CodingAgentOutput
        output = CodingAgentOutput()
        assert output.scripts == []
        assert output.results == []
        assert output.language == "python"

    def test_output_with_results(self):
        """Test output with test results."""
        from agents.protocols import CodingAgentOutput, TestResult, ExecutionSummary
        result = TestResult(concept="attention", success=True)
        output = CodingAgentOutput(
            results=[result],
            summary=ExecutionSummary(total_tests=1, passed=1)
        )
        assert len(output.results) == 1
        assert output.summary.passed == 1


# ============================================================================
# TestResult and GeneratedScript Tests
# ============================================================================

class TestTestResult:
    """Tests for TestResult model."""

    def test_result_requires_concept(self):
        """Test that concept is required."""
        from agents.protocols import TestResult
        with pytest.raises(ValidationError):
            TestResult()

    def test_result_defaults(self):
        """Test result default values."""
        from agents.protocols import TestResult
        result = TestResult(concept="attention")
        assert result.concept == "attention"
        assert result.success is False
        assert result.stdout == ""
        assert result.return_code == -1

    def test_result_full_creation(self):
        """Test result with all fields."""
        from agents.protocols import TestResult
        result = TestResult(
            concept="attention",
            code_element="MultiHeadAttention",
            success=True,
            stdout="All tests passed",
            stderr="",
            execution_time=1.5,
            return_code=0,
            isolation_level="subprocess"
        )
        assert result.success is True
        assert result.execution_time == 1.5
        assert result.return_code == 0


class TestGeneratedScript:
    """Tests for GeneratedScript model."""

    def test_script_requires_fields(self):
        """Test required fields."""
        from agents.protocols import GeneratedScript
        with pytest.raises(ValidationError):
            GeneratedScript()

    def test_script_creation(self):
        """Test script creation."""
        from agents.protocols import GeneratedScript
        script = GeneratedScript(
            concept="attention",
            code="import torch\nprint('test')",
            file_name="test_attention.py"
        )
        assert script.concept == "attention"
        assert "import torch" in script.code
        assert script.syntax_valid is True  # Default


# ============================================================================
# AgentStats Tests
# ============================================================================

class TestAgentStats:
    """Tests for AgentStats model."""

    def test_stats_defaults(self):
        """Test stats default values."""
        from agents.protocols import AgentStats
        stats = AgentStats()
        assert stats.operations == 0
        assert stats.total_duration_ms == 0.0
        assert stats.errors == 0
        assert stats.last_operation is None

    def test_stats_avg_duration(self):
        """Test average duration calculation."""
        from agents.protocols import AgentStats
        stats = AgentStats(operations=10, total_duration_ms=1000.0)
        assert stats.avg_duration_ms == 100.0

    def test_stats_avg_duration_zero_operations(self):
        """Test avg duration with zero operations."""
        from agents.protocols import AgentStats
        stats = AgentStats(operations=0, total_duration_ms=0.0)
        assert stats.avg_duration_ms == 0.0


# ============================================================================
# Validation Helper Tests
# ============================================================================

class TestValidationHelpers:
    """Tests for validation helper functions."""

    def test_validate_paper_parser_input_valid(self):
        """Test validating valid paper parser input."""
        from agents.protocols import validate_paper_parser_input
        result = validate_paper_parser_input({"paper_source": "2301.00001"})
        assert result.paper_source == "2301.00001"

    def test_validate_paper_parser_input_missing(self):
        """Test validating missing paper_source."""
        from agents.protocols import validate_paper_parser_input, ValidationError as VE
        with pytest.raises(VE) as exc_info:
            validate_paper_parser_input({})
        assert "paper_source is required" in str(exc_info.value)

    def test_validate_repo_analyzer_input_valid(self):
        """Test validating valid repo analyzer input."""
        from agents.protocols import validate_repo_analyzer_input
        result = validate_repo_analyzer_input({"repo_url": "https://github.com/test/repo"})
        assert result.repo_url == "https://github.com/test/repo"

    def test_validate_repo_analyzer_input_missing(self):
        """Test validating missing repo_url."""
        from agents.protocols import validate_repo_analyzer_input, ValidationError as VE
        with pytest.raises(VE) as exc_info:
            validate_repo_analyzer_input({})
        assert "repo_url is required" in str(exc_info.value)

    def test_validate_semantic_mapper_input_valid(self):
        """Test validating valid semantic mapper input."""
        from agents.protocols import validate_semantic_mapper_input
        result = validate_semantic_mapper_input({
            "paper_data": {"title": "Test"},
            "repo_data": {"name": "test"}
        })
        assert result.paper_data is not None
        assert result.repo_data is not None

    def test_validate_semantic_mapper_input_missing_paper(self):
        """Test validating missing paper_data."""
        from agents.protocols import validate_semantic_mapper_input, ValidationError as VE
        with pytest.raises(VE) as exc_info:
            validate_semantic_mapper_input({"repo_data": {"name": "test"}})
        assert "paper_data is required" in str(exc_info.value)

    def test_validate_semantic_mapper_input_missing_repo(self):
        """Test validating missing repo_data."""
        from agents.protocols import validate_semantic_mapper_input, ValidationError as VE
        with pytest.raises(VE) as exc_info:
            validate_semantic_mapper_input({"paper_data": {"title": "test"}})
        assert "repo_data is required" in str(exc_info.value)

    def test_validate_coding_agent_input_valid(self, valid_mapping_result):
        """Test validating valid coding agent input."""
        from agents.protocols import validate_coding_agent_input
        result = validate_coding_agent_input({
            "mappings": [valid_mapping_result.model_dump()],
            "repo_data": {"name": "test"}
        })
        assert len(result.mappings) == 1

    def test_validate_coding_agent_input_missing_mappings(self):
        """Test validating missing mappings."""
        from agents.protocols import validate_coding_agent_input, ValidationError as VE
        with pytest.raises(VE) as exc_info:
            validate_coding_agent_input({"repo_data": {"name": "test"}})
        assert "mappings is required" in str(exc_info.value)


# ============================================================================
# CodeElement and ParsedFile Tests
# ============================================================================

class TestCodeElement:
    """Tests for CodeElement model."""

    def test_element_requires_fields(self):
        """Test required fields."""
        from agents.protocols import CodeElement
        with pytest.raises(ValidationError):
            CodeElement()

    def test_element_creation(self):
        """Test element creation."""
        from agents.protocols import CodeElement
        element = CodeElement(
            name="MultiHeadAttention",
            element_type="class",
            file_path="model.py"
        )
        assert element.name == "MultiHeadAttention"
        assert element.element_type == "class"
        assert element.args == []
        assert element.decorators == []

    def test_element_full(self):
        """Test element with all fields."""
        from agents.protocols import CodeElement
        element = CodeElement(
            name="forward",
            element_type="function",
            file_path="model.py",
            line_number=42,
            docstring="Forward pass",
            signature="def forward(self, x)",
            args=["self", "x"],
            return_type="Tensor",
            decorators=["@torch.no_grad()"],
            language="python"
        )
        assert element.line_number == 42
        assert len(element.args) == 2
        assert len(element.decorators) == 1


class TestParsedFile:
    """Tests for ParsedFile model."""

    def test_parsed_file_requires_fields(self):
        """Test required fields."""
        from agents.protocols import ParsedFile
        with pytest.raises(ValidationError):
            ParsedFile()

    def test_parsed_file_creation(self):
        """Test parsed file creation."""
        from agents.protocols import ParsedFile
        pf = ParsedFile(file_path="model.py", language="python")
        assert pf.file_path == "model.py"
        assert pf.language == "python"
        assert pf.classes == []
        assert pf.functions == []

    def test_parsed_file_with_elements(self):
        """Test parsed file with code elements."""
        from agents.protocols import ParsedFile, CodeElement
        element = CodeElement(
            name="Model",
            element_type="class",
            file_path="model.py"
        )
        pf = ParsedFile(
            file_path="model.py",
            language="python",
            classes=[element]
        )
        assert len(pf.classes) == 1


# ============================================================================
# Methodology and Reproducibility Tests
# ============================================================================

class TestMethodology:
    """Tests for Methodology model."""

    def test_methodology_defaults(self):
        """Test methodology default values."""
        from agents.protocols import Methodology
        m = Methodology()
        assert m.approach == ""
        assert m.datasets == []
        assert m.evaluation_metrics == []
        assert m.baselines == []

    def test_methodology_full(self):
        """Test methodology with all fields."""
        from agents.protocols import Methodology
        m = Methodology(
            approach="Supervised learning",
            datasets=["ImageNet", "COCO"],
            evaluation_metrics=["accuracy", "F1"],
            baselines=["ResNet", "VGG"]
        )
        assert len(m.datasets) == 2
        assert len(m.baselines) == 2


class TestReproducibility:
    """Tests for Reproducibility model."""

    def test_reproducibility_defaults(self):
        """Test reproducibility default values."""
        from agents.protocols import Reproducibility
        r = Reproducibility()
        assert r.code_available is False
        assert r.data_available is False
        assert r.hardware_requirements == ""

    def test_reproducibility_full(self):
        """Test reproducibility with all fields."""
        from agents.protocols import Reproducibility
        r = Reproducibility(
            code_available=True,
            data_available=True,
            hardware_requirements="GPU with 16GB VRAM",
            estimated_time="2 hours"
        )
        assert r.code_available is True
        assert "GPU" in r.hardware_requirements


# ============================================================================
# Sub-model Tests
# ============================================================================

class TestFileStats:
    """Tests for FileStats model."""

    def test_file_stats_defaults(self):
        """Test file stats default values."""
        from agents.protocols import FileStats
        stats = FileStats()
        assert stats.total_files == 0
        assert stats.code_files == 0
        assert stats.classes == 0
        assert stats.functions == 0

    def test_file_stats_full(self):
        """Test file stats with values."""
        from agents.protocols import FileStats
        stats = FileStats(
            total_files=100,
            code_files=50,
            classes=20,
            functions=150
        )
        assert stats.total_files == 100
        assert stats.functions == 150


class TestDependencyInfo:
    """Tests for DependencyInfo model."""

    def test_dependency_info_defaults(self):
        """Test dependency info default values."""
        from agents.protocols import DependencyInfo
        deps = DependencyInfo()
        assert deps.python == []
        assert deps.julia == []
        assert deps.r == []
        assert deps.javascript == []
        assert deps.system == []

    def test_dependency_info_full(self):
        """Test dependency info with values."""
        from agents.protocols import DependencyInfo
        deps = DependencyInfo(
            python=["torch", "numpy"],
            julia=["Flux"],
            system=["cuda"]
        )
        assert len(deps.python) == 2
        assert len(deps.julia) == 1


class TestExecutionSummary:
    """Tests for ExecutionSummary model."""

    def test_execution_summary_defaults(self):
        """Test execution summary defaults."""
        from agents.protocols import ExecutionSummary
        summary = ExecutionSummary()
        assert summary.total_tests == 0
        assert summary.passed == 0
        assert summary.failed == 0
        assert summary.skipped == 0
        assert summary.total_time == 0.0

    def test_execution_summary_full(self):
        """Test execution summary with values."""
        from agents.protocols import ExecutionSummary
        summary = ExecutionSummary(
            total_tests=10,
            passed=8,
            failed=1,
            skipped=1,
            total_time=5.5
        )
        assert summary.total_tests == 10
        assert summary.passed == 8
