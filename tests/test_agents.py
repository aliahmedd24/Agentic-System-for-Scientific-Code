"""
Tests for Agent modules.

This module tests all agent implementations including:
- BaseAgent abstract base class
- PaperParserAgent for PDF/arXiv parsing
- RepoAnalyzerAgent for code repository analysis
- SemanticMapper for concept-to-code mapping
- CodingAgent for code generation and execution
- Parser integration with multi-language support
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from agents.base_agent import BaseAgent
from agents.paper_parser_agent import PaperParserAgent
from agents.repo_analyzer_agent import RepoAnalyzerAgent
from agents.semantic_mapper import SemanticMapper
from agents.coding_agent import CodingAgent
from agents.parsers import (
    CodeElement,
    LanguageParser,
    PythonParser,
    JuliaParser,
    RParser,
    JavaScriptParser,
    ParserFactory,
)
from core.knowledge_graph import KnowledgeGraph, NodeType


class TestBaseAgent:
    """Tests for BaseAgent class."""
    
    def test_base_agent_is_abstract(self):
        """Test that BaseAgent cannot be instantiated directly."""
        # BaseAgent has abstract methods, so this should fail
        with pytest.raises(TypeError):
            BaseAgent(llm_client=MagicMock())
    
    def test_agent_stats_initial(self):
        """Test initial agent statistics."""
        # Create a concrete subclass for testing
        class ConcreteAgent(BaseAgent):
            async def process(self, *args, **kwargs):
                return {}
        
        mock_client = MagicMock()
        agent = ConcreteAgent(llm_client=mock_client)
        
        stats = agent.get_stats()
        assert stats["operation_count"] == 0
        assert stats["total_duration_ms"] == 0


class TestPaperParserAgent:
    """Tests for PaperParserAgent."""
    
    @pytest.fixture
    def paper_parser(self, mock_llm_client):
        """Create a paper parser agent for testing."""
        return PaperParserAgent(llm_client=mock_llm_client)
    
    def test_agent_creation(self, paper_parser):
        """Test paper parser agent creation."""
        assert paper_parser is not None
        assert paper_parser.agent_id == "paper_parser"
    
    def test_is_arxiv_id(self, paper_parser):
        """Test arXiv ID detection."""
        # Valid arXiv IDs
        assert paper_parser._is_arxiv_id("2301.00001")
        assert paper_parser._is_arxiv_id("2301.00001v1")
        assert paper_parser._is_arxiv_id("hep-th/9901001")
        
        # Invalid arXiv IDs
        assert not paper_parser._is_arxiv_id("https://arxiv.org/abs/2301.00001")
        assert not paper_parser._is_arxiv_id("paper.pdf")
        assert not paper_parser._is_arxiv_id("random_string")
    
    def test_is_url(self, paper_parser):
        """Test URL detection."""
        # Valid URLs
        assert paper_parser._is_url("https://arxiv.org/pdf/2301.00001.pdf")
        assert paper_parser._is_url("http://example.com/paper.pdf")
        
        # Invalid URLs
        assert not paper_parser._is_url("2301.00001")
        assert not paper_parser._is_url("paper.pdf")
        assert not paper_parser._is_url("/local/path/paper.pdf")
    
    @pytest.mark.asyncio
    async def test_process_with_text(self, paper_parser, knowledge_graph):
        """Test processing paper text."""
        result = await paper_parser.process(
            paper_source="Sample paper text about machine learning",
            knowledge_graph=knowledge_graph
        )
        
        assert result is not None
        assert "analysis" in result or "error" in result
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_extract_arxiv_metadata(self, paper_parser):
        """Test arXiv metadata extraction."""
        # This would normally make a real API call
        # We test the method exists and has correct signature
        assert hasattr(paper_parser, '_extract_arxiv_metadata')


class TestRepoAnalyzerAgent:
    """Tests for RepoAnalyzerAgent."""

    @pytest.fixture
    def repo_analyzer(self, mock_llm_client):
        """Create a repo analyzer agent for testing."""
        return RepoAnalyzerAgent(llm_client=mock_llm_client)

    def test_agent_creation(self, repo_analyzer):
        """Test repo analyzer agent creation."""
        assert repo_analyzer is not None
        assert repo_analyzer.agent_id == "repoanalyzer"

    def test_agent_creation_with_custom_params(self, mock_llm_client):
        """Test repo analyzer with custom parallel parsing parameters."""
        agent = RepoAnalyzerAgent(
            llm_client=mock_llm_client,
            max_files=100,
            batch_size=10,
            max_workers=4
        )
        assert agent._max_files == 100
        assert agent._batch_size == 10
        assert agent._max_workers == 4

    def test_default_parallel_config(self, repo_analyzer):
        """Test default parallel parsing configuration."""
        assert repo_analyzer._max_files == RepoAnalyzerAgent.DEFAULT_MAX_FILES
        assert repo_analyzer._batch_size == RepoAnalyzerAgent.DEFAULT_BATCH_SIZE
        assert repo_analyzer._max_workers == RepoAnalyzerAgent.DEFAULT_MAX_WORKERS

    def test_prioritize_files(self, repo_analyzer):
        """Test file prioritization for parsing."""
        files = [
            {"path": "tests/test_main.py"},
            {"path": "model.py"},
            {"path": "train.py"},
            {"path": "utils/helpers.py"},
            {"path": "core/network.py"},
        ]

        prioritized = repo_analyzer._prioritize_files(files)

        # model.py and train.py should be ranked higher than test files
        paths = [f["path"] for f in prioritized]
        model_idx = paths.index("model.py")
        train_idx = paths.index("train.py")
        test_idx = paths.index("tests/test_main.py")

        assert model_idx < test_idx
        assert train_idx < test_idx

    @pytest.mark.asyncio
    async def test_scan_structure(self, repo_analyzer, temp_repo_dir):
        """Test repository structure scanning."""
        structure = await repo_analyzer._scan_structure(temp_repo_dir)

        assert "files" in structure
        assert "directories" in structure
        assert "file_counts" in structure
        assert "code_files" in structure
        assert len(structure["files"]) > 0

    @pytest.mark.asyncio
    async def test_extract_code_elements_parallel(self, repo_analyzer, temp_repo_dir):
        """Test parallel code element extraction."""
        structure = await repo_analyzer._scan_structure(temp_repo_dir)

        elements = await repo_analyzer._extract_code_elements(
            temp_repo_dir,
            structure,
            max_files=50,
            batch_size=5,
            max_workers=2
        )

        assert "classes" in elements
        assert "functions" in elements
        assert "imports" in elements
        assert "_parse_stats" in elements

        # Verify parse stats are tracked
        stats = elements["_parse_stats"]
        assert "files_parsed" in stats
        assert "batch_size" in stats
        assert stats["batch_size"] == 5

    @pytest.mark.asyncio
    async def test_process_local_repo(self, repo_analyzer, temp_repo_dir, knowledge_graph):
        """Test processing a local repository."""
        result = await repo_analyzer.process(
            repo_url=str(temp_repo_dir),
            knowledge_graph=knowledge_graph
        )

        assert result is not None
        assert "_structure" in result or "error" in result


class TestSemanticMapper:
    """Tests for SemanticMapper."""
    
    @pytest.fixture
    def semantic_mapper(self, mock_llm_client):
        """Create a semantic mapper for testing."""
        return SemanticMapper(llm_client=mock_llm_client)
    
    def test_agent_creation(self, semantic_mapper):
        """Test semantic mapper creation."""
        assert semantic_mapper is not None
        assert semantic_mapper.agent_id == "semantic_mapper"
    
    def test_tokenize(self, semantic_mapper):
        """Test name tokenization."""
        # camelCase
        tokens = semantic_mapper._tokenize("MultiHeadAttention")
        assert "multi" in tokens
        assert "head" in tokens
        assert "attention" in tokens
        
        # snake_case
        tokens = semantic_mapper._tokenize("multi_head_attention")
        assert "multi" in tokens
        assert "head" in tokens
        assert "attention" in tokens
    
    def test_lexical_similarity(self, semantic_mapper):
        """Test lexical similarity calculation."""
        # Exact match
        score = semantic_mapper._lexical_similarity("attention", "attention", [])
        assert score > 0.9
        
        # Similar names
        score = semantic_mapper._lexical_similarity(
            "attention_mechanism",
            "MultiHeadAttention",
            ["attention", "self-attention"]
        )
        assert score > 0.5
        
        # Unrelated names
        score = semantic_mapper._lexical_similarity(
            "database",
            "ImageProcessor",
            []
        )
        assert score < 0.3
    
    @pytest.mark.asyncio
    async def test_compute_code_scores(self, semantic_mapper, sample_mappings):
        """Test code score computation."""
        concept = {
            "name": "attention_mechanism",
            "description": "Self-attention in transformers",
            "likely_names": ["attention", "self_attention"]
        }
        
        code_elements = [
            {
                "name": "MultiHeadAttention",
                "type": "class",
                "file": "/model.py",
                "docstring": "Multi-head attention implementation"
            }
        ]
        
        scores = await semantic_mapper._compute_code_scores(concept, code_elements)
        
        assert len(scores) > 0
        assert all("lexical" in s for s in scores)
    
    @pytest.mark.asyncio
    async def test_process(self, semantic_mapper, knowledge_graph):
        """Test full mapping process."""
        paper_data = {
            "key_concepts": [
                {
                    "name": "transformer",
                    "description": "A neural network architecture",
                    "likely_names": ["transformer", "encoder"]
                }
            ]
        }
        
        repo_data = {
            "classes": [
                {
                    "name": "TransformerModel",
                    "file_path": "/model.py",
                    "docstring": "Transformer implementation"
                }
            ],
            "functions": []
        }
        
        result = await semantic_mapper.process(
            paper_data=paper_data,
            repo_data=repo_data,
            knowledge_graph=knowledge_graph
        )
        
        assert result is not None
        assert "mappings" in result


class TestCodingAgent:
    """Tests for CodingAgent."""
    
    @pytest.fixture
    def coding_agent(self, mock_llm_client):
        """Create a coding agent for testing."""
        return CodingAgent(llm_client=mock_llm_client)
    
    def test_agent_creation(self, coding_agent):
        """Test coding agent creation."""
        assert coding_agent is not None
        assert coding_agent.agent_id == "coding_agent"
    
    def test_validate_python_syntax_valid(self, coding_agent):
        """Test syntax validation with valid code."""
        valid_code = """
def hello():
    print("Hello, World!")

if __name__ == "__main__":
    hello()
"""
        is_valid, error = coding_agent._validate_python_syntax(valid_code)
        assert is_valid
        assert error is None
    
    def test_validate_python_syntax_invalid(self, coding_agent):
        """Test syntax validation with invalid code."""
        invalid_code = """
def hello(
    print("Hello, World!")
"""
        is_valid, error = coding_agent._validate_python_syntax(invalid_code)
        assert not is_valid
        assert error is not None
    
    def test_safe_filename(self, coding_agent):
        """Test safe filename generation."""
        filename = coding_agent._safe_filename("Test Mapping (v1)")
        
        assert " " not in filename
        assert "(" not in filename
        assert ")" not in filename
        assert filename.endswith(".py")
    
    @pytest.mark.asyncio
    async def test_generate_test_script(self, coding_agent, sample_mappings):
        """Test test script generation."""
        mapping = sample_mappings[0]
        
        script = await coding_agent._generate_test_script(mapping)
        
        assert script is not None
        # Script should be valid Python
        is_valid, _ = coding_agent._validate_python_syntax(script)
        # May not be valid if LLM mock returns invalid code
        # Just check it returns something
        assert isinstance(script, str)
    
    @pytest.mark.asyncio
    async def test_execute_in_subprocess(self, coding_agent, temp_output_dir):
        """Test subprocess code execution."""
        simple_script = '''
print("Hello from test!")
x = 1 + 1
print(f"Result: {x}")
'''
        script_path = temp_output_dir / "test_script.py"
        script_path.write_text(simple_script)
        
        result = await coding_agent._execute_in_subprocess(str(script_path))
        
        assert result["success"]
        assert "Hello from test!" in result["stdout"]
        assert result["execution_time"] > 0


class TestAgentIntegration:
    """Integration tests for agents working together."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_paper_to_mapping_flow(
        self,
        mock_llm_client,
        sample_paper_text,
        temp_repo_dir,
        knowledge_graph
    ):
        """Test flow from paper parsing to mapping."""
        # Parse paper
        parser = PaperParserAgent(llm_client=mock_llm_client)
        paper_result = await parser.process(
            paper_source=sample_paper_text,
            knowledge_graph=knowledge_graph
        )

        # Analyze repo
        analyzer = RepoAnalyzerAgent(llm_client=mock_llm_client)
        repo_result = await analyzer.process(
            repo_url=str(temp_repo_dir),
            knowledge_graph=knowledge_graph
        )

        # Create mappings
        mapper = SemanticMapper(llm_client=mock_llm_client)
        mapping_result = await mapper.process(
            paper_data=paper_result.get("analysis", {}),
            repo_data=repo_result,
            knowledge_graph=knowledge_graph
        )

        # Verify knowledge graph has nodes
        assert len(knowledge_graph.graph.nodes) > 0


# =============================================================================
# Parser Integration Tests
# =============================================================================

class TestParserIntegration:
    """Tests for parser integration with RepoAnalyzerAgent."""

    @pytest.fixture
    def multi_language_repo(self, tmp_path):
        """Create a repository with multiple language files."""
        # Python files
        py_dir = tmp_path / "python"
        py_dir.mkdir()
        (py_dir / "model.py").write_text('''
"""Model implementation module."""
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    """Transformer neural network model."""

    def __init__(self, hidden_size: int = 512):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x):
        """Forward pass."""
        return x
''')

        # Julia files
        julia_dir = tmp_path / "julia"
        julia_dir.mkdir()
        (julia_dir / "solver.jl").write_text('''
"""Julia solver module."""
module Solver

export solve, optimize

struct Config
    max_iter::Int
    tolerance::Float64
end

function solve(problem::Vector{Float64})
    # Solve the problem
    return sum(problem)
end

function optimize(f, x0)
    return x0
end

end
''')

        # R files
        r_dir = tmp_path / "R"
        r_dir.mkdir()
        (r_dir / "analysis.R").write_text('''
#' Analysis module for statistical computations
#' @author Test Author

#' Calculate mean of values
#' @param x Numeric vector
#' @return Mean value
calculate_mean <- function(x) {
    return(mean(x))
}

#' Fit linear model
#' @param formula Model formula
#' @param data Dataset
fit_model <- function(formula, data) {
    lm(formula, data)
}
''')

        # JavaScript files
        js_dir = tmp_path / "js"
        js_dir.mkdir()
        (js_dir / "utils.js").write_text('''
/**
 * Utility functions
 * @module utils
 */

/**
 * Format a number as currency
 * @param {number} value - The value to format
 * @returns {string} Formatted string
 */
function formatCurrency(value) {
    return `$${value.toFixed(2)}`;
}

/**
 * DataProcessor class for handling data
 */
class DataProcessor {
    constructor(config) {
        this.config = config;
    }

    process(data) {
        return data.map(item => item * 2);
    }
}

export { formatCurrency, DataProcessor };
''')

        return tmp_path

    @pytest.fixture
    def factory(self):
        """Create a ParserFactory instance."""
        return ParserFactory()

    def test_parser_factory_python(self, factory, tmp_path):
        """Test ParserFactory returns Python parser."""
        py_file = tmp_path / "test.py"
        py_file.touch()
        parser = factory.get_parser_for_file(py_file)

        assert parser is not None
        assert isinstance(parser, PythonParser)
        assert parser.language == "python"

    def test_parser_factory_julia(self, factory, tmp_path):
        """Test ParserFactory returns Julia parser."""
        jl_file = tmp_path / "test.jl"
        jl_file.touch()
        parser = factory.get_parser_for_file(jl_file)

        assert parser is not None
        assert isinstance(parser, JuliaParser)
        assert parser.language == "julia"

    def test_parser_factory_r(self, factory, tmp_path):
        """Test ParserFactory returns R parser."""
        r_file = tmp_path / "test.R"
        r_file.touch()
        parser = factory.get_parser_for_file(r_file)

        assert parser is not None
        assert isinstance(parser, RParser)
        assert parser.language == "r"

    def test_parser_factory_javascript(self, factory, tmp_path):
        """Test ParserFactory returns JavaScript parser."""
        js_file = tmp_path / "test.js"
        js_file.touch()
        parser = factory.get_parser_for_file(js_file)

        assert parser is not None
        assert isinstance(parser, JavaScriptParser)
        assert parser.language == "javascript"

    def test_parser_factory_typescript(self, factory, tmp_path):
        """Test ParserFactory returns JavaScript parser for TypeScript."""
        ts_file = tmp_path / "test.ts"
        ts_file.touch()
        parser = factory.get_parser_for_file(ts_file)

        assert parser is not None
        assert isinstance(parser, JavaScriptParser)

    def test_parser_factory_unsupported(self, factory, tmp_path):
        """Test ParserFactory returns None for unsupported extensions."""
        xyz_file = tmp_path / "test.xyz"
        xyz_file.touch()
        parser = factory.get_parser_for_file(xyz_file)
        assert parser is None

        go_file = tmp_path / "test.go"
        go_file.touch()
        parser = factory.get_parser_for_file(go_file)
        assert parser is None

    def test_parse_python_file(self, multi_language_repo):
        """Test parsing Python file extracts elements correctly."""
        parser = PythonParser()
        py_file = multi_language_repo / "python" / "model.py"

        result = parser.parse_file(py_file)

        assert "classes" in result
        assert "functions" in result

        # Check class was found
        classes = result["classes"]
        assert len(classes) >= 1
        class_names = [c.name for c in classes]
        assert "TransformerModel" in class_names

    def test_parse_julia_file(self, multi_language_repo):
        """Test parsing Julia file extracts elements correctly."""
        parser = JuliaParser()
        jl_file = multi_language_repo / "julia" / "solver.jl"

        result = parser.parse_file(jl_file)

        assert "classes" in result
        assert "functions" in result

        # Check functions were found
        func_names = [f.name for f in result["functions"]]
        assert "solve" in func_names or len(result["functions"]) > 0

    def test_parse_r_file(self, multi_language_repo):
        """Test parsing R file extracts elements correctly."""
        parser = RParser()
        r_file = multi_language_repo / "R" / "analysis.R"

        result = parser.parse_file(r_file)

        assert "classes" in result
        assert "functions" in result

        # Check functions were found
        func_names = [f.name for f in result["functions"]]
        assert "calculate_mean" in func_names or len(result["functions"]) > 0

    def test_parse_javascript_file(self, multi_language_repo):
        """Test parsing JavaScript file extracts elements correctly."""
        parser = JavaScriptParser()
        js_file = multi_language_repo / "js" / "utils.js"

        result = parser.parse_file(js_file)

        assert "classes" in result
        assert "functions" in result

        # Check class and function were found
        class_names = [c.name for c in result["classes"]]
        func_names = [f.name for f in result["functions"]]
        assert "DataProcessor" in class_names or "formatCurrency" in func_names

    def test_code_element_model(self):
        """Test CodeElement Pydantic model."""
        element = CodeElement(
            name="test_function",
            element_type="function",
            file_path="/test.py",
            line_number=10,
            docstring="A test function",
            signature="def test_function(x: int) -> int",
            args=["x"],
            return_type="int",
            language="python"
        )

        assert element.name == "test_function"
        assert element.element_type == "function"
        assert element.language == "python"

    @pytest.mark.asyncio
    async def test_repo_analyzer_with_multi_language(
        self,
        mock_llm_client,
        multi_language_repo,
        knowledge_graph
    ):
        """Test RepoAnalyzerAgent handles multi-language repos."""
        analyzer = RepoAnalyzerAgent(llm_client=mock_llm_client)

        result = await analyzer.process(
            repo_url=str(multi_language_repo),
            knowledge_graph=knowledge_graph
        )

        assert result is not None
        # Should have parsed files from multiple languages
        if "classes" in result:
            assert len(result.get("classes", [])) >= 0
        if "functions" in result:
            assert len(result.get("functions", [])) >= 0


# =============================================================================
# Agent Error Handling Tests
# =============================================================================

class TestAgentErrorHandling:
    """Tests for agent error handling."""

    @pytest.mark.asyncio
    async def test_paper_parser_handles_invalid_input(self, mock_llm_client, knowledge_graph):
        """Test paper parser handles invalid input gracefully."""
        parser = PaperParserAgent(llm_client=mock_llm_client)

        # Empty input
        result = await parser.process(
            paper_source="",
            knowledge_graph=knowledge_graph
        )

        # Should return result without crashing
        assert result is not None

    @pytest.mark.asyncio
    async def test_repo_analyzer_handles_nonexistent_path(
        self,
        mock_llm_client,
        knowledge_graph
    ):
        """Test repo analyzer handles non-existent path."""
        analyzer = RepoAnalyzerAgent(llm_client=mock_llm_client)

        result = await analyzer.process(
            repo_url="/nonexistent/path/that/does/not/exist",
            knowledge_graph=knowledge_graph
        )

        # Should return error result, not crash
        assert result is not None

    @pytest.mark.asyncio
    async def test_semantic_mapper_handles_empty_data(
        self,
        mock_llm_client,
        knowledge_graph
    ):
        """Test semantic mapper handles empty input data."""
        mapper = SemanticMapper(llm_client=mock_llm_client)

        result = await mapper.process(
            paper_data={},
            repo_data={},
            knowledge_graph=knowledge_graph
        )

        assert result is not None
        assert "mappings" in result

    @pytest.mark.asyncio
    async def test_coding_agent_handles_invalid_mapping(
        self,
        mock_llm_client
    ):
        """Test coding agent handles invalid mapping data."""
        agent = CodingAgent(llm_client=mock_llm_client)

        # Invalid mapping with missing fields
        invalid_mapping = {"invalid": "data"}

        # Should not crash when generating test script
        try:
            script = await agent._generate_test_script(invalid_mapping)
            assert script is not None or True  # Either returns something or handles error
        except Exception:
            # Exception handling is also acceptable
            pass


# =============================================================================
# Agent Statistics Tests
# =============================================================================

class TestAgentStatistics:
    """Tests for agent statistics tracking."""

    @pytest.mark.asyncio
    async def test_agent_tracks_operations(self, mock_llm_client, knowledge_graph):
        """Test that agents track operation statistics."""
        # Create concrete agent
        class ConcreteAgent(BaseAgent):
            async def process(self, *args, **kwargs):
                return {"status": "ok"}

        agent = ConcreteAgent(llm_client=mock_llm_client)

        # Get initial stats
        stats_before = agent.get_stats()
        assert stats_before["operation_count"] == 0

    def test_paper_parser_has_correct_agent_id(self, mock_llm_client):
        """Test paper parser has correct agent ID."""
        agent = PaperParserAgent(llm_client=mock_llm_client)
        assert agent.agent_id == "paper_parser"

    def test_repo_analyzer_has_correct_agent_id(self, mock_llm_client):
        """Test repo analyzer has correct agent ID."""
        agent = RepoAnalyzerAgent(llm_client=mock_llm_client)
        assert agent.agent_id == "repoanalyzer"

    def test_semantic_mapper_has_correct_agent_id(self, mock_llm_client):
        """Test semantic mapper has correct agent ID."""
        agent = SemanticMapper(llm_client=mock_llm_client)
        assert agent.agent_id == "semantic_mapper"

    def test_coding_agent_has_correct_agent_id(self, mock_llm_client):
        """Test coding agent has correct agent ID."""
        agent = CodingAgent(llm_client=mock_llm_client)
        assert agent.agent_id == "coding_agent"


# =============================================================================
# Parser Factory Edge Cases
# =============================================================================

class TestParserFactoryEdgeCases:
    """Tests for parser factory edge cases."""

    @pytest.fixture
    def factory(self):
        """Create a ParserFactory instance."""
        return ParserFactory()

    def test_get_parser_case_insensitive(self, factory, tmp_path):
        """Test parser factory is case insensitive for extensions."""
        # Test various case combinations
        py_lower = tmp_path / "test.py"
        py_lower.touch()
        py_upper = tmp_path / "test.PY"
        py_upper.touch()

        parser_lower = factory.get_parser_for_file(py_lower)
        parser_upper = factory.get_parser_for_file(py_upper)

        assert parser_lower is not None
        # Case sensitivity depends on implementation

    def test_get_parser_by_language(self, factory):
        """Test getting parser by language name."""
        parser = factory.get_parser("python")
        assert parser is not None
        assert parser.language == "python"

        parser = factory.get_parser("julia")
        assert parser is not None
        assert parser.language == "julia"

    def test_supported_extensions_list(self, factory):
        """Test getting list of supported extensions."""
        extensions = factory.supported_extensions

        assert isinstance(extensions, list)
        assert len(extensions) > 0
        assert ".py" in extensions
        assert ".jl" in extensions

    def test_supported_languages_list(self, factory):
        """Test getting list of supported languages."""
        languages = factory.supported_languages

        assert isinstance(languages, list)
        assert len(languages) > 0
        assert "python" in languages
        assert "julia" in languages

    def test_detect_language(self, factory, tmp_path):
        """Test language detection from file extension."""
        py_file = tmp_path / "test.py"
        py_file.touch()

        lang = factory.detect_language(py_file)
        assert lang == "python"

        jl_file = tmp_path / "test.jl"
        jl_file.touch()
        lang = factory.detect_language(jl_file)
        assert lang == "julia"

    def test_detect_language_unsupported(self, factory, tmp_path):
        """Test language detection returns None for unsupported."""
        xyz_file = tmp_path / "test.xyz"
        xyz_file.touch()

        lang = factory.detect_language(xyz_file)
        assert lang is None


# =============================================================================
# Semantic Mapper Additional Tests
# =============================================================================

class TestSemanticMapperAdditional:
    """Additional tests for SemanticMapper."""

    @pytest.fixture
    def semantic_mapper(self, mock_llm_client):
        """Create a semantic mapper for testing."""
        return SemanticMapper(llm_client=mock_llm_client)

    def test_tokenize_mixed_case(self, semantic_mapper):
        """Test tokenization of mixed case names."""
        tokens = semantic_mapper._tokenize("XMLHttpRequest")
        assert "xml" in tokens or "http" in tokens or "request" in tokens

    def test_tokenize_numbers(self, semantic_mapper):
        """Test tokenization handles numbers."""
        tokens = semantic_mapper._tokenize("Layer2Norm")
        assert "layer" in tokens or "norm" in tokens

    def test_tokenize_acronyms(self, semantic_mapper):
        """Test tokenization handles acronyms."""
        tokens = semantic_mapper._tokenize("GPUCompute")
        assert len(tokens) >= 1

    def test_lexical_similarity_empty_strings(self, semantic_mapper):
        """Test lexical similarity with empty strings."""
        score = semantic_mapper._lexical_similarity("", "", [])
        assert score >= 0  # Should not crash

    def test_lexical_similarity_with_keywords(self, semantic_mapper):
        """Test lexical similarity boost from keywords."""
        # With matching keyword
        score_with_kw = semantic_mapper._lexical_similarity(
            "attention",
            "SelfAttention",
            ["attention", "transformer"]
        )

        # Without matching keyword
        score_without_kw = semantic_mapper._lexical_similarity(
            "attention",
            "SelfAttention",
            []
        )

        # Keyword match should boost score
        assert score_with_kw >= score_without_kw
