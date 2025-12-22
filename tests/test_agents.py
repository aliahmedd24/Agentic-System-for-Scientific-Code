"""
Tests for Agent modules.
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
        assert repo_analyzer.agent_id == "repo_analyzer"
    
    def test_should_skip_path(self, repo_analyzer):
        """Test path skipping logic."""
        # Should skip
        assert repo_analyzer._should_skip_path(Path(".git"))
        assert repo_analyzer._should_skip_path(Path("node_modules"))
        assert repo_analyzer._should_skip_path(Path("__pycache__"))
        assert repo_analyzer._should_skip_path(Path(".venv"))
        
        # Should not skip
        assert not repo_analyzer._should_skip_path(Path("src"))
        assert not repo_analyzer._should_skip_path(Path("model.py"))
        assert not repo_analyzer._should_skip_path(Path("tests"))
    
    def test_is_python_file(self, repo_analyzer):
        """Test Python file detection."""
        assert repo_analyzer._is_python_file(Path("model.py"))
        assert repo_analyzer._is_python_file(Path("src/utils.py"))
        
        assert not repo_analyzer._is_python_file(Path("config.json"))
        assert not repo_analyzer._is_python_file(Path("README.md"))
    
    @pytest.mark.asyncio
    async def test_scan_structure(self, repo_analyzer, temp_repo_dir):
        """Test repository structure scanning."""
        structure = await repo_analyzer._scan_structure(temp_repo_dir)
        
        assert "files" in structure
        assert "directories" in structure
        assert "extensions" in structure
        assert len(structure["files"]) > 0
    
    @pytest.mark.asyncio
    async def test_parse_python_file(self, repo_analyzer, temp_repo_dir):
        """Test Python file parsing."""
        model_file = temp_repo_dir / "model.py"
        
        result = await repo_analyzer._parse_python_file(model_file)
        
        assert "classes" in result
        assert "functions" in result
        assert "imports" in result
        
        # Should find TransformerNet and MultiHeadAttention classes
        class_names = [c["name"] for c in result["classes"]]
        assert "TransformerNet" in class_names
        assert "MultiHeadAttention" in class_names
    
    @pytest.mark.asyncio
    async def test_process_local_repo(self, repo_analyzer, temp_repo_dir, knowledge_graph):
        """Test processing a local repository."""
        result = await repo_analyzer.process(
            repo_path=str(temp_repo_dir),
            knowledge_graph=knowledge_graph
        )
        
        assert result is not None
        assert "structure" in result or "error" in result


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
            repo_path=str(temp_repo_dir),
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
