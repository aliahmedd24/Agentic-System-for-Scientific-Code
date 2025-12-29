"""
Pytest configuration and shared fixtures.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Event Loop Fixture
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Mock LLM Client Fixture
# ============================================================================

@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    client = MagicMock()
    client.generate = AsyncMock(return_value="Mock LLM response")
    client.generate_structured = AsyncMock(return_value={
        "title": "Test Paper",
        "abstract": "Test abstract",
        "key_concepts": [{"name": "test_concept", "description": "A test concept"}],
        "algorithms": [],
        "methodology": "Test methodology",
        "expected_implementations": ["test_function"]
    })
    client.generate_stream = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


# ============================================================================
# Knowledge Graph Fixtures
# ============================================================================

@pytest.fixture
def knowledge_graph():
    """Create a fresh knowledge graph for testing."""
    from core.knowledge_graph import KnowledgeGraph
    return KnowledgeGraph()


@pytest.fixture
def populated_knowledge_graph(knowledge_graph):
    """Create a knowledge graph with sample data."""
    from core.knowledge_graph import (
        create_paper_node,
        create_concept_node,
        create_function_node,
        create_mapping_node,
        EdgeType
    )

    # Add paper node (helper function takes kg as first arg and returns node_id)
    paper_node_id = create_paper_node(
        knowledge_graph,
        title="Test Paper on Machine Learning",
        authors=["Alice Smith", "Bob Jones"],
        abstract="This paper presents a novel approach to machine learning."
    )

    # Add concept nodes
    concept1_id = create_concept_node(
        knowledge_graph,
        name="neural_network",
        description="A computational model inspired by biological neural networks"
    )
    concept2_id = create_concept_node(
        knowledge_graph,
        name="backpropagation",
        description="Algorithm for training neural networks"
    )

    # Add function node
    func_node_id = create_function_node(
        knowledge_graph,
        name="train_model",
        file_path="/repo/train.py",
        signature="def train_model(data, epochs=10)",
        docstring="Train the neural network model"
    )

    # Add relationships
    knowledge_graph.add_edge(paper_node_id, concept1_id, EdgeType.CONTAINS)
    knowledge_graph.add_edge(paper_node_id, concept2_id, EdgeType.CONTAINS)
    knowledge_graph.add_edge(concept2_id, concept1_id, EdgeType.DEPENDS_ON)

    # Add mapping (create_mapping_node automatically adds the edges)
    create_mapping_node(
        knowledge_graph,
        concept_id=concept1_id,
        code_id=func_node_id,
        confidence=0.85,
        evidence=["Function implements neural network training"]
    )

    return knowledge_graph


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_paper_text():
    """Sample paper text for testing."""
    return """
    Title: A Novel Approach to Deep Learning
    
    Abstract:
    This paper presents a novel deep learning architecture called TransformerNet
    that achieves state-of-the-art results on various NLP tasks.
    
    1. Introduction
    Deep learning has revolutionized the field of artificial intelligence.
    Our approach introduces a new attention mechanism that improves performance.
    
    2. Methodology
    We propose the TransformerNet architecture which consists of:
    - Multi-head self-attention layers
    - Feed-forward neural networks
    - Layer normalization
    
    3. Experiments
    We evaluated our model on the following benchmarks:
    - GLUE benchmark
    - SQuAD dataset
    - MNLI dataset
    
    4. Conclusion
    TransformerNet demonstrates significant improvements over baseline models.
    """


@pytest.fixture
def sample_repo_structure():
    """Sample repository structure for testing."""
    return {
        "name": "test-repo",
        "files": [
            {"path": "main.py", "size": 1500},
            {"path": "model.py", "size": 3000},
            {"path": "train.py", "size": 2500},
            {"path": "utils.py", "size": 800},
            {"path": "requirements.txt", "size": 200},
            {"path": "README.md", "size": 1000},
        ],
        "directories": ["src", "tests", "data"],
        "total_files": 6,
        "total_size": 9000,
        "extensions": {".py": 4, ".txt": 1, ".md": 1}
    }


@pytest.fixture
def sample_mappings():
    """Sample concept-to-code mappings for testing."""
    return [
        {
            "concept": "attention_mechanism",
            "code_element": "MultiHeadAttention",
            "element_type": "class",
            "file_path": "/repo/model.py",
            "confidence": 0.92,
            "evidence": "Class implements multi-head attention as described in paper"
        },
        {
            "concept": "transformer",
            "code_element": "TransformerBlock",
            "element_type": "class",
            "file_path": "/repo/model.py",
            "confidence": 0.88,
            "evidence": "Core transformer architecture implementation"
        },
        {
            "concept": "training_loop",
            "code_element": "train_epoch",
            "element_type": "function",
            "file_path": "/repo/train.py",
            "confidence": 0.75,
            "evidence": "Training loop implementation"
        }
    ]


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def temp_repo_dir(tmp_path):
    """Create a temporary repository directory with sample files."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    # Create sample Python files
    (repo_dir / "main.py").write_text('''
"""Main entry point."""

from model import TransformerNet

def main():
    model = TransformerNet()
    model.train()

if __name__ == "__main__":
    main()
''')
    
    (repo_dir / "model.py").write_text('''
"""Model definitions."""

import torch
import torch.nn as nn

class TransformerNet(nn.Module):
    """Transformer-based neural network."""
    
    def __init__(self, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = MultiHeadAttention(hidden_size)
    
    def forward(self, x):
        return self.attention(x)
    
    def train(self):
        """Train the model."""
        pass

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
    
    def forward(self, x):
        return x
''')
    
    (repo_dir / "requirements.txt").write_text('''
torch>=2.0.0
numpy>=1.24.0
transformers>=4.30.0
''')
    
    return repo_dir


# ============================================================================
# API Testing Fixtures
# ============================================================================

@pytest.fixture
def test_client():
    """Create a test client for the FastAPI application."""
    from fastapi.testclient import TestClient
    from api.server import app
    return TestClient(app)


# ============================================================================
# Environment Setup
# ============================================================================

@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables after each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


# ============================================================================
# Sandbox Fixtures
# ============================================================================

@pytest.fixture
def sandbox_config():
    """Create a basic sandbox configuration for testing."""
    from core.bubblewrap_sandbox import SandboxConfig, IsolationLevel
    return SandboxConfig(
        isolation_level=IsolationLevel.SUBPROCESS,
        timeout_seconds=30,
        memory_limit_mb=512,
        network_enabled=False,
        language="python"
    )


@pytest.fixture
def mock_sandbox_manager():
    """Create a mock sandbox manager."""
    from core.bubblewrap_sandbox import ExecutionResult, IsolationLevel

    manager = MagicMock()
    manager.available_backends = [IsolationLevel.SUBPROCESS]
    manager.best_backend = IsolationLevel.SUBPROCESS
    manager.execute = AsyncMock(return_value=ExecutionResult(
        success=True,
        stdout="Test output",
        stderr="",
        exit_code=0,
        execution_time=0.1,
        isolation_level=IsolationLevel.SUBPROCESS
    ))
    manager.execute_code = AsyncMock(return_value=ExecutionResult(
        success=True,
        stdout="Code output",
        stderr="",
        exit_code=0,
        execution_time=0.1,
        isolation_level=IsolationLevel.SUBPROCESS
    ))
    return manager


# ============================================================================
# Checkpoint Fixtures
# ============================================================================

@pytest.fixture
def checkpoint_manager(tmp_path):
    """Create a checkpoint manager with temp directory."""
    from core.checkpointing import CheckpointManager
    return CheckpointManager(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        max_checkpoints=5,
        compress=True
    )


@pytest.fixture
def sample_checkpoint_data():
    """Sample data for checkpoint tests."""
    return {
        "paper_data": {
            "title": "Test Paper on Neural Networks",
            "abstract": "A" * 150,  # Meets minimum length requirement
            "authors": ["Author One", "Author Two"],
            "key_concepts": [{"name": "attention", "description": "Attention mechanism"}]
        },
        "repo_data": {
            "name": "test-repo",
            "url": "https://github.com/test/repo",
            "overview": {"purpose": "Test repository"},
            "dependencies": {"python": ["torch", "numpy"]}
        },
        "mappings": [
            {
                "concept_name": "attention",
                "code_element": "Attention",
                "code_file": "model.py",
                "confidence": 0.85,
                "evidence": ["Function implements attention mechanism"]
            }
        ]
    }


# ============================================================================
# Metrics Fixtures
# ============================================================================

@pytest.fixture
def metrics_collector():
    """Create a fresh metrics collector."""
    from core.metrics import MetricsCollector
    return MetricsCollector(max_history=100)


@pytest.fixture
def populated_metrics_collector(metrics_collector):
    """Create a metrics collector with sample data."""
    from core.metrics import MetricType

    # Add sample metrics
    metrics_collector.record(MetricType.ACCURACY, "mapping_confidence", 0.85)
    metrics_collector.record(MetricType.ACCURACY, "mapping_confidence", 0.92)
    metrics_collector.record(MetricType.TIMING, "test_execution", 1500.0)
    metrics_collector.record(MetricType.SUCCESS_RATE, "test_execution", 1.0)
    metrics_collector.record(MetricType.SUCCESS_RATE, "test_execution", 0.0)

    return metrics_collector


# ============================================================================
# Parser Fixtures
# ============================================================================

@pytest.fixture
def parser_factory():
    """Create a parser factory instance."""
    from agents.parsers import ParserFactory
    return ParserFactory()


@pytest.fixture
def sample_python_file(tmp_path):
    """Create a sample Python file for parsing."""
    content = '''"""Sample module docstring."""

import os
from typing import List

CONSTANT_VALUE = 42

class SampleClass:
    """A sample class."""

    def __init__(self, value: int):
        self.value = value

    def get_value(self) -> int:
        """Get the value."""
        return self.value

def sample_function(arg1: str, arg2: int = 10) -> List[str]:
    """A sample function."""
    return [arg1] * arg2

async def async_function():
    """An async function."""
    pass
'''
    file_path = tmp_path / "sample.py"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_julia_file(tmp_path):
    """Create a sample Julia file for parsing."""
    content = '''module SampleModule

struct Point
    x::Float64
    y::Float64
end

function calculate_distance(p1::Point, p2::Point)::Float64
    return sqrt((p2.x - p1.x)^2 + (p2.y - p1.y)^2)
end

end
'''
    file_path = tmp_path / "sample.jl"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_r_file(tmp_path):
    """Create a sample R file for parsing."""
    content = '''# Sample R file

calculate_mean <- function(x) {
    mean(x)
}

process_data <- function(data, threshold = 0.5) {
    data[data > threshold]
}
'''
    file_path = tmp_path / "sample.R"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_js_file(tmp_path):
    """Create a sample JavaScript file for parsing."""
    content = '''// Sample JavaScript file

class Calculator {
    constructor(value) {
        this.value = value;
    }

    add(x) {
        return this.value + x;
    }
}

function multiply(a, b) {
    return a * b;
}

const divide = (a, b) => a / b;

export { Calculator, multiply, divide };
'''
    file_path = tmp_path / "sample.js"
    file_path.write_text(content)
    return file_path


# ============================================================================
# Orchestrator Fixtures
# ============================================================================

@pytest.fixture
def mock_orchestrator(mock_llm_client):
    """Create a mock pipeline orchestrator."""
    from core.orchestrator import PipelineOrchestrator, PipelineResult, PipelineStage

    orchestrator = MagicMock(spec=PipelineOrchestrator)
    orchestrator.llm_provider = "gemini"
    orchestrator._current_stage = PipelineStage.INITIALIZED
    orchestrator._current_progress = 0
    orchestrator._event_callbacks = []

    async def mock_run(*args, **kwargs):
        return PipelineResult(
            status="completed",
            paper_data=None,
            repo_data=None,
            mappings=[],
            code_results=[],
            knowledge_graph=None
        )

    orchestrator.run = AsyncMock(side_effect=mock_run)
    return orchestrator


@pytest.fixture
def pipeline_event_collector():
    """Collect pipeline events for testing."""
    events = []

    async def callback(event):
        events.append(event)

    return {"callback": callback, "events": events}


# ============================================================================
# Resource Estimator Fixtures
# ============================================================================

@pytest.fixture
def resource_estimator():
    """Create a resource estimator instance."""
    from core.resource_estimator import ResourceEstimator
    return ResourceEstimator()


@pytest.fixture
def sample_repo_with_torch():
    """Sample repo data with PyTorch dependencies."""
    return {
        "name": "torch-project",
        "dependencies": {
            "python": ["torch", "numpy", "transformers"]
        },
        "stats": {
            "classes": 10,
            "functions": 20
        }
    }


@pytest.fixture
def sample_repo_minimal():
    """Sample repo data with minimal dependencies."""
    return {
        "name": "simple-project",
        "dependencies": {
            "python": ["requests", "json"]
        },
        "stats": {
            "classes": 2,
            "functions": 5
        }
    }


# ============================================================================
# LLM Client Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_config():
    """Create a mock LLM configuration."""
    return {
        "provider": "gemini",
        "model": "gemini-pro",
        "max_tokens": 4096,
        "temperature": 0.7
    }


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx async client."""
    import httpx

    client = MagicMock(spec=httpx.AsyncClient)
    client.post = AsyncMock()
    client.get = AsyncMock()
    client.aclose = AsyncMock()
    return client


# ============================================================================
# Protocol Fixtures
# ============================================================================

@pytest.fixture
def valid_paper_parser_output():
    """Create a valid PaperParserOutput for testing."""
    from agents.protocols import (
        PaperParserOutput, Concept, Algorithm,
        Methodology, Reproducibility, SourceMetadata
    )
    return PaperParserOutput(
        title="Test Paper on Deep Learning",
        authors=["Author One", "Author Two"],
        abstract="A" * 150,  # Minimum 100 chars
        key_concepts=[
            Concept(name="attention", description="Attention mechanism")
        ],
        algorithms=[
            Algorithm(name="transformer", description="Transformer architecture")
        ],
        methodology=Methodology(approach="Supervised learning"),
        reproducibility=Reproducibility(code_available=True),
        source_metadata=SourceMetadata(source_type="arxiv")
    )


@pytest.fixture
def valid_repo_analyzer_output():
    """Create a valid RepoAnalyzerOutput for testing."""
    from agents.protocols import (
        RepoAnalyzerOutput, OverviewInfo, KeyComponent,
        DependencyInfo, FileStats
    )
    return RepoAnalyzerOutput(
        name="test-repo",
        url="https://github.com/test/repo",
        overview=OverviewInfo(purpose="Test repository"),
        key_components=[
            KeyComponent(name="model", path="model.py", description="Model definitions")
        ],
        dependencies=DependencyInfo(python=["torch", "numpy"]),
        stats=FileStats(total_files=10, code_files=5, classes=3, functions=15)
    )


@pytest.fixture
def valid_mapping_result():
    """Create a valid MappingResult for testing."""
    from agents.protocols import MappingResult, MatchSignals
    return MappingResult(
        concept_name="attention",
        concept_description="Attention mechanism from paper",
        code_element="MultiHeadAttention",
        code_file="model.py",
        confidence=0.85,
        match_signals=MatchSignals(lexical=0.7, semantic=0.9, documentary=0.8),
        evidence=["Implements attention as described"],
        reasoning="High semantic similarity with paper concept"
    )


# ============================================================================
# QEMU Fixtures
# ============================================================================

@pytest.fixture
def qemu_vm_config():
    """Create a QEMU VM configuration for testing."""
    from core.qemu_backend import QEMUVMConfig, ExecutionMode
    return QEMUVMConfig(
        name="test-vm",
        memory="1G",
        cpus=1,
        timeout_seconds=60,
        execution_mode=ExecutionMode.VIRTFS,
        network_enabled=False
    )


@pytest.fixture
def mock_qemu_image_manager(tmp_path):
    """Create a mock QEMU image manager."""
    manager = MagicMock()
    manager.images_dir = tmp_path / "qemu-images"
    manager.images_dir.mkdir(exist_ok=True)
    manager.create_snapshot = MagicMock(return_value=tmp_path / "snapshot.qcow2")
    manager.delete_snapshot = MagicMock(return_value=True)
    manager.get_image_info = MagicMock(return_value={"format": "qcow2", "size": 1024})
    return manager


# ============================================================================
# Schema Utils Fixtures
# ============================================================================

@pytest.fixture
def sample_pydantic_model():
    """Create a sample Pydantic model for schema testing."""
    from pydantic import BaseModel, Field

    class InnerModel(BaseModel):
        value: int = Field(..., description="Inner value")

    class OuterModel(BaseModel):
        name: str = Field(..., description="Name field")
        inner: InnerModel = Field(..., description="Nested model")
        items: list = Field(default_factory=list, description="List of items")

    return OuterModel


# ============================================================================
# Additional Pytest Markers
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_llm: marks tests that require an LLM API key"
    )
    config.addinivalue_line(
        "markers", "sandbox: marks tests requiring sandbox execution"
    )
    config.addinivalue_line(
        "markers", "qemu: marks tests requiring QEMU (very slow)"
    )
