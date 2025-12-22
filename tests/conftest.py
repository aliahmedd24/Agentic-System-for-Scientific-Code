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
# Pytest Configuration
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
        NodeType,
        EdgeType
    )
    
    # Add paper node
    paper_node = create_paper_node(
        paper_id="test_paper_001",
        title="Test Paper on Machine Learning",
        authors=["Alice Smith", "Bob Jones"],
        abstract="This paper presents a novel approach to machine learning."
    )
    knowledge_graph.add_node(paper_node)
    
    # Add concept nodes
    concept1 = create_concept_node(
        name="neural_network",
        description="A computational model inspired by biological neural networks"
    )
    concept2 = create_concept_node(
        name="backpropagation",
        description="Algorithm for training neural networks"
    )
    knowledge_graph.add_node(concept1)
    knowledge_graph.add_node(concept2)
    
    # Add function node
    func_node = create_function_node(
        name="train_model",
        file_path="/repo/train.py",
        signature="def train_model(data, epochs=10)",
        docstring="Train the neural network model"
    )
    knowledge_graph.add_node(func_node)
    
    # Add relationships
    knowledge_graph.add_edge(paper_node.id, concept1.id, EdgeType.CONTAINS)
    knowledge_graph.add_edge(paper_node.id, concept2.id, EdgeType.CONTAINS)
    knowledge_graph.add_edge(concept2.id, concept1.id, EdgeType.DEPENDS_ON)
    
    # Add mapping
    mapping_node = create_mapping_node(
        concept_name="neural_network",
        code_element="train_model",
        confidence=0.85,
        evidence="Function implements neural network training"
    )
    knowledge_graph.add_node(mapping_node)
    knowledge_graph.add_edge(concept1.id, mapping_node.id, EdgeType.MAPS_TO)
    knowledge_graph.add_edge(mapping_node.id, func_node.id, EdgeType.MAPS_TO)
    
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
