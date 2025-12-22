"""
Tests for the Knowledge Graph module.
"""

import pytest
import json
from pathlib import Path

from core.knowledge_graph import (
    KnowledgeGraph,
    KGNode,
    NodeType,
    EdgeType,
    create_paper_node,
    create_concept_node,
    create_function_node,
    create_class_node,
    create_mapping_node,
)


class TestKGNode:
    """Tests for KGNode dataclass."""
    
    def test_create_node(self):
        """Test basic node creation."""
        node = KGNode(
            id="test_node_1",
            type=NodeType.CONCEPT,
            label="test_concept",
            properties={"description": "A test concept"}
        )
        
        assert node.id == "test_node_1"
        assert node.type == NodeType.CONCEPT
        assert node.label == "test_concept"
        assert node.properties["description"] == "A test concept"
        assert node.embedding is None
    
    def test_node_with_embedding(self):
        """Test node with embedding vector."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        node = KGNode(
            id="embed_node",
            type=NodeType.CONCEPT,
            label="embedded",
            properties={},
            embedding=embedding
        )
        
        assert node.embedding == embedding
        assert len(node.embedding) == 5


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph class."""
    
    def test_empty_graph(self, knowledge_graph):
        """Test newly created graph is empty."""
        assert len(knowledge_graph.graph.nodes) == 0
        assert len(knowledge_graph.graph.edges) == 0
    
    def test_add_node(self, knowledge_graph):
        """Test adding a node to the graph."""
        node = create_concept_node("test_concept", "A test concept")
        knowledge_graph.add_node(node)
        
        assert knowledge_graph.get_node(node.id) is not None
        assert len(knowledge_graph.graph.nodes) == 1
    
    def test_add_duplicate_node(self, knowledge_graph):
        """Test adding duplicate node updates existing."""
        node1 = create_concept_node("test", "First description")
        node2 = create_concept_node("test", "Updated description")
        
        knowledge_graph.add_node(node1)
        knowledge_graph.add_node(node2)
        
        # Should still have one node (updated)
        assert len(knowledge_graph.graph.nodes) == 1
    
    def test_add_edge(self, knowledge_graph):
        """Test adding an edge between nodes."""
        node1 = create_concept_node("concept1", "First concept")
        node2 = create_concept_node("concept2", "Second concept")
        
        knowledge_graph.add_node(node1)
        knowledge_graph.add_node(node2)
        knowledge_graph.add_edge(node1.id, node2.id, EdgeType.DEPENDS_ON)
        
        assert len(knowledge_graph.graph.edges) == 1
        assert knowledge_graph.graph.has_edge(node1.id, node2.id)
    
    def test_add_edge_missing_node(self, knowledge_graph):
        """Test adding edge with missing node raises error or handles gracefully."""
        node1 = create_concept_node("concept1", "First concept")
        knowledge_graph.add_node(node1)
        
        # Should handle missing target node
        knowledge_graph.add_edge(node1.id, "nonexistent", EdgeType.DEPENDS_ON)
        
        # Edge should not be added for missing node
        assert len(knowledge_graph.graph.edges) == 0
    
    def test_get_neighbors(self, populated_knowledge_graph):
        """Test getting neighbors of a node."""
        kg = populated_knowledge_graph
        
        # Find paper node
        paper_nodes = [n for n, d in kg.graph.nodes(data=True) 
                       if d.get('type') == NodeType.PAPER]
        assert len(paper_nodes) > 0
        
        neighbors = kg.get_neighbors(paper_nodes[0])
        assert len(neighbors) > 0
    
    def test_find_path(self, populated_knowledge_graph):
        """Test finding path between nodes."""
        kg = populated_knowledge_graph
        
        # Get paper and function nodes
        paper_node = None
        func_node = None
        
        for node_id, data in kg.graph.nodes(data=True):
            if data.get('type') == NodeType.PAPER:
                paper_node = node_id
            elif data.get('type') == NodeType.FUNCTION:
                func_node = node_id
        
        if paper_node and func_node:
            path = kg.find_path(paper_node, func_node)
            # Path may or may not exist depending on graph structure
            assert path is None or len(path) > 0
    
    def test_text_search(self, populated_knowledge_graph):
        """Test text-based search."""
        kg = populated_knowledge_graph
        
        # Search for "neural"
        results = kg.text_search("neural")
        assert len(results) > 0
    
    def test_text_search_no_results(self, knowledge_graph):
        """Test search with no results."""
        node = create_concept_node("python", "Programming language")
        knowledge_graph.add_node(node)
        
        results = knowledge_graph.text_search("nonexistent_term_xyz")
        assert len(results) == 0
    
    def test_get_subgraph(self, populated_knowledge_graph):
        """Test extracting subgraph."""
        kg = populated_knowledge_graph
        
        # Get nodes by type
        concept_nodes = [n for n, d in kg.graph.nodes(data=True) 
                        if d.get('type') == NodeType.CONCEPT]
        
        if concept_nodes:
            subgraph = kg.get_subgraph(concept_nodes)
            assert len(subgraph.graph.nodes) <= len(kg.graph.nodes)
    
    def test_to_d3_format(self, populated_knowledge_graph):
        """Test D3.js format export."""
        kg = populated_knowledge_graph
        d3_data = kg.to_d3_format()
        
        assert "nodes" in d3_data
        assert "links" in d3_data
        assert len(d3_data["nodes"]) > 0
        
        # Check node structure
        if d3_data["nodes"]:
            node = d3_data["nodes"][0]
            assert "id" in node
            assert "type" in node
            assert "label" in node
    
    def test_json_serialization(self, populated_knowledge_graph, tmp_path):
        """Test JSON save and load."""
        kg = populated_knowledge_graph
        
        # Save
        json_path = tmp_path / "kg.json"
        kg.to_json(str(json_path))
        
        assert json_path.exists()
        
        # Load
        loaded_kg = KnowledgeGraph.from_json(str(json_path))
        
        assert len(loaded_kg.graph.nodes) == len(kg.graph.nodes)
        assert len(loaded_kg.graph.edges) == len(kg.graph.edges)


class TestNodeCreationHelpers:
    """Tests for node creation helper functions."""
    
    def test_create_paper_node(self):
        """Test paper node creation."""
        node = create_paper_node(
            paper_id="arxiv:2301.00001",
            title="Test Paper",
            authors=["Author One", "Author Two"],
            abstract="This is a test abstract."
        )
        
        assert node.type == NodeType.PAPER
        assert "2301.00001" in node.id
        assert node.label == "Test Paper"
        assert node.properties["title"] == "Test Paper"
        assert len(node.properties["authors"]) == 2
    
    def test_create_concept_node(self):
        """Test concept node creation."""
        node = create_concept_node(
            name="attention_mechanism",
            description="Self-attention in transformers"
        )
        
        assert node.type == NodeType.CONCEPT
        assert "attention_mechanism" in node.id
        assert node.label == "attention_mechanism"
    
    def test_create_function_node(self):
        """Test function node creation."""
        node = create_function_node(
            name="forward",
            file_path="/model.py",
            signature="def forward(self, x)",
            docstring="Forward pass of the model"
        )
        
        assert node.type == NodeType.FUNCTION
        assert node.label == "forward"
        assert node.properties["file_path"] == "/model.py"
    
    def test_create_class_node(self):
        """Test class node creation."""
        node = create_class_node(
            name="TransformerModel",
            file_path="/model.py",
            methods=["forward", "train", "eval"],
            bases=["nn.Module"],
            docstring="A transformer model implementation"
        )
        
        assert node.type == NodeType.CLASS
        assert node.label == "TransformerModel"
        assert "nn.Module" in node.properties["bases"]
    
    def test_create_mapping_node(self):
        """Test mapping node creation."""
        node = create_mapping_node(
            concept_name="attention",
            code_element="MultiHeadAttention",
            confidence=0.95,
            evidence="Direct implementation of attention mechanism"
        )
        
        assert node.type == NodeType.MAPPING
        assert node.properties["confidence"] == 0.95
        assert "attention" in node.properties["concept"]


class TestNodeTypes:
    """Tests for NodeType enum."""
    
    def test_all_node_types_exist(self):
        """Verify all expected node types exist."""
        expected_types = [
            "PAPER", "SECTION", "CONCEPT", "ALGORITHM", "EQUATION",
            "FIGURE", "CITATION", "REPOSITORY", "FILE", "CLASS",
            "FUNCTION", "DEPENDENCY", "TEST", "RESULT", "VISUALIZATION",
            "ERROR", "AGENT", "INSIGHT", "MAPPING"
        ]
        
        for type_name in expected_types:
            assert hasattr(NodeType, type_name)


class TestEdgeTypes:
    """Tests for EdgeType enum."""
    
    def test_all_edge_types_exist(self):
        """Verify all expected edge types exist."""
        expected_types = [
            "CONTAINS", "IMPLEMENTS", "DEPENDS_ON", "REFERENCES",
            "SIMILAR_TO", "GENERATES", "VALIDATES", "MAPS_TO",
            "IMPORTS", "CALLS", "INHERITS", "PRODUCED_BY", "DESCRIBES"
        ]
        
        for type_name in expected_types:
            assert hasattr(EdgeType, type_name)
