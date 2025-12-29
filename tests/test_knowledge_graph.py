"""
Tests for the Knowledge Graph module.

This module tests the knowledge graph functionality including:
- KGNode and KGEdge Pydantic models
- Node and edge creation with metadata
- Semantic search using embeddings
- Text-based search
- Graph traversal and path finding
- Subgraph extraction
- Serialization (JSON, D3 format)
- Statistics and analysis
"""

import pytest
import json
import numpy as np
from pathlib import Path

from core.knowledge_graph import (
    KnowledgeGraph,
    KGNode,
    KGEdge,
    NodeType,
    EdgeType,
    create_paper_node,
    create_concept_node,
    create_function_node,
    create_mapping_node,
)


class TestKGNode:
    """Tests for KGNode dataclass."""

    def test_create_node(self):
        """Test basic node creation."""
        node = KGNode(
            id="test_node_1",
            type=NodeType.CONCEPT,
            name="test_concept",
            description="A test concept"
        )

        assert node.id == "test_node_1"
        assert node.type == NodeType.CONCEPT
        assert node.name == "test_concept"
        assert node.description == "A test concept"
        assert node.embedding is None

    def test_node_with_embedding(self):
        """Test node with embedding vector."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        node = KGNode(
            id="embed_node",
            type=NodeType.CONCEPT,
            name="embedded",
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
        # create_concept_node already adds node to graph and returns node_id
        node_id = create_concept_node(knowledge_graph, name="test_concept", description="A test concept")

        assert knowledge_graph.get_node(node_id) is not None
        assert len(knowledge_graph.graph.nodes) == 1

    def test_add_duplicate_node(self, knowledge_graph):
        """Test adding node with same name creates separate nodes."""
        node1_id = create_concept_node(knowledge_graph, name="test", description="First description")
        node2_id = create_concept_node(knowledge_graph, name="test", description="Updated description")

        # Each call creates a new node with unique ID
        assert len(knowledge_graph.graph.nodes) == 2
        assert node1_id != node2_id

    def test_add_edge(self, knowledge_graph):
        """Test adding an edge between nodes."""
        node1_id = create_concept_node(knowledge_graph, name="concept1", description="First concept")
        node2_id = create_concept_node(knowledge_graph, name="concept2", description="Second concept")

        knowledge_graph.add_edge(node1_id, node2_id, EdgeType.DEPENDS_ON)

        assert len(knowledge_graph.graph.edges) == 1
        assert knowledge_graph.graph.has_edge(node1_id, node2_id)

    def test_add_edge_missing_node(self, knowledge_graph):
        """Test adding edge with missing node raises error."""
        node1_id = create_concept_node(knowledge_graph, name="concept1", description="First concept")

        # Should raise error for missing target node
        with pytest.raises(ValueError):
            knowledge_graph.add_edge(node1_id, "nonexistent", EdgeType.DEPENDS_ON)

    def test_get_neighbors(self, populated_knowledge_graph):
        """Test getting neighbors of a node."""
        kg = populated_knowledge_graph

        # Find paper node
        paper_nodes = [n for n, d in kg.graph.nodes(data=True)
                       if d.get('type') == 'paper']
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
        create_concept_node(knowledge_graph, name="python", description="Programming language")

        results = knowledge_graph.text_search("nonexistent_term_xyz")
        assert len(results) == 0

    def test_get_subgraph(self, populated_knowledge_graph):
        """Test extracting subgraph."""
        kg = populated_knowledge_graph

        # Get nodes by type (type is stored as string value)
        concept_nodes = [n for n, d in kg.graph.nodes(data=True)
                        if d.get('type') == 'concept']

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
            assert "name" in node  # field is 'name', not 'label'

    def test_json_serialization(self, populated_knowledge_graph, tmp_path):
        """Test JSON save and load."""
        kg = populated_knowledge_graph

        # Save
        json_path = tmp_path / "kg.json"
        kg.save(str(json_path))

        assert json_path.exists()

        # Load
        loaded_kg = KnowledgeGraph.load(str(json_path))

        assert len(loaded_kg.graph.nodes) == len(kg.graph.nodes)
        assert len(loaded_kg.graph.edges) == len(kg.graph.edges)


class TestNodeCreationHelpers:
    """Tests for node creation helper functions."""
    
    def test_create_paper_node(self, knowledge_graph):
        """Test paper node creation."""
        node_id = create_paper_node(
            knowledge_graph,
            title="Test Paper",
            authors=["Author One", "Author Two"],
            abstract="This is a test abstract."
        )

        assert isinstance(node_id, str)
        node = knowledge_graph.get_node(node_id)
        assert node.type == NodeType.PAPER
        assert node.name == "Test Paper"

    def test_create_concept_node(self, knowledge_graph):
        """Test concept node creation."""
        node_id = create_concept_node(
            knowledge_graph,
            name="attention_mechanism",
            description="Self-attention in transformers"
        )

        assert isinstance(node_id, str)
        node = knowledge_graph.get_node(node_id)
        assert node.type == NodeType.CONCEPT
        assert node.name == "attention_mechanism"

    def test_create_function_node(self, knowledge_graph):
        """Test function node creation."""
        node_id = create_function_node(
            knowledge_graph,
            name="forward",
            file_path="/model.py",
            signature="def forward(self, x)",
            docstring="Forward pass of the model"
        )

        assert isinstance(node_id, str)
        node = knowledge_graph.get_node(node_id)
        assert node.type == NodeType.FUNCTION
        assert node.name == "forward"
        assert node.metadata["file_path"] == "/model.py"

    def test_create_mapping_node(self, knowledge_graph):
        """Test mapping node creation."""
        # First create concept and code nodes
        concept_id = create_concept_node(knowledge_graph, name="attention", description="Attention mechanism")
        code_id = create_function_node(knowledge_graph, name="MultiHeadAttention", file_path="/model.py")

        mapping_id = create_mapping_node(
            knowledge_graph,
            concept_id=concept_id,
            code_id=code_id,
            confidence=0.95,
            evidence=["Direct implementation of attention mechanism"]
        )

        assert isinstance(mapping_id, str)
        node = knowledge_graph.get_node(mapping_id)
        assert node.type == NodeType.MAPPING
        assert node.metadata["confidence"] == 0.95


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


# =============================================================================
# KGEdge Model Tests
# =============================================================================

class TestKGEdge:
    """Tests for KGEdge Pydantic model."""

    def test_create_edge(self):
        """Test basic edge creation."""
        edge = KGEdge(
            source="node1",
            target="node2",
            type=EdgeType.IMPLEMENTS
        )

        assert edge.source == "node1"
        assert edge.target == "node2"
        assert edge.type == EdgeType.IMPLEMENTS
        assert edge.weight == 1.0
        assert edge.metadata == {}

    def test_edge_with_weight(self):
        """Test edge with custom weight."""
        edge = KGEdge(
            source="node1",
            target="node2",
            type=EdgeType.SIMILAR_TO,
            weight=0.85
        )

        assert edge.weight == 0.85

    def test_edge_with_metadata(self):
        """Test edge with metadata."""
        edge = KGEdge(
            source="node1",
            target="node2",
            type=EdgeType.CALLS,
            metadata={"line_number": 42, "context": "function call"}
        )

        assert edge.metadata["line_number"] == 42
        assert edge.metadata["context"] == "function call"

    def test_edge_has_timestamp(self):
        """Test edge has creation timestamp."""
        edge = KGEdge(
            source="node1",
            target="node2",
            type=EdgeType.CONTAINS
        )

        assert edge.created_at is not None


# =============================================================================
# Semantic Search Tests
# =============================================================================

class TestSemanticSearch:
    """Tests for semantic search functionality."""

    def test_semantic_search_basic(self, knowledge_graph):
        """Test basic semantic search with embeddings."""
        # Create nodes with embeddings
        emb1 = [0.1, 0.2, 0.3, 0.4, 0.5]
        emb2 = [0.1, 0.2, 0.3, 0.4, 0.6]  # Similar to emb1
        emb3 = [0.9, 0.8, 0.7, 0.6, 0.5]  # Different from emb1

        knowledge_graph.add_node(
            type=NodeType.CONCEPT,
            name="concept1",
            description="First concept",
            embedding=emb1
        )
        knowledge_graph.add_node(
            type=NodeType.CONCEPT,
            name="concept2",
            description="Second concept",
            embedding=emb2
        )
        knowledge_graph.add_node(
            type=NodeType.CONCEPT,
            name="concept3",
            description="Third concept",
            embedding=emb3
        )

        # Search with query similar to emb1 and emb2
        query = [0.1, 0.2, 0.3, 0.4, 0.55]
        results = knowledge_graph.semantic_search(query, top_k=2)

        assert len(results) == 2
        # First result should be more similar
        assert results[0][1] >= results[1][1]

    def test_semantic_search_empty_index(self, knowledge_graph):
        """Test semantic search on graph with no embeddings."""
        # Add node without embedding
        knowledge_graph.add_node(
            type=NodeType.CONCEPT,
            name="no_embedding",
            description="Node without embedding"
        )

        results = knowledge_graph.semantic_search([0.1, 0.2, 0.3])
        assert len(results) == 0

    def test_semantic_search_with_type_filter(self, knowledge_graph):
        """Test semantic search filtered by node type."""
        emb = [0.1, 0.2, 0.3]

        knowledge_graph.add_node(
            type=NodeType.CONCEPT,
            name="concept",
            embedding=emb
        )
        knowledge_graph.add_node(
            type=NodeType.FUNCTION,
            name="function",
            embedding=emb
        )

        # Search only for concepts
        results = knowledge_graph.semantic_search(
            emb,
            node_type=NodeType.CONCEPT,
            top_k=10
        )

        assert all(node.type == NodeType.CONCEPT for node, _ in results)

    def test_semantic_search_top_k(self, knowledge_graph):
        """Test semantic search respects top_k parameter."""
        emb = [0.1, 0.2, 0.3]

        for i in range(10):
            knowledge_graph.add_node(
                type=NodeType.CONCEPT,
                name=f"concept_{i}",
                embedding=[0.1 + i * 0.01, 0.2, 0.3]
            )

        results = knowledge_graph.semantic_search(emb, top_k=3)
        assert len(results) == 3

    def test_semantic_search_similarity_score(self, knowledge_graph):
        """Test that semantic search returns valid similarity scores."""
        emb = [1.0, 0.0, 0.0]

        knowledge_graph.add_node(
            type=NodeType.CONCEPT,
            name="identical",
            embedding=[1.0, 0.0, 0.0]
        )
        knowledge_graph.add_node(
            type=NodeType.CONCEPT,
            name="orthogonal",
            embedding=[0.0, 1.0, 0.0]
        )

        results = knowledge_graph.semantic_search(emb, top_k=2)

        # Identical vector should have similarity close to 1
        assert results[0][1] > 0.99
        # Orthogonal vector should have similarity close to 0
        assert results[1][1] < 0.1


# =============================================================================
# Graph Statistics Tests
# =============================================================================

class TestGraphStatistics:
    """Tests for graph statistics functionality."""

    def test_statistics_empty_graph(self, knowledge_graph):
        """Test statistics on empty graph."""
        stats = knowledge_graph.get_statistics()

        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0
        assert stats["has_embeddings"] == 0
        assert stats["is_connected"] is True  # Empty graph is technically connected

    def test_statistics_with_nodes(self, knowledge_graph):
        """Test statistics with nodes."""
        knowledge_graph.add_node(type=NodeType.CONCEPT, name="c1")
        knowledge_graph.add_node(type=NodeType.CONCEPT, name="c2")
        knowledge_graph.add_node(type=NodeType.FUNCTION, name="f1")

        stats = knowledge_graph.get_statistics()

        assert stats["total_nodes"] == 3
        assert "concept" in stats["node_types"]
        assert stats["node_types"]["concept"] == 2

    def test_statistics_with_edges(self, knowledge_graph):
        """Test statistics with edges."""
        n1 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="c1")
        n2 = knowledge_graph.add_node(type=NodeType.FUNCTION, name="f1")
        knowledge_graph.add_edge(n1, n2, EdgeType.IMPLEMENTS)

        stats = knowledge_graph.get_statistics()

        assert stats["total_edges"] == 1
        assert stats["density"] > 0

    def test_statistics_with_embeddings(self, knowledge_graph):
        """Test statistics counts embeddings."""
        knowledge_graph.add_node(
            type=NodeType.CONCEPT,
            name="with_emb",
            embedding=[0.1, 0.2, 0.3]
        )
        knowledge_graph.add_node(
            type=NodeType.CONCEPT,
            name="without_emb"
        )

        stats = knowledge_graph.get_statistics()

        assert stats["has_embeddings"] == 1

    def test_statistics_connected_graph(self, knowledge_graph):
        """Test connectivity detection."""
        n1 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="c1")
        n2 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="c2")
        n3 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="c3")

        # Connect all nodes
        knowledge_graph.add_edge(n1, n2, EdgeType.REFERENCES)
        knowledge_graph.add_edge(n2, n3, EdgeType.REFERENCES)

        stats = knowledge_graph.get_statistics()
        assert stats["is_connected"] is True

    def test_statistics_disconnected_graph(self, knowledge_graph):
        """Test disconnected graph detection."""
        knowledge_graph.add_node(type=NodeType.CONCEPT, name="c1")
        knowledge_graph.add_node(type=NodeType.CONCEPT, name="c2")
        # No edges - disconnected

        stats = knowledge_graph.get_statistics()
        assert stats["is_connected"] is False


# =============================================================================
# Subgraph Extraction Tests
# =============================================================================

class TestSubgraphExtraction:
    """Tests for subgraph extraction functionality."""

    def test_get_subgraph_depth_0(self, knowledge_graph):
        """Test subgraph with depth 0 (just the node)."""
        n1 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="center")
        n2 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="neighbor")
        knowledge_graph.add_edge(n1, n2, EdgeType.REFERENCES)

        subgraph = knowledge_graph.get_subgraph([n1], depth=0)

        # With depth 0, only expand within the initial set
        assert len(subgraph.graph.nodes) >= 1

    def test_get_subgraph_depth_1(self, knowledge_graph):
        """Test subgraph with depth 1 (node + immediate neighbors)."""
        n1 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="center")
        n2 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="neighbor1")
        n3 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="neighbor2")
        n4 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="distant")

        knowledge_graph.add_edge(n1, n2, EdgeType.REFERENCES)
        knowledge_graph.add_edge(n1, n3, EdgeType.REFERENCES)
        knowledge_graph.add_edge(n3, n4, EdgeType.REFERENCES)

        subgraph = knowledge_graph.get_subgraph([n1], depth=1)

        # Should include center and neighbors, not distant
        assert n1 in subgraph.graph.nodes
        assert n2 in subgraph.graph.nodes
        assert n3 in subgraph.graph.nodes

    def test_get_subgraph_depth_2(self, knowledge_graph):
        """Test subgraph with depth 2."""
        n1 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="center")
        n2 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="neighbor")
        n3 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="level2")

        knowledge_graph.add_edge(n1, n2, EdgeType.REFERENCES)
        knowledge_graph.add_edge(n2, n3, EdgeType.REFERENCES)

        subgraph = knowledge_graph.get_subgraph([n1], depth=2)

        # Should include all nodes at depth 2
        assert len(subgraph.graph.nodes) == 3
        assert n3 in subgraph.graph.nodes

    def test_subgraph_preserves_edges(self, knowledge_graph):
        """Test that subgraph preserves edge data."""
        n1 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="c1")
        n2 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="c2")
        knowledge_graph.add_edge(n1, n2, EdgeType.IMPLEMENTS, weight=0.9)

        subgraph = knowledge_graph.get_subgraph([n1], depth=1)

        assert subgraph.graph.has_edge(n1, n2)
        edge_data = subgraph.graph.edges[n1, n2]
        assert edge_data["weight"] == 0.9


# =============================================================================
# Neighbor Traversal Tests
# =============================================================================

class TestNeighborTraversal:
    """Tests for neighbor traversal functionality."""

    def test_get_neighbors_outgoing(self, knowledge_graph):
        """Test getting outgoing neighbors only."""
        n1 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="source")
        n2 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="target")
        n3 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="predecessor")

        knowledge_graph.add_edge(n1, n2, EdgeType.REFERENCES)  # n1 -> n2
        knowledge_graph.add_edge(n3, n1, EdgeType.REFERENCES)  # n3 -> n1

        neighbors = knowledge_graph.get_neighbors(n1, direction="out")

        # Should only include n2 (outgoing)
        neighbor_names = [n.name for n in neighbors]
        assert "target" in neighbor_names
        assert "predecessor" not in neighbor_names

    def test_get_neighbors_incoming(self, knowledge_graph):
        """Test getting incoming neighbors only."""
        n1 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="center")
        n2 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="successor")
        n3 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="predecessor")

        knowledge_graph.add_edge(n1, n2, EdgeType.REFERENCES)  # n1 -> n2
        knowledge_graph.add_edge(n3, n1, EdgeType.REFERENCES)  # n3 -> n1

        neighbors = knowledge_graph.get_neighbors(n1, direction="in")

        # Should only include n3 (incoming)
        neighbor_names = [n.name for n in neighbors]
        assert "predecessor" in neighbor_names
        assert "successor" not in neighbor_names

    def test_get_neighbors_both_directions(self, knowledge_graph):
        """Test getting neighbors in both directions."""
        n1 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="center")
        n2 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="successor")
        n3 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="predecessor")

        knowledge_graph.add_edge(n1, n2, EdgeType.REFERENCES)
        knowledge_graph.add_edge(n3, n1, EdgeType.REFERENCES)

        neighbors = knowledge_graph.get_neighbors(n1, direction="both")

        # Should include both
        assert len(neighbors) == 2

    def test_get_neighbors_by_edge_type(self, knowledge_graph):
        """Test filtering neighbors by edge type."""
        n1 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="concept")
        n2 = knowledge_graph.add_node(type=NodeType.FUNCTION, name="function")
        n3 = knowledge_graph.add_node(type=NodeType.CONCEPT, name="reference")

        knowledge_graph.add_edge(n1, n2, EdgeType.IMPLEMENTS)
        knowledge_graph.add_edge(n1, n3, EdgeType.REFERENCES)

        # Get only IMPLEMENTS neighbors
        neighbors = knowledge_graph.get_neighbors(n1, edge_type=EdgeType.IMPLEMENTS)

        assert len(neighbors) == 1
        assert neighbors[0].name == "function"


# =============================================================================
# Get Nodes By Type Tests
# =============================================================================

class TestGetNodesByType:
    """Tests for getting nodes by type."""

    def test_get_nodes_by_type_empty(self, knowledge_graph):
        """Test getting nodes of type with no matches."""
        knowledge_graph.add_node(type=NodeType.CONCEPT, name="concept")

        functions = knowledge_graph.get_nodes_by_type(NodeType.FUNCTION)
        assert len(functions) == 0

    def test_get_nodes_by_type_multiple(self, knowledge_graph):
        """Test getting multiple nodes of same type."""
        knowledge_graph.add_node(type=NodeType.CONCEPT, name="c1")
        knowledge_graph.add_node(type=NodeType.CONCEPT, name="c2")
        knowledge_graph.add_node(type=NodeType.FUNCTION, name="f1")

        concepts = knowledge_graph.get_nodes_by_type(NodeType.CONCEPT)
        assert len(concepts) == 2

        functions = knowledge_graph.get_nodes_by_type(NodeType.FUNCTION)
        assert len(functions) == 1


# =============================================================================
# JSON Serialization Tests
# =============================================================================

class TestJSONSerialization:
    """Tests for JSON serialization functionality."""

    def test_to_json_and_back(self, knowledge_graph):
        """Test round-trip JSON serialization."""
        n1 = knowledge_graph.add_node(
            type=NodeType.CONCEPT,
            name="test_concept",
            description="A test concept",
            metadata={"key": "value"}
        )
        n2 = knowledge_graph.add_node(
            type=NodeType.FUNCTION,
            name="test_function"
        )
        knowledge_graph.add_edge(n1, n2, EdgeType.IMPLEMENTS, weight=0.8)

        # Serialize
        json_str = knowledge_graph.to_json()
        assert isinstance(json_str, str)

        # Deserialize
        loaded_kg = KnowledgeGraph.from_json(json_str)

        assert len(loaded_kg.graph.nodes) == 2
        assert len(loaded_kg.graph.edges) == 1

    def test_json_preserves_metadata(self, knowledge_graph):
        """Test that JSON preserves node metadata."""
        knowledge_graph.add_node(
            type=NodeType.FUNCTION,
            name="func",
            metadata={"file_path": "/test.py", "line": 42}
        )

        json_str = knowledge_graph.to_json()
        loaded_kg = KnowledgeGraph.from_json(json_str)

        nodes = loaded_kg.get_nodes_by_type(NodeType.FUNCTION)
        assert len(nodes) == 1
        assert nodes[0].metadata["file_path"] == "/test.py"

    def test_json_includes_statistics(self, knowledge_graph):
        """Test that to_json includes statistics."""
        knowledge_graph.add_node(type=NodeType.CONCEPT, name="c1")

        json_str = knowledge_graph.to_json()
        data = json.loads(json_str)

        assert "statistics" in data
        assert data["statistics"]["total_nodes"] == 1


# =============================================================================
# Additional Text Search Tests
# =============================================================================

class TestTextSearchAdditional:
    """Additional tests for text search functionality."""

    def test_text_search_case_insensitive(self, knowledge_graph):
        """Test that text search is case insensitive."""
        knowledge_graph.add_node(
            type=NodeType.CONCEPT,
            name="UPPERCASE",
            description="lowercase description"
        )

        results = knowledge_graph.text_search("uppercase")
        assert len(results) == 1

        results = knowledge_graph.text_search("LOWERCASE")
        assert len(results) == 1

    def test_text_search_partial_match(self, knowledge_graph):
        """Test text search matches partial strings."""
        knowledge_graph.add_node(
            type=NodeType.CONCEPT,
            name="transformer_attention",
            description="Multi-head attention mechanism"
        )

        results = knowledge_graph.text_search("attention")
        assert len(results) == 1

    def test_text_search_respects_top_k(self, knowledge_graph):
        """Test text search respects top_k limit."""
        for i in range(10):
            knowledge_graph.add_node(
                type=NodeType.CONCEPT,
                name=f"concept_{i}",
                description="similar description"
            )

        results = knowledge_graph.text_search("concept", top_k=5)
        assert len(results) == 5

    def test_text_search_prioritizes_name_match(self, knowledge_graph):
        """Test that name matches are ranked higher than description matches."""
        knowledge_graph.add_node(
            type=NodeType.CONCEPT,
            name="attention",
            description="Not the query"
        )
        knowledge_graph.add_node(
            type=NodeType.CONCEPT,
            name="other",
            description="This mentions attention"
        )

        results = knowledge_graph.text_search("attention")

        # Name match should come first
        assert results[0].name == "attention"

    def test_text_search_with_node_type_filter(self, knowledge_graph):
        """Test text search with node type filter."""
        knowledge_graph.add_node(
            type=NodeType.CONCEPT,
            name="attention_concept",
            description="Concept"
        )
        knowledge_graph.add_node(
            type=NodeType.FUNCTION,
            name="attention_function",
            description="Function"
        )

        results = knowledge_graph.text_search("attention", node_type=NodeType.CONCEPT)

        assert len(results) == 1
        assert results[0].type == NodeType.CONCEPT
