"""
Knowledge Graph - Shared memory system for the multi-agent pipeline.
"""

import json
import uuid
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any, List, Set, Tuple
from pathlib import Path

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class NodeType(Enum):
    """Node types for the knowledge graph."""
    # Paper-related
    PAPER = "paper"
    SECTION = "section"
    CONCEPT = "concept"
    ALGORITHM = "algorithm"
    EQUATION = "equation"
    FIGURE = "figure"
    CITATION = "citation"
    
    # Code-related
    REPOSITORY = "repository"
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    DEPENDENCY = "dependency"
    MODULE = "module"
    
    # Execution-related
    TEST = "test"
    RESULT = "result"
    VISUALIZATION = "visualization"
    ERROR = "error"
    
    # Meta
    AGENT = "agent"
    INSIGHT = "insight"
    MAPPING = "mapping"


class EdgeType(Enum):
    """Relationship types for edges."""
    CONTAINS = "contains"
    IMPLEMENTS = "implements"
    DEPENDS_ON = "depends_on"
    REFERENCES = "references"
    SIMILAR_TO = "similar_to"
    GENERATES = "generates"
    VALIDATES = "validates"
    MAPS_TO = "maps_to"
    IMPORTS = "imports"
    CALLS = "calls"
    INHERITS = "inherits"
    PRODUCED_BY = "produced_by"
    DESCRIBES = "describes"


class KGNode(BaseModel):
    """Knowledge graph node with metadata."""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Unique node identifier")
    type: NodeType = Field(..., description="Node type")
    name: str = Field(..., description="Node name")
    description: str = Field("", description="Node description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Update timestamp")


class KGEdge(BaseModel):
    """Knowledge graph edge with metadata."""
    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    type: EdgeType = Field(..., description="Edge type")
    weight: float = Field(1.0, description="Edge weight")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class KnowledgeGraph:
    """
    NetworkX-based knowledge graph for storing and querying
    paper concepts, code elements, and their relationships.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self._node_registry: Dict[str, KGNode] = {}
        self._embeddings_index: Dict[str, np.ndarray] = {}
        self._type_index: Dict[NodeType, Set[str]] = {t: set() for t in NodeType}
    
    def add_node(
        self,
        type: NodeType,
        name: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        node_id: Optional[str] = None
    ) -> str:
        """Add a node to the knowledge graph."""
        node_id = node_id or str(uuid.uuid4())[:8]
        
        node = KGNode(
            id=node_id,
            type=type,
            name=name,
            description=description,
            metadata=metadata or {},
            embedding=embedding
        )
        
        self._node_registry[node_id] = node
        self._type_index[type].add(node_id)
        
        if embedding:
            self._embeddings_index[node_id] = np.array(embedding)
        
        self.graph.add_node(
            node_id,
            type=type.value,
            name=name,
            description=description,
            metadata=metadata or {}
        )
        
        return node_id
    
    def add_edge(
        self,
        source: str,
        target: str,
        type: EdgeType,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """Add an edge between two nodes."""
        if source not in self._node_registry:
            raise ValueError(f"Source node {source} not found")
        if target not in self._node_registry:
            raise ValueError(f"Target node {target} not found")
        
        edge = KGEdge(
            source=source,
            target=target,
            type=type,
            weight=weight,
            metadata=metadata or {}
        )
        
        self.graph.add_edge(
            source,
            target,
            type=type.value,
            weight=weight,
            metadata=metadata or {}
        )
        
        return (source, target)
    
    def get_node(self, node_id: str) -> Optional[KGNode]:
        """Get a node by ID."""
        return self._node_registry.get(node_id)
    
    def get_nodes_by_type(self, type: NodeType) -> List[KGNode]:
        """Get all nodes of a specific type."""
        node_ids = self._type_index.get(type, set())
        return [self._node_registry[nid] for nid in node_ids if nid in self._node_registry]
    
    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        direction: str = "both"
    ) -> List[KGNode]:
        """Get neighboring nodes, optionally filtered by edge type."""
        neighbors = []
        
        if direction in ("both", "out"):
            for _, target, data in self.graph.out_edges(node_id, data=True):
                if edge_type is None or data.get("type") == edge_type.value:
                    if target in self._node_registry:
                        neighbors.append(self._node_registry[target])
        
        if direction in ("both", "in"):
            for source, _, data in self.graph.in_edges(node_id, data=True):
                if edge_type is None or data.get("type") == edge_type.value:
                    if source in self._node_registry:
                        neighbors.append(self._node_registry[source])
        
        return neighbors
    
    def find_path(
        self,
        source: str,
        target: str
    ) -> Optional[List[str]]:
        """Find shortest path between two nodes."""
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None
    
    def semantic_search(
        self,
        query_embedding: List[float],
        node_type: Optional[NodeType] = None,
        top_k: int = 5
    ) -> List[Tuple[KGNode, float]]:
        """Search for similar nodes using embedding similarity."""
        if not self._embeddings_index:
            return []
        
        query_vec = np.array(query_embedding)
        
        candidates = []
        for node_id, emb in self._embeddings_index.items():
            node = self._node_registry.get(node_id)
            if node and (node_type is None or node.type == node_type):
                # Cosine similarity
                similarity = np.dot(query_vec, emb) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(emb) + 1e-8
                )
                candidates.append((node, float(similarity)))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def text_search(
        self,
        query: str,
        node_type: Optional[NodeType] = None,
        top_k: int = 10
    ) -> List[KGNode]:
        """Simple text-based search on node names and descriptions."""
        query_lower = query.lower()
        results = []
        
        for node in self._node_registry.values():
            if node_type and node.type != node_type:
                continue
            
            # Score based on query presence
            score = 0
            if query_lower in node.name.lower():
                score += 2
            if query_lower in node.description.lower():
                score += 1
            
            if score > 0:
                results.append((node, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in results[:top_k]]
    
    def get_subgraph(
        self,
        node_ids: List[str],
        depth: int = 1
    ) -> 'KnowledgeGraph':
        """Extract a subgraph around specified nodes."""
        expanded_nodes = set(node_ids)
        
        for _ in range(depth):
            new_nodes = set()
            for node_id in expanded_nodes:
                if node_id in self.graph:
                    new_nodes.update(self.graph.predecessors(node_id))
                    new_nodes.update(self.graph.successors(node_id))
            expanded_nodes.update(new_nodes)
        
        subgraph = KnowledgeGraph()
        
        for node_id in expanded_nodes:
            if node_id in self._node_registry:
                node = self._node_registry[node_id]
                subgraph.add_node(
                    type=node.type,
                    name=node.name,
                    description=node.description,
                    metadata=node.metadata,
                    embedding=node.embedding,
                    node_id=node.id
                )
        
        for source, target, data in self.graph.edges(data=True):
            if source in expanded_nodes and target in expanded_nodes:
                subgraph.add_edge(
                    source=source,
                    target=target,
                    type=EdgeType(data["type"]),
                    weight=data.get("weight", 1.0),
                    metadata=data.get("metadata", {})
                )
        
        return subgraph
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        type_counts = {t.value: len(ids) for t, ids in self._type_index.items() if ids}
        
        return {
            "total_nodes": len(self._node_registry),
            "total_edges": self.graph.number_of_edges(),
            "node_types": type_counts,
            "has_embeddings": len(self._embeddings_index),
            "is_connected": nx.is_weakly_connected(self.graph) if len(self.graph) > 0 else True,
            "density": nx.density(self.graph) if len(self.graph) > 0 else 0
        }
    
    def to_d3_format(self) -> Dict[str, Any]:
        """Export graph in D3.js force-directed format."""
        nodes = []
        for node in self._node_registry.values():
            nodes.append({
                "id": node.id,
                "name": node.name,
                "type": node.type.value,
                "description": node.description,
                "metadata": node.metadata
            })
        
        edges = []
        for source, target, data in self.graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "type": data.get("type", "unknown"),
                "weight": data.get("weight", 1.0)
            })
        
        return {"nodes": nodes, "links": edges}
    
    def to_json(self) -> str:
        """Serialize graph to JSON."""
        data = {
            "nodes": [node.model_dump(mode='json', exclude={'embedding'}) for node in self._node_registry.values()],
            "edges": [],
            "statistics": self.get_statistics()
        }

        for source, target, edge_data in self.graph.edges(data=True):
            data["edges"].append({
                "source": source,
                "target": target,
                **edge_data
            })

        return json.dumps(data, indent=2, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'KnowledgeGraph':
        """Load graph from JSON."""
        data = json.loads(json_str)
        kg = cls()
        
        for node_data in data.get("nodes", []):
            kg.add_node(
                type=NodeType(node_data["type"]),
                name=node_data["name"],
                description=node_data.get("description", ""),
                metadata=node_data.get("metadata", {}),
                node_id=node_data["id"]
            )
        
        for edge_data in data.get("edges", []):
            kg.add_edge(
                source=edge_data["source"],
                target=edge_data["target"],
                type=EdgeType(edge_data["type"]),
                weight=edge_data.get("weight", 1.0),
                metadata=edge_data.get("metadata", {})
            )
        
        return kg
    
    def save(self, path: str) -> None:
        """Save graph to file."""
        Path(path).write_text(self.to_json())
    
    @classmethod
    def load(cls, path: str) -> 'KnowledgeGraph':
        """Load graph from file."""
        return cls.from_json(Path(path).read_text())


# Convenience functions for common operations
def create_paper_node(kg: KnowledgeGraph, title: str, **metadata) -> str:
    """Create a paper node."""
    return kg.add_node(
        type=NodeType.PAPER,
        name=title,
        description=metadata.get("abstract", ""),
        metadata=metadata
    )


def create_concept_node(
    kg: KnowledgeGraph,
    name: str,
    description: str = "",
    embedding: Optional[List[float]] = None,
    **metadata
) -> str:
    """Create a concept node."""
    return kg.add_node(
        type=NodeType.CONCEPT,
        name=name,
        description=description,
        embedding=embedding,
        metadata=metadata
    )


def create_function_node(
    kg: KnowledgeGraph,
    name: str,
    file_path: str,
    signature: str = "",
    docstring: str = "",
    **metadata
) -> str:
    """Create a function node."""
    return kg.add_node(
        type=NodeType.FUNCTION,
        name=name,
        description=docstring,
        metadata={
            "file_path": file_path,
            "signature": signature,
            **metadata
        }
    )


def create_mapping_node(
    kg: KnowledgeGraph,
    concept_id: str,
    code_id: str,
    confidence: float,
    evidence: List[str],
    **metadata
) -> str:
    """Create a mapping node linking concept to code."""
    concept = kg.get_node(concept_id)
    code = kg.get_node(code_id)
    
    mapping_id = kg.add_node(
        type=NodeType.MAPPING,
        name=f"Mapping: {concept.name if concept else concept_id} -> {code.name if code else code_id}",
        description=f"Confidence: {confidence:.2f}",
        metadata={
            "concept_id": concept_id,
            "code_id": code_id,
            "confidence": confidence,
            "evidence": evidence,
            **metadata
        }
    )
    
    # Connect mapping to concept and code
    kg.add_edge(concept_id, mapping_id, EdgeType.MAPS_TO)
    kg.add_edge(mapping_id, code_id, EdgeType.IMPLEMENTS)
    
    return mapping_id
