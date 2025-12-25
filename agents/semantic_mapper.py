"""
Semantic Mapper - Maps paper concepts to code implementations.
"""

import re
from typing import Optional, Dict, Any, List, Tuple
from difflib import SequenceMatcher

from .base_agent import BaseAgent
from core.llm_client import LLMClient
from core.knowledge_graph import (
    KnowledgeGraph, NodeType, EdgeType,
    create_mapping_node
)
from core.agent_prompts import (
    SEMANTIC_MAPPER_SYSTEM_PROMPT,
    SEMANTIC_MAPPER_PROMPT
)
from core.error_handling import logger, LogCategory


class SemanticMapper(BaseAgent):
    """
    Maps paper concepts to their code implementations using multiple signals.

    Signals used:
    1. Lexical matching - Name and terminology similarity
    2. Semantic similarity - Embedding-based similarity
    3. Structural patterns - Code structure matching algorithms
    4. Documentary evidence - Docstrings, comments mentioning paper
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client, name="SemanticMapper")
        self._embedder = None
        self._embeddings_cache: Dict[str, List[float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _init_embedder(self):
        """Initialize sentence transformer for embeddings."""
        if self._embedder is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.log_info("Initialized embedding model")
        except ImportError:
            self.log_warning("sentence-transformers not available, using LLM-only mapping")
            self._embedder = None
    
    async def process(
        self,
        *,
        paper_data: Dict[str, Any],
        repo_data: Dict[str, Any],
        knowledge_graph: KnowledgeGraph = None
    ) -> Dict[str, Any]:
        """
        Map paper concepts to code implementations.

        Args:
            paper_data: Extracted paper information (REQUIRED)
            repo_data: Analyzed repository data (REQUIRED)
            knowledge_graph: Optional knowledge graph to populate

        Returns:
            Dict with mappings list (SemanticMapperOutput)
        """
        if not paper_data:
            raise ValueError("paper_data is required")
        if not repo_data:
            raise ValueError("repo_data is required")
        if knowledge_graph is None:
            knowledge_graph = KnowledgeGraph()
        mappings = await self.map_concepts(paper_data, repo_data, knowledge_graph)
        return {"mappings": mappings}
    
    async def map_concepts(
        self,
        paper_data: Optional[Dict[str, Any]],
        repo_data: Optional[Dict[str, Any]],
        kg: KnowledgeGraph
    ) -> List[Dict[str, Any]]:
        """
        Map paper concepts to code elements.
        
        Args:
            paper_data: Extracted paper information
            repo_data: Analyzed repository data
            kg: Knowledge graph
            
        Returns:
            List of concept-code mappings with confidence scores
        """
        self.log_info("Starting concept-code mapping")
        
        if not paper_data or not repo_data:
            self.log_warning("Missing paper or repo data, returning empty mappings")
            return []
        
        # Extract concepts from paper
        concepts = self._extract_concepts(paper_data)
        if not concepts:
            self.log_warning("No concepts found in paper data")
            return []
        
        # Extract code elements from repo
        code_elements = self._extract_code_elements(repo_data)
        if not code_elements:
            self.log_warning("No code elements found in repo data")
            return []
        
        self.log_info(f"Mapping {len(concepts)} concepts to {len(code_elements)} code elements")
        
        # Step 1: Compute lexical similarity
        lexical_scores = await self._timed_operation(
            "lexical_matching",
            self._compute_lexical_similarity(concepts, code_elements)
        )
        
        # Step 2: Compute semantic similarity (if embedder available)
        semantic_scores = await self._timed_operation(
            "semantic_matching",
            self._compute_semantic_similarity(concepts, code_elements)
        )
        
        # Step 3: Check documentary evidence
        documentary_scores = await self._timed_operation(
            "documentary_matching",
            self._check_documentary_evidence(concepts, code_elements, paper_data)
        )
        
        # Step 4: Use LLM for final mapping with reasoning
        mappings = await self._timed_operation(
            "llm_mapping",
            self._llm_mapping(
                concepts, code_elements,
                lexical_scores, semantic_scores, documentary_scores
            )
        )
        
        # Step 5: Update knowledge graph
        await self._update_knowledge_graph(mappings, kg)
        
        return mappings
    
    def _extract_concepts(self, paper_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract concepts from paper data."""
        concepts = []
        
        # Key concepts
        for concept in paper_data.get("key_concepts", []):
            concepts.append({
                "name": concept.get("name", ""),
                "description": concept.get("description", ""),
                "type": "concept",
                "importance": concept.get("importance", "medium")
            })
        
        # Algorithms
        for algo in paper_data.get("algorithms", []):
            concepts.append({
                "name": algo.get("name", ""),
                "description": algo.get("description", ""),
                "type": "algorithm",
                "importance": "high"
            })
        
        # Expected implementations
        for impl in paper_data.get("expected_implementations", []):
            concepts.append({
                "name": impl.get("component", ""),
                "description": impl.get("description", ""),
                "type": "implementation",
                "importance": "high",
                "likely_names": impl.get("likely_function_names", [])
            })
        
        # Filter empty concepts
        concepts = [c for c in concepts if c["name"]]
        
        return concepts
    
    def _extract_code_elements(self, repo_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract code elements from repo data."""
        elements = []
        code_data = repo_data.get("_code_elements", {})
        
        # Classes
        for cls in code_data.get("classes", []):
            elements.append({
                "name": cls.get("name", ""),
                "type": "class",
                "file_path": cls.get("file_path", ""),
                "description": cls.get("docstring", ""),
                "methods": cls.get("methods", []),
                "signature": f"class {cls.get('name', '')}",
            })
        
        # Functions
        for func in code_data.get("functions", []):
            args = func.get("args", [])
            signature = f"def {func.get('name', '')}({', '.join(args)})"
            elements.append({
                "name": func.get("name", ""),
                "type": "function",
                "file_path": func.get("file_path", ""),
                "description": func.get("docstring", ""),
                "signature": signature,
            })
        
        # Filter empty elements
        elements = [e for e in elements if e["name"]]
        
        return elements
    
    async def _compute_lexical_similarity(
        self,
        concepts: List[Dict[str, Any]],
        code_elements: List[Dict[str, Any]]
    ) -> Dict[Tuple[str, str], float]:
        """Compute lexical similarity between concepts and code elements."""
        scores = {}
        
        for concept in concepts:
            concept_name = concept["name"].lower()
            concept_tokens = set(self._tokenize(concept_name))
            
            # Add likely names if available
            likely_names = concept.get("likely_names", [])
            likely_tokens = set()
            for name in likely_names:
                likely_tokens.update(self._tokenize(name.lower()))
            
            for element in code_elements:
                element_name = element["name"].lower()
                element_tokens = set(self._tokenize(element_name))
                
                # Direct name match
                direct_score = SequenceMatcher(
                    None, concept_name, element_name
                ).ratio()
                
                # Token overlap
                if concept_tokens and element_tokens:
                    token_overlap = len(concept_tokens & element_tokens) / \
                                  len(concept_tokens | element_tokens)
                else:
                    token_overlap = 0
                
                # Likely name match
                likely_score = 0
                if likely_tokens and element_tokens:
                    likely_score = len(likely_tokens & element_tokens) / \
                                  max(1, len(likely_tokens))
                
                # Combined score
                score = max(direct_score, token_overlap, likely_score)
                
                key = (concept["name"], element["name"])
                scores[key] = score
        
        return scores
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words, handling camelCase and snake_case."""
        # Split camelCase
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        # Split snake_case
        text = text.replace('_', ' ')
        # Split on non-alphanumeric
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text, using cache if available."""
        if text in self._embeddings_cache:
            self._cache_hits += 1
            return self._embeddings_cache[text]

        self._cache_misses += 1

        if self._embedder is None:
            return None

        try:
            embedding = self._embedder.encode(text).tolist()
            self._embeddings_cache[text] = embedding
            return embedding
        except Exception:
            return None

    async def _compute_semantic_similarity(
        self,
        concepts: List[Dict[str, Any]],
        code_elements: List[Dict[str, Any]]
    ) -> Dict[Tuple[str, str], float]:
        """Compute semantic similarity using embeddings with caching."""
        scores = {}

        self._init_embedder()

        if self._embedder is None:
            return scores

        try:
            import numpy as np

            # Prepare texts and get embeddings (with caching)
            concept_embeddings = []
            for c in concepts:
                text = f"{c['name']}: {c['description']}"
                emb = self._get_embedding(text)
                if emb:
                    concept_embeddings.append(np.array(emb))
                else:
                    concept_embeddings.append(None)

            element_embeddings = []
            for e in code_elements:
                text = f"{e['name']}: {e['description']}"
                emb = self._get_embedding(text)
                if emb:
                    element_embeddings.append(np.array(emb))
                else:
                    element_embeddings.append(None)

            # Compute similarities
            for i, concept in enumerate(concepts):
                c_emb = concept_embeddings[i]
                if c_emb is None:
                    continue

                for j, element in enumerate(code_elements):
                    e_emb = element_embeddings[j]
                    if e_emb is None:
                        continue

                    # Cosine similarity
                    similarity = np.dot(c_emb, e_emb) / (
                        np.linalg.norm(c_emb) * np.linalg.norm(e_emb) + 1e-8
                    )

                    key = (concept["name"], element["name"])
                    scores[key] = float(similarity)

            self.log_debug(
                f"Embedding cache stats: {self._cache_hits} hits, {self._cache_misses} misses"
            )

        except Exception as e:
            self.log_warning(f"Semantic similarity failed: {e}")

        return scores
    
    async def _check_documentary_evidence(
        self,
        concepts: List[Dict[str, Any]],
        code_elements: List[Dict[str, Any]],
        paper_data: Dict[str, Any]
    ) -> Dict[Tuple[str, str], float]:
        """Check for documentary evidence (paper references in code)."""
        scores = {}
        
        # Get paper title and key terms
        paper_title = paper_data.get("title", "").lower()
        paper_terms = set(self._tokenize(paper_title))
        
        # Add author names
        authors = paper_data.get("authors", [])
        author_terms = set()
        for author in authors:
            author_terms.update(self._tokenize(author.lower()))
        
        for concept in concepts:
            concept_terms = set(self._tokenize(concept["name"].lower()))
            
            for element in code_elements:
                score = 0
                doc = element.get("description", "").lower()
                
                # Check if docstring mentions concept
                if concept["name"].lower() in doc:
                    score += 0.8
                elif any(term in doc for term in concept_terms if len(term) > 3):
                    score += 0.4
                
                # Check for paper references
                if any(term in doc for term in paper_terms if len(term) > 4):
                    score += 0.3
                
                # Check for author mentions
                if any(author in doc for author in author_terms if len(author) > 4):
                    score += 0.2
                
                key = (concept["name"], element["name"])
                scores[key] = min(1.0, score)
        
        return scores
    
    async def _llm_mapping(
        self,
        concepts: List[Dict[str, Any]],
        code_elements: List[Dict[str, Any]],
        lexical_scores: Dict[Tuple[str, str], float],
        semantic_scores: Dict[Tuple[str, str], float],
        documentary_scores: Dict[Tuple[str, str], float]
    ) -> List[Dict[str, Any]]:
        """Use LLM for final mapping with reasoning."""
        # Prepare concepts summary
        concepts_text = ""
        for c in concepts[:20]:  # Limit
            concepts_text += f"- {c['name']} ({c['type']}): {c['description'][:200]}\n"
        
        # Prepare code elements summary with pre-computed scores
        elements_text = ""
        for e in code_elements[:50]:  # Limit
            # Find best matching concept for this element
            best_scores = []
            for c in concepts:
                key = (c["name"], e["name"])
                lex = lexical_scores.get(key, 0)
                sem = semantic_scores.get(key, 0)
                doc = documentary_scores.get(key, 0)
                combined = max(lex, sem, doc)
                if combined > 0.2:
                    best_scores.append((c["name"], lex, sem, doc))
            
            score_hint = ""
            if best_scores:
                best = max(best_scores, key=lambda x: max(x[1], x[2], x[3]))
                score_hint = f" [potential match: {best[0]}]"
            
            elements_text += f"- {e['name']} ({e['type']} in {e['file_path']}): {e['description'][:150]}{score_hint}\n"
        
        prompt = SEMANTIC_MAPPER_PROMPT.format(
            concepts=concepts_text,
            code_elements=elements_text
        )
        
        try:
            result = await self.llm.generate_structured(
                prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "mappings": {"type": "array"},
                        "unmapped_concepts": {"type": "array"},
                        "unmapped_code": {"type": "array"}
                    }
                },
                system_instruction=SEMANTIC_MAPPER_SYSTEM_PROMPT
            )
            
            mappings = result.get("mappings", [])
            
            # Enhance mappings with pre-computed scores
            for mapping in mappings:
                concept_name = mapping.get("concept_name", "")
                code_name = mapping.get("code_element", "")
                key = (concept_name, code_name)
                
                # Add match signals if not present
                if "match_signals" not in mapping:
                    mapping["match_signals"] = {}
                
                mapping["match_signals"]["lexical"] = lexical_scores.get(key, 0)
                mapping["match_signals"]["semantic"] = semantic_scores.get(key, 0)
                mapping["match_signals"]["documentary"] = documentary_scores.get(key, 0)
                
                # Ensure confidence is reasonable
                if "confidence" not in mapping:
                    signals = mapping["match_signals"]
                    mapping["confidence"] = max(
                        signals["lexical"],
                        signals["semantic"],
                        signals["documentary"]
                    )
            
            # Sort by confidence
            mappings.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            
            self.log_info(f"Generated {len(mappings)} mappings")
            
            return mappings
            
        except Exception as e:
            self.log_error(f"LLM mapping failed: {e}")
            return self._fallback_mapping(
                concepts, code_elements,
                lexical_scores, semantic_scores, documentary_scores
            )
    
    def _fallback_mapping(
        self,
        concepts: List[Dict[str, Any]],
        code_elements: List[Dict[str, Any]],
        lexical_scores: Dict[Tuple[str, str], float],
        semantic_scores: Dict[Tuple[str, str], float],
        documentary_scores: Dict[Tuple[str, str], float]
    ) -> List[Dict[str, Any]]:
        """Fallback mapping using only pre-computed scores with adaptive weighting."""
        mappings = []

        # Check if we have semantic scores available
        has_semantic = len(semantic_scores) > 0

        for concept in concepts:
            best_match = None
            best_score = 0
            best_signals = {}

            for element in code_elements:
                key = (concept["name"], element["name"])

                lex = lexical_scores.get(key, 0)
                sem = semantic_scores.get(key, 0)
                doc = documentary_scores.get(key, 0)

                # Adaptive weighting based on signal availability
                if has_semantic and sem > 0:
                    # When semantic scores are available, weight them higher
                    # Semantic similarity is most reliable for concept matching
                    combined = (lex * 0.25 + sem * 0.50 + doc * 0.25)
                else:
                    # Without semantic, rely more on lexical and documentary
                    combined = (lex * 0.5 + doc * 0.5)

                # Boost if multiple signals agree
                signals_agreeing = sum(1 for s in [lex, sem, doc] if s > 0.3)
                if signals_agreeing >= 2:
                    combined *= 1.1  # 10% boost for agreement

                if combined > best_score:
                    best_score = combined
                    best_match = element
                    best_signals = {"lexical": lex, "semantic": sem, "documentary": doc}

            if best_match and best_score > 0.2:
                key = (concept["name"], best_match["name"])

                # Determine primary matching reason
                signals = best_signals
                primary_signal = max(signals.items(), key=lambda x: x[1])[0] if signals else "unknown"

                mappings.append({
                    "concept_name": concept["name"],
                    "concept_description": concept["description"],
                    "code_element": best_match["name"],
                    "code_file": best_match["file_path"],
                    "confidence": min(1.0, best_score),
                    "match_signals": signals,
                    "evidence": [f"Primary signal: {primary_signal} ({signals.get(primary_signal, 0):.2f})"],
                    "reasoning": f"Matched via {primary_signal} similarity (score: {best_score:.2f})"
                })

        return mappings
    
    async def _update_knowledge_graph(
        self,
        mappings: List[Dict[str, Any]],
        kg: KnowledgeGraph
    ):
        """Add mappings to knowledge graph."""
        # Find concept and code nodes
        concept_nodes = kg.get_nodes_by_type(NodeType.CONCEPT)
        algo_nodes = kg.get_nodes_by_type(NodeType.ALGORITHM)
        function_nodes = kg.get_nodes_by_type(NodeType.FUNCTION)
        class_nodes = kg.get_nodes_by_type(NodeType.CLASS)
        
        concept_map = {n.name.lower(): n.id for n in concept_nodes + algo_nodes}
        code_map = {n.name.lower(): n.id for n in function_nodes + class_nodes}
        
        for mapping in mappings:
            concept_name = mapping.get("concept_name", "").lower()
            code_name = mapping.get("code_element", "").lower()
            
            concept_id = concept_map.get(concept_name)
            code_id = code_map.get(code_name)
            
            if concept_id and code_id:
                try:
                    create_mapping_node(
                        kg,
                        concept_id,
                        code_id,
                        confidence=mapping.get("confidence", 0),
                        evidence=mapping.get("evidence", [])
                    )
                except Exception as e:
                    self.log_warning(f"Failed to create mapping node: {e}")
        
        self.log_info(f"Knowledge graph now has {kg.get_statistics()['total_nodes']} nodes")