"""
Paper Parser Agent - Analyzes scientific papers and extracts structured information.
"""

import re
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

import httpx

from .base_agent import BaseAgent
from core.llm_client import LLMClient
from core.knowledge_graph import (
    KnowledgeGraph, NodeType, EdgeType,
    create_paper_node, create_concept_node
)
from core.agent_prompts import (
    PAPER_PARSER_SYSTEM_PROMPT,
    PAPER_PARSER_EXTRACTION_PROMPT
)
from core.error_handling import (
    logger, LogCategory, ErrorCategory, create_error, AgentError, with_retry
)


class PaperParserAgent(BaseAgent):
    """
    Analyzes scientific papers from various sources.
    
    Supports:
    - arXiv IDs and URLs
    - Direct PDF URLs
    - Local PDF files
    
    Uses multiple PDF parsing backends for robustness.
    """
    
    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client, name="PaperParser")
        self._pdf_backends = self._init_pdf_backends()
    
    def _init_pdf_backends(self) -> List[str]:
        """Initialize available PDF parsing backends."""
        backends = []
        
        try:
            import fitz  # PyMuPDF
            backends.append("pymupdf")
        except ImportError:
            pass
        
        try:
            import pdfplumber
            backends.append("pdfplumber")
        except ImportError:
            pass
        
        try:
            import pypdf
            backends.append("pypdf")
        except ImportError:
            pass
        
        if not backends:
            self.log_warning("No PDF parsing backends available. Install PyMuPDF, pdfplumber, or pypdf.")
        else:
            self.log_info(f"Available PDF backends: {backends}")
        
        return backends
    
    async def process(self, paper_source: str = None, paper_url: str = None, knowledge_graph: KnowledgeGraph = None, kg: KnowledgeGraph = None) -> Dict[str, Any]:
        """Main processing method."""
        # Support both parameter names
        source = paper_source or paper_url
        graph = knowledge_graph or kg
        if not graph:
            graph = KnowledgeGraph()
        return await self.parse(source, graph)
    
    async def parse(
        self,
        paper_source: str,
        kg: KnowledgeGraph
    ) -> Dict[str, Any]:
        """
        Parse a scientific paper and extract structured information.
        
        Args:
            paper_source: arXiv ID, URL, or local file path
            kg: Knowledge graph to populate
            
        Returns:
            Dictionary with extracted paper information
        """
        self.log_info(f"Parsing paper: {paper_source}")
        
        # Step 1: Download/load PDF
        pdf_path, metadata = await self._timed_operation(
            "download_paper",
            self._get_pdf(paper_source)
        )
        
        # Step 2: Extract text
        text = await self._timed_operation(
            "extract_text",
            self._extract_text(pdf_path)
        )
        
        if not text or len(text.strip()) < 100:
            raise AgentError(create_error(
                ErrorCategory.PARSING,
                "Failed to extract meaningful text from PDF",
                suggestion="The PDF might be image-based or corrupted"
            ))
        
        self.log_info(f"Extracted {len(text)} characters from PDF")
        
        # Step 3: Use LLM to analyze paper
        paper_data = await self._timed_operation(
            "llm_analysis",
            self._analyze_paper(text, metadata)
        )
        
        # Step 4: Populate knowledge graph
        await self._populate_knowledge_graph(paper_data, kg)
        
        # Clean up temp file
        if pdf_path and Path(pdf_path).exists() and "temp" in str(pdf_path):
            try:
                Path(pdf_path).unlink()
            except:
                pass
        
        return paper_data
    
    @with_retry(ErrorCategory.NETWORK, "PDF download")
    async def _get_pdf(self, source: str) -> tuple[str, Dict[str, Any]]:
        """
        Get PDF from various sources.
        
        Returns:
            Tuple of (pdf_path, metadata)
        """
        metadata = {"source": source, "source_type": "unknown"}
        
        # Check if it's a local file
        if Path(source).exists():
            metadata["source_type"] = "local"
            return str(source), metadata
        
        # Check if it's an arXiv ID
        arxiv_pattern = r'^(\d{4}\.\d{4,5})(v\d+)?$'
        if re.match(arxiv_pattern, source):
            return await self._download_arxiv(source, metadata)
        
        # Check if it's an arXiv URL
        if "arxiv.org" in source:
            arxiv_id = self._extract_arxiv_id(source)
            if arxiv_id:
                return await self._download_arxiv(arxiv_id, metadata)
        
        # Assume it's a direct PDF URL
        return await self._download_pdf_url(source, metadata)
    
    async def _download_arxiv(
        self,
        arxiv_id: str,
        metadata: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """Download paper from arXiv."""
        self.log_info(f"Downloading from arXiv: {arxiv_id}")
        metadata["source_type"] = "arxiv"
        metadata["arxiv_id"] = arxiv_id
        
        # Try to get metadata from arXiv API
        try:
            import arxiv
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results())
            
            metadata["title"] = paper.title
            metadata["authors"] = [a.name for a in paper.authors]
            metadata["abstract"] = paper.summary
            metadata["published"] = paper.published.isoformat() if paper.published else None
            metadata["categories"] = paper.categories
            
            # Download PDF
            temp_path = tempfile.mktemp(suffix=".pdf")
            paper.download_pdf(filename=temp_path)
            
            return temp_path, metadata
            
        except ImportError:
            # Fall back to direct download
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            return await self._download_pdf_url(pdf_url, metadata)
        except Exception as e:
            self.log_warning(f"arXiv API failed: {e}, falling back to direct download")
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            return await self._download_pdf_url(pdf_url, metadata)
    
    async def _download_pdf_url(
        self,
        url: str,
        metadata: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """Download PDF from URL."""
        self.log_info(f"Downloading PDF from URL: {url}")
        metadata["source_type"] = "url"
        metadata["url"] = url
        
        async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type.lower() and not url.endswith(".pdf"):
                self.log_warning(f"URL may not be a PDF (content-type: {content_type})")
            
            # Save to temp file
            temp_path = tempfile.mktemp(suffix=".pdf")
            Path(temp_path).write_bytes(response.content)
            
            return temp_path, metadata
    
    def _extract_arxiv_id(self, url: str) -> Optional[str]:
        """Extract arXiv ID from URL."""
        patterns = [
            r'arxiv\.org/abs/(\d{4}\.\d{4,5})',
            r'arxiv\.org/pdf/(\d{4}\.\d{4,5})',
            r'ar5iv\.labs\.arxiv\.org/html/(\d{4}\.\d{4,5})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    async def _extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF using available backends."""
        text = ""
        
        for backend in self._pdf_backends:
            try:
                if backend == "pymupdf":
                    text = self._extract_with_pymupdf(pdf_path)
                elif backend == "pdfplumber":
                    text = self._extract_with_pdfplumber(pdf_path)
                elif backend == "pypdf":
                    text = self._extract_with_pypdf(pdf_path)
                
                if text and len(text.strip()) > 100:
                    self.log_info(f"Extracted text using {backend}")
                    return text
                    
            except Exception as e:
                self.log_warning(f"Backend {backend} failed: {e}")
                continue
        
        return text
    
    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF."""
        import fitz
        
        text_parts = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text_parts.append(page.get_text())
        
        return "\n\n".join(text_parts)
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber."""
        import pdfplumber
        
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    def _extract_with_pypdf(self, pdf_path: str) -> str:
        """Extract text using pypdf."""
        from pypdf import PdfReader
        
        text_parts = []
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    async def _analyze_paper(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM to analyze paper and extract structured information."""
        # Truncate text if too long (keep first and last parts)
        max_chars = 50000
        if len(text) > max_chars:
            half = max_chars // 2
            text = text[:half] + "\n\n[...TRUNCATED...]\n\n" + text[-half:]
        
        prompt = PAPER_PARSER_EXTRACTION_PROMPT.format(paper_text=text)
        
        try:
            result = await self.llm.generate_structured(
                prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "authors": {"type": "array", "items": {"type": "string"}},
                        "abstract": {"type": "string"},
                        "key_concepts": {"type": "array"},
                        "algorithms": {"type": "array"},
                        "methodology": {"type": "object"},
                        "reproducibility": {"type": "object"},
                        "expected_implementations": {"type": "array"}
                    }
                },
                system_instruction=PAPER_PARSER_SYSTEM_PROMPT
            )
            
            # Merge with metadata
            if metadata.get("title"):
                result["title"] = result.get("title") or metadata["title"]
            if metadata.get("authors"):
                result["authors"] = result.get("authors") or metadata["authors"]
            if metadata.get("abstract"):
                result["abstract"] = result.get("abstract") or metadata["abstract"]
            
            result["source_metadata"] = metadata
            
            return result
            
        except Exception as e:
            self.log_error(f"LLM analysis failed: {e}")
            # Return minimal structure with metadata
            return {
                "title": metadata.get("title", "Unknown"),
                "authors": metadata.get("authors", []),
                "abstract": metadata.get("abstract", ""),
                "key_concepts": [],
                "algorithms": [],
                "methodology": {},
                "reproducibility": {},
                "expected_implementations": [],
                "source_metadata": metadata,
                "analysis_error": str(e)
            }
    
    async def _populate_knowledge_graph(
        self,
        paper_data: Dict[str, Any],
        kg: KnowledgeGraph
    ):
        """Add paper information to knowledge graph."""
        # Create paper node
        paper_id = create_paper_node(
            kg,
            paper_data.get("title", "Unknown Paper"),
            abstract=paper_data.get("abstract", ""),
            authors=paper_data.get("authors", []),
            source=paper_data.get("source_metadata", {})
        )
        
        # Create concept nodes
        for concept in paper_data.get("key_concepts", []):
            concept_id = create_concept_node(
                kg,
                concept.get("name", ""),
                description=concept.get("description", ""),
                importance=concept.get("importance", "medium"),
                related_sections=concept.get("related_sections", [])
            )
            kg.add_edge(paper_id, concept_id, EdgeType.CONTAINS)
        
        # Create algorithm nodes
        for algo in paper_data.get("algorithms", []):
            algo_id = kg.add_node(
                type=NodeType.ALGORITHM,
                name=algo.get("name", ""),
                description=algo.get("description", ""),
                metadata={
                    "complexity": algo.get("complexity"),
                    "pseudocode": algo.get("pseudocode_summary")
                }
            )
            kg.add_edge(paper_id, algo_id, EdgeType.CONTAINS)
        
        paper_data["_kg_paper_id"] = paper_id
        self.log_info(f"Added {kg.get_statistics()['total_nodes']} nodes to knowledge graph")