"""
Paper Parser Agent - Analyzes scientific papers and extracts structured information.
"""

import re
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

import httpx
from pydantic import ValidationError

from .base_agent import BaseAgent
from .protocols import PaperParserOutput
from core.llm_client import LLMClient
from core.schema_utils import generate_llm_schema
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
    
    async def process(
        self,
        *,
        paper_source: str,
        knowledge_graph: KnowledgeGraph = None
    ) -> PaperParserOutput:
        """
        Parse a scientific paper and extract structured information.

        Args:
            paper_source: arXiv ID, URL, or local file path (REQUIRED)
            knowledge_graph: Optional knowledge graph to populate

        Returns:
            PaperParserOutput with extracted paper information
        """
        if not paper_source:
            raise ValueError("paper_source is required")
        if knowledge_graph is None:
            knowledge_graph = KnowledgeGraph()
        return await self.parse(paper_source, knowledge_graph)

    async def parse(
        self,
        paper_source: str,
        kg: KnowledgeGraph
    ) -> PaperParserOutput:
        """
        Parse a scientific paper and extract structured information.

        Args:
            paper_source: arXiv ID, URL, or local file path
            kg: Knowledge graph to populate

        Returns:
            PaperParserOutput with extracted paper information
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
            except OSError as e:
                self.log_debug(f"Failed to clean up temp file {pdf_path}: {e}")
        
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
        """
        Extract text from PDF using available backends with retry and quality validation.

        Tries all backends with retries for each, and selects the one with best quality score.
        """
        results = []  # Store results from each backend
        backend_errors = {}  # Track errors per backend

        for backend in self._pdf_backends:
            result = await self._try_backend_with_retry(pdf_path, backend, max_retries=2)
            if result:
                results.append(result)
            else:
                backend_errors[backend] = "Failed after retries"

        if not results:
            raise AgentError(create_error(
                ErrorCategory.PARSING,
                "All PDF extraction backends failed",
                context={
                    "tried_backends": self._pdf_backends,
                    "pdf_path": pdf_path,
                    "errors": backend_errors
                },
                suggestion="The PDF may be image-based (scanned), corrupted, or password-protected."
            ))

        # Select best result based on quality score
        best_result = max(results, key=lambda r: r["quality"])

        if best_result["quality"] < 0.3:
            self.log_warning(f"Low quality extraction ({best_result['quality']:.2f}), results may be unreliable")

        self.log_info(f"Selected {best_result['backend']} extraction (quality={best_result['quality']:.2f})")
        return best_result["text"]

    async def _try_backend_with_retry(
        self,
        pdf_path: str,
        backend: str,
        max_retries: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        Try a PDF backend with retry logic.

        Args:
            pdf_path: Path to the PDF file
            backend: Backend name (pymupdf, pdfplumber, pypdf)
            max_retries: Maximum number of retry attempts

        Returns:
            Dict with extraction result or None if all attempts failed
        """
        import asyncio

        last_error = None

        for attempt in range(max_retries + 1):
            try:
                if backend == "pymupdf":
                    text = self._extract_with_pymupdf(pdf_path)
                elif backend == "pdfplumber":
                    text = self._extract_with_pdfplumber(pdf_path)
                elif backend == "pypdf":
                    text = self._extract_with_pypdf(pdf_path)
                else:
                    return None

                if text:
                    quality_score = self._assess_text_quality(text)
                    self.log_info(
                        f"Backend {backend}: {len(text)} chars, quality={quality_score:.2f}"
                        f"{' (retry ' + str(attempt) + ')' if attempt > 0 else ''}"
                    )
                    return {
                        "backend": backend,
                        "text": text,
                        "quality": quality_score,
                        "length": len(text),
                        "attempts": attempt + 1
                    }

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = 0.5 * (attempt + 1)  # Exponential backoff
                    self.log_warning(
                        f"Backend {backend} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.log_warning(
                        f"Backend {backend} failed after {max_retries + 1} attempts: {last_error}"
                    )

        return None

    def _assess_text_quality(self, text: str) -> float:
        """
        Assess quality of extracted text (0-1 score).

        Checks for:
        - Minimum length
        - Special character ratio
        - Words per line
        - Section markers
        - Text repetition
        """
        if not text or len(text) < 100:
            return 0.0

        score = 1.0

        # 1. Too many special characters (indicates OCR issues)
        special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_ratio > 0.3:
            score -= 0.3

        # 2. Too few words per line (indicates column merge issues)
        lines = [l for l in text.split('\n') if l.strip()]
        if lines:
            avg_words_per_line = sum(len(l.split()) for l in lines) / len(lines)
            if avg_words_per_line < 3:
                score -= 0.2

        # 3. Missing section markers (Abstract, Introduction, etc.)
        section_markers = ["abstract", "introduction", "method", "result", "conclusion", "reference"]
        found_markers = sum(1 for m in section_markers if m in text.lower())
        if found_markers < 2:
            score -= 0.2

        # 4. Excessive repetition (indicates loop extraction bug)
        words = text.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                score -= 0.3

        return max(0.0, score)
    
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
    
    def _smart_truncate(self, text: str, max_chars: int = 50000) -> str:
        """
        Intelligently truncate paper text while preserving critical sections.

        Priority order:
        1. Abstract (always keep full)
        2. Methodology/Methods (keep full)
        3. Introduction (keep first half)
        4. Results/Experiments (keep key portions)
        5. Conclusion (keep full)
        6. Related Work (truncate heavily)
        7. References (remove entirely)
        """
        if len(text) <= max_chars:
            return text

        # Split into sections
        sections = self._identify_sections(text)

        # Define section priorities and target sizes
        priorities = {
            "abstract": (1.0, 1.0),      # (priority, keep_ratio)
            "introduction": (0.8, 0.6),
            "methodology": (1.0, 1.0),
            "methods": (1.0, 1.0),
            "experiments": (0.7, 0.5),
            "results": (0.7, 0.5),
            "discussion": (0.5, 0.3),
            "conclusion": (0.9, 1.0),
            "related_work": (0.3, 0.2),
            "references": (0.0, 0.0),    # Remove entirely
            "full_text": (0.5, 0.5),     # Fallback
        }

        truncated_sections = []
        remaining_chars = max_chars

        # Process sections by priority (highest first)
        sorted_sections = sorted(
            [(name, priorities.get(name, (0.5, 0.5))) for name in sections.keys()],
            key=lambda x: -x[1][0]
        )

        for section_name, (_, keep_ratio) in sorted_sections:
            if section_name not in sections:
                continue

            section_text = sections[section_name]
            target_length = int(len(section_text) * keep_ratio)
            target_length = min(target_length, remaining_chars)

            if target_length > 100:
                truncated_text = self._truncate_section(section_text, target_length)
                truncated_sections.append(f"\n\n## {section_name.upper()}\n{truncated_text}")
                remaining_chars -= len(truncated_text)

            if remaining_chars <= 0:
                break

        result = "".join(truncated_sections)

        # If still too long, do final truncation
        if len(result) > max_chars:
            half = max_chars // 2
            result = result[:half] + "\n\n[...TRUNCATED FOR LENGTH...]\n\n" + result[-half:]

        return result

    def _identify_sections(self, text: str) -> Dict[str, str]:
        """Identify and extract paper sections using regex patterns."""
        sections = {}

        # Common section patterns
        patterns = {
            "abstract": r"(?i)(?:^|\n)\s*abstract\s*\n(.*?)(?=\n\s*(?:1\.?\s*)?introduction|\n\s*keywords|\Z)",
            "introduction": r"(?i)(?:^|\n)\s*(?:1\.?\s*)?introduction\s*\n(.*?)(?=\n\s*(?:2\.?\s*)?(?:related|background|method|approach)|\Z)",
            "methodology": r"(?i)(?:^|\n)\s*(?:\d\.?\s*)?(?:method(?:ology)?|approach|model)\s*\n(.*?)(?=\n\s*(?:\d\.?\s*)?(?:experiment|result|evaluation)|\Z)",
            "experiments": r"(?i)(?:^|\n)\s*(?:\d\.?\s*)?(?:experiment|evaluation|result)s?\s*\n(.*?)(?=\n\s*(?:\d\.?\s*)?(?:discussion|conclusion|related)|\Z)",
            "conclusion": r"(?i)(?:^|\n)\s*(?:\d\.?\s*)?conclusion\s*\n(.*?)(?=\n\s*(?:acknowledge|reference|appendix)|\Z)",
            "related_work": r"(?i)(?:^|\n)\s*(?:\d\.?\s*)?related\s+work\s*\n(.*?)(?=\n\s*(?:\d\.?\s*)?(?:method|approach|conclusion)|\Z)",
        }

        for section_name, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()

        # If no sections found, treat entire text as one section
        if not sections:
            sections["full_text"] = text

        return sections

    def _truncate_section(self, text: str, max_length: int) -> str:
        """Truncate a section intelligently at sentence boundaries."""
        if len(text) <= max_length:
            return text

        # Try to truncate at sentence boundary
        sentences = re.split(r'(?<=[.!?])\s+', text)

        result = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) > max_length:
                break
            result.append(sentence)
            current_length += len(sentence) + 1

        if result:
            return " ".join(result) + "..."
        else:
            return text[:max_length] + "..."

    async def _analyze_paper(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> PaperParserOutput:
        """Use LLM to analyze paper and extract structured information."""
        # Use smart truncation instead of simple split
        text = self._smart_truncate(text, max_chars=50000)

        prompt = PAPER_PARSER_EXTRACTION_PROMPT.format(paper_text=text)

        # Generate schema appropriate for the LLM provider
        provider_name = self.llm.config.provider.value
        schema = generate_llm_schema(PaperParserOutput, provider_name)

        try:
            result_dict = await self.llm.generate_structured(
                prompt,
                schema=schema,
                system_instruction=PAPER_PARSER_SYSTEM_PROMPT
            )

            # Merge with metadata (prefer LLM results, fallback to metadata)
            if metadata.get("title") and not result_dict.get("title"):
                result_dict["title"] = metadata["title"]
            if metadata.get("authors") and not result_dict.get("authors"):
                result_dict["authors"] = metadata["authors"]
            if metadata.get("abstract") and not result_dict.get("abstract"):
                result_dict["abstract"] = metadata["abstract"]

            # Build source metadata
            result_dict["source_metadata"] = {
                "source_type": metadata.get("source_type", ""),
                "arxiv_id": metadata.get("arxiv_id"),
                "url": metadata.get("url"),
                "file_path": metadata.get("file_path"),
                "extraction_date": metadata.get("extraction_date")
            }

            # Validate with Pydantic - raises ValidationError on failure
            try:
                output = PaperParserOutput.model_validate(result_dict)
            except ValidationError as ve:
                raise AgentError(create_error(
                    ErrorCategory.VALIDATION,
                    f"LLM output validation failed: {ve}",
                    original_error=ve,
                    recoverable=False,
                    suggestion="LLM returned data that doesn't match expected schema"
                ))

            return output

        except AgentError:
            raise  # Re-raise AgentError as-is
        except Exception as e:
            self.log_error(f"LLM analysis failed: {e}")
            raise AgentError(create_error(
                ErrorCategory.LLM,
                f"Paper analysis failed: {e}",
                original_error=e,
                recoverable=False
            ))
    
    async def _populate_knowledge_graph(
        self,
        paper_data: PaperParserOutput,
        kg: KnowledgeGraph
    ):
        """Add paper information to knowledge graph."""
        # Create paper node
        paper_id = create_paper_node(
            kg,
            paper_data.title,
            abstract=paper_data.abstract,
            authors=paper_data.authors,
            source=paper_data.source_metadata.model_dump() if paper_data.source_metadata else {}
        )

        # Create concept nodes
        for concept in paper_data.key_concepts:
            concept_id = create_concept_node(
                kg,
                concept.name,
                description=concept.description,
                importance=concept.importance,
                related_sections=concept.related_sections
            )
            kg.add_edge(paper_id, concept_id, EdgeType.CONTAINS)

        # Create algorithm nodes
        for algo in paper_data.algorithms:
            algo_id = kg.add_node(
                type=NodeType.ALGORITHM,
                name=algo.name,
                description=algo.description,
                metadata={
                    "complexity": algo.complexity,
                    "pseudocode": algo.pseudocode
                }
            )
            kg.add_edge(paper_id, algo_id, EdgeType.CONTAINS)

        paper_data._kg_paper_id = paper_id
        self.log_info(f"Added {kg.get_statistics()['total_nodes']} nodes to knowledge graph")