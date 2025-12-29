"""
Multi-Language Parsers for Code Analysis

This module provides language-specific parsers for extracting
code elements (classes, functions, types) from various programming languages.

Supported languages:
- Python (AST-based)
- Julia (regex-based)
- R (regex-based)
- JavaScript/TypeScript (regex-based)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field, ConfigDict


class CodeElement(BaseModel):
    """Represents a parsed code element from any language."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Element name")
    element_type: str = Field(..., description="Type: class, function, struct, type, etc.")
    file_path: str = Field(..., description="Path to source file")
    line_number: int = Field(0, description="Line number where defined")
    docstring: str = Field("", description="Documentation string")
    signature: str = Field("", description="Function/method signature")
    args: List[str] = Field(default_factory=list, description="Function arguments")
    return_type: str = Field("", description="Return type annotation")
    decorators: List[str] = Field(default_factory=list, description="Decorators applied")
    bases: List[str] = Field(default_factory=list, description="Base classes (for classes)")
    methods: List[str] = Field(default_factory=list, description="Methods (for classes)")
    language: str = Field("python", description="Programming language")


class LanguageParser(ABC):
    """Abstract base class for language-specific parsers."""

    @property
    @abstractmethod
    def extensions(self) -> List[str]:
        """File extensions this parser handles."""
        pass

    @property
    @abstractmethod
    def language(self) -> str:
        """Language name."""
        pass

    @abstractmethod
    def parse_file(self, file_path: Path) -> Dict[str, List[CodeElement]]:
        """
        Parse a file and extract code elements.

        Returns:
            Dict with keys: "classes", "functions", "imports", "constants"
        """
        pass

    def _safe_read(self, file_path: Path) -> str:
        """Safely read file content with encoding fallback."""
        try:
            return file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return ""


# Import parsers for convenience
from .python_parser import PythonParser
from .julia_parser import JuliaParser
from .r_parser import RParser
from .javascript_parser import JavaScriptParser
from .parser_factory import ParserFactory

__all__ = [
    'CodeElement',
    'LanguageParser',
    'PythonParser',
    'JuliaParser',
    'RParser',
    'JavaScriptParser',
    'ParserFactory',
]
