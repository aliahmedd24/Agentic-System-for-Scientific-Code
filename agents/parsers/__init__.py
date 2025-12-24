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
from dataclasses import dataclass, field


@dataclass
class CodeElement:
    """Represents a parsed code element from any language."""
    name: str
    element_type: str  # "class", "function", "struct", "type", etc.
    file_path: str
    line_number: int
    docstring: str = ""
    signature: str = ""
    args: List[str] = field(default_factory=list)
    return_type: str = ""
    decorators: List[str] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)  # For classes
    methods: List[str] = field(default_factory=list)  # For classes
    language: str = "python"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "type": self.element_type,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "docstring": self.docstring,
            "signature": self.signature,
            "args": self.args,
            "return_type": self.return_type,
            "decorators": self.decorators,
            "bases": self.bases,
            "methods": self.methods,
            "language": self.language
        }


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
