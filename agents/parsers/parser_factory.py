"""
Parser Factory - Creates appropriate parsers for different languages.
"""

from typing import Dict, List, Optional
from pathlib import Path

from . import LanguageParser, CodeElement
from .python_parser import PythonParser
from .julia_parser import JuliaParser
from .r_parser import RParser
from .javascript_parser import JavaScriptParser


class ParserFactory:
    """
    Factory for creating and managing language-specific parsers.

    Provides a unified interface for parsing code files regardless of language.
    """

    def __init__(self):
        """Initialize available parsers."""
        self._parsers: Dict[str, LanguageParser] = {}
        self._extension_map: Dict[str, str] = {}

        # Register all available parsers
        self._register_parser(PythonParser())
        self._register_parser(JuliaParser())
        self._register_parser(RParser())
        self._register_parser(JavaScriptParser())

    def _register_parser(self, parser: LanguageParser) -> None:
        """Register a parser and its extensions."""
        self._parsers[parser.language] = parser
        for ext in parser.extensions:
            self._extension_map[ext.lower()] = parser.language

    @property
    def supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return list(self._parsers.keys())

    @property
    def supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return list(self._extension_map.keys())

    def get_parser(self, language: str) -> Optional[LanguageParser]:
        """Get parser for a specific language."""
        return self._parsers.get(language.lower())

    def get_parser_for_file(self, file_path: Path) -> Optional[LanguageParser]:
        """Get appropriate parser for a file based on extension."""
        ext = file_path.suffix.lower()
        language = self._extension_map.get(ext)
        if language:
            return self._parsers.get(language)
        return None

    def detect_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language from file extension."""
        ext = file_path.suffix.lower()
        return self._extension_map.get(ext)

    def parse_file(self, file_path: Path) -> Dict[str, List[CodeElement]]:
        """
        Parse a file using the appropriate parser.

        Returns:
            Dict with keys: "classes", "functions", "imports", "constants"
            Empty dict if no parser is available for the file type.
        """
        parser = self.get_parser_for_file(file_path)
        if parser:
            return parser.parse_file(file_path)
        return {"classes": [], "functions": [], "imports": [], "constants": []}

    def parse_directory(
        self,
        directory: Path,
        max_files: int = 100,
        skip_dirs: Optional[set] = None
    ) -> Dict[str, List[CodeElement]]:
        """
        Parse all supported files in a directory.

        Args:
            directory: Path to directory to parse
            max_files: Maximum number of files to parse
            skip_dirs: Set of directory names to skip

        Returns:
            Aggregated dict with all code elements
        """
        if skip_dirs is None:
            skip_dirs = {
                '.git', '.github', '__pycache__', 'node_modules',
                '.venv', 'venv', 'env', 'build', 'dist', '.tox',
                '.pytest_cache', '.mypy_cache', 'htmlcov',
                '.eggs', '.cache'
            }

        result = {
            "classes": [],
            "functions": [],
            "imports": [],
            "constants": []
        }

        files_parsed = 0

        for file_path in directory.rglob('*'):
            if files_parsed >= max_files:
                break

            # Skip directories in skip list
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue

            if not file_path.is_file():
                continue

            # Check if we have a parser for this file
            if self.get_parser_for_file(file_path):
                try:
                    file_result = self.parse_file(file_path)
                    result["classes"].extend(file_result.get("classes", []))
                    result["functions"].extend(file_result.get("functions", []))
                    result["imports"].extend(file_result.get("imports", []))
                    result["constants"].extend(file_result.get("constants", []))
                    files_parsed += 1
                except Exception:
                    # Skip files that fail to parse
                    pass

        return result

    def get_language_stats(self, directory: Path, skip_dirs: Optional[set] = None) -> Dict[str, int]:
        """
        Get statistics about language usage in a directory.

        Returns:
            Dict mapping language names to file counts
        """
        if skip_dirs is None:
            skip_dirs = {
                '.git', '.github', '__pycache__', 'node_modules',
                '.venv', 'venv', 'env', 'build', 'dist'
            }

        stats = {lang: 0 for lang in self._parsers.keys()}

        for file_path in directory.rglob('*'):
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue

            if not file_path.is_file():
                continue

            language = self.detect_language(file_path)
            if language:
                stats[language] += 1

        return stats

    def detect_primary_language(self, directory: Path) -> str:
        """
        Detect the primary programming language in a directory.

        Returns:
            Language name (defaults to "python" if no files found)
        """
        stats = self.get_language_stats(directory)

        if not stats or all(v == 0 for v in stats.values()):
            return "python"

        return max(stats, key=stats.get)
