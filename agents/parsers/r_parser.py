"""
R Parser - Regex-based R code parser.
"""

import re
from typing import Dict, List
from pathlib import Path

from . import LanguageParser, CodeElement


class RParser(LanguageParser):
    """Regex-based R parser for extracting code elements."""

    @property
    def extensions(self) -> List[str]:
        return [".r", ".R"]

    @property
    def language(self) -> str:
        return "r"

    def parse_file(self, file_path: Path) -> Dict[str, List[CodeElement]]:
        """Parse an R file using regex patterns."""
        result = {
            "classes": [],
            "functions": [],
            "imports": [],
            "constants": []
        }

        content = self._safe_read(file_path)
        if not content:
            return result

        lines = content.split('\n')

        # Parse functions (name <- function(...))
        func_pattern = r'(\w+)\s*<-\s*function\s*\(([^)]*)\)'
        for i, line in enumerate(lines):
            match = re.match(func_pattern, line.strip())
            if match:
                name = match.group(1)
                args_str = match.group(2)
                args = self._parse_r_args(args_str)

                # Get roxygen docstring
                docstring = self._get_roxygen_docstring(lines, i)

                result["functions"].append(CodeElement(
                    name=name,
                    element_type="function",
                    file_path=str(file_path),
                    line_number=i + 1,
                    docstring=docstring,
                    signature=f"{name} <- function({args_str})",
                    args=args,
                    language="r"
                ))

        # Alternative function syntax (name = function(...))
        func_pattern2 = r'(\w+)\s*=\s*function\s*\(([^)]*)\)'
        for i, line in enumerate(lines):
            match = re.match(func_pattern2, line.strip())
            if match:
                name = match.group(1)
                # Skip if already found
                if any(f.name == name for f in result["functions"]):
                    continue

                args_str = match.group(2)
                args = self._parse_r_args(args_str)
                docstring = self._get_roxygen_docstring(lines, i)

                result["functions"].append(CodeElement(
                    name=name,
                    element_type="function",
                    file_path=str(file_path),
                    line_number=i + 1,
                    docstring=docstring,
                    signature=f"{name} = function({args_str})",
                    args=args,
                    language="r"
                ))

        # Parse S4 classes
        s4_class_pattern = r'setClass\s*\(\s*["\'](\w+)["\']'
        for i, line in enumerate(lines):
            match = re.search(s4_class_pattern, line)
            if match:
                name = match.group(1)
                result["classes"].append(CodeElement(
                    name=name,
                    element_type="S4_class",
                    file_path=str(file_path),
                    line_number=i + 1,
                    language="r"
                ))

        # Parse R6 classes
        r6_pattern = r'(\w+)\s*<-\s*R6Class\s*\('
        for i, line in enumerate(lines):
            match = re.match(r6_pattern, line.strip())
            if match:
                name = match.group(1)
                result["classes"].append(CodeElement(
                    name=name,
                    element_type="R6_class",
                    file_path=str(file_path),
                    line_number=i + 1,
                    language="r"
                ))

        # Parse Reference Classes
        refclass_pattern = r'setRefClass\s*\(\s*["\'](\w+)["\']'
        for i, line in enumerate(lines):
            match = re.search(refclass_pattern, line)
            if match:
                name = match.group(1)
                result["classes"].append(CodeElement(
                    name=name,
                    element_type="reference_class",
                    file_path=str(file_path),
                    line_number=i + 1,
                    language="r"
                ))

        # Parse library/require statements
        lib_pattern = r'(?:library|require)\s*\(\s*["\']?(\w+)["\']?\s*\)'
        for i, line in enumerate(lines):
            match = re.search(lib_pattern, line)
            if match:
                result["imports"].append({
                    "module": match.group(1),
                    "file_path": str(file_path),
                    "line_number": i + 1
                })

        return result

    def _parse_r_args(self, args_str: str) -> List[str]:
        """Parse R function arguments."""
        if not args_str.strip():
            return []

        args = []
        for arg in args_str.split(','):
            arg = arg.strip()
            # Remove default values
            if '=' in arg:
                arg = arg.split('=')[0].strip()
            if arg:
                args.append(arg)
        return args

    def _get_roxygen_docstring(self, lines: List[str], func_line: int) -> str:
        """Extract roxygen2 documentation."""
        docstring_lines = []
        i = func_line - 1

        while i >= 0:
            line = lines[i].strip()
            if line.startswith("#'"):
                docstring_lines.append(line[2:].strip())
            elif line.startswith("#"):
                pass  # Skip regular comments
            elif line:
                break
            i -= 1

        return "\n".join(reversed(docstring_lines))[:500]
