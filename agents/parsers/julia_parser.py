"""
Julia Parser - Regex-based Julia code parser.
"""

import re
from typing import Dict, List
from pathlib import Path

from . import LanguageParser, CodeElement


class JuliaParser(LanguageParser):
    """Regex-based Julia parser for extracting code elements."""

    @property
    def extensions(self) -> List[str]:
        return [".jl"]

    @property
    def language(self) -> str:
        return "julia"

    def parse_file(self, file_path: Path) -> Dict[str, List[CodeElement]]:
        """Parse a Julia file using regex patterns."""
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

        # Parse structs (Julia's equivalent of classes)
        struct_pattern = r'(?:mutable\s+)?struct\s+(\w+)(?:\s*<:\s*(\w+))?'
        for i, line in enumerate(lines):
            match = re.match(struct_pattern, line.strip())
            if match:
                name = match.group(1)
                base = match.group(2) or ""
                docstring = self._get_julia_docstring(lines, i)
                is_mutable = "mutable" in line

                result["classes"].append(CodeElement(
                    name=name,
                    element_type="mutable_struct" if is_mutable else "struct",
                    file_path=str(file_path),
                    line_number=i + 1,
                    docstring=docstring,
                    bases=[base] if base else [],
                    language="julia"
                ))

        # Parse abstract types
        abstract_pattern = r'abstract\s+type\s+(\w+)(?:\s*<:\s*(\w+))?'
        for i, line in enumerate(lines):
            match = re.match(abstract_pattern, line.strip())
            if match:
                name = match.group(1)
                base = match.group(2) or ""

                result["classes"].append(CodeElement(
                    name=name,
                    element_type="abstract_type",
                    file_path=str(file_path),
                    line_number=i + 1,
                    bases=[base] if base else [],
                    language="julia"
                ))

        # Parse functions (multi-line definition)
        func_pattern = r'function\s+(\w+)\s*\(([^)]*)\)'
        for i, line in enumerate(lines):
            match = re.match(func_pattern, line.strip())
            if match:
                name = match.group(1)
                args_str = match.group(2)
                args = self._parse_julia_args(args_str)
                docstring = self._get_julia_docstring(lines, i)

                result["functions"].append(CodeElement(
                    name=name,
                    element_type="function",
                    file_path=str(file_path),
                    line_number=i + 1,
                    docstring=docstring,
                    signature=f"function {name}({args_str})",
                    args=args,
                    language="julia"
                ))

        # Parse one-line functions
        oneline_pattern = r'^(\w+)\s*\(([^)]*)\)\s*='
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            match = re.match(oneline_pattern, stripped)
            if match:
                name = match.group(1)
                args_str = match.group(2)
                args = self._parse_julia_args(args_str)

                # Skip if already found as full function
                if not any(f.name == name for f in result["functions"]):
                    result["functions"].append(CodeElement(
                        name=name,
                        element_type="function",
                        file_path=str(file_path),
                        line_number=i + 1,
                        signature=stripped[:100],
                        args=args,
                        language="julia"
                    ))

        # Parse imports/using statements
        import_pattern = r'(?:using|import)\s+(.+)'
        for i, line in enumerate(lines):
            match = re.match(import_pattern, line.strip())
            if match:
                modules = match.group(1).split(',')
                for module in modules:
                    module = module.strip().split(':')[0].strip()
                    if module:
                        result["imports"].append({
                            "module": module,
                            "file_path": str(file_path),
                            "line_number": i + 1
                        })

        # Parse constants (const declarations)
        const_pattern = r'const\s+(\w+)\s*='
        for i, line in enumerate(lines):
            match = re.match(const_pattern, line.strip())
            if match:
                result["constants"].append({
                    "name": match.group(1),
                    "file_path": str(file_path),
                    "line_number": i + 1
                })

        return result

    def _parse_julia_args(self, args_str: str) -> List[str]:
        """Parse Julia function arguments."""
        if not args_str.strip():
            return []

        args = []
        for arg in args_str.split(','):
            arg = arg.strip()
            # Remove type annotations
            if '::' in arg:
                arg = arg.split('::')[0].strip()
            # Remove default values
            if '=' in arg:
                arg = arg.split('=')[0].strip()
            if arg:
                args.append(arg)
        return args

    def _get_julia_docstring(self, lines: List[str], func_line: int) -> str:
        """Extract Julia docstring (triple quotes before function)."""
        if func_line == 0:
            return ""

        docstring_lines = []
        i = func_line - 1
        in_docstring = False

        while i >= 0:
            line = lines[i].strip()
            if line.endswith('"""') and not in_docstring:
                in_docstring = True
                docstring_lines.append(line[:-3])
            elif line.startswith('"""') and in_docstring:
                docstring_lines.append(line[3:])
                break
            elif in_docstring:
                docstring_lines.append(line)
            elif line and not line.startswith("#"):
                break
            i -= 1

        return "\n".join(reversed(docstring_lines))[:500]
