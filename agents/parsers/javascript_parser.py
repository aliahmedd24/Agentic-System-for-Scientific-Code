"""
JavaScript/TypeScript Parser - Regex-based JS/TS code parser.
"""

import re
from typing import Dict, List
from pathlib import Path

from . import LanguageParser, CodeElement


class JavaScriptParser(LanguageParser):
    """Regex-based JavaScript/TypeScript parser for extracting code elements."""

    @property
    def extensions(self) -> List[str]:
        return [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"]

    @property
    def language(self) -> str:
        return "javascript"

    def parse_file(self, file_path: Path) -> Dict[str, List[CodeElement]]:
        """Parse a JavaScript/TypeScript file using regex patterns."""
        result = {
            "classes": [],
            "functions": [],
            "imports": [],
            "constants": []
        }

        content = self._safe_read(file_path)
        if not content:
            return result

        is_typescript = str(file_path).endswith(('.ts', '.tsx'))
        lines = content.split('\n')

        # Parse classes
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?'
        for i, line in enumerate(lines):
            match = re.search(class_pattern, line)
            if match:
                name = match.group(1)
                base = match.group(2) or ""
                implements = match.group(3) or ""

                # Get JSDoc comment
                docstring = self._get_jsdoc(lines, i)

                # Extract methods (simple approach)
                methods = self._extract_class_methods(lines, i)

                result["classes"].append(CodeElement(
                    name=name,
                    element_type="class",
                    file_path=str(file_path),
                    line_number=i + 1,
                    docstring=docstring,
                    bases=[base] if base else [],
                    methods=methods,
                    language="typescript" if is_typescript else "javascript"
                ))

        # Parse function declarations
        func_pattern = r'(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)'
        for i, line in enumerate(lines):
            match = re.search(func_pattern, line)
            if match:
                name = match.group(1)
                args_str = match.group(2)
                args = self._parse_js_args(args_str)
                docstring = self._get_jsdoc(lines, i)
                is_async = "async" in line[:line.find("function")]

                result["functions"].append(CodeElement(
                    name=name,
                    element_type="async_function" if is_async else "function",
                    file_path=str(file_path),
                    line_number=i + 1,
                    docstring=docstring,
                    signature=f"function {name}({args_str})",
                    args=args,
                    language="typescript" if is_typescript else "javascript"
                ))

        # Parse arrow functions assigned to const/let/var
        arrow_pattern = r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*\w+)?\s*=>'
        for i, line in enumerate(lines):
            match = re.search(arrow_pattern, line)
            if match:
                name = match.group(1)
                # Skip if already found
                if any(f.name == name for f in result["functions"]):
                    continue

                docstring = self._get_jsdoc(lines, i)
                is_async = "async" in line

                result["functions"].append(CodeElement(
                    name=name,
                    element_type="arrow_function",
                    file_path=str(file_path),
                    line_number=i + 1,
                    docstring=docstring,
                    language="typescript" if is_typescript else "javascript"
                ))

        # Parse TypeScript interfaces
        if is_typescript:
            interface_pattern = r'(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+([^{]+))?'
            for i, line in enumerate(lines):
                match = re.search(interface_pattern, line)
                if match:
                    name = match.group(1)
                    extends = match.group(2) or ""
                    docstring = self._get_jsdoc(lines, i)

                    result["classes"].append(CodeElement(
                        name=name,
                        element_type="interface",
                        file_path=str(file_path),
                        line_number=i + 1,
                        docstring=docstring,
                        bases=[e.strip() for e in extends.split(',') if e.strip()],
                        language="typescript"
                    ))

            # Parse TypeScript types
            type_pattern = r'(?:export\s+)?type\s+(\w+)\s*='
            for i, line in enumerate(lines):
                match = re.search(type_pattern, line)
                if match:
                    name = match.group(1)
                    docstring = self._get_jsdoc(lines, i)

                    result["classes"].append(CodeElement(
                        name=name,
                        element_type="type_alias",
                        file_path=str(file_path),
                        line_number=i + 1,
                        docstring=docstring,
                        language="typescript"
                    ))

        # Parse imports
        import_pattern = r"import\s+(?:{[^}]+}|\*\s+as\s+\w+|\w+)\s+from\s+['\"]([^'\"]+)['\"]"
        for i, line in enumerate(lines):
            match = re.search(import_pattern, line)
            if match:
                result["imports"].append({
                    "module": match.group(1),
                    "file_path": str(file_path),
                    "line_number": i + 1
                })

        # Parse require statements
        require_pattern = r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"
        for i, line in enumerate(lines):
            match = re.search(require_pattern, line)
            if match:
                result["imports"].append({
                    "module": match.group(1),
                    "file_path": str(file_path),
                    "line_number": i + 1
                })

        # Parse constants
        const_pattern = r'(?:export\s+)?const\s+([A-Z][A-Z0-9_]+)\s*='
        for i, line in enumerate(lines):
            match = re.search(const_pattern, line)
            if match:
                result["constants"].append({
                    "name": match.group(1),
                    "file_path": str(file_path),
                    "line_number": i + 1
                })

        return result

    def _parse_js_args(self, args_str: str) -> List[str]:
        """Parse JavaScript function arguments."""
        if not args_str.strip():
            return []

        args = []
        for arg in args_str.split(','):
            arg = arg.strip()
            # Remove type annotations (TypeScript)
            if ':' in arg:
                arg = arg.split(':')[0].strip()
            # Remove default values
            if '=' in arg:
                arg = arg.split('=')[0].strip()
            if arg:
                args.append(arg)
        return args

    def _get_jsdoc(self, lines: List[str], func_line: int) -> str:
        """Extract JSDoc comment before function/class."""
        if func_line == 0:
            return ""

        docstring_lines = []
        i = func_line - 1
        in_jsdoc = False

        while i >= 0:
            line = lines[i].strip()
            if line.endswith('*/') and not in_jsdoc:
                in_jsdoc = True
                if line.startswith('/**'):
                    # Single-line JSDoc
                    return line[3:-2].strip()
                docstring_lines.append(line[:-2])
            elif line.startswith('/**') and in_jsdoc:
                docstring_lines.append(line[3:])
                break
            elif in_jsdoc:
                # Remove leading * from JSDoc lines
                if line.startswith('*'):
                    line = line[1:].strip()
                docstring_lines.append(line)
            elif line and not in_jsdoc:
                break
            i -= 1

        return "\n".join(reversed(docstring_lines))[:500]

    def _extract_class_methods(self, lines: List[str], class_line: int) -> List[str]:
        """Extract method names from a class (simple approach)."""
        methods = []
        brace_count = 0
        started = False

        for i in range(class_line, min(class_line + 100, len(lines))):
            line = lines[i]
            brace_count += line.count('{') - line.count('}')

            if '{' in line:
                started = True

            if started:
                # Look for method definitions
                method_match = re.match(r'\s*(?:async\s+)?(\w+)\s*\([^)]*\)\s*{', line)
                if method_match and method_match.group(1) not in ('if', 'for', 'while', 'switch'):
                    methods.append(method_match.group(1))

            if started and brace_count == 0:
                break

        return methods
