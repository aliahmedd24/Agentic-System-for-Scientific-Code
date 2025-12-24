"""
Python Parser - AST-based Python code parser.
"""

import ast
from typing import Dict, List
from pathlib import Path

from . import LanguageParser, CodeElement


class PythonParser(LanguageParser):
    """AST-based Python parser for accurate code element extraction."""

    @property
    def extensions(self) -> List[str]:
        return [".py", ".pyx"]

    @property
    def language(self) -> str:
        return "python"

    def parse_file(self, file_path: Path) -> Dict[str, List[CodeElement]]:
        """Parse a Python file using AST."""
        result = {
            "classes": [],
            "functions": [],
            "imports": [],
            "constants": []
        }

        content = self._safe_read(file_path)
        if not content:
            return result

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return result
        except Exception:
            return result

        # Add parent references for context
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child._parent = node

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [
                    n.name for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                bases = [self._get_name(b) for b in node.bases]
                docstring = ast.get_docstring(node) or ""

                result["classes"].append(CodeElement(
                    name=node.name,
                    element_type="class",
                    file_path=str(file_path),
                    line_number=node.lineno,
                    docstring=docstring[:500],
                    signature=f"class {node.name}",
                    bases=bases,
                    methods=methods,
                    decorators=[self._get_name(d) for d in node.decorator_list],
                    language="python"
                ))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if this is a method (inside a class)
                parent = getattr(node, '_parent', None)
                if isinstance(parent, ast.ClassDef):
                    continue  # Skip methods, they're handled with classes

                args = [arg.arg for arg in node.args.args]
                docstring = ast.get_docstring(node) or ""
                return_type = self._get_annotation(node.returns) if node.returns else ""
                is_async = isinstance(node, ast.AsyncFunctionDef)

                result["functions"].append(CodeElement(
                    name=node.name,
                    element_type="async_function" if is_async else "function",
                    file_path=str(file_path),
                    line_number=node.lineno,
                    docstring=docstring[:500],
                    signature=f"{'async ' if is_async else ''}def {node.name}({', '.join(args)})",
                    args=args,
                    return_type=return_type,
                    decorators=[self._get_name(d) for d in node.decorator_list],
                    language="python"
                ))

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                # Track imports for dependency analysis
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        result["imports"].append({
                            "module": alias.name,
                            "alias": alias.asname,
                            "file_path": str(file_path),
                            "line_number": node.lineno
                        })
                else:
                    module = node.module or ""
                    for alias in node.names:
                        full_name = f"{module}.{alias.name}" if module else alias.name
                        result["imports"].append({
                            "module": full_name,
                            "alias": alias.asname,
                            "file_path": str(file_path),
                            "line_number": node.lineno
                        })

            elif isinstance(node, ast.Assign):
                # Look for module-level constants (ALL_CAPS names)
                parent = getattr(node, '_parent', None)
                if isinstance(parent, ast.Module):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            result["constants"].append({
                                "name": target.id,
                                "file_path": str(file_path),
                                "line_number": node.lineno
                            })

        return result

    def _get_name(self, node) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[...]"
        return str(type(node).__name__)

    def _get_annotation(self, node) -> str:
        """Get type annotation as string."""
        if node is None:
            return ""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[...]"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return self._get_name(node)
