"""
Repository Analyzer Agent - Analyzes code repositories and extracts structure.
"""

import os
import re
import ast
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
from urllib.parse import urlparse

from .base_agent import BaseAgent
from core.llm_client import LLMClient
from core.knowledge_graph import (
    KnowledgeGraph, NodeType, EdgeType,
    create_function_node
)
from core.agent_prompts import (
    REPO_ANALYZER_SYSTEM_PROMPT,
    REPO_ANALYZER_STRUCTURE_PROMPT,
    REPO_ANALYZER_CODE_EXTRACTION_PROMPT
)
from core.error_handling import (
    logger, LogCategory, ErrorCategory, create_error, AgentError, with_retry
)


# File extensions to analyze
CODE_EXTENSIONS = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.jsx': 'javascript',
    '.tsx': 'typescript',
    '.r': 'r',
    '.R': 'r',
    '.jl': 'julia',
    '.cpp': 'cpp',
    '.c': 'c',
    '.h': 'c',
    '.hpp': 'cpp',
}

# Files to look for
IMPORTANT_FILES = [
    'README.md', 'README.rst', 'README.txt',
    'requirements.txt', 'setup.py', 'setup.cfg', 'pyproject.toml',
    'package.json', 'Cargo.toml',
    'main.py', 'app.py', 'run.py', '__main__.py',
    'config.py', 'settings.py',
    'train.py', 'test.py', 'evaluate.py', 'inference.py',
    'model.py', 'models.py', 'network.py',
]

# Directories to skip
SKIP_DIRS = {
    '.git', '.github', '__pycache__', 'node_modules', '.venv', 'venv',
    'env', '.env', 'build', 'dist', '.eggs', '*.egg-info', '.tox',
    '.pytest_cache', '.mypy_cache', 'htmlcov', '.coverage',
}


class RepoAnalyzerAgent(BaseAgent):
    """
    Analyzes code repositories to understand structure and implementations.
    """
    
    def __init__(self, llm_client: LLMClient, github_token: Optional[str] = None):
        super().__init__(llm_client, name="RepoAnalyzer")
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self._temp_dirs: List[str] = []
    
    async def process(self, repo_url: str, kg: KnowledgeGraph) -> Dict[str, Any]:
        """Main processing method."""
        return await self.analyze(repo_url, kg)
    
    async def analyze(
        self,
        repo_url: str,
        kg: KnowledgeGraph
    ) -> Dict[str, Any]:
        """
        Analyze a code repository.
        
        Args:
            repo_url: GitHub URL or local path
            kg: Knowledge graph to populate
            
        Returns:
            Dictionary with repository analysis
        """
        self.log_info(f"Analyzing repository: {repo_url}")
        
        # Step 1: Clone/locate repository
        repo_path = await self._timed_operation(
            "get_repository",
            self._get_repository(repo_url)
        )
        
        # Step 2: Scan structure
        structure = await self._timed_operation(
            "scan_structure",
            self._scan_structure(repo_path)
        )
        
        # Step 3: Extract code elements
        code_elements = await self._timed_operation(
            "extract_code",
            self._extract_code_elements(repo_path, structure)
        )
        
        # Step 4: Read key files
        key_files_content = await self._timed_operation(
            "read_key_files",
            self._read_key_files(repo_path, structure)
        )
        
        # Step 5: LLM analysis
        repo_data = await self._timed_operation(
            "llm_analysis",
            self._analyze_repository(
                repo_url, structure, code_elements, key_files_content
            )
        )
        
        # Step 6: Populate knowledge graph
        await self._populate_knowledge_graph(repo_data, code_elements, kg)
        
        # Add raw data to result
        repo_data["_structure"] = structure
        repo_data["_code_elements"] = code_elements
        repo_data["_repo_path"] = str(repo_path)
        
        return repo_data
    
    @with_retry(ErrorCategory.NETWORK, "Repository clone")
    async def _get_repository(self, source: str) -> Path:
        """Get repository from URL or local path."""
        # Check if local path
        if Path(source).exists():
            self.log_info("Using local repository")
            return Path(source)
        
        # Parse GitHub URL
        parsed = urlparse(source)
        if "github.com" not in parsed.netloc:
            raise AgentError(create_error(
                ErrorCategory.VALIDATION,
                f"Only GitHub repositories are supported: {source}",
                suggestion="Provide a GitHub URL like https://github.com/owner/repo"
            ))
        
        # Clone repository
        return await self._clone_github(source)
    
    async def _clone_github(self, url: str) -> Path:
        """Clone a GitHub repository."""
        import subprocess
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="repo_")
        self._temp_dirs.append(temp_dir)
        
        # Parse URL for repo name
        parts = urlparse(url).path.strip('/').split('/')
        if len(parts) >= 2:
            repo_name = parts[1].replace('.git', '')
        else:
            repo_name = "repo"
        
        clone_path = Path(temp_dir) / repo_name
        
        # Build clone command
        clone_url = url
        if self.github_token and "github.com" in url:
            # Add token for private repos
            clone_url = url.replace(
                "https://github.com",
                f"https://{self.github_token}@github.com"
            )
        
        self.log_info(f"Cloning repository to {clone_path}")
        
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", clone_url, str(clone_path)],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                raise AgentError(create_error(
                    ErrorCategory.NETWORK,
                    f"Git clone failed: {result.stderr}",
                    suggestion="Check the repository URL and access permissions"
                ))
            
            return clone_path
            
        except subprocess.TimeoutExpired:
            raise AgentError(create_error(
                ErrorCategory.TIMEOUT,
                "Repository clone timed out",
                suggestion="The repository might be too large"
            ))
    
    async def _scan_structure(self, repo_path: Path) -> Dict[str, Any]:
        """Scan repository structure."""
        structure = {
            "root": str(repo_path),
            "name": repo_path.name,
            "files": [],
            "directories": [],
            "file_counts": {},
            "total_files": 0,
            "total_size": 0,
            "important_files": [],
            "code_files": [],
        }
        
        for root, dirs, files in os.walk(repo_path):
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]
            
            rel_root = Path(root).relative_to(repo_path)
            
            for file in files:
                if file.startswith('.'):
                    continue
                
                file_path = Path(root) / file
                rel_path = str(rel_root / file) if str(rel_root) != '.' else file
                
                try:
                    size = file_path.stat().st_size
                except:
                    size = 0
                
                structure["total_files"] += 1
                structure["total_size"] += size
                
                ext = file_path.suffix.lower()
                structure["file_counts"][ext] = structure["file_counts"].get(ext, 0) + 1
                
                file_info = {
                    "path": rel_path,
                    "size": size,
                    "extension": ext
                }
                
                structure["files"].append(file_info)
                
                # Track important files
                if file in IMPORTANT_FILES:
                    structure["important_files"].append(file_info)
                
                # Track code files
                if ext in CODE_EXTENSIONS:
                    structure["code_files"].append(file_info)
            
            # Track directories
            for d in dirs:
                rel_dir = str(rel_root / d) if str(rel_root) != '.' else d
                structure["directories"].append(rel_dir)
        
        self.log_info(
            f"Scanned {structure['total_files']} files, "
            f"{len(structure['code_files'])} code files"
        )
        
        return structure
    
    async def _extract_code_elements(
        self,
        repo_path: Path,
        structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract classes, functions, and other code elements."""
        elements = {
            "classes": [],
            "functions": [],
            "imports": [],
            "constants": [],
        }
        
        # Focus on Python files for now
        python_files = [f for f in structure["code_files"] if f["extension"] == ".py"]
        
        # Limit to reasonable number of files
        if len(python_files) > 50:
            # Prioritize important-looking files
            def file_importance(f):
                path = f["path"].lower()
                score = 0
                if "model" in path: score += 3
                if "train" in path: score += 2
                if "main" in path: score += 2
                if "core" in path: score += 1
                if "util" in path: score -= 1
                if "test" in path: score -= 2
                return -score
            
            python_files = sorted(python_files, key=file_importance)[:50]
        
        for file_info in python_files:
            file_path = repo_path / file_info["path"]
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                file_elements = self._parse_python_file(content, file_info["path"])
                
                elements["classes"].extend(file_elements.get("classes", []))
                elements["functions"].extend(file_elements.get("functions", []))
                elements["imports"].extend(file_elements.get("imports", []))
                elements["constants"].extend(file_elements.get("constants", []))
                
            except Exception as e:
                self.log_warning(f"Failed to parse {file_info['path']}: {e}")
        
        self.log_info(
            f"Extracted {len(elements['classes'])} classes, "
            f"{len(elements['functions'])} functions"
        )
        
        return elements
    
    def _parse_python_file(self, content: str, file_path: str) -> Dict[str, Any]:
        """Parse a Python file using AST."""
        elements = {
            "classes": [],
            "functions": [],
            "imports": [],
            "constants": [],
        }
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return elements
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "file_path": file_path,
                    "line": node.lineno,
                    "docstring": ast.get_docstring(node) or "",
                    "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                    "bases": [self._get_name(b) for b in node.bases],
                    "decorators": [self._get_name(d) for d in node.decorator_list],
                }
                elements["classes"].append(class_info)
            
            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                # Top-level functions only
                func_info = {
                    "name": node.name,
                    "file_path": file_path,
                    "line": node.lineno,
                    "docstring": ast.get_docstring(node) or "",
                    "args": [arg.arg for arg in node.args.args],
                    "decorators": [self._get_name(d) for d in node.decorator_list],
                    "returns": self._get_annotation(node.returns),
                }
                elements["functions"].append(func_info)
            
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        elements["imports"].append({
                            "module": alias.name,
                            "alias": alias.asname,
                            "file_path": file_path
                        })
                else:
                    module = node.module or ""
                    for alias in node.names:
                        elements["imports"].append({
                            "module": f"{module}.{alias.name}" if module else alias.name,
                            "alias": alias.asname,
                            "file_path": file_path
                        })
            
            elif isinstance(node, ast.Assign):
                # Look for module-level constants (ALL_CAPS names)
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        elements["constants"].append({
                            "name": target.id,
                            "file_path": file_path,
                            "line": node.lineno
                        })
        
        return elements
    
    def _get_name(self, node) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        return str(type(node).__name__)
    
    def _get_annotation(self, node) -> Optional[str]:
        """Get type annotation as string."""
        if node is None:
            return None
        return self._get_name(node)
    
    async def _read_key_files(
        self,
        repo_path: Path,
        structure: Dict[str, Any]
    ) -> Dict[str, str]:
        """Read content of important files."""
        key_files = {}
        
        for file_info in structure["important_files"][:10]:  # Limit
            file_path = repo_path / file_info["path"]
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                # Truncate very long files
                if len(content) > 10000:
                    content = content[:10000] + "\n\n[...TRUNCATED...]"
                key_files[file_info["path"]] = content
            except Exception as e:
                self.log_warning(f"Failed to read {file_info['path']}: {e}")
        
        return key_files
    
    async def _analyze_repository(
        self,
        repo_url: str,
        structure: Dict[str, Any],
        code_elements: Dict[str, Any],
        key_files: Dict[str, str]
    ) -> Dict[str, Any]:
        """Use LLM to analyze repository."""
        # Prepare structure summary
        structure_summary = f"""
Repository: {structure['name']}
Total files: {structure['total_files']}
Code files: {len(structure['code_files'])}

File types:
{chr(10).join(f'  {ext}: {count}' for ext, count in sorted(structure['file_counts'].items(), key=lambda x: -x[1])[:10])}

Directories:
{chr(10).join('  ' + d for d in structure['directories'][:20])}
"""
        
        # Prepare key files summary
        key_files_summary = ""
        for path, content in key_files.items():
            key_files_summary += f"\n--- {path} ---\n{content[:3000]}\n"
        
        prompt = REPO_ANALYZER_STRUCTURE_PROMPT.format(
            repo_name=structure['name'],
            structure=structure_summary,
            key_files=key_files_summary
        )
        
        try:
            result = await self.llm.generate_structured(
                prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "overview": {"type": "object"},
                        "key_components": {"type": "array"},
                        "entry_points": {"type": "array"},
                        "dependencies": {"type": "object"},
                        "setup_complexity": {"type": "object"},
                        "compute_requirements": {"type": "object"}
                    }
                },
                system_instruction=REPO_ANALYZER_SYSTEM_PROMPT
            )
            
            result["name"] = structure["name"]
            result["url"] = repo_url
            result["stats"] = {
                "total_files": structure["total_files"],
                "code_files": len(structure["code_files"]),
                "classes": len(code_elements["classes"]),
                "functions": len(code_elements["functions"])
            }
            
            return result
            
        except Exception as e:
            self.log_error(f"LLM analysis failed: {e}")
            return {
                "name": structure["name"],
                "url": repo_url,
                "overview": {"purpose": "Analysis failed"},
                "key_components": [],
                "entry_points": [],
                "dependencies": {},
                "setup_complexity": {"level": "unknown"},
                "compute_requirements": {},
                "stats": {
                    "total_files": structure["total_files"],
                    "code_files": len(structure["code_files"]),
                    "classes": len(code_elements["classes"]),
                    "functions": len(code_elements["functions"])
                },
                "analysis_error": str(e)
            }
    
    async def _populate_knowledge_graph(
        self,
        repo_data: Dict[str, Any],
        code_elements: Dict[str, Any],
        kg: KnowledgeGraph
    ):
        """Add repository information to knowledge graph."""
        # Create repository node
        repo_id = kg.add_node(
            type=NodeType.REPOSITORY,
            name=repo_data.get("name", "Unknown"),
            description=repo_data.get("overview", {}).get("purpose", ""),
            metadata={
                "url": repo_data.get("url"),
                "stats": repo_data.get("stats", {})
            }
        )
        
        # Create class nodes
        for cls in code_elements.get("classes", [])[:50]:  # Limit
            cls_id = kg.add_node(
                type=NodeType.CLASS,
                name=cls["name"],
                description=cls.get("docstring", "")[:500],
                metadata={
                    "file_path": cls["file_path"],
                    "methods": cls.get("methods", []),
                    "bases": cls.get("bases", [])
                }
            )
            kg.add_edge(repo_id, cls_id, EdgeType.CONTAINS)
        
        # Create function nodes
        for func in code_elements.get("functions", [])[:100]:  # Limit
            func_id = create_function_node(
                kg,
                func["name"],
                func["file_path"],
                signature=f"({', '.join(func.get('args', []))})",
                docstring=func.get("docstring", "")[:500]
            )
            kg.add_edge(repo_id, func_id, EdgeType.CONTAINS)
        
        repo_data["_kg_repo_id"] = repo_id
        self.log_info(f"Knowledge graph now has {kg.get_statistics()['total_nodes']} nodes")
    
    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self._temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        self._temp_dirs = []
