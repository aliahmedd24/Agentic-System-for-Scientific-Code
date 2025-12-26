"""
Coding Agent - Generates and executes validation tests for paper concepts.

Supports multiple languages: Python, Julia, R, MATLAB/Octave
Executes tests in Docker sandbox with repo dependencies installed.
"""

import os
import ast
import sys
import asyncio
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import BaseAgent
from .protocols import (
    MappingResult, RepoAnalyzerOutput,
    CodingAgentOutput, GeneratedScript, TestResult, ExecutionSummary
)
from core.llm_client import LLMClient
from core.knowledge_graph import KnowledgeGraph, NodeType, EdgeType
from core.agent_prompts import (
    CODING_AGENT_SYSTEM_PROMPT,
    CODING_AGENT_TEST_GENERATION_PROMPT,
    CODING_AGENT_DEBUG_PROMPT
)
from core.error_handling import logger, LogCategory
from core.resource_estimator import estimate_resources
from core.bubblewrap_sandbox import (
    SandboxManager, SandboxConfig, ExecutionResult, IsolationLevel
)


# Supported languages and their configurations
LANGUAGE_CONFIG = {
    "python": {
        "extensions": [".py"],
        "docker_image": "python:3.11-slim",
        "install_cmd": "pip install",
        "run_cmd": "python",
        "dep_files": ["requirements.txt", "setup.py", "pyproject.toml"],
        "pkg_manager": "pip"
    },
    "julia": {
        "extensions": [".jl"],
        "docker_image": "julia:1.10",
        "install_cmd": "julia -e 'using Pkg; Pkg.add'",
        "run_cmd": "julia",
        "dep_files": ["Project.toml", "Manifest.toml"],
        "pkg_manager": "Pkg"
    },
    "r": {
        "extensions": [".r", ".R"],
        "docker_image": "r-base:4.3.0",
        "install_cmd": "Rscript -e 'install.packages'",
        "run_cmd": "Rscript",
        "dep_files": ["DESCRIPTION", "renv.lock"],
        "pkg_manager": "CRAN"
    },
    "matlab": {
        "extensions": [".m"],
        "docker_image": "gnuoctave/octave:8.4.0",  # Use Octave as MATLAB alternative
        "install_cmd": "octave --eval 'pkg install'",
        "run_cmd": "octave --no-gui",
        "dep_files": [],
        "pkg_manager": "octave-forge"
    }
}


class CodingAgent(BaseAgent):
    """
    Generates and executes code to validate paper implementations.
    
    Features:
    - Multi-language support: Python, Julia, R, MATLAB/Octave
    - Parses repo dependency files for each language
    - Builds Docker sandbox with dependencies installed
    - Mounts cloned repo and configures language paths
    - Generates tests that import and validate actual repo code
    - Self-correction on errors
    """
    
    def __init__(self, llm_client: LLMClient, max_retries: int = 3):
        super().__init__(llm_client, name="CodingAgent")
        self.max_retries = max_retries
        self._docker_available = self._check_docker()
        self._sandbox_image = None

        # Initialize SandboxManager for secure execution
        self._sandbox_manager = SandboxManager()
        self.log_info(
            f"SandboxManager initialized with backends: {self._sandbox_manager.available_backends}"
        )
    
    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5
            )
            available = result.returncode == 0
            if available:
                self.log_info("Docker is available for sandboxed execution")
            else:
                self.log_warning("Docker not available, using subprocess execution")
            return available
        except Exception as e:
            self.log_warning(f"Docker check failed: {e}, using subprocess execution")
            return False
    
    def _detect_primary_language(self, repo_path: str, repo_data: Dict[str, Any]) -> str:
        """Detect the primary programming language of the repository."""
        if not repo_path:
            return "python"
        
        repo_path = Path(repo_path)
        
        # Count files by language
        lang_counts = {lang: 0 for lang in LANGUAGE_CONFIG.keys()}
        
        for lang, config in LANGUAGE_CONFIG.items():
            for ext in config["extensions"]:
                count = len(list(repo_path.rglob(f"*{ext}")))
                lang_counts[lang] += count
        
        # Check for language-specific dependency files
        for lang, config in LANGUAGE_CONFIG.items():
            for dep_file in config["dep_files"]:
                if (repo_path / dep_file).exists():
                    lang_counts[lang] += 50  # Boost for having dep file
        
        # Get language with most files
        primary = max(lang_counts, key=lang_counts.get)
        
        # Default to python if no clear winner
        if lang_counts[primary] == 0:
            primary = "python"
        
        self.log_info(f"Detected primary language: {primary} (counts: {lang_counts})")
        return primary
    
    async def process(
        self,
        *,
        mappings: List[MappingResult],
        repo_data: Any,
        knowledge_graph: KnowledgeGraph = None,
        execute: bool = True
    ) -> CodingAgentOutput:
        """
        Generate and execute validation tests for paper concepts.

        Args:
            mappings: Concept-to-code mappings (REQUIRED) - List[MappingResult]
            repo_data: Repository analysis data (REQUIRED) - RepoAnalyzerOutput or dict
            knowledge_graph: Optional knowledge graph
            execute: Whether to execute generated tests (default True)

        Returns:
            CodingAgentOutput with scripts, results, and language
        """
        if not mappings:
            self.log_warning("No mappings provided for code generation")
            return CodingAgentOutput(scripts=[], results=[], language="python")
        if not repo_data:
            raise ValueError("repo_data is required")
        if knowledge_graph is None:
            knowledge_graph = KnowledgeGraph()

        # Get repo path - handle both Pydantic and dict
        if isinstance(repo_data, RepoAnalyzerOutput):
            repo_path = repo_data._repo_path or ""
        else:
            repo_path = repo_data.get("_repo_path", "")

        # Detect primary language
        language = self._detect_primary_language(repo_path, repo_data)

        # Parse repository dependencies (language-aware)
        dependencies = await self._parse_repo_dependencies(repo_path, repo_data, language)
        dependencies["language"] = language

        # Generate validation tests
        scripts = await self.generate_tests(mappings, repo_data, knowledge_graph, dependencies)

        # Estimate resources for the generated code
        resource_estimate = None
        if scripts:
            # Combine all script code for resource estimation
            all_code = "\n".join(script.code for script in scripts)
            resource_estimate = estimate_resources(code=all_code)

            # Log resource warnings
            for warning in resource_estimate.warnings:
                self.log_warning(f"Resource warning: {warning}")

            if resource_estimate.gpu_required:
                self.log_warning(
                    f"Code may require GPU ({resource_estimate.gpu_memory_gb}GB VRAM)"
                )

            if resource_estimate.memory_gb > 8:
                self.log_warning(
                    f"High memory requirement: {resource_estimate.memory_gb}GB"
                )

            # Check feasibility (16GB RAM, no GPU assumed for local execution)
            if not resource_estimate.is_feasible(available_memory_gb=16, has_gpu=False):
                self.log_warning(
                    "Code may require more resources than available - proceeding anyway"
                )

        # Execute in sandbox using SandboxManager
        results: List[TestResult] = []
        if execute and scripts:
            output_dir = Path(tempfile.mkdtemp(prefix="validation_"))
            try:
                # Use SandboxManager for secure execution with proper isolation
                results = await self._execute_with_sandbox_manager(
                    scripts, repo_path, dependencies, output_dir, resource_estimate
                )
            except Exception as e:
                self.log_error(f"Sandbox execution failed: {e}")
                # Fallback to legacy execution methods if SandboxManager fails
                try:
                    if self._docker_available:
                        results = await self._execute_in_docker_sandbox(
                            scripts, repo_path, dependencies, output_dir
                        )
                    else:
                        results = await self._execute_in_venv_sandbox(
                            scripts, repo_path, dependencies, output_dir
                        )
                except Exception as fallback_error:
                    self.log_error(f"Fallback execution also failed: {fallback_error}")

        # Build execution summary
        summary = ExecutionSummary(
            total_tests=len(results),
            passed=sum(1 for r in results if r.success),
            failed=sum(1 for r in results if not r.success),
            skipped=0,
            total_time=sum(r.execution_time for r in results)
        )

        return CodingAgentOutput(
            scripts=scripts,
            results=results,
            language=language,
            summary=summary,
            resource_estimate=None  # TODO: Convert resource_estimate to Pydantic
        )
    
    async def _parse_repo_dependencies(
        self, 
        repo_path: str, 
        repo_data: Dict[str, Any],
        language: str = "python"
    ) -> Dict[str, Any]:
        """Parse repository dependencies based on detected language."""
        deps = {
            "packages": [],
            "dep_file": None,
            "language": language
        }
        
        if not repo_path:
            return deps
        
        repo_path = Path(repo_path)
        
        if language == "python":
            deps.update(await self._parse_python_deps(repo_path, repo_data))
        elif language == "julia":
            deps.update(await self._parse_julia_deps(repo_path))
        elif language == "r":
            deps.update(await self._parse_r_deps(repo_path))
        elif language == "matlab":
            deps.update(await self._parse_matlab_deps(repo_path))
        
        self.log_info(f"Found {len(deps['packages'])} {language} dependencies")
        return deps
    
    async def _parse_python_deps(self, repo_path: Path, repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Python dependencies."""
        deps = {"packages": [], "dep_file": None}
        
        # Check requirements.txt variants
        req_files = ["requirements.txt", "requirements-dev.txt", "requirements_dev.txt"]
        for req_file in req_files:
            req_path = repo_path / req_file
            if req_path.exists():
                deps["dep_file"] = str(req_path)
                try:
                    content = req_path.read_text()
                    for line in content.split('\n'):
                        line = line.strip()
                        if line and not line.startswith('#') and not line.startswith('-'):
                            pkg = line.split('==')[0].split('>=')[0].split('<=')[0].split('[')[0].strip()
                            if pkg and not pkg.startswith('git+'):
                                deps["packages"].append(pkg)
                except (OSError, UnicodeDecodeError) as e:
                    self.log_debug(f"Failed to parse {req_file}: {e}")
                break
        
        # Check setup.py
        setup_py = repo_path / "setup.py"
        if setup_py.exists():
            deps["setup_py"] = str(setup_py)
        
        # Check pyproject.toml
        pyproject = repo_path / "pyproject.toml"
        if pyproject.exists():
            deps["pyproject"] = str(pyproject)
            try:
                content = pyproject.read_text()
                # Simple extraction of dependencies
                if "dependencies" in content:
                    import re
                    matches = re.findall(r'"([a-zA-Z0-9_-]+)', content)
                    deps["packages"].extend(matches[:20])
            except (OSError, UnicodeDecodeError) as e:
                self.log_debug(f"Failed to parse pyproject.toml: {e}")
        
        # Get from repo_data
        if repo_data and "dependencies" in repo_data:
            repo_deps = repo_data["dependencies"]
            if isinstance(repo_deps, dict):
                deps["packages"].extend(repo_deps.get("python", []))
            elif isinstance(repo_deps, list):
                deps["packages"].extend(repo_deps)
        
        deps["packages"] = list(set(deps["packages"]))
        return deps
    
    async def _parse_julia_deps(self, repo_path: Path) -> Dict[str, Any]:
        """Parse Julia Project.toml dependencies."""
        deps = {"packages": [], "dep_file": None}
        
        project_toml = repo_path / "Project.toml"
        if project_toml.exists():
            deps["dep_file"] = str(project_toml)
            try:
                content = project_toml.read_text()
                # Parse [deps] section
                in_deps = False
                for line in content.split('\n'):
                    if line.strip() == "[deps]":
                        in_deps = True
                        continue
                    if line.startswith("[") and in_deps:
                        break
                    if in_deps and "=" in line:
                        pkg = line.split("=")[0].strip()
                        if pkg:
                            deps["packages"].append(pkg)
            except (OSError, UnicodeDecodeError) as e:
                self.log_debug(f"Failed to parse Julia Project.toml: {e}")
        
        return deps
    
    async def _parse_r_deps(self, repo_path: Path) -> Dict[str, Any]:
        """Parse R DESCRIPTION file dependencies."""
        deps = {"packages": [], "dep_file": None}
        
        # Check DESCRIPTION file
        desc_file = repo_path / "DESCRIPTION"
        if desc_file.exists():
            deps["dep_file"] = str(desc_file)
            try:
                content = desc_file.read_text()
                # Parse Imports and Depends
                import re
                for field in ["Imports:", "Depends:"]:
                    match = re.search(f"{field}([^A-Z]+)", content)
                    if match:
                        pkgs = match.group(1).replace("\n", " ").split(",")
                        for pkg in pkgs:
                            pkg = pkg.strip().split("(")[0].strip()
                            if pkg and pkg != "R":
                                deps["packages"].append(pkg)
            except (OSError, UnicodeDecodeError) as e:
                self.log_debug(f"Failed to parse R DESCRIPTION: {e}")
        
        # Check renv.lock
        renv_lock = repo_path / "renv.lock"
        if renv_lock.exists():
            deps["renv_lock"] = str(renv_lock)
            try:
                import json
                content = json.loads(renv_lock.read_text())
                for pkg_name in content.get("Packages", {}).keys():
                    deps["packages"].append(pkg_name)
            except (OSError, json.JSONDecodeError) as e:
                self.log_debug(f"Failed to parse renv.lock: {e}")
        
        deps["packages"] = list(set(deps["packages"]))
        return deps
    
    async def _parse_matlab_deps(self, repo_path: Path) -> Dict[str, Any]:
        """Parse MATLAB/Octave dependencies (limited support)."""
        deps = {"packages": [], "dep_file": None, "toolboxes": []}
        
        # Look for common toolbox indicators in .m files
        toolbox_patterns = {
            "signal": "Signal Processing",
            "image": "Image Processing",
            "optim": "Optimization",
            "stats": "Statistics",
            "neural": "Neural Network",
            "deep": "Deep Learning"
        }
        
        for m_file in list(repo_path.rglob("*.m"))[:20]:
            try:
                content = m_file.read_text()
                for pattern, toolbox in toolbox_patterns.items():
                    if pattern in content.lower():
                        deps["toolboxes"].append(toolbox)
            except (OSError, UnicodeDecodeError):
                # Skip files that can't be read
                continue
        
        deps["toolboxes"] = list(set(deps["toolboxes"]))
        return deps
        
        return deps
    
    async def generate_tests(
        self,
        mappings: List[MappingResult],
        repo_data: Any,
        kg: KnowledgeGraph,
        dependencies: Dict[str, Any]
    ) -> List[GeneratedScript]:
        """Generate validation test scripts for concept-code mappings."""
        if not mappings:
            self.log_warning("No mappings provided for code generation")
            return []

        generated_tests: List[GeneratedScript] = []

        # Get repo path and name - handle both Pydantic and dict
        if isinstance(repo_data, RepoAnalyzerOutput):
            repo_path = repo_data._repo_path or ""
            repo_name = repo_data.name or "repo"
        else:
            repo_path = repo_data.get("_repo_path", "") if repo_data else ""
            repo_name = repo_data.get("name", "repo") if repo_data else "repo"

        language = dependencies.get("language", "python")

        # Get file extension for this language
        extensions = {
            "python": ".py",
            "julia": ".jl",
            "r": ".R",
            "matlab": ".m"
        }
        ext = extensions.get(language, ".py")

        # Get mappings - prefer high-confidence but fall back to any
        high_confidence = [m for m in mappings if m.confidence >= 0.3]
        if not high_confidence:
            high_confidence = sorted(mappings, key=lambda x: x.confidence, reverse=True)
        high_confidence = high_confidence[:5]

        self.log_info(f"Generating {language} tests for {len(high_confidence)} mappings")

        for i, mapping in enumerate(high_confidence):
            concept_name = mapping.concept_name
            self.log_info(f"Generating test {i+1}/{len(high_confidence)}: {concept_name}")

            try:
                code_file = mapping.code_file
                code_element = mapping.code_element
                actual_code = await self._extract_code_context(repo_path, code_file, code_element)

                # Compute correct import path
                import_info = self._compute_import_path(repo_path, code_file, code_element)

                test_code = await self._generate_validation_test(
                    mapping, repo_name, actual_code, dependencies, import_info
                )

                if test_code:
                    # Multi-stage validation
                    validation_result = await self._validate_test(
                        test_code, language, repo_path, code_file
                    )

                    if not validation_result["valid"]:
                        # Attempt to fix issues
                        test_code = await self._fix_code(
                            test_code,
                            validation_result["error"],
                            validation_result["error_type"]
                        )
                        validation_result = await self._validate_test(
                            test_code, language, repo_path, code_file
                        )

                    generated_tests.append(GeneratedScript(
                        concept=concept_name,
                        code_element=code_element,
                        code_file=code_file,
                        confidence=mapping.confidence,
                        code=test_code,
                        syntax_valid=validation_result["syntax_valid"],
                        import_valid=validation_result.get("import_valid", True),
                        validation_error=validation_result.get("error"),
                        file_name=f"test_{i+1}_{self._safe_filename(concept_name)}{ext}",
                        language=language,
                        import_path=import_info.get("import_statement", "")
                    ))

            except Exception as e:
                self.log_error(f"Test generation failed for {concept_name}: {e}")

        self.log_info(f"Generated {len(generated_tests)} test scripts")
        return generated_tests

    def _compute_import_path(
        self,
        repo_path: str,
        code_file: str,
        code_element: str
    ) -> Dict[str, Any]:
        """
        Compute the correct import path for a code element.

        Returns:
            Dict with import_statement, module_path, and element_name
        """
        if not repo_path or not code_file:
            return {"import_statement": "", "module_path": "", "element_name": code_element}

        # Normalize the file path
        file_path = Path(code_file)

        # Remove .py extension and convert path separators to dots
        if file_path.suffix == ".py":
            module_path = str(file_path.with_suffix("")).replace("/", ".").replace("\\", ".")

            # Remove leading dots
            module_path = module_path.lstrip(".")

            # Handle __init__.py files
            if module_path.endswith(".__init__"):
                module_path = module_path[:-9]

            if code_element:
                import_statement = f"from {module_path} import {code_element}"
            else:
                import_statement = f"import {module_path}"

            return {
                "import_statement": import_statement,
                "module_path": module_path,
                "element_name": code_element
            }

        # For non-Python files, return basic info
        return {
            "import_statement": "",
            "module_path": str(file_path),
            "element_name": code_element
        }

    async def _validate_test(
        self,
        code: str,
        language: str,
        repo_path: str,
        code_file: str = ""  # Reserved for future use
    ) -> Dict[str, Any]:
        """
        Multi-stage test validation.

        Validates:
        1. Syntax correctness
        2. Import statements
        3. Basic structure
        """
        result = {
            "valid": True,
            "syntax_valid": True,
            "import_valid": True,
            "error": None,
            "error_type": None
        }

        # Stage 1: Syntax validation
        syntax_valid, syntax_error = self._validate_syntax(code, language)
        result["syntax_valid"] = syntax_valid
        if not syntax_valid:
            result["valid"] = False
            result["error"] = syntax_error
            result["error_type"] = "SyntaxError"
            return result

        # Stage 2: Import validation (Python only for now)
        if language == "python":
            import_valid, import_error = self._validate_imports(code, repo_path)
            result["import_valid"] = import_valid
            if not import_valid:
                result["valid"] = False
                result["error"] = import_error
                result["error_type"] = "ImportError"
                return result

        # Stage 3: Structure validation
        structure_valid, structure_error = self._validate_test_structure(code, language)
        if not structure_valid:
            result["valid"] = False
            result["error"] = structure_error
            result["error_type"] = "StructureError"
            return result

        return result

    def _validate_imports(self, code: str, repo_path: str) -> tuple:
        """Validate that import statements reference existing modules."""
        if not repo_path:
            return True, None

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return True, None  # Syntax errors handled elsewhere

        repo_path = Path(repo_path)
        errors = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    # Check if this is a repo-relative import
                    module_path = node.module.replace(".", "/")
                    potential_paths = [
                        repo_path / f"{module_path}.py",
                        repo_path / module_path / "__init__.py"
                    ]

                    # Only flag as error if it looks like a repo import but doesn't exist
                    if not any(p.exists() for p in potential_paths):
                        # Check if it's a standard library or third-party import
                        first_part = node.module.split(".")[0]
                        if (repo_path / first_part).exists() or (repo_path / f"{first_part}.py").exists():
                            errors.append(f"Module not found: {node.module}")

        if errors:
            return False, "; ".join(errors)
        return True, None

    def _validate_test_structure(self, code: str, language: str) -> tuple:
        """Validate that the test has proper structure."""
        if language != "python":
            return True, None

        # Check for basic test structure elements
        has_main_guard = "if __name__" in code
        has_print_or_assert = "print(" in code or "assert " in code

        if not has_main_guard and not has_print_or_assert:
            return False, "Test has no output mechanism (print/assert)"

        return True, None
    
    async def _extract_code_context(
        self, 
        repo_path: str, 
        code_file: str, 
        code_element: str
    ) -> str:
        """Extract relevant code from the repository."""
        if not repo_path or not code_file:
            return ""
        
        full_path = Path(repo_path) / code_file
        if not full_path.exists():
            return ""
        
        try:
            content = full_path.read_text(encoding='utf-8', errors='ignore')
            
            # Try to extract just the relevant class/function
            if code_element:
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            if node.name == code_element:
                                start = node.lineno - 1
                                end = node.end_lineno if hasattr(node, 'end_lineno') else start + 50
                                lines = content.split('\n')
                                return '\n'.join(lines[start:end])
                except:
                    pass
            
            # Return truncated file content
            if len(content) > 4000:
                content = content[:4000] + "\n# ... (truncated)"
            return content
            
        except Exception as e:
            self.log_warning(f"Failed to read code file: {e}")
            return ""
    
    async def _generate_validation_test(
        self,
        mapping: MappingResult,
        repo_name: str,
        actual_code: str,
        dependencies: Dict[str, Any],
        import_info: Dict[str, Any] = None
    ) -> str:
        """Generate a test that validates the paper concept using actual repo code."""
        language = dependencies.get("language", "python")
        import_info = import_info or {}

        # Build import hint for the prompt
        import_hint = ""
        if import_info.get("import_statement"):
            import_hint = f"\nUse this import: {import_info['import_statement']}"

        prompt = CODING_AGENT_TEST_GENERATION_PROMPT.format(
            language=language,
            concept_name=mapping.concept_name,
            concept_description=mapping.concept_description,
            code_element=mapping.code_element,
            code_file=mapping.code_file,
            repo_name=repo_name,
            actual_code=actual_code or "# Code not available",
            packages=", ".join(dependencies.get("packages", [])[:15])
        )

        # Add import hint to prompt
        if import_hint:
            prompt += import_hint

        try:
            code = await self.llm.generate(
                prompt,
                system_instruction=CODING_AGENT_SYSTEM_PROMPT
            )
            return self._clean_code(code)
        except Exception as e:
            self.log_error(f"Code generation failed: {e}")
            return ""

    async def _execute_with_sandbox_manager(
        self,
        scripts: List[GeneratedScript],
        repo_path: str,
        dependencies: Dict[str, Any],
        output_dir: Path,
        resource_estimate=None
    ) -> List[TestResult]:
        """
        Execute tests using SandboxManager for proper isolation.

        This method uses the unified SandboxManager which selects the best
        available backend (QEMU > Bubblewrap > Docker > Subprocess).
        """
        results: List[TestResult] = []
        language = dependencies.get("language", "python")

        # Build SandboxConfig from resource estimate and dependencies
        config = SandboxConfig(
            language=language,
            timeout_seconds=120,
            network_enabled=False,
        )

        # Apply resource limits from estimate if available
        if resource_estimate:
            config.memory_limit_mb = int(resource_estimate.memory_gb * 1024)
            config.cpu_limit = max(1.0, resource_estimate.complexity_score * 4)

        # Add repo path as read-only for imports
        if repo_path:
            config.read_only_paths.append(repo_path)
            config.env_vars["PYTHONPATH"] = repo_path

        # Set up working directory
        scripts_dir = output_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        config.read_write_paths.append(str(scripts_dir))

        self.log_info(
            f"Executing {len(scripts)} scripts via SandboxManager "
            f"(best backend: {self._sandbox_manager.best_backend})"
        )

        # Execute each script
        for script in scripts:
            if not script.syntax_valid:
                results.append(TestResult(
                    concept=script.concept,
                    code_element=script.code_element,
                    success=False,
                    error="Syntax validation failed",
                    stdout="",
                    stderr=script.validation_error or "",
                    execution_time=0,
                    isolation_level="none"
                ))
                continue

            script_code = script.code
            concept = script.concept

            try:
                # Execute using SandboxManager
                exec_result: ExecutionResult = await self._sandbox_manager.execute_code(
                    code=script_code,
                    language=language,
                    config=config
                )

                results.append(TestResult(
                    concept=concept,
                    code_element=script.code_element,
                    success=exec_result.success,
                    stdout=exec_result.stdout,
                    stderr=exec_result.stderr,
                    return_code=exec_result.exit_code,
                    execution_time=exec_result.execution_time,
                    isolation_level=exec_result.isolation_level.value,
                    error=exec_result.error or ""
                ))

                if exec_result.success:
                    self.log_info(f"Script for '{concept}' passed")
                else:
                    self.log_warning(
                        f"Script for '{concept}' failed: {exec_result.error or exec_result.stderr[:200]}"
                    )

            except Exception as e:
                self.log_error(f"Execution error for '{concept}': {e}")
                results.append(TestResult(
                    concept=concept,
                    code_element=script.code_element,
                    success=False,
                    error=str(e),
                    stdout="",
                    stderr=str(e),
                    execution_time=0,
                    isolation_level="none"
                ))

        return results

    async def _execute_in_docker_sandbox(
        self,
        scripts: List[GeneratedScript],
        repo_path: str,
        dependencies: Dict[str, Any],
        output_dir: Path
    ) -> List[TestResult]:
        """Execute tests in Docker sandbox with repo mounted and deps installed."""
        results: List[TestResult] = []

        # Create scripts directory
        scripts_dir = output_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        # Write scripts to disk
        for script in scripts:
            if script.syntax_valid:
                script_path = scripts_dir / script.file_name
                script_path.write_text(script.code)

        # Create Dockerfile for sandbox
        dockerfile_content = self._generate_dockerfile(dependencies)
        dockerfile_path = output_dir / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)

        # Build sandbox image
        image_name = f"paper-validator-{os.getpid()}"
        self.log_info("Building Docker sandbox image...")

        try:
            build_process = await asyncio.create_subprocess_exec(
                "docker", "build", "-t", image_name, "-f", str(dockerfile_path), str(output_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(build_process.communicate(), timeout=300)

            if build_process.returncode != 0:
                self.log_error(f"Docker build failed: {stderr.decode()}")
                return await self._execute_in_venv_sandbox(scripts, repo_path, dependencies, output_dir)

            self.log_info("Docker sandbox ready")

        except asyncio.TimeoutError:
            self.log_error("Docker build timed out")
            return []
        except Exception as e:
            self.log_error(f"Docker build error: {e}")
            return await self._execute_in_venv_sandbox(scripts, repo_path, dependencies, output_dir)

        # Execute each test in the sandbox
        for script in scripts:
            if not script.syntax_valid:
                results.append(TestResult(
                    concept=script.concept,
                    code_element=script.code_element,
                    success=False,
                    error=f"Syntax error: {script.validation_error}",
                    stdout="",
                    stderr=""
                ))
                continue

            script_name = script.file_name
            self.log_info(f"Executing: {script_name}")

            result_dict = await self._run_in_docker(
                image_name, scripts_dir, repo_path, script_name, output_dir,
                language=dependencies.get("language", "python")
            )
            results.append(TestResult(
                concept=script.concept,
                code_element=script.code_element,
                success=result_dict.get("success", False),
                stdout=result_dict.get("stdout", ""),
                stderr=result_dict.get("stderr", ""),
                execution_time=result_dict.get("execution_time", 0),
                return_code=result_dict.get("return_code", -1),
                output_files=result_dict.get("output_files", []),
                error=result_dict.get("error", "")
            ))

        # Cleanup image
        try:
            subprocess.run(["docker", "rmi", "-f", image_name], capture_output=True)
        except:
            pass

        success_count = sum(1 for r in results if r.success)
        self.log_info(f"Execution complete: {success_count}/{len(results)} succeeded")

        return results
    
    def _generate_dockerfile(self, dependencies: Dict[str, Any]) -> str:
        """Generate Dockerfile for sandbox environment based on language."""
        language = dependencies.get("language", "python")
        packages = dependencies.get("packages", [])
        
        if language == "python":
            return self._generate_python_dockerfile(packages)
        elif language == "julia":
            return self._generate_julia_dockerfile(packages)
        elif language == "r":
            return self._generate_r_dockerfile(packages)
        elif language == "matlab":
            return self._generate_octave_dockerfile(packages)
        else:
            return self._generate_python_dockerfile(packages)
    
    def _generate_python_dockerfile(self, packages: List[str]) -> str:
        """Generate Dockerfile for Python environment."""
        # Filter problematic packages
        skip_packages = {"tensorflow-gpu", "torch", "torchvision", "jax", "jaxlib"}
        safe_packages = [p for p in packages[:30] if p.lower().split('[')[0] not in skip_packages]
        pip_install = " ".join(safe_packages) if safe_packages else ""
        
        install_cmds = [
            "RUN pip install --no-cache-dir --upgrade pip",
            "RUN pip install --no-cache-dir numpy scipy matplotlib pandas scikit-learn || true"
        ]
        if pip_install:
            install_cmds.append(f"RUN pip install --no-cache-dir {pip_install} || true")
        
        return f'''FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc g++ make git libhdf5-dev libopenblas-dev \\
    && rm -rf /var/lib/apt/lists/*

{chr(10).join(install_cmds)}

ENV PYTHONPATH="/repo:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

CMD ["python"]
'''
    
    def _generate_julia_dockerfile(self, packages: List[str]) -> str:
        """Generate Dockerfile for Julia environment."""
        # Install packages via Pkg
        pkg_installs = ""
        if packages:
            pkg_list = ", ".join([f'"{p}"' for p in packages[:20]])
            pkg_installs = f'RUN julia -e \'using Pkg; Pkg.add([{pkg_list}])\' || true'
        
        return f'''FROM julia:1.10

WORKDIR /workspace

# Install common scientific packages
RUN julia -e 'using Pkg; Pkg.add(["LinearAlgebra", "Statistics", "Plots", "DataFrames"])' || true

{pkg_installs}

ENV JULIA_DEPOT_PATH="/root/.julia"
ENV JULIA_LOAD_PATH="/repo:@:@stdlib"

CMD ["julia"]
'''
    
    def _generate_r_dockerfile(self, packages: List[str]) -> str:
        """Generate Dockerfile for R environment."""
        pkg_installs = ""
        if packages:
            pkg_list = ", ".join([f'"{p}"' for p in packages[:20]])
            pkg_installs = f'RUN Rscript -e \'install.packages(c({pkg_list}), repos="https://cran.r-project.org")\' || true'
        
        return f'''FROM r-base:4.3.0

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \\
    libcurl4-openssl-dev libssl-dev libxml2-dev \\
    && rm -rf /var/lib/apt/lists/*

# Install common packages
RUN Rscript -e 'install.packages(c("ggplot2", "dplyr", "tidyr", "data.table"), repos="https://cran.r-project.org")' || true

{pkg_installs}

ENV R_LIBS_USER="/repo"

CMD ["Rscript"]
'''
    
    def _generate_octave_dockerfile(self, packages: List[str]) -> str:
        """Generate Dockerfile for Octave (MATLAB alternative) environment."""
        return '''FROM gnuoctave/octave:8.4.0

WORKDIR /workspace

# Install common packages from Octave Forge
RUN octave --eval "pkg install -forge control signal image statistics" || true

ENV OCTAVE_PATH="/repo"

CMD ["octave", "--no-gui"]
'''
    
    async def _run_in_docker(
        self,
        image_name: str,
        scripts_dir: Path,
        repo_path: str,
        script_name: str,
        output_dir: Path,
        language: str = "python"
    ) -> Dict[str, Any]:
        """Run a single test script in Docker container."""
        result = {
            "success": False,
            "stdout": "",
            "stderr": "",
            "execution_time": 0,
            "output_files": []
        }
        
        import time
        start_time = time.time()
        
        # Get the run command for this language
        run_commands = {
            "python": ["python"],
            "julia": ["julia"],
            "r": ["Rscript"],
            "matlab": ["octave", "--no-gui"]
        }
        run_cmd = run_commands.get(language, ["python"])
        
        # Get environment variable for module path
        path_vars = {
            "python": ("PYTHONPATH", "/repo"),
            "julia": ("JULIA_LOAD_PATH", "/repo:@:@stdlib"),
            "r": ("R_LIBS_USER", "/repo"),
            "matlab": ("OCTAVE_PATH", "/repo")
        }
        
        # Build docker run command
        cmd = [
            "docker", "run",
            "--rm",
            "-v", f"{scripts_dir}:/scripts:ro",
            "-v", f"{output_dir}:/outputs",
            "-w", "/outputs",
            "--cpus", "2",
            "--memory", "2g",
        ]
        
        # Mount repository if available
        if repo_path and Path(repo_path).exists():
            cmd.extend(["-v", f"{repo_path}:/repo:ro"])
            env_var, env_val = path_vars.get(language, ("PYTHONPATH", "/repo"))
            cmd.extend(["-e", f"{env_var}={env_val}"])
        
        cmd.append(image_name)
        cmd.extend(run_cmd)
        cmd.append(f"/scripts/{script_name}")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=120
            )
            
            result["stdout"] = stdout.decode('utf-8', errors='ignore')
            result["stderr"] = stderr.decode('utf-8', errors='ignore')
            result["success"] = process.returncode == 0
            result["return_code"] = process.returncode
            
        except asyncio.TimeoutError:
            result["error"] = "Execution timed out (120s)"
            result["stderr"] = "Timeout"
        except Exception as e:
            result["error"] = str(e)
            result["stderr"] = str(e)
        
        result["execution_time"] = time.time() - start_time
        
        # Collect output files
        for f in output_dir.glob("*.png"):
            result["output_files"].append(str(f))
        for f in output_dir.glob("*.svg"):
            result["output_files"].append(str(f))
        for f in output_dir.glob("*.pdf"):
            result["output_files"].append(str(f))
        
        return result
    
    async def _execute_in_venv_sandbox(
        self,
        scripts: List[GeneratedScript],
        repo_path: str,
        dependencies: Dict[str, Any],
        output_dir: Path
    ) -> List[TestResult]:
        """Fallback: Execute in virtual environment sandbox."""
        results: List[TestResult] = []

        self.log_info("Using virtual environment sandbox (Docker unavailable)")

        # Create venv
        venv_dir = output_dir / "venv"
        scripts_dir = output_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        try:
            # Create virtual environment
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

            # Get pip path
            if sys.platform == "win32":
                pip_path = venv_dir / "Scripts" / "pip"
                python_path = venv_dir / "Scripts" / "python"
            else:
                pip_path = venv_dir / "bin" / "pip"
                python_path = venv_dir / "bin" / "python"

            # Install base dependencies
            subprocess.run([str(pip_path), "install", "--quiet", "numpy", "scipy", "matplotlib"],
                          capture_output=True)

            # Install repo dependencies
            packages = dependencies.get("packages", [])[:20]
            if packages:
                subprocess.run([str(pip_path), "install", "--quiet"] + packages,
                              capture_output=True)

            # Install repo itself if setup.py exists
            if repo_path and (Path(repo_path) / "setup.py").exists():
                subprocess.run([str(pip_path), "install", "-e", repo_path],
                              capture_output=True)

        except Exception as e:
            self.log_error(f"Venv setup failed: {e}")
            python_path = Path(sys.executable)

        # Write and execute scripts
        for script in scripts:
            if not script.syntax_valid:
                results.append(TestResult(
                    concept=script.concept,
                    code_element=script.code_element,
                    success=False,
                    error=f"Syntax error: {script.validation_error}"
                ))
                continue

            script_path = scripts_dir / script.file_name
            script_path.write_text(script.code)

            self.log_info(f"Executing: {script.file_name}")

            result_dict = await self._run_in_venv(python_path, script_path, repo_path, output_dir)
            results.append(TestResult(
                concept=script.concept,
                code_element=script.code_element,
                success=result_dict.get("success", False),
                stdout=result_dict.get("stdout", ""),
                stderr=result_dict.get("stderr", ""),
                execution_time=result_dict.get("execution_time", 0),
                return_code=result_dict.get("return_code", -1),
                output_files=result_dict.get("output_files", []),
                error=result_dict.get("error", "")
            ))

        success_count = sum(1 for r in results if r.success)
        self.log_info(f"Execution complete: {success_count}/{len(results)} succeeded")

        return results
    
    async def _run_in_venv(
        self,
        python_path: Path,
        script_path: Path,
        repo_path: str,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Run script in virtual environment."""
        result = {
            "success": False,
            "stdout": "",
            "stderr": "",
            "execution_time": 0,
            "output_files": []
        }
        
        import time
        start_time = time.time()
        
        env = os.environ.copy()
        if repo_path:
            env["PYTHONPATH"] = repo_path + os.pathsep + env.get("PYTHONPATH", "")
        
        try:
            process = await asyncio.create_subprocess_exec(
                str(python_path), str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(output_dir),
                env=env
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=60
            )
            
            result["stdout"] = stdout.decode('utf-8', errors='ignore')
            result["stderr"] = stderr.decode('utf-8', errors='ignore')
            result["success"] = process.returncode == 0
            result["return_code"] = process.returncode
            
        except asyncio.TimeoutError:
            result["error"] = "Execution timed out (60s)"
        except Exception as e:
            result["error"] = str(e)
        
        result["execution_time"] = time.time() - start_time
        
        return result
    
    def _clean_code(self, code: str) -> str:
        """Clean up generated code."""
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()
    
    def _validate_syntax(self, code: str, language: str = "python") -> Tuple[bool, Optional[str]]:
        """Validate code syntax for the given language."""
        if language == "python":
            return self._validate_python_syntax(code)
        elif language == "julia":
            return self._validate_julia_syntax(code)
        elif language == "r":
            return self._validate_r_syntax(code)
        else:
            # For languages we can't validate locally, assume valid
            return True, None
    
    def _validate_python_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate Python syntax using AST."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
    
    def _validate_julia_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Basic Julia syntax validation."""
        # Check for common syntax issues
        errors = []
        
        # Check balanced keywords
        opens = code.count("function ") + code.count("if ") + code.count("for ") + code.count("while ")
        ends = code.count("\nend") + code.count(" end") + (1 if code.strip().endswith("end") else 0)
        
        if opens > ends:
            errors.append("Missing 'end' keyword")
        
        # Check string quotes are balanced
        if code.count('"') % 2 != 0:
            errors.append("Unbalanced double quotes")
        
        return (len(errors) == 0, "; ".join(errors) if errors else None)
    
    def _validate_r_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Basic R syntax validation."""
        errors = []
        
        # Check balanced braces
        if code.count('{') != code.count('}'):
            errors.append("Unbalanced curly braces")
        if code.count('(') != code.count(')'):
            errors.append("Unbalanced parentheses")
        
        return (len(errors) == 0, "; ".join(errors) if errors else None)
    
    async def _fix_code(self, code: str, error: str, error_type: str) -> str:
        """Attempt to fix code using LLM."""
        self.log_info(f"Attempting to fix {error_type}: {error}")
        
        prompt = CODING_AGENT_DEBUG_PROMPT.format(
            code=code,
            error=error,
            error_type=error_type
        )
        
        try:
            fixed = await self.llm.generate(
                prompt,
                system_instruction=CODING_AGENT_SYSTEM_PROMPT
            )
            return self._clean_code(fixed)
        except:
            return code
    
    def _safe_filename(self, name: str) -> str:
        """Convert concept name to safe filename."""
        safe = "".join(c if c.isalnum() else "_" for c in name.lower())
        return safe[:30]