"""
Coding Agent - Generates and executes test code for paper concepts.
"""

import os
import ast
import sys
import asyncio
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from .base_agent import BaseAgent
from core.llm_client import LLMClient
from core.knowledge_graph import KnowledgeGraph, NodeType, EdgeType
from core.agent_prompts import (
    CODING_AGENT_SYSTEM_PROMPT,
    CODING_AGENT_TEST_GENERATION_PROMPT,
    CODING_AGENT_VISUALIZATION_PROMPT,
    CODING_AGENT_DEBUG_PROMPT
)
from core.error_handling import (
    logger, LogCategory, ErrorCategory, create_error, AgentError
)


class CodingAgent(BaseAgent):
    """
    Generates and executes code to demonstrate paper concepts.
    
    Features:
    - Test script generation
    - Syntax validation
    - Sandbox execution (Docker or subprocess)
    - Self-correction on errors
    - Visualization generation
    """
    
    def __init__(self, llm_client: LLMClient, max_retries: int = 3):
        super().__init__(llm_client, name="CodingAgent")
        self.max_retries = max_retries
        self._docker_available = self._check_docker()
    
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
            return available
        except:
            self.log_warning("Docker not available, using subprocess execution")
            return False
    
    async def process(
        self,
        mappings: List[Dict[str, Any]],
        repo_data: Dict[str, Any],
        kg: KnowledgeGraph
    ) -> List[Dict[str, Any]]:
        """Main processing method."""
        return await self.generate_tests(mappings, repo_data, kg)
    
    async def generate_tests(
        self,
        mappings: Optional[List[Dict[str, Any]]],
        repo_data: Optional[Dict[str, Any]],
        kg: KnowledgeGraph
    ) -> List[Dict[str, Any]]:
        """
        Generate test scripts for concept-code mappings.
        
        Args:
            mappings: List of concept-code mappings
            repo_data: Repository analysis data
            kg: Knowledge graph
            
        Returns:
            List of generated test scripts with metadata
        """
        if not mappings:
            self.log_warning("No mappings provided for code generation")
            return []
        
        generated_tests = []
        repo_path = repo_data.get("_repo_path", "") if repo_data else ""
        
        # Get dependencies
        dependencies = []
        if repo_data:
            deps = repo_data.get("dependencies", {})
            dependencies = deps.get("python", [])
        
        # Generate tests for high-confidence mappings
        high_confidence_mappings = [
            m for m in mappings if m.get("confidence", 0) >= 0.5
        ][:5]  # Limit to top 5
        
        self.log_info(f"Generating tests for {len(high_confidence_mappings)} mappings")
        
        for i, mapping in enumerate(high_confidence_mappings):
            self.log_info(f"Generating test {i+1}/{len(high_confidence_mappings)}: {mapping.get('concept_name', 'Unknown')}")
            
            try:
                test_code = await self._generate_test_code(
                    mapping, repo_path, dependencies
                )
                
                if test_code:
                    # Validate syntax
                    is_valid, error = self._validate_python_syntax(test_code)
                    
                    if not is_valid:
                        # Try to fix
                        test_code = await self._fix_code(test_code, error)
                        is_valid, error = self._validate_python_syntax(test_code)
                    
                    generated_tests.append({
                        "concept": mapping.get("concept_name", ""),
                        "code_element": mapping.get("code_element", ""),
                        "confidence": mapping.get("confidence", 0),
                        "code": test_code,
                        "syntax_valid": is_valid,
                        "syntax_error": error if not is_valid else None,
                        "file_name": f"test_{i+1}_{self._safe_filename(mapping.get('concept_name', 'concept'))}.py"
                    })
                    
            except Exception as e:
                self.log_error(f"Failed to generate test for {mapping.get('concept_name')}: {e}")
        
        self.log_info(f"Generated {len(generated_tests)} test scripts")
        return generated_tests
    
    async def _generate_test_code(
        self,
        mapping: Dict[str, Any],
        repo_path: str,
        dependencies: List[str]
    ) -> str:
        """Generate test code using LLM."""
        # Get code context if available
        code_file = mapping.get("code_file", "")
        mapped_code = ""
        
        if repo_path and code_file:
            full_path = Path(repo_path) / code_file
            if full_path.exists():
                try:
                    mapped_code = full_path.read_text(encoding='utf-8', errors='ignore')
                    # Truncate if too long
                    if len(mapped_code) > 5000:
                        mapped_code = mapped_code[:5000] + "\n# ... (truncated)"
                except:
                    pass
        
        prompt = CODING_AGENT_TEST_GENERATION_PROMPT.format(
            concept_name=mapping.get("concept_name", "Unknown"),
            concept_description=mapping.get("concept_description", ""),
            mapped_code=mapped_code or "Code not available",
            repo_path=repo_path or "./",
            dependencies=", ".join(dependencies[:10]) or "None specified"
        )
        
        try:
            code = await self.llm.generate(
                prompt,
                system_instruction=CODING_AGENT_SYSTEM_PROMPT
            )
            
            # Clean up code
            code = self._clean_code(code)
            return code
            
        except Exception as e:
            self.log_error(f"Code generation failed: {e}")
            return ""
    
    def _clean_code(self, code: str) -> str:
        """Clean up generated code."""
        # Remove markdown code blocks
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        
        if code.endswith("```"):
            code = code[:-3]
        
        return code.strip()
    
    def _validate_python_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate Python syntax using AST."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
    
    async def _fix_code(self, code: str, error: str) -> str:
        """Attempt to fix code using LLM."""
        self.log_info(f"Attempting to fix syntax error: {error}")
        
        prompt = CODING_AGENT_DEBUG_PROMPT.format(
            code=code,
            error=error,
            error_type="SyntaxError"
        )
        
        try:
            fixed_code = await self.llm.generate(
                prompt,
                system_instruction=CODING_AGENT_SYSTEM_PROMPT
            )
            return self._clean_code(fixed_code)
        except:
            return code
    
    def _safe_filename(self, name: str) -> str:
        """Convert name to safe filename."""
        import re
        # Remove non-alphanumeric characters
        safe = re.sub(r'[^\w\s-]', '', name.lower())
        # Replace spaces with underscores
        safe = re.sub(r'[\s]+', '_', safe)
        return safe[:30]
    
    async def execute_tests(
        self,
        generated_tests: Optional[List[Dict[str, Any]]],
        output_dir: Path,
        use_docker: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Execute generated test scripts.
        
        Args:
            generated_tests: List of generated tests
            output_dir: Directory for outputs
            use_docker: Whether to use Docker sandbox
            
        Returns:
            Tuple of (execution results, visualization paths)
        """
        if not generated_tests:
            return [], []
        
        results = []
        visualizations = []
        
        # Create scripts directory
        scripts_dir = output_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Create visualizations directory
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        for test in generated_tests:
            if not test.get("syntax_valid", False):
                results.append({
                    "concept": test.get("concept"),
                    "success": False,
                    "error": f"Syntax error: {test.get('syntax_error')}",
                    "stdout": "",
                    "stderr": "",
                    "execution_time": 0
                })
                continue
            
            # Save script
            script_path = scripts_dir / test["file_name"]
            script_path.write_text(test["code"])
            
            self.log_info(f"Executing: {test['file_name']}")
            
            # Execute
            if use_docker and self._docker_available:
                result = await self._execute_docker(script_path, viz_dir)
            else:
                result = await self._execute_subprocess(script_path, viz_dir)
            
            result["concept"] = test.get("concept")
            result["code_element"] = test.get("code_element")
            result["script_path"] = str(script_path)
            
            results.append(result)
            
            # Collect visualizations
            if result.get("success") and result.get("output_files"):
                for f in result["output_files"]:
                    if f.endswith(('.png', '.jpg', '.svg', '.pdf')):
                        visualizations.append(f)
        
        success_count = sum(1 for r in results if r.get("success"))
        self.log_info(f"Execution complete: {success_count}/{len(results)} succeeded")
        
        return results, visualizations
    
    async def _execute_subprocess(
        self,
        script_path: Path,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Execute script using subprocess."""
        result = {
            "success": False,
            "stdout": "",
            "stderr": "",
            "execution_time": 0,
            "output_files": []
        }
        
        try:
            import time
            start_time = time.time()
            
            # Run with timeout
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(output_dir)
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=60  # 60 second timeout
                )
                
                result["stdout"] = stdout.decode('utf-8', errors='ignore')
                result["stderr"] = stderr.decode('utf-8', errors='ignore')
                result["success"] = process.returncode == 0
                result["return_code"] = process.returncode
                
            except asyncio.TimeoutError:
                process.kill()
                result["error"] = "Execution timed out (60s)"
                result["stderr"] = "Timeout"
            
            result["execution_time"] = time.time() - start_time
            
            # Check for output files
            for f in output_dir.glob("*.png"):
                result["output_files"].append(str(f))
            for f in output_dir.glob("*.svg"):
                result["output_files"].append(str(f))
            
        except Exception as e:
            result["error"] = str(e)
            result["stderr"] = str(e)
        
        return result
    
    async def _execute_docker(
        self,
        script_path: Path,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Execute script in Docker container."""
        result = {
            "success": False,
            "stdout": "",
            "stderr": "",
            "execution_time": 0,
            "output_files": []
        }
        
        try:
            import time
            start_time = time.time()
            
            # Build Docker command
            cmd = [
                "docker", "run",
                "--rm",
                "-v", f"{script_path.parent}:/scripts:ro",
                "-v", f"{output_dir}:/outputs",
                "--cpus", "1",
                "--memory", "512m",
                "--network", "none",
                "python:3.11-slim",
                "python", f"/scripts/{script_path.name}"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=120
                )
                
                result["stdout"] = stdout.decode('utf-8', errors='ignore')
                result["stderr"] = stderr.decode('utf-8', errors='ignore')
                result["success"] = process.returncode == 0
                result["return_code"] = process.returncode
                
            except asyncio.TimeoutError:
                # Kill container
                subprocess.run(["docker", "kill", "-s", "KILL"], capture_output=True)
                result["error"] = "Execution timed out (120s)"
            
            result["execution_time"] = time.time() - start_time
            
            # Check for output files
            for f in output_dir.glob("*.png"):
                result["output_files"].append(str(f))
            for f in output_dir.glob("*.svg"):
                result["output_files"].append(str(f))
            
        except Exception as e:
            result["error"] = str(e)
            # Fall back to subprocess
            self.log_warning(f"Docker execution failed, falling back to subprocess: {e}")
            return await self._execute_subprocess(script_path, output_dir)
        
        return result
    
    async def generate_visualization(
        self,
        results: Dict[str, Any],
        concept: str,
        output_dir: Path
    ) -> Optional[str]:
        """Generate visualization code for results."""
        prompt = CODING_AGENT_VISUALIZATION_PROMPT.format(
            results=str(results)[:2000],
            concept=concept
        )
        
        try:
            code = await self.llm.generate(
                prompt,
                system_instruction=CODING_AGENT_SYSTEM_PROMPT
            )
            code = self._clean_code(code)
            
            # Validate
            is_valid, _ = self._validate_python_syntax(code)
            if not is_valid:
                return None
            
            # Save and execute
            viz_script = output_dir / f"viz_{self._safe_filename(concept)}.py"
            viz_script.write_text(code)
            
            result = await self._execute_subprocess(viz_script, output_dir)
            
            if result.get("output_files"):
                return result["output_files"][0]
            
        except Exception as e:
            self.log_error(f"Visualization generation failed: {e}")
        
        return None
