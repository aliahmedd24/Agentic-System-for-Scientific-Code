"""
Resource Estimator - Estimates computational requirements for code execution.

This module provides utilities for estimating:
- Memory requirements
- GPU needs
- Expected execution time
- Dependency complexity
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum


class ComputeLevel(Enum):
    """Compute requirement levels."""
    MINIMAL = "minimal"      # < 1GB RAM, no GPU, < 1 min
    LOW = "low"              # 1-4GB RAM, no GPU, 1-10 min
    MEDIUM = "medium"        # 4-16GB RAM, optional GPU, 10-60 min
    HIGH = "high"            # 16-64GB RAM, GPU recommended, 1-6 hours
    EXTREME = "extreme"      # > 64GB RAM, multi-GPU, > 6 hours


@dataclass
class ResourceEstimate:
    """Estimated resource requirements."""
    compute_level: ComputeLevel = ComputeLevel.LOW
    memory_gb: float = 2.0
    gpu_required: bool = False
    gpu_memory_gb: float = 0.0
    estimated_time_minutes: float = 5.0
    disk_space_gb: float = 1.0
    dependency_count: int = 0
    complexity_score: float = 0.5
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "compute_level": self.compute_level.value,
            "memory_gb": self.memory_gb,
            "gpu_required": self.gpu_required,
            "gpu_memory_gb": self.gpu_memory_gb,
            "estimated_time_minutes": self.estimated_time_minutes,
            "disk_space_gb": self.disk_space_gb,
            "dependency_count": self.dependency_count,
            "complexity_score": self.complexity_score,
            "warnings": self.warnings,
            "recommendations": self.recommendations
        }

    def is_feasible(self, available_memory_gb: float = 16.0, has_gpu: bool = False) -> bool:
        """Check if resources are available to run this task."""
        if self.memory_gb > available_memory_gb:
            return False
        if self.gpu_required and not has_gpu:
            return False
        return True


# Library-specific resource requirements
LIBRARY_REQUIREMENTS = {
    # Deep Learning frameworks (high memory, GPU beneficial)
    "torch": {"memory_gb": 4.0, "gpu_beneficial": True, "gpu_memory_gb": 4.0},
    "pytorch": {"memory_gb": 4.0, "gpu_beneficial": True, "gpu_memory_gb": 4.0},
    "tensorflow": {"memory_gb": 4.0, "gpu_beneficial": True, "gpu_memory_gb": 4.0},
    "keras": {"memory_gb": 3.0, "gpu_beneficial": True, "gpu_memory_gb": 3.0},
    "jax": {"memory_gb": 4.0, "gpu_beneficial": True, "gpu_memory_gb": 4.0},
    "flax": {"memory_gb": 4.0, "gpu_beneficial": True, "gpu_memory_gb": 4.0},

    # Transformers (very high memory)
    "transformers": {"memory_gb": 8.0, "gpu_beneficial": True, "gpu_memory_gb": 8.0},
    "huggingface": {"memory_gb": 8.0, "gpu_beneficial": True, "gpu_memory_gb": 8.0},

    # Computer Vision
    "opencv": {"memory_gb": 2.0, "gpu_beneficial": False},
    "cv2": {"memory_gb": 2.0, "gpu_beneficial": False},
    "pillow": {"memory_gb": 1.0, "gpu_beneficial": False},
    "torchvision": {"memory_gb": 4.0, "gpu_beneficial": True, "gpu_memory_gb": 4.0},

    # Scientific computing
    "numpy": {"memory_gb": 1.0, "gpu_beneficial": False},
    "scipy": {"memory_gb": 2.0, "gpu_beneficial": False},
    "pandas": {"memory_gb": 2.0, "gpu_beneficial": False},
    "sklearn": {"memory_gb": 2.0, "gpu_beneficial": False},
    "scikit-learn": {"memory_gb": 2.0, "gpu_beneficial": False},

    # Large-scale ML
    "xgboost": {"memory_gb": 4.0, "gpu_beneficial": True, "gpu_memory_gb": 2.0},
    "lightgbm": {"memory_gb": 4.0, "gpu_beneficial": True, "gpu_memory_gb": 2.0},
    "catboost": {"memory_gb": 4.0, "gpu_beneficial": True, "gpu_memory_gb": 2.0},

    # Distributed computing
    "dask": {"memory_gb": 8.0, "gpu_beneficial": False},
    "ray": {"memory_gb": 8.0, "gpu_beneficial": False},
    "spark": {"memory_gb": 16.0, "gpu_beneficial": False},

    # NLP
    "spacy": {"memory_gb": 4.0, "gpu_beneficial": True, "gpu_memory_gb": 2.0},
    "nltk": {"memory_gb": 2.0, "gpu_beneficial": False},
    "gensim": {"memory_gb": 4.0, "gpu_beneficial": False},
}

# Code patterns that indicate high resource usage
HIGH_RESOURCE_PATTERNS = [
    (r'\.cuda\(\)', "GPU usage detected", True, 4.0),
    (r'\.to\([\'"]cuda[\'"]\)', "GPU usage detected", True, 4.0),
    (r'torch\.nn\.DataParallel', "Multi-GPU training", True, 8.0),
    (r'torch\.distributed', "Distributed training", True, 16.0),
    (r'large_?model|huge_?model', "Large model indicated", True, 16.0),
    (r'batch_size\s*[=:]\s*(\d+)', "Batch size", False, 0.0),
    (r'num_workers\s*[=:]\s*(\d+)', "Multiple workers", False, 2.0),
    (r'load_?in_?8bit|bitsandbytes', "Quantized model", True, 8.0),
    (r'gradient_checkpointing', "Gradient checkpointing", True, 4.0),
]


class ResourceEstimator:
    """
    Estimates computational resources required for code execution.
    """

    def __init__(self):
        self._library_requirements = LIBRARY_REQUIREMENTS.copy()
        self._patterns = HIGH_RESOURCE_PATTERNS.copy()

    def estimate_from_repo(
        self,
        repo_data: Dict[str, Any],
        code_elements: Dict[str, Any] = None
    ) -> ResourceEstimate:
        """
        Estimate resources from repository analysis.

        Args:
            repo_data: Repository analysis from RepoAnalyzerAgent
            code_elements: Optional extracted code elements

        Returns:
            ResourceEstimate with predicted requirements
        """
        estimate = ResourceEstimate()

        # Get dependencies
        dependencies = repo_data.get("dependencies", {})
        if isinstance(dependencies, dict):
            packages = dependencies.get("python", [])
        elif isinstance(dependencies, list):
            packages = dependencies
        else:
            packages = []

        estimate.dependency_count = len(packages)

        # Analyze dependencies
        memory_estimates = []
        gpu_needed = False
        gpu_memory = 0.0

        for pkg in packages:
            pkg_lower = pkg.lower().split("[")[0].split(">=")[0].split("==")[0].strip()
            if pkg_lower in self._library_requirements:
                req = self._library_requirements[pkg_lower]
                memory_estimates.append(req.get("memory_gb", 1.0))
                if req.get("gpu_beneficial", False):
                    gpu_needed = True
                    gpu_memory = max(gpu_memory, req.get("gpu_memory_gb", 0.0))

        # Estimate memory (use max of library requirements, minimum 2GB)
        if memory_estimates:
            estimate.memory_gb = max(2.0, max(memory_estimates))
        else:
            estimate.memory_gb = 2.0

        estimate.gpu_required = gpu_needed
        estimate.gpu_memory_gb = gpu_memory

        # Analyze code complexity
        if code_elements:
            classes = code_elements.get("classes", [])
            functions = code_elements.get("functions", [])
            estimate.complexity_score = min(1.0, (len(classes) + len(functions)) / 100)

            # Estimate time based on complexity
            estimate.estimated_time_minutes = 5.0 + (estimate.complexity_score * 30.0)

        # Determine compute level
        estimate.compute_level = self._determine_compute_level(estimate)

        # Add warnings and recommendations
        self._add_warnings(estimate, packages)

        return estimate

    def estimate_from_code(self, code: str, language: str = "python") -> ResourceEstimate:
        """
        Estimate resources from code content.

        Args:
            code: Source code to analyze
            language: Programming language

        Returns:
            ResourceEstimate with predicted requirements
        """
        estimate = ResourceEstimate()

        if language != "python":
            # Basic estimation for non-Python code
            estimate.memory_gb = 2.0
            estimate.complexity_score = min(1.0, len(code) / 10000)
            return estimate

        # Extract imports
        imports = self._extract_imports(code)
        estimate.dependency_count = len(imports)

        # Analyze imports
        memory_estimates = []
        gpu_needed = False
        gpu_memory = 0.0

        for imp in imports:
            imp_lower = imp.lower()
            for lib_name, req in self._library_requirements.items():
                if lib_name in imp_lower:
                    memory_estimates.append(req.get("memory_gb", 1.0))
                    if req.get("gpu_beneficial", False):
                        gpu_needed = True
                        gpu_memory = max(gpu_memory, req.get("gpu_memory_gb", 0.0))
                    break

        # Check for high-resource patterns
        for pattern, message, needs_gpu, extra_memory in self._patterns:
            if re.search(pattern, code, re.IGNORECASE):
                estimate.warnings.append(message)
                if needs_gpu:
                    gpu_needed = True
                if extra_memory > 0:
                    memory_estimates.append(extra_memory)

        # Set estimates
        if memory_estimates:
            estimate.memory_gb = max(2.0, max(memory_estimates))
        else:
            estimate.memory_gb = 2.0

        estimate.gpu_required = gpu_needed
        estimate.gpu_memory_gb = gpu_memory

        # Estimate complexity from code length and structure
        lines = code.split('\n')
        estimate.complexity_score = min(1.0, len(lines) / 500)

        # Estimate time
        base_time = 5.0
        if gpu_needed:
            base_time = 15.0
        estimate.estimated_time_minutes = base_time + (estimate.complexity_score * 30.0)

        # Determine compute level
        estimate.compute_level = self._determine_compute_level(estimate)

        return estimate

    def estimate_for_mapping(
        self,
        mapping: Dict[str, Any],
        repo_data: Dict[str, Any]
    ) -> ResourceEstimate:
        """
        Estimate resources for a specific concept-code mapping.

        Args:
            mapping: Concept-to-code mapping
            repo_data: Repository analysis data

        Returns:
            ResourceEstimate for executing this mapping's tests
        """
        # Start with base repo estimate
        estimate = self.estimate_from_repo(repo_data)

        # Adjust based on mapping specifics
        code_element = mapping.get("code_element", "")
        concept_type = mapping.get("concept_type", "")

        # Neural network concepts typically need more resources
        if any(keyword in code_element.lower() for keyword in
               ["network", "model", "train", "neural", "layer", "transformer"]):
            estimate.memory_gb = max(estimate.memory_gb, 8.0)
            estimate.gpu_required = True
            estimate.gpu_memory_gb = max(estimate.gpu_memory_gb, 4.0)

        # Algorithm implementations might need less
        if concept_type == "algorithm":
            estimate.memory_gb = min(estimate.memory_gb, 4.0)

        # Update compute level
        estimate.compute_level = self._determine_compute_level(estimate)

        return estimate

    def _determine_compute_level(self, estimate: ResourceEstimate) -> ComputeLevel:
        """Determine compute level from estimates."""
        if estimate.memory_gb >= 64 or estimate.gpu_memory_gb >= 40:
            return ComputeLevel.EXTREME
        elif estimate.memory_gb >= 16 or estimate.gpu_memory_gb >= 12:
            return ComputeLevel.HIGH
        elif estimate.memory_gb >= 4 or estimate.gpu_required:
            return ComputeLevel.MEDIUM
        elif estimate.memory_gb >= 1:
            return ComputeLevel.LOW
        else:
            return ComputeLevel.MINIMAL

    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements from Python code."""
        imports = []
        lines = code.split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith("import "):
                # import x, y, z
                parts = line[7:].split(",")
                for part in parts:
                    part = part.split(" as ")[0].strip()
                    imports.append(part.split(".")[0])
            elif line.startswith("from "):
                # from x import y
                match = re.match(r"from\s+(\S+)\s+import", line)
                if match:
                    imports.append(match.group(1).split(".")[0])

        return list(set(imports))

    def _add_warnings(self, estimate: ResourceEstimate, packages: List[str]) -> None:
        """Add warnings and recommendations based on estimate."""
        if estimate.gpu_required and estimate.gpu_memory_gb >= 8:
            estimate.warnings.append(
                f"Requires GPU with {estimate.gpu_memory_gb}GB+ VRAM"
            )

        if estimate.memory_gb >= 16:
            estimate.warnings.append(
                f"High memory requirement: {estimate.memory_gb}GB RAM"
            )

        if estimate.compute_level in [ComputeLevel.HIGH, ComputeLevel.EXTREME]:
            estimate.recommendations.append(
                "Consider using cloud compute (AWS, GCP, Colab Pro)"
            )

        if any(pkg.lower() in ["torch", "tensorflow", "jax"] for pkg in packages):
            if not estimate.gpu_required:
                estimate.recommendations.append(
                    "GPU recommended for faster execution"
                )

        if estimate.dependency_count > 20:
            estimate.warnings.append(
                f"Many dependencies ({estimate.dependency_count}), installation may take time"
            )


def estimate_resources(
    repo_data: Dict[str, Any] = None,
    code: str = None,
    mapping: Dict[str, Any] = None
) -> ResourceEstimate:
    """
    Convenience function to estimate resources.

    Provide one of: repo_data, code, or mapping+repo_data
    """
    estimator = ResourceEstimator()

    if mapping and repo_data:
        return estimator.estimate_for_mapping(mapping, repo_data)
    elif repo_data:
        return estimator.estimate_from_repo(repo_data)
    elif code:
        return estimator.estimate_from_code(code)
    else:
        return ResourceEstimate()
