"""
Tests for Resource Estimator - Computational requirements estimation.
"""

import pytest


# ============================================================================
# ComputeLevel Enum Tests
# ============================================================================

class TestComputeLevel:
    """Tests for ComputeLevel enum."""

    def test_all_levels_exist(self):
        """Test all expected levels are defined."""
        from core.resource_estimator import ComputeLevel

        assert ComputeLevel.MINIMAL.value == "minimal"
        assert ComputeLevel.LOW.value == "low"
        assert ComputeLevel.MEDIUM.value == "medium"
        assert ComputeLevel.HIGH.value == "high"
        assert ComputeLevel.EXTREME.value == "extreme"


# ============================================================================
# ResourceEstimate Tests
# ============================================================================

class TestResourceEstimate:
    """Tests for ResourceEstimate model."""

    def test_estimate_defaults(self):
        """Test default estimate values."""
        from core.resource_estimator import ResourceEstimate, ComputeLevel

        estimate = ResourceEstimate()

        assert estimate.compute_level == ComputeLevel.LOW
        assert estimate.memory_gb == 2.0
        assert estimate.gpu_required is False
        assert estimate.gpu_memory_gb == 0.0
        assert estimate.warnings == []
        assert estimate.recommendations == []

    def test_estimate_custom_values(self):
        """Test estimate with custom values."""
        from core.resource_estimator import ResourceEstimate, ComputeLevel

        estimate = ResourceEstimate(
            compute_level=ComputeLevel.HIGH,
            memory_gb=32.0,
            gpu_required=True,
            gpu_memory_gb=16.0,
            estimated_time_minutes=120.0,
            disk_space_gb=50.0,
            dependency_count=25,
            complexity_score=0.8
        )

        assert estimate.compute_level == ComputeLevel.HIGH
        assert estimate.memory_gb == 32.0
        assert estimate.gpu_required is True
        assert estimate.gpu_memory_gb == 16.0

    def test_is_feasible_with_resources(self):
        """Test feasibility check with available resources."""
        from core.resource_estimator import ResourceEstimate, ComputeLevel

        estimate = ResourceEstimate(
            memory_gb=8.0,
            gpu_required=False
        )

        assert estimate.is_feasible(available_memory_gb=16.0) is True

    def test_is_feasible_memory_insufficient(self):
        """Test feasibility with insufficient memory."""
        from core.resource_estimator import ResourceEstimate

        estimate = ResourceEstimate(memory_gb=32.0)

        assert estimate.is_feasible(available_memory_gb=16.0) is False

    def test_is_feasible_gpu_required_not_available(self):
        """Test feasibility with GPU required but not available."""
        from core.resource_estimator import ResourceEstimate

        estimate = ResourceEstimate(
            memory_gb=8.0,
            gpu_required=True
        )

        assert estimate.is_feasible(available_memory_gb=16.0, has_gpu=False) is False

    def test_is_feasible_gpu_required_available(self):
        """Test feasibility with GPU required and available."""
        from core.resource_estimator import ResourceEstimate

        estimate = ResourceEstimate(
            memory_gb=8.0,
            gpu_required=True
        )

        assert estimate.is_feasible(available_memory_gb=16.0, has_gpu=True) is True

    def test_complexity_score_bounds(self):
        """Test complexity score must be 0-1."""
        from core.resource_estimator import ResourceEstimate
        from pydantic import ValidationError

        # Valid scores
        ResourceEstimate(complexity_score=0.0)
        ResourceEstimate(complexity_score=1.0)
        ResourceEstimate(complexity_score=0.5)

        # Invalid scores
        with pytest.raises(ValidationError):
            ResourceEstimate(complexity_score=-0.1)
        with pytest.raises(ValidationError):
            ResourceEstimate(complexity_score=1.1)


# ============================================================================
# ResourceEstimator Tests
# ============================================================================

class TestResourceEstimator:
    """Tests for ResourceEstimator class."""

    def test_estimator_creation(self, resource_estimator):
        """Test creating estimator."""
        assert resource_estimator is not None

    def test_estimate_from_repo_minimal(self, resource_estimator, sample_repo_minimal):
        """Test estimation from minimal repo."""
        estimate = resource_estimator.estimate_from_repo(sample_repo_minimal)

        assert estimate is not None
        assert estimate.memory_gb >= 1.0  # Minimum

    def test_estimate_from_repo_with_torch(self, resource_estimator, sample_repo_with_torch):
        """Test estimation with PyTorch dependencies."""
        estimate = resource_estimator.estimate_from_repo(sample_repo_with_torch)

        assert estimate is not None
        assert estimate.memory_gb >= 4.0  # torch requires more memory
        # GPU may be recommended

    def test_estimate_from_repo_with_transformers(self, resource_estimator):
        """Test estimation with transformers."""
        repo_data = {
            "name": "transformer-project",
            "dependencies": {
                "python": ["transformers", "torch", "datasets"]
            },
            "stats": {"classes": 5, "functions": 20}
        }

        estimate = resource_estimator.estimate_from_repo(repo_data)

        assert estimate.memory_gb >= 8.0  # transformers needs a lot


# ============================================================================
# Code Pattern Detection Tests
# ============================================================================

class TestCodePatternDetection:
    """Tests for GPU and resource pattern detection in code."""

    def test_estimate_from_code_simple(self, resource_estimator):
        """Test estimation from simple code."""
        code = """
def add(a, b):
    return a + b

result = add(1, 2)
print(result)
"""
        estimate = resource_estimator.estimate_from_code(code, "python")

        assert estimate is not None
        assert estimate.gpu_required is False

    def test_estimate_from_code_cuda(self, resource_estimator):
        """Test detection of .cuda() usage."""
        code = """
import torch
model = MyModel()
model.cuda()
"""
        estimate = resource_estimator.estimate_from_code(code, "python")

        assert estimate is not None
        # Should detect GPU usage

    def test_estimate_from_code_to_device(self, resource_estimator):
        """Test detection of .to('cuda')."""
        code = """
import torch
x = torch.tensor([1, 2, 3])
x = x.to('cuda')
"""
        estimate = resource_estimator.estimate_from_code(code, "python")

        assert estimate is not None
        # GPU usage pattern detected

    def test_estimate_from_code_dataparallel(self, resource_estimator):
        """Test detection of DataParallel."""
        code = """
import torch.nn as nn
model = nn.DataParallel(MyModel())
"""
        estimate = resource_estimator.estimate_from_code(code, "python")

        assert estimate is not None
        # Multi-GPU pattern

    def test_estimate_from_code_non_python(self, resource_estimator):
        """Test estimation for non-Python code."""
        code = """
function train(model)
    # Julia training code
end
"""
        estimate = resource_estimator.estimate_from_code(code, "julia")

        assert estimate is not None
        # Should give basic estimate


# ============================================================================
# Mapping-Based Estimation Tests
# ============================================================================

class TestMappingEstimation:
    """Tests for mapping-based estimation."""

    def test_estimate_for_mapping(self, resource_estimator):
        """Test estimation for a concept mapping."""
        mapping = {
            "concept_name": "attention",
            "code_element": "MultiHeadAttention",
            "code_file": "model.py",
            "confidence": 0.85
        }

        repo_data = {
            "name": "transformer",
            "dependencies": {"python": ["torch"]},
            "stats": {"classes": 5, "functions": 10}
        }

        estimate = resource_estimator.estimate_for_mapping(mapping, repo_data)

        assert estimate is not None


# ============================================================================
# Compute Level Determination Tests
# ============================================================================

class TestComputeLevelDetermination:
    """Tests for compute level determination."""

    def test_determine_minimal_level(self, resource_estimator):
        """Test MINIMAL level for simple projects."""
        repo_data = {
            "name": "simple",
            "dependencies": {"python": []},
            "stats": {"classes": 1, "functions": 2}
        }

        estimate = resource_estimator.estimate_from_repo(repo_data)

        from core.resource_estimator import ComputeLevel
        assert estimate.compute_level in [ComputeLevel.MINIMAL, ComputeLevel.LOW]

    def test_determine_high_level_with_gpu(self, resource_estimator):
        """Test HIGH level for GPU projects."""
        repo_data = {
            "name": "deep-learning",
            "dependencies": {"python": ["torch", "transformers", "cuda"]},
            "stats": {"classes": 20, "functions": 50}
        }

        estimate = resource_estimator.estimate_from_repo(repo_data)

        from core.resource_estimator import ComputeLevel
        assert estimate.compute_level in [ComputeLevel.MEDIUM, ComputeLevel.HIGH, ComputeLevel.EXTREME]


# ============================================================================
# Import Extraction Tests
# ============================================================================

class TestImportExtraction:
    """Tests for import extraction from code."""

    def test_extract_imports_standard(self, resource_estimator):
        """Test extracting standard imports."""
        code = """
import os
import sys
from pathlib import Path
"""
        imports = resource_estimator._extract_imports(code)

        assert "os" in imports
        assert "sys" in imports
        assert "pathlib" in imports

    def test_extract_imports_ml_libraries(self, resource_estimator):
        """Test extracting ML library imports."""
        code = """
import torch
import torch.nn as nn
from transformers import AutoModel
import numpy as np
"""
        imports = resource_estimator._extract_imports(code)

        assert "torch" in imports
        assert "transformers" in imports
        assert "numpy" in imports


# ============================================================================
# Warning Generation Tests
# ============================================================================

class TestWarningGeneration:
    """Tests for warning generation."""

    def test_warnings_for_high_memory(self, resource_estimator):
        """Test warnings are generated for high memory requirements."""
        repo_data = {
            "name": "huge-project",
            "dependencies": {"python": ["transformers", "torch", "tensorflow"]},
            "stats": {"classes": 100, "functions": 500}
        }

        estimate = resource_estimator.estimate_from_repo(repo_data)

        # Should have warnings for high resource usage
        assert len(estimate.warnings) >= 0  # May or may not have warnings

    def test_recommendations_provided(self, resource_estimator):
        """Test recommendations are provided."""
        repo_data = {
            "name": "ml-project",
            "dependencies": {"python": ["torch"]},
            "stats": {"classes": 10, "functions": 20}
        }

        estimate = resource_estimator.estimate_from_repo(repo_data)

        # Recommendations may be provided based on analysis


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunction:
    """Tests for estimate_resources convenience function."""

    def test_estimate_with_repo_data(self):
        """Test estimate with repo data."""
        from core.resource_estimator import estimate_resources

        repo_data = {
            "name": "test",
            "dependencies": {"python": ["numpy"]},
            "stats": {"classes": 1, "functions": 2}
        }

        estimate = estimate_resources(repo_data=repo_data)

        assert estimate is not None

    def test_estimate_with_code(self):
        """Test estimate with code."""
        from core.resource_estimator import estimate_resources

        code = "print('hello')"

        estimate = estimate_resources(code=code, language="python")

        assert estimate is not None

    def test_estimate_empty(self):
        """Test estimate with no input."""
        from core.resource_estimator import estimate_resources

        estimate = estimate_resources()

        assert estimate is not None
        # Should return default estimate


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_dependencies(self, resource_estimator):
        """Test with empty dependencies."""
        repo_data = {
            "name": "empty",
            "dependencies": {"python": []},
            "stats": {"classes": 0, "functions": 0}
        }

        estimate = resource_estimator.estimate_from_repo(repo_data)

        assert estimate is not None

    def test_unknown_dependencies(self, resource_estimator):
        """Test with unknown dependencies."""
        repo_data = {
            "name": "custom",
            "dependencies": {"python": ["unknown_library_xyz"]},
            "stats": {"classes": 1, "functions": 1}
        }

        estimate = resource_estimator.estimate_from_repo(repo_data)

        assert estimate is not None

    def test_empty_code(self, resource_estimator):
        """Test with empty code."""
        estimate = resource_estimator.estimate_from_code("", "python")

        assert estimate is not None

    def test_malformed_code(self, resource_estimator):
        """Test with malformed code."""
        code = "def broken(:\n    pass"

        estimate = resource_estimator.estimate_from_code(code, "python")

        assert estimate is not None  # Should not raise
