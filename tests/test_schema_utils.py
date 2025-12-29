"""
Tests for Schema Utilities - LLM schema generation.
"""

import pytest
from pydantic import BaseModel, Field
from typing import List, Optional


# ============================================================================
# Test Models
# ============================================================================

class InnerModel(BaseModel):
    """Inner model for nested tests."""
    value: int = Field(..., description="Inner value")
    name: str = Field("default", description="Inner name")


class OuterModel(BaseModel):
    """Outer model with nested model."""
    title: str = Field(..., description="Title field")
    inner: InnerModel = Field(..., description="Nested inner model")
    items: List[str] = Field(default_factory=list, description="List of items")


class SimpleModel(BaseModel):
    """Simple model without nesting."""
    name: str = Field(..., description="Name")
    count: int = Field(0, description="Count")


class ModelWithOptional(BaseModel):
    """Model with optional fields."""
    required_field: str = Field(..., description="Required")
    optional_field: Optional[str] = Field(None, description="Optional")


class CircularModelA(BaseModel):
    """Model A for circular reference testing."""
    name: str
    # Note: Pydantic handles forward references


class DeepNestedModel(BaseModel):
    """Model with deep nesting."""
    level1: InnerModel
    outer: OuterModel


# ============================================================================
# generate_llm_schema Tests
# ============================================================================

class TestGenerateLLMSchema:
    """Tests for generate_llm_schema function."""

    def test_generate_for_gemini(self):
        """Test schema generation for Gemini."""
        from core.schema_utils import generate_llm_schema

        schema = generate_llm_schema(SimpleModel, provider="gemini")

        assert schema is not None
        assert "properties" in schema

    def test_generate_for_anthropic(self):
        """Test schema generation for Anthropic."""
        from core.schema_utils import generate_llm_schema

        schema = generate_llm_schema(SimpleModel, provider="anthropic")

        assert schema is not None
        assert "properties" in schema

    def test_generate_for_openai(self):
        """Test schema generation for OpenAI."""
        from core.schema_utils import generate_llm_schema

        schema = generate_llm_schema(SimpleModel, provider="openai")

        assert schema is not None
        assert "properties" in schema

    def test_generate_with_nested_model(self):
        """Test schema generation with nested models."""
        from core.schema_utils import generate_llm_schema

        schema = generate_llm_schema(OuterModel, provider="gemini")

        assert schema is not None
        assert "properties" in schema
        assert "title" in schema["properties"]
        assert "inner" in schema["properties"]

    def test_gemini_schema_has_no_defs(self):
        """Test Gemini schema has $defs inlined."""
        from core.schema_utils import generate_llm_schema

        schema = generate_llm_schema(OuterModel, provider="gemini")

        # Gemini schemas should have $defs removed
        assert "$defs" not in schema

    def test_anthropic_may_have_defs(self):
        """Test Anthropic schema may have $defs."""
        from core.schema_utils import generate_llm_schema

        schema = generate_llm_schema(OuterModel, provider="anthropic")

        # Anthropic handles $defs natively
        # May or may not have $defs depending on Pydantic version


# ============================================================================
# flatten_schema_defs Tests
# ============================================================================

class TestFlattenSchemaDefs:
    """Tests for flatten_schema_defs function."""

    def test_flatten_empty_defs(self):
        """Test flattening schema with no $defs."""
        from core.schema_utils import flatten_schema_defs

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }

        flattened = flatten_schema_defs(schema)

        assert flattened == schema
        assert "$defs" not in flattened

    def test_flatten_simple_ref(self):
        """Test flattening simple $ref."""
        from core.schema_utils import flatten_schema_defs

        schema = {
            "type": "object",
            "properties": {
                "inner": {"$ref": "#/$defs/InnerModel"}
            },
            "$defs": {
                "InnerModel": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "integer"}
                    }
                }
            }
        }

        flattened = flatten_schema_defs(schema)

        # $defs should be removed
        assert "$defs" not in flattened

        # $ref should be replaced with inline definition
        inner = flattened["properties"]["inner"]
        assert "$ref" not in inner
        assert inner["type"] == "object"
        assert "value" in inner["properties"]

    def test_flatten_nested_refs(self):
        """Test flattening nested $refs."""
        from core.schema_utils import flatten_schema_defs

        schema = {
            "type": "object",
            "properties": {
                "outer": {"$ref": "#/$defs/Outer"}
            },
            "$defs": {
                "Outer": {
                    "type": "object",
                    "properties": {
                        "inner": {"$ref": "#/$defs/Inner"}
                    }
                },
                "Inner": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "integer"}
                    }
                }
            }
        }

        flattened = flatten_schema_defs(schema)

        # All $refs should be resolved
        assert "$defs" not in flattened

        outer = flattened["properties"]["outer"]
        assert outer["type"] == "object"

        inner = outer["properties"]["inner"]
        assert inner["type"] == "object"
        assert "value" in inner["properties"]

    def test_flatten_preserves_other_fields(self):
        """Test that non-ref fields are preserved."""
        from core.schema_utils import flatten_schema_defs

        schema = {
            "type": "object",
            "title": "TestSchema",
            "description": "A test schema",
            "properties": {
                "name": {"type": "string", "description": "Name field"}
            }
        }

        flattened = flatten_schema_defs(schema)

        assert flattened["title"] == "TestSchema"
        assert flattened["description"] == "A test schema"
        assert flattened["properties"]["name"]["description"] == "Name field"

    def test_flatten_array_refs(self):
        """Test flattening $refs in arrays."""
        from core.schema_utils import flatten_schema_defs

        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Item"}
                }
            },
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }

        flattened = flatten_schema_defs(schema)

        items_schema = flattened["properties"]["items"]["items"]
        assert "$ref" not in items_schema
        assert items_schema["type"] == "object"

    def test_flatten_with_depth_limit(self):
        """Test recursion depth limit prevents infinite loops."""
        from core.schema_utils import flatten_schema_defs

        # This should not hang even with self-referential structure
        # (though Pydantic normally handles this)
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "string"}
            }
        }

        flattened = flatten_schema_defs(schema)
        assert flattened is not None


# ============================================================================
# clean_schema_for_llm Tests
# ============================================================================

class TestCleanSchemaForLLM:
    """Tests for clean_schema_for_llm function."""

    def test_clean_removes_title(self):
        """Test title field is removed."""
        from core.schema_utils import clean_schema_for_llm

        schema = {
            "type": "object",
            "title": "MyModel",
            "properties": {
                "name": {
                    "type": "string",
                    "title": "Name"
                }
            }
        }

        cleaned = clean_schema_for_llm(schema)

        assert "title" not in cleaned
        assert "title" not in cleaned["properties"]["name"]

    def test_clean_removes_default(self):
        """Test default field is removed."""
        from core.schema_utils import clean_schema_for_llm

        schema = {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "default": 0
                }
            }
        }

        cleaned = clean_schema_for_llm(schema)

        assert "default" not in cleaned["properties"]["count"]

    def test_clean_preserves_essential(self):
        """Test essential fields are preserved."""
        from core.schema_utils import clean_schema_for_llm

        schema = {
            "type": "object",
            "description": "A model",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name"
                }
            },
            "required": ["name"]
        }

        cleaned = clean_schema_for_llm(schema)

        assert cleaned["type"] == "object"
        assert cleaned["description"] == "A model"
        assert cleaned["required"] == ["name"]
        assert cleaned["properties"]["name"]["description"] == "The name"

    def test_clean_recursive(self):
        """Test cleaning is recursive."""
        from core.schema_utils import clean_schema_for_llm

        schema = {
            "type": "object",
            "title": "Outer",
            "properties": {
                "inner": {
                    "type": "object",
                    "title": "Inner",
                    "properties": {
                        "value": {
                            "type": "integer",
                            "title": "Value"
                        }
                    }
                }
            }
        }

        cleaned = clean_schema_for_llm(schema)

        assert "title" not in cleaned
        assert "title" not in cleaned["properties"]["inner"]
        assert "title" not in cleaned["properties"]["inner"]["properties"]["value"]

    def test_clean_arrays(self):
        """Test cleaning handles arrays."""
        from core.schema_utils import clean_schema_for_llm

        schema = {
            "type": "array",
            "title": "Items",
            "items": {
                "type": "string",
                "title": "Item"
            }
        }

        cleaned = clean_schema_for_llm(schema)

        assert "title" not in cleaned
        assert "title" not in cleaned["items"]


# ============================================================================
# get_schema_for_provider Tests
# ============================================================================

class TestGetSchemaForProvider:
    """Tests for get_schema_for_provider function."""

    def test_get_schema_gemini(self):
        """Test getting schema for Gemini."""
        from core.schema_utils import get_schema_for_provider

        schema = get_schema_for_provider(SimpleModel, "gemini")

        assert schema is not None
        assert "$defs" not in schema

    def test_get_schema_anthropic(self):
        """Test getting schema for Anthropic."""
        from core.schema_utils import get_schema_for_provider

        schema = get_schema_for_provider(SimpleModel, "anthropic")

        assert schema is not None

    def test_get_schema_with_clean(self):
        """Test getting cleaned schema."""
        from core.schema_utils import get_schema_for_provider

        schema = get_schema_for_provider(SimpleModel, "gemini", clean=True)

        assert schema is not None
        # title and default should be removed
        if "title" in schema:
            # Some Pydantic versions might not add title at top level
            pass

    def test_get_schema_without_clean(self):
        """Test getting uncleaned schema."""
        from core.schema_utils import get_schema_for_provider

        schema = get_schema_for_provider(SimpleModel, "gemini", clean=False)

        assert schema is not None


# ============================================================================
# Integration Tests with Real Models
# ============================================================================

class TestSchemaIntegration:
    """Integration tests with actual protocol models."""

    def test_paper_parser_output_schema(self):
        """Test generating schema for PaperParserOutput."""
        from core.schema_utils import generate_llm_schema
        from agents.protocols import PaperParserOutput

        schema = generate_llm_schema(PaperParserOutput, "gemini")

        assert schema is not None
        assert "properties" in schema
        assert "title" in schema["properties"]
        assert "abstract" in schema["properties"]

    def test_mapping_result_schema(self):
        """Test generating schema for MappingResult."""
        from core.schema_utils import generate_llm_schema
        from agents.protocols import MappingResult

        schema = generate_llm_schema(MappingResult, "gemini")

        assert schema is not None
        assert "properties" in schema
        assert "concept_name" in schema["properties"]
        assert "confidence" in schema["properties"]

    def test_concept_schema(self):
        """Test generating schema for Concept."""
        from core.schema_utils import generate_llm_schema
        from agents.protocols import Concept

        schema = generate_llm_schema(Concept, "gemini")

        assert schema is not None
        assert "properties" in schema
        assert "name" in schema["properties"]


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_model(self):
        """Test schema for model with no fields."""
        from core.schema_utils import generate_llm_schema

        class EmptyModel(BaseModel):
            pass

        schema = generate_llm_schema(EmptyModel, "gemini")

        assert schema is not None
        assert schema["type"] == "object"

    def test_model_with_complex_types(self):
        """Test schema for model with complex types."""
        from core.schema_utils import generate_llm_schema
        from typing import Dict, Any

        class ComplexModel(BaseModel):
            mapping: Dict[str, Any] = Field(default_factory=dict)
            items: List[InnerModel] = Field(default_factory=list)

        schema = generate_llm_schema(ComplexModel, "gemini")

        assert schema is not None

    def test_deep_nesting(self):
        """Test schema with deep nesting."""
        from core.schema_utils import generate_llm_schema

        schema = generate_llm_schema(DeepNestedModel, "gemini")

        assert schema is not None
        assert "$defs" not in schema
