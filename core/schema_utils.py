"""
Schema utilities for LLM providers.

This module provides utilities for generating JSON schemas from Pydantic models
with provider-specific optimizations (e.g., $defs inlining for Gemini).
"""

from typing import Type, Dict, Any, Union
from copy import deepcopy

from pydantic import BaseModel


def generate_llm_schema(
    model: Type[BaseModel],
    provider: str = "gemini"
) -> Dict[str, Any]:
    """
    Generate JSON schema suitable for LLM consumption.

    Args:
        model: Pydantic model class
        provider: LLM provider name (gemini, anthropic, openai)

    Returns:
        JSON schema dict, simplified for Gemini, native for others
    """
    schema = model.model_json_schema()

    if provider == "gemini":
        # Gemini may struggle with $defs - inline definitions
        return flatten_schema_defs(schema)
    else:
        # Anthropic and OpenAI handle $defs properly
        return schema


def flatten_schema_defs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simplify schema by inlining $defs for Gemini compatibility.

    Recursively resolves $ref pointers and removes $defs.

    Args:
        schema: JSON schema with potential $defs

    Returns:
        Flattened schema with all $refs resolved inline
    """
    schema = deepcopy(schema)
    defs = schema.pop("$defs", {})

    if not defs:
        return schema

    def resolve_refs(obj: Any, depth: int = 0) -> Any:
        """Recursively resolve $ref pointers."""
        # Prevent infinite recursion for circular references
        if depth > 50:
            return obj

        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_path = obj["$ref"]
                # Extract definition name from "#/$defs/DefinitionName"
                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path.split("/")[-1]
                    if def_name in defs:
                        # Return resolved definition with refs resolved
                        resolved = deepcopy(defs[def_name])
                        return resolve_refs(resolved, depth + 1)
                return obj

            return {k: resolve_refs(v, depth) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [resolve_refs(item, depth) for item in obj]

        return obj

    return resolve_refs(schema)


def clean_schema_for_llm(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean up schema for better LLM understanding.

    Removes Pydantic-specific fields that may confuse LLMs.

    Args:
        schema: JSON schema dict

    Returns:
        Cleaned schema
    """
    schema = deepcopy(schema)

    # Fields to remove at any level
    fields_to_remove = {"title", "default"}

    def clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: clean(v)
                for k, v in obj.items()
                if k not in fields_to_remove
            }
        elif isinstance(obj, list):
            return [clean(item) for item in obj]
        return obj

    return clean(schema)


def get_schema_for_provider(
    model: Type[BaseModel],
    provider: str,
    clean: bool = False
) -> Dict[str, Any]:
    """
    Get the appropriate schema for a given LLM provider.

    Args:
        model: Pydantic model class
        provider: Provider name (gemini, anthropic, openai)
        clean: Whether to remove non-essential fields

    Returns:
        Provider-appropriate JSON schema
    """
    schema = generate_llm_schema(model, provider)

    if clean:
        schema = clean_schema_for_llm(schema)

    return schema
