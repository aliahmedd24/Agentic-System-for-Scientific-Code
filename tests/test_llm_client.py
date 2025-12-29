"""
Tests for LLM Client - Multi-provider LLM interface.
"""

import pytest
import os
import json
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import ValidationError


# ============================================================================
# LLMProvider Enum Tests
# ============================================================================

class TestLLMProvider:
    """Tests for LLMProvider enum."""

    def test_all_providers_exist(self):
        """Test all expected providers are defined."""
        from core.llm_client import LLMProvider
        assert LLMProvider.GEMINI.value == "gemini"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.OPENAI.value == "openai"

    def test_provider_from_string(self):
        """Test creating provider from string."""
        from core.llm_client import LLMProvider
        assert LLMProvider("gemini") == LLMProvider.GEMINI
        assert LLMProvider("anthropic") == LLMProvider.ANTHROPIC
        assert LLMProvider("openai") == LLMProvider.OPENAI

    def test_invalid_provider(self):
        """Test invalid provider raises ValueError."""
        from core.llm_client import LLMProvider
        with pytest.raises(ValueError):
            LLMProvider("invalid_provider")


# ============================================================================
# LLMConfig Tests
# ============================================================================

class TestLLMConfig:
    """Tests for LLMConfig model."""

    def test_config_creation(self):
        """Test valid config creation."""
        from core.llm_client import LLMConfig, LLMProvider
        config = LLMConfig(
            provider=LLMProvider.GEMINI,
            api_key="test-key",
            model="gemini-pro"
        )
        assert config.provider == LLMProvider.GEMINI
        assert config.api_key == "test-key"
        assert config.model == "gemini-pro"
        assert config.max_tokens == 8192  # Default
        assert config.temperature == 0.7  # Default
        assert config.timeout == 120  # Default

    def test_config_custom_values(self):
        """Test config with custom values."""
        from core.llm_client import LLMConfig, LLMProvider
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            model="gpt-4",
            max_tokens=4096,
            temperature=0.5,
            timeout=60
        )
        assert config.max_tokens == 4096
        assert config.temperature == 0.5
        assert config.timeout == 60

    def test_config_api_key_required(self):
        """Test that api_key is required."""
        from core.llm_client import LLMConfig, LLMProvider
        with pytest.raises(ValidationError):
            LLMConfig(provider=LLMProvider.GEMINI, model="gemini-pro")

    def test_config_api_key_not_empty(self):
        """Test that api_key cannot be empty."""
        from core.llm_client import LLMConfig, LLMProvider
        with pytest.raises(ValidationError):
            LLMConfig(provider=LLMProvider.GEMINI, api_key="", model="gemini-pro")

    def test_config_temperature_bounds(self):
        """Test temperature must be between 0 and 2."""
        from core.llm_client import LLMConfig, LLMProvider
        # Valid temperatures
        LLMConfig(provider=LLMProvider.GEMINI, api_key="key", model="m", temperature=0)
        LLMConfig(provider=LLMProvider.GEMINI, api_key="key", model="m", temperature=2)

        # Invalid temperatures
        with pytest.raises(ValidationError):
            LLMConfig(provider=LLMProvider.GEMINI, api_key="key", model="m", temperature=-0.1)
        with pytest.raises(ValidationError):
            LLMConfig(provider=LLMProvider.GEMINI, api_key="key", model="m", temperature=2.1)

    def test_config_max_tokens_positive(self):
        """Test max_tokens must be >= 1."""
        from core.llm_client import LLMConfig, LLMProvider
        with pytest.raises(ValidationError):
            LLMConfig(provider=LLMProvider.GEMINI, api_key="key", model="m", max_tokens=0)

    def test_config_timeout_positive(self):
        """Test timeout must be >= 1."""
        from core.llm_client import LLMConfig, LLMProvider
        with pytest.raises(ValidationError):
            LLMConfig(provider=LLMProvider.GEMINI, api_key="key", model="m", timeout=0)


# ============================================================================
# BaseLLMProvider Tests
# ============================================================================

class TestBaseLLMProvider:
    """Tests for BaseLLMProvider abstract class."""

    def test_provider_is_abstract(self):
        """Test that BaseLLMProvider cannot be instantiated."""
        from core.llm_client import BaseLLMProvider, LLMConfig, LLMProvider
        config = LLMConfig(provider=LLMProvider.GEMINI, api_key="key", model="m")

        # BaseLLMProvider is abstract and should not be instantiated directly
        with pytest.raises(TypeError):
            BaseLLMProvider(config)


# ============================================================================
# GeminiProvider Tests
# ============================================================================

class TestGeminiProvider:
    """Tests for GeminiProvider."""

    @pytest.fixture
    def gemini_config(self):
        """Create Gemini config."""
        from core.llm_client import LLMConfig, LLMProvider
        return LLMConfig(
            provider=LLMProvider.GEMINI,
            api_key="test-gemini-key",
            model="gemini-pro"
        )

    @pytest.fixture
    def gemini_provider(self, gemini_config):
        """Create Gemini provider."""
        from core.llm_client import GeminiProvider
        return GeminiProvider(gemini_config)

    @pytest.mark.asyncio
    async def test_generate_success(self, gemini_provider, mock_httpx_client):
        """Test successful generation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello, world!"}]
                }
            }]
        }
        mock_httpx_client.post.return_value = mock_response
        gemini_provider.client = mock_httpx_client

        result = await gemini_provider.generate("Say hello")
        assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_generate_api_error(self, gemini_provider, mock_httpx_client):
        """Test API error handling."""
        from core.error_handling import AgentError

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_httpx_client.post.return_value = mock_response
        gemini_provider.client = mock_httpx_client

        with pytest.raises(AgentError) as exc_info:
            await gemini_provider.generate("test")
        assert "401" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_structured_json_cleanup(self, gemini_provider, mock_httpx_client):
        """Test JSON cleanup in structured generation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Response with markdown code fences
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": '```json\n{"key": "value"}\n```'}]
                }
            }]
        }
        mock_httpx_client.post.return_value = mock_response
        gemini_provider.client = mock_httpx_client

        result = await gemini_provider.generate_structured("test", {"type": "object"})
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_generate_structured_invalid_json(self, gemini_provider, mock_httpx_client):
        """Test error on invalid JSON response."""
        from core.error_handling import AgentError

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "not valid json"}]
                }
            }]
        }
        mock_httpx_client.post.return_value = mock_response
        gemini_provider.client = mock_httpx_client

        with pytest.raises(AgentError) as exc_info:
            await gemini_provider.generate_structured("test", {"type": "object"})
        assert "Failed to parse" in str(exc_info.value)


# ============================================================================
# AnthropicProvider Tests
# ============================================================================

class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    @pytest.fixture
    def anthropic_config(self):
        """Create Anthropic config."""
        from core.llm_client import LLMConfig, LLMProvider
        return LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            api_key="test-anthropic-key",
            model="claude-3-sonnet"
        )

    @pytest.fixture
    def anthropic_provider(self, anthropic_config):
        """Create Anthropic provider."""
        from core.llm_client import AnthropicProvider
        return AnthropicProvider(anthropic_config)

    @pytest.mark.asyncio
    async def test_generate_success(self, anthropic_provider, mock_httpx_client):
        """Test successful generation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hello from Claude!"}]
        }
        mock_httpx_client.post.return_value = mock_response
        anthropic_provider.client = mock_httpx_client

        result = await anthropic_provider.generate("Say hello")
        assert result == "Hello from Claude!"

    @pytest.mark.asyncio
    async def test_generate_api_error(self, anthropic_provider, mock_httpx_client):
        """Test API error handling."""
        from core.error_handling import AgentError

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_httpx_client.post.return_value = mock_response
        anthropic_provider.client = mock_httpx_client

        with pytest.raises(AgentError) as exc_info:
            await anthropic_provider.generate("test")
        assert "400" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_structured_tool_use(self, anthropic_provider, mock_httpx_client):
        """Test structured output using tool_use."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "test_id",
                    "name": "structured_output",
                    "input": {"key": "value", "count": 42}
                }
            ]
        }
        mock_httpx_client.post.return_value = mock_response
        anthropic_provider.client = mock_httpx_client

        result = await anthropic_provider.generate_structured("test", {"type": "object"})
        assert result == {"key": "value", "count": 42}


# ============================================================================
# OpenAIProvider Tests
# ============================================================================

class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    @pytest.fixture
    def openai_config(self):
        """Create OpenAI config."""
        from core.llm_client import LLMConfig, LLMProvider
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="test-openai-key",
            model="gpt-4"
        )

    @pytest.fixture
    def openai_provider(self, openai_config):
        """Create OpenAI provider."""
        from core.llm_client import OpenAIProvider
        return OpenAIProvider(openai_config)

    @pytest.mark.asyncio
    async def test_generate_success(self, openai_provider, mock_httpx_client):
        """Test successful generation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": "Hello from GPT!"}
            }]
        }
        mock_httpx_client.post.return_value = mock_response
        openai_provider.client = mock_httpx_client

        result = await openai_provider.generate("Say hello")
        assert result == "Hello from GPT!"

    @pytest.mark.asyncio
    async def test_generate_api_error(self, openai_provider, mock_httpx_client):
        """Test API error handling."""
        from core.error_handling import AgentError

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limited"
        mock_httpx_client.post.return_value = mock_response
        openai_provider.client = mock_httpx_client

        with pytest.raises(AgentError) as exc_info:
            await openai_provider.generate("test")
        assert "429" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_structured_json_schema(self, openai_provider, mock_httpx_client):
        """Test structured output with json_schema response_format."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": '{"name": "test", "value": 123}'}
            }]
        }
        mock_httpx_client.post.return_value = mock_response
        openai_provider.client = mock_httpx_client

        result = await openai_provider.generate_structured("test", {"type": "object"})
        assert result == {"name": "test", "value": 123}

    @pytest.mark.asyncio
    async def test_generate_with_system_instruction(self, openai_provider, mock_httpx_client):
        """Test generation with system instruction."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": "Helpful response"}
            }]
        }
        mock_httpx_client.post.return_value = mock_response
        openai_provider.client = mock_httpx_client

        result = await openai_provider.generate(
            "Hello",
            system_instruction="You are helpful"
        )
        assert result == "Helpful response"

        # Verify system instruction was included in request
        call_args = mock_httpx_client.post.call_args
        body = call_args.kwargs.get('json', {})
        messages = body.get('messages', [])
        assert any(m.get('role') == 'system' for m in messages)


# ============================================================================
# LLMClient Tests
# ============================================================================

class TestLLMClient:
    """Tests for LLMClient unified interface."""

    def test_client_creation_gemini(self):
        """Test creating client with Gemini provider."""
        from core.llm_client import LLMClient
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            client = LLMClient(provider="gemini")
            assert client.config.provider.value == "gemini"

    def test_client_creation_anthropic(self):
        """Test creating client with Anthropic provider."""
        from core.llm_client import LLMClient
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            client = LLMClient(provider="anthropic")
            assert client.config.provider.value == "anthropic"

    def test_client_creation_openai(self):
        """Test creating client with OpenAI provider."""
        from core.llm_client import LLMClient
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = LLMClient(provider="openai")
            assert client.config.provider.value == "openai"

    def test_client_unknown_provider(self):
        """Test error on unknown provider."""
        from core.llm_client import LLMClient
        from core.error_handling import AgentError

        with pytest.raises(AgentError) as exc_info:
            LLMClient(provider="unknown_provider")
        assert "Unknown provider" in str(exc_info.value)

    def test_client_missing_api_key(self):
        """Test error when no API key available."""
        from core.llm_client import LLMClient
        from core.error_handling import AgentError

        # Clear environment
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(AgentError) as exc_info:
                LLMClient(provider="gemini")
            assert "API key required" in str(exc_info.value)

    def test_client_env_api_key_gemini(self):
        """Test loading API key from GOOGLE_API_KEY."""
        from core.llm_client import LLMClient
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "google-test-key"}):
            client = LLMClient(provider="gemini")
            assert client.config.api_key == "google-test-key"

    def test_client_env_api_key_gemini_alternative(self):
        """Test loading API key from GEMINI_API_KEY."""
        from core.llm_client import LLMClient
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-test-key"}, clear=True):
            client = LLMClient(provider="gemini")
            assert client.config.api_key == "gemini-test-key"

    def test_client_direct_api_key(self):
        """Test passing API key directly."""
        from core.llm_client import LLMClient
        client = LLMClient(provider="gemini", api_key="direct-key")
        assert client.config.api_key == "direct-key"

    def test_client_custom_model(self):
        """Test using custom model."""
        from core.llm_client import LLMClient
        client = LLMClient(provider="gemini", api_key="key", model="gemini-1.5-pro")
        assert client.config.model == "gemini-1.5-pro"

    def test_client_default_model(self):
        """Test default model is used when not specified."""
        from core.llm_client import LLMClient, DEFAULT_MODELS, LLMProvider
        client = LLMClient(provider="gemini", api_key="key")
        assert client.config.model == DEFAULT_MODELS[LLMProvider.GEMINI]

    @pytest.mark.asyncio
    async def test_client_context_manager(self, mock_httpx_client):
        """Test async context manager."""
        from core.llm_client import LLMClient

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            async with LLMClient(provider="gemini") as client:
                assert client._provider.client is not None

    @pytest.mark.asyncio
    async def test_generate_delegates_to_provider(self, mock_httpx_client):
        """Test generate() delegates to provider."""
        from core.llm_client import LLMClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "test response"}]}}]
        }
        mock_httpx_client.post.return_value = mock_response

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            client = LLMClient(provider="gemini")
            client._provider.client = mock_httpx_client

            result = await client.generate("test prompt")
            assert result == "test response"

    @pytest.mark.asyncio
    async def test_generate_structured_delegates(self, mock_httpx_client):
        """Test generate_structured() delegates to provider."""
        from core.llm_client import LLMClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"result": true}'}]}}]
        }
        mock_httpx_client.post.return_value = mock_response

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            client = LLMClient(provider="gemini")
            client._provider.client = mock_httpx_client

            result = await client.generate_structured("test", {"type": "object"})
            assert result == {"result": True}

    @pytest.mark.asyncio
    async def test_generate_stream(self, mock_httpx_client):
        """Test streaming generation."""
        from core.llm_client import LLMClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "streamed response"}]}}]
        }
        mock_httpx_client.post.return_value = mock_response

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            client = LLMClient(provider="gemini")
            client._provider.client = mock_httpx_client

            chunks = []
            async for chunk in client.generate_stream("test"):
                chunks.append(chunk)

            # Default implementation returns full response as single chunk
            assert len(chunks) == 1
            assert chunks[0] == "streamed response"


# ============================================================================
# Provider Map and Constants Tests
# ============================================================================

class TestProviderConstants:
    """Tests for provider maps and constants."""

    def test_provider_map_complete(self):
        """Test PROVIDER_MAP has all providers."""
        from core.llm_client import PROVIDER_MAP, LLMProvider
        assert LLMProvider.GEMINI in PROVIDER_MAP
        assert LLMProvider.ANTHROPIC in PROVIDER_MAP
        assert LLMProvider.OPENAI in PROVIDER_MAP

    def test_default_models_complete(self):
        """Test DEFAULT_MODELS has all providers."""
        from core.llm_client import DEFAULT_MODELS, LLMProvider
        assert LLMProvider.GEMINI in DEFAULT_MODELS
        assert LLMProvider.ANTHROPIC in DEFAULT_MODELS
        assert LLMProvider.OPENAI in DEFAULT_MODELS

    def test_default_models_not_empty(self):
        """Test default models are non-empty strings."""
        from core.llm_client import DEFAULT_MODELS
        for provider, model in DEFAULT_MODELS.items():
            assert model, f"Default model for {provider} is empty"
            assert isinstance(model, str)
