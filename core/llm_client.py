"""
Unified LLM interface supporting multiple providers.
"""

import os
import json
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, AsyncIterator, Type
from dataclasses import dataclass
from enum import Enum

import httpx

from .error_handling import (
    logger, LogCategory, ErrorCategory, ErrorSeverity,
    create_error, AgentError, with_retry
)


class LLMProvider(Enum):
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: LLMProvider
    api_key: str
    model: str
    max_tokens: int = 8192
    temperature: float = 0.7
    timeout: int = 120


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(self.config.timeout))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a structured JSON response."""
        pass

    async def generate_stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Generate a streaming response. Default implementation returns full response."""
        response = await self.generate(prompt, system_instruction)
        yield response


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider."""
    
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
    @with_retry(ErrorCategory.LLM, "Gemini API call")
    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> str:
        url = f"{self.BASE_URL}/models/{self.config.model}:generateContent"
        
        contents = [{"parts": [{"text": prompt}]}]
        
        body = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            }
        }
        
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        
        headers = {"Content-Type": "application/json"}
        params = {"key": self.config.api_key}
        
        response = await self.client.post(url, json=body, headers=headers, params=params)
        
        if response.status_code != 200:
            raise AgentError(create_error(
                ErrorCategory.LLM,
                f"Gemini API error: {response.status_code} - {response.text}",
                suggestion="Check your API key and try again"
            ))
        
        data = response.json()
        
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise AgentError(create_error(
                ErrorCategory.LLM,
                f"Unexpected Gemini response format: {data}",
                original_error=e
            ))

    @with_retry(ErrorCategory.LLM, "Gemini structured API call")
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        # Add JSON instruction to system prompt
        json_instruction = (
            f"You must respond with valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Only output the JSON, no other text or markdown formatting."
        )
        
        full_system = f"{system_instruction}\n\n{json_instruction}" if system_instruction else json_instruction
        
        response = await self.generate(prompt, full_system)
        
        # Clean up response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise AgentError(create_error(
                ErrorCategory.PARSING,
                f"Failed to parse LLM response as JSON: {response[:200]}...",
                original_error=e
            ))


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""
    
    BASE_URL = "https://api.anthropic.com/v1"
    
    @with_retry(ErrorCategory.LLM, "Anthropic API call")
    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> str:
        url = f"{self.BASE_URL}/messages"
        
        body = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system_instruction:
            body["system"] = system_instruction
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        response = await self.client.post(url, json=body, headers=headers)
        
        if response.status_code != 200:
            raise AgentError(create_error(
                ErrorCategory.LLM,
                f"Anthropic API error: {response.status_code} - {response.text}",
                suggestion="Check your API key and try again"
            ))
        
        data = response.json()
        
        try:
            return data["content"][0]["text"]
        except (KeyError, IndexError) as e:
            raise AgentError(create_error(
                ErrorCategory.LLM,
                f"Unexpected Anthropic response format: {data}",
                original_error=e
            ))

    @with_retry(ErrorCategory.LLM, "Anthropic structured API call")
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        json_instruction = (
            f"You must respond with valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Only output the JSON, no other text or markdown formatting."
        )
        
        full_system = f"{system_instruction}\n\n{json_instruction}" if system_instruction else json_instruction
        
        response = await self.generate(prompt, full_system)
        
        # Clean up response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise AgentError(create_error(
                ErrorCategory.PARSING,
                f"Failed to parse LLM response as JSON: {response[:200]}...",
                original_error=e
            ))


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""
    
    BASE_URL = "https://api.openai.com/v1"
    
    @with_retry(ErrorCategory.LLM, "OpenAI API call")
    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> str:
        url = f"{self.BASE_URL}/chat/completions"
        
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        body = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": messages
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        
        response = await self.client.post(url, json=body, headers=headers)
        
        if response.status_code != 200:
            raise AgentError(create_error(
                ErrorCategory.LLM,
                f"OpenAI API error: {response.status_code} - {response.text}",
                suggestion="Check your API key and try again"
            ))
        
        data = response.json()
        
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise AgentError(create_error(
                ErrorCategory.LLM,
                f"Unexpected OpenAI response format: {data}",
                original_error=e
            ))

    @with_retry(ErrorCategory.LLM, "OpenAI structured API call")
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        json_instruction = (
            f"You must respond with valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Only output the JSON, no other text or markdown formatting."
        )
        
        full_system = f"{system_instruction}\n\n{json_instruction}" if system_instruction else json_instruction
        
        response = await self.generate(prompt, full_system)
        
        # Clean up response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise AgentError(create_error(
                ErrorCategory.PARSING,
                f"Failed to parse LLM response as JSON: {response[:200]}...",
                original_error=e
            ))


PROVIDER_MAP: Dict[LLMProvider, Type[BaseLLMProvider]] = {
    LLMProvider.GEMINI: GeminiProvider,
    LLMProvider.ANTHROPIC: AnthropicProvider,
    LLMProvider.OPENAI: OpenAIProvider,
}

DEFAULT_MODELS: Dict[LLMProvider, str] = {
    LLMProvider.GEMINI: "gemini-2.0-flash-exp",
    LLMProvider.ANTHROPIC: "claude-sonnet-4-20250514",
    LLMProvider.OPENAI: "gpt-4o",
}


class LLMClient:
    """
    Unified LLM interface with multi-provider support.
    
    Usage:
        async with LLMClient(provider="gemini", api_key="...") as client:
            response = await client.generate("Hello!")
    """
    
    def __init__(
        self,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        timeout: int = 120
    ):
        # Parse provider
        try:
            llm_provider = LLMProvider(provider.lower())
        except ValueError:
            raise AgentError(create_error(
                ErrorCategory.CONFIGURATION,
                f"Unknown provider: {provider}. Supported: {[p.value for p in LLMProvider]}",
                recoverable=False
            ))
        
        # Get API key from environment if not provided
        if not api_key:
            env_keys = {
                LLMProvider.GEMINI: "GEMINI_API_KEY",
                LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
                LLMProvider.OPENAI: "OPENAI_API_KEY",
            }
            api_key = os.getenv(env_keys.get(llm_provider, ""))
        
        if not api_key:
            raise AgentError(create_error(
                ErrorCategory.CONFIGURATION,
                f"API key required for {provider}",
                suggestion=f"Set the API key in environment variables or pass it directly",
                recoverable=False
            ))
        
        # Use default model if not specified
        if not model:
            model = DEFAULT_MODELS.get(llm_provider, "")
        
        self.config = LLMConfig(
            provider=llm_provider,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout
        )
        
        provider_class = PROVIDER_MAP.get(llm_provider)
        if not provider_class:
            raise AgentError(create_error(
                ErrorCategory.CONFIGURATION,
                f"Provider {provider} not implemented",
                recoverable=False
            ))
        
        self._provider = provider_class(self.config)

    async def __aenter__(self):
        await self._provider.__aenter__()
        logger.info(
            LogCategory.LLM,
            f"Initialized {self.config.provider.value} client with model {self.config.model}"
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._provider.__aexit__(exc_type, exc_val, exc_tb)

    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> str:
        """Generate a response from the LLM."""
        logger.debug(
            LogCategory.LLM,
            f"Generating response (prompt length: {len(prompt)} chars)"
        )
        response = await self._provider.generate(prompt, system_instruction)
        logger.debug(
            LogCategory.LLM,
            f"Received response (length: {len(response)} chars)"
        )
        return response

    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a structured JSON response."""
        logger.debug(
            LogCategory.LLM,
            f"Generating structured response"
        )
        return await self._provider.generate_structured(prompt, schema, system_instruction)

    async def generate_stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        async for chunk in self._provider.generate_stream(prompt, system_instruction):
            yield chunk
