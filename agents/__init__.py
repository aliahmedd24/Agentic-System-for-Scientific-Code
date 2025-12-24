"""
Agents module for Scientific Paper Analysis System.

Contains specialized agents for different pipeline stages:
- PaperParserAgent: Extracts and analyzes scientific papers
- RepoAnalyzerAgent: Analyzes code repositories
- SemanticMapper: Maps paper concepts to code implementations
- CodingAgent: Generates and executes test/validation code

Also includes:
- Multi-language parsers for Python, Julia, R, JavaScript/TypeScript
- Agent protocols and type definitions
"""

from .base_agent import BaseAgent, AgentStats
from .paper_parser_agent import PaperParserAgent
from .repo_analyzer_agent import RepoAnalyzerAgent
from .semantic_mapper import SemanticMapper
from .coding_agent import CodingAgent
from .protocols import (
    PaperParserInput, PaperParserOutput,
    RepoAnalyzerInput, RepoAnalyzerOutput,
    SemanticMapperInput, SemanticMapperOutput,
    CodingAgentInput, CodingAgentOutput,
    AgentProtocol, ValidationError
)
from .parsers import (
    CodeElement, LanguageParser, ParserFactory,
    PythonParser, JuliaParser, RParser, JavaScriptParser
)

__all__ = [
    # Core agents
    "BaseAgent",
    "AgentStats",
    "PaperParserAgent",
    "RepoAnalyzerAgent",
    "SemanticMapper",
    "CodingAgent",

    # Protocols and types
    "PaperParserInput",
    "PaperParserOutput",
    "RepoAnalyzerInput",
    "RepoAnalyzerOutput",
    "SemanticMapperInput",
    "SemanticMapperOutput",
    "CodingAgentInput",
    "CodingAgentOutput",
    "AgentProtocol",
    "ValidationError",

    # Parsers
    "CodeElement",
    "LanguageParser",
    "ParserFactory",
    "PythonParser",
    "JuliaParser",
    "RParser",
    "JavaScriptParser",
]
