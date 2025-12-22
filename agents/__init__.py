"""
Agents module for Scientific Paper Analysis System.

Contains specialized agents for different pipeline stages:
- PaperParserAgent: Extracts and analyzes scientific papers
- RepoAnalyzerAgent: Analyzes code repositories
- SemanticMapper: Maps paper concepts to code implementations
- CodingAgent: Generates and executes test/validation code
"""

from .base_agent import BaseAgent
from .paper_parser_agent import PaperParserAgent
from .repo_analyzer_agent import RepoAnalyzerAgent
from .semantic_mapper import SemanticMapper
from .coding_agent import CodingAgent

__all__ = [
    "BaseAgent",
    "PaperParserAgent",
    "RepoAnalyzerAgent",
    "SemanticMapper",
    "CodingAgent",
]
