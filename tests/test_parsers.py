"""
Tests for Multi-Language Parsers - Code element extraction.
"""

import pytest
from pathlib import Path


# ============================================================================
# CodeElement Tests
# ============================================================================

class TestCodeElement:
    """Tests for CodeElement model."""

    def test_element_creation_minimal(self):
        """Test creating element with minimal fields."""
        from agents.parsers import CodeElement

        element = CodeElement(
            name="test_func",
            element_type="function",
            file_path="test.py"
        )

        assert element.name == "test_func"
        assert element.element_type == "function"
        assert element.file_path == "test.py"
        assert element.line_number == 0
        assert element.language == "python"

    def test_element_creation_full(self):
        """Test creating element with all fields."""
        from agents.parsers import CodeElement

        element = CodeElement(
            name="MyClass",
            element_type="class",
            file_path="model.py",
            line_number=42,
            docstring="A sample class",
            signature="class MyClass(BaseClass)",
            args=[],
            return_type="",
            decorators=["@dataclass"],
            bases=["BaseClass"],
            methods=["__init__", "process"],
            language="python"
        )

        assert element.name == "MyClass"
        assert element.line_number == 42
        assert len(element.bases) == 1
        assert len(element.methods) == 2


# ============================================================================
# PythonParser Tests
# ============================================================================

class TestPythonParser:
    """Tests for PythonParser."""

    @pytest.fixture
    def parser(self):
        """Create Python parser."""
        from agents.parsers import PythonParser
        return PythonParser()

    def test_extensions(self, parser):
        """Test supported extensions."""
        assert ".py" in parser.extensions
        assert ".pyx" in parser.extensions

    def test_language(self, parser):
        """Test language name."""
        assert parser.language == "python"

    def test_parse_class(self, parser, sample_python_file):
        """Test class extraction."""
        result = parser.parse_file(sample_python_file)

        assert "classes" in result
        assert len(result["classes"]) == 1

        cls = result["classes"][0]
        assert cls.name == "SampleClass"
        assert cls.element_type == "class"
        assert "A sample class" in cls.docstring

    def test_parse_class_with_methods(self, parser, sample_python_file):
        """Test methods are captured."""
        result = parser.parse_file(sample_python_file)

        cls = result["classes"][0]
        assert "__init__" in cls.methods
        assert "get_value" in cls.methods

    def test_parse_function(self, parser, sample_python_file):
        """Test function extraction."""
        result = parser.parse_file(sample_python_file)

        assert "functions" in result
        function_names = [f.name for f in result["functions"]]

        assert "sample_function" in function_names

    def test_parse_async_function(self, parser, sample_python_file):
        """Test async function extraction."""
        result = parser.parse_file(sample_python_file)

        function_names = [f.name for f in result["functions"]]
        assert "async_function" in function_names

    def test_parse_function_with_args(self, parser, sample_python_file):
        """Test function arguments are captured."""
        result = parser.parse_file(sample_python_file)

        func = next(f for f in result["functions"] if f.name == "sample_function")
        assert "arg1" in func.args
        assert "arg2" in func.args

    def test_parse_function_return_type(self, parser, sample_python_file):
        """Test return type annotation is captured."""
        result = parser.parse_file(sample_python_file)

        func = next(f for f in result["functions"] if f.name == "sample_function")
        assert func.return_type  # Should have return type

    def test_parse_imports(self, parser, sample_python_file):
        """Test import extraction."""
        result = parser.parse_file(sample_python_file)

        assert "imports" in result
        # Imports are tracked

    def test_parse_syntax_error(self, parser, tmp_path):
        """Test handling of syntax errors."""
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken(:\n    pass")

        result = parser.parse_file(bad_file)

        # Should return empty result, not raise
        assert result["classes"] == []
        assert result["functions"] == []

    def test_parse_empty_file(self, parser, tmp_path):
        """Test parsing empty file."""
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")

        result = parser.parse_file(empty_file)

        assert result["classes"] == []
        assert result["functions"] == []

    def test_parse_nonexistent_file(self, parser, tmp_path):
        """Test parsing nonexistent file."""
        fake_path = tmp_path / "nonexistent.py"

        result = parser.parse_file(fake_path)

        assert result["classes"] == []
        assert result["functions"] == []


# ============================================================================
# JuliaParser Tests
# ============================================================================

class TestJuliaParser:
    """Tests for JuliaParser."""

    @pytest.fixture
    def parser(self):
        """Create Julia parser."""
        from agents.parsers import JuliaParser
        return JuliaParser()

    def test_extensions(self, parser):
        """Test supported extensions."""
        assert ".jl" in parser.extensions

    def test_language(self, parser):
        """Test language name."""
        assert parser.language == "julia"

    def test_parse_struct(self, parser, sample_julia_file):
        """Test struct extraction."""
        result = parser.parse_file(sample_julia_file)

        assert "classes" in result
        struct_names = [c.name for c in result["classes"]]
        assert "Point" in struct_names

    def test_parse_function(self, parser, sample_julia_file):
        """Test function extraction."""
        result = parser.parse_file(sample_julia_file)

        assert "functions" in result
        function_names = [f.name for f in result["functions"]]
        assert "calculate_distance" in function_names

    def test_parse_empty_file(self, parser, tmp_path):
        """Test parsing empty file."""
        empty_file = tmp_path / "empty.jl"
        empty_file.write_text("")

        result = parser.parse_file(empty_file)

        assert result["classes"] == []
        assert result["functions"] == []


# ============================================================================
# RParser Tests
# ============================================================================

class TestRParser:
    """Tests for RParser."""

    @pytest.fixture
    def parser(self):
        """Create R parser."""
        from agents.parsers import RParser
        return RParser()

    def test_extensions(self, parser):
        """Test supported extensions."""
        assert ".R" in parser.extensions
        assert ".r" in parser.extensions

    def test_language(self, parser):
        """Test language name."""
        assert parser.language == "r"

    def test_parse_function(self, parser, sample_r_file):
        """Test function extraction."""
        result = parser.parse_file(sample_r_file)

        assert "functions" in result
        function_names = [f.name for f in result["functions"]]
        assert "calculate_mean" in function_names
        assert "process_data" in function_names

    def test_parse_empty_file(self, parser, tmp_path):
        """Test parsing empty file."""
        empty_file = tmp_path / "empty.R"
        empty_file.write_text("")

        result = parser.parse_file(empty_file)

        assert result["functions"] == []


# ============================================================================
# JavaScriptParser Tests
# ============================================================================

class TestJavaScriptParser:
    """Tests for JavaScriptParser."""

    @pytest.fixture
    def parser(self):
        """Create JavaScript parser."""
        from agents.parsers import JavaScriptParser
        return JavaScriptParser()

    def test_extensions(self, parser):
        """Test supported extensions."""
        assert ".js" in parser.extensions
        assert ".ts" in parser.extensions
        assert ".jsx" in parser.extensions
        assert ".tsx" in parser.extensions

    def test_language(self, parser):
        """Test language name."""
        assert parser.language == "javascript"

    def test_parse_class(self, parser, sample_js_file):
        """Test class extraction."""
        result = parser.parse_file(sample_js_file)

        assert "classes" in result
        class_names = [c.name for c in result["classes"]]
        assert "Calculator" in class_names

    def test_parse_function(self, parser, sample_js_file):
        """Test function extraction."""
        result = parser.parse_file(sample_js_file)

        assert "functions" in result
        function_names = [f.name for f in result["functions"]]
        assert "multiply" in function_names

    def test_parse_arrow_function(self, parser, sample_js_file):
        """Test arrow function extraction."""
        result = parser.parse_file(sample_js_file)

        function_names = [f.name for f in result["functions"]]
        assert "divide" in function_names

    def test_parse_empty_file(self, parser, tmp_path):
        """Test parsing empty file."""
        empty_file = tmp_path / "empty.js"
        empty_file.write_text("")

        result = parser.parse_file(empty_file)

        assert result["classes"] == []
        assert result["functions"] == []


# ============================================================================
# ParserFactory Tests
# ============================================================================

class TestParserFactory:
    """Tests for ParserFactory."""

    def test_supported_languages(self, parser_factory):
        """Test all languages are registered."""
        languages = parser_factory.supported_languages

        assert "python" in languages
        assert "julia" in languages
        assert "r" in languages
        assert "javascript" in languages

    def test_supported_extensions(self, parser_factory):
        """Test all extensions are registered."""
        extensions = parser_factory.supported_extensions

        assert ".py" in extensions
        assert ".jl" in extensions
        assert ".R" in extensions or ".r" in extensions
        assert ".js" in extensions
        assert ".ts" in extensions

    def test_get_parser_python(self, parser_factory):
        """Test getting Python parser."""
        from agents.parsers import PythonParser

        parser = parser_factory.get_parser("python")

        assert parser is not None
        assert isinstance(parser, PythonParser)

    def test_get_parser_case_insensitive(self, parser_factory):
        """Test parser lookup is case insensitive."""
        parser1 = parser_factory.get_parser("Python")
        parser2 = parser_factory.get_parser("PYTHON")
        parser3 = parser_factory.get_parser("python")

        assert parser1 is parser2 is parser3

    def test_get_parser_unknown(self, parser_factory):
        """Test getting unknown parser returns None."""
        parser = parser_factory.get_parser("unknown_language")

        assert parser is None

    def test_get_parser_for_file_python(self, parser_factory, sample_python_file):
        """Test getting parser by file path."""
        parser = parser_factory.get_parser_for_file(sample_python_file)

        assert parser is not None
        assert parser.language == "python"

    def test_get_parser_for_file_julia(self, parser_factory, sample_julia_file):
        """Test getting parser for Julia file."""
        parser = parser_factory.get_parser_for_file(sample_julia_file)

        assert parser is not None
        assert parser.language == "julia"

    def test_get_parser_for_file_unknown(self, parser_factory, tmp_path):
        """Test getting parser for unknown extension."""
        unknown_file = tmp_path / "file.xyz"
        unknown_file.write_text("content")

        parser = parser_factory.get_parser_for_file(unknown_file)

        assert parser is None

    def test_detect_language_python(self, parser_factory, sample_python_file):
        """Test language detection for Python."""
        language = parser_factory.detect_language(sample_python_file)

        assert language == "python"

    def test_detect_language_unknown(self, parser_factory, tmp_path):
        """Test language detection for unknown extension."""
        unknown_file = tmp_path / "file.xyz"

        language = parser_factory.detect_language(unknown_file)

        assert language is None

    def test_parse_file_python(self, parser_factory, sample_python_file):
        """Test parsing Python file through factory."""
        result = parser_factory.parse_file(sample_python_file)

        assert "classes" in result
        assert "functions" in result
        assert len(result["classes"]) > 0

    def test_parse_file_unknown_extension(self, parser_factory, tmp_path):
        """Test parsing file with unknown extension."""
        unknown_file = tmp_path / "file.xyz"
        unknown_file.write_text("some content")

        result = parser_factory.parse_file(unknown_file)

        # Should return empty result
        assert result["classes"] == []
        assert result["functions"] == []


# ============================================================================
# ParserFactory Directory Parsing Tests
# ============================================================================

class TestParserFactoryDirectory:
    """Tests for ParserFactory directory parsing."""

    @pytest.fixture
    def sample_directory(self, tmp_path, sample_python_file, sample_julia_file):
        """Create a sample directory with multiple files."""
        # Create subdirectory
        subdir = tmp_path / "src"
        subdir.mkdir()

        # Create additional files
        (tmp_path / "main.py").write_text("def main(): pass")
        (tmp_path / "utils.py").write_text("def helper(): pass")
        (subdir / "model.py").write_text("class Model: pass")
        (tmp_path / "README.md").write_text("# Readme")  # Non-code file

        return tmp_path

    def test_parse_directory(self, parser_factory, sample_directory):
        """Test parsing a directory."""
        results = parser_factory.parse_directory(sample_directory)

        # Should find Python files
        assert len(results) > 0

        # All results should be from Python files
        for file_path, elements in results.items():
            assert file_path.suffix == ".py"

    def test_parse_directory_max_files(self, parser_factory, sample_directory):
        """Test max_files limit."""
        results = parser_factory.parse_directory(sample_directory, max_files=2)

        assert len(results) <= 2

    def test_parse_directory_skip_patterns(self, parser_factory, tmp_path):
        """Test skip directory patterns."""
        # Create node_modules directory (should be skipped)
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        (node_modules / "test.py").write_text("def npm_func(): pass")

        # Create regular file
        (tmp_path / "main.py").write_text("def main(): pass")

        results = parser_factory.parse_directory(
            tmp_path,
            skip_dirs=["node_modules"]
        )

        # Should not include files from node_modules
        for file_path in results.keys():
            assert "node_modules" not in str(file_path)

    def test_get_language_stats(self, parser_factory, sample_directory):
        """Test language statistics."""
        stats = parser_factory.get_language_stats(sample_directory)

        assert "python" in stats
        assert stats["python"] > 0

    def test_detect_primary_language(self, parser_factory, sample_directory):
        """Test primary language detection."""
        primary = parser_factory.detect_primary_language(sample_directory)

        assert primary == "python"  # Most files are Python


# ============================================================================
# LanguageParser Base Class Tests
# ============================================================================

class TestLanguageParser:
    """Tests for LanguageParser base class."""

    def test_parser_is_abstract(self):
        """Test LanguageParser cannot be instantiated."""
        from agents.parsers import LanguageParser

        with pytest.raises(TypeError):
            LanguageParser()

    def test_safe_read_utf8(self, tmp_path):
        """Test _safe_read with UTF-8 file."""
        from agents.parsers import PythonParser

        parser = PythonParser()
        test_file = tmp_path / "utf8.py"
        test_file.write_text("# Unicode: \u00e9\u00e0\u00f1", encoding="utf-8")

        content = parser._safe_read(test_file)

        assert "\u00e9" in content

    def test_safe_read_nonexistent(self, tmp_path):
        """Test _safe_read with nonexistent file."""
        from agents.parsers import PythonParser

        parser = PythonParser()
        fake_file = tmp_path / "nonexistent.py"

        content = parser._safe_read(fake_file)

        assert content == ""


# ============================================================================
# Integration Tests
# ============================================================================

class TestParserIntegration:
    """Integration tests for parsers."""

    @pytest.mark.integration
    def test_parse_real_python_code(self, parser_factory, tmp_path):
        """Test parsing realistic Python code."""
        code = '''
"""Module docstring."""

import os
import sys
from typing import List, Optional

CONSTANT = 42

class DataProcessor:
    """Process data from various sources."""

    def __init__(self, config: dict):
        self.config = config

    def process(self, data: List[dict]) -> Optional[dict]:
        """Process the data."""
        if not data:
            return None
        return {"processed": len(data)}

def main():
    """Main entry point."""
    processor = DataProcessor({})
    result = processor.process([{"x": 1}])
    print(result)

if __name__ == "__main__":
    main()
'''
        test_file = tmp_path / "processor.py"
        test_file.write_text(code)

        result = parser_factory.parse_file(test_file)

        # Check classes
        assert len(result["classes"]) == 1
        cls = result["classes"][0]
        assert cls.name == "DataProcessor"
        assert "__init__" in cls.methods
        assert "process" in cls.methods

        # Check functions
        function_names = [f.name for f in result["functions"]]
        assert "main" in function_names

    @pytest.mark.integration
    def test_multi_language_project(self, parser_factory, tmp_path):
        """Test parsing multi-language project."""
        # Create Python file
        (tmp_path / "main.py").write_text("class App: pass")

        # Create JavaScript file
        (tmp_path / "app.js").write_text("class App {}")

        # Create Julia file
        (tmp_path / "compute.jl").write_text("function compute(x) x^2 end")

        # Parse directory
        results = parser_factory.parse_directory(tmp_path)

        # Should find files from all languages
        extensions = {p.suffix for p in results.keys()}
        assert ".py" in extensions
        assert ".js" in extensions
        assert ".jl" in extensions

        # Check language stats
        stats = parser_factory.get_language_stats(tmp_path)
        assert "python" in stats
        assert "javascript" in stats
        assert "julia" in stats
