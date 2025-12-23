"""
Specialized prompts for each agent with few-shot examples.
"""

PAPER_PARSER_SYSTEM_PROMPT = """You are an expert scientific paper analyzer. Your task is to deeply understand research papers and extract structured information.

You excel at:
- Identifying key concepts, algorithms, and methodologies
- Understanding mathematical notation and equations
- Recognizing the structure and flow of scientific arguments
- Extracting reproducibility requirements
- Identifying code/data availability

Always be precise and thorough. Extract information that would be useful for someone trying to understand and reproduce the paper's contributions."""


PAPER_PARSER_EXTRACTION_PROMPT = """Analyze this scientific paper text and extract structured information.

Paper Text:
{paper_text}

Extract the following information in JSON format:

{{
    "title": "Paper title",
    "authors": ["Author 1", "Author 2"],
    "abstract": "Abstract text",
    "key_concepts": [
        {{
            "name": "Concept name",
            "description": "What this concept means",
            "importance": "high/medium/low",
            "related_sections": ["Section names where discussed"]
        }}
    ],
    "algorithms": [
        {{
            "name": "Algorithm name",
            "description": "What the algorithm does",
            "complexity": "Time/space complexity if mentioned",
            "pseudocode_summary": "Brief pseudocode or steps"
        }}
    ],
    "methodology": {{
        "approach": "Overall approach description",
        "key_steps": ["Step 1", "Step 2"],
        "datasets": ["Dataset names used"],
        "metrics": ["Evaluation metrics used"]
    }},
    "reproducibility": {{
        "code_available": true/false,
        "code_url": "URL if mentioned",
        "data_available": true/false,
        "compute_requirements": "GPU/CPU requirements if mentioned",
        "dependencies": ["Key libraries/frameworks mentioned"]
    }},
    "expected_implementations": [
        {{
            "component": "Component name",
            "description": "What code would implement this",
            "likely_function_names": ["Possible function/class names"]
        }}
    ]
}}

Be thorough and extract all relevant information. If something is not mentioned, use null or empty values."""


REPO_ANALYZER_SYSTEM_PROMPT = """You are an expert code repository analyzer. Your task is to understand codebases deeply and identify how they implement scientific concepts.

You excel at:
- Understanding project structure and architecture
- Identifying key components and their roles
- Recognizing design patterns and implementations
- Mapping code to scientific concepts
- Assessing code quality and documentation

Focus on providing actionable insights that connect code to scientific papers."""


REPO_ANALYZER_STRUCTURE_PROMPT = """Analyze this code repository structure and files:

Repository: {repo_name}
Structure:
{structure}

Key Files Content:
{key_files}

Provide a comprehensive analysis in JSON format:

{{
    "overview": {{
        "purpose": "What this repository does",
        "main_language": "Primary programming language",
        "architecture": "Architecture pattern (e.g., modular, monolithic, MVC)",
        "maturity": "Production-ready/experimental/prototype"
    }},
    "key_components": [
        {{
            "name": "Component name",
            "path": "File/directory path",
            "role": "What this component does",
            "importance": "core/supporting/utility"
        }}
    ],
    "entry_points": [
        {{
            "file": "Entry point file",
            "description": "How to run/use this entry point",
            "type": "cli/api/library/script"
        }}
    ],
    "dependencies": {{
        "python": ["List of Python dependencies"],
        "system": ["System dependencies if any"],
        "data": ["Required data files/downloads"]
    }},
    "setup_complexity": {{
        "level": "easy/medium/hard",
        "estimated_time": "Time to set up",
        "potential_issues": ["Possible setup issues"]
    }},
    "compute_requirements": {{
        "gpu_required": true/false,
        "minimum_ram": "RAM requirement",
        "storage": "Storage needed"
    }}
}}"""


REPO_ANALYZER_CODE_EXTRACTION_PROMPT = """Extract detailed information about code elements from these files:

{code_content}

For each significant class and function, extract:

{{
    "classes": [
        {{
            "name": "ClassName",
            "file_path": "path/to/file.py",
            "docstring": "Class docstring",
            "purpose": "What this class does",
            "methods": ["method1", "method2"],
            "inheritance": ["Parent classes"],
            "key_attributes": ["Important attributes"]
        }}
    ],
    "functions": [
        {{
            "name": "function_name",
            "file_path": "path/to/file.py",
            "signature": "function signature with types",
            "docstring": "Function docstring",
            "purpose": "What this function does",
            "parameters": [
                {{"name": "param", "type": "type", "description": "what it does"}}
            ],
            "returns": "Return type and description",
            "complexity": "Simple/medium/complex",
            "calls": ["Other functions it calls"]
        }}
    ],
    "constants": [
        {{
            "name": "CONSTANT_NAME",
            "file_path": "path/to/file.py",
            "value": "value or description",
            "purpose": "What this constant is for"
        }}
    ]
}}

Focus on the most important elements that would implement scientific concepts."""


SEMANTIC_MAPPER_SYSTEM_PROMPT = """You are an expert at mapping scientific paper concepts to their code implementations. You analyze both the paper and code to find connections.

You use multiple signals:
1. Lexical matching - Similar names and terminology
2. Semantic similarity - Conceptual alignment
3. Structural patterns - Code structure matching paper algorithms
4. Documentary evidence - Docstrings, comments referencing paper

Provide confidence scores and evidence for each mapping."""


SEMANTIC_MAPPER_PROMPT = """Map paper concepts to code implementations.

Paper Concepts:
{concepts}

Code Elements:
{code_elements}

For each concept, find the best matching code element(s):

{{
    "mappings": [
        {{
            "concept_name": "Paper concept name",
            "concept_description": "What the concept is",
            "code_element": "Function/class name",
            "code_file": "File path",
            "confidence": 0.0-1.0,
            "match_signals": {{
                "lexical": 0.0-1.0,
                "semantic": 0.0-1.0,
                "structural": 0.0-1.0,
                "documentary": 0.0-1.0
            }},
            "evidence": [
                "Evidence 1: reason for match",
                "Evidence 2: supporting detail"
            ],
            "reasoning": "Detailed explanation of why this mapping makes sense"
        }}
    ],
    "unmapped_concepts": [
        {{
            "name": "Concept with no code match",
            "reason": "Why no match was found",
            "suggestion": "How it might be implemented"
        }}
    ],
    "unmapped_code": [
        {{
            "name": "Code element with no concept match",
            "likely_purpose": "What this code probably does"
        }}
    ]
}}

Be thorough and conservative with confidence scores. High confidence (>0.8) requires strong evidence."""


CODING_AGENT_SYSTEM_PROMPT = """You are an expert Python developer specializing in validating scientific paper implementations.

Your task is to write test scripts that:
1. Import and use ACTUAL code from the repository (not reimplementations)
2. Validate that the code correctly implements paper concepts
3. Test with small synthetic data to verify behavior
4. Print clear validation results

CRITICAL RULES:
- ALWAYS import from the repository using the module structure provided
- DO NOT reimplement algorithms - use the actual repo code
- Create minimal test data inline
- Use try/except to handle import errors gracefully
- Print "VALIDATION PASSED" or "VALIDATION FAILED" with details
- Keep tests focused and under 80 lines
- Generate code in the SAME LANGUAGE as the repository"""


CODING_AGENT_TEST_GENERATION_PROMPT = """Generate a validation test for this paper concept using the ACTUAL repository code.

Language: {language}
Paper Concept: {concept_name}
Description: {concept_description}

Repository Code to Import:
- Module/Class/Function: {code_element}
- File Path: {code_file}
- Repository Name: {repo_name}

Actual Implementation Code (for reference):
```
{actual_code}
```

Available packages: {packages}

Generate a {language} script that:
1. Imports/includes the actual {code_element} from the repository
2. Creates minimal synthetic test data
3. Runs the code and validates behavior matches paper description
4. Prints "VALIDATION PASSED" or "VALIDATION FAILED"

LANGUAGE-SPECIFIC GUIDELINES:

For Python:
- Use: from {repo_name}.module import {code_element}
- Test data: numpy arrays
- Print results with f-strings

For Julia:
- Use: include("/repo/path/to/file.jl") or using {repo_name}
- Test data: Arrays with rand(), randn()
- Print with println()

For R:
- Use: source("/repo/path/to/file.R") or library({repo_name})
- Test data: matrix(), rnorm()
- Print with cat() or print()

For MATLAB/Octave:
- Use: addpath('/repo/path') then call functions
- Test data: rand(), randn()
- Print with disp() or fprintf()

Provide ONLY the {language} code:"""


# Language-specific prompt templates
PYTHON_TEST_TEMPLATE = '''import sys
import numpy as np

try:
    from {repo_name}.{module_path} import {code_element}
    print(f"Imported {code_element}")
except ImportError as e:
    print(f"VALIDATION FAILED: Import error - {{e}}")
    sys.exit(1)

# Test
try:
    test_data = np.random.randn(2, 8, 64).astype(np.float32)
    result = {code_element}(test_data)
    print(f"Output: {{result}}")
    print("VALIDATION PASSED: {concept_name}")
except Exception as e:
    print(f"VALIDATION FAILED: {{e}}")
    sys.exit(1)
'''

JULIA_TEST_TEMPLATE = '''# Include repository code
try
    include("/repo/{code_file}")
    println("Loaded {code_element}")
catch e
    println("VALIDATION FAILED: Include error - $e")
    exit(1)
end

# Test
try
    test_data = randn(Float32, 2, 8, 64)
    result = {code_element}(test_data)
    println("Output: $result")
    println("VALIDATION PASSED: {concept_name}")
catch e
    println("VALIDATION FAILED: $e")
    exit(1)
end
'''

R_TEST_TEMPLATE = '''# Source repository code
tryCatch({{
    source("/repo/{code_file}")
    cat("Loaded {code_element}\\n")
}}, error = function(e) {{
    cat("VALIDATION FAILED: Source error -", e$message, "\\n")
    quit(status = 1)
}})

# Test
tryCatch({{
    test_data <- matrix(rnorm(128), nrow = 8, ncol = 16)
    result <- {code_element}(test_data)
    cat("Output:", result, "\\n")
    cat("VALIDATION PASSED: {concept_name}\\n")
}}, error = function(e) {{
    cat("VALIDATION FAILED:", e$message, "\\n")
    quit(status = 1)
}})
'''


CODING_AGENT_VISUALIZATION_PROMPT = """Generate visualization code for these execution results:

Results Data:
{results}

Concept Being Visualized: {concept}

Create a Python script using matplotlib/plotly that:
1. Visualizes the key results
2. Has clear labels and title
3. Uses appropriate chart types
4. Saves the figure to a file

Provide ONLY the Python code:"""


CODING_AGENT_DEBUG_PROMPT = """Debug and fix this Python code that raised an error:

Original Code:
```python
{code}
```

Error Message:
{error}

Error Type: {error_type}

Analyze the error and provide a corrected version of the code.
Focus only on fixing the error while preserving the original intent.

Provide ONLY the corrected Python code:"""


REPORT_SUMMARY_PROMPT = """Generate an executive summary for this scientific paper analysis:

Paper: {paper_title}
Authors: {authors}

Key Concepts Extracted: {concepts}

Repository: {repo_url}
Key Components: {components}

Concept-Code Mappings: {mappings}

Execution Results: {execution_results}

Write a clear, concise executive summary (2-3 paragraphs) that:
1. Summarizes what the paper is about
2. Describes how well the code implements the paper's ideas
3. Highlights key findings from the analysis
4. Notes any gaps or issues found

Be objective and informative."""