# Scientific Paper Analysis System

A production-ready multi-agent system for analyzing scientific papers and mapping them to their code implementations.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)

## Overview

This system automates the analysis of scientific papers (from arXiv or PDF uploads) and their corresponding code repositories. It uses a multi-agent architecture powered by LLMs to:

1. **Parse Papers** - Extract key concepts, algorithms, methodology, and implementation requirements
2. **Analyze Repositories** - Scan code structure, parse Python AST, identify classes and functions
3. **Semantic Mapping** - Map paper concepts to code implementations using multi-signal analysis
4. **Code Generation** - Generate and execute validation/test scripts
5. **Report Generation** - Create comprehensive HTML reports with interactive knowledge graph visualization

## Features

- 🤖 **Multi-Agent Architecture** - Specialized agents for each pipeline stage
- 🔌 **Multi-LLM Support** - Works with Gemini, Claude, and OpenAI
- 📊 **Knowledge Graph** - NetworkX-based graph for tracking relationships
- 🎨 **Modern UI** - Dark theme with glassmorphism design
- 📡 **Real-time Updates** - WebSocket-based progress streaming
- 🔒 **Safe Execution** - Docker sandbox for running generated code
- 📝 **Comprehensive Reports** - HTML reports with D3.js visualizations

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional, for sandboxed code execution)
- At least one LLM API key (Gemini, Anthropic, or OpenAI)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/scientific-agent-system.git
cd scientific-agent-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running the Application

#### Web Interface

```bash
# Start the server
python main.py server --port 8000

# Open http://localhost:8000 in your browser
```

#### Command Line

```bash
# Analyze a paper from arXiv with its GitHub implementation
python main.py analyze --paper 2301.00001 --repo https://github.com/user/repo

# Use a specific LLM provider
python main.py analyze --paper https://arxiv.org/abs/2301.00001 --repo https://github.com/user/repo --llm anthropic

# Skip automatic code execution
python main.py analyze --paper paper.pdf --repo https://github.com/user/repo --no-execute
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop
docker-compose down
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Pipeline Orchestrator                    │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────┤
│   Paper     │    Repo     │  Semantic   │   Coding    │ Report  │
│   Parser    │  Analyzer   │   Mapper    │   Agent     │ Engine  │
│   Agent     │   Agent     │             │             │         │
└─────┬───────┴──────┬──────┴──────┬──────┴──────┬──────┴────┬────┘
      │              │             │             │           │
      └──────────────┴──────┬──────┴─────────────┴───────────┘
                           │
                   ┌───────┴───────┐
                   │  Knowledge    │
                   │    Graph      │
                   └───────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **PaperParserAgent** | Extracts and analyzes scientific papers from arXiv, URLs, or PDFs |
| **RepoAnalyzerAgent** | Clones and analyzes code repositories using AST parsing |
| **SemanticMapper** | Maps paper concepts to code using lexical, semantic, and documentary signals |
| **CodingAgent** | Generates and executes validation scripts with Docker sandboxing |
| **ReportGenerator** | Creates comprehensive HTML reports with D3.js knowledge graph visualization |
| **KnowledgeGraph** | NetworkX-based graph storing all discovered relationships |
| **PipelineOrchestrator** | Coordinates agents and manages pipeline execution |

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google Gemini API key | One of these |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | must be |
| `OPENAI_API_KEY` | OpenAI API key | provided |
| `GITHUB_TOKEN` | GitHub token for private repos | Optional |
| `PORT` | Server port (default: 8000) | Optional |
| `LOG_LEVEL` | Logging level (default: INFO) | Optional |

### LLM Providers

The system supports multiple LLM providers:

- **Gemini** (default) - `gemini-2.0-flash-exp`
- **Anthropic** - `claude-sonnet-4`
- **OpenAI** - `gpt-4o`

## API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze` | POST | Start new analysis |
| `/api/analyze/upload` | POST | Start analysis with PDF upload |
| `/api/jobs` | GET | List all jobs |
| `/api/jobs/{id}/status` | GET | Get job status |
| `/api/jobs/{id}/result` | GET | Get job result |
| `/api/jobs/{id}/report` | GET | Download HTML report |
| `/api/jobs/{id}/knowledge-graph` | GET | Get knowledge graph data |
| `/api/jobs/{id}` | DELETE | Cancel job |
| `/health` | GET | Health check |

### WebSocket

Connect to `/ws/{job_id}` for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/your-job-id');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data.stage, data.progress, data.message);
};
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_knowledge_graph.py

# Run excluding slow tests
pytest -m "not slow"
```

### Code Quality

```bash
# Format code
black .

# Sort imports
isort .

# Type checking
mypy .
```

### Project Structure

```
scientific-agent-system/
├── agents/                 # Agent implementations
│   ├── base_agent.py
│   ├── paper_parser_agent.py
│   ├── repo_analyzer_agent.py
│   ├── semantic_mapper.py
│   └── coding_agent.py
├── api/                    # FastAPI server
│   └── server.py
├── core/                   # Core infrastructure
│   ├── error_handling.py
│   ├── llm_client.py
│   ├── knowledge_graph.py
│   ├── orchestrator.py
│   └── agent_prompts.py
├── reports/                # Report generation
│   └── template_engine.py
├── ui/                     # Web UI
│   └── index.html
├── tests/                  # Test suite
├── main.py                 # CLI entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Examples

### Analyzing a Transformer Paper

```bash
python main.py analyze \
    --paper "Attention Is All You Need" \
    --repo https://github.com/tensorflow/tensor2tensor \
    --llm gemini
```

### Analyzing a Local PDF

```bash
python main.py analyze \
    --paper /path/to/paper.pdf \
    --repo https://github.com/author/implementation \
    --no-execute
```

### Using the API

```python
import requests

# Start analysis
response = requests.post("http://localhost:8000/api/analyze", json={
    "paper_source": "2301.00001",
    "repo_url": "https://github.com/user/repo",
    "llm_provider": "gemini",
    "auto_execute": True
})

job_id = response.json()["job_id"]

# Check status
status = requests.get(f"http://localhost:8000/api/jobs/{job_id}/status").json()
print(f"Stage: {status['stage']}, Progress: {status['progress']}%")
```

## Troubleshooting

### Common Issues

**API Key Not Found**
```
Error: No API key configured for provider
```
Solution: Ensure the appropriate API key is set in your `.env` file.

**PDF Parsing Failed**
```
Error: Failed to extract text from PDF
```
Solution: The system uses multiple PDF backends (PyMuPDF, pdfplumber, pypdf). If one fails, it tries the next. Ensure the PDF is not corrupted or password-protected.

**Repository Clone Failed**
```
Error: Failed to clone repository
```
Solution: Check the repository URL is correct. For private repositories, ensure `GITHUB_TOKEN` is set.

**Docker Sandbox Unavailable**
```
Warning: Docker not available, falling back to subprocess
```
Solution: This is not an error - code will execute in a subprocess with limited isolation. Install Docker for full sandboxing.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Knowledge graph powered by [NetworkX](https://networkx.org/)
- UI styling inspired by modern glassmorphism design
- LLM integrations via official provider SDKs
