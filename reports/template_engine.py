"""
Report Generation Engine.

Generates comprehensive HTML reports with D3.js knowledge graph visualization.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from jinja2 import Environment, FileSystemLoader, BaseLoader


# ============================================================================
# Report Templates (embedded for portability)
# ============================================================================

REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Report: {{ paper.title | default('Scientific Paper') }}</title>
    <style>
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a25;
            --accent-primary: #6366f1;
            --accent-secondary: #818cf8;
            --accent-glow: rgba(99, 102, 241, 0.3);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
            --border-color: rgba(255, 255, 255, 0.1);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        /* Header */
        .report-header {
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
            border-radius: 20px;
            border: 1px solid var(--border-color);
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }
        
        .report-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
        }
        
        .report-header h1 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--text-primary), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .report-header .subtitle {
            color: var(--text-secondary);
            font-size: 1rem;
        }
        
        .meta-info {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 1.5rem;
            flex-wrap: wrap;
        }
        
        .meta-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--text-secondary);
            font-size: 0.875rem;
        }
        
        /* Sections */
        .section {
            background: var(--bg-secondary);
            border-radius: 16px;
            border: 1px solid var(--border-color);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .section-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border-color);
        }
        
        .section-header h2 {
            font-size: 1.25rem;
            font-weight: 600;
        }
        
        .section-icon {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            background: var(--accent-glow);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--accent-primary);
        }
        
        /* Cards */
        .card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
        }
        
        .card {
            background: var(--bg-tertiary);
            border-radius: 12px;
            padding: 1.25rem;
            border: 1px solid var(--border-color);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }
        
        .card h3 {
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .card p {
            color: var(--text-secondary);
            font-size: 0.875rem;
        }
        
        /* Mappings Table */
        .mappings-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .mappings-table th,
        .mappings-table td {
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        .mappings-table th {
            background: var(--bg-tertiary);
            font-weight: 600;
            font-size: 0.875rem;
            color: var(--text-secondary);
        }
        
        .mappings-table td {
            font-size: 0.875rem;
        }
        
        .confidence-bar {
            height: 6px;
            background: var(--bg-primary);
            border-radius: 3px;
            overflow: hidden;
            width: 100px;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-primary), var(--success));
            border-radius: 3px;
        }
        
        /* Code Blocks */
        .code-block {
            background: var(--bg-primary);
            border-radius: 8px;
            padding: 1rem;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.8125rem;
            overflow-x: auto;
            border: 1px solid var(--border-color);
        }
        
        /* Knowledge Graph */
        #knowledge-graph {
            width: 100%;
            height: 500px;
            background: var(--bg-primary);
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }
        
        .graph-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-top: 1rem;
            justify-content: center;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.75rem;
            color: var(--text-secondary);
        }
        
        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        
        /* Results */
        .result-item {
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }
        
        .result-status {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 500;
        }
        
        .status-success { background: rgba(34, 197, 94, 0.2); color: var(--success); }
        .status-error { background: rgba(239, 68, 68, 0.2); color: var(--error); }
        .status-warning { background: rgba(245, 158, 11, 0.2); color: var(--warning); }
        
        /* Visualizations */
        .visualization-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
        }
        
        .visualization-item {
            background: var(--bg-tertiary);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }
        
        .visualization-item img {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .visualization-caption {
            padding: 0.75rem;
            font-size: 0.875rem;
            color: var(--text-secondary);
        }
        
        /* Footer */
        .report-footer {
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            font-size: 0.875rem;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .container { padding: 1rem; }
            .report-header h1 { font-size: 1.5rem; }
            .meta-info { gap: 1rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="report-header">
            <h1>{{ paper.title | default('Analysis Report') }}</h1>
            <p class="subtitle">Scientific Paper Analysis Report</p>
            <div class="meta-info">
                <div class="meta-item">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
                        <line x1="16" y1="2" x2="16" y2="6"></line>
                        <line x1="8" y1="2" x2="8" y2="6"></line>
                        <line x1="3" y1="10" x2="21" y2="10"></line>
                    </svg>
                    {{ generated_at }}
                </div>
                <div class="meta-item">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2L2 7l10 5 10-5-10-5z"></path>
                        <path d="M2 17l10 5 10-5"></path>
                        <path d="M2 12l10 5 10-5"></path>
                    </svg>
                    {{ mappings | length }} Concept Mappings
                </div>
                <div class="meta-item">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="16 18 22 12 16 6"></polyline>
                        <polyline points="8 6 2 12 8 18"></polyline>
                    </svg>
                    {{ code_results | length }} Code Results
                </div>
            </div>
        </header>
        
        <!-- Paper Overview -->
        <section class="section">
            <div class="section-header">
                <div class="section-icon">📄</div>
                <h2>Paper Overview</h2>
            </div>
            
            {% if paper.authors %}
            <p style="margin-bottom: 0.75rem; color: var(--text-secondary);">
                <strong>Authors:</strong> {{ paper.authors | join(', ') }}
            </p>
            {% endif %}
            
            {% if paper.abstract %}
            <div class="card">
                <h3>Abstract</h3>
                <p>{{ paper.abstract }}</p>
            </div>
            {% endif %}
            
            {% if paper.key_concepts %}
            <h3 style="margin-top: 1rem; margin-bottom: 0.75rem;">Key Concepts</h3>
            <div class="card-grid">
                {% for concept in paper.key_concepts %}
                <div class="card">
                    <h3>{{ concept.name }}</h3>
                    <p>{{ concept.description | default('') }}</p>
                </div>
                {% endfor %}
            </div>
            {% endif %}
        </section>
        
        <!-- Repository Analysis -->
        <section class="section">
            <div class="section-header">
                <div class="section-icon">📁</div>
                <h2>Repository Analysis</h2>
            </div>
            
            {% if repository %}
            <div class="card-grid">
                <div class="card">
                    <h3>Repository</h3>
                    <p>{{ repository.url | default('N/A') }}</p>
                </div>
                <div class="card">
                    <h3>Primary Language</h3>
                    <p>{{ repository.primary_language | default('Python') }}</p>
                </div>
                <div class="card">
                    <h3>Total Files</h3>
                    <p>{{ repository.total_files | default(0) }}</p>
                </div>
                <div class="card">
                    <h3>Setup Complexity</h3>
                    <p>{{ repository.setup_complexity | default('Medium') }}</p>
                </div>
            </div>
            
            {% if repository.key_components %}
            <h3 style="margin-top: 1.5rem; margin-bottom: 0.75rem;">Key Components</h3>
            <div class="card-grid">
                {% for component in repository.key_components[:6] %}
                <div class="card">
                    <h3>{{ component.name }}</h3>
                    <p>{{ component.description | default('') }}</p>
                </div>
                {% endfor %}
            </div>
            {% endif %}
            {% endif %}
        </section>
        
        <!-- Concept-to-Code Mappings -->
        <section class="section">
            <div class="section-header">
                <div class="section-icon">🔗</div>
                <h2>Concept-to-Code Mappings</h2>
            </div>
            
            {% if mappings %}
            <div style="overflow-x: auto;">
                <table class="mappings-table">
                    <thead>
                        <tr>
                            <th>Concept</th>
                            <th>Code Implementation</th>
                            <th>File</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for mapping in mappings %}
                        <tr>
                            <td><strong>{{ mapping.concept }}</strong></td>
                            <td><code>{{ mapping.code_element }}</code></td>
                            <td style="color: var(--text-muted);">{{ mapping.file_path | default('') }}</td>
                            <td>
                                <div style="display: flex; align-items: center; gap: 0.5rem;">
                                    <div class="confidence-bar">
                                        <div class="confidence-fill" style="width: {{ (mapping.confidence * 100) | int }}%;"></div>
                                    </div>
                                    <span style="font-size: 0.75rem; color: var(--text-secondary);">
                                        {{ (mapping.confidence * 100) | int }}%
                                    </span>
                                </div>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            {% else %}
            <p style="color: var(--text-secondary);">No mappings found.</p>
            {% endif %}
        </section>
        
        <!-- Code Execution Results -->
        <section class="section">
            <div class="section-header">
                <div class="section-icon">⚡</div>
                <h2>Code Execution Results</h2>
            </div>
            
            {% if code_results %}
            {% for result in code_results %}
            <div class="result-item">
                <div class="result-header">
                    <h3>{{ result.script_name | default('Test Script') }}</h3>
                    <span class="result-status {{ 'status-success' if result.success else 'status-error' }}">
                        {{ 'Success' if result.success else 'Failed' }}
                    </span>
                </div>
                
                {% if result.stdout %}
                <div class="code-block" style="max-height: 200px; overflow-y: auto;">
                    <pre>{{ result.stdout }}</pre>
                </div>
                {% endif %}
                
                {% if result.error and not result.success %}
                <div class="code-block" style="border-color: var(--error); margin-top: 0.5rem;">
                    <pre style="color: var(--error);">{{ result.error }}</pre>
                </div>
                {% endif %}
            </div>
            {% endfor %}
            {% else %}
            <p style="color: var(--text-secondary);">No code execution results.</p>
            {% endif %}
        </section>
        
        <!-- Visualizations -->
        {% if visualizations %}
        <section class="section">
            <div class="section-header">
                <div class="section-icon">📊</div>
                <h2>Generated Visualizations</h2>
            </div>
            
            <div class="visualization-grid">
                {% for viz in visualizations %}
                <div class="visualization-item">
                    <img src="{{ viz.path }}" alt="{{ viz.caption | default('Visualization') }}">
                    <div class="visualization-caption">{{ viz.caption | default('') }}</div>
                </div>
                {% endfor %}
            </div>
        </section>
        {% endif %}
        
        <!-- Knowledge Graph -->
        <section class="section">
            <div class="section-header">
                <div class="section-icon">🌐</div>
                <h2>Knowledge Graph</h2>
            </div>
            
            <div id="knowledge-graph"></div>
            
            <div class="graph-legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #6366f1;"></div>
                    Paper
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #22c55e;"></div>
                    Concept
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #f59e0b;"></div>
                    Algorithm
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #06b6d4;"></div>
                    Repository
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ec4899;"></div>
                    Function/Class
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #8b5cf6;"></div>
                    Mapping
                </div>
            </div>
        </section>
        
        <!-- Footer -->
        <footer class="report-footer">
            <p>Generated by Scientific Paper Analysis System</p>
            <p>{{ generated_at }}</p>
        </footer>
    </div>
    
    <!-- D3.js Knowledge Graph Visualization -->
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
        const graphData = {{ knowledge_graph_json | safe }};
        
        if (graphData && graphData.nodes && graphData.nodes.length > 0) {
            const container = document.getElementById('knowledge-graph');
            const width = container.clientWidth;
            const height = 500;
            
            const nodeColors = {
                paper: '#6366f1',
                concept: '#22c55e',
                algorithm: '#f59e0b',
                repository: '#06b6d4',
                function: '#ec4899',
                class: '#ec4899',
                mapping: '#8b5cf6',
                file: '#64748b',
                default: '#94a3b8'
            };
            
            const svg = d3.select('#knowledge-graph')
                .append('svg')
                .attr('width', width)
                .attr('height', height);
            
            const g = svg.append('g');
            
            // Zoom
            svg.call(d3.zoom()
                .extent([[0, 0], [width, height]])
                .scaleExtent([0.1, 4])
                .on('zoom', (event) => g.attr('transform', event.transform)));
            
            // Simulation
            const simulation = d3.forceSimulation(graphData.nodes)
                .force('link', d3.forceLink(graphData.links).id(d => d.id).distance(80))
                .force('charge', d3.forceManyBody().strength(-200))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(30));
            
            // Links
            const link = g.append('g')
                .attr('stroke', '#334155')
                .attr('stroke-opacity', 0.6)
                .selectAll('line')
                .data(graphData.links)
                .join('line')
                .attr('stroke-width', 1);
            
            // Nodes
            const node = g.append('g')
                .selectAll('g')
                .data(graphData.nodes)
                .join('g')
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended));
            
            node.append('circle')
                .attr('r', d => d.type === 'paper' ? 12 : 8)
                .attr('fill', d => nodeColors[d.type] || nodeColors.default)
                .attr('stroke', '#0a0a0f')
                .attr('stroke-width', 2);
            
            node.append('text')
                .text(d => d.label ? (d.label.length > 15 ? d.label.slice(0, 15) + '...' : d.label) : '')
                .attr('x', 12)
                .attr('y', 4)
                .attr('fill', '#94a3b8')
                .attr('font-size', '10px');
            
            // Tooltip
            node.append('title')
                .text(d => `${d.type}: ${d.label || d.id}`);
            
            simulation.on('tick', () => {
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
                
                node.attr('transform', d => `translate(${d.x},${d.y})`);
            });
            
            function dragstarted(event) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }
            
            function dragged(event) {
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }
            
            function dragended(event) {
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }
        } else {
            document.getElementById('knowledge-graph').innerHTML = 
                '<p style="text-align: center; padding: 2rem; color: #64748b;">No knowledge graph data available</p>';
        }
    </script>
</body>
</html>
"""


# ============================================================================
# Report Generator Class
# ============================================================================

class ReportGenerator:
    """Generates comprehensive HTML reports from analysis results."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path(__file__).parent / "generated"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Jinja environment with string loader
        self.env = Environment(loader=BaseLoader())
        self.template = self.env.from_string(REPORT_TEMPLATE)
    
    async def generate(
        self,
        job_id: str,
        paper_data: Optional[dict] = None,
        repo_data: Optional[dict] = None,
        mappings: Optional[list] = None,
        code_results: Optional[list] = None,
        knowledge_graph: Optional[Any] = None,
        visualizations: Optional[list] = None
    ) -> Path:
        """Generate a complete HTML report."""
        
        # Prepare knowledge graph JSON
        kg_data = {"nodes": [], "links": []}
        if knowledge_graph:
            try:
                kg_data = knowledge_graph.to_d3_format()
            except Exception:
                pass
        
        # Prepare template context
        context = {
            "job_id": job_id,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "paper": paper_data or {},
            "repository": repo_data or {},
            "mappings": mappings or [],
            "code_results": code_results or [],
            "visualizations": visualizations or [],
            "knowledge_graph_json": json.dumps(kg_data)
        }
        
        # Render template
        html_content = self.template.render(**context)
        
        # Write report file
        report_path = self.output_dir / f"{job_id}_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return report_path
    
    def generate_summary_card(
        self,
        paper_title: str,
        num_concepts: int,
        num_mappings: int,
        num_successful: int,
        num_failed: int
    ) -> str:
        """Generate a summary card HTML snippet."""
        return f"""
        <div class="summary-card">
            <h2>{paper_title}</h2>
            <div class="stats">
                <div class="stat">
                    <span class="stat-value">{num_concepts}</span>
                    <span class="stat-label">Concepts</span>
                </div>
                <div class="stat">
                    <span class="stat-value">{num_mappings}</span>
                    <span class="stat-label">Mappings</span>
                </div>
                <div class="stat">
                    <span class="stat-value">{num_successful}</span>
                    <span class="stat-label">Successful</span>
                </div>
                <div class="stat">
                    <span class="stat-value">{num_failed}</span>
                    <span class="stat-label">Failed</span>
                </div>
            </div>
        </div>
        """


# ============================================================================
# Standalone usage
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test_report():
        generator = ReportGenerator()
        
        # Sample data
        paper_data = {
            "title": "Test Paper: A Novel Approach",
            "authors": ["Alice Smith", "Bob Jones"],
            "abstract": "This paper presents a novel approach to solving complex problems.",
            "key_concepts": [
                {"name": "Concept A", "description": "Description of concept A"},
                {"name": "Concept B", "description": "Description of concept B"}
            ]
        }
        
        repo_data = {
            "url": "https://github.com/example/repo",
            "primary_language": "Python",
            "total_files": 42,
            "setup_complexity": "Medium",
            "key_components": [
                {"name": "Model", "description": "Main model implementation"},
                {"name": "Training", "description": "Training loop"}
            ]
        }
        
        mappings = [
            {"concept": "Concept A", "code_element": "ClassA", "file_path": "model.py", "confidence": 0.85},
            {"concept": "Concept B", "code_element": "function_b", "file_path": "utils.py", "confidence": 0.72}
        ]
        
        code_results = [
            {"script_name": "test_concept_a.py", "success": True, "stdout": "All tests passed!"},
            {"script_name": "test_concept_b.py", "success": False, "error": "ImportError: No module named 'xyz'"}
        ]
        
        path = await generator.generate(
            job_id="test123",
            paper_data=paper_data,
            repo_data=repo_data,
            mappings=mappings,
            code_results=code_results
        )
        
        print(f"Report generated: {path}")
    
    asyncio.run(test_report())
