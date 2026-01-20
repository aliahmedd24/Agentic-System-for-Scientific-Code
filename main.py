#!/usr/bin/env python3
"""
Scientific Paper Analysis System - CLI Entry Point.

A multi-agent system for analyzing scientific papers and mapping
them to their code implementations.

Usage:
    python main.py analyze --paper <arxiv_id_or_url> --repo <github_url>
    python main.py server [--port 8000]
    python main.py --help
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.layout import Layout
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for better terminal output: pip install rich")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.orchestrator import PipelineOrchestrator, PipelineEvent, PipelineStage
from core.error_handling import SystemLogger, LogCategory


# Console for rich output (force_terminal for Windows compatibility)
console = Console(force_terminal=True, legacy_windows=False) if RICH_AVAILABLE else None


def print_banner():
    """Print the application banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ███████╗ ██████╗██╗███████╗███╗   ██╗████████╗██╗███████╗██╗ ██████╗   ║
║   ██╔════╝██╔════╝██║██╔════╝████╗  ██║╚══██╔══╝██║██╔════╝██║██╔════╝   ║
║   ███████╗██║     ██║█████╗  ██╔██╗ ██║   ██║   ██║█████╗  ██║██║        ║
║   ╚════██║██║     ██║██╔══╝  ██║╚██╗██║   ██║   ██║██╔══╝  ██║██║        ║
║   ███████║╚██████╗██║███████╗██║ ╚████║   ██║   ██║██║     ██║╚██████╗   ║
║   ╚══════╝ ╚═════╝╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝     ╚═╝ ╚═════╝   ║
║                                                                           ║
║              PAPER ANALYSIS SYSTEM v1.0.0                                 ║
║              Multi-Agent Scientific Paper Analyzer                        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""
    if RICH_AVAILABLE:
        console.print(banner, style="bold blue")
    else:
        print(banner)


def create_status_table(events: list[dict]) -> Table:
    """Create a status table from events."""
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Stage", style="cyan", width=20)
    table.add_column("Status", style="green", width=12)
    table.add_column("Progress", width=15)
    table.add_column("Message", style="white")
    
    stage_status = {}
    for event in events:
        stage = event.get("stage", "unknown")
        stage_status[stage] = {
            "progress": event.get("progress", 0),
            "message": event.get("message", "")
        }
    
    stage_order = [
        "initialized", "parsing_paper", "analyzing_repo",
        "mapping_concepts", "generating_code", "setting_up_env",
        "executing_code", "generating_report", "completed"
    ]
    
    for stage in stage_order:
        if stage in stage_status:
            info = stage_status[stage]
            progress = info["progress"]
            status = "✓ Done" if progress >= 100 else "⋯ Running" if progress > 0 else "○ Pending"
            progress_bar = f"[{'█' * (progress // 10)}{'░' * (10 - progress // 10)}] {progress}%"
            table.add_row(stage, status, progress_bar, info["message"][:50])
    
    return table


async def run_analysis(paper_source: str, repo_url: str, llm_provider: str, auto_execute: bool):
    """Run the full analysis pipeline with live progress display."""
    events = []
    
    async def event_callback(event: PipelineEvent):
        events.append({
            "stage": event.stage.value,
            "progress": event.progress,
            "message": event.message
        })
        
        if not RICH_AVAILABLE:
            print(f"[{event.stage.value}] {event.progress}% - {event.message}")
    
    if RICH_AVAILABLE:
        console.print()
        console.print(Panel.fit(
            f"[bold]Paper:[/bold] {paper_source}\n[bold]Repository:[/bold] {repo_url}\n[bold]LLM Provider:[/bold] {llm_provider}",
            title="📊 Analysis Configuration",
            border_style="blue"
        ))
        console.print()
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        llm_provider=llm_provider,
        event_callback=event_callback
    )
    
    # Run with progress display
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Analyzing...", total=100)
            
            # Run analysis
            result_task = asyncio.create_task(orchestrator.run(
                paper_source=paper_source,
                repo_url=repo_url,
                auto_execute=auto_execute
            ))
            
            # Update progress
            while not result_task.done():
                if events:
                    latest = events[-1]
                    progress.update(task, completed=latest["progress"], description=f"[cyan]{latest['stage']}")
                await asyncio.sleep(0.1)
            
            result = await result_task
            progress.update(task, completed=100)
    else:
        print("Starting analysis...")
        result = await orchestrator.run(
            paper_source=paper_source,
            repo_url=repo_url,
            auto_execute=auto_execute
        )
    
    return result


def display_results(result):
    """Display analysis results."""
    if not RICH_AVAILABLE:
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        if result.paper_data:
            print(f"\nPaper: {result.paper_data.get('title', 'Unknown')}")
        if result.mappings:
            print(f"Mappings found: {len(result.mappings)}")
        if result.report_paths:
            print(f"Reports: {result.report_paths}")
        return
    
    console.print()
    console.print(Panel.fit("✅ [bold green]Analysis Complete!", border_style="green"))
    
    # Paper info
    if result.paper_data:
        paper_info = Table(box=box.SIMPLE, show_header=False)
        paper_info.add_column("Field", style="cyan")
        paper_info.add_column("Value")
        
        paper_info.add_row("Title", result.paper_data.get("title", "Unknown")[:80])
        paper_info.add_row("Authors", ", ".join(result.paper_data.get("authors", [])[:3]))
        
        if result.paper_data.get("key_concepts"):
            concepts = result.paper_data["key_concepts"][:5]
            paper_info.add_row("Key Concepts", ", ".join(c.get("name", c) if isinstance(c, dict) else c for c in concepts))
        
        console.print(Panel(paper_info, title="📄 Paper Information", border_style="blue"))
    
    # Repository info
    if result.repo_data:
        repo_info = Table(box=box.SIMPLE, show_header=False)
        repo_info.add_column("Field", style="cyan")
        repo_info.add_column("Value")
        
        analysis = result.repo_data.get("analysis", {})
        repo_info.add_row("Architecture", analysis.get("architecture", "Unknown")[:60])
        repo_info.add_row("Setup Complexity", analysis.get("setup_complexity", "Unknown"))
        
        if result.repo_data.get("classes"):
            repo_info.add_row("Classes", str(len(result.repo_data["classes"])))
        if result.repo_data.get("functions"):
            repo_info.add_row("Functions", str(len(result.repo_data["functions"])))
        
        console.print(Panel(repo_info, title="📁 Repository Analysis", border_style="blue"))
    
    # Mappings
    if result.mappings:
        mapping_table = Table(box=box.ROUNDED, title="🔗 Concept-to-Code Mappings")
        mapping_table.add_column("Concept", style="cyan")
        mapping_table.add_column("Code Element", style="green")
        mapping_table.add_column("Confidence", style="yellow")
        mapping_table.add_column("Evidence", style="white")
        
        for mapping in result.mappings[:10]:
            confidence = mapping.get("confidence", 0)
            conf_str = f"{confidence:.0%}" if isinstance(confidence, float) else str(confidence)
            evidence = mapping.get("evidence", "")[:40]
            mapping_table.add_row(
                mapping.get("concept", "?")[:30],
                mapping.get("code_element", "?")[:30],
                conf_str,
                evidence
            )
        
        console.print(mapping_table)
        
        if len(result.mappings) > 10:
            console.print(f"[dim]... and {len(result.mappings) - 10} more mappings[/dim]")
    
    # Code execution results
    if result.code_results:
        exec_table = Table(box=box.ROUNDED, title="⚡ Code Execution Results")
        exec_table.add_column("Script", style="cyan")
        exec_table.add_column("Status", style="green")
        exec_table.add_column("Duration", style="yellow")
        
        for exec_result in result.code_results[:5]:
            status = "✓ Success" if exec_result.get("success") else "✗ Failed"
            duration = exec_result.get("execution_time", 0)
            exec_table.add_row(
                exec_result.get("script_name", "unknown")[:30],
                status,
                f"{duration:.2f}s"
            )
        
        console.print(exec_table)
    
    # Errors
    if result.errors:
        console.print()
        console.print(Panel(
            "\n".join(str(e)[:100] for e in result.errors[:5]),
            title="⚠️ Errors",
            border_style="yellow"
        ))
    
    # Report paths
    if result.report_paths:
        console.print()
        console.print(Panel.fit(
            "\n".join(f"📎 {p}" for p in result.report_paths),
            title="📊 Generated Reports",
            border_style="green"
        ))


def run_server(host: str, port: int):
    """Run the FastAPI server."""
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            f"Starting server at [bold]http://{host}:{port}[/bold]\n"
            f"API docs at [bold]http://{host}:{port}/docs[/bold]\n"
            f"Press [bold]Ctrl+C[/bold] to stop",
            title="🚀 Server Starting",
            border_style="green"
        ))
    else:
        print(f"Starting server at http://{host}:{port}")
    
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scientific Paper Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a paper from arXiv with its GitHub implementation
  python main.py analyze --paper 2301.00001 --repo https://github.com/user/repo
  
  # Use a specific LLM provider
  python main.py analyze --paper https://arxiv.org/abs/2301.00001 --repo https://github.com/user/repo --llm anthropic
  
  # Start the web server
  python main.py server --port 8080
  
  # Skip automatic code execution
  python main.py analyze --paper paper.pdf --repo https://github.com/user/repo --no-execute
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a paper and repository")
    analyze_parser.add_argument(
        "--paper", "-p",
        required=True,
        help="Paper source: arXiv ID (e.g., 2301.00001), URL, or local PDF path"
    )
    analyze_parser.add_argument(
        "--repo", "-r",
        required=True,
        help="GitHub repository URL"
    )
    analyze_parser.add_argument(
        "--llm", "-l",
        default="gemini",
        choices=["gemini", "anthropic", "openai"],
        help="LLM provider to use (default: gemini)"
    )
    analyze_parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Skip automatic code execution"
    )
    analyze_parser.add_argument(
        "--output", "-o",
        default="./outputs",
        help="Output directory for results"
    )
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start the web server")
    server_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    server_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to run on (default: 8000)"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        print_banner()
        parser.print_help()
        return
    
    print_banner()
    
    if args.command == "analyze":
        try:
            result = asyncio.run(run_analysis(
                paper_source=args.paper,
                repo_url=args.repo,
                llm_provider=args.llm,
                auto_execute=not args.no_execute
            ))
            display_results(result)
        except KeyboardInterrupt:
            if RICH_AVAILABLE:
                console.print("\n[yellow]Analysis cancelled by user[/yellow]")
            else:
                print("\nAnalysis cancelled by user")
            sys.exit(1)
        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"\n[red]Error: {e}[/red]")
            else:
                print(f"\nError: {e}")
            sys.exit(1)
    
    elif args.command == "server":
        try:
            run_server(args.host, args.port)
        except KeyboardInterrupt:
            if RICH_AVAILABLE:
                console.print("\n[yellow]Server stopped[/yellow]")
            else:
                print("\nServer stopped")


if __name__ == "__main__":
    main()
