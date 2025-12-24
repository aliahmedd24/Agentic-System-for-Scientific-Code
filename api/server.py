"""
FastAPI server for Scientific Paper Analysis System.

Provides REST API endpoints and WebSocket for real-time updates.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestrator import PipelineOrchestrator, PipelineStage, PipelineEvent
from core.knowledge_graph import KnowledgeGraph
from core.error_handling import SystemLogger, LogLevel, LogCategory
from core.metrics import get_metrics_collector


# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title="Scientific Paper Analysis System",
    description="Multi-agent system for analyzing scientific papers and their implementations",
    version="1.0.0"
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (UI)
UI_DIR = Path(__file__).parent.parent / "ui"
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

# Directories
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
REPORTS_DIR = Path(__file__).parent.parent / "reports" / "generated"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Logger
logger = SystemLogger()


# ============================================================================
# Connection Manager for WebSockets
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
        logger.info(f"WebSocket connected for job {job_id}", category=LogCategory.SYSTEM)
    
    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            if websocket in self.active_connections[job_id]:
                self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
        logger.info(f"WebSocket disconnected for job {job_id}", category=LogCategory.SYSTEM)
    
    async def broadcast(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)
            for conn in disconnected:
                self.disconnect(conn, job_id)


manager = ConnectionManager()


# ============================================================================
# Job Storage (in-memory for simplicity)
# ============================================================================

class JobStatus:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = "pending"
        self.stage = "initialized"
        self.progress = 0
        self.events: list[dict] = []
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.paper_source: Optional[str] = None
        self.repo_url: Optional[str] = None


jobs: dict[str, JobStatus] = {}


# ============================================================================
# Request/Response Models
# ============================================================================

class AnalysisRequest(BaseModel):
    """Request model for starting an analysis."""
    paper_source: str = Field(..., description="arXiv ID, URL, or 'upload' for file upload")
    repo_url: str = Field(..., description="GitHub repository URL")
    llm_provider: str = Field(default="gemini", description="LLM provider: gemini, anthropic, openai")
    auto_execute: bool = Field(default=True, description="Automatically execute generated code")


class JobResponse(BaseModel):
    """Response model for job creation."""
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    stage: str
    progress: int
    events: list[dict]
    error: Optional[str]
    created_at: str
    updated_at: str


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Serve the main UI."""
    index_file = UI_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Scientific Paper Analysis System API", "docs": "/docs"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len([j for j in jobs.values() if j.status == "running"])
    }


@app.post("/api/analyze", response_model=JobResponse)
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Start a new paper analysis job."""
    job_id = str(uuid.uuid4())[:8]
    
    job = JobStatus(job_id)
    job.paper_source = request.paper_source
    job.repo_url = request.repo_url
    jobs[job_id] = job
    
    logger.info(
        f"Starting analysis job {job_id}",
        category=LogCategory.SYSTEM,
        context={"paper": request.paper_source, "repo": request.repo_url}
    )
    
    # Run analysis in background
    background_tasks.add_task(
        run_analysis,
        job_id,
        request.paper_source,
        request.repo_url,
        request.llm_provider,
        request.auto_execute
    )
    
    return JobResponse(
        job_id=job_id,
        status="started",
        message=f"Analysis job {job_id} started"
    )


@app.post("/api/analyze/upload", response_model=JobResponse)
async def start_analysis_with_upload(
    background_tasks: BackgroundTasks,
    paper: UploadFile = File(...),
    repo_url: str = Form(...),
    llm_provider: str = Form(default="gemini"),
    auto_execute: bool = Form(default=True)
):
    """Start analysis with an uploaded PDF file."""
    job_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file
    file_path = UPLOAD_DIR / f"{job_id}_{paper.filename}"
    content = await paper.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    job = JobStatus(job_id)
    job.paper_source = str(file_path)
    job.repo_url = repo_url
    jobs[job_id] = job
    
    logger.info(
        f"Starting analysis job {job_id} with uploaded file",
        category=LogCategory.SYSTEM,
        context={"file": paper.filename, "repo": repo_url}
    )
    
    background_tasks.add_task(
        run_analysis,
        job_id,
        str(file_path),
        repo_url,
        llm_provider,
        auto_execute
    )
    
    return JobResponse(
        job_id=job_id,
        status="started",
        message=f"Analysis job {job_id} started with uploaded file"
    )


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a specific job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs[job_id]
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
        events=job.events[-50:],  # Last 50 events
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at
    )


@app.get("/api/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Get the full result of a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs[job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job {job_id} is not completed yet")
    
    return job.result


@app.get("/api/jobs/{job_id}/report")
async def get_job_report(job_id: str):
    """Get the generated report for a job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs[job_id]
    
    # Check for report file
    report_path = REPORTS_DIR / f"{job_id}_report.html"
    if report_path.exists():
        return FileResponse(
            str(report_path),
            media_type="text/html",
            filename=f"analysis_report_{job_id}.html"
        )
    
    raise HTTPException(status_code=404, detail="Report not yet generated")


@app.get("/api/jobs/{job_id}/knowledge-graph")
async def get_knowledge_graph(job_id: str):
    """Get the knowledge graph data for visualization."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs[job_id]
    if job.result and "knowledge_graph" in job.result:
        return job.result["knowledge_graph"]
    
    raise HTTPException(status_code=404, detail="Knowledge graph not available")


@app.get("/api/jobs")
async def list_jobs(limit: int = 20):
    """List recent jobs."""
    sorted_jobs = sorted(
        jobs.values(),
        key=lambda j: j.created_at,
        reverse=True
    )[:limit]
    
    return [
        {
            "job_id": j.job_id,
            "status": j.status,
            "stage": j.stage,
            "progress": j.progress,
            "paper_source": j.paper_source,
            "repo_url": j.repo_url,
            "created_at": j.created_at,
            "updated_at": j.updated_at
        }
        for j in sorted_jobs
    ]


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs[job_id]
    if job.status == "running":
        job.status = "cancelled"
        job.updated_at = datetime.now().isoformat()
        await manager.broadcast(job_id, {
            "type": "cancelled",
            "job_id": job_id,
            "message": "Job cancelled by user"
        })
    
    return {"message": f"Job {job_id} cancelled"}


# ============================================================================
# Metrics Endpoints
# ============================================================================

@app.get("/api/metrics")
async def get_metrics():
    """Get aggregated metrics summary for all pipeline operations."""
    metrics = get_metrics_collector()
    return metrics.get_summary()


@app.get("/api/metrics/agents")
async def get_agent_metrics():
    """Get metrics for all agent operations."""
    metrics = get_metrics_collector()
    all_metrics = metrics.get_all_metrics()

    # Filter to agent-related metrics
    agent_metrics = {
        key: metric.to_dict()
        for key, metric in all_metrics.items()
        if "agent" in key.lower()
    }

    return {
        "agent_metrics": agent_metrics,
        "total_operations": sum(m["count"] for m in agent_metrics.values()) if agent_metrics else 0
    }


@app.get("/api/metrics/pipeline")
async def get_pipeline_metrics():
    """Get metrics for pipeline stages."""
    metrics = get_metrics_collector()
    all_metrics = metrics.get_all_metrics()

    # Filter to pipeline-related metrics
    pipeline_metrics = {
        key: metric.to_dict()
        for key, metric in all_metrics.items()
        if "pipeline" in key.lower()
    }

    return {
        "pipeline_metrics": pipeline_metrics,
        "overall_accuracy": metrics.calculate_accuracy_score()
    }


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates."""
    await manager.connect(websocket, job_id)
    
    # Send current status immediately
    if job_id in jobs:
        job = jobs[job_id]
        await websocket.send_json({
            "type": "status",
            "job_id": job_id,
            "status": job.status,
            "stage": job.stage,
            "progress": job.progress
        })
    
    try:
        while True:
            # Keep connection alive, handle client messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)


# ============================================================================
# Background Task: Run Analysis
# ============================================================================

async def run_analysis(
    job_id: str,
    paper_source: str,
    repo_url: str,
    llm_provider: str,
    auto_execute: bool
):
    """Run the full analysis pipeline."""
    job = jobs[job_id]
    job.status = "running"
    
    try:
        # Create orchestrator with event callback
        async def event_callback(event: PipelineEvent):
            job.stage = event.stage.value
            job.progress = event.progress
            job.updated_at = datetime.now().isoformat()
            
            event_data = {
                "type": "event",
                "job_id": job_id,
                "stage": event.stage.value,
                "progress": event.progress,
                "message": event.message,
                "timestamp": event.timestamp.isoformat(),
                "data": event.data
            }
            job.events.append(event_data)
            await manager.broadcast(job_id, event_data)
        
        # Create and run orchestrator
        orchestrator = PipelineOrchestrator(
            llm_provider=llm_provider,
            event_callback=event_callback
        )
        
        # Run pipeline
        result = await orchestrator.run(
            paper_source=paper_source,
            repo_url=repo_url,
            auto_execute=auto_execute
        )
        
        # Store result
        job.result = {
            "paper": result.paper_data,
            "repository": result.repo_data,
            "mappings": result.mappings,
            "code_results": result.code_results,
            "knowledge_graph": result.knowledge_graph.to_d3_format() if result.knowledge_graph else None,
            "report_paths": result.report_paths,
            "errors": [str(e) for e in result.errors]
        }
        
        # Generate report
        from reports.template_engine import ReportGenerator
        report_gen = ReportGenerator()
        report_path = await report_gen.generate(
            job_id=job_id,
            paper_data=result.paper_data,
            repo_data=result.repo_data,
            mappings=result.mappings,
            code_results=result.code_results,
            knowledge_graph=result.knowledge_graph
        )
        
        job.result["report_path"] = str(report_path)
        job.status = "completed"
        job.progress = 100
        
        await manager.broadcast(job_id, {
            "type": "completed",
            "job_id": job_id,
            "message": "Analysis completed successfully",
            "report_url": f"/api/jobs/{job_id}/report"
        })
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.updated_at = datetime.now().isoformat()
        
        logger.error(
            f"Job {job_id} failed: {e}",
            category=LogCategory.SYSTEM
        )
        
        await manager.broadcast(job_id, {
            "type": "error",
            "job_id": job_id,
            "message": str(e)
        })


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Scientific Paper Analysis System starting...", category=LogCategory.SYSTEM)
    
    # Verify directories
    for dir_path in [UPLOAD_DIR, OUTPUT_DIR, REPORTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("System ready", category=LogCategory.SYSTEM)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("System shutting down...", category=LogCategory.SYSTEM)


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
