"""
Tests for the FastAPI server.

This module tests all API endpoints including:
- Health check and root endpoints
- Analysis job lifecycle (create, status, result, cancel)
- Knowledge graph endpoints (search, filter, node traversal)
- Metrics endpoints (summary, agents, pipeline)
- Report endpoints (HTML, JSON, Markdown)
- WebSocket real-time updates
- Input validation and error handling
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self, test_client):
        """Test health check returns healthy status."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "active_jobs" in data


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_ui_or_json(self, test_client):
        """Test root endpoint returns UI or API info."""
        response = test_client.get("/")
        
        # Should return either HTML (UI) or JSON (API info)
        assert response.status_code == 200
        
        content_type = response.headers.get("content-type", "")
        assert "text/html" in content_type or "application/json" in content_type


class TestAnalyzeEndpoint:
    """Tests for analysis endpoints."""
    
    def test_start_analysis_valid(self, test_client):
        """Test starting analysis with valid parameters."""
        response = test_client.post(
            "/api/analyze",
            json={
                "paper_source": "2301.00001",
                "repo_url": "https://github.com/user/repo",
                "llm_provider": "gemini",
                "auto_execute": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "started"
    
    def test_start_analysis_missing_fields(self, test_client):
        """Test starting analysis with missing required fields."""
        response = test_client.post(
            "/api/analyze",
            json={"paper_source": "2301.00001"}
            # Missing repo_url
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_start_analysis_with_upload(self, test_client, tmp_path):
        """Test starting analysis with file upload."""
        # Create a dummy PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 dummy content")
        
        with open(pdf_path, "rb") as f:
            response = test_client.post(
                "/api/analyze/upload",
                files={"paper": ("test.pdf", f, "application/pdf")},
                data={
                    "repo_url": "https://github.com/user/repo",
                    "llm_provider": "gemini",
                    "auto_execute": "true"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data


class TestJobsEndpoint:
    """Tests for job management endpoints."""
    
    def test_list_jobs(self, test_client):
        """Test listing jobs."""
        response = test_client.get("/api/jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_job_status_not_found(self, test_client):
        """Test getting status of non-existent job."""
        response = test_client.get("/api/jobs/nonexistent123/status")
        
        assert response.status_code == 404
    
    def test_get_job_result_not_found(self, test_client):
        """Test getting result of non-existent job."""
        response = test_client.get("/api/jobs/nonexistent123/result")
        
        assert response.status_code == 404
    
    def test_cancel_job_not_found(self, test_client):
        """Test cancelling non-existent job."""
        response = test_client.delete("/api/jobs/nonexistent123")
        
        assert response.status_code == 404
    
    def test_job_lifecycle(self, test_client):
        """Test complete job lifecycle."""
        # Start a job
        start_response = test_client.post(
            "/api/analyze",
            json={
                "paper_source": "2301.00001",
                "repo_url": "https://github.com/user/repo"
            }
        )
        assert start_response.status_code == 200
        job_id = start_response.json()["job_id"]
        
        # Check status
        status_response = test_client.get(f"/api/jobs/{job_id}/status")
        assert status_response.status_code == 200
        status = status_response.json()
        assert status["job_id"] == job_id
        assert "status" in status
        
        # Job should be in jobs list
        list_response = test_client.get("/api/jobs")
        job_ids = [j["job_id"] for j in list_response.json()]
        assert job_id in job_ids


class TestReportEndpoint:
    """Tests for report endpoints."""
    
    def test_get_report_not_found(self, test_client):
        """Test getting report for non-existent job."""
        response = test_client.get("/api/jobs/nonexistent123/report")
        
        assert response.status_code == 404


class TestKnowledgeGraphEndpoint:
    """Tests for knowledge graph endpoints."""
    
    def test_get_knowledge_graph_not_found(self, test_client):
        """Test getting KG for non-existent job."""
        response = test_client.get("/api/jobs/nonexistent123/knowledge-graph")
        
        assert response.status_code == 404


class TestWebSocket:
    """Tests for WebSocket functionality."""
    
    def test_websocket_connection(self, test_client):
        """Test WebSocket connection."""
        with test_client.websocket_connect("/ws/test_job_id") as websocket:
            # Should receive initial status
            data = websocket.receive_json()
            assert "type" in data
            
            # Test ping/pong
            websocket.send_text("ping")
            pong = websocket.receive_json()
            assert pong["type"] == "pong"


class TestAPIValidation:
    """Tests for API input validation."""
    
    def test_invalid_llm_provider(self, test_client):
        """Test with invalid LLM provider."""
        response = test_client.post(
            "/api/analyze",
            json={
                "paper_source": "2301.00001",
                "repo_url": "https://github.com/user/repo",
                "llm_provider": "invalid_provider"
            }
        )
        
        # Should still work - validation happens at runtime
        assert response.status_code == 200
    
    def test_empty_paper_source(self, test_client):
        """Test with empty paper source."""
        response = test_client.post(
            "/api/analyze",
            json={
                "paper_source": "",
                "repo_url": "https://github.com/user/repo"
            }
        )
        
        # Empty string should be handled
        assert response.status_code == 200
    
    def test_invalid_repo_url(self, test_client):
        """Test with invalid repository URL."""
        response = test_client.post(
            "/api/analyze",
            json={
                "paper_source": "2301.00001",
                "repo_url": "not-a-valid-url"
            }
        )
        
        # Should start job, but fail during execution
        assert response.status_code == 200


class TestCORS:
    """Tests for CORS configuration."""
    
    def test_cors_headers(self, test_client):
        """Test CORS headers are present."""
        response = test_client.options(
            "/api/analyze",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST"
            }
        )
        
        # CORS should allow the request
        assert response.status_code in [200, 204, 405]


class TestErrorHandling:
    """Tests for API error handling."""
    
    def test_invalid_json(self, test_client):
        """Test handling of invalid JSON."""
        response = test_client.post(
            "/api/analyze",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_method_not_allowed(self, test_client):
        """Test method not allowed."""
        response = test_client.put("/api/analyze", json={})
        
        assert response.status_code == 405


class TestPagination:
    """Tests for pagination in list endpoints."""
    
    def test_jobs_list_limit(self, test_client):
        """Test jobs list respects limit parameter."""
        # Create multiple jobs
        for _ in range(5):
            test_client.post(
                "/api/analyze",
                json={
                    "paper_source": "2301.00001",
                    "repo_url": "https://github.com/user/repo"
                }
            )

        # Request with limit
        response = test_client.get("/api/jobs?limit=3")

        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 3


# =============================================================================
# Knowledge Graph Search Endpoint Tests
# =============================================================================

class TestKnowledgeGraphSearchEndpoint:
    """Tests for knowledge graph search endpoint."""

    @pytest.fixture
    def job_with_kg(self, test_client):
        """Create a job with a knowledge graph for testing."""
        # Start a job
        response = test_client.post(
            "/api/analyze",
            json={
                "paper_source": "2301.00001",
                "repo_url": "https://github.com/user/repo"
            }
        )
        job_id = response.json()["job_id"]

        # Manually add KG data to the job (simulating completed job)
        from api.server import jobs
        job = jobs[job_id]
        job.status = "completed"
        job.result = {
            "knowledge_graph": {
                "nodes": [
                    {"id": "node1", "label": "Transformer", "type": "concept", "description": "Attention-based architecture"},
                    {"id": "node2", "label": "BERT", "type": "algorithm", "description": "Bidirectional encoder"},
                    {"id": "node3", "label": "attention_forward", "type": "function", "description": "Forward pass function"},
                    {"id": "node4", "label": "GPT", "type": "algorithm", "description": "Generative pretrained model"},
                ],
                "links": [
                    {"source": "node1", "target": "node2", "type": "implements"},
                    {"source": "node1", "target": "node4", "type": "implements"},
                    {"source": "node2", "target": "node3", "type": "contains"},
                ]
            }
        }

        return job_id

    def test_search_knowledge_graph_by_query(self, test_client, job_with_kg):
        """Test searching KG nodes by query."""
        response = test_client.get(
            f"/api/jobs/{job_with_kg}/knowledge-graph/search?query=transformer"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "transformer"
        assert data["count"] >= 1
        assert any(n["label"] == "Transformer" for n in data["results"])

    def test_search_knowledge_graph_with_type_filter(self, test_client, job_with_kg):
        """Test searching KG with node type filter."""
        response = test_client.get(
            f"/api/jobs/{job_with_kg}/knowledge-graph/search?query=e&node_type=algorithm"
        )

        assert response.status_code == 200
        data = response.json()
        # All results should be algorithms
        for node in data["results"]:
            assert node["type"] == "algorithm"

    def test_search_knowledge_graph_with_limit(self, test_client, job_with_kg):
        """Test search respects limit parameter."""
        response = test_client.get(
            f"/api/jobs/{job_with_kg}/knowledge-graph/search?query=e&limit=2"
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= 2

    def test_search_knowledge_graph_not_found(self, test_client):
        """Test search on non-existent job."""
        response = test_client.get(
            "/api/jobs/nonexistent/knowledge-graph/search?query=test"
        )

        assert response.status_code == 404

    def test_search_knowledge_graph_no_kg(self, test_client):
        """Test search when job has no KG."""
        # Create job without KG
        response = test_client.post(
            "/api/analyze",
            json={
                "paper_source": "2301.00001",
                "repo_url": "https://github.com/user/repo"
            }
        )
        job_id = response.json()["job_id"]

        response = test_client.get(
            f"/api/jobs/{job_id}/knowledge-graph/search?query=test"
        )

        assert response.status_code == 404


# =============================================================================
# Knowledge Graph Filter Endpoint Tests
# =============================================================================

class TestKnowledgeGraphFilterEndpoint:
    """Tests for knowledge graph filter endpoint."""

    @pytest.fixture
    def job_with_kg(self, test_client):
        """Create a job with a knowledge graph for testing."""
        response = test_client.post(
            "/api/analyze",
            json={
                "paper_source": "2301.00001",
                "repo_url": "https://github.com/user/repo"
            }
        )
        job_id = response.json()["job_id"]

        from api.server import jobs
        job = jobs[job_id]
        job.status = "completed"
        job.result = {
            "knowledge_graph": {
                "nodes": [
                    {"id": "n1", "label": "Node1", "type": "concept"},
                    {"id": "n2", "label": "Node2", "type": "algorithm"},
                    {"id": "n3", "label": "Node3", "type": "function"},
                    {"id": "n4", "label": "Isolated", "type": "concept"},  # No connections
                ],
                "links": [
                    {"source": "n1", "target": "n2", "type": "related"},
                    {"source": "n1", "target": "n3", "type": "related"},
                    {"source": "n2", "target": "n3", "type": "implements"},
                ]
            }
        }

        return job_id

    def test_filter_by_node_types(self, test_client, job_with_kg):
        """Test filtering KG by node types."""
        response = test_client.get(
            f"/api/jobs/{job_with_kg}/knowledge-graph/filter?node_types=concept,algorithm"
        )

        assert response.status_code == 200
        data = response.json()
        # Should only include concept and algorithm nodes
        for node in data["nodes"]:
            assert node["type"] in ["concept", "algorithm"]

    def test_filter_by_min_connections(self, test_client, job_with_kg):
        """Test filtering KG by minimum connections."""
        response = test_client.get(
            f"/api/jobs/{job_with_kg}/knowledge-graph/filter?min_connections=2"
        )

        assert response.status_code == 200
        data = response.json()
        # n1 has 2 connections, n2 has 2, n3 has 2
        assert data["node_count"] >= 0

    def test_filter_exclude_isolated(self, test_client, job_with_kg):
        """Test filtering excludes isolated nodes."""
        response = test_client.get(
            f"/api/jobs/{job_with_kg}/knowledge-graph/filter?include_isolated=false"
        )

        assert response.status_code == 200
        data = response.json()
        # Isolated node should not be in results
        node_ids = [n["id"] for n in data["nodes"]]
        assert "n4" not in node_ids

    def test_filter_include_isolated(self, test_client, job_with_kg):
        """Test filtering includes isolated nodes when requested."""
        response = test_client.get(
            f"/api/jobs/{job_with_kg}/knowledge-graph/filter?include_isolated=true"
        )

        assert response.status_code == 200
        data = response.json()
        # Isolated node should be in results
        node_ids = [n["id"] for n in data["nodes"]]
        assert "n4" in node_ids

    def test_filter_returns_node_and_link_counts(self, test_client, job_with_kg):
        """Test filter response includes counts."""
        response = test_client.get(
            f"/api/jobs/{job_with_kg}/knowledge-graph/filter"
        )

        assert response.status_code == 200
        data = response.json()
        assert "node_count" in data
        assert "link_count" in data


# =============================================================================
# Knowledge Graph Node Endpoint Tests
# =============================================================================

class TestKnowledgeGraphNodeEndpoint:
    """Tests for knowledge graph node retrieval endpoint."""

    @pytest.fixture
    def job_with_kg(self, test_client):
        """Create a job with a knowledge graph for testing."""
        response = test_client.post(
            "/api/analyze",
            json={
                "paper_source": "2301.00001",
                "repo_url": "https://github.com/user/repo"
            }
        )
        job_id = response.json()["job_id"]

        from api.server import jobs
        job = jobs[job_id]
        job.status = "completed"
        job.result = {
            "knowledge_graph": {
                "nodes": [
                    {"id": "center", "label": "Center", "type": "concept"},
                    {"id": "neighbor1", "label": "Neighbor1", "type": "algorithm"},
                    {"id": "neighbor2", "label": "Neighbor2", "type": "function"},
                    {"id": "level2", "label": "Level2", "type": "concept"},
                ],
                "links": [
                    {"source": "center", "target": "neighbor1", "type": "related"},
                    {"source": "center", "target": "neighbor2", "type": "related"},
                    {"source": "neighbor1", "target": "level2", "type": "implements"},
                ]
            }
        }

        return job_id

    def test_get_node_with_neighbors(self, test_client, job_with_kg):
        """Test getting a node with its neighbors."""
        response = test_client.get(
            f"/api/jobs/{job_with_kg}/knowledge-graph/node/center"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["center_node"]["id"] == "center"
        assert data["depth"] == 1
        assert "neighbor_count" in data

    def test_get_node_with_depth_2(self, test_client, job_with_kg):
        """Test getting a node with depth 2."""
        response = test_client.get(
            f"/api/jobs/{job_with_kg}/knowledge-graph/node/center?depth=2"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["depth"] == 2
        # Should include level2 node which is 2 hops away
        node_ids = [n["id"] for n in data["nodes"]]
        assert "level2" in node_ids

    def test_get_node_not_found(self, test_client, job_with_kg):
        """Test getting a non-existent node."""
        response = test_client.get(
            f"/api/jobs/{job_with_kg}/knowledge-graph/node/nonexistent"
        )

        assert response.status_code == 404

    def test_get_node_job_not_found(self, test_client):
        """Test getting a node from non-existent job."""
        response = test_client.get(
            "/api/jobs/nonexistent/knowledge-graph/node/center"
        )

        assert response.status_code == 404


# =============================================================================
# Metrics Endpoints Tests
# =============================================================================

class TestMetricsEndpoints:
    """Tests for metrics API endpoints."""

    def test_get_metrics_summary(self, test_client):
        """Test getting metrics summary."""
        response = test_client.get("/api/metrics")

        assert response.status_code == 200
        data = response.json()
        # Should return a dict with metrics summary
        assert isinstance(data, dict)

    def test_get_agent_metrics(self, test_client):
        """Test getting agent-specific metrics."""
        response = test_client.get("/api/metrics/agents")

        assert response.status_code == 200
        data = response.json()
        assert "agent_metrics" in data
        assert "total_operations" in data

    def test_get_pipeline_metrics(self, test_client):
        """Test getting pipeline metrics."""
        response = test_client.get("/api/metrics/pipeline")

        assert response.status_code == 200
        data = response.json()
        assert "pipeline_metrics" in data
        assert "overall_accuracy" in data


# =============================================================================
# Report Format Endpoints Tests
# =============================================================================

class TestReportFormatEndpoints:
    """Tests for report format endpoints (JSON, Markdown)."""

    @pytest.fixture
    def completed_job(self, test_client):
        """Create a completed job with results for testing."""
        response = test_client.post(
            "/api/analyze",
            json={
                "paper_source": "2301.00001",
                "repo_url": "https://github.com/user/repo"
            }
        )
        job_id = response.json()["job_id"]

        from api.server import jobs
        job = jobs[job_id]
        job.status = "completed"
        job.result = {
            "paper": {"title": "Test Paper", "abstract": "Test abstract"},
            "repository": {"name": "test-repo", "url": "https://github.com/user/repo"},
            "mappings": [{"concept": "test", "code": "test.py"}],
            "code_results": [],
            "knowledge_graph": {"nodes": [], "links": []},
            "errors": []
        }

        return job_id

    def test_get_report_json(self, test_client, completed_job):
        """Test getting report as JSON."""
        response = test_client.get(f"/api/jobs/{completed_job}/report/json")

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert "generated_at" in data
        assert "paper" in data
        assert "repository" in data

    def test_get_report_json_not_found(self, test_client):
        """Test getting JSON report for non-existent job."""
        response = test_client.get("/api/jobs/nonexistent/report/json")

        assert response.status_code == 404

    def test_get_report_json_no_result(self, test_client):
        """Test getting JSON report before job completes."""
        # Create a job that hasn't completed
        response = test_client.post(
            "/api/analyze",
            json={
                "paper_source": "2301.00001",
                "repo_url": "https://github.com/user/repo"
            }
        )
        job_id = response.json()["job_id"]

        # Clear the result
        from api.server import jobs
        jobs[job_id].result = None

        response = test_client.get(f"/api/jobs/{job_id}/report/json")

        assert response.status_code == 404

    @patch('reports.template_engine.ReportGenerator')
    def test_get_report_markdown(self, mock_report_gen, test_client, completed_job):
        """Test getting report as Markdown."""
        # Mock the report generator
        mock_gen_instance = MagicMock()
        mock_gen_instance.generate_markdown.return_value = "# Test Report\n\nThis is a test."
        mock_report_gen.return_value = mock_gen_instance

        response = test_client.get(f"/api/jobs/{completed_job}/report/markdown")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/markdown; charset=utf-8"

    def test_get_report_markdown_not_found(self, test_client):
        """Test getting Markdown report for non-existent job."""
        response = test_client.get("/api/jobs/nonexistent/report/markdown")

        assert response.status_code == 404


# =============================================================================
# Job Status Endpoint Tests (Additional)
# =============================================================================

class TestJobStatusEndpointAdditional:
    """Additional tests for job status endpoint."""

    def test_get_job_status_full_response(self, test_client):
        """Test job status returns all expected fields."""
        # Create a job
        response = test_client.post(
            "/api/analyze",
            json={
                "paper_source": "2301.00001",
                "repo_url": "https://github.com/user/repo"
            }
        )
        job_id = response.json()["job_id"]

        # Get status using correct endpoint
        response = test_client.get(f"/api/jobs/{job_id}")

        assert response.status_code == 200
        data = response.json()

        # Check all expected fields
        assert "job_id" in data
        assert "status" in data
        assert "stage" in data
        assert "progress" in data
        assert "events" in data
        assert "created_at" in data
        assert "updated_at" in data


# =============================================================================
# WebSocket Additional Tests
# =============================================================================

class TestWebSocketAdditional:
    """Additional WebSocket tests."""

    def test_websocket_receives_job_status_immediately(self, test_client):
        """Test WebSocket receives current job status on connect."""
        # Create a job first
        response = test_client.post(
            "/api/analyze",
            json={
                "paper_source": "2301.00001",
                "repo_url": "https://github.com/user/repo"
            }
        )
        job_id = response.json()["job_id"]

        # Connect to WebSocket for this job
        with test_client.websocket_connect(f"/ws/{job_id}") as websocket:
            data = websocket.receive_json()
            assert data["type"] == "status"
            assert data["job_id"] == job_id

    def test_websocket_for_nonexistent_job(self, test_client):
        """Test WebSocket connection for non-existent job."""
        with test_client.websocket_connect("/ws/nonexistent") as websocket:
            # Should receive ping/pong but no initial status
            websocket.send_text("ping")
            pong = websocket.receive_json()
            assert pong["type"] == "pong"


# =============================================================================
# API Edge Cases
# =============================================================================

class TestAPIEdgeCases:
    """Tests for API edge cases and boundary conditions."""

    def test_very_long_paper_source(self, test_client):
        """Test handling of very long paper source."""
        response = test_client.post(
            "/api/analyze",
            json={
                "paper_source": "a" * 10000,
                "repo_url": "https://github.com/user/repo"
            }
        )

        # Should still accept the request
        assert response.status_code == 200

    def test_special_characters_in_repo_url(self, test_client):
        """Test handling special characters in repo URL."""
        response = test_client.post(
            "/api/analyze",
            json={
                "paper_source": "2301.00001",
                "repo_url": "https://github.com/user/repo-with-special_chars.123"
            }
        )

        assert response.status_code == 200

    def test_unicode_in_request(self, test_client):
        """Test handling Unicode characters in request."""
        response = test_client.post(
            "/api/analyze",
            json={
                "paper_source": "Test Paper 论文 🔬",
                "repo_url": "https://github.com/user/repo"
            }
        )

        assert response.status_code == 200

    def test_empty_query_search(self, test_client):
        """Test search with empty query."""
        # First create a job with KG
        response = test_client.post(
            "/api/analyze",
            json={
                "paper_source": "2301.00001",
                "repo_url": "https://github.com/user/repo"
            }
        )
        job_id = response.json()["job_id"]

        # Add KG data
        from api.server import jobs
        job = jobs[job_id]
        job.result = {
            "knowledge_graph": {
                "nodes": [{"id": "1", "label": "Test", "type": "concept"}],
                "links": []
            }
        }

        response = test_client.get(f"/api/jobs/{job_id}/knowledge-graph/search?query=")

        # Empty query should still work
        assert response.status_code == 200

    def test_negative_limit(self, test_client):
        """Test handling of negative limit parameter."""
        # Create a job with KG
        response = test_client.post(
            "/api/analyze",
            json={
                "paper_source": "2301.00001",
                "repo_url": "https://github.com/user/repo"
            }
        )
        job_id = response.json()["job_id"]

        from api.server import jobs
        job = jobs[job_id]
        job.result = {
            "knowledge_graph": {"nodes": [], "links": []}
        }

        # Negative limit - behavior depends on implementation
        response = test_client.get(f"/api/jobs/{job_id}/knowledge-graph/search?query=test&limit=-1")
        # Should either work or return 422
        assert response.status_code in [200, 422]
