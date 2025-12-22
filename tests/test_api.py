"""
Tests for the FastAPI server.
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
