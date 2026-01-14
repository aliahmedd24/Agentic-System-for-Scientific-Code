import apiClient from './client'
import type {
  AnalysisRequest,
  JobResponse,
  Job,
  JobStatusResponse,
  JobResult,
  KnowledgeGraphData,
  GraphNode,
  MetricsSummary,
  AgentMetrics,
  PipelineMetrics,
  HealthStatus,
} from './types'

// ============================================================================
// Analysis Endpoints
// ============================================================================

/**
 * Start a new analysis with arXiv ID or URL
 */
export async function startAnalysis(request: AnalysisRequest): Promise<JobResponse> {
  const response = await apiClient.post<JobResponse>('/analyze', request)
  return response.data
}

/**
 * Start a new analysis with PDF upload
 */
export async function startAnalysisWithUpload(
  file: File,
  repoUrl: string,
  llmProvider: string = 'gemini',
  autoExecute: boolean = true
): Promise<JobResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('repo_url', repoUrl)
  formData.append('llm_provider', llmProvider)
  formData.append('auto_execute', String(autoExecute))

  const response = await apiClient.post<JobResponse>('/analyze/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 60000, // Longer timeout for file upload
  })
  return response.data
}

// ============================================================================
// Job Management Endpoints
// ============================================================================

/**
 * Get list of recent jobs
 */
export async function getJobs(limit: number = 20): Promise<Job[]> {
  const response = await apiClient.get<Job[]>('/jobs', {
    params: { limit },
  })
  return response.data
}

/**
 * Get job status with events
 */
export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const response = await apiClient.get<JobStatusResponse>(`/jobs/${jobId}`)
  return response.data
}

/**
 * Get full job result (after completion)
 */
export async function getJobResult(jobId: string): Promise<JobResult> {
  const response = await apiClient.get<JobResult>(`/jobs/${jobId}/result`)
  return response.data
}

/**
 * Cancel a running job
 */
export async function cancelJob(jobId: string): Promise<{ message: string }> {
  const response = await apiClient.delete<{ message: string }>(`/jobs/${jobId}`)
  return response.data
}

// ============================================================================
// Report Endpoints
// ============================================================================

/**
 * Get report download URL
 */
export function getReportUrl(jobId: string, format: 'html' | 'json' | 'markdown' = 'html'): string {
  const suffix = format === 'html' ? '' : `/${format}`
  return `/api/jobs/${jobId}/report${suffix}`
}

/**
 * Download report as blob
 */
export async function downloadReport(jobId: string, format: 'html' | 'json' | 'markdown' = 'html'): Promise<Blob> {
  const suffix = format === 'html' ? '' : `/${format}`
  const response = await apiClient.get(`/jobs/${jobId}/report${suffix}`, {
    responseType: 'blob',
  })
  return response.data
}

// ============================================================================
// Knowledge Graph Endpoints
// ============================================================================

/**
 * Get full knowledge graph
 */
export async function getKnowledgeGraph(jobId: string): Promise<KnowledgeGraphData> {
  const response = await apiClient.get<KnowledgeGraphData>(`/jobs/${jobId}/knowledge-graph`)
  return response.data
}

/**
 * Search knowledge graph nodes
 */
export async function searchKnowledgeGraph(
  jobId: string,
  query: string,
  nodeType?: string,
  limit: number = 20
): Promise<{ nodes: GraphNode[]; total: number }> {
  const response = await apiClient.get(`/jobs/${jobId}/knowledge-graph/search`, {
    params: { query, node_type: nodeType, limit },
  })
  return response.data
}

/**
 * Filter knowledge graph by node types
 */
export async function filterKnowledgeGraph(
  jobId: string,
  nodeTypes?: string[],
  minConnections?: number,
  includeIsolated: boolean = false
): Promise<KnowledgeGraphData> {
  const response = await apiClient.get<KnowledgeGraphData>(`/jobs/${jobId}/knowledge-graph/filter`, {
    params: {
      node_types: nodeTypes?.join(','),
      min_connections: minConnections,
      include_isolated: includeIsolated,
    },
  })
  return response.data
}

/**
 * Get node with neighbors
 */
export async function getGraphNode(
  jobId: string,
  nodeId: string,
  depth: number = 1
): Promise<{ node: GraphNode; neighbors: GraphNode[] }> {
  const response = await apiClient.get(`/jobs/${jobId}/knowledge-graph/node/${nodeId}`, {
    params: { depth },
  })
  return response.data
}

// ============================================================================
// Metrics Endpoints
// ============================================================================

/**
 * Get aggregated metrics summary
 */
export async function getMetrics(): Promise<MetricsSummary> {
  const response = await apiClient.get<MetricsSummary>('/metrics')
  return response.data
}

/**
 * Get agent-specific metrics
 */
export async function getAgentMetrics(): Promise<AgentMetrics> {
  const response = await apiClient.get<AgentMetrics>('/metrics/agents')
  return response.data
}

/**
 * Get pipeline stage metrics
 */
export async function getPipelineMetrics(): Promise<PipelineMetrics> {
  const response = await apiClient.get<PipelineMetrics>('/metrics/pipeline')
  return response.data
}

// ============================================================================
// Health Endpoint
// ============================================================================

/**
 * Get health status
 */
export async function getHealth(): Promise<HealthStatus> {
  const response = await apiClient.get<HealthStatus>('/health')
  return response.data
}
