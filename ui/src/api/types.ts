// ============================================================================
// API Response Types - Generated from backend Pydantic models
// ============================================================================

// Job Status Types
export type JobStatusType = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

// LLM Provider Types
export type LLMProvider = 'gemini' | 'anthropic' | 'openai'

// ============================================================================
// Request Types
// ============================================================================

export interface AnalysisRequest {
  paper_source: string
  repo_url: string
  llm_provider?: LLMProvider
  auto_execute?: boolean
}

// ============================================================================
// Job Types
// ============================================================================

export interface Job {
  job_id: string
  status: JobStatusType
  stage: string
  progress: number
  paper_source: string | null
  repo_url: string | null
  created_at: string
  updated_at: string
  error: string | null
}

export interface JobEvent {
  timestamp: string
  stage: string
  progress: number
  message: string
  data?: Record<string, unknown>
}

export interface JobStatusResponse extends Job {
  events: JobEvent[]
  llm_provider?: LLMProvider
  auto_execute?: boolean
  error_message?: string | null
}

export interface JobResponse {
  job_id: string
  status: string
  message: string
}

// ============================================================================
// Paper Parser Types
// ============================================================================

export interface Concept {
  name: string
  description: string
  importance: 'low' | 'medium' | 'high' | 'critical'
  related_sections: string[]
  likely_names: string[]
}

export interface Algorithm {
  name: string
  description: string
  pseudocode: string
  complexity: string
  inputs: string[]
  outputs: string[]
}

export interface Methodology {
  approach: string
  datasets: string[]
  evaluation_metrics: string[]
  baselines: string[]
}

export interface Reproducibility {
  code_available: boolean
  data_available: boolean
  hardware_requirements: string
  estimated_time: string
}

export interface ExpectedImplementation {
  component_name: string
  description: string
  priority: 'low' | 'medium' | 'high'
  dependencies: string[]
}

export interface SourceMetadata {
  source_type: string
  arxiv_id: string | null
  url: string | null
  file_path: string | null
  extraction_date: string | null
}

export interface PaperData {
  title: string
  authors: string[]
  abstract: string
  key_concepts: Concept[]
  algorithms: Algorithm[]
  methodology: Methodology
  reproducibility: Reproducibility
  expected_implementations: ExpectedImplementation[]
  source_metadata: SourceMetadata
}

// ============================================================================
// Repository Analyzer Types
// ============================================================================

export interface FileStats {
  total_files: number
  code_files: number
  classes: number
  functions: number
}

export interface DependencyInfo {
  python: string[]
  julia: string[]
  r: string[]
  javascript: string[]
  system: string[]
}

export interface OverviewInfo {
  purpose: string
  architecture: string
  key_features: string[]
}

export interface KeyComponent {
  name: string
  path: string
  description: string
  importance: string
}

export interface EntryPoint {
  name: string
  path: string
  description: string
  arguments: string[]
}

export interface SetupComplexity {
  level: 'easy' | 'medium' | 'hard' | 'expert'
  steps: string[]
  estimated_time: string
}

export interface ComputeRequirements {
  cpu_cores: number
  memory_gb: number
  gpu_required: boolean
  gpu_memory_gb: number
}

export interface RepoData {
  name: string
  url: string
  overview: OverviewInfo
  key_components: KeyComponent[]
  entry_points: EntryPoint[]
  dependencies: DependencyInfo
  setup_complexity: SetupComplexity
  compute_requirements: ComputeRequirements
  stats: FileStats
}

// ============================================================================
// Semantic Mapping Types
// ============================================================================

export interface MatchSignals {
  lexical: number
  semantic: number
  documentary: number
}

export interface MappingResult {
  concept_name: string
  concept_description: string
  code_element: string
  code_file: string
  confidence: number
  match_signals: MatchSignals
  evidence: string[]
  reasoning: string
}

export interface UnmappedItem {
  name: string
  description: string
  reason: string
}

// ============================================================================
// Code Execution Types
// ============================================================================

export interface TestResult {
  concept: string
  code_element: string
  success: boolean
  stdout: string
  stderr: string
  execution_time: number
  return_code: number
  output_files: string[]
  error: string
  isolation_level: string
}

export interface GeneratedScript {
  concept: string
  code_element: string
  code_file: string
  confidence: number
  code: string
  file_name: string
  language: string
  syntax_valid: boolean
  import_valid: boolean
}

export interface ExecutionSummary {
  total_tests: number
  passed: number
  failed: number
  skipped: number
  total_time: number
}

export interface ResourceEstimate {
  compute_level: string
  memory_gb: number
  gpu_required: boolean
  gpu_memory_gb: number
  estimated_time_minutes: number
  disk_space_gb: number
  dependency_count: number
  complexity_score: number
  warnings: string[]
  recommendations: string[]
}

export interface CodeResults {
  scripts: GeneratedScript[]
  results: TestResult[]
  language: string
  summary: ExecutionSummary
  resource_estimate: ResourceEstimate | null
}

// ============================================================================
// Full Job Result
// ============================================================================

export interface JobResult {
  job_id: string
  paper_data: PaperData
  repo_data: RepoData
  mappings: MappingResult[]
  unmapped_concepts: UnmappedItem[]
  unmapped_code: UnmappedItem[]
  code_results: CodeResults
}

// ============================================================================
// Knowledge Graph Types
// ============================================================================

export interface GraphNode {
  id: string
  name: string
  type: string
  description?: string
  metadata?: Record<string, unknown>
}

export interface GraphLink {
  source: string
  target: string
  type: string
  weight?: number
}

export interface KnowledgeGraphData {
  nodes: GraphNode[]
  links: GraphLink[]
}

export interface GraphSearchResult {
  nodes: GraphNode[]
  total: number
}

// ============================================================================
// Metrics Types
// ============================================================================

export interface MetricsSummary {
  total_jobs: number
  completed_jobs: number
  failed_jobs: number
  average_accuracy: number
  average_duration_seconds: number
}

export interface AgentMetric {
  agent_name: string
  operations: number
  total_duration_ms: number
  avg_duration_ms: number
  errors: number
  last_operation: string | null
}

export interface AgentMetrics {
  agents: AgentMetric[]
}

export interface PipelineStageMetric {
  stage: string
  count: number
  avg_duration_ms: number
  success_rate: number
}

export interface PipelineMetrics {
  stages: PipelineStageMetric[]
  accuracy_score: number
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy'
  timestamp: string
  active_jobs: number
}

// ============================================================================
// WebSocket Message Types
// ============================================================================

export type WebSocketMessageType = 'event' | 'status' | 'completed' | 'error' | 'cancelled'

export interface WebSocketMessage {
  type: WebSocketMessageType
  timestamp: string
  stage?: string
  progress?: number
  message?: string
  data?: Record<string, unknown>
  error?: string
}
