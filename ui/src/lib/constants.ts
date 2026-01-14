// Pipeline stages with progress percentages
export const PIPELINE_STAGES = [
  { id: 'initialized', label: 'Initialized', progress: 0 },
  { id: 'parsing_paper', label: 'Paper Parsing', progress: 15 },
  { id: 'analyzing_repo', label: 'Repo Analysis', progress: 35 },
  { id: 'mapping_concepts', label: 'Mapping', progress: 50 },
  { id: 'generating_code', label: 'Code Gen', progress: 65 },
  { id: 'setting_up_env', label: 'Setup', progress: 75 },
  { id: 'executing_code', label: 'Execution', progress: 85 },
  { id: 'generating_report', label: 'Report', progress: 95 },
  { id: 'completed', label: 'Completed', progress: 100 },
] as const

export type PipelineStageId = typeof PIPELINE_STAGES[number]['id']

// Job status types
export const JOB_STATUSES = {
  pending: { label: 'Pending', color: 'text-text-muted', bgColor: 'bg-text-muted/20' },
  running: { label: 'Running', color: 'text-accent-primary', bgColor: 'bg-accent-primary/20' },
  completed: { label: 'Completed', color: 'text-status-success', bgColor: 'bg-status-success/20' },
  failed: { label: 'Failed', color: 'text-status-error', bgColor: 'bg-status-error/20' },
  cancelled: { label: 'Cancelled', color: 'text-status-warning', bgColor: 'bg-status-warning/20' },
} as const

export type JobStatus = keyof typeof JOB_STATUSES

// LLM providers
export const LLM_PROVIDERS = [
  { id: 'gemini', label: 'Google Gemini', description: 'Fast and efficient' },
  { id: 'anthropic', label: 'Anthropic Claude', description: 'High quality reasoning' },
  { id: 'openai', label: 'OpenAI GPT-4', description: 'Versatile and capable' },
] as const

export type LLMProvider = typeof LLM_PROVIDERS[number]['id']

// Knowledge graph node types
export const NODE_TYPES = {
  // Paper nodes
  PAPER: { label: 'Paper', color: '#6366f1' },
  SECTION: { label: 'Section', color: '#6366f1' },
  CONCEPT: { label: 'Concept', color: '#22c55e' },
  ALGORITHM: { label: 'Algorithm', color: '#f59e0b' },
  EQUATION: { label: 'Equation', color: '#f59e0b' },
  FIGURE: { label: 'Figure', color: '#06b6d4' },
  CITATION: { label: 'Citation', color: '#64748b' },
  // Code nodes
  REPOSITORY: { label: 'Repository', color: '#06b6d4' },
  FILE: { label: 'File', color: '#64748b' },
  CLASS: { label: 'Class', color: '#ec4899' },
  FUNCTION: { label: 'Function', color: '#ec4899' },
  DEPENDENCY: { label: 'Dependency', color: '#64748b' },
  MODULE: { label: 'Module', color: '#8b5cf6' },
  // Execution nodes
  TEST: { label: 'Test', color: '#22c55e' },
  RESULT: { label: 'Result', color: '#22c55e' },
  VISUALIZATION: { label: 'Visualization', color: '#06b6d4' },
  ERROR: { label: 'Error', color: '#ef4444' },
  // Meta nodes
  AGENT: { label: 'Agent', color: '#8b5cf6' },
  INSIGHT: { label: 'Insight', color: '#f59e0b' },
  MAPPING: { label: 'Mapping', color: '#8b5cf6' },
} as const

export type NodeType = keyof typeof NODE_TYPES

// API base URL
export const API_BASE_URL = '/api'
export const WS_BASE_URL = `ws://${window.location.host}/ws`
