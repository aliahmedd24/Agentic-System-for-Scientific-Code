import type { GraphNode, GraphLink } from '@/api/types'
import type { SimulationNodeDatum, SimulationLinkDatum } from 'd3'

// ============================================================================
// D3.js Simulation Types
// ============================================================================

export interface D3Node extends SimulationNodeDatum {
  id: string
  name: string
  type: string
  description?: string
  metadata?: Record<string, unknown>
  // D3 adds these during simulation
  x?: number
  y?: number
  fx?: number | null
  fy?: number | null
  vx?: number
  vy?: number
  index?: number
}

export interface D3Link extends SimulationLinkDatum<D3Node> {
  source: D3Node | string
  target: D3Node | string
  type: string
  weight?: number
}

// ============================================================================
// Canvas State Types
// ============================================================================

export interface Transform {
  x: number
  y: number
  k: number // scale
}

export interface CanvasState {
  width: number
  height: number
  transform: Transform
  isSimulationRunning: boolean
}

export interface CanvasDimensions {
  width: number
  height: number
}

// ============================================================================
// Selection & Interaction Types
// ============================================================================

export interface NodeSelection {
  node: D3Node
  x: number
  y: number
}

export interface HoveredNode {
  node: D3Node
  x: number
  y: number
}

export interface SearchResult {
  nodes: GraphNode[]
  total: number
  highlightedIds: Set<string>
}

// ============================================================================
// Filter Types
// ============================================================================

export interface NodeTypeFilterState {
  activeTypes: Set<string>
  minConnections: number
  includeIsolated: boolean
}

// ============================================================================
// Graph Statistics
// ============================================================================

export interface GraphStats {
  totalNodes: number
  totalLinks: number
  nodesByType: Record<string, number>
  avgConnections: number
  isolatedNodes: number
}

// ============================================================================
// Component Props Types
// ============================================================================

export interface KnowledgeGraphCanvasProps {
  nodes: GraphNode[]
  links: GraphLink[]
  selectedNode: D3Node | null
  highlightedIds: Set<string>
  activeFilters: Set<string>
  onNodeClick: (node: D3Node) => void
  onNodeHover: (node: D3Node | null) => void
  onBackgroundClick: () => void
  className?: string
}

export interface GraphControlsProps {
  onZoomIn: () => void
  onZoomOut: () => void
  onResetView: () => void
  onToggleFullscreen: () => void
  onFitToView: () => void
  isFullscreen: boolean
  zoomLevel: number
  className?: string
}

export interface NodeTypeFilterProps {
  activeTypes: Set<string>
  onToggleType: (type: string) => void
  onSelectAll: () => void
  onClearAll: () => void
  nodeCounts?: Record<string, number>
  className?: string
}

export interface GraphSearchProps {
  jobId: string
  onSearchResults: (results: SearchResult) => void
  onClearSearch: () => void
  className?: string
}

export interface NodeDetailsProps {
  node: D3Node | null
  neighbors?: GraphNode[]
  onClose: () => void
  onNavigateToNode: (nodeId: string) => void
  isLoading?: boolean
  className?: string
}

export interface GraphLegendProps {
  activeTypes: Set<string>
  compact?: boolean
  className?: string
}

// ============================================================================
// Hook Return Types
// ============================================================================

export interface UseKnowledgeGraphReturn {
  graphData: { nodes: GraphNode[]; links: GraphLink[] } | null
  isLoading: boolean
  error: Error | null
  refetch: () => Promise<void>
  stats: GraphStats | null
}

export interface UseGraphSearchReturn {
  search: (query: string, nodeType?: string) => Promise<void>
  results: SearchResult
  isSearching: boolean
  clearSearch: () => void
}

export interface UseGraphFilterReturn {
  filteredData: { nodes: GraphNode[]; links: GraphLink[] }
  filterState: NodeTypeFilterState
  setActiveTypes: (types: Set<string>) => void
  toggleType: (type: string) => void
  setMinConnections: (min: number) => void
  setIncludeIsolated: (include: boolean) => void
  resetFilters: () => void
}

// ============================================================================
// Performance Optimization Types
// ============================================================================

export interface RenderConfig {
  nodeRadius: number
  linkWidth: number
  labelThreshold: number // Only show labels above this zoom level
  maxVisibleNodes: number // For viewport culling
  useWebGL: boolean
}

export interface ViewportBounds {
  minX: number
  maxX: number
  minY: number
  maxY: number
}

// ============================================================================
// Event Types
// ============================================================================

export type GraphEventType =
  | 'node:click'
  | 'node:hover'
  | 'node:drag'
  | 'zoom'
  | 'pan'
  | 'simulation:tick'
  | 'simulation:end'

export interface GraphEvent {
  type: GraphEventType
  data?: unknown
  timestamp: number
}
