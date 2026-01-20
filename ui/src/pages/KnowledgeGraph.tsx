import { useCallback, useMemo, useRef, useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { PageHeader } from '@/components/layout/PageHeader'
import { GlassCard } from '@/components/ui/GlassCard'
import { Button } from '@/components/ui/Button'
import { LoadingSpinner } from '@/components/data-display/LoadingSpinner'
import { NODE_TYPES } from '@/lib/constants'
import { cn } from '@/lib/cn'
import * as api from '@/api/endpoints'

import {
  KnowledgeGraphCanvas,
  GraphControls,
  NodeTypeFilter,
  GraphSearch,
  NodeDetails,
  GraphLegend,
  useKnowledgeGraph,
  useGraphSearch,
  useGraphFilter,
  useFullscreen,
} from '@/components/graph'
import type { D3Node, Transform } from '@/components/graph'
import type { GraphNode } from '@/api/types'

export default function KnowledgeGraph() {
  const { jobId } = useParams<{ jobId: string }>()

  // Refs
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasContainerRef = useRef<HTMLDivElement>(null)

  // Graph data hooks
  const { graphData, isLoading, error, refetch, stats } = useKnowledgeGraph(jobId)
  const { search, results: searchResults, isSearching, clearSearch } = useGraphSearch(jobId)

  // Filter state
  const { filteredData, filterState, toggleType, setActiveTypes, resetFilters } = useGraphFilter(
    graphData?.nodes || [],
    graphData?.links || []
  )

  // Fullscreen hook
  const { isFullscreen, toggleFullscreen } = useFullscreen(containerRef)

  // Local state
  const [selectedNode, setSelectedNode] = useState<D3Node | null>(null)
  const [neighbors, setNeighbors] = useState<GraphNode[]>([])
  const [isLoadingNeighbors, setIsLoadingNeighbors] = useState(false)
  const [transform, setTransform] = useState<Transform>({ x: 0, y: 0, k: 1 })
  const [hoveredNode, setHoveredNode] = useState<D3Node | null>(null)

  // Calculate node counts for filter
  const nodeCounts = useMemo(() => {
    if (!graphData?.nodes) return {}
    const counts: Record<string, number> = {}
    graphData.nodes.forEach(node => {
      const type = node.type.toUpperCase()
      counts[type] = (counts[type] || 0) + 1
    })
    return counts
  }, [graphData?.nodes])

  // Fetch neighbors when a node is selected
  useEffect(() => {
    if (!selectedNode || !jobId) {
      setNeighbors([])
      return
    }

    const fetchNeighbors = async () => {
      setIsLoadingNeighbors(true)
      try {
        const result = await api.getGraphNode(jobId, selectedNode.id, 1)
        setNeighbors(result.neighbors || [])
      } catch (err) {
        console.error('Failed to fetch neighbors:', err)
        setNeighbors([])
      } finally {
        setIsLoadingNeighbors(false)
      }
    }

    fetchNeighbors()
  }, [selectedNode, jobId])

  // Handlers
  const handleNodeClick = useCallback((node: D3Node) => {
    setSelectedNode(node)
  }, [])

  const handleNodeHover = useCallback((node: D3Node | null) => {
    setHoveredNode(node)
  }, [])

  const handleBackgroundClick = useCallback(() => {
    setSelectedNode(null)
  }, [])

  const handleCloseDetails = useCallback(() => {
    setSelectedNode(null)
  }, [])

  const handleNavigateToNode = useCallback((nodeId: string) => {
    // Find node in data and select it
    const node = graphData?.nodes.find(n => n.id === nodeId)
    if (node) {
      setSelectedNode(node as D3Node)
    }
  }, [graphData?.nodes])


  const handleSelectAll = useCallback(() => {
    setActiveTypes(new Set(Object.keys(NODE_TYPES)))
  }, [setActiveTypes])

  const handleClearAll = useCallback(() => {
    setActiveTypes(new Set())
  }, [setActiveTypes])

  // Canvas control handlers
  const handleZoomIn = useCallback(() => {
    const canvas = canvasContainerRef.current as unknown as { zoomIn?: () => void }
    canvas?.zoomIn?.()
  }, [])

  const handleZoomOut = useCallback(() => {
    const canvas = canvasContainerRef.current as unknown as { zoomOut?: () => void }
    canvas?.zoomOut?.()
  }, [])

  const handleResetView = useCallback(() => {
    const canvas = canvasContainerRef.current as unknown as { resetView?: () => void }
    canvas?.resetView?.()
  }, [])

  const handleFitToView = useCallback(() => {
    const canvas = canvasContainerRef.current as unknown as { fitToView?: () => void }
    canvas?.fitToView?.()
  }, [])

  const handleTransformChange = useCallback((newTransform: Transform) => {
    setTransform(newTransform)
  }, [])

  // Loading state
  if (isLoading) {
    return (
      <div className="animate-in">
        <PageHeader
          title="Knowledge Graph"
          breadcrumbs={[
            { label: 'Dashboard', href: '/' },
            { label: 'Jobs', href: '/jobs' },
            { label: jobId?.slice(0, 8) || '', href: `/jobs/${jobId}` },
            { label: 'Knowledge Graph' },
          ]}
        />
        <div className="flex items-center justify-center h-[600px]">
          <LoadingSpinner size="xl" label="Loading knowledge graph..." />
        </div>
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="animate-in">
        <PageHeader
          title="Knowledge Graph"
          breadcrumbs={[
            { label: 'Dashboard', href: '/' },
            { label: 'Jobs', href: '/jobs' },
            { label: jobId?.slice(0, 8) || '', href: `/jobs/${jobId}` },
            { label: 'Knowledge Graph' },
          ]}
        />
        <GlassCard className="text-center py-12">
          <p className="text-status-error mb-4">Failed to load knowledge graph</p>
          <p className="text-text-muted mb-6">{error.message}</p>
          <Button onClick={refetch}>Try Again</Button>
        </GlassCard>
      </div>
    )
  }

  // Empty state
  if (!graphData || graphData.nodes.length === 0) {
    return (
      <div className="animate-in">
        <PageHeader
          title="Knowledge Graph"
          breadcrumbs={[
            { label: 'Dashboard', href: '/' },
            { label: 'Jobs', href: '/jobs' },
            { label: jobId?.slice(0, 8) || '', href: `/jobs/${jobId}` },
            { label: 'Knowledge Graph' },
          ]}
        />
        <GlassCard className="text-center py-12">
          <p className="text-text-muted">No knowledge graph data available</p>
          <p className="text-body-sm text-text-muted mt-2">
            The graph will be generated after analysis completes
          </p>
        </GlassCard>
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className={cn(
        'animate-in',
        isFullscreen && 'fixed inset-0 z-50 bg-bg-primary p-6'
      )}
    >
      <PageHeader
        title="Knowledge Graph"
        breadcrumbs={
          isFullscreen
            ? undefined
            : [
                { label: 'Dashboard', href: '/' },
                { label: 'Jobs', href: '/jobs' },
                { label: jobId?.slice(0, 8) || '', href: `/jobs/${jobId}` },
                { label: 'Knowledge Graph' },
              ]
        }
        actions={
          <div className="flex items-center gap-3">
            {/* Stats Summary */}
            <div className="hidden md:flex items-center gap-4 text-body-sm text-text-muted">
              <span>
                <span className="text-text-primary font-medium">{stats?.totalNodes || 0}</span> nodes
              </span>
              <span>
                <span className="text-text-primary font-medium">{stats?.totalLinks || 0}</span> links
              </span>
            </div>

            <Link to={`/jobs/${jobId}/results`}>
              <Button variant="secondary">View Results</Button>
            </Link>
            <Link to={`/jobs/${jobId}/reports`}>
              <Button>Download Report</Button>
            </Link>
          </div>
        }
      />

      <div className={cn(
        'grid grid-cols-1 lg:grid-cols-4 gap-6',
        isFullscreen && 'h-[calc(100vh-120px)]'
      )}>
        {/* Controls Sidebar */}
        <div className={cn(
          'space-y-6',
          isFullscreen && 'lg:col-span-1 overflow-y-auto'
        )}>
          {/* Search */}
          <GlassCard title="Search">
            <GraphSearch
              searchFunction={search}
              results={searchResults}
              isSearching={isSearching}
              onClearSearch={clearSearch}
            />
          </GlassCard>

          {/* Node Type Filters */}
          <GlassCard title="Node Types">
            <NodeTypeFilter
              activeTypes={filterState.activeTypes}
              onToggleType={toggleType}
              onSelectAll={handleSelectAll}
              onClearAll={handleClearAll}
              nodeCounts={nodeCounts}
            />
          </GlassCard>

          {/* Selected Node Details */}
          {selectedNode && (
            <NodeDetails
              node={selectedNode}
              neighbors={neighbors}
              neighborCount={neighbors.length}
              onClose={handleCloseDetails}
              onNavigateToNode={handleNavigateToNode}
              isLoading={isLoadingNeighbors}
            />
          )}
        </div>

        {/* Graph Canvas */}
        <div className={cn(
          'lg:col-span-3',
          isFullscreen && 'h-full'
        )}>
          <GlassCard
            noPadding
            className={cn(
              'relative overflow-hidden',
              isFullscreen ? 'h-full' : 'h-[700px]'
            )}
          >
            {/* Canvas */}
            <div ref={canvasContainerRef} className="absolute inset-0">
              <KnowledgeGraphCanvas
                nodes={filteredData.nodes}
                links={filteredData.links}
                selectedNode={selectedNode}
                highlightedIds={searchResults.highlightedIds}
                activeFilters={filterState.activeTypes}
                onNodeClick={handleNodeClick}
                onNodeHover={handleNodeHover}
                onBackgroundClick={handleBackgroundClick}
                onTransformChange={handleTransformChange}
              />
            </div>

            {/* Graph Controls */}
            <GraphControls
              onZoomIn={handleZoomIn}
              onZoomOut={handleZoomOut}
              onResetView={handleResetView}
              onFitToView={handleFitToView}
              onToggleFullscreen={toggleFullscreen}
              isFullscreen={isFullscreen}
              zoomLevel={transform.k}
              className="absolute top-4 right-4"
            />

            {/* Legend */}
            <GraphLegend
              activeTypes={filterState.activeTypes}
              compact
              className="absolute bottom-4 left-4"
            />

            {/* Hover Tooltip */}
            {hoveredNode && (
              <div className="absolute top-4 left-4 max-w-xs pointer-events-none">
                <div className="p-3 rounded-lg bg-bg-glass/95 backdrop-blur-[20px] border border-border shadow-lg">
                  <div className="flex items-center gap-2 mb-1">
                    <span
                      className="w-2.5 h-2.5 rounded-full"
                      style={{
                        backgroundColor:
                          NODE_TYPES[hoveredNode.type.toUpperCase() as keyof typeof NODE_TYPES]?.color ||
                          '#64748b',
                      }}
                    />
                    <span className="text-body-sm font-medium text-text-primary truncate">
                      {hoveredNode.name}
                    </span>
                  </div>
                  <span className="text-caption text-text-muted">
                    {NODE_TYPES[hoveredNode.type.toUpperCase() as keyof typeof NODE_TYPES]?.label ||
                      hoveredNode.type}
                  </span>
                  {hoveredNode.description && (
                    <p className="text-caption text-text-secondary mt-1 line-clamp-2">
                      {hoveredNode.description}
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* Filter indicator */}
            {filteredData.nodes.length < (graphData?.nodes.length || 0) && (
              <div className="absolute top-4 left-1/2 -translate-x-1/2">
                <div className="px-3 py-1.5 rounded-full bg-bg-glass/90 backdrop-blur-[20px] border border-border text-body-sm text-text-secondary">
                  Showing {filteredData.nodes.length} of {graphData?.nodes.length} nodes
                  <button
                    onClick={resetFilters}
                    className="ml-2 text-accent-primary hover:text-accent-secondary transition-colors"
                  >
                    Reset
                  </button>
                </div>
              </div>
            )}
          </GlassCard>
        </div>
      </div>
    </div>
  )
}
