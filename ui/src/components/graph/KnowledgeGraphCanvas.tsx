import { useRef, useEffect, useCallback, useState, useMemo } from 'react'
import * as d3 from 'd3'
import { cn } from '@/lib/cn'
import { NODE_TYPES } from '@/lib/constants'
import type { GraphNode, GraphLink } from '@/api/types'
import type { D3Node, D3Link, Transform } from './types'

interface KnowledgeGraphCanvasProps {
  nodes: GraphNode[]
  links: GraphLink[]
  selectedNode: D3Node | null
  highlightedIds: Set<string>
  activeFilters: Set<string>
  onNodeClick: (node: D3Node) => void
  onNodeHover: (node: D3Node | null, event?: MouseEvent) => void
  onBackgroundClick: () => void
  onTransformChange?: (transform: Transform) => void
  className?: string
}

// Performance constants
const LARGE_GRAPH_THRESHOLD = 500
const LABEL_ZOOM_THRESHOLD = 0.8
const NODE_RADIUS = {
  default: 8,
  selected: 12,
  highlighted: 10,
}
const LINK_WIDTH = {
  default: 1,
  highlighted: 2,
}

export function KnowledgeGraphCanvas({
  nodes,
  links,
  selectedNode,
  highlightedIds,
  activeFilters,
  onNodeClick,
  onNodeHover,
  onBackgroundClick,
  onTransformChange,
  className,
}: KnowledgeGraphCanvasProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const simulationRef = useRef<d3.Simulation<D3Node, D3Link> | null>(null)
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null)

  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })
  const [currentTransform, setCurrentTransform] = useState<Transform>({ x: 0, y: 0, k: 1 })

  // Determine if we should optimize for large graphs
  const isLargeGraph = nodes.length > LARGE_GRAPH_THRESHOLD

  // Convert to D3 format and filter
  const { d3Nodes, d3Links } = useMemo(() => {
    // Filter nodes by active types
    const filteredNodes = nodes.filter(node =>
      activeFilters.has(node.type.toUpperCase())
    )
    const nodeIds = new Set(filteredNodes.map(n => n.id))

    // Filter links
    const filteredLinks = links.filter(link => {
      const sourceId = typeof link.source === 'string' ? link.source : link.source
      const targetId = typeof link.target === 'string' ? link.target : link.target
      return nodeIds.has(sourceId) && nodeIds.has(targetId)
    })

    const d3Nodes: D3Node[] = filteredNodes.map(node => ({
      ...node,
      x: undefined,
      y: undefined,
    }))

    const d3Links: D3Link[] = filteredLinks.map(link => ({
      source: typeof link.source === 'string' ? link.source : link.source,
      target: typeof link.target === 'string' ? link.target : link.target,
      type: link.type,
      weight: link.weight,
    }))

    return { d3Nodes, d3Links }
  }, [nodes, links, activeFilters])

  // Resize observer
  useEffect(() => {
    if (!containerRef.current) return

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (entry) {
        const { width, height } = entry.contentRect
        setDimensions({ width, height })
      }
    })

    observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [])

  // Get node color
  const getNodeColor = useCallback((node: D3Node) => {
    const typeKey = node.type.toUpperCase()
    const config = NODE_TYPES[typeKey as keyof typeof NODE_TYPES]
    return config?.color || '#64748b'
  }, [])

  // Get node radius
  const getNodeRadius = useCallback((node: D3Node) => {
    if (selectedNode?.id === node.id) return NODE_RADIUS.selected
    if (highlightedIds.has(node.id)) return NODE_RADIUS.highlighted
    return NODE_RADIUS.default
  }, [selectedNode, highlightedIds])

  // Get node opacity
  const getNodeOpacity = useCallback((node: D3Node) => {
    if (highlightedIds.size === 0) return 1
    return highlightedIds.has(node.id) ? 1 : 0.2
  }, [highlightedIds])

  // Get link opacity
  const getLinkOpacity = useCallback((link: D3Link) => {
    if (highlightedIds.size === 0) return 0.6
    const sourceId = typeof link.source === 'string' ? link.source : (link.source as D3Node).id
    const targetId = typeof link.target === 'string' ? link.target : (link.target as D3Node).id
    return highlightedIds.has(sourceId) && highlightedIds.has(targetId) ? 0.8 : 0.1
  }, [highlightedIds])

  // Initialize simulation and rendering
  useEffect(() => {
    if (!svgRef.current || d3Nodes.length === 0) return

    const svg = d3.select(svgRef.current)
    const width = dimensions.width
    const height = dimensions.height

    // Clear existing content
    svg.selectAll('*').remove()

    // Create container group for zoom/pan
    const g = svg.append('g').attr('class', 'graph-container')

    // Create arrow markers for directed edges
    svg.append('defs')
      .append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .append('path')
      .attr('d', 'M 0,-5 L 10,0 L 0,5')
      .attr('fill', '#64748b')
      .attr('opacity', 0.5)

    // Create links
    const linkGroup = g.append('g').attr('class', 'links')
    const link = linkGroup
      .selectAll<SVGLineElement, D3Link>('line')
      .data(d3Links)
      .join('line')
      .attr('stroke', '#64748b')
      .attr('stroke-width', LINK_WIDTH.default)
      .attr('stroke-opacity', 0.6)
      .attr('marker-end', 'url(#arrowhead)')

    // Create nodes
    const nodeGroup = g.append('g').attr('class', 'nodes')
    const node = nodeGroup
      .selectAll<SVGGElement, D3Node>('g')
      .data(d3Nodes)
      .join('g')
      .attr('class', 'node')
      .attr('cursor', 'pointer')

    // Node circles
    node
      .append('circle')
      .attr('r', d => getNodeRadius(d))
      .attr('fill', d => getNodeColor(d))
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .attr('opacity', d => getNodeOpacity(d))

    // Node labels (only visible at certain zoom levels)
    const labels = node
      .append('text')
      .attr('dx', 12)
      .attr('dy', 4)
      .attr('font-size', '11px')
      .attr('fill', '#e2e8f0')
      .attr('pointer-events', 'none')
      .attr('opacity', currentTransform.k >= LABEL_ZOOM_THRESHOLD ? 1 : 0)
      .text(d => d.name.length > 20 ? d.name.slice(0, 20) + '...' : d.name)

    // Node interaction handlers
    node
      .on('click', (event, d) => {
        event.stopPropagation()
        onNodeClick(d)
      })
      .on('mouseenter', (event, d) => {
        onNodeHover(d, event)
        d3.select(event.currentTarget)
          .select('circle')
          .transition()
          .duration(150)
          .attr('r', getNodeRadius(d) * 1.3)
      })
      .on('mouseleave', (event, d) => {
        onNodeHover(null)
        d3.select(event.currentTarget)
          .select('circle')
          .transition()
          .duration(150)
          .attr('r', getNodeRadius(d))
      })

    // Drag behavior
    const drag = d3.drag<SVGGElement, D3Node>()
      .on('start', (event, d) => {
        if (!event.active) simulationRef.current?.alphaTarget(0.3).restart()
        d.fx = d.x
        d.fy = d.y
      })
      .on('drag', (event, d) => {
        d.fx = event.x
        d.fy = event.y
      })
      .on('end', (event, d) => {
        if (!event.active) simulationRef.current?.alphaTarget(0)
        d.fx = null
        d.fy = null
      })

    node.call(drag)

    // Background click handler
    svg.on('click', () => {
      onBackgroundClick()
    })

    // Create force simulation
    const simulation = d3.forceSimulation<D3Node>(d3Nodes)
      .force('link', d3.forceLink<D3Node, D3Link>(d3Links)
        .id(d => d.id)
        .distance(isLargeGraph ? 80 : 100)
        .strength(0.5)
      )
      .force('charge', d3.forceManyBody()
        .strength(isLargeGraph ? -100 : -200)
        .distanceMax(300)
      )
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(15))

    // Reduce simulation intensity for large graphs
    if (isLargeGraph) {
      simulation.alphaDecay(0.05)
    }

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as D3Node).x ?? 0)
        .attr('y1', d => (d.source as D3Node).y ?? 0)
        .attr('x2', d => (d.target as D3Node).x ?? 0)
        .attr('y2', d => (d.target as D3Node).y ?? 0)

      node.attr('transform', d => `translate(${d.x ?? 0},${d.y ?? 0})`)
    })

    simulationRef.current = simulation

    // Setup zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform.toString())
        const newTransform = {
          x: event.transform.x,
          y: event.transform.y,
          k: event.transform.k,
        }
        setCurrentTransform(newTransform)
        onTransformChange?.(newTransform)

        // Toggle labels based on zoom level
        labels.attr('opacity', event.transform.k >= LABEL_ZOOM_THRESHOLD ? 1 : 0)
      })

    svg.call(zoom)
    zoomRef.current = zoom

    // Apply initial transform if exists
    if (currentTransform.k !== 1 || currentTransform.x !== 0 || currentTransform.y !== 0) {
      svg.call(zoom.transform, d3.zoomIdentity
        .translate(currentTransform.x, currentTransform.y)
        .scale(currentTransform.k)
      )
    }

    // Cleanup
    return () => {
      simulation.stop()
      simulationRef.current = null
    }
  }, [d3Nodes, d3Links, dimensions, isLargeGraph, getNodeColor, getNodeRadius, getNodeOpacity, getLinkOpacity, onNodeClick, onNodeHover, onBackgroundClick, onTransformChange])

  // Update node styles when selection/highlight changes
  useEffect(() => {
    if (!svgRef.current) return

    const svg = d3.select(svgRef.current)

    svg.selectAll<SVGCircleElement, D3Node>('.nodes circle')
      .transition()
      .duration(200)
      .attr('r', d => getNodeRadius(d))
      .attr('opacity', d => getNodeOpacity(d))
      .attr('stroke-width', d => selectedNode?.id === d.id ? 3 : 1.5)
      .attr('stroke', d => selectedNode?.id === d.id ? '#fff' : '#fff')

    svg.selectAll<SVGLineElement, D3Link>('.links line')
      .transition()
      .duration(200)
      .attr('stroke-opacity', d => getLinkOpacity(d))
      .attr('stroke-width', d => {
        if (highlightedIds.size === 0) return LINK_WIDTH.default
        const sourceId = typeof d.source === 'string' ? d.source : (d.source as D3Node).id
        const targetId = typeof d.target === 'string' ? d.target : (d.target as D3Node).id
        return highlightedIds.has(sourceId) && highlightedIds.has(targetId)
          ? LINK_WIDTH.highlighted
          : LINK_WIDTH.default
      })
  }, [selectedNode, highlightedIds, getNodeRadius, getNodeOpacity, getLinkOpacity])

  // Expose zoom controls
  const zoomIn = useCallback(() => {
    if (!svgRef.current || !zoomRef.current) return
    d3.select(svgRef.current)
      .transition()
      .duration(300)
      .call(zoomRef.current.scaleBy, 1.3)
  }, [])

  const zoomOut = useCallback(() => {
    if (!svgRef.current || !zoomRef.current) return
    d3.select(svgRef.current)
      .transition()
      .duration(300)
      .call(zoomRef.current.scaleBy, 0.7)
  }, [])

  const resetView = useCallback(() => {
    if (!svgRef.current || !zoomRef.current) return
    d3.select(svgRef.current)
      .transition()
      .duration(500)
      .call(zoomRef.current.transform, d3.zoomIdentity)
  }, [])

  const fitToView = useCallback(() => {
    if (!svgRef.current || !zoomRef.current || d3Nodes.length === 0) return

    const svg = d3.select(svgRef.current)
    const containerNode = svg.select('.graph-container').node() as SVGGElement | null
    const bounds = containerNode?.getBBox()
    if (!bounds) return

    const { width, height } = dimensions
    const padding = 50

    const scale = Math.min(
      (width - padding * 2) / bounds.width,
      (height - padding * 2) / bounds.height,
      2
    )
    const translateX = (width - bounds.width * scale) / 2 - bounds.x * scale
    const translateY = (height - bounds.height * scale) / 2 - bounds.y * scale

    svg
      .transition()
      .duration(500)
      .call(
        zoomRef.current.transform,
        d3.zoomIdentity.translate(translateX, translateY).scale(scale)
      )
  }, [d3Nodes.length, dimensions])

  // Expose controls via ref
  useEffect(() => {
    // Store refs on the container for external access
    const container = containerRef.current
    if (container) {
      (container as unknown as { zoomIn: typeof zoomIn }).zoomIn = zoomIn;
      (container as unknown as { zoomOut: typeof zoomOut }).zoomOut = zoomOut;
      (container as unknown as { resetView: typeof resetView }).resetView = resetView;
      (container as unknown as { fitToView: typeof fitToView }).fitToView = fitToView;
      (container as unknown as { transform: Transform }).transform = currentTransform
    }
  }, [zoomIn, zoomOut, resetView, fitToView, currentTransform])

  return (
    <div
      ref={containerRef}
      className={cn('relative w-full h-full overflow-hidden bg-bg-secondary/50', className)}
    >
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        className="absolute inset-0"
      />

      {/* Empty state */}
      {d3Nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <p className="text-body text-text-muted">No nodes match current filters</p>
            <p className="text-body-sm text-text-muted mt-1">Try adjusting your filters</p>
          </div>
        </div>
      )}

      {/* Node count indicator for large graphs */}
      {isLargeGraph && (
        <div className="absolute top-2 left-2 px-2 py-1 rounded bg-status-warning/20 text-status-warning text-caption">
          Large graph: {d3Nodes.length} nodes
        </div>
      )}
    </div>
  )
}

export default KnowledgeGraphCanvas
