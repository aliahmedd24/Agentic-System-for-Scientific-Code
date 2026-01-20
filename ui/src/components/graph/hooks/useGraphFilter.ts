import { useState, useMemo, useCallback } from 'react'
import { NODE_TYPES } from '@/lib/constants'
import type { GraphNode, GraphLink } from '@/api/types'
import type { UseGraphFilterReturn, NodeTypeFilterState } from '../types'

const ALL_NODE_TYPES = new Set(Object.keys(NODE_TYPES))

export function useGraphFilter(
  nodes: GraphNode[],
  links: GraphLink[]
): UseGraphFilterReturn {
  const [filterState, setFilterState] = useState<NodeTypeFilterState>({
    activeTypes: ALL_NODE_TYPES,
    minConnections: 0,
    includeIsolated: true,
  })

  const filteredData = useMemo(() => {
    const { activeTypes, minConnections, includeIsolated } = filterState

    // First filter by type
    const typeFilteredNodes = nodes.filter(node =>
      activeTypes.has(node.type.toUpperCase())
    )
    const typeFilteredIds = new Set(typeFilteredNodes.map(n => n.id))

    // Filter links to only include those between filtered nodes
    const filteredLinks = links.filter(link => {
      const sourceId = typeof link.source === 'string' ? link.source : (link.source as GraphNode).id
      const targetId = typeof link.target === 'string' ? link.target : (link.target as GraphNode).id
      return typeFilteredIds.has(sourceId) && typeFilteredIds.has(targetId)
    })

    // Calculate connection counts
    const connectionCounts = new Map<string, number>()
    filteredLinks.forEach(link => {
      const sourceId = typeof link.source === 'string' ? link.source : (link.source as GraphNode).id
      const targetId = typeof link.target === 'string' ? link.target : (link.target as GraphNode).id
      connectionCounts.set(sourceId, (connectionCounts.get(sourceId) || 0) + 1)
      connectionCounts.set(targetId, (connectionCounts.get(targetId) || 0) + 1)
    })

    // Filter by connection count
    const finalNodes = typeFilteredNodes.filter(node => {
      const connections = connectionCounts.get(node.id) || 0
      if (connections === 0) {
        return includeIsolated
      }
      return connections >= minConnections
    })

    // Final set of node IDs
    const finalNodeIds = new Set(finalNodes.map(n => n.id))

    // Final links
    const finalLinks = filteredLinks.filter(link => {
      const sourceId = typeof link.source === 'string' ? link.source : (link.source as GraphNode).id
      const targetId = typeof link.target === 'string' ? link.target : (link.target as GraphNode).id
      return finalNodeIds.has(sourceId) && finalNodeIds.has(targetId)
    })

    return { nodes: finalNodes, links: finalLinks }
  }, [nodes, links, filterState])

  const setActiveTypes = useCallback((types: Set<string>) => {
    setFilterState(prev => ({ ...prev, activeTypes: types }))
  }, [])

  const toggleType = useCallback((type: string) => {
    setFilterState(prev => {
      const newTypes = new Set(prev.activeTypes)
      if (newTypes.has(type)) {
        newTypes.delete(type)
      } else {
        newTypes.add(type)
      }
      return { ...prev, activeTypes: newTypes }
    })
  }, [])

  const setMinConnections = useCallback((min: number) => {
    setFilterState(prev => ({ ...prev, minConnections: min }))
  }, [])

  const setIncludeIsolated = useCallback((include: boolean) => {
    setFilterState(prev => ({ ...prev, includeIsolated: include }))
  }, [])

  const resetFilters = useCallback(() => {
    setFilterState({
      activeTypes: ALL_NODE_TYPES,
      minConnections: 0,
      includeIsolated: true,
    })
  }, [])

  return {
    filteredData,
    filterState,
    setActiveTypes,
    toggleType,
    setMinConnections,
    setIncludeIsolated,
    resetFilters,
  }
}

export default useGraphFilter
