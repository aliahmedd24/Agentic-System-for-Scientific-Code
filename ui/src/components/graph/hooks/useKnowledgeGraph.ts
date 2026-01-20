import { useState, useEffect, useCallback, useMemo } from 'react'
import * as api from '@/api/endpoints'
import type { GraphNode, GraphLink } from '@/api/types'
import type { UseKnowledgeGraphReturn, GraphStats } from '../types'

export function useKnowledgeGraph(jobId: string | undefined): UseKnowledgeGraphReturn {
  const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; links: GraphLink[] } | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  const fetchGraph = useCallback(async () => {
    if (!jobId) {
      setIsLoading(false)
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const data = await api.getKnowledgeGraph(jobId)
      setGraphData(data)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load knowledge graph'
      setError(new Error(message))
      console.error('Failed to load knowledge graph:', err)
    } finally {
      setIsLoading(false)
    }
  }, [jobId])

  useEffect(() => {
    fetchGraph()
  }, [fetchGraph])

  const stats = useMemo((): GraphStats | null => {
    if (!graphData) return null

    const { nodes, links } = graphData

    // Count nodes by type
    const nodesByType: Record<string, number> = {}
    nodes.forEach(node => {
      nodesByType[node.type] = (nodesByType[node.type] || 0) + 1
    })

    // Calculate connection counts per node
    const connectionCounts = new Map<string, number>()
    links.forEach(link => {
      const sourceId = typeof link.source === 'string' ? link.source : link.source
      const targetId = typeof link.target === 'string' ? link.target : link.target
      connectionCounts.set(sourceId, (connectionCounts.get(sourceId) || 0) + 1)
      connectionCounts.set(targetId, (connectionCounts.get(targetId) || 0) + 1)
    })

    const connectionValues = Array.from(connectionCounts.values())
    const avgConnections = connectionValues.length > 0
      ? connectionValues.reduce((a, b) => a + b, 0) / connectionValues.length
      : 0

    // Count isolated nodes (no connections)
    const connectedNodeIds = new Set(connectionCounts.keys())
    const isolatedNodes = nodes.filter(n => !connectedNodeIds.has(n.id)).length

    return {
      totalNodes: nodes.length,
      totalLinks: links.length,
      nodesByType,
      avgConnections: Math.round(avgConnections * 10) / 10,
      isolatedNodes,
    }
  }, [graphData])

  return {
    graphData,
    isLoading,
    error,
    refetch: fetchGraph,
    stats,
  }
}

export default useKnowledgeGraph
