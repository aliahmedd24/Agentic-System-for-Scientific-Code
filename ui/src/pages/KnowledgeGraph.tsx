import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { PageHeader } from '@/components/layout/PageHeader'
import { GlassCard } from '@/components/ui/GlassCard'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { NODE_TYPES } from '@/lib/constants'
import * as api from '@/api/endpoints'
import type { KnowledgeGraphData, GraphNode } from '@/api/types'
import {
  MagnifyingGlassIcon,
  ArrowsPointingOutIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline'

export default function KnowledgeGraph() {
  const { jobId } = useParams<{ jobId: string }>()
  const [graphData, setGraphData] = useState<KnowledgeGraphData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [activeFilters, setActiveFilters] = useState<string[]>(Object.keys(NODE_TYPES))

  useEffect(() => {
    if (jobId) {
      loadGraph()
    }
  }, [jobId])

  const loadGraph = async () => {
    if (!jobId) return
    setIsLoading(true)
    try {
      const data = await api.getKnowledgeGraph(jobId)
      setGraphData(data)
    } catch (err) {
      console.error('Failed to load knowledge graph:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSearch = async () => {
    if (!jobId || !searchQuery.trim()) return
    try {
      const results = await api.searchKnowledgeGraph(jobId, searchQuery)
      // Highlight matching nodes
      console.log('Search results:', results)
    } catch (err) {
      console.error('Search failed:', err)
    }
  }

  const toggleFilter = (nodeType: string) => {
    setActiveFilters((prev) =>
      prev.includes(nodeType)
        ? prev.filter((t) => t !== nodeType)
        : [...prev, nodeType]
    )
  }

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
        actions={
          <div className="flex items-center gap-3">
            <Link to={`/jobs/${jobId}/results`}>
              <Button variant="secondary">View Results</Button>
            </Link>
            <Link to={`/jobs/${jobId}/reports`}>
              <Button>Download Report</Button>
            </Link>
          </div>
        }
      />

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Controls Sidebar */}
        <div className="space-y-6">
          {/* Search */}
          <GlassCard title="Search">
            <div className="flex gap-2">
              <Input
                placeholder="Search nodes..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              />
              <Button variant="icon" onClick={handleSearch}>
                <MagnifyingGlassIcon className="h-5 w-5" />
              </Button>
            </div>
          </GlassCard>

          {/* Filters */}
          <GlassCard title="Node Types">
            <div className="space-y-2">
              {Object.entries(NODE_TYPES).map(([key, config]) => (
                <label
                  key={key}
                  className="flex items-center gap-3 p-2 rounded-lg hover:bg-bg-tertiary/50 cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={activeFilters.includes(key)}
                    onChange={() => toggleFilter(key)}
                    className="w-4 h-4 rounded border-border bg-bg-tertiary text-accent-primary focus:ring-accent-primary"
                  />
                  <span
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: config.color }}
                  />
                  <span className="text-body-sm text-text-primary">{config.label}</span>
                </label>
              ))}
            </div>
          </GlassCard>

          {/* Selected Node */}
          {selectedNode && (
            <GlassCard title="Selected Node">
              <div className="space-y-3">
                <div>
                  <p className="text-caption text-text-muted">Name</p>
                  <p className="text-body-sm text-text-primary font-medium">{selectedNode.name}</p>
                </div>
                <div>
                  <p className="text-caption text-text-muted">Type</p>
                  <p className="text-body-sm text-text-primary">{selectedNode.type}</p>
                </div>
                {selectedNode.description && (
                  <div>
                    <p className="text-caption text-text-muted">Description</p>
                    <p className="text-body-sm text-text-secondary">{selectedNode.description}</p>
                  </div>
                )}
              </div>
            </GlassCard>
          )}
        </div>

        {/* Graph Canvas */}
        <div className="lg:col-span-3">
          <GlassCard noPadding className="relative overflow-hidden" style={{ height: '600px' }}>
            {isLoading ? (
              <div className="absolute inset-0 flex items-center justify-center text-text-muted">
                Loading knowledge graph...
              </div>
            ) : !graphData || graphData.nodes.length === 0 ? (
              <div className="absolute inset-0 flex items-center justify-center text-text-muted">
                No graph data available
              </div>
            ) : (
              <>
                {/* Placeholder for D3.js visualization */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <div className="w-48 h-48 mx-auto mb-4 rounded-full bg-gradient-to-br from-accent-primary/20 to-purple-500/20 flex items-center justify-center">
                      <div className="text-center">
                        <p className="text-heading-1 text-accent-primary">{graphData.nodes.length}</p>
                        <p className="text-body-sm text-text-muted">nodes</p>
                      </div>
                    </div>
                    <p className="text-body text-text-secondary">
                      Interactive D3.js visualization
                    </p>
                    <p className="text-body-sm text-text-muted mt-1">
                      {graphData.links.length} connections
                    </p>
                  </div>
                </div>

                {/* Controls */}
                <div className="absolute top-4 right-4 flex gap-2">
                  <Button variant="icon" title="Reset View" onClick={loadGraph}>
                    <ArrowPathIcon className="h-5 w-5" />
                  </Button>
                  <Button variant="icon" title="Fullscreen">
                    <ArrowsPointingOutIcon className="h-5 w-5" />
                  </Button>
                </div>

                {/* Legend */}
                <div className="absolute bottom-4 left-4 flex flex-wrap gap-3 p-3 rounded-lg bg-bg-secondary/90 backdrop-blur-sm">
                  {Object.entries(NODE_TYPES)
                    .filter(([key]) => activeFilters.includes(key))
                    .slice(0, 6)
                    .map(([key, config]) => (
                      <div key={key} className="flex items-center gap-1.5">
                        <span
                          className="w-2.5 h-2.5 rounded-full"
                          style={{ backgroundColor: config.color }}
                        />
                        <span className="text-caption text-text-muted">{config.label}</span>
                      </div>
                    ))}
                </div>
              </>
            )}
          </GlassCard>
        </div>
      </div>
    </div>
  )
}
