import { cn } from '@/lib/cn'
import { NODE_TYPES } from '@/lib/constants'
import { Button } from '@/components/ui/Button'
import { GlassCard } from '@/components/ui/GlassCard'
import { LoadingSpinner } from '@/components/data-display/LoadingSpinner'
import {
  XMarkIcon,
  ArrowTopRightOnSquareIcon,
  LinkIcon,
  DocumentTextIcon,
  CodeBracketIcon,
  CubeIcon,
} from '@heroicons/react/24/outline'
import type { GraphNode } from '@/api/types'
import type { D3Node } from './types'

interface NodeDetailsProps {
  node: D3Node | null
  neighbors?: GraphNode[]
  neighborCount?: number
  onClose: () => void
  onNavigateToNode: (nodeId: string) => void
  isLoading?: boolean
  className?: string
}

// Icons for different node type categories
const TYPE_ICONS: Record<string, React.ReactNode> = {
  PAPER: <DocumentTextIcon className="h-5 w-5" />,
  SECTION: <DocumentTextIcon className="h-5 w-5" />,
  CONCEPT: <CubeIcon className="h-5 w-5" />,
  ALGORITHM: <CubeIcon className="h-5 w-5" />,
  EQUATION: <CubeIcon className="h-5 w-5" />,
  REPOSITORY: <CodeBracketIcon className="h-5 w-5" />,
  FILE: <CodeBracketIcon className="h-5 w-5" />,
  CLASS: <CodeBracketIcon className="h-5 w-5" />,
  FUNCTION: <CodeBracketIcon className="h-5 w-5" />,
  MODULE: <CodeBracketIcon className="h-5 w-5" />,
  MAPPING: <LinkIcon className="h-5 w-5" />,
}

export function NodeDetails({
  node,
  neighbors,
  neighborCount,
  onClose,
  onNavigateToNode,
  isLoading,
  className,
}: NodeDetailsProps) {
  if (!node) return null

  const typeKey = node.type.toUpperCase()
  const typeConfig = NODE_TYPES[typeKey as keyof typeof NODE_TYPES]
  const icon = TYPE_ICONS[typeKey] || <CubeIcon className="h-5 w-5" />

  return (
    <GlassCard
      className={cn('relative', className)}
      noPadding
    >
      {/* Header */}
      <div
        className="p-4 border-b border-border"
        style={{ borderLeftColor: typeConfig?.color, borderLeftWidth: 4 }}
      >
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-start gap-3 min-w-0">
            <div
              className="flex-shrink-0 p-2 rounded-lg"
              style={{ backgroundColor: `${typeConfig?.color}20` }}
            >
              <span style={{ color: typeConfig?.color }}>{icon}</span>
            </div>
            <div className="min-w-0">
              <h3 className="text-heading-3 text-text-primary truncate pr-2">
                {node.name}
              </h3>
              <div className="flex items-center gap-2 mt-1">
                <span
                  className="inline-flex items-center px-2 py-0.5 rounded text-caption font-medium"
                  style={{
                    backgroundColor: `${typeConfig?.color}20`,
                    color: typeConfig?.color,
                  }}
                >
                  {typeConfig?.label || node.type}
                </span>
                {neighborCount !== undefined && (
                  <span className="text-caption text-text-muted">
                    {neighborCount} connections
                  </span>
                )}
              </div>
            </div>
          </div>
          <Button
            variant="icon"
            size="sm"
            onClick={onClose}
            aria-label="Close details"
          >
            <XMarkIcon className="h-5 w-5" />
          </Button>
        </div>
      </div>

      {/* Content */}
      <div className="p-4 space-y-4">
        {/* Description */}
        {node.description && (
          <div>
            <h4 className="text-caption text-text-muted uppercase tracking-wider mb-1">
              Description
            </h4>
            <p className="text-body-sm text-text-secondary">{node.description}</p>
          </div>
        )}

        {/* Metadata */}
        {node.metadata && Object.keys(node.metadata).length > 0 && (
          <div>
            <h4 className="text-caption text-text-muted uppercase tracking-wider mb-2">
              Properties
            </h4>
            <div className="space-y-2">
              {Object.entries(node.metadata).map(([key, value]) => (
                <div key={key} className="flex justify-between items-start gap-2">
                  <span className="text-body-sm text-text-muted">{formatKey(key)}</span>
                  <span className="text-body-sm text-text-primary text-right truncate max-w-[60%]">
                    {formatValue(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Connected Nodes */}
        <div>
          <h4 className="text-caption text-text-muted uppercase tracking-wider mb-2">
            Connected Nodes
          </h4>
          {isLoading ? (
            <div className="flex items-center justify-center py-4">
              <LoadingSpinner size="sm" label="Loading connections..." />
            </div>
          ) : neighbors && neighbors.length > 0 ? (
            <div className="space-y-1 max-h-48 overflow-y-auto">
              {neighbors.map((neighbor) => {
                const neighborType = neighbor.type.toUpperCase()
                const neighborConfig = NODE_TYPES[neighborType as keyof typeof NODE_TYPES]

                return (
                  <button
                    key={neighbor.id}
                    onClick={() => onNavigateToNode(neighbor.id)}
                    className={cn(
                      'w-full flex items-center gap-2 p-2 rounded-lg text-left',
                      'bg-bg-tertiary/50 hover:bg-bg-tertiary transition-colors',
                      'group'
                    )}
                  >
                    <span
                      className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                      style={{ backgroundColor: neighborConfig?.color || '#64748b' }}
                    />
                    <span className="flex-1 text-body-sm text-text-primary truncate">
                      {neighbor.name}
                    </span>
                    <span className="text-caption text-text-muted flex-shrink-0">
                      {neighborConfig?.label || neighbor.type}
                    </span>
                    <ArrowTopRightOnSquareIcon className="h-3.5 w-3.5 text-text-muted opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0" />
                  </button>
                )
              })}
            </div>
          ) : (
            <p className="text-body-sm text-text-muted py-2">No connected nodes</p>
          )}
        </div>

        {/* Node ID (for debugging) */}
        <div className="pt-2 border-t border-border">
          <div className="flex justify-between items-center">
            <span className="text-caption text-text-muted">Node ID</span>
            <code className="text-caption text-text-muted bg-bg-tertiary px-1.5 py-0.5 rounded font-mono">
              {node.id.length > 12 ? `${node.id.slice(0, 6)}...${node.id.slice(-6)}` : node.id}
            </code>
          </div>
        </div>
      </div>
    </GlassCard>
  )
}

// Helper functions
function formatKey(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/([A-Z])/g, ' $1')
    .replace(/^./, (str) => str.toUpperCase())
    .trim()
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '-'
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (typeof value === 'number') return value.toLocaleString()
  if (Array.isArray(value)) return value.length > 0 ? value.join(', ') : '-'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

export default NodeDetails
