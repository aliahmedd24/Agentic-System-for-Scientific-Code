import { useState, useMemo } from 'react'
import { cn } from '@/lib/cn'
import { GlassCard } from '@/components/ui/GlassCard'
import { formatDuration, formatNumber } from '@/lib/formatters'
import {
  CpuChipIcon,
  ChevronUpIcon,
  ChevronDownIcon,
} from '@heroicons/react/24/outline'
import type { AgentMetric } from '@/api/types'

type SortColumn = 'agent_name' | 'operations' | 'total_duration_ms' | 'avg_duration_ms' | 'errors'
type SortDirection = 'asc' | 'desc'

interface AgentTableProps {
  agents: AgentMetric[]
  isLoading?: boolean
  className?: string
}

interface SortState {
  column: SortColumn
  direction: SortDirection
}

export function AgentTable({ agents, isLoading, className }: AgentTableProps) {
  const [sortState, setSortState] = useState<SortState>({
    column: 'operations',
    direction: 'desc',
  })

  const handleSort = (column: SortColumn) => {
    setSortState((prev) => ({
      column,
      direction: prev.column === column && prev.direction === 'desc' ? 'asc' : 'desc',
    }))
  }

  const sortedAgents = useMemo(() => {
    const sorted = [...agents].sort((a, b) => {
      const { column, direction } = sortState
      let comparison = 0

      switch (column) {
        case 'agent_name':
          comparison = a.agent_name.localeCompare(b.agent_name)
          break
        case 'operations':
          comparison = a.operations - b.operations
          break
        case 'total_duration_ms':
          comparison = a.total_duration_ms - b.total_duration_ms
          break
        case 'avg_duration_ms':
          comparison = a.avg_duration_ms - b.avg_duration_ms
          break
        case 'errors':
          comparison = a.errors - b.errors
          break
      }

      return direction === 'asc' ? comparison : -comparison
    })

    return sorted
  }, [agents, sortState])

  const SortIcon = ({ column }: { column: SortColumn }) => {
    if (sortState.column !== column) {
      return <ChevronUpIcon className="h-4 w-4 text-text-muted opacity-0 group-hover:opacity-50" />
    }
    return sortState.direction === 'asc' ? (
      <ChevronUpIcon className="h-4 w-4 text-accent-primary" />
    ) : (
      <ChevronDownIcon className="h-4 w-4 text-accent-primary" />
    )
  }

  const columns: { key: SortColumn; label: string; align: 'left' | 'right' }[] = [
    { key: 'agent_name', label: 'Agent', align: 'left' },
    { key: 'operations', label: 'Operations', align: 'right' },
    { key: 'total_duration_ms', label: 'Total Time', align: 'right' },
    { key: 'avg_duration_ms', label: 'Avg Time', align: 'right' },
    { key: 'errors', label: 'Errors', align: 'right' },
  ]

  if (isLoading) {
    return (
      <GlassCard title="Agent Details" noPadding className={className}>
        <div className="p-8 text-center text-text-muted">Loading agent data...</div>
      </GlassCard>
    )
  }

  if (!agents || agents.length === 0) {
    return (
      <GlassCard title="Agent Details" noPadding className={className}>
        <div className="py-12 text-center">
          <CpuChipIcon className="h-12 w-12 mx-auto text-text-muted mb-3" />
          <p className="text-text-secondary">No agent data available</p>
        </div>
      </GlassCard>
    )
  }

  return (
    <GlassCard title="Agent Details" noPadding className={className}>
      <div className="overflow-x-auto">
        <table className="w-full" role="grid">
          <thead>
            <tr className="border-b border-border">
              {columns.map((col) => (
                <th
                  key={col.key}
                  scope="col"
                  aria-sort={
                    sortState.column === col.key
                      ? sortState.direction === 'asc'
                        ? 'ascending'
                        : 'descending'
                      : 'none'
                  }
                  className={cn(
                    'px-6 py-4',
                    col.align === 'left' ? 'text-left' : 'text-right'
                  )}
                >
                  <button
                    onClick={() => handleSort(col.key)}
                    className={cn(
                      'group inline-flex items-center gap-1.5 text-body-sm font-semibold text-text-secondary',
                      'hover:text-text-primary transition-colors',
                      col.align === 'right' && 'flex-row-reverse'
                    )}
                    aria-label={`Sort by ${col.label}`}
                  >
                    {col.label}
                    <SortIcon column={col.key} />
                  </button>
                </th>
              ))}
              <th
                scope="col"
                className="px-6 py-4 text-right text-body-sm font-semibold text-text-secondary"
              >
                Error Rate
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedAgents.map((agent) => {
              const errorRate = agent.operations > 0 ? agent.errors / agent.operations : 0

              return (
                <tr
                  key={agent.agent_name}
                  className="border-b border-border/50 hover:bg-bg-tertiary/30 transition-colors"
                >
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-lg bg-accent-primary/20 flex items-center justify-center">
                        <CpuChipIcon className="h-4 w-4 text-accent-primary" />
                      </div>
                      <span className="text-body-sm font-medium text-text-primary">
                        {agent.agent_name}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <span className="text-body-sm text-text-primary font-medium">
                      {formatNumber(agent.operations)}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <span className="text-body-sm text-text-secondary">
                      {formatDuration(agent.total_duration_ms)}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <span className="text-body-sm text-text-secondary">
                      {formatDuration(agent.avg_duration_ms)}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <span
                      className={cn(
                        'text-body-sm',
                        agent.errors > 0 ? 'text-status-error' : 'text-text-muted'
                      )}
                    >
                      {agent.errors}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <span
                      className={cn(
                        'inline-flex items-center px-2 py-0.5 rounded-full text-caption font-medium',
                        errorRate > 0.1
                          ? 'bg-status-error/20 text-status-error'
                          : errorRate > 0
                          ? 'bg-status-warning/20 text-status-warning'
                          : 'bg-status-success/20 text-status-success'
                      )}
                    >
                      {(errorRate * 100).toFixed(1)}%
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </GlassCard>
  )
}

export default AgentTable
