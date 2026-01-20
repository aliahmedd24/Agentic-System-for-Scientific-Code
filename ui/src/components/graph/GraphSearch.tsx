import { useState, useCallback, useEffect } from 'react'
import { cn } from '@/lib/cn'
import { Input } from '@/components/ui/Input'
import { Button } from '@/components/ui/Button'
import { NODE_TYPES } from '@/lib/constants'
import { LoadingSpinner } from '@/components/data-display/LoadingSpinner'
import {
  MagnifyingGlassIcon,
  XMarkIcon,
  FunnelIcon,
} from '@heroicons/react/24/outline'
import type { GraphNode } from '@/api/types'
import type { SearchResult } from './types'

interface GraphSearchProps {
  onSearchResults?: (results: SearchResult) => void
  onClearSearch: () => void
  searchFunction: (query: string, nodeType?: string) => Promise<void>
  results: SearchResult
  isSearching: boolean
  className?: string
}

export function GraphSearch({
  onClearSearch,
  searchFunction,
  results,
  isSearching,
  className,
}: GraphSearchProps) {
  const [query, setQuery] = useState('')
  const [selectedType, setSelectedType] = useState<string | undefined>(undefined)
  const [showTypeFilter, setShowTypeFilter] = useState(false)

  // Trigger search when query or type changes
  useEffect(() => {
    searchFunction(query, selectedType)
  }, [query, selectedType, searchFunction])

  const handleClear = useCallback(() => {
    setQuery('')
    setSelectedType(undefined)
    onClearSearch()
  }, [onClearSearch])

  const handleTypeSelect = useCallback((type: string | undefined) => {
    setSelectedType(type)
    setShowTypeFilter(false)
  }, [])

  const hasResults = results.nodes.length > 0
  const hasQuery = query.trim().length > 0

  return (
    <div className={cn('space-y-3', className)}>
      {/* Search Input */}
      <div className="flex gap-2">
        <div className="flex-1 relative">
          <Input
            placeholder="Search nodes..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            leftIcon={<MagnifyingGlassIcon className="h-5 w-5" />}
            rightIcon={
              isSearching ? (
                <LoadingSpinner size="sm" />
              ) : hasQuery ? (
                <button
                  onClick={handleClear}
                  className="text-text-muted hover:text-text-primary transition-colors"
                  aria-label="Clear search"
                >
                  <XMarkIcon className="h-5 w-5" />
                </button>
              ) : undefined
            }
            inputSize="sm"
          />
        </div>

        {/* Type Filter Toggle */}
        <Button
          variant={selectedType ? 'secondary' : 'icon'}
          size="sm"
          onClick={() => setShowTypeFilter(!showTypeFilter)}
          title="Filter by type"
          className={cn(
            selectedType && 'border-accent-primary text-accent-primary'
          )}
        >
          <FunnelIcon className="h-4 w-4" />
          {selectedType && (
            <span className="ml-1 text-body-sm">
              {NODE_TYPES[selectedType as keyof typeof NODE_TYPES]?.label || selectedType}
            </span>
          )}
        </Button>
      </div>

      {/* Type Filter Dropdown */}
      {showTypeFilter && (
        <div className="p-2 rounded-lg bg-bg-tertiary border border-border">
          <div className="text-caption text-text-muted mb-2">Filter by node type</div>
          <div className="flex flex-wrap gap-1">
            <button
              onClick={() => handleTypeSelect(undefined)}
              className={cn(
                'px-2 py-1 rounded text-body-sm transition-colors',
                !selectedType
                  ? 'bg-accent-primary text-white'
                  : 'bg-bg-secondary text-text-secondary hover:text-text-primary'
              )}
            >
              All Types
            </button>
            {Object.entries(NODE_TYPES).map(([key, config]) => (
              <button
                key={key}
                onClick={() => handleTypeSelect(key)}
                className={cn(
                  'px-2 py-1 rounded text-body-sm transition-colors flex items-center gap-1.5',
                  selectedType === key
                    ? 'bg-accent-primary text-white'
                    : 'bg-bg-secondary text-text-secondary hover:text-text-primary'
                )}
              >
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: config.color }}
                />
                {config.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Search Results Summary */}
      {hasQuery && (
        <div className="flex items-center justify-between text-body-sm">
          {isSearching ? (
            <span className="text-text-muted">Searching...</span>
          ) : hasResults ? (
            <span className="text-text-secondary">
              Found <span className="text-accent-primary font-medium">{results.total}</span> nodes
              {results.total > results.nodes.length && (
                <span className="text-text-muted"> (showing {results.nodes.length})</span>
              )}
            </span>
          ) : (
            <span className="text-text-muted">No nodes found</span>
          )}

          {hasResults && (
            <button
              onClick={handleClear}
              className="text-text-muted hover:text-text-primary transition-colors text-body-sm"
            >
              Clear
            </button>
          )}
        </div>
      )}

      {/* Results Preview */}
      {hasResults && results.nodes.length <= 10 && (
        <div className="space-y-1 max-h-48 overflow-y-auto">
          {results.nodes.slice(0, 10).map((node) => (
            <SearchResultItem key={node.id} node={node} />
          ))}
        </div>
      )}
    </div>
  )
}

// Individual search result item
function SearchResultItem({ node }: { node: GraphNode }) {
  const typeConfig = NODE_TYPES[node.type.toUpperCase() as keyof typeof NODE_TYPES]

  return (
    <div className="flex items-center gap-2 p-2 rounded-lg bg-bg-tertiary/50 hover:bg-bg-tertiary transition-colors">
      <span
        className="w-2.5 h-2.5 rounded-full flex-shrink-0"
        style={{ backgroundColor: typeConfig?.color || '#64748b' }}
      />
      <div className="flex-1 min-w-0">
        <p className="text-body-sm text-text-primary truncate">{node.name}</p>
        {node.description && (
          <p className="text-caption text-text-muted truncate">{node.description}</p>
        )}
      </div>
      <span className="text-caption text-text-muted flex-shrink-0">
        {typeConfig?.label || node.type}
      </span>
    </div>
  )
}

export default GraphSearch
