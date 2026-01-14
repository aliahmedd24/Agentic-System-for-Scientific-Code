import { useState, useMemo } from 'react'
import { MagnifyingGlassIcon, FunnelIcon } from '@heroicons/react/24/outline'
import { cn } from '@/lib/cn'
import { Input } from '@/components/ui/Input'
import { Select } from '@/components/ui/Select'
import { ConceptMappingCard } from './ConceptMappingCard'
import type { MappingResult } from '@/api/types'

interface MappingsListProps {
  mappings: MappingResult[]
  className?: string
}

type SortOption = 'confidence-desc' | 'confidence-asc' | 'concept' | 'file'

const sortOptions = [
  { id: 'confidence-desc', label: 'Confidence (High to Low)' },
  { id: 'confidence-asc', label: 'Confidence (Low to High)' },
  { id: 'concept', label: 'Concept Name (A-Z)' },
  { id: 'file', label: 'Code File (A-Z)' },
]

export function MappingsList({ mappings, className }: MappingsListProps) {
  const [search, setSearch] = useState('')
  const [minConfidence, setMinConfidence] = useState(0)
  const [sortBy, setSortBy] = useState<SortOption>('confidence-desc')
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const filteredAndSorted = useMemo(() => {
    let result = [...mappings]

    // Filter by search term
    if (search) {
      const searchLower = search.toLowerCase()
      result = result.filter(
        (m) =>
          m.concept_name.toLowerCase().includes(searchLower) ||
          m.code_element.toLowerCase().includes(searchLower) ||
          m.code_file.toLowerCase().includes(searchLower)
      )
    }

    // Filter by minimum confidence
    if (minConfidence > 0) {
      result = result.filter((m) => m.confidence >= minConfidence / 100)
    }

    // Sort
    result.sort((a, b) => {
      switch (sortBy) {
        case 'confidence-desc':
          return b.confidence - a.confidence
        case 'confidence-asc':
          return a.confidence - b.confidence
        case 'concept':
          return a.concept_name.localeCompare(b.concept_name)
        case 'file':
          return a.code_file.localeCompare(b.code_file)
        default:
          return 0
      }
    })

    return result
  }, [mappings, search, minConfidence, sortBy])

  const getMappingId = (m: MappingResult) => `${m.concept_name}-${m.code_element}`

  return (
    <div className={cn('space-y-4', className)}>
      {/* Filters */}
      <div className="flex flex-col lg:flex-row gap-4">
        {/* Search */}
        <div className="flex-1">
          <Input
            placeholder="Search concepts, code elements, or files..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            leftIcon={<MagnifyingGlassIcon className="h-4 w-4" />}
          />
        </div>

        {/* Confidence slider */}
        <div className="w-full lg:w-64">
          <div className="flex items-center gap-2 mb-1">
            <FunnelIcon className="h-4 w-4 text-text-muted" />
            <label className="text-body-sm text-text-secondary">
              Min Confidence: {minConfidence}%
            </label>
          </div>
          <input
            type="range"
            min="0"
            max="100"
            value={minConfidence}
            onChange={(e) => setMinConfidence(Number(e.target.value))}
            className={cn(
              'w-full h-2 rounded-full appearance-none cursor-pointer',
              'bg-bg-tertiary',
              '[&::-webkit-slider-thumb]:appearance-none',
              '[&::-webkit-slider-thumb]:w-4',
              '[&::-webkit-slider-thumb]:h-4',
              '[&::-webkit-slider-thumb]:rounded-full',
              '[&::-webkit-slider-thumb]:bg-accent-primary',
              '[&::-webkit-slider-thumb]:cursor-pointer',
              '[&::-webkit-slider-thumb]:transition-all',
              '[&::-webkit-slider-thumb]:hover:bg-accent-secondary'
            )}
          />
        </div>

        {/* Sort */}
        <div className="w-full lg:w-56">
          <Select
            options={sortOptions}
            value={sortBy}
            onChange={(value) => setSortBy(value as SortOption)}
            placeholder="Sort by..."
          />
        </div>
      </div>

      {/* Results count */}
      <div className="flex items-center justify-between">
        <p className="text-body-sm text-text-muted">
          Showing {filteredAndSorted.length} of {mappings.length} mappings
        </p>
        {(search || minConfidence > 0) && (
          <button
            onClick={() => {
              setSearch('')
              setMinConfidence(0)
            }}
            className="text-body-sm text-accent-secondary hover:text-accent-primary transition-colors"
          >
            Clear filters
          </button>
        )}
      </div>

      {/* Mappings grid */}
      {filteredAndSorted.length > 0 ? (
        <div className="grid gap-4">
          {filteredAndSorted.map((mapping) => {
            const id = getMappingId(mapping)
            return (
              <ConceptMappingCard
                key={id}
                mapping={mapping}
                expanded={expandedId === id}
                onToggle={() => setExpandedId(expandedId === id ? null : id)}
              />
            )
          })}
        </div>
      ) : (
        <div className="py-12 text-center">
          <p className="text-body text-text-muted">
            No mappings match your filters
          </p>
        </div>
      )}
    </div>
  )
}

export default MappingsList
