import {
  MagnifyingGlassIcon,
  XMarkIcon,
  FunnelIcon,
} from '@heroicons/react/24/outline'
import { cn } from '@/lib/cn'
import { Input } from '@/components/ui/Input'
import { Select } from '@/components/ui/Select'
import { Button } from '@/components/ui/Button'
import type { JobStatusType } from '@/api/types'

export interface JobFilterState {
  status: JobStatusType[]
  search: string
  sortBy: 'created' | 'updated' | 'status' | 'progress'
  sortOrder: 'asc' | 'desc'
}

interface JobFiltersProps {
  filters: JobFilterState
  onChange: (filters: JobFilterState) => void
  totalCount?: number
  filteredCount?: number
  className?: string
}

const statusOptions: { id: JobStatusType; label: string; color: string }[] = [
  { id: 'pending', label: 'Pending', color: 'bg-status-warning' },
  { id: 'running', label: 'Running', color: 'bg-accent-primary' },
  { id: 'completed', label: 'Completed', color: 'bg-status-success' },
  { id: 'failed', label: 'Failed', color: 'bg-status-error' },
  { id: 'cancelled', label: 'Cancelled', color: 'bg-text-muted' },
]

const sortOptions = [
  { id: 'created-desc', label: 'Newest First' },
  { id: 'created-asc', label: 'Oldest First' },
  { id: 'updated-desc', label: 'Recently Updated' },
  { id: 'status-asc', label: 'Status (A-Z)' },
  { id: 'progress-desc', label: 'Progress (High-Low)' },
]

export function JobFilters({
  filters,
  onChange,
  totalCount,
  filteredCount,
  className,
}: JobFiltersProps) {
  const hasActiveFilters = filters.status.length > 0 || filters.search

  const toggleStatus = (status: JobStatusType) => {
    const newStatus = filters.status.includes(status)
      ? filters.status.filter((s) => s !== status)
      : [...filters.status, status]
    onChange({ ...filters, status: newStatus })
  }

  const handleSortChange = (value: string) => {
    const [sortBy, sortOrder] = value.split('-') as [
      JobFilterState['sortBy'],
      JobFilterState['sortOrder']
    ]
    onChange({ ...filters, sortBy, sortOrder })
  }

  const clearFilters = () => {
    onChange({
      status: [],
      search: '',
      sortBy: 'created',
      sortOrder: 'desc',
    })
  }

  return (
    <div className={cn('space-y-4', className)}>
      {/* Search and Sort Row */}
      <div className="flex flex-col sm:flex-row gap-4">
        {/* Search */}
        <div className="flex-1">
          <Input
            placeholder="Search by job ID or paper source..."
            value={filters.search}
            onChange={(e) => onChange({ ...filters, search: e.target.value })}
            leftIcon={<MagnifyingGlassIcon className="h-4 w-4" />}
          />
        </div>

        {/* Sort */}
        <div className="w-full sm:w-48">
          <Select
            options={sortOptions}
            value={`${filters.sortBy}-${filters.sortOrder}`}
            onChange={handleSortChange}
            placeholder="Sort by..."
          />
        </div>
      </div>

      {/* Status Filter Chips */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="flex items-center gap-1.5 text-body-sm text-text-muted">
          <FunnelIcon className="h-4 w-4" />
          Status:
        </span>
        {statusOptions.map((option) => {
          const isActive = filters.status.includes(option.id)
          return (
            <button
              key={option.id}
              onClick={() => toggleStatus(option.id)}
              className={cn(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-full text-body-sm font-medium',
                'transition-all duration-200',
                isActive
                  ? 'bg-accent-primary text-white'
                  : 'bg-bg-tertiary text-text-muted hover:text-text-secondary hover:bg-bg-secondary'
              )}
            >
              <span
                className={cn(
                  'w-2 h-2 rounded-full',
                  isActive ? 'bg-white' : option.color
                )}
              />
              {option.label}
            </button>
          )
        })}
      </div>

      {/* Results count and Clear */}
      <div className="flex items-center justify-between">
        <p className="text-body-sm text-text-muted">
          {filteredCount !== undefined && totalCount !== undefined ? (
            filteredCount === totalCount ? (
              `${totalCount} job${totalCount !== 1 ? 's' : ''}`
            ) : (
              `Showing ${filteredCount} of ${totalCount} jobs`
            )
          ) : null}
        </p>

        {hasActiveFilters && (
          <Button
            variant="ghost"
            size="sm"
            onClick={clearFilters}
            leftIcon={<XMarkIcon className="h-4 w-4" />}
          >
            Clear filters
          </Button>
        )}
      </div>
    </div>
  )
}

export default JobFilters
