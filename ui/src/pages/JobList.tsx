import { useState } from 'react'
import { Link } from 'react-router-dom'
import { PageHeader } from '@/components/layout/PageHeader'
import { Button } from '@/components/ui/Button'
import { LoadingSpinner } from '@/components/data-display/LoadingSpinner'
import { EmptyState } from '@/components/data-display/EmptyState'
import { JobCard } from '@/components/jobs/JobCard'
import { JobFilters, type JobFilterState } from '@/components/jobs/JobFilters'
import { useJobs, useFilteredJobs } from '@/hooks/useJobs'
import { useJobMutations } from '@/hooks/useJobMutations'
import {
  PlusCircleIcon,
  Squares2X2Icon,
  ListBulletIcon,
} from '@heroicons/react/24/outline'

type ViewMode = 'grid' | 'list'

const defaultFilters: JobFilterState = {
  status: [],
  search: '',
  sortBy: 'created',
  sortOrder: 'desc',
}

export default function JobList() {
  const [filters, setFilters] = useState<JobFilterState>(defaultFilters)
  const [viewMode, setViewMode] = useState<ViewMode>('grid')

  const { data: jobs, isLoading, error } = useJobs({ limit: 100 })
  const { cancelJob } = useJobMutations()
  const filteredJobs = useFilteredJobs(jobs, filters)

  const handleCancel = (jobId: string) => {
    if (confirm('Are you sure you want to cancel this job?')) {
      cancelJob(jobId)
    }
  }

  const handleDelete = (jobId: string) => {
    if (confirm('Are you sure you want to delete this job?')) {
      // TODO: Add delete mutation when backend supports it
      console.log('Delete job:', jobId)
    }
  }

  return (
    <div className="animate-in">
      <PageHeader
        title="Job History"
        subtitle="View and manage all analysis jobs"
        actions={
          <div className="flex items-center gap-3">
            {/* View toggle */}
            <div className="flex items-center rounded-lg bg-bg-tertiary p-1">
              <button
                onClick={() => setViewMode('grid')}
                className={`p-2 rounded-md transition-colors ${
                  viewMode === 'grid'
                    ? 'bg-accent-primary/20 text-accent-secondary'
                    : 'text-text-muted hover:text-text-secondary'
                }`}
                title="Grid view"
              >
                <Squares2X2Icon className="h-4 w-4" />
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`p-2 rounded-md transition-colors ${
                  viewMode === 'list'
                    ? 'bg-accent-primary/20 text-accent-secondary'
                    : 'text-text-muted hover:text-text-secondary'
                }`}
                title="List view"
              >
                <ListBulletIcon className="h-4 w-4" />
              </button>
            </div>

            <Link to="/analyze">
              <Button leftIcon={<PlusCircleIcon className="h-5 w-5" />}>
                New Analysis
              </Button>
            </Link>
          </div>
        }
      />

      {/* Filters */}
      <JobFilters
        filters={filters}
        onChange={setFilters}
        totalCount={jobs?.length}
        filteredCount={filteredJobs.length}
        className="mb-6"
      />

      {/* Content */}
      {isLoading ? (
        <div className="flex items-center justify-center py-20">
          <LoadingSpinner size="lg" />
        </div>
      ) : error ? (
        <EmptyState
          icon="error"
          title="Failed to load jobs"
          description={error.message}
          action={{
            label: 'Try Again',
            onClick: () => window.location.reload(),
          }}
        />
      ) : filteredJobs.length === 0 ? (
        jobs && jobs.length > 0 ? (
          // Has jobs but none match filters
          <EmptyState
            icon="search"
            title="No matching jobs"
            description="Try adjusting your filters to see more results"
            action={{
              label: 'Clear Filters',
              onClick: () => setFilters(defaultFilters),
            }}
          />
        ) : (
          // No jobs at all
          <EmptyState
            icon="document"
            title="No jobs yet"
            description="Start your first analysis to see jobs here"
            action={{
              label: 'New Analysis',
              onClick: () => window.location.href = '/analyze',
            }}
          />
        )
      ) : viewMode === 'grid' ? (
        // Grid view
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {filteredJobs.map((job) => (
            <JobCard
              key={job.job_id}
              job={job}
              onCancel={handleCancel}
              onDelete={handleDelete}
            />
          ))}
        </div>
      ) : (
        // List view - compact cards
        <div className="space-y-3">
          {filteredJobs.map((job) => (
            <JobCard
              key={job.job_id}
              job={job}
              onCancel={handleCancel}
              onDelete={handleDelete}
            />
          ))}
        </div>
      )}
    </div>
  )
}
