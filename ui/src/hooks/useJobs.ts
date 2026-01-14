import { useQuery } from '@tanstack/react-query'
import { getJobs, getJobStatus } from '@/api/endpoints'
import type { Job, JobStatusResponse, JobStatusType } from '@/api/types'
import { useMemo } from 'react'

interface UseJobsOptions {
  limit?: number
  enabled?: boolean
  staleTime?: number
  refetchInterval?: number | false
}

/**
 * React Query hook for fetching job list
 */
export function useJobs(options: UseJobsOptions = {}) {
  const {
    limit = 50,
    enabled = true,
    staleTime = 30 * 1000, // 30 seconds
    refetchInterval = false,
  } = options

  return useQuery<Job[], Error>({
    queryKey: ['jobs', { limit }],
    queryFn: () => getJobs(limit),
    enabled,
    staleTime,
    refetchInterval,
  })
}

interface UseJobStatusOptions {
  enabled?: boolean
  refetchInterval?: number | false
}

/**
 * React Query hook for fetching single job status with events
 */
export function useJobStatus(
  jobId: string | undefined,
  options: UseJobStatusOptions = {}
) {
  const { enabled = true, refetchInterval = false } = options

  return useQuery<JobStatusResponse, Error>({
    queryKey: ['jobStatus', jobId],
    queryFn: () => getJobStatus(jobId!),
    enabled: enabled && !!jobId,
    staleTime: 10 * 1000, // 10 seconds
    refetchInterval,
  })
}

interface JobFilterState {
  status: JobStatusType[]
  search: string
  sortBy: 'created' | 'updated' | 'status' | 'progress'
  sortOrder: 'asc' | 'desc'
}

/**
 * Hook for filtering and sorting jobs client-side
 */
export function useFilteredJobs(
  jobs: Job[] | undefined,
  filters: JobFilterState
) {
  return useMemo(() => {
    if (!jobs) return []

    let filtered = [...jobs]

    // Filter by status
    if (filters.status.length > 0) {
      filtered = filtered.filter((job) => filters.status.includes(job.status))
    }

    // Filter by search term (job ID or paper source)
    if (filters.search) {
      const searchLower = filters.search.toLowerCase()
      filtered = filtered.filter(
        (job) =>
          job.job_id.toLowerCase().includes(searchLower) ||
          job.paper_source?.toLowerCase().includes(searchLower) ||
          job.repo_url?.toLowerCase().includes(searchLower)
      )
    }

    // Sort
    filtered.sort((a, b) => {
      let comparison = 0

      switch (filters.sortBy) {
        case 'created':
          comparison = new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
          break
        case 'updated':
          comparison = new Date(a.updated_at).getTime() - new Date(b.updated_at).getTime()
          break
        case 'status':
          comparison = a.status.localeCompare(b.status)
          break
        case 'progress':
          comparison = a.progress - b.progress
          break
      }

      return filters.sortOrder === 'desc' ? -comparison : comparison
    })

    return filtered
  }, [jobs, filters])
}

export default useJobs
