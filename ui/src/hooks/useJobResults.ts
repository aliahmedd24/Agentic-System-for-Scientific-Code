import { useQuery } from '@tanstack/react-query'
import { getJobResult } from '@/api/endpoints'
import type { JobResult } from '@/api/types'

interface UseJobResultsOptions {
  enabled?: boolean
  staleTime?: number
  refetchOnWindowFocus?: boolean
}

/**
 * React Query hook for fetching job results
 * Results are cached for 5 minutes since they don't change after completion
 */
export function useJobResults(
  jobId: string | undefined,
  options: UseJobResultsOptions = {}
) {
  const {
    enabled = true,
    staleTime = 5 * 60 * 1000, // 5 minutes - results don't change
    refetchOnWindowFocus = false,
  } = options

  return useQuery<JobResult, Error>({
    queryKey: ['jobResults', jobId],
    queryFn: () => getJobResult(jobId!),
    enabled: enabled && !!jobId,
    staleTime,
    refetchOnWindowFocus,
  })
}

export default useJobResults
