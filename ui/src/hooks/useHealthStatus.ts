import { useQuery } from '@tanstack/react-query'
import { getHealth } from '@/api/endpoints'
import type { HealthStatus } from '@/api/types'

interface UseHealthStatusOptions {
  pollInterval?: number
  enabled?: boolean
}

interface UseHealthStatusReturn {
  data: HealthStatus | undefined
  isLoading: boolean
  isError: boolean
  error: Error | null
  refetch: () => void
}

export function useHealthStatus(options: UseHealthStatusOptions = {}): UseHealthStatusReturn {
  const { pollInterval = 30000, enabled = true } = options

  const query = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: pollInterval,
    staleTime: 10000, // Consider data stale after 10 seconds
    enabled,
    retry: 1,
  })

  return {
    data: query.data,
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
  }
}

export default useHealthStatus
