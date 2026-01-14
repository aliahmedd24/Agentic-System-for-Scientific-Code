import { useEffect, useCallback } from 'react'
import { useWebSocketStore } from '@/stores/websocketStore'
import { useJobsStore } from '@/stores/jobsStore'

interface UseJobWebSocketOptions {
  autoRefreshOnComplete?: boolean
  onComplete?: () => void
  onError?: (error: string) => void
}

/**
 * Custom hook for managing WebSocket connection to a job
 * Handles connection lifecycle, event tracking, and automatic job refresh
 */
export function useJobWebSocket(
  jobId: string | null | undefined,
  options: UseJobWebSocketOptions = {}
) {
  const { autoRefreshOnComplete = true, onComplete, onError } = options

  const {
    connect,
    disconnect,
    connectionStatus,
    events,
    currentStage,
    currentProgress,
    lastMessage,
  } = useWebSocketStore()

  const { fetchJob } = useJobsStore()

  // Connect when jobId changes
  useEffect(() => {
    if (jobId) {
      connect(jobId)
    }

    return () => {
      disconnect()
    }
  }, [jobId, connect, disconnect])

  // Handle completion and error events
  useEffect(() => {
    if (events.length === 0) return

    const lastEvent = events[events.length - 1]
    const stage = lastEvent.stage?.toLowerCase()

    if (stage === 'completed') {
      if (autoRefreshOnComplete && jobId) {
        fetchJob(jobId)
      }
      onComplete?.()
    } else if (stage === 'failed' || stage === 'error') {
      if (autoRefreshOnComplete && jobId) {
        fetchJob(jobId)
      }
      onError?.(lastEvent.message || 'Job failed')
    }
  }, [events, jobId, autoRefreshOnComplete, fetchJob, onComplete, onError])

  // Manual refresh function
  const refresh = useCallback(() => {
    if (jobId) {
      fetchJob(jobId)
    }
  }, [jobId, fetchJob])

  // Reconnect function
  const reconnect = useCallback(() => {
    if (jobId) {
      disconnect()
      setTimeout(() => connect(jobId), 100)
    }
  }, [jobId, connect, disconnect])

  return {
    // Connection state
    isConnected: connectionStatus === 'connected',
    isConnecting: connectionStatus === 'connecting',
    hasError: connectionStatus === 'error',
    connectionStatus,

    // Event data
    events,
    currentStage,
    currentProgress,
    lastMessage,

    // Derived state
    eventCount: events.length,
    latestEvent: events.length > 0 ? events[events.length - 1] : null,

    // Actions
    disconnect,
    reconnect,
    refresh,
  }
}

export default useJobWebSocket
