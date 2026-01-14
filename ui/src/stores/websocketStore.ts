import { create } from 'zustand'
import type { WebSocketMessage, JobEvent } from '@/api/types'
import { WebSocketManager } from '@/api/websocket'

type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error'

interface WebSocketState {
  // Connection state
  connectionStatus: ConnectionStatus
  currentJobId: string | null

  // Events
  events: JobEvent[]
  currentStage: string
  currentProgress: number
  lastMessage: string

  // Actions
  connect: (jobId: string) => void
  disconnect: () => void
  clearEvents: () => void
}

// Create WebSocket manager instance
const wsManager = new WebSocketManager()

export const useWebSocketStore = create<WebSocketState>((set, get) => {
  // Setup handlers
  wsManager.setHandlers({
    onMessage: (message: WebSocketMessage) => {
      const event: JobEvent = {
        timestamp: message.timestamp || new Date().toISOString(),
        stage: message.stage || get().currentStage,
        progress: message.progress ?? get().currentProgress,
        message: message.message || '',
        data: message.data,
      }

      set((state) => ({
        events: [...state.events, event],
        currentStage: event.stage,
        currentProgress: event.progress,
        lastMessage: event.message,
      }))
    },
    onStatusChange: (status: ConnectionStatus) => {
      set({ connectionStatus: status })
    },
  })

  return {
    // Initial state
    connectionStatus: 'disconnected',
    currentJobId: null,
    events: [],
    currentStage: 'initialized',
    currentProgress: 0,
    lastMessage: '',

    // Connect to job
    connect: (jobId: string) => {
      set({
        currentJobId: jobId,
        events: [],
        currentStage: 'initialized',
        currentProgress: 0,
        lastMessage: '',
      })
      wsManager.connect(jobId)
    },

    // Disconnect
    disconnect: () => {
      wsManager.disconnect()
      set({
        currentJobId: null,
        connectionStatus: 'disconnected',
      })
    },

    // Clear events
    clearEvents: () => {
      set({
        events: [],
        currentStage: 'initialized',
        currentProgress: 0,
        lastMessage: '',
      })
    },
  }
})

export default useWebSocketStore
