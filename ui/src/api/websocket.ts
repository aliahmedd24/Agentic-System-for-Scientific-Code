import type { WebSocketMessage } from './types'

type WebSocketEventHandler = (message: WebSocketMessage) => void
type ConnectionStatusHandler = (status: 'connecting' | 'connected' | 'disconnected' | 'error') => void

interface WebSocketManagerConfig {
  onMessage?: WebSocketEventHandler
  onStatusChange?: ConnectionStatusHandler
  reconnectAttempts?: number
  reconnectDelay?: number
  heartbeatInterval?: number
}

/**
 * WebSocket connection manager for real-time job updates
 */
export class WebSocketManager {
  private socket: WebSocket | null = null
  private jobId: string | null = null
  private reconnectAttempts: number
  private reconnectDelay: number
  private heartbeatInterval: number
  private currentAttempt = 0
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private onMessage?: WebSocketEventHandler
  private onStatusChange?: ConnectionStatusHandler

  constructor(config: WebSocketManagerConfig = {}) {
    this.onMessage = config.onMessage
    this.onStatusChange = config.onStatusChange
    this.reconnectAttempts = config.reconnectAttempts ?? 5
    this.reconnectDelay = config.reconnectDelay ?? 3000
    this.heartbeatInterval = config.heartbeatInterval ?? 30000
  }

  /**
   * Connect to WebSocket for a specific job
   */
  connect(jobId: string): void {
    // Disconnect existing connection if any
    if (this.socket) {
      this.disconnect()
    }

    this.jobId = jobId
    this.currentAttempt = 0
    this.createConnection()
  }

  /**
   * Disconnect from WebSocket
   */
  disconnect(): void {
    this.clearTimers()

    if (this.socket) {
      this.socket.onclose = null // Prevent reconnect on intentional close
      this.socket.close()
      this.socket = null
    }

    this.jobId = null
    this.onStatusChange?.('disconnected')
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN
  }

  /**
   * Update event handlers
   */
  setHandlers(config: Pick<WebSocketManagerConfig, 'onMessage' | 'onStatusChange'>): void {
    if (config.onMessage) this.onMessage = config.onMessage
    if (config.onStatusChange) this.onStatusChange = config.onStatusChange
  }

  private createConnection(): void {
    if (!this.jobId) return

    this.onStatusChange?.('connecting')

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/${this.jobId}`

    try {
      this.socket = new WebSocket(wsUrl)
      this.setupEventListeners()
    } catch (error) {
      console.error('WebSocket connection error:', error)
      this.onStatusChange?.('error')
      this.scheduleReconnect()
    }
  }

  private setupEventListeners(): void {
    if (!this.socket) return

    this.socket.onopen = () => {
      console.log(`WebSocket connected for job: ${this.jobId}`)
      this.currentAttempt = 0
      this.onStatusChange?.('connected')
      this.startHeartbeat()
    }

    this.socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as WebSocketMessage
        this.onMessage?.(message)

        // Auto-disconnect on completion or error
        if (message.type === 'completed' || message.type === 'error' || message.type === 'cancelled') {
          // Keep connection for a bit to ensure all messages are received
          setTimeout(() => {
            if (this.socket?.readyState === WebSocket.OPEN) {
              this.disconnect()
            }
          }, 1000)
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error)
      }
    }

    this.socket.onclose = (event) => {
      console.log(`WebSocket closed: ${event.code} - ${event.reason}`)
      this.clearTimers()
      this.onStatusChange?.('disconnected')

      // Only reconnect if it wasn't a clean close and we have a job ID
      if (!event.wasClean && this.jobId) {
        this.scheduleReconnect()
      }
    }

    this.socket.onerror = (error) => {
      console.error('WebSocket error:', error)
      this.onStatusChange?.('error')
    }
  }

  private startHeartbeat(): void {
    this.clearHeartbeat()
    this.heartbeatTimer = setInterval(() => {
      if (this.socket?.readyState === WebSocket.OPEN) {
        this.socket.send('ping')
      }
    }, this.heartbeatInterval)
  }

  private clearHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = null
    }
  }

  private clearTimers(): void {
    this.clearHeartbeat()
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
  }

  private scheduleReconnect(): void {
    if (this.currentAttempt >= this.reconnectAttempts) {
      console.log('Max reconnect attempts reached')
      this.onStatusChange?.('error')
      return
    }

    this.currentAttempt++
    const delay = this.reconnectDelay * Math.pow(2, this.currentAttempt - 1) // Exponential backoff

    console.log(`Scheduling reconnect attempt ${this.currentAttempt}/${this.reconnectAttempts} in ${delay}ms`)

    this.reconnectTimer = setTimeout(() => {
      this.createConnection()
    }, delay)
  }
}

// Singleton instance for global use
export const wsManager = new WebSocketManager()

export default wsManager
