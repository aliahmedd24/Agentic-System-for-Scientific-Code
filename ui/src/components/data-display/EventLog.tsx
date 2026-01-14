import { useRef, useEffect } from 'react'
import { cn } from '@/lib/cn'
import type { JobEvent } from '@/api/types'

interface EventLogProps {
  events: JobEvent[]
  maxHeight?: number | string
  autoScroll?: boolean
  showTimestamp?: boolean
  showStage?: boolean
  className?: string
}

const stageColors: Record<string, string> = {
  initialized: 'text-text-muted',
  parsing_paper: 'text-blue-400',
  analyzing_repo: 'text-cyan-400',
  mapping_concepts: 'text-purple-400',
  generating_code: 'text-orange-400',
  setting_up_env: 'text-yellow-400',
  executing_code: 'text-green-400',
  generating_report: 'text-pink-400',
  completed: 'text-status-success',
  failed: 'text-status-error',
  error: 'text-status-error',
}

export function EventLog({
  events,
  maxHeight = 400,
  autoScroll = true,
  showTimestamp = true,
  showStage = true,
  className,
}: EventLogProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom when new events arrive
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [events, autoScroll])

  const formatTimestamp = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      })
    } catch {
      return '--:--:--'
    }
  }

  const formatStage = (stage: string) => {
    return stage.replace(/_/g, ' ').toUpperCase()
  }

  if (events.length === 0) {
    return (
      <div
        className={cn(
          'flex items-center justify-center p-8',
          'bg-bg-tertiary/30 rounded-lg border border-border',
          className
        )}
        style={{ maxHeight }}
      >
        <div className="text-center">
          <div className="w-8 h-8 mx-auto mb-3 rounded-full border-2 border-accent-primary border-t-transparent animate-spin" />
          <p className="text-body-sm text-text-muted">Waiting for events...</p>
        </div>
      </div>
    )
  }

  return (
    <div
      ref={scrollRef}
      className={cn(
        'overflow-y-auto font-mono text-body-sm',
        'bg-bg-tertiary/30 rounded-lg border border-border p-4',
        className
      )}
      style={{ maxHeight }}
    >
      <div className="space-y-1">
        {events.map((event, idx) => (
          <div
            key={idx}
            className={cn(
              'flex items-start gap-3 py-1.5 px-2 rounded',
              'hover:bg-bg-tertiary/50 transition-colors',
              idx === events.length - 1 && 'bg-accent-primary/10'
            )}
          >
            {showTimestamp && (
              <span className="text-text-muted flex-shrink-0 tabular-nums">
                {formatTimestamp(event.timestamp)}
              </span>
            )}
            {showStage && (
              <span
                className={cn(
                  'flex-shrink-0 text-caption font-semibold min-w-[100px]',
                  stageColors[event.stage.toLowerCase()] || 'text-text-secondary'
                )}
              >
                [{formatStage(event.stage)}]
              </span>
            )}
            <span className="text-text-primary flex-1 break-words">
              {event.message}
            </span>
            {event.progress !== undefined && (
              <span className="text-text-muted flex-shrink-0">
                {Math.round(event.progress)}%
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

export default EventLog
