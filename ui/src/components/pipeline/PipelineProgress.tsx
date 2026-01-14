import { cn } from '@/lib/cn'

interface PipelineProgressProps {
  progress: number
  status?: 'running' | 'completed' | 'failed' | 'pending'
  showLabel?: boolean
  size?: 'sm' | 'md' | 'lg'
  animated?: boolean
  className?: string
}

const sizeStyles = {
  sm: 'h-1.5',
  md: 'h-2.5',
  lg: 'h-4',
}

export function PipelineProgress({
  progress,
  status = 'running',
  showLabel = true,
  size = 'md',
  animated = true,
  className,
}: PipelineProgressProps) {
  const clampedProgress = Math.min(100, Math.max(0, progress))

  const getBarColor = () => {
    switch (status) {
      case 'completed':
        return 'bg-status-success'
      case 'failed':
        return 'bg-status-error'
      case 'pending':
        return 'bg-text-muted'
      default:
        return 'bg-gradient-to-r from-accent-primary to-purple-500'
    }
  }

  return (
    <div className={cn('w-full', className)}>
      {showLabel && (
        <div className="flex items-center justify-between mb-2">
          <span className="text-body-sm text-text-secondary">Progress</span>
          <span className="text-body-sm font-medium text-text-primary">
            {Math.round(clampedProgress)}%
          </span>
        </div>
      )}
      <div
        className={cn(
          'w-full bg-bg-tertiary rounded-full overflow-hidden',
          sizeStyles[size]
        )}
      >
        <div
          className={cn(
            'h-full rounded-full transition-all duration-500 ease-out',
            getBarColor(),
            animated && status === 'running' && 'animate-pulse'
          )}
          style={{ width: `${clampedProgress}%` }}
        >
          {/* Shimmer effect for running state */}
          {animated && status === 'running' && clampedProgress > 0 && (
            <div className="h-full w-full relative overflow-hidden">
              <div
                className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
                style={{
                  animation: 'shimmer 2s infinite',
                }}
              />
            </div>
          )}
        </div>
      </div>

      <style>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  )
}

export default PipelineProgress
