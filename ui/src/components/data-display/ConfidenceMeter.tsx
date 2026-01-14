import { cn } from '@/lib/cn'

interface ConfidenceMeterProps {
  value: number // 0-1
  showLabel?: boolean
  showValue?: boolean
  size?: 'sm' | 'md' | 'lg'
  variant?: 'bar' | 'circle'
  className?: string
}

const sizeStyles = {
  bar: {
    sm: 'h-1.5',
    md: 'h-2',
    lg: 'h-3',
  },
  circle: {
    sm: 'w-8 h-8',
    md: 'w-12 h-12',
    lg: 'w-16 h-16',
  },
}

const getColor = (value: number) => {
  if (value >= 0.8) return 'text-status-success'
  if (value >= 0.6) return 'text-status-warning'
  if (value >= 0.4) return 'text-orange-500'
  return 'text-status-error'
}

const getBgColor = (value: number) => {
  if (value >= 0.8) return 'bg-status-success'
  if (value >= 0.6) return 'bg-status-warning'
  if (value >= 0.4) return 'bg-orange-500'
  return 'bg-status-error'
}

export function ConfidenceMeter({
  value,
  showLabel = false,
  showValue = true,
  size = 'md',
  variant = 'bar',
  className,
}: ConfidenceMeterProps) {
  const clampedValue = Math.min(1, Math.max(0, value))
  const percentage = Math.round(clampedValue * 100)

  if (variant === 'circle') {
    const strokeWidth = size === 'sm' ? 3 : size === 'md' ? 4 : 5
    const radius = size === 'sm' ? 12 : size === 'md' ? 20 : 28
    const circumference = 2 * Math.PI * radius
    const strokeDashoffset = circumference * (1 - clampedValue)

    return (
      <div className={cn('flex items-center gap-2', className)}>
        {showLabel && (
          <span className="text-body-sm text-text-secondary">Confidence</span>
        )}
        <div className={cn('relative', sizeStyles.circle[size])}>
          <svg className="w-full h-full -rotate-90">
            {/* Background circle */}
            <circle
              cx="50%"
              cy="50%"
              r={radius}
              fill="none"
              stroke="currentColor"
              strokeWidth={strokeWidth}
              className="text-bg-tertiary"
            />
            {/* Progress circle */}
            <circle
              cx="50%"
              cy="50%"
              r={radius}
              fill="none"
              stroke="currentColor"
              strokeWidth={strokeWidth}
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              className={cn('transition-all duration-500', getColor(clampedValue))}
            />
          </svg>
          {showValue && (
            <span
              className={cn(
                'absolute inset-0 flex items-center justify-center font-medium',
                size === 'sm' ? 'text-caption' : 'text-body-sm',
                getColor(clampedValue)
              )}
            >
              {percentage}
            </span>
          )}
        </div>
      </div>
    )
  }

  // Bar variant
  return (
    <div className={cn('flex items-center gap-2', className)}>
      {showLabel && (
        <span className="text-body-sm text-text-secondary flex-shrink-0">
          Confidence
        </span>
      )}
      <div className="flex-1 flex items-center gap-2 min-w-0">
        <div
          className={cn(
            'flex-1 bg-bg-tertiary rounded-full overflow-hidden',
            sizeStyles.bar[size]
          )}
        >
          <div
            className={cn(
              'h-full rounded-full transition-all duration-300',
              getBgColor(clampedValue)
            )}
            style={{ width: `${percentage}%` }}
          />
        </div>
        {showValue && (
          <span
            className={cn(
              'text-body-sm font-medium tabular-nums flex-shrink-0 min-w-[3ch]',
              getColor(clampedValue)
            )}
          >
            {percentage}%
          </span>
        )}
      </div>
    </div>
  )
}

export default ConfidenceMeter
