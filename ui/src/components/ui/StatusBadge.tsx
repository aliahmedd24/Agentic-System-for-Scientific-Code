import { cn } from '@/lib/cn'
import { JOB_STATUSES, type JobStatus } from '@/lib/constants'

interface StatusBadgeProps {
  status: JobStatus
  size?: 'sm' | 'md' | 'lg'
  showDot?: boolean
  className?: string
}

const sizeStyles = {
  sm: 'px-2 py-0.5 text-caption',
  md: 'px-2.5 py-1 text-body-sm',
  lg: 'px-3 py-1.5 text-body',
}

export function StatusBadge({
  status,
  size = 'md',
  showDot = true,
  className,
}: StatusBadgeProps) {
  const config = JOB_STATUSES[status] || JOB_STATUSES.pending

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 font-medium rounded-full',
        config.bgColor,
        config.color,
        sizeStyles[size],
        className
      )}
    >
      {showDot && (
        <span
          className={cn(
            'w-1.5 h-1.5 rounded-full',
            status === 'running' && 'animate-pulse',
            config.color.replace('text-', 'bg-')
          )}
        />
      )}
      {config.label}
    </span>
  )
}

export default StatusBadge
