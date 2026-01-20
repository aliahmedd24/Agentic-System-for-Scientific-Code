import { forwardRef } from 'react'
import { cn } from '@/lib/cn'
import { GlassCard } from '@/components/ui/GlassCard'
import {
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  MinusIcon,
} from '@heroicons/react/24/outline'

interface KPICardProps {
  title: string
  value: string | number
  icon: React.ReactNode
  iconBgColor?: string
  iconColor?: string
  trend?: {
    value: number
    direction: 'up' | 'down' | 'neutral'
    label?: string
  }
  loading?: boolean
  subtitle?: string
  className?: string
}

export const KPICard = forwardRef<HTMLDivElement, KPICardProps>(
  (
    {
      title,
      value,
      icon,
      iconBgColor = 'bg-accent-primary/20',
      iconColor = 'text-accent-primary',
      trend,
      loading = false,
      subtitle,
      className,
    },
    ref
  ) => {
    const trendConfig = {
      up: {
        icon: ArrowTrendingUpIcon,
        color: 'text-status-success',
        bgColor: 'bg-status-success/10',
      },
      down: {
        icon: ArrowTrendingDownIcon,
        color: 'text-status-error',
        bgColor: 'bg-status-error/10',
      },
      neutral: {
        icon: MinusIcon,
        color: 'text-text-muted',
        bgColor: 'bg-bg-tertiary',
      },
    }

    const TrendIcon = trend ? trendConfig[trend.direction].icon : null

    return (
      <GlassCard ref={ref} className={className}>
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-4">
            {/* Icon */}
            <div
              className={cn(
                'flex items-center justify-center w-12 h-12 rounded-xl',
                iconBgColor
              )}
            >
              <span className={iconColor}>{icon}</span>
            </div>

            {/* Content */}
            <div>
              <p className="text-caption text-text-muted">{title}</p>
              {loading ? (
                <div className="h-8 w-20 mt-1 bg-bg-tertiary rounded animate-pulse" />
              ) : (
                <p className="text-heading-2 text-text-primary">{value}</p>
              )}
              {subtitle && (
                <p className="text-caption text-text-muted mt-0.5">{subtitle}</p>
              )}
            </div>
          </div>

          {/* Trend Indicator */}
          {trend && !loading && (
            <div
              className={cn(
                'flex items-center gap-1 px-2 py-1 rounded-full',
                trendConfig[trend.direction].bgColor
              )}
            >
              {TrendIcon && (
                <TrendIcon
                  className={cn('h-3.5 w-3.5', trendConfig[trend.direction].color)}
                />
              )}
              <span
                className={cn(
                  'text-caption font-medium',
                  trendConfig[trend.direction].color
                )}
              >
                {trend.value > 0 ? '+' : ''}
                {trend.value}%
              </span>
            </div>
          )}
        </div>

        {/* Optional trend label */}
        {trend?.label && !loading && (
          <p className="text-caption text-text-muted mt-3">{trend.label}</p>
        )}
      </GlassCard>
    )
  }
)

KPICard.displayName = 'KPICard'

export default KPICard
