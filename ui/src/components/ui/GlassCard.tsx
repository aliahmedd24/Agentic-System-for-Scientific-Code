import { forwardRef } from 'react'
import { cn } from '@/lib/cn'

type CardVariant = 'glass' | 'solid' | 'outline' | 'interactive'

interface GlassCardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: CardVariant
  title?: string
  subtitle?: string
  icon?: React.ReactNode
  headerActions?: React.ReactNode
  noPadding?: boolean
}

const variantStyles: Record<CardVariant, string> = {
  glass: 'bg-bg-glass backdrop-blur-[20px] border border-border hover:border-border-glow hover:shadow-glow',
  solid: 'bg-bg-tertiary border border-border',
  outline: 'bg-transparent border border-border hover:border-border-glow',
  interactive: 'bg-bg-glass backdrop-blur-[20px] border border-border cursor-pointer hover:border-border-glow hover:shadow-glow active:scale-[0.99]',
}

export const GlassCard = forwardRef<HTMLDivElement, GlassCardProps>(
  (
    {
      variant = 'glass',
      title,
      subtitle,
      icon,
      headerActions,
      noPadding = false,
      className,
      children,
      ...props
    },
    ref
  ) => {
    const hasHeader = title || subtitle || icon || headerActions

    return (
      <div
        ref={ref}
        className={cn(
          'rounded-2xl transition-all duration-300',
          variantStyles[variant],
          !noPadding && 'p-6',
          className
        )}
        {...props}
      >
        {hasHeader && (
          <div className={cn('flex items-start justify-between gap-4', children && 'mb-4')}>
            <div className="flex items-start gap-3 min-w-0">
              {icon && (
                <div className="flex-shrink-0 flex items-center justify-center w-10 h-10 rounded-lg bg-accent-primary/20 text-accent-primary">
                  {icon}
                </div>
              )}
              <div className="min-w-0">
                {title && (
                  <h3 className="text-heading-3 text-text-primary truncate">{title}</h3>
                )}
                {subtitle && (
                  <p className="mt-0.5 text-body-sm text-text-secondary">{subtitle}</p>
                )}
              </div>
            </div>
            {headerActions && (
              <div className="flex-shrink-0 flex items-center gap-2">
                {headerActions}
              </div>
            )}
          </div>
        )}
        {children}
      </div>
    )
  }
)

GlassCard.displayName = 'GlassCard'

export default GlassCard
