import { cn } from '@/lib/cn'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl'
  variant?: 'spinner' | 'dots' | 'pulse'
  label?: string
  className?: string
}

const sizeStyles = {
  sm: 'h-4 w-4',
  md: 'h-6 w-6',
  lg: 'h-8 w-8',
  xl: 'h-12 w-12',
}

const borderStyles = {
  sm: 'border-2',
  md: 'border-2',
  lg: 'border-3',
  xl: 'border-4',
}

export function LoadingSpinner({
  size = 'md',
  variant = 'spinner',
  label,
  className,
}: LoadingSpinnerProps) {
  if (variant === 'dots') {
    return (
      <div className={cn('flex items-center justify-center gap-1', className)}>
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className={cn(
              'rounded-full bg-accent-primary',
              size === 'sm' && 'w-1.5 h-1.5',
              size === 'md' && 'w-2 h-2',
              size === 'lg' && 'w-2.5 h-2.5',
              size === 'xl' && 'w-3 h-3'
            )}
            style={{
              animation: 'bounce 1.4s infinite ease-in-out both',
              animationDelay: `${i * 0.16}s`,
            }}
          />
        ))}
        {label && (
          <span className="ml-2 text-body-sm text-text-secondary">{label}</span>
        )}
        <style>{`
          @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
          }
        `}</style>
      </div>
    )
  }

  if (variant === 'pulse') {
    return (
      <div className={cn('flex items-center justify-center', className)}>
        <div
          className={cn(
            'rounded-full bg-accent-primary/30 animate-ping',
            sizeStyles[size]
          )}
        />
        <div
          className={cn(
            'absolute rounded-full bg-accent-primary',
            size === 'sm' && 'w-2 h-2',
            size === 'md' && 'w-3 h-3',
            size === 'lg' && 'w-4 h-4',
            size === 'xl' && 'w-6 h-6'
          )}
        />
        {label && (
          <span className="ml-3 text-body-sm text-text-secondary">{label}</span>
        )}
      </div>
    )
  }

  // Default spinner variant
  return (
    <div className={cn('flex items-center justify-center', className)}>
      <div
        className={cn(
          'rounded-full border-accent-primary border-t-transparent animate-spin',
          sizeStyles[size],
          borderStyles[size]
        )}
      />
      {label && (
        <span className="ml-3 text-body-sm text-text-secondary">{label}</span>
      )}
    </div>
  )
}

// Full page loading overlay
interface LoadingOverlayProps {
  label?: string
  className?: string
}

export function LoadingOverlay({ label = 'Loading...', className }: LoadingOverlayProps) {
  return (
    <div
      className={cn(
        'fixed inset-0 z-50 flex items-center justify-center',
        'bg-bg-primary/80 backdrop-blur-sm',
        className
      )}
    >
      <div className="flex flex-col items-center gap-4">
        <LoadingSpinner size="xl" />
        <p className="text-body text-text-secondary">{label}</p>
      </div>
    </div>
  )
}

export default LoadingSpinner
