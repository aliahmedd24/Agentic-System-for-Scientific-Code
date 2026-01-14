import { forwardRef } from 'react'
import { cn } from '@/lib/cn'

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'icon'
type ButtonSize = 'sm' | 'md' | 'lg'

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?: ButtonSize
  loading?: boolean
  leftIcon?: React.ReactNode
  rightIcon?: React.ReactNode
}

const variantStyles: Record<ButtonVariant, string> = {
  primary: cn(
    'bg-gradient-to-r from-accent-primary to-purple-500',
    'text-white font-semibold',
    'hover:from-accent-secondary hover:to-purple-400',
    'focus:ring-2 focus:ring-accent-primary focus:ring-offset-2 focus:ring-offset-bg-primary'
  ),
  secondary: cn(
    'bg-bg-tertiary border border-border',
    'text-text-primary font-medium',
    'hover:border-accent-primary hover:text-accent-secondary',
    'focus:ring-2 focus:ring-accent-primary focus:ring-offset-2 focus:ring-offset-bg-primary'
  ),
  ghost: cn(
    'text-text-secondary font-medium',
    'hover:bg-bg-tertiary hover:text-text-primary',
    'focus:ring-2 focus:ring-accent-primary'
  ),
  danger: cn(
    'bg-status-error/20 border border-status-error/50',
    'text-status-error font-medium',
    'hover:bg-status-error hover:text-white',
    'focus:ring-2 focus:ring-status-error focus:ring-offset-2 focus:ring-offset-bg-primary'
  ),
  icon: cn(
    'text-text-secondary',
    'hover:bg-bg-tertiary hover:text-text-primary',
    'focus:ring-2 focus:ring-accent-primary'
  ),
}

const sizeStyles: Record<ButtonSize, string> = {
  sm: 'px-3 py-1.5 text-body-sm gap-1.5',
  md: 'px-4 py-2.5 text-body gap-2',
  lg: 'px-6 py-3 text-body-lg gap-2',
}

const iconSizeStyles: Record<ButtonSize, string> = {
  sm: 'p-1.5',
  md: 'p-2',
  lg: 'p-2.5',
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      variant = 'primary',
      size = 'md',
      loading = false,
      leftIcon,
      rightIcon,
      disabled,
      className,
      children,
      ...props
    },
    ref
  ) => {
    const isDisabled = disabled || loading
    const isIconButton = variant === 'icon'

    return (
      <button
        ref={ref}
        disabled={isDisabled}
        className={cn(
          'inline-flex items-center justify-center rounded-lg',
          'transition-all duration-200',
          'focus:outline-none',
          'disabled:opacity-50 disabled:cursor-not-allowed',
          variantStyles[variant],
          isIconButton ? iconSizeStyles[size] : sizeStyles[size],
          className
        )}
        {...props}
      >
        {loading ? (
          <LoadingSpinner size={size} />
        ) : (
          <>
            {leftIcon && <span className="flex-shrink-0">{leftIcon}</span>}
            {children}
            {rightIcon && <span className="flex-shrink-0">{rightIcon}</span>}
          </>
        )}
      </button>
    )
  }
)

Button.displayName = 'Button'

function LoadingSpinner({ size }: { size: ButtonSize }) {
  const sizeClass = size === 'sm' ? 'h-4 w-4' : size === 'lg' ? 'h-6 w-6' : 'h-5 w-5'

  return (
    <svg
      className={cn('animate-spin', sizeClass)}
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  )
}

export default Button
