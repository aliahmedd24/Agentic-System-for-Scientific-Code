import { forwardRef, useId } from 'react'
import { cn } from '@/lib/cn'
import { ExclamationCircleIcon } from '@heroicons/react/24/outline'

interface InputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'size'> {
  label?: string
  error?: string
  hint?: string
  leftIcon?: React.ReactNode
  rightIcon?: React.ReactNode
  inputSize?: 'sm' | 'md' | 'lg'
}

const sizeStyles = {
  sm: 'px-3 py-2 text-body-sm',
  md: 'px-4 py-3 text-body',
  lg: 'px-4 py-4 text-body-lg',
}

const iconSizeStyles = {
  sm: 'h-4 w-4',
  md: 'h-5 w-5',
  lg: 'h-6 w-6',
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      label,
      error,
      hint,
      leftIcon,
      rightIcon,
      inputSize = 'md',
      className,
      id,
      ...props
    },
    ref
  ) => {
    const generatedId = useId()
    const inputId = id || generatedId

    return (
      <div className="w-full">
        {label && (
          <label
            htmlFor={inputId}
            className="block mb-2 text-body-sm font-medium text-text-primary"
          >
            {label}
          </label>
        )}
        <div className="relative">
          {leftIcon && (
            <div className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted">
              <span className={iconSizeStyles[inputSize]}>{leftIcon}</span>
            </div>
          )}
          <input
            ref={ref}
            id={inputId}
            className={cn(
              'w-full bg-bg-tertiary border rounded-lg',
              'text-text-primary placeholder:text-text-muted',
              'focus:outline-none focus:border-accent-primary focus:ring-1 focus:ring-accent-primary',
              'transition-colors duration-200',
              sizeStyles[inputSize],
              leftIcon && 'pl-10',
              (rightIcon || error) && 'pr-10',
              error
                ? 'border-status-error focus:border-status-error focus:ring-status-error'
                : 'border-border',
              props.disabled && 'opacity-50 cursor-not-allowed',
              className
            )}
            {...props}
          />
          {(rightIcon || error) && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2">
              {error ? (
                <ExclamationCircleIcon className={cn(iconSizeStyles[inputSize], 'text-status-error')} />
              ) : (
                <span className={cn(iconSizeStyles[inputSize], 'text-text-muted')}>{rightIcon}</span>
              )}
            </div>
          )}
        </div>
        {(error || hint) && (
          <p
            className={cn(
              'mt-1.5 text-body-sm',
              error ? 'text-status-error' : 'text-text-muted'
            )}
          >
            {error || hint}
          </p>
        )}
      </div>
    )
  }
)

Input.displayName = 'Input'

export default Input
