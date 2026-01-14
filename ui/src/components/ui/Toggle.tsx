import { Switch } from '@headlessui/react'
import { cn } from '@/lib/cn'

interface ToggleProps {
  label?: string
  description?: string
  checked: boolean
  onChange: (checked: boolean) => void
  disabled?: boolean
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

const sizeConfig = {
  sm: {
    track: 'h-5 w-9',
    thumb: 'h-3 w-3',
    translate: 'translate-x-4',
  },
  md: {
    track: 'h-6 w-11',
    thumb: 'h-4 w-4',
    translate: 'translate-x-5',
  },
  lg: {
    track: 'h-7 w-14',
    thumb: 'h-5 w-5',
    translate: 'translate-x-7',
  },
}

export function Toggle({
  label,
  description,
  checked,
  onChange,
  disabled = false,
  size = 'md',
  className,
}: ToggleProps) {
  const config = sizeConfig[size]

  return (
    <Switch.Group>
      <div className={cn('flex items-center justify-between', className)}>
        {(label || description) && (
          <div className="flex-1 mr-4">
            {label && (
              <Switch.Label className="text-body font-medium text-text-primary cursor-pointer">
                {label}
              </Switch.Label>
            )}
            {description && (
              <Switch.Description className="text-body-sm text-text-muted">
                {description}
              </Switch.Description>
            )}
          </div>
        )}
        <Switch
          checked={checked}
          onChange={onChange}
          disabled={disabled}
          className={cn(
            'relative inline-flex shrink-0 cursor-pointer rounded-full',
            'border-2 border-transparent transition-colors duration-200 ease-in-out',
            'focus:outline-none focus:ring-2 focus:ring-accent-primary focus:ring-offset-2 focus:ring-offset-bg-primary',
            config.track,
            checked ? 'bg-accent-primary' : 'bg-bg-tertiary',
            disabled && 'opacity-50 cursor-not-allowed'
          )}
        >
          <span
            aria-hidden="true"
            className={cn(
              'pointer-events-none inline-block rounded-full bg-white shadow-lg',
              'ring-0 transition duration-200 ease-in-out',
              config.thumb,
              checked ? config.translate : 'translate-x-1'
            )}
            style={{ marginTop: '1px' }}
          />
        </Switch>
      </div>
    </Switch.Group>
  )
}

export default Toggle
