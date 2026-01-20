import { cn } from '@/lib/cn'
import { Button } from '@/components/ui/Button'
import { InboxIcon } from '@heroicons/react/24/outline'

interface ActionConfig {
  label: string
  onClick: () => void
}

interface EmptyStateProps {
  icon?: React.ReactNode | 'error' | 'search' | 'document'
  title: string
  description?: string
  action?: React.ReactNode | ActionConfig
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

const sizeStyles = {
  sm: {
    container: 'py-8',
    icon: 'h-10 w-10',
    title: 'text-body font-medium',
    description: 'text-body-sm',
  },
  md: {
    container: 'py-12',
    icon: 'h-14 w-14',
    title: 'text-heading-3',
    description: 'text-body',
  },
  lg: {
    container: 'py-16',
    icon: 'h-20 w-20',
    title: 'text-heading-2',
    description: 'text-body-lg',
  },
}

function isActionConfig(action: unknown): action is ActionConfig {
  return (
    typeof action === 'object' &&
    action !== null &&
    'label' in action &&
    'onClick' in action
  )
}

export function EmptyState({
  icon,
  title,
  description,
  action,
  size = 'md',
  className,
}: EmptyStateProps) {
  const styles = sizeStyles[size]

  // Render action - either as JSX or from config object
  const renderAction = () => {
    if (!action) return null
    if (isActionConfig(action)) {
      return (
        <Button onClick={action.onClick}>
          {action.label}
        </Button>
      )
    }
    return action
  }

  return (
    <div className={cn('text-center', styles.container, className)}>
      <div className="flex justify-center mb-4">
        {icon || (
          <InboxIcon
            className={cn('text-text-muted', styles.icon)}
          />
        )}
      </div>
      <h3 className={cn('text-text-primary mb-2', styles.title)}>{title}</h3>
      {description && (
        <p className={cn('text-text-secondary mb-6 max-w-md mx-auto', styles.description)}>
          {description}
        </p>
      )}
      {action && <div className="flex justify-center">{renderAction()}</div>}
    </div>
  )
}

export default EmptyState
