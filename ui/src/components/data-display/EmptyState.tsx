import { cn } from '@/lib/cn'
import { InboxIcon } from '@heroicons/react/24/outline'

interface EmptyStateProps {
  icon?: React.ReactNode
  title: string
  description?: string
  action?: React.ReactNode
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

export function EmptyState({
  icon,
  title,
  description,
  action,
  size = 'md',
  className,
}: EmptyStateProps) {
  const styles = sizeStyles[size]

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
      {action && <div className="flex justify-center">{action}</div>}
    </div>
  )
}

export default EmptyState
