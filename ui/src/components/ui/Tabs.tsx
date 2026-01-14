import { Tab } from '@headlessui/react'
import { cn } from '@/lib/cn'

export interface TabItem {
  label: string
  icon?: React.ComponentType<{ className?: string }>
  count?: number
  disabled?: boolean
}

interface TabsProps {
  items: TabItem[]
  children: React.ReactNode
  defaultIndex?: number
  selectedIndex?: number
  onChange?: (index: number) => void
  variant?: 'default' | 'pills' | 'underline'
  fullWidth?: boolean
  className?: string
}

const variantStyles = {
  default: {
    list: 'bg-bg-tertiary rounded-xl p-1',
    tab: cn(
      'rounded-lg py-2.5 px-4',
      'text-body-sm font-medium',
      'focus:outline-none focus:ring-2 focus:ring-accent-primary focus:ring-offset-2 focus:ring-offset-bg-tertiary'
    ),
    selected: 'bg-bg-secondary text-text-primary shadow-sm',
    unselected: 'text-text-muted hover:text-text-secondary hover:bg-bg-secondary/50',
  },
  pills: {
    list: 'space-x-2',
    tab: cn(
      'rounded-full py-2 px-4',
      'text-body-sm font-medium',
      'transition-all duration-200',
      'focus:outline-none focus:ring-2 focus:ring-accent-primary'
    ),
    selected: 'bg-accent-primary text-white',
    unselected: 'bg-bg-tertiary text-text-muted hover:text-text-secondary hover:bg-bg-secondary',
  },
  underline: {
    list: 'border-b border-border space-x-6',
    tab: cn(
      'py-3 px-1 -mb-px',
      'text-body-sm font-medium',
      'border-b-2 transition-colors duration-200',
      'focus:outline-none'
    ),
    selected: 'border-accent-primary text-accent-secondary',
    unselected: 'border-transparent text-text-muted hover:text-text-secondary hover:border-border',
  },
}

export function Tabs({
  items,
  children,
  defaultIndex = 0,
  selectedIndex,
  onChange,
  variant = 'default',
  fullWidth = false,
  className,
}: TabsProps) {
  const styles = variantStyles[variant]

  return (
    <Tab.Group
      defaultIndex={defaultIndex}
      selectedIndex={selectedIndex}
      onChange={onChange}
    >
      <Tab.List
        className={cn(
          'flex',
          fullWidth && 'w-full',
          styles.list,
          className
        )}
      >
        {items.map((item, idx) => (
          <Tab
            key={idx}
            disabled={item.disabled}
            className={({ selected }) =>
              cn(
                styles.tab,
                selected ? styles.selected : styles.unselected,
                fullWidth && 'flex-1',
                item.disabled && 'opacity-50 cursor-not-allowed'
              )
            }
          >
            <span className="flex items-center justify-center gap-2">
              {item.icon && <item.icon className="h-4 w-4" />}
              <span>{item.label}</span>
              {item.count !== undefined && (
                <span
                  className={cn(
                    'ml-1 rounded-full px-2 py-0.5 text-caption',
                    'bg-accent-primary/20 text-accent-secondary'
                  )}
                >
                  {item.count}
                </span>
              )}
            </span>
          </Tab>
        ))}
      </Tab.List>
      <Tab.Panels className="mt-4">{children}</Tab.Panels>
    </Tab.Group>
  )
}

export function TabPanel({
  children,
  className,
}: {
  children: React.ReactNode
  className?: string
}) {
  return (
    <Tab.Panel
      className={cn(
        'rounded-xl focus:outline-none focus:ring-2 focus:ring-accent-primary focus:ring-offset-2 focus:ring-offset-bg-primary',
        className
      )}
    >
      {children}
    </Tab.Panel>
  )
}

export default Tabs
