import { cn } from '@/lib/cn'

interface PageHeaderProps {
  title: string
  subtitle?: string
  actions?: React.ReactNode
  breadcrumbs?: Array<{ label: string; href?: string }>
  className?: string
}

export function PageHeader({
  title,
  subtitle,
  actions,
  breadcrumbs,
  className,
}: PageHeaderProps) {
  return (
    <header className={cn('mb-8', className)}>
      {/* Breadcrumbs */}
      {breadcrumbs && breadcrumbs.length > 0 && (
        <nav className="mb-3">
          <ol className="flex items-center gap-2 text-body-sm">
            {breadcrumbs.map((crumb, idx) => (
              <li key={idx} className="flex items-center gap-2">
                {idx > 0 && <span className="text-text-muted">/</span>}
                {crumb.href ? (
                  <a
                    href={crumb.href}
                    className="text-text-secondary hover:text-accent-secondary transition-colors"
                  >
                    {crumb.label}
                  </a>
                ) : (
                  <span className="text-text-muted">{crumb.label}</span>
                )}
              </li>
            ))}
          </ol>
        </nav>
      )}

      {/* Title and Actions */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <h1 className="text-display gradient-text truncate">{title}</h1>
          {subtitle && (
            <p className="mt-2 text-body-lg text-text-secondary">{subtitle}</p>
          )}
        </div>
        {actions && (
          <div className="flex items-center gap-3 flex-shrink-0">
            {actions}
          </div>
        )}
      </div>
    </header>
  )
}

export default PageHeader
