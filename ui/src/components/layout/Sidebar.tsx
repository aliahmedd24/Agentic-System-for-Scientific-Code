import { NavLink, useLocation } from 'react-router-dom'
import { cn } from '@/lib/cn'
import { useHealthStatus } from '@/hooks/useHealthStatus'
import {
  HomeIcon,
  PlusCircleIcon,
  ClipboardDocumentListIcon,
  ChartBarIcon,
  CpuChipIcon,
  Cog6ToothIcon,
  BeakerIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
} from '@heroicons/react/24/outline'

interface NavItem {
  name: string
  href: string
  icon: React.ComponentType<{ className?: string }>
  badge?: number
}

interface NavSection {
  title: string
  items: NavItem[]
}

const navigation: NavSection[] = [
  {
    title: '',
    items: [
      { name: 'Dashboard', href: '/', icon: HomeIcon },
    ],
  },
  {
    title: 'Analysis',
    items: [
      { name: 'New Analysis', href: '/analyze', icon: PlusCircleIcon },
      { name: 'Job History', href: '/jobs', icon: ClipboardDocumentListIcon },
    ],
  },
  {
    title: 'Monitoring',
    items: [
      { name: 'System Metrics', href: '/metrics', icon: ChartBarIcon },
      { name: 'Agent Performance', href: '/metrics/agents', icon: CpuChipIcon },
      { name: 'Pipeline Stats', href: '/metrics/pipeline', icon: BeakerIcon },
    ],
  },
  {
    title: 'Settings',
    items: [
      { name: 'Configuration', href: '/settings', icon: Cog6ToothIcon },
    ],
  },
]

interface HealthIndicatorProps {
  status: 'healthy' | 'degraded' | 'unhealthy' | 'unknown'
  activeJobs?: number
}

function HealthIndicator({ status, activeJobs = 0 }: HealthIndicatorProps) {
  const statusConfig = {
    healthy: { color: 'text-status-success', bg: 'bg-status-success', label: 'Healthy' },
    degraded: { color: 'text-status-warning', bg: 'bg-status-warning', label: 'Degraded' },
    unhealthy: { color: 'text-status-error', bg: 'bg-status-error', label: 'Unhealthy' },
    unknown: { color: 'text-text-muted', bg: 'bg-text-muted', label: 'Unknown' },
  }

  const config = statusConfig[status]
  const Icon = status === 'healthy' ? CheckCircleIcon : ExclamationCircleIcon

  return (
    <div
      className="px-4 py-3 border-t border-border"
      role="status"
      aria-live="polite"
      aria-label={`System status: ${config.label}${activeJobs > 0 ? `, ${activeJobs} active jobs` : ''}`}
    >
      <div className="flex items-center gap-3">
        <div className={cn('w-2 h-2 rounded-full animate-pulse', config.bg)} />
        <div className="flex-1 min-w-0">
          <p className={cn('text-body-sm font-medium', config.color)}>{config.label}</p>
          {activeJobs > 0 && (
            <p className="text-caption text-text-muted">{activeJobs} active job{activeJobs !== 1 ? 's' : ''}</p>
          )}
        </div>
        <Icon className={cn('h-5 w-5', config.color)} />
      </div>
    </div>
  )
}

interface SidebarProps {
  className?: string
}

export function Sidebar({ className }: SidebarProps) {
  const location = useLocation()
  const { data: healthStatus } = useHealthStatus({ pollInterval: 30000 })

  // Determine health status or default to unknown
  const status = healthStatus?.status || 'unknown'
  const activeJobs = healthStatus?.active_jobs || 0

  return (
    <aside
      className={cn(
        'fixed left-0 top-0 z-40 h-screen w-64 flex flex-col',
        'bg-bg-secondary/95 backdrop-blur-xl border-r border-border',
        className
      )}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-6 py-5 border-b border-border">
        <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-accent-primary to-purple-500">
          <BeakerIcon className="h-6 w-6 text-white" />
        </div>
        <div className="flex-1 min-w-0">
          <h1 className="text-body font-bold text-text-primary truncate">Scientific Agent</h1>
          <p className="text-caption text-text-muted">Paper Analysis System</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-4 px-3">
        {navigation.map((section, sectionIdx) => (
          <div key={sectionIdx} className={cn(sectionIdx > 0 && 'mt-6')}>
            {section.title && (
              <h2 className="px-3 mb-2 text-caption font-semibold text-text-muted uppercase tracking-wider">
                {section.title}
              </h2>
            )}
            <ul className="space-y-1">
              {section.items.map((item) => {
                const isActive = location.pathname === item.href ||
                  (item.href !== '/' && location.pathname.startsWith(item.href))

                return (
                  <li key={item.name}>
                    <NavLink
                      to={item.href}
                      className={cn(
                        'flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200',
                        'text-body-sm font-medium',
                        isActive
                          ? 'bg-accent-primary/20 text-accent-secondary border border-accent-primary/30'
                          : 'text-text-secondary hover:bg-bg-tertiary hover:text-text-primary'
                      )}
                    >
                      <item.icon
                        className={cn(
                          'h-5 w-5 flex-shrink-0',
                          isActive ? 'text-accent-secondary' : 'text-text-muted'
                        )}
                      />
                      <span className="flex-1">{item.name}</span>
                      {item.badge !== undefined && item.badge > 0 && (
                        <span className="flex items-center justify-center px-2 py-0.5 text-caption font-semibold rounded-full bg-accent-primary text-white">
                          {item.badge}
                        </span>
                      )}
                    </NavLink>
                  </li>
                )
              })}
            </ul>
          </div>
        ))}
      </nav>

      {/* Health Indicator */}
      <HealthIndicator status={status} activeJobs={activeJobs} />
    </aside>
  )
}

export default Sidebar
