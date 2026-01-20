import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { PageHeader } from '@/components/layout/PageHeader'
import { GlassCard } from '@/components/ui/GlassCard'
import { Button } from '@/components/ui/Button'
import { StatusBadge } from '@/components/ui/StatusBadge'
import { KPICard } from '@/components/metrics'
import { useJobsStore } from '@/stores/jobsStore'
import { useMetricsStore } from '@/stores/metricsStore'
import { formatRelativeTime, formatPercent } from '@/lib/formatters'
import {
  PlusCircleIcon,
  ChartBarIcon,
  ClipboardDocumentListIcon,
  CheckCircleIcon,
  CpuChipIcon,
} from '@heroicons/react/24/outline'

export default function Dashboard() {
  const { jobs, fetchJobs, isLoadingJobs } = useJobsStore()
  const { summary, healthStatus, fetchSummary, fetchHealth, isLoadingSummary } = useMetricsStore()

  useEffect(() => {
    fetchJobs(5)
    fetchSummary()
    fetchHealth()
  }, [])

  const recentJobs = jobs.slice(0, 5)

  return (
    <div className="animate-in">
      <PageHeader
        title="Dashboard"
        subtitle="Welcome to the Scientific Paper Analysis System"
        actions={
          <Link to="/analyze">
            <Button leftIcon={<PlusCircleIcon className="h-5 w-5" />}>
              New Analysis
            </Button>
          </Link>
        }
      />

      {/* KPI Cards using new component */}
      <div
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
        role="region"
        aria-label="Key performance indicators"
      >
        <KPICard
          title="Total Jobs"
          value={summary?.total_jobs ?? '-'}
          icon={<ClipboardDocumentListIcon className="h-6 w-6" />}
          iconBgColor="bg-accent-primary/20"
          iconColor="text-accent-primary"
          loading={isLoadingSummary}
        />

        <KPICard
          title="Completed"
          value={summary?.completed_jobs ?? '-'}
          icon={<CheckCircleIcon className="h-6 w-6" />}
          iconBgColor="bg-status-success/20"
          iconColor="text-status-success"
          loading={isLoadingSummary}
          trend={
            summary && summary.total_jobs > 0
              ? {
                  value: Math.round((summary.completed_jobs / summary.total_jobs) * 100),
                  direction:
                    summary.completed_jobs / summary.total_jobs >= 0.8
                      ? 'up'
                      : summary.completed_jobs / summary.total_jobs >= 0.5
                      ? 'neutral'
                      : 'down',
                  label: 'Success rate',
                }
              : undefined
          }
        />

        <KPICard
          title="Avg Accuracy"
          value={summary?.average_accuracy ? formatPercent(summary.average_accuracy) : '-'}
          icon={<ChartBarIcon className="h-6 w-6" />}
          iconBgColor="bg-status-warning/20"
          iconColor="text-status-warning"
          loading={isLoadingSummary}
        />

        <KPICard
          title="Active Jobs"
          value={healthStatus?.active_jobs ?? 0}
          icon={<CpuChipIcon className="h-6 w-6" />}
          iconBgColor="bg-status-info/20"
          iconColor="text-status-info"
          loading={isLoadingSummary}
          subtitle={healthStatus?.status === 'healthy' ? 'System healthy' : 'Check system status'}
        />
      </div>

      {/* Two column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Activity */}
        <div className="lg:col-span-2">
          <GlassCard
            title="Recent Activity"
            subtitle="Latest analysis jobs"
            headerActions={
              <Link to="/jobs">
                <Button variant="ghost" size="sm">View All</Button>
              </Link>
            }
          >
            {isLoadingJobs ? (
              <div className="py-8 text-center text-text-muted" role="status" aria-live="polite">
                Loading...
              </div>
            ) : recentJobs.length === 0 ? (
              <div className="py-8 text-center">
                <ClipboardDocumentListIcon className="h-12 w-12 mx-auto text-text-muted mb-3" aria-hidden="true" />
                <p className="text-text-secondary">No jobs yet</p>
                <p className="text-body-sm text-text-muted mt-1">
                  Start your first analysis to see activity here
                </p>
              </div>
            ) : (
              <nav className="space-y-3" aria-label="Recent jobs">
                {recentJobs.map((job) => (
                  <Link
                    key={job.job_id}
                    to={`/jobs/${job.job_id}`}
                    className="flex items-center justify-between p-3 rounded-lg bg-bg-tertiary/50 hover:bg-bg-tertiary transition-colors"
                  >
                    <div className="min-w-0 flex-1">
                      <p className="text-body-sm font-medium text-text-primary truncate">
                        {job.paper_source || 'Unknown paper'}
                      </p>
                      <p className="text-caption text-text-muted truncate">
                        {job.repo_url || 'Unknown repo'}
                      </p>
                    </div>
                    <div className="flex items-center gap-3 ml-4">
                      <StatusBadge status={job.status as any} size="sm" />
                      <span className="text-caption text-text-muted">
                        {formatRelativeTime(job.created_at)}
                      </span>
                    </div>
                  </Link>
                ))}
              </nav>
            )}
          </GlassCard>
        </div>

        {/* Quick Actions */}
        <GlassCard title="Quick Actions">
          <nav className="space-y-3" aria-label="Quick actions">
            <Link to="/analyze" className="block">
              <Button
                variant="secondary"
                className="w-full justify-start"
                leftIcon={<PlusCircleIcon className="h-5 w-5" />}
              >
                New Analysis
              </Button>
            </Link>
            <Link to="/jobs" className="block">
              <Button
                variant="secondary"
                className="w-full justify-start"
                leftIcon={<ClipboardDocumentListIcon className="h-5 w-5" />}
              >
                View All Jobs
              </Button>
            </Link>
            <Link to="/metrics" className="block">
              <Button
                variant="secondary"
                className="w-full justify-start"
                leftIcon={<ChartBarIcon className="h-5 w-5" />}
              >
                System Metrics
              </Button>
            </Link>
          </nav>
        </GlassCard>
      </div>
    </div>
  )
}
