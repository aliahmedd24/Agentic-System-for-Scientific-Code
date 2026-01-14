import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { PageHeader } from '@/components/layout/PageHeader'
import { GlassCard } from '@/components/ui/GlassCard'
import { Button } from '@/components/ui/Button'
import { StatusBadge } from '@/components/ui/StatusBadge'
import { useJobsStore } from '@/stores/jobsStore'
import { useMetricsStore } from '@/stores/metricsStore'
import { formatRelativeTime, formatPercent } from '@/lib/formatters'
import {
  PlusCircleIcon,
  ChartBarIcon,
  ClipboardDocumentListIcon,
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  CpuChipIcon,
} from '@heroicons/react/24/outline'

export default function Dashboard() {
  const { jobs, fetchJobs, isLoadingJobs } = useJobsStore()
  const { summary, healthStatus, fetchSummary, fetchHealth } = useMetricsStore()

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

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <GlassCard>
          <div className="flex items-center gap-4">
            <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-accent-primary/20">
              <ClipboardDocumentListIcon className="h-6 w-6 text-accent-primary" />
            </div>
            <div>
              <p className="text-body-sm text-text-secondary">Total Jobs</p>
              <p className="text-heading-2 text-text-primary">
                {summary?.total_jobs ?? '-'}
              </p>
            </div>
          </div>
        </GlassCard>

        <GlassCard>
          <div className="flex items-center gap-4">
            <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-status-success/20">
              <CheckCircleIcon className="h-6 w-6 text-status-success" />
            </div>
            <div>
              <p className="text-body-sm text-text-secondary">Completed</p>
              <p className="text-heading-2 text-text-primary">
                {summary?.completed_jobs ?? '-'}
              </p>
            </div>
          </div>
        </GlassCard>

        <GlassCard>
          <div className="flex items-center gap-4">
            <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-status-warning/20">
              <ChartBarIcon className="h-6 w-6 text-status-warning" />
            </div>
            <div>
              <p className="text-body-sm text-text-secondary">Avg Accuracy</p>
              <p className="text-heading-2 text-text-primary">
                {summary?.average_accuracy ? formatPercent(summary.average_accuracy) : '-'}
              </p>
            </div>
          </div>
        </GlassCard>

        <GlassCard>
          <div className="flex items-center gap-4">
            <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-status-info/20">
              <CpuChipIcon className="h-6 w-6 text-status-info" />
            </div>
            <div>
              <p className="text-body-sm text-text-secondary">Active Jobs</p>
              <p className="text-heading-2 text-text-primary">
                {healthStatus?.active_jobs ?? 0}
              </p>
            </div>
          </div>
        </GlassCard>
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
              <div className="py-8 text-center text-text-muted">Loading...</div>
            ) : recentJobs.length === 0 ? (
              <div className="py-8 text-center">
                <ClipboardDocumentListIcon className="h-12 w-12 mx-auto text-text-muted mb-3" />
                <p className="text-text-secondary">No jobs yet</p>
                <p className="text-body-sm text-text-muted mt-1">
                  Start your first analysis to see activity here
                </p>
              </div>
            ) : (
              <div className="space-y-3">
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
              </div>
            )}
          </GlassCard>
        </div>

        {/* Quick Actions */}
        <GlassCard title="Quick Actions">
          <div className="space-y-3">
            <Link to="/analyze" className="block">
              <Button variant="secondary" className="w-full justify-start" leftIcon={<PlusCircleIcon className="h-5 w-5" />}>
                New Analysis
              </Button>
            </Link>
            <Link to="/jobs" className="block">
              <Button variant="secondary" className="w-full justify-start" leftIcon={<ClipboardDocumentListIcon className="h-5 w-5" />}>
                View All Jobs
              </Button>
            </Link>
            <Link to="/metrics" className="block">
              <Button variant="secondary" className="w-full justify-start" leftIcon={<ChartBarIcon className="h-5 w-5" />}>
                System Metrics
              </Button>
            </Link>
          </div>
        </GlassCard>
      </div>
    </div>
  )
}
