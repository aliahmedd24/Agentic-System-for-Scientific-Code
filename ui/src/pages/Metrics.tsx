import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { PageHeader } from '@/components/layout/PageHeader'
import { GlassCard } from '@/components/ui/GlassCard'
import { Button } from '@/components/ui/Button'
import { useMetricsStore } from '@/stores/metricsStore'
import { formatPercent, formatDuration } from '@/lib/formatters'
import {
  ChartBarIcon,
  CpuChipIcon,
  BeakerIcon,
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  ArrowTrendingUpIcon,
} from '@heroicons/react/24/outline'

export default function Metrics() {
  const { summary, healthStatus, fetchSummary, fetchHealth, isLoadingSummary } = useMetricsStore()

  useEffect(() => {
    fetchSummary()
    fetchHealth()
  }, [])

  return (
    <div className="animate-in">
      <PageHeader
        title="System Metrics"
        subtitle="Monitor system health and performance"
        actions={
          <div className="flex items-center gap-3">
            <Link to="/metrics/agents">
              <Button variant="secondary" leftIcon={<CpuChipIcon className="h-5 w-5" />}>
                Agent Metrics
              </Button>
            </Link>
            <Link to="/metrics/pipeline">
              <Button variant="secondary" leftIcon={<BeakerIcon className="h-5 w-5" />}>
                Pipeline Stats
              </Button>
            </Link>
          </div>
        }
      />

      {/* Health Status */}
      <GlassCard className="mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div
              className={`w-4 h-4 rounded-full ${
                healthStatus?.status === 'healthy'
                  ? 'bg-status-success'
                  : healthStatus?.status === 'degraded'
                  ? 'bg-status-warning'
                  : 'bg-status-error'
              }`}
            />
            <div>
              <h3 className="text-heading-3 text-text-primary capitalize">
                System {healthStatus?.status || 'Unknown'}
              </h3>
              <p className="text-body-sm text-text-secondary">
                {healthStatus?.active_jobs || 0} active jobs
              </p>
            </div>
          </div>
          <Button variant="ghost" onClick={() => { fetchSummary(); fetchHealth() }}>
            Refresh
          </Button>
        </div>
      </GlassCard>

      {/* KPI Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <GlassCard>
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-accent-primary/20 flex items-center justify-center">
              <ChartBarIcon className="h-6 w-6 text-accent-primary" />
            </div>
            <div>
              <p className="text-caption text-text-muted">Total Jobs</p>
              <p className="text-heading-2 text-text-primary">
                {isLoadingSummary ? '-' : summary?.total_jobs ?? 0}
              </p>
            </div>
          </div>
        </GlassCard>

        <GlassCard>
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-status-success/20 flex items-center justify-center">
              <CheckCircleIcon className="h-6 w-6 text-status-success" />
            </div>
            <div>
              <p className="text-caption text-text-muted">Completed</p>
              <p className="text-heading-2 text-text-primary">
                {isLoadingSummary ? '-' : summary?.completed_jobs ?? 0}
              </p>
            </div>
          </div>
        </GlassCard>

        <GlassCard>
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-status-error/20 flex items-center justify-center">
              <XCircleIcon className="h-6 w-6 text-status-error" />
            </div>
            <div>
              <p className="text-caption text-text-muted">Failed</p>
              <p className="text-heading-2 text-text-primary">
                {isLoadingSummary ? '-' : summary?.failed_jobs ?? 0}
              </p>
            </div>
          </div>
        </GlassCard>

        <GlassCard>
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-status-warning/20 flex items-center justify-center">
              <ArrowTrendingUpIcon className="h-6 w-6 text-status-warning" />
            </div>
            <div>
              <p className="text-caption text-text-muted">Avg Accuracy</p>
              <p className="text-heading-2 text-text-primary">
                {isLoadingSummary ? '-' : summary?.average_accuracy ? formatPercent(summary.average_accuracy) : '-'}
              </p>
            </div>
          </div>
        </GlassCard>
      </div>

      {/* Additional Stats */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <GlassCard title="Performance" icon={<ClockIcon className="h-5 w-5" />}>
          <div className="space-y-4">
            <div className="flex items-center justify-between py-3 border-b border-border">
              <span className="text-body-sm text-text-secondary">Average Duration</span>
              <span className="text-body-sm font-medium text-text-primary">
                {summary?.average_duration_seconds
                  ? formatDuration(summary.average_duration_seconds * 1000)
                  : '-'}
              </span>
            </div>
            <div className="flex items-center justify-between py-3 border-b border-border">
              <span className="text-body-sm text-text-secondary">Success Rate</span>
              <span className="text-body-sm font-medium text-status-success">
                {summary && summary.total_jobs > 0
                  ? formatPercent(summary.completed_jobs / summary.total_jobs)
                  : '-'}
              </span>
            </div>
            <div className="flex items-center justify-between py-3">
              <span className="text-body-sm text-text-secondary">Failure Rate</span>
              <span className="text-body-sm font-medium text-status-error">
                {summary && summary.total_jobs > 0
                  ? formatPercent(summary.failed_jobs / summary.total_jobs)
                  : '-'}
              </span>
            </div>
          </div>
        </GlassCard>

        <GlassCard title="Quick Links" icon={<ChartBarIcon className="h-5 w-5" />}>
          <div className="space-y-3">
            <Link to="/metrics/agents" className="block">
              <div className="flex items-center justify-between p-4 rounded-lg bg-bg-tertiary/50 hover:bg-bg-tertiary transition-colors">
                <div className="flex items-center gap-3">
                  <CpuChipIcon className="h-5 w-5 text-accent-primary" />
                  <div>
                    <p className="text-body-sm font-medium text-text-primary">Agent Performance</p>
                    <p className="text-caption text-text-muted">View per-agent metrics</p>
                  </div>
                </div>
                <span className="text-text-muted">→</span>
              </div>
            </Link>
            <Link to="/metrics/pipeline" className="block">
              <div className="flex items-center justify-between p-4 rounded-lg bg-bg-tertiary/50 hover:bg-bg-tertiary transition-colors">
                <div className="flex items-center gap-3">
                  <BeakerIcon className="h-5 w-5 text-accent-primary" />
                  <div>
                    <p className="text-body-sm font-medium text-text-primary">Pipeline Statistics</p>
                    <p className="text-caption text-text-muted">View stage-by-stage metrics</p>
                  </div>
                </div>
                <span className="text-text-muted">→</span>
              </div>
            </Link>
          </div>
        </GlassCard>
      </div>
    </div>
  )
}
