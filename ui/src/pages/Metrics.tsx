import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { PageHeader } from '@/components/layout/PageHeader'
import { GlassCard } from '@/components/ui/GlassCard'
import { Button } from '@/components/ui/Button'
import { KPICard } from '@/components/metrics'
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
  ArrowPathIcon,
} from '@heroicons/react/24/outline'

export default function Metrics() {
  const { summary, healthStatus, fetchSummary, fetchHealth, isLoadingSummary } = useMetricsStore()

  useEffect(() => {
    fetchSummary()
    fetchHealth()
  }, [])

  const handleRefresh = () => {
    fetchSummary()
    fetchHealth()
  }

  // Calculate success rate for trend
  const successRate = summary && summary.total_jobs > 0
    ? (summary.completed_jobs / summary.total_jobs) * 100
    : 0

  return (
    <div className="animate-in">
      <PageHeader
        title="System Metrics"
        subtitle="Monitor system health and performance"
        actions={
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              onClick={handleRefresh}
              leftIcon={<ArrowPathIcon className="h-5 w-5" />}
              aria-label="Refresh metrics data"
            >
              Refresh
            </Button>
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
      <GlassCard className="mb-8" aria-label="System health status">
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
              role="status"
              aria-label={`System status: ${healthStatus?.status || 'unknown'}`}
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
        </div>
      </GlassCard>

      {/* KPI Grid using new KPICard component */}
      <div
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
        role="region"
        aria-label="Key performance indicators"
      >
        <KPICard
          title="Total Jobs"
          value={isLoadingSummary ? '-' : summary?.total_jobs ?? 0}
          icon={<ChartBarIcon className="h-6 w-6" />}
          iconBgColor="bg-accent-primary/20"
          iconColor="text-accent-primary"
          loading={isLoadingSummary}
        />

        <KPICard
          title="Completed"
          value={isLoadingSummary ? '-' : summary?.completed_jobs ?? 0}
          icon={<CheckCircleIcon className="h-6 w-6" />}
          iconBgColor="bg-status-success/20"
          iconColor="text-status-success"
          loading={isLoadingSummary}
          trend={
            summary && summary.total_jobs > 0
              ? {
                  value: Math.round(successRate),
                  direction: successRate >= 80 ? 'up' : successRate >= 50 ? 'neutral' : 'down',
                  label: 'Success rate',
                }
              : undefined
          }
        />

        <KPICard
          title="Failed"
          value={isLoadingSummary ? '-' : summary?.failed_jobs ?? 0}
          icon={<XCircleIcon className="h-6 w-6" />}
          iconBgColor="bg-status-error/20"
          iconColor="text-status-error"
          loading={isLoadingSummary}
        />

        <KPICard
          title="Avg Accuracy"
          value={
            isLoadingSummary
              ? '-'
              : summary?.average_accuracy
              ? formatPercent(summary.average_accuracy)
              : '-'
          }
          icon={<ArrowTrendingUpIcon className="h-6 w-6" />}
          iconBgColor="bg-status-warning/20"
          iconColor="text-status-warning"
          loading={isLoadingSummary}
          trend={
            summary?.average_accuracy
              ? {
                  value: Math.round(summary.average_accuracy * 100),
                  direction:
                    summary.average_accuracy >= 0.8
                      ? 'up'
                      : summary.average_accuracy >= 0.6
                      ? 'neutral'
                      : 'down',
                }
              : undefined
          }
        />
      </div>

      {/* Additional Stats */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <GlassCard title="Performance" icon={<ClockIcon className="h-5 w-5" />}>
          <div className="space-y-4" role="list" aria-label="Performance metrics">
            <div
              className="flex items-center justify-between py-3 border-b border-border"
              role="listitem"
            >
              <span className="text-body-sm text-text-secondary">Average Duration</span>
              <span className="text-body-sm font-medium text-text-primary">
                {summary?.average_duration_seconds
                  ? formatDuration(summary.average_duration_seconds * 1000)
                  : '-'}
              </span>
            </div>
            <div
              className="flex items-center justify-between py-3 border-b border-border"
              role="listitem"
            >
              <span className="text-body-sm text-text-secondary">Success Rate</span>
              <span className="text-body-sm font-medium text-status-success">
                {summary && summary.total_jobs > 0
                  ? formatPercent(summary.completed_jobs / summary.total_jobs)
                  : '-'}
              </span>
            </div>
            <div
              className="flex items-center justify-between py-3"
              role="listitem"
            >
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
          <nav className="space-y-3" aria-label="Metrics navigation">
            <Link to="/metrics/agents" className="block">
              <div className="flex items-center justify-between p-4 rounded-lg bg-bg-tertiary/50 hover:bg-bg-tertiary transition-colors">
                <div className="flex items-center gap-3">
                  <CpuChipIcon className="h-5 w-5 text-accent-primary" aria-hidden="true" />
                  <div>
                    <p className="text-body-sm font-medium text-text-primary">Agent Performance</p>
                    <p className="text-caption text-text-muted">View per-agent metrics</p>
                  </div>
                </div>
                <span className="text-text-muted" aria-hidden="true">→</span>
              </div>
            </Link>
            <Link to="/metrics/pipeline" className="block">
              <div className="flex items-center justify-between p-4 rounded-lg bg-bg-tertiary/50 hover:bg-bg-tertiary transition-colors">
                <div className="flex items-center gap-3">
                  <BeakerIcon className="h-5 w-5 text-accent-primary" aria-hidden="true" />
                  <div>
                    <p className="text-body-sm font-medium text-text-primary">Pipeline Statistics</p>
                    <p className="text-caption text-text-muted">View stage-by-stage metrics</p>
                  </div>
                </div>
                <span className="text-text-muted" aria-hidden="true">→</span>
              </div>
            </Link>
          </nav>
        </GlassCard>
      </div>
    </div>
  )
}
