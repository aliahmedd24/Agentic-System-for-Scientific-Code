import { useEffect } from 'react'
import { PageHeader } from '@/components/layout/PageHeader'
import { GlassCard } from '@/components/ui/GlassCard'
import { KPICard, PipelineChart } from '@/components/metrics'
import { useMetricsStore } from '@/stores/metricsStore'
import { formatDuration, formatPercent } from '@/lib/formatters'
import {
  BeakerIcon,
  ChartBarIcon,
  CheckCircleIcon,
  ClockIcon,
  ArrowTrendingUpIcon,
} from '@heroicons/react/24/outline'

export default function MetricsPipeline() {
  const { pipelineMetrics, fetchPipelineMetrics, isLoadingPipeline } = useMetricsStore()

  useEffect(() => {
    fetchPipelineMetrics()
  }, [])

  // Calculate summary stats (totalExecutions reserved for future use)
  void (pipelineMetrics?.stages.reduce((sum, s) => sum + s.count, 0) || 0)
  const avgSuccessRate = pipelineMetrics?.stages.length
    ? pipelineMetrics.stages.reduce((sum, s) => sum + s.success_rate, 0) / pipelineMetrics.stages.length
    : 0
  const totalDuration = pipelineMetrics?.stages.reduce((sum, s) => sum + s.avg_duration_ms, 0) || 0

  return (
    <div className="animate-in">
      <PageHeader
        title="Pipeline Statistics"
        subtitle="Stage-by-stage performance analysis"
        breadcrumbs={[
          { label: 'Dashboard', href: '/' },
          { label: 'Metrics', href: '/metrics' },
          { label: 'Pipeline' },
        ]}
      />

      {isLoadingPipeline ? (
        <div className="py-16 text-center text-text-muted" role="status" aria-live="polite">
          Loading pipeline metrics...
        </div>
      ) : !pipelineMetrics || pipelineMetrics.stages.length === 0 ? (
        <GlassCard>
          <div className="py-16 text-center">
            <BeakerIcon className="h-16 w-16 mx-auto text-text-muted mb-4" aria-hidden="true" />
            <h3 className="text-heading-3 text-text-primary mb-2">No Pipeline Data</h3>
            <p className="text-body text-text-secondary">
              Run some analyses to see pipeline metrics
            </p>
          </div>
        </GlassCard>
      ) : (
        <div className="space-y-6">
          {/* Summary KPI Cards */}
          <div
            className="grid grid-cols-1 md:grid-cols-4 gap-6"
            role="region"
            aria-label="Pipeline summary statistics"
          >
            <KPICard
              title="Accuracy Score"
              value={formatPercent(pipelineMetrics.accuracy_score)}
              icon={<ArrowTrendingUpIcon className="h-6 w-6" />}
              iconBgColor="bg-accent-primary/20"
              iconColor="text-accent-primary"
              trend={{
                value: Math.round(pipelineMetrics.accuracy_score * 100),
                direction:
                  pipelineMetrics.accuracy_score >= 0.8
                    ? 'up'
                    : pipelineMetrics.accuracy_score >= 0.6
                    ? 'neutral'
                    : 'down',
              }}
            />
            <KPICard
              title="Total Stages"
              value={pipelineMetrics.stages.length}
              icon={<BeakerIcon className="h-6 w-6" />}
              iconBgColor="bg-status-info/20"
              iconColor="text-status-info"
            />
            <KPICard
              title="Avg Success Rate"
              value={formatPercent(avgSuccessRate)}
              icon={<CheckCircleIcon className="h-6 w-6" />}
              iconBgColor="bg-status-success/20"
              iconColor="text-status-success"
            />
            <KPICard
              title="Total Pipeline Time"
              value={formatDuration(totalDuration)}
              icon={<ClockIcon className="h-6 w-6" />}
              iconBgColor="bg-status-warning/20"
              iconColor="text-status-warning"
              subtitle="Average per job"
            />
          </div>

          {/* D3.js Pipeline Chart */}
          <PipelineChart stages={pipelineMetrics.stages} />

          {/* Stage Metrics Table */}
          <GlassCard
            title="Stage Performance"
            icon={<ChartBarIcon className="h-5 w-5" />}
            noPadding
          >
            <div className="overflow-x-auto">
              <table
                className="w-full"
                role="table"
                aria-label="Pipeline stage performance details"
              >
                <thead>
                  <tr className="border-b border-border">
                    <th
                      scope="col"
                      className="px-6 py-4 text-left text-body-sm font-semibold text-text-secondary"
                    >
                      Stage
                    </th>
                    <th
                      scope="col"
                      className="px-6 py-4 text-right text-body-sm font-semibold text-text-secondary"
                    >
                      Executions
                    </th>
                    <th
                      scope="col"
                      className="px-6 py-4 text-right text-body-sm font-semibold text-text-secondary"
                    >
                      Avg Duration
                    </th>
                    <th
                      scope="col"
                      className="px-6 py-4 text-right text-body-sm font-semibold text-text-secondary"
                    >
                      Success Rate
                    </th>
                    <th
                      scope="col"
                      className="px-6 py-4 text-left text-body-sm font-semibold text-text-secondary"
                    >
                      Performance
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {pipelineMetrics.stages.map((stage) => (
                    <tr
                      key={stage.stage}
                      className="border-b border-border/50 hover:bg-bg-tertiary/30"
                    >
                      <td className="px-6 py-4">
                        <span className="text-body-sm font-medium text-text-primary capitalize">
                          {stage.stage.replace(/_/g, ' ')}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-right">
                        <span className="text-body-sm text-text-primary">{stage.count}</span>
                      </td>
                      <td className="px-6 py-4 text-right">
                        <span className="text-body-sm text-text-secondary">
                          {formatDuration(stage.avg_duration_ms)}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-right">
                        <span
                          className={`text-body-sm ${
                            stage.success_rate >= 0.9
                              ? 'text-status-success'
                              : stage.success_rate >= 0.7
                              ? 'text-status-warning'
                              : 'text-status-error'
                          }`}
                        >
                          {formatPercent(stage.success_rate)}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-2">
                          <div
                            className="w-32 h-2 bg-bg-tertiary rounded-full overflow-hidden"
                            role="progressbar"
                            aria-valuenow={Math.round(stage.success_rate * 100)}
                            aria-valuemin={0}
                            aria-valuemax={100}
                            aria-label={`${stage.stage} success rate: ${formatPercent(stage.success_rate)}`}
                          >
                            <div
                              className={`h-full rounded-full ${
                                stage.success_rate >= 0.9
                                  ? 'bg-status-success'
                                  : stage.success_rate >= 0.7
                                  ? 'bg-status-warning'
                                  : 'bg-status-error'
                              }`}
                              style={{ width: `${stage.success_rate * 100}%` }}
                            />
                          </div>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </GlassCard>
        </div>
      )}
    </div>
  )
}
