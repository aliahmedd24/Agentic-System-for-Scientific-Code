import { useEffect } from 'react'
import { PageHeader } from '@/components/layout/PageHeader'
import { GlassCard } from '@/components/ui/GlassCard'
import { useMetricsStore } from '@/stores/metricsStore'
import { formatDuration, formatPercent } from '@/lib/formatters'
import { BeakerIcon, ChartBarIcon } from '@heroicons/react/24/outline'

export default function MetricsPipeline() {
  const { pipelineMetrics, fetchPipelineMetrics, isLoadingPipeline } = useMetricsStore()

  useEffect(() => {
    fetchPipelineMetrics()
  }, [])

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
        <div className="py-16 text-center text-text-muted">Loading pipeline metrics...</div>
      ) : !pipelineMetrics || pipelineMetrics.stages.length === 0 ? (
        <GlassCard>
          <div className="py-16 text-center">
            <BeakerIcon className="h-16 w-16 mx-auto text-text-muted mb-4" />
            <h3 className="text-heading-3 text-text-primary mb-2">No Pipeline Data</h3>
            <p className="text-body text-text-secondary">
              Run some analyses to see pipeline metrics
            </p>
          </div>
        </GlassCard>
      ) : (
        <div className="space-y-6">
          {/* Accuracy Score */}
          <GlassCard>
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-heading-3 text-text-primary">Overall Accuracy Score</h3>
                <p className="text-body-sm text-text-secondary mt-1">
                  Aggregate mapping accuracy across all jobs
                </p>
              </div>
              <div className="text-right">
                <p className="text-display text-accent-primary">
                  {formatPercent(pipelineMetrics.accuracy_score)}
                </p>
              </div>
            </div>
          </GlassCard>

          {/* Stage Metrics */}
          <GlassCard title="Stage Performance" icon={<ChartBarIcon className="h-5 w-5" />} noPadding>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="px-6 py-4 text-left text-body-sm font-semibold text-text-secondary">Stage</th>
                    <th className="px-6 py-4 text-right text-body-sm font-semibold text-text-secondary">Executions</th>
                    <th className="px-6 py-4 text-right text-body-sm font-semibold text-text-secondary">Avg Duration</th>
                    <th className="px-6 py-4 text-right text-body-sm font-semibold text-text-secondary">Success Rate</th>
                    <th className="px-6 py-4 text-left text-body-sm font-semibold text-text-secondary">Performance</th>
                  </tr>
                </thead>
                <tbody>
                  {pipelineMetrics.stages.map((stage) => (
                    <tr key={stage.stage} className="border-b border-border/50 hover:bg-bg-tertiary/30">
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
                        <span className={`text-body-sm ${
                          stage.success_rate >= 0.9 ? 'text-status-success' :
                          stage.success_rate >= 0.7 ? 'text-status-warning' :
                          'text-status-error'
                        }`}>
                          {formatPercent(stage.success_rate)}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-2">
                          <div className="w-32 h-2 bg-bg-tertiary rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${
                                stage.success_rate >= 0.9 ? 'bg-status-success' :
                                stage.success_rate >= 0.7 ? 'bg-status-warning' :
                                'bg-status-error'
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

          {/* Timing Chart Placeholder */}
          <GlassCard title="Stage Timing Distribution">
            <div className="space-y-4">
              {pipelineMetrics.stages.map((stage) => {
                const maxDuration = Math.max(...pipelineMetrics.stages.map((s) => s.avg_duration_ms))
                const widthPercent = (stage.avg_duration_ms / maxDuration) * 100

                return (
                  <div key={stage.stage} className="flex items-center gap-4">
                    <div className="w-32 text-body-sm text-text-secondary capitalize">
                      {stage.stage.replace(/_/g, ' ')}
                    </div>
                    <div className="flex-1">
                      <div className="h-8 bg-bg-tertiary rounded-lg overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-accent-primary to-purple-500 rounded-lg flex items-center justify-end px-2"
                          style={{ width: `${widthPercent}%` }}
                        >
                          <span className="text-caption text-white font-medium">
                            {formatDuration(stage.avg_duration_ms)}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </GlassCard>
        </div>
      )}
    </div>
  )
}
