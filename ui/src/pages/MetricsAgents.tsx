import { useEffect } from 'react'
import { PageHeader } from '@/components/layout/PageHeader'
import { GlassCard } from '@/components/ui/GlassCard'
import { useMetricsStore } from '@/stores/metricsStore'
import { formatDuration, formatNumber } from '@/lib/formatters'
import { CpuChipIcon } from '@heroicons/react/24/outline'

export default function MetricsAgents() {
  const { agentMetrics, fetchAgentMetrics, isLoadingAgents } = useMetricsStore()

  useEffect(() => {
    fetchAgentMetrics()
  }, [])

  return (
    <div className="animate-in">
      <PageHeader
        title="Agent Performance"
        subtitle="Detailed metrics for each agent"
        breadcrumbs={[
          { label: 'Dashboard', href: '/' },
          { label: 'Metrics', href: '/metrics' },
          { label: 'Agents' },
        ]}
      />

      {isLoadingAgents ? (
        <div className="py-16 text-center text-text-muted">Loading agent metrics...</div>
      ) : !agentMetrics || agentMetrics.agents.length === 0 ? (
        <GlassCard>
          <div className="py-16 text-center">
            <CpuChipIcon className="h-16 w-16 mx-auto text-text-muted mb-4" />
            <h3 className="text-heading-3 text-text-primary mb-2">No Agent Data</h3>
            <p className="text-body text-text-secondary">
              Run some analyses to see agent performance metrics
            </p>
          </div>
        </GlassCard>
      ) : (
        <div className="space-y-6">
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {agentMetrics.agents.map((agent) => (
              <GlassCard key={agent.agent_name}>
                <div className="text-center">
                  <div className="w-12 h-12 rounded-xl bg-accent-primary/20 flex items-center justify-center mx-auto mb-3">
                    <CpuChipIcon className="h-6 w-6 text-accent-primary" />
                  </div>
                  <h3 className="text-body font-semibold text-text-primary mb-1">
                    {agent.agent_name}
                  </h3>
                  <p className="text-heading-2 text-accent-secondary">
                    {formatNumber(agent.operations)}
                  </p>
                  <p className="text-caption text-text-muted">operations</p>
                </div>
              </GlassCard>
            ))}
          </div>

          {/* Detailed Table */}
          <GlassCard title="Agent Details" noPadding>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="px-6 py-4 text-left text-body-sm font-semibold text-text-secondary">Agent</th>
                    <th className="px-6 py-4 text-right text-body-sm font-semibold text-text-secondary">Operations</th>
                    <th className="px-6 py-4 text-right text-body-sm font-semibold text-text-secondary">Total Time</th>
                    <th className="px-6 py-4 text-right text-body-sm font-semibold text-text-secondary">Avg Time</th>
                    <th className="px-6 py-4 text-right text-body-sm font-semibold text-text-secondary">Errors</th>
                    <th className="px-6 py-4 text-right text-body-sm font-semibold text-text-secondary">Error Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {agentMetrics.agents.map((agent) => {
                    const errorRate = agent.operations > 0 ? agent.errors / agent.operations : 0

                    return (
                      <tr key={agent.agent_name} className="border-b border-border/50 hover:bg-bg-tertiary/30">
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-lg bg-accent-primary/20 flex items-center justify-center">
                              <CpuChipIcon className="h-4 w-4 text-accent-primary" />
                            </div>
                            <span className="text-body-sm font-medium text-text-primary">
                              {agent.agent_name}
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4 text-right">
                          <span className="text-body-sm text-text-primary">
                            {formatNumber(agent.operations)}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-right">
                          <span className="text-body-sm text-text-secondary">
                            {formatDuration(agent.total_duration_ms)}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-right">
                          <span className="text-body-sm text-text-secondary">
                            {formatDuration(agent.avg_duration_ms)}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-right">
                          <span className={`text-body-sm ${agent.errors > 0 ? 'text-status-error' : 'text-text-muted'}`}>
                            {agent.errors}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-right">
                          <span className={`text-body-sm ${errorRate > 0.1 ? 'text-status-error' : errorRate > 0 ? 'text-status-warning' : 'text-status-success'}`}>
                            {(errorRate * 100).toFixed(1)}%
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </GlassCard>
        </div>
      )}
    </div>
  )
}
