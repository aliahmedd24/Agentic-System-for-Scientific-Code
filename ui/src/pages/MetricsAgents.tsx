import { useEffect } from 'react'
import { PageHeader } from '@/components/layout/PageHeader'
import { GlassCard } from '@/components/ui/GlassCard'
import { KPICard, AgentTable } from '@/components/metrics'
import { useMetricsStore } from '@/stores/metricsStore'
import { formatNumber } from '@/lib/formatters'
import { CpuChipIcon, ClockIcon, ExclamationTriangleIcon, CheckCircleIcon } from '@heroicons/react/24/outline'

export default function MetricsAgents() {
  const { agentMetrics, fetchAgentMetrics, isLoadingAgents } = useMetricsStore()

  useEffect(() => {
    fetchAgentMetrics()
  }, [])

  // Calculate summary stats
  const totalOperations = agentMetrics?.agents.reduce((sum, a) => sum + a.operations, 0) || 0
  const totalErrors = agentMetrics?.agents.reduce((sum, a) => sum + a.errors, 0) || 0
  const avgDuration = agentMetrics?.agents.length
    ? agentMetrics.agents.reduce((sum, a) => sum + a.avg_duration_ms, 0) / agentMetrics.agents.length
    : 0
  const errorRate = totalOperations > 0 ? (totalErrors / totalOperations) * 100 : 0

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
        <div className="py-16 text-center text-text-muted" role="status" aria-live="polite">
          Loading agent metrics...
        </div>
      ) : !agentMetrics || agentMetrics.agents.length === 0 ? (
        <GlassCard>
          <div className="py-16 text-center">
            <CpuChipIcon className="h-16 w-16 mx-auto text-text-muted mb-4" aria-hidden="true" />
            <h3 className="text-heading-3 text-text-primary mb-2">No Agent Data</h3>
            <p className="text-body text-text-secondary">
              Run some analyses to see agent performance metrics
            </p>
          </div>
        </GlassCard>
      ) : (
        <div className="space-y-6">
          {/* Summary KPI Cards */}
          <div
            className="grid grid-cols-1 md:grid-cols-4 gap-6"
            role="region"
            aria-label="Agent summary statistics"
          >
            <KPICard
              title="Total Agents"
              value={agentMetrics.agents.length}
              icon={<CpuChipIcon className="h-6 w-6" />}
              iconBgColor="bg-accent-primary/20"
              iconColor="text-accent-primary"
            />
            <KPICard
              title="Total Operations"
              value={formatNumber(totalOperations)}
              icon={<CheckCircleIcon className="h-6 w-6" />}
              iconBgColor="bg-status-success/20"
              iconColor="text-status-success"
            />
            <KPICard
              title="Avg Duration"
              value={`${(avgDuration / 1000).toFixed(1)}s`}
              icon={<ClockIcon className="h-6 w-6" />}
              iconBgColor="bg-status-info/20"
              iconColor="text-status-info"
              subtitle="Per operation"
            />
            <KPICard
              title="Error Rate"
              value={`${errorRate.toFixed(1)}%`}
              icon={<ExclamationTriangleIcon className="h-6 w-6" />}
              iconBgColor={errorRate > 5 ? 'bg-status-error/20' : 'bg-status-warning/20'}
              iconColor={errorRate > 5 ? 'text-status-error' : 'text-status-warning'}
              trend={{
                value: errorRate,
                direction: errorRate > 5 ? 'down' : errorRate > 0 ? 'neutral' : 'up',
              }}
            />
          </div>

          {/* Agent Cards Grid */}
          <div
            className="grid grid-cols-1 md:grid-cols-4 gap-6"
            role="region"
            aria-label="Individual agent summaries"
          >
            {agentMetrics.agents.map((agent) => (
              <GlassCard key={agent.agent_name}>
                <div className="text-center">
                  <div className="w-12 h-12 rounded-xl bg-accent-primary/20 flex items-center justify-center mx-auto mb-3">
                    <CpuChipIcon className="h-6 w-6 text-accent-primary" aria-hidden="true" />
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

          {/* Detailed Agent Table */}
          <AgentTable
            agents={agentMetrics.agents}
            isLoading={isLoadingAgents}
          />
        </div>
      )}
    </div>
  )
}
