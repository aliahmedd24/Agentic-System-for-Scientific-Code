import { create } from 'zustand'
import type { MetricsSummary, AgentMetrics, PipelineMetrics, HealthStatus } from '@/api/types'
import * as api from '@/api/endpoints'

interface MetricsState {
  // Data
  summary: MetricsSummary | null
  agentMetrics: AgentMetrics | null
  pipelineMetrics: PipelineMetrics | null
  healthStatus: HealthStatus | null

  // Loading states
  isLoadingSummary: boolean
  isLoadingAgents: boolean
  isLoadingPipeline: boolean
  isLoadingHealth: boolean

  // Error
  error: string | null

  // Actions
  fetchSummary: () => Promise<void>
  fetchAgentMetrics: () => Promise<void>
  fetchPipelineMetrics: () => Promise<void>
  fetchHealth: () => Promise<void>
  fetchAll: () => Promise<void>
  clearError: () => void
}

export const useMetricsStore = create<MetricsState>((set) => ({
  // Initial state
  summary: null,
  agentMetrics: null,
  pipelineMetrics: null,
  healthStatus: null,
  isLoadingSummary: false,
  isLoadingAgents: false,
  isLoadingPipeline: false,
  isLoadingHealth: false,
  error: null,

  // Fetch summary metrics
  fetchSummary: async () => {
    set({ isLoadingSummary: true, error: null })
    try {
      const summary = await api.getMetrics()
      set({ summary, isLoadingSummary: false })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch metrics'
      set({ error: message, isLoadingSummary: false })
    }
  },

  // Fetch agent metrics
  fetchAgentMetrics: async () => {
    set({ isLoadingAgents: true, error: null })
    try {
      const agentMetrics = await api.getAgentMetrics()
      set({ agentMetrics, isLoadingAgents: false })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch agent metrics'
      set({ error: message, isLoadingAgents: false })
    }
  },

  // Fetch pipeline metrics
  fetchPipelineMetrics: async () => {
    set({ isLoadingPipeline: true, error: null })
    try {
      const pipelineMetrics = await api.getPipelineMetrics()
      set({ pipelineMetrics, isLoadingPipeline: false })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch pipeline metrics'
      set({ error: message, isLoadingPipeline: false })
    }
  },

  // Fetch health status
  fetchHealth: async () => {
    set({ isLoadingHealth: true, error: null })
    try {
      const healthStatus = await api.getHealth()
      set({ healthStatus, isLoadingHealth: false })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch health status'
      set({ error: message, isLoadingHealth: false })
    }
  },

  // Fetch all metrics
  fetchAll: async () => {
    const store = useMetricsStore.getState()
    await Promise.all([
      store.fetchSummary(),
      store.fetchAgentMetrics(),
      store.fetchPipelineMetrics(),
      store.fetchHealth(),
    ])
  },

  // Clear error
  clearError: () => set({ error: null }),
}))

export default useMetricsStore
