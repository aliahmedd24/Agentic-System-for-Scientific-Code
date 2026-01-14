import { Routes, Route } from 'react-router-dom'
import { AppShell } from '@/components/layout/AppShell'
import { ToastContainer } from '@/components/ui/Toast'

// Pages
import Dashboard from '@/pages/Dashboard'
import NewAnalysis from '@/pages/NewAnalysis'
import JobList from '@/pages/JobList'
import JobDetail from '@/pages/JobDetail'
import Results from '@/pages/Results'
import KnowledgeGraph from '@/pages/KnowledgeGraph'
import Reports from '@/pages/Reports'
import Metrics from '@/pages/Metrics'
import MetricsAgents from '@/pages/MetricsAgents'
import MetricsPipeline from '@/pages/MetricsPipeline'
import Settings from '@/pages/Settings'

function App() {
  return (
    <>
      <AppShell>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analyze" element={<NewAnalysis />} />
          <Route path="/jobs" element={<JobList />} />
          <Route path="/jobs/:jobId" element={<JobDetail />} />
          <Route path="/jobs/:jobId/results" element={<Results />} />
          <Route path="/jobs/:jobId/graph" element={<KnowledgeGraph />} />
          <Route path="/jobs/:jobId/reports" element={<Reports />} />
          <Route path="/metrics" element={<Metrics />} />
          <Route path="/metrics/agents" element={<MetricsAgents />} />
          <Route path="/metrics/pipeline" element={<MetricsPipeline />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </AppShell>
      <ToastContainer />
    </>
  )
}

export default App
