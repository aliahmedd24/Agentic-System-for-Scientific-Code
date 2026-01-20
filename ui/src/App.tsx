import { Suspense, lazy } from 'react'
import { Routes, Route } from 'react-router-dom'
import { AppShell } from '@/components/layout/AppShell'
import { ToastContainer } from '@/components/ui/Toast'
import { LoadingSpinner } from '@/components/data-display/LoadingSpinner'

// Lazy load pages for code splitting
const Dashboard = lazy(() => import('@/pages/Dashboard'))
const NewAnalysis = lazy(() => import('@/pages/NewAnalysis'))
const JobList = lazy(() => import('@/pages/JobList'))
const JobDetail = lazy(() => import('@/pages/JobDetail'))
const Results = lazy(() => import('@/pages/Results'))
const KnowledgeGraph = lazy(() => import('@/pages/KnowledgeGraph'))
const Reports = lazy(() => import('@/pages/Reports'))
const Metrics = lazy(() => import('@/pages/Metrics'))
const MetricsAgents = lazy(() => import('@/pages/MetricsAgents'))
const MetricsPipeline = lazy(() => import('@/pages/MetricsPipeline'))
const Settings = lazy(() => import('@/pages/Settings'))

// Loading fallback component
function PageLoader() {
  return (
    <div className="flex items-center justify-center min-h-[400px]">
      <LoadingSpinner size="xl" label="Loading..." />
    </div>
  )
}

function App() {
  return (
    <>
      <AppShell>
        <Suspense fallback={<PageLoader />}>
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
        </Suspense>
      </AppShell>
      <ToastContainer />
    </>
  )
}

export default App
