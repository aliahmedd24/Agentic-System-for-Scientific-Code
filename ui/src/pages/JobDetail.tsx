import { useEffect, useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import { PageHeader } from '@/components/layout/PageHeader'
import { GlassCard } from '@/components/ui/GlassCard'
import { Button } from '@/components/ui/Button'
import { StatusBadge } from '@/components/ui/StatusBadge'
import { LoadingSpinner } from '@/components/data-display/LoadingSpinner'
import { EmptyState } from '@/components/data-display/EmptyState'
import { PipelineProgress } from '@/components/pipeline/PipelineProgress'
import { PipelineSteps } from '@/components/pipeline/PipelineSteps'
import { EventLog } from '@/components/data-display/EventLog'
import { useJobWebSocket } from '@/hooks/useJobWebSocket'
import { useJobsStore } from '@/stores/jobsStore'
import { formatDate, formatRelativeTime } from '@/lib/formatters'
import { toast } from '@/components/ui/Toast'
import type { JobStatus } from '@/lib/constants'
import {
  ArrowTopRightOnSquareIcon,
  XMarkIcon,
  DocumentTextIcon,
  CodeBracketIcon,
  ClockIcon,
  ArrowPathIcon,
  SignalIcon,
  SignalSlashIcon,
  ExclamationTriangleIcon,
  ChartBarIcon,
  CubeTransparentIcon,
  DocumentChartBarIcon,
} from '@heroicons/react/24/outline'

export default function JobDetail() {
  const { jobId } = useParams<{ jobId: string }>()
  const { currentJob, fetchJob, cancelJob, isLoadingJob, error } = useJobsStore()

  // Use the custom WebSocket hook
  const {
    isConnected,
    isConnecting,
    hasError: wsHasError,
    events: wsEvents,
    currentStage: wsStage,
    currentProgress: wsProgress,
    reconnect,
    latestEvent,
  } = useJobWebSocket(jobId, {
    autoRefreshOnComplete: true,
    onComplete: () => {
      toast.success('Analysis Complete', 'Your analysis has finished successfully')
    },
    onError: (errorMessage) => {
      toast.error('Analysis Failed', errorMessage)
    },
  })

  // Initial fetch
  useEffect(() => {
    if (jobId) {
      fetchJob(jobId)
    }
  }, [jobId, fetchJob])

  const handleCancel = async () => {
    if (jobId && confirm('Are you sure you want to cancel this job?')) {
      try {
        await cancelJob(jobId)
        toast.success('Job Cancelled', 'The analysis job has been cancelled')
      } catch (err) {
        toast.error('Failed to Cancel', err instanceof Error ? err.message : 'Unknown error')
      }
    }
  }

  // Compute job status
  const jobStatus = useMemo(() => {
    if (!currentJob) return null
    return {
      isRunning: currentJob.status === 'running',
      isCompleted: currentJob.status === 'completed',
      isFailed: currentJob.status === 'failed',
      isPending: currentJob.status === 'pending',
    }
  }, [currentJob?.status])

  // Use WebSocket data for running jobs, otherwise use stored data
  const displayProgress = jobStatus?.isRunning ? wsProgress : (currentJob?.progress ?? 0)
  const displayStage = jobStatus?.isRunning ? wsStage : (currentJob?.stage ?? 'initialized')
  const displayEvents = jobStatus?.isRunning ? wsEvents : (currentJob?.events ?? [])

  // Connection status indicator
  const ConnectionIndicator = () => {
    if (!jobStatus?.isRunning) return null

    return (
      <div className="flex items-center gap-2">
        {isConnected && (
          <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-status-success/10 text-status-success">
            <SignalIcon className="h-4 w-4" />
            <span className="text-caption font-medium">Live</span>
          </div>
        )}
        {isConnecting && (
          <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-status-warning/10 text-status-warning">
            <ArrowPathIcon className="h-4 w-4 animate-spin" />
            <span className="text-caption font-medium">Connecting</span>
          </div>
        )}
        {wsHasError && (
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-status-error/10 text-status-error">
              <SignalSlashIcon className="h-4 w-4" />
              <span className="text-caption font-medium">Disconnected</span>
            </div>
            <Button variant="ghost" size="sm" onClick={reconnect}>
              Reconnect
            </Button>
          </div>
        )}
      </div>
    )
  }

  // Loading state
  if (isLoadingJob && !currentJob) {
    return (
      <div className="animate-in">
        <PageHeader title="Loading Job..." />
        <div className="flex items-center justify-center py-20">
          <LoadingSpinner size="xl" label="Loading job details..." />
        </div>
      </div>
    )
  }

  // Error state
  if (error && !currentJob) {
    return (
      <div className="animate-in">
        <PageHeader
          title="Job Not Found"
          breadcrumbs={[
            { label: 'Dashboard', href: '/' },
            { label: 'Jobs', href: '/jobs' },
            { label: 'Error' },
          ]}
        />
        <EmptyState
          icon={<ExclamationTriangleIcon className="h-16 w-16 text-status-error" />}
          title="Failed to Load Job"
          description={error}
          action={
            <div className="flex gap-3">
              <Link to="/jobs">
                <Button variant="secondary">Back to Jobs</Button>
              </Link>
              <Button onClick={() => jobId && fetchJob(jobId)}>
                Retry
              </Button>
            </div>
          }
        />
      </div>
    )
  }

  if (!currentJob) {
    return (
      <div className="animate-in">
        <PageHeader title="Job Not Found" />
        <EmptyState
          title="Job Not Found"
          description="The requested job could not be found"
          action={
            <Link to="/jobs">
              <Button>Back to Jobs</Button>
            </Link>
          }
        />
      </div>
    )
  }

  return (
    <div className="animate-in">
      <PageHeader
        title={`Job ${currentJob.job_id.slice(0, 8)}`}
        subtitle={latestEvent?.message}
        breadcrumbs={[
          { label: 'Dashboard', href: '/' },
          { label: 'Jobs', href: '/jobs' },
          { label: currentJob.job_id.slice(0, 8) },
        ]}
        actions={
          <div className="flex items-center gap-3">
            <ConnectionIndicator />
            {jobStatus?.isRunning && (
              <Button
                variant="danger"
                onClick={handleCancel}
                leftIcon={<XMarkIcon className="h-5 w-5" />}
              >
                Cancel
              </Button>
            )}
            {jobStatus?.isCompleted && (
              <>
                <Link to={`/jobs/${jobId}/results`}>
                  <Button variant="secondary" leftIcon={<ChartBarIcon className="h-5 w-5" />}>
                    Results
                  </Button>
                </Link>
                <Link to={`/jobs/${jobId}/graph`}>
                  <Button variant="secondary" leftIcon={<CubeTransparentIcon className="h-5 w-5" />}>
                    Graph
                  </Button>
                </Link>
                <Link to={`/jobs/${jobId}/reports`}>
                  <Button leftIcon={<DocumentChartBarIcon className="h-5 w-5" />}>
                    Report
                  </Button>
                </Link>
              </>
            )}
          </div>
        }
      />

      <div className="space-y-6">
        {/* Status Header */}
        <GlassCard>
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-6">
            <div className="flex items-center gap-4">
              <StatusBadge status={currentJob.status as JobStatus} size="lg" />
              <div>
                <h3 className="text-heading-3 text-text-primary">
                  {jobStatus?.isRunning && 'Analysis in Progress'}
                  {jobStatus?.isCompleted && 'Analysis Complete'}
                  {jobStatus?.isFailed && 'Analysis Failed'}
                  {jobStatus?.isPending && 'Analysis Pending'}
                </h3>
                <p className="text-body-sm text-text-secondary">
                  {displayStage.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                </p>
              </div>
            </div>
            {currentJob.created_at && (
              <div className="text-body-sm text-text-muted">
                Started {formatRelativeTime(currentJob.created_at)}
              </div>
            )}
          </div>

          {/* Progress Bar */}
          <PipelineProgress
            progress={displayProgress}
            status={
              jobStatus?.isCompleted
                ? 'completed'
                : jobStatus?.isFailed
                ? 'failed'
                : jobStatus?.isRunning
                ? 'running'
                : 'pending'
            }
            size="lg"
            animated={jobStatus?.isRunning}
            className="mb-8"
          />

          {/* Pipeline Steps */}
          <PipelineSteps
            currentStage={displayStage}
            progress={displayProgress}
            status={
              jobStatus?.isCompleted
                ? 'completed'
                : jobStatus?.isFailed
                ? 'failed'
                : jobStatus?.isRunning
                ? 'running'
                : 'pending'
            }
          />
        </GlassCard>

        {/* Job Information Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Paper Info */}
          <GlassCard>
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-lg bg-accent-primary/10">
                <DocumentTextIcon className="h-5 w-5 text-accent-primary" />
              </div>
              <div className="flex-1 min-w-0">
                <h4 className="text-body-sm font-medium text-text-secondary mb-1">
                  Paper Source
                </h4>
                <p className="text-body text-text-primary truncate" title={currentJob.paper_source ?? undefined}>
                  {currentJob.paper_source || 'Not specified'}
                </p>
                {currentJob.paper_source?.startsWith('http') && (
                  <a
                    href={currentJob.paper_source}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 text-caption text-accent-secondary hover:underline mt-1"
                  >
                    Open Paper <ArrowTopRightOnSquareIcon className="h-3 w-3" />
                  </a>
                )}
              </div>
            </div>
          </GlassCard>

          {/* Repository Info */}
          <GlassCard>
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-lg bg-purple-500/10">
                <CodeBracketIcon className="h-5 w-5 text-purple-400" />
              </div>
              <div className="flex-1 min-w-0">
                <h4 className="text-body-sm font-medium text-text-secondary mb-1">
                  Repository
                </h4>
                <p className="text-body text-text-primary truncate" title={currentJob.repo_url ?? undefined}>
                  {currentJob.repo_url
                    ? currentJob.repo_url.replace('https://github.com/', '')
                    : 'Not specified'}
                </p>
                {currentJob.repo_url && (
                  <a
                    href={currentJob.repo_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 text-caption text-accent-secondary hover:underline mt-1"
                  >
                    View on GitHub <ArrowTopRightOnSquareIcon className="h-3 w-3" />
                  </a>
                )}
              </div>
            </div>
          </GlassCard>

          {/* Timing Info */}
          <GlassCard>
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-lg bg-cyan-500/10">
                <ClockIcon className="h-5 w-5 text-cyan-400" />
              </div>
              <div className="flex-1">
                <h4 className="text-body-sm font-medium text-text-secondary mb-1">
                  Timing
                </h4>
                <div className="space-y-1">
                  <div className="flex justify-between text-body-sm">
                    <span className="text-text-muted">Created:</span>
                    <span className="text-text-primary">{formatDate(currentJob.created_at)}</span>
                  </div>
                  <div className="flex justify-between text-body-sm">
                    <span className="text-text-muted">Updated:</span>
                    <span className="text-text-primary">{formatDate(currentJob.updated_at)}</span>
                  </div>
                </div>
              </div>
            </div>
          </GlassCard>
        </div>

        {/* Configuration Details */}
        <GlassCard title="Configuration">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-3 rounded-lg bg-bg-tertiary/50">
              <div className="text-caption text-text-muted mb-1">LLM Provider</div>
              <div className="text-body-sm text-text-primary font-medium capitalize">
                {currentJob.llm_provider || 'Default'}
              </div>
            </div>
            <div className="p-3 rounded-lg bg-bg-tertiary/50">
              <div className="text-caption text-text-muted mb-1">Auto Execute</div>
              <div className="text-body-sm text-text-primary font-medium">
                {currentJob.auto_execute ? 'Enabled' : 'Disabled'}
              </div>
            </div>
            <div className="p-3 rounded-lg bg-bg-tertiary/50">
              <div className="text-caption text-text-muted mb-1">Events</div>
              <div className="text-body-sm text-text-primary font-medium">
                {displayEvents.length} logged
              </div>
            </div>
            <div className="p-3 rounded-lg bg-bg-tertiary/50">
              <div className="text-caption text-text-muted mb-1">Job ID</div>
              <div className="text-body-sm text-text-primary font-mono truncate" title={currentJob.job_id}>
                {currentJob.job_id}
              </div>
            </div>
          </div>
        </GlassCard>

        {/* Event Log */}
        <GlassCard
          title="Event Log"
          subtitle="Real-time updates from the analysis pipeline"
          headerActions={
            jobStatus?.isRunning ? (
              <div className="flex items-center gap-2">
                <PipelineSteps
                  currentStage={displayStage}
                  progress={displayProgress}
                  status="running"
                  compact
                />
              </div>
            ) : undefined
          }
        >
          <EventLog
            events={displayEvents}
            maxHeight={400}
            autoScroll={jobStatus?.isRunning}
            showTimestamp
            showStage
          />
        </GlassCard>

        {/* Error Details (if failed) */}
        {jobStatus?.isFailed && (currentJob.error_message || currentJob.error) && (
          <GlassCard className="border-status-error/30">
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-lg bg-status-error/10">
                <ExclamationTriangleIcon className="h-5 w-5 text-status-error" />
              </div>
              <div className="flex-1">
                <h4 className="text-body font-medium text-status-error mb-2">Error Details</h4>
                <p className="text-body-sm text-text-secondary font-mono whitespace-pre-wrap">
                  {currentJob.error_message || currentJob.error}
                </p>
              </div>
            </div>
          </GlassCard>
        )}

        {/* Quick Actions for Completed Jobs */}
        {jobStatus?.isCompleted && (
          <GlassCard>
            <h4 className="text-body font-medium text-text-primary mb-4">Next Steps</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Link
                to={`/jobs/${jobId}/results`}
                className="flex items-center gap-3 p-4 rounded-lg border border-border hover:border-accent-primary/50 hover:bg-accent-primary/5 transition-all group"
              >
                <div className="p-2 rounded-lg bg-accent-primary/10 group-hover:bg-accent-primary/20 transition-colors">
                  <ChartBarIcon className="h-5 w-5 text-accent-primary" />
                </div>
                <div>
                  <div className="text-body-sm font-medium text-text-primary">View Results</div>
                  <div className="text-caption text-text-muted">Explore concept mappings</div>
                </div>
              </Link>

              <Link
                to={`/jobs/${jobId}/graph`}
                className="flex items-center gap-3 p-4 rounded-lg border border-border hover:border-purple-500/50 hover:bg-purple-500/5 transition-all group"
              >
                <div className="p-2 rounded-lg bg-purple-500/10 group-hover:bg-purple-500/20 transition-colors">
                  <CubeTransparentIcon className="h-5 w-5 text-purple-400" />
                </div>
                <div>
                  <div className="text-body-sm font-medium text-text-primary">Knowledge Graph</div>
                  <div className="text-caption text-text-muted">Visualize relationships</div>
                </div>
              </Link>

              <Link
                to={`/jobs/${jobId}/reports`}
                className="flex items-center gap-3 p-4 rounded-lg border border-border hover:border-cyan-500/50 hover:bg-cyan-500/5 transition-all group"
              >
                <div className="p-2 rounded-lg bg-cyan-500/10 group-hover:bg-cyan-500/20 transition-colors">
                  <DocumentChartBarIcon className="h-5 w-5 text-cyan-400" />
                </div>
                <div>
                  <div className="text-body-sm font-medium text-text-primary">Download Report</div>
                  <div className="text-caption text-text-muted">Get detailed analysis</div>
                </div>
              </Link>
            </div>
          </GlassCard>
        )}
      </div>
    </div>
  )
}
