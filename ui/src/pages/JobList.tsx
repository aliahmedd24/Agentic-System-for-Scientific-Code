import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { PageHeader } from '@/components/layout/PageHeader'
import { GlassCard } from '@/components/ui/GlassCard'
import { Button } from '@/components/ui/Button'
import { StatusBadge } from '@/components/ui/StatusBadge'
import { useJobsStore } from '@/stores/jobsStore'
import { formatDate, formatJobId } from '@/lib/formatters'
import type { JobStatus } from '@/lib/constants'
import {
  PlusCircleIcon,
  ClipboardDocumentListIcon,
  EyeIcon,
  TrashIcon,
} from '@heroicons/react/24/outline'

export default function JobList() {
  const { jobs, fetchJobs, cancelJob, isLoadingJobs } = useJobsStore()

  useEffect(() => {
    fetchJobs(50)
  }, [])

  const handleCancel = async (jobId: string, e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (confirm('Are you sure you want to cancel this job?')) {
      await cancelJob(jobId)
    }
  }

  return (
    <div className="animate-in">
      <PageHeader
        title="Job History"
        subtitle="View and manage all analysis jobs"
        actions={
          <Link to="/analyze">
            <Button leftIcon={<PlusCircleIcon className="h-5 w-5" />}>
              New Analysis
            </Button>
          </Link>
        }
      />

      <GlassCard noPadding>
        {isLoadingJobs ? (
          <div className="py-16 text-center text-text-muted">Loading jobs...</div>
        ) : jobs.length === 0 ? (
          <div className="py-16 text-center">
            <ClipboardDocumentListIcon className="h-16 w-16 mx-auto text-text-muted mb-4" />
            <h3 className="text-heading-3 text-text-primary mb-2">No jobs yet</h3>
            <p className="text-body text-text-secondary mb-6">
              Start your first analysis to see jobs here
            </p>
            <Link to="/analyze">
              <Button leftIcon={<PlusCircleIcon className="h-5 w-5" />}>
                New Analysis
              </Button>
            </Link>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border">
                  <th className="px-6 py-4 text-left text-body-sm font-semibold text-text-secondary">Job ID</th>
                  <th className="px-6 py-4 text-left text-body-sm font-semibold text-text-secondary">Paper</th>
                  <th className="px-6 py-4 text-left text-body-sm font-semibold text-text-secondary">Repository</th>
                  <th className="px-6 py-4 text-left text-body-sm font-semibold text-text-secondary">Status</th>
                  <th className="px-6 py-4 text-left text-body-sm font-semibold text-text-secondary">Progress</th>
                  <th className="px-6 py-4 text-left text-body-sm font-semibold text-text-secondary">Created</th>
                  <th className="px-6 py-4 text-right text-body-sm font-semibold text-text-secondary">Actions</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr
                    key={job.job_id}
                    className="border-b border-border/50 hover:bg-bg-tertiary/30 transition-colors"
                  >
                    <td className="px-6 py-4">
                      <span className="font-mono text-body-sm text-accent-secondary">
                        {formatJobId(job.job_id)}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-body-sm text-text-primary truncate max-w-[200px] block">
                        {job.paper_source || '-'}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-body-sm text-text-secondary truncate max-w-[200px] block">
                        {job.repo_url || '-'}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <StatusBadge status={job.status as JobStatus} size="sm" />
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <div className="w-24 h-2 bg-bg-tertiary rounded-full overflow-hidden">
                          <div
                            className="h-full bg-accent-primary rounded-full transition-all duration-300"
                            style={{ width: `${job.progress}%` }}
                          />
                        </div>
                        <span className="text-caption text-text-muted">{job.progress}%</span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-body-sm text-text-muted">
                        {formatDate(job.created_at)}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center justify-end gap-2">
                        <Link to={`/jobs/${job.job_id}`}>
                          <Button variant="icon" size="sm" title="View Details">
                            <EyeIcon className="h-4 w-4" />
                          </Button>
                        </Link>
                        {job.status === 'running' && (
                          <Button
                            variant="icon"
                            size="sm"
                            onClick={(e) => handleCancel(job.job_id, e)}
                            title="Cancel Job"
                          >
                            <TrashIcon className="h-4 w-4 text-status-error" />
                          </Button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </GlassCard>
    </div>
  )
}
