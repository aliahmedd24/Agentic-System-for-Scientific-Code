import { Link } from 'react-router-dom'
import {
  EyeIcon,
  XMarkIcon,
  TrashIcon,
  DocumentTextIcon,
  FolderIcon,
} from '@heroicons/react/24/outline'
import { cn } from '@/lib/cn'
import { GlassCard } from '@/components/ui/GlassCard'
import { Button } from '@/components/ui/Button'
import { StatusBadge } from '@/components/ui/StatusBadge'
import { formatRelativeTime, formatJobId } from '@/lib/formatters'
import type { Job } from '@/api/types'

interface JobCardProps {
  job: Job
  onCancel?: (jobId: string) => void
  onDelete?: (jobId: string) => void
  className?: string
}

export function JobCard({ job, onCancel, onDelete, className }: JobCardProps) {
  const isRunning = job.status === 'running' || job.status === 'pending'
  const isFinished = job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled'

  return (
    <GlassCard
      variant="interactive"
      noPadding
      className={cn('overflow-hidden', className)}
    >
      <div className="p-4">
        {/* Header */}
        <div className="flex items-start justify-between gap-4 mb-3">
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-10 h-10 rounded-lg bg-accent-primary/20 flex items-center justify-center shrink-0">
              <DocumentTextIcon className="h-5 w-5 text-accent-primary" />
            </div>
            <div className="min-w-0">
              <h4 className="text-body font-medium text-text-primary font-mono">
                {formatJobId(job.job_id)}
              </h4>
              <p className="text-caption text-text-muted">
                {formatRelativeTime(job.created_at)}
              </p>
            </div>
          </div>
          <StatusBadge status={job.status} size="sm" />
        </div>

        {/* Paper source */}
        {job.paper_source && (
          <div className="flex items-start gap-2 mb-2">
            <DocumentTextIcon className="h-4 w-4 text-text-muted shrink-0 mt-0.5" />
            <p className="text-body-sm text-text-secondary truncate">
              {job.paper_source}
            </p>
          </div>
        )}

        {/* Repo URL */}
        {job.repo_url && (
          <div className="flex items-start gap-2 mb-3">
            <FolderIcon className="h-4 w-4 text-text-muted shrink-0 mt-0.5" />
            <p className="text-body-sm text-text-muted truncate font-mono">
              {job.repo_url.replace('https://github.com/', '')}
            </p>
          </div>
        )}

        {/* Progress bar */}
        {isRunning && (
          <div className="mb-3">
            <div className="flex items-center justify-between mb-1">
              <span className="text-caption text-text-muted capitalize">
                {job.stage || 'Initializing'}
              </span>
              <span className="text-caption text-text-muted">
                {Math.round(job.progress)}%
              </span>
            </div>
            <div className="h-1.5 bg-bg-tertiary rounded-full overflow-hidden">
              <div
                className={cn(
                  'h-full rounded-full transition-all duration-500',
                  job.status === 'running'
                    ? 'bg-accent-primary'
                    : 'bg-status-warning'
                )}
                style={{ width: `${job.progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Error message */}
        {job.error && (
          <div className="mb-3 p-2 rounded bg-status-error/10 border border-status-error/20">
            <p className="text-caption text-status-error line-clamp-2">
              {job.error}
            </p>
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center gap-2 pt-2 border-t border-border">
          <Link to={`/jobs/${job.job_id}`} className="flex-1">
            <Button variant="ghost" size="sm" className="w-full" leftIcon={<EyeIcon className="h-4 w-4" />}>
              View
            </Button>
          </Link>

          {isRunning && onCancel && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onCancel(job.job_id)}
              leftIcon={<XMarkIcon className="h-4 w-4" />}
              className="text-status-warning hover:text-status-warning"
            >
              Cancel
            </Button>
          )}

          {isFinished && onDelete && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onDelete(job.job_id)}
              leftIcon={<TrashIcon className="h-4 w-4" />}
              className="text-status-error hover:text-status-error"
            >
              Delete
            </Button>
          )}
        </div>
      </div>
    </GlassCard>
  )
}

export default JobCard
