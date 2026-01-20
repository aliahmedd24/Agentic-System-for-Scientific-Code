import { useState, useEffect } from 'react'
import {
  ArrowDownTrayIcon,
  ClipboardDocumentIcon,
  CheckIcon,
  ArrowTopRightOnSquareIcon,
} from '@heroicons/react/24/outline'
import { cn } from '@/lib/cn'
import { GlassCard } from '@/components/ui/GlassCard'
import { Button } from '@/components/ui/Button'
import { CodeBlock } from '@/components/code/CodeBlock'
import { LoadingSpinner } from '@/components/data-display/LoadingSpinner'
import { EmptyState } from '@/components/data-display/EmptyState'
import { useToastStore } from '@/components/ui/Toast'
import { getReportUrl, downloadReport } from '@/api/endpoints'
import type { ReportFormat } from './ReportGenerator'

interface ReportPreviewProps {
  jobId: string
  format: ReportFormat
  className?: string
}

export function ReportPreview({ jobId, format, className }: ReportPreviewProps) {
  const [content, setContent] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const addToast = useToastStore((s) => s.addToast)

  // Fetch content for markdown/json preview
  useEffect(() => {
    if (format === 'html') {
      setContent(null)
      return
    }

    const fetchContent = async () => {
      setIsLoading(true)
      setError(null)
      try {
        const blob = await downloadReport(jobId, format)
        const text = await blob.text()
        setContent(text)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load report')
      } finally {
        setIsLoading(false)
      }
    }

    fetchContent()
  }, [jobId, format])

  const handleDownload = () => {
    const url = getReportUrl(jobId, format)
    window.open(url, '_blank')
  }

  const handleCopy = async () => {
    if (!content) return
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      addToast({
        type: 'success',
        title: 'Copied to clipboard',
      })
      setTimeout(() => setCopied(false), 2000)
    } catch {
      addToast({
        type: 'error',
        title: 'Failed to copy',
        message: 'Could not copy content to clipboard',
      })
    }
  }

  const handleOpenInNewTab = () => {
    const url = getReportUrl(jobId, format)
    window.open(url, '_blank')
  }

  return (
    <div className={cn('space-y-4', className)}>
      {/* Action Buttons */}
      <div className="flex items-center justify-between">
        <h3 className="text-heading-3 text-text-primary">
          Preview: {format.toUpperCase()}
        </h3>
        <div className="flex items-center gap-2">
          {format !== 'html' && content && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCopy}
              leftIcon={
                copied ? (
                  <CheckIcon className="h-4 w-4 text-status-success" />
                ) : (
                  <ClipboardDocumentIcon className="h-4 w-4" />
                )
              }
            >
              {copied ? 'Copied' : 'Copy'}
            </Button>
          )}
          {format === 'html' && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleOpenInNewTab}
              leftIcon={<ArrowTopRightOnSquareIcon className="h-4 w-4" />}
            >
              Open in New Tab
            </Button>
          )}
          <Button
            variant="secondary"
            size="sm"
            onClick={handleDownload}
            leftIcon={<ArrowDownTrayIcon className="h-4 w-4" />}
          >
            Download
          </Button>
        </div>
      </div>

      {/* Preview Content */}
      <GlassCard noPadding className="overflow-hidden">
        {format === 'html' ? (
          // HTML iframe preview
          <div className="relative" style={{ height: '600px' }}>
            <iframe
              src={getReportUrl(jobId, 'html')}
              className="w-full h-full border-0"
              title="Report Preview"
            />
          </div>
        ) : isLoading ? (
          // Loading state
          <div className="flex items-center justify-center py-20">
            <LoadingSpinner size="lg" />
          </div>
        ) : error ? (
          // Error state
          <div className="p-6">
            <EmptyState
              icon="error"
              title="Failed to load preview"
              description={error}
              action={{
                label: 'Try Again',
                onClick: () => window.location.reload(),
              }}
            />
          </div>
        ) : content ? (
          // Content preview
          format === 'json' ? (
            <CodeBlock
              code={content}
              language="json"
              maxHeight="600px"
              copyButton={false}
            />
          ) : (
            // Markdown preview - show as formatted text
            <div className="p-6 prose prose-invert max-w-none">
              <pre className="whitespace-pre-wrap text-body-sm text-text-secondary font-mono bg-transparent p-0 m-0">
                {content}
              </pre>
            </div>
          )
        ) : (
          // Empty state
          <div className="p-6">
            <EmptyState
              icon="document"
              title="No preview available"
              description="Generate a report to see the preview"
            />
          </div>
        )}
      </GlassCard>
    </div>
  )
}

export default ReportPreview
