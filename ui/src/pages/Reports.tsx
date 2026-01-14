import { useParams, Link } from 'react-router-dom'
import { PageHeader } from '@/components/layout/PageHeader'
import { GlassCard } from '@/components/ui/GlassCard'
import { Button } from '@/components/ui/Button'
import { getReportUrl } from '@/api/endpoints'
import {
  DocumentTextIcon,
  CodeBracketIcon,
  DocumentArrowDownIcon,
} from '@heroicons/react/24/outline'

const reportFormats = [
  {
    id: 'html',
    name: 'HTML Report',
    description: 'Full interactive report with knowledge graph visualization',
    icon: DocumentTextIcon,
    extension: '.html',
  },
  {
    id: 'json',
    name: 'JSON Export',
    description: 'Structured data for programmatic access',
    icon: CodeBracketIcon,
    extension: '.json',
  },
  {
    id: 'markdown',
    name: 'Markdown Report',
    description: 'Readable text format for documentation',
    icon: DocumentTextIcon,
    extension: '.md',
  },
] as const

export default function Reports() {
  const { jobId } = useParams<{ jobId: string }>()

  const handleDownload = (format: 'html' | 'json' | 'markdown') => {
    if (!jobId) return
    const url = getReportUrl(jobId, format)
    window.open(url, '_blank')
  }

  return (
    <div className="animate-in">
      <PageHeader
        title="Download Reports"
        breadcrumbs={[
          { label: 'Dashboard', href: '/' },
          { label: 'Jobs', href: '/jobs' },
          { label: jobId?.slice(0, 8) || '', href: `/jobs/${jobId}` },
          { label: 'Reports' },
        ]}
        actions={
          <div className="flex items-center gap-3">
            <Link to={`/jobs/${jobId}/results`}>
              <Button variant="secondary">View Results</Button>
            </Link>
            <Link to={`/jobs/${jobId}/graph`}>
              <Button variant="secondary">Knowledge Graph</Button>
            </Link>
          </div>
        }
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {reportFormats.map((format) => (
          <GlassCard
            key={format.id}
            variant="interactive"
            className="text-center"
            onClick={() => handleDownload(format.id)}
          >
            <div className="flex flex-col items-center py-4">
              <div className="w-16 h-16 rounded-2xl bg-accent-primary/20 flex items-center justify-center mb-4">
                <format.icon className="h-8 w-8 text-accent-primary" />
              </div>
              <h3 className="text-heading-3 text-text-primary mb-2">{format.name}</h3>
              <p className="text-body-sm text-text-secondary mb-4">{format.description}</p>
              <Button
                variant="secondary"
                leftIcon={<DocumentArrowDownIcon className="h-5 w-5" />}
              >
                Download {format.extension}
              </Button>
            </div>
          </GlassCard>
        ))}
      </div>

      {/* Preview Section */}
      <div className="mt-8">
        <GlassCard title="Report Preview" subtitle="HTML report preview">
          <div className="aspect-video bg-bg-tertiary rounded-lg flex items-center justify-center">
            {jobId ? (
              <iframe
                src={getReportUrl(jobId, 'html')}
                className="w-full h-full rounded-lg"
                title="Report Preview"
              />
            ) : (
              <p className="text-text-muted">No report available</p>
            )}
          </div>
        </GlassCard>
      </div>
    </div>
  )
}
