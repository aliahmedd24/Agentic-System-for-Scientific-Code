import { useParams, Link } from 'react-router-dom'
import { PageHeader } from '@/components/layout/PageHeader'
import { GlassCard } from '@/components/ui/GlassCard'
import { Button } from '@/components/ui/Button'
import { Tabs, TabPanel } from '@/components/ui/Tabs'
import { LoadingSpinner } from '@/components/data-display/LoadingSpinner'
import { EmptyState } from '@/components/data-display/EmptyState'
import {
  MappingsList,
  UnmappedSection,
  PaperSummary,
  RepoSummary,
  CodeExecutionResults,
  ConceptMappingCard,
} from '@/components/results'
import { useJobResults } from '@/hooks/useJobResults'
import { formatConfidence } from '@/lib/formatters'
import {
  DocumentTextIcon,
  CodeBracketIcon,
  ArrowsRightLeftIcon,
  BeakerIcon,
  ChartBarIcon,
  ArrowRightIcon,
} from '@heroicons/react/24/outline'

const tabItems = [
  { label: 'Overview', icon: ChartBarIcon },
  { label: 'Mappings', icon: ArrowsRightLeftIcon },
  { label: 'Paper', icon: DocumentTextIcon },
  { label: 'Repository', icon: CodeBracketIcon },
  { label: 'Execution', icon: BeakerIcon },
]

export default function Results() {
  const { jobId } = useParams<{ jobId: string }>()
  const { data: result, isLoading, error } = useJobResults(jobId)

  if (isLoading) {
    return (
      <div className="animate-in">
        <PageHeader title="Loading Results..." />
        <div className="flex items-center justify-center py-20">
          <LoadingSpinner size="lg" />
        </div>
      </div>
    )
  }

  if (error || !result) {
    return (
      <div className="animate-in">
        <PageHeader title="Results" />
        <EmptyState
          icon="error"
          title="Failed to load results"
          description={error?.message || 'Could not load analysis results'}
          action={{
            label: 'Go to Jobs',
            onClick: () => window.location.href = '/jobs',
          }}
        />
      </div>
    )
  }

  const { paper_data, repo_data, mappings, unmapped_concepts, unmapped_code, code_results } = result

  // Calculate stats
  const avgConfidence = mappings.length > 0
    ? mappings.reduce((sum, m) => sum + m.confidence, 0) / mappings.length
    : 0
  const highConfidenceMappings = mappings.filter(m => m.confidence >= 0.8).length
  const passRate = code_results.summary.total_tests > 0
    ? code_results.summary.passed / code_results.summary.total_tests
    : 0

  // Get top 5 mappings by confidence
  const topMappings = [...mappings]
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 5)

  return (
    <div className="animate-in">
      <PageHeader
        title="Analysis Results"
        breadcrumbs={[
          { label: 'Dashboard', href: '/' },
          { label: 'Jobs', href: '/jobs' },
          { label: jobId?.slice(0, 8) || '', href: `/jobs/${jobId}` },
          { label: 'Results' },
        ]}
        actions={
          <div className="flex items-center gap-3">
            <Link to={`/jobs/${jobId}/graph`}>
              <Button variant="secondary">Knowledge Graph</Button>
            </Link>
            <Link to={`/jobs/${jobId}/reports`}>
              <Button>Download Report</Button>
            </Link>
          </div>
        }
      />

      <Tabs items={tabItems} variant="default">
        {/* Overview Tab */}
        <TabPanel>
          <div className="space-y-6">
            {/* Summary Stats */}
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <GlassCard className="text-center">
                <p className="text-caption text-text-muted mb-1">Total Mappings</p>
                <p className="text-heading-1 text-text-primary">{mappings.length}</p>
              </GlassCard>

              <GlassCard className="text-center">
                <p className="text-caption text-text-muted mb-1">Avg Confidence</p>
                <p className="text-heading-1 text-accent-secondary">
                  {formatConfidence(avgConfidence)}
                </p>
              </GlassCard>

              <GlassCard className="text-center">
                <p className="text-caption text-text-muted mb-1">High Confidence</p>
                <p className="text-heading-1 text-status-success">{highConfidenceMappings}</p>
              </GlassCard>

              <GlassCard className="text-center">
                <p className="text-caption text-text-muted mb-1">Test Pass Rate</p>
                <p className="text-heading-1 text-status-info">
                  {formatConfidence(passRate)}
                </p>
              </GlassCard>
            </div>

            {/* Quick Info Cards */}
            <div className="grid gap-4 lg:grid-cols-2">
              <GlassCard
                title={paper_data.title}
                subtitle={`${paper_data.authors.slice(0, 3).join(', ')}${paper_data.authors.length > 3 ? ' et al.' : ''}`}
                icon={<DocumentTextIcon className="h-5 w-5 text-graph-paper" />}
              >
                <p className="text-body-sm text-text-secondary line-clamp-3 mb-4">
                  {paper_data.abstract}
                </p>
                <div className="flex items-center gap-4 text-caption text-text-muted">
                  <span>{paper_data.key_concepts.length} concepts</span>
                  <span>{paper_data.algorithms.length} algorithms</span>
                </div>
              </GlassCard>

              <GlassCard
                title={repo_data.name}
                subtitle={repo_data.url.replace('https://github.com/', '')}
                icon={<CodeBracketIcon className="h-5 w-5 text-graph-repository" />}
              >
                <p className="text-body-sm text-text-secondary line-clamp-3 mb-4">
                  {repo_data.overview.purpose}
                </p>
                <div className="flex items-center gap-4 text-caption text-text-muted">
                  <span>{repo_data.stats.code_files} files</span>
                  <span>{repo_data.stats.classes} classes</span>
                  <span>{repo_data.stats.functions} functions</span>
                </div>
              </GlassCard>
            </div>

            {/* Top Mappings */}
            {topMappings.length > 0 && (
              <div>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-heading-3 text-text-primary">
                    Top Mappings by Confidence
                  </h3>
                  <Link
                    to="#"
                    onClick={(e) => { e.preventDefault(); /* switch to mappings tab */ }}
                    className="flex items-center gap-1 text-body-sm text-accent-secondary hover:text-accent-primary transition-colors"
                  >
                    View all
                    <ArrowRightIcon className="h-4 w-4" />
                  </Link>
                </div>
                <div className="grid gap-3">
                  {topMappings.map((mapping, idx) => (
                    <ConceptMappingCard key={idx} mapping={mapping} />
                  ))}
                </div>
              </div>
            )}

            {/* Unmapped Items Summary */}
            {(unmapped_concepts.length > 0 || unmapped_code.length > 0) && (
              <GlassCard
                title="Unmapped Items"
                subtitle={`${unmapped_concepts.length + unmapped_code.length} items could not be mapped`}
              >
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="p-4 rounded-lg bg-bg-tertiary/50">
                    <p className="text-body-sm text-text-primary font-medium mb-1">
                      Unmapped Concepts
                    </p>
                    <p className="text-heading-2 text-status-warning">
                      {unmapped_concepts.length}
                    </p>
                  </div>
                  <div className="p-4 rounded-lg bg-bg-tertiary/50">
                    <p className="text-body-sm text-text-primary font-medium mb-1">
                      Unmapped Code Elements
                    </p>
                    <p className="text-heading-2 text-status-warning">
                      {unmapped_code.length}
                    </p>
                  </div>
                </div>
              </GlassCard>
            )}
          </div>
        </TabPanel>

        {/* Mappings Tab */}
        <TabPanel>
          <div className="space-y-6">
            <MappingsList mappings={mappings} />
            <UnmappedSection concepts={unmapped_concepts} code={unmapped_code} />
          </div>
        </TabPanel>

        {/* Paper Tab */}
        <TabPanel>
          <PaperSummary paper={paper_data} />
        </TabPanel>

        {/* Repository Tab */}
        <TabPanel>
          <RepoSummary repo={repo_data} />
        </TabPanel>

        {/* Execution Tab */}
        <TabPanel>
          <CodeExecutionResults codeResults={code_results} />
        </TabPanel>
      </Tabs>
    </div>
  )
}
