import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { Tab } from '@headlessui/react'
import { PageHeader } from '@/components/layout/PageHeader'
import { GlassCard } from '@/components/ui/GlassCard'
import { Button } from '@/components/ui/Button'
import { useJobsStore } from '@/stores/jobsStore'
import { formatConfidence } from '@/lib/formatters'
import { cn } from '@/lib/cn'
import {
  DocumentTextIcon,
  CodeBracketIcon,
  ArrowsRightLeftIcon,
  BeakerIcon,
  CheckCircleIcon,
  XCircleIcon,
} from '@heroicons/react/24/outline'

const tabs = [
  { name: 'Paper', icon: DocumentTextIcon },
  { name: 'Repository', icon: CodeBracketIcon },
  { name: 'Mappings', icon: ArrowsRightLeftIcon },
  { name: 'Tests', icon: BeakerIcon },
]

export default function Results() {
  const { jobId } = useParams<{ jobId: string }>()
  const { currentResult, fetchResult, isLoadingResult } = useJobsStore()

  useEffect(() => {
    if (jobId) {
      fetchResult(jobId)
    }
  }, [jobId])

  if (isLoadingResult || !currentResult) {
    return (
      <div className="animate-in">
        <PageHeader title="Loading Results..." />
        <div className="py-16 text-center text-text-muted">Loading analysis results...</div>
      </div>
    )
  }

  const { paper_data, repo_data, mappings, code_results } = currentResult

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

      <Tab.Group>
        <Tab.List className="flex gap-2 mb-6">
          {tabs.map((tab) => (
            <Tab
              key={tab.name}
              className={({ selected }) =>
                cn(
                  'flex items-center gap-2 px-4 py-2.5 rounded-lg text-body-sm font-medium transition-colors',
                  selected
                    ? 'bg-accent-primary/20 text-accent-secondary border border-accent-primary/30'
                    : 'text-text-secondary hover:bg-bg-tertiary'
                )
              }
            >
              <tab.icon className="h-5 w-5" />
              {tab.name}
            </Tab>
          ))}
        </Tab.List>

        <Tab.Panels>
          {/* Paper Tab */}
          <Tab.Panel>
            <div className="space-y-6">
              <GlassCard title={paper_data.title} subtitle={`by ${paper_data.authors.join(', ')}`}>
                <div className="space-y-6">
                  <div>
                    <h4 className="text-body-sm font-semibold text-text-secondary mb-2">Abstract</h4>
                    <p className="text-body text-text-primary">{paper_data.abstract}</p>
                  </div>

                  <div>
                    <h4 className="text-body-sm font-semibold text-text-secondary mb-3">Key Concepts ({paper_data.key_concepts.length})</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {paper_data.key_concepts.map((concept, idx) => (
                        <div key={idx} className="p-4 rounded-lg bg-bg-tertiary/50">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-body-sm font-medium text-text-primary">{concept.name}</span>
                            <span className={cn(
                              'text-caption px-2 py-0.5 rounded-full',
                              concept.importance === 'critical' && 'bg-status-error/20 text-status-error',
                              concept.importance === 'high' && 'bg-status-warning/20 text-status-warning',
                              concept.importance === 'medium' && 'bg-accent-primary/20 text-accent-primary',
                              concept.importance === 'low' && 'bg-text-muted/20 text-text-muted'
                            )}>
                              {concept.importance}
                            </span>
                          </div>
                          <p className="text-caption text-text-secondary">{concept.description}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {paper_data.algorithms.length > 0 && (
                    <div>
                      <h4 className="text-body-sm font-semibold text-text-secondary mb-3">Algorithms ({paper_data.algorithms.length})</h4>
                      <div className="space-y-3">
                        {paper_data.algorithms.map((algo, idx) => (
                          <div key={idx} className="p-4 rounded-lg bg-bg-tertiary/50">
                            <h5 className="text-body-sm font-medium text-text-primary mb-1">{algo.name}</h5>
                            <p className="text-caption text-text-secondary">{algo.description}</p>
                            {algo.complexity && (
                              <p className="text-caption text-text-muted mt-1">Complexity: {algo.complexity}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </GlassCard>
            </div>
          </Tab.Panel>

          {/* Repository Tab */}
          <Tab.Panel>
            <div className="space-y-6">
              <GlassCard title={repo_data.name} subtitle={repo_data.url}>
                <div className="space-y-6">
                  <div>
                    <h4 className="text-body-sm font-semibold text-text-secondary mb-2">Overview</h4>
                    <p className="text-body text-text-primary">{repo_data.overview.purpose}</p>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-4 rounded-lg bg-bg-tertiary/50 text-center">
                      <p className="text-heading-2 text-text-primary">{repo_data.stats.total_files}</p>
                      <p className="text-caption text-text-muted">Total Files</p>
                    </div>
                    <div className="p-4 rounded-lg bg-bg-tertiary/50 text-center">
                      <p className="text-heading-2 text-text-primary">{repo_data.stats.code_files}</p>
                      <p className="text-caption text-text-muted">Code Files</p>
                    </div>
                    <div className="p-4 rounded-lg bg-bg-tertiary/50 text-center">
                      <p className="text-heading-2 text-text-primary">{repo_data.stats.classes}</p>
                      <p className="text-caption text-text-muted">Classes</p>
                    </div>
                    <div className="p-4 rounded-lg bg-bg-tertiary/50 text-center">
                      <p className="text-heading-2 text-text-primary">{repo_data.stats.functions}</p>
                      <p className="text-caption text-text-muted">Functions</p>
                    </div>
                  </div>

                  {repo_data.key_components.length > 0 && (
                    <div>
                      <h4 className="text-body-sm font-semibold text-text-secondary mb-3">Key Components</h4>
                      <div className="space-y-2">
                        {repo_data.key_components.map((comp, idx) => (
                          <div key={idx} className="flex items-center justify-between p-3 rounded-lg bg-bg-tertiary/50">
                            <div>
                              <span className="text-body-sm font-medium text-text-primary">{comp.name}</span>
                              <span className="text-caption text-text-muted ml-2">{comp.path}</span>
                            </div>
                            <span className="text-caption text-text-secondary">{comp.description}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </GlassCard>
            </div>
          </Tab.Panel>

          {/* Mappings Tab */}
          <Tab.Panel>
            <GlassCard title={`Concept-to-Code Mappings (${mappings.length})`}>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="px-4 py-3 text-left text-body-sm font-semibold text-text-secondary">Concept</th>
                      <th className="px-4 py-3 text-left text-body-sm font-semibold text-text-secondary">Code Element</th>
                      <th className="px-4 py-3 text-left text-body-sm font-semibold text-text-secondary">File</th>
                      <th className="px-4 py-3 text-left text-body-sm font-semibold text-text-secondary">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mappings.map((mapping, idx) => (
                      <tr key={idx} className="border-b border-border/50 hover:bg-bg-tertiary/30">
                        <td className="px-4 py-3">
                          <span className="text-body-sm text-text-primary">{mapping.concept_name}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="font-mono text-body-sm text-accent-secondary">{mapping.code_element}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-body-sm text-text-muted">{mapping.code_file}</span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2">
                            <div className="w-20 h-2 bg-bg-tertiary rounded-full overflow-hidden">
                              <div
                                className={cn(
                                  'h-full rounded-full',
                                  mapping.confidence >= 0.8 && 'bg-status-success',
                                  mapping.confidence >= 0.5 && mapping.confidence < 0.8 && 'bg-status-warning',
                                  mapping.confidence < 0.5 && 'bg-status-error'
                                )}
                                style={{ width: `${mapping.confidence * 100}%` }}
                              />
                            </div>
                            <span className="text-caption text-text-muted">{formatConfidence(mapping.confidence)}</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </GlassCard>
          </Tab.Panel>

          {/* Tests Tab */}
          <Tab.Panel>
            <div className="space-y-6">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <GlassCard>
                  <div className="text-center">
                    <p className="text-heading-1 text-text-primary">{code_results.summary.total_tests}</p>
                    <p className="text-body-sm text-text-muted">Total Tests</p>
                  </div>
                </GlassCard>
                <GlassCard>
                  <div className="text-center">
                    <p className="text-heading-1 text-status-success">{code_results.summary.passed}</p>
                    <p className="text-body-sm text-text-muted">Passed</p>
                  </div>
                </GlassCard>
                <GlassCard>
                  <div className="text-center">
                    <p className="text-heading-1 text-status-error">{code_results.summary.failed}</p>
                    <p className="text-body-sm text-text-muted">Failed</p>
                  </div>
                </GlassCard>
                <GlassCard>
                  <div className="text-center">
                    <p className="text-heading-1 text-text-muted">{code_results.summary.skipped}</p>
                    <p className="text-body-sm text-text-muted">Skipped</p>
                  </div>
                </GlassCard>
              </div>

              <GlassCard title="Test Results">
                <div className="space-y-3">
                  {code_results.results.map((result, idx) => (
                    <div key={idx} className="flex items-start gap-4 p-4 rounded-lg bg-bg-tertiary/50">
                      {result.success ? (
                        <CheckCircleIcon className="h-6 w-6 text-status-success flex-shrink-0" />
                      ) : (
                        <XCircleIcon className="h-6 w-6 text-status-error flex-shrink-0" />
                      )}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-body-sm font-medium text-text-primary">{result.concept}</span>
                          <span className="text-caption text-text-muted">→</span>
                          <span className="font-mono text-body-sm text-accent-secondary">{result.code_element}</span>
                        </div>
                        {result.error && (
                          <p className="text-caption text-status-error">{result.error}</p>
                        )}
                        <p className="text-caption text-text-muted">
                          {result.execution_time.toFixed(2)}s • {result.isolation_level}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </GlassCard>
            </div>
          </Tab.Panel>
        </Tab.Panels>
      </Tab.Group>
    </div>
  )
}
