import { Disclosure } from '@headlessui/react'
import {
  ChevronDownIcon,
  DocumentTextIcon,
  BeakerIcon,
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  CpuChipIcon,
} from '@heroicons/react/24/outline'
import { cn } from '@/lib/cn'
import { GlassCard } from '@/components/ui/GlassCard'
import { CodeBlock } from '@/components/code/CodeBlock'
import type { PaperData, Concept, Algorithm } from '@/api/types'

interface PaperSummaryProps {
  paper: PaperData
  className?: string
}

const importanceColors: Record<Concept['importance'], { bg: string; text: string }> = {
  critical: { bg: 'bg-status-error/20', text: 'text-status-error' },
  high: { bg: 'bg-status-warning/20', text: 'text-status-warning' },
  medium: { bg: 'bg-status-info/20', text: 'text-status-info' },
  low: { bg: 'bg-text-muted/20', text: 'text-text-muted' },
}

interface ConceptBadgeProps {
  concept: Concept
}

function ConceptBadge({ concept }: ConceptBadgeProps) {
  const colors = importanceColors[concept.importance]
  return (
    <div
      className={cn(
        'p-3 rounded-lg border border-border',
        'hover:border-border-glow transition-colors duration-200'
      )}
    >
      <div className="flex items-center gap-2 mb-1">
        <h5 className="text-body-sm font-medium text-text-primary">
          {concept.name}
        </h5>
        <span className={cn('px-1.5 py-0.5 rounded text-caption', colors.bg, colors.text)}>
          {concept.importance}
        </span>
      </div>
      <p className="text-caption text-text-muted line-clamp-2">
        {concept.description}
      </p>
    </div>
  )
}

interface AlgorithmCardProps {
  algorithm: Algorithm
}

function AlgorithmCard({ algorithm }: AlgorithmCardProps) {
  return (
    <Disclosure>
      {({ open }) => (
        <GlassCard noPadding className="overflow-hidden">
          <Disclosure.Button
            className={cn(
              'w-full flex items-center justify-between gap-4 p-4',
              'hover:bg-bg-tertiary/50 transition-colors duration-200'
            )}
          >
            <div className="flex items-center gap-3">
              <BeakerIcon className="h-5 w-5 text-graph-algorithm" />
              <div className="text-left">
                <h5 className="text-body font-medium text-text-primary">
                  {algorithm.name}
                </h5>
                <p className="text-caption text-text-muted">
                  Complexity: {algorithm.complexity}
                </p>
              </div>
            </div>
            <ChevronDownIcon
              className={cn(
                'h-5 w-5 text-text-muted transition-transform duration-200',
                open && 'rotate-180'
              )}
            />
          </Disclosure.Button>

          <Disclosure.Panel>
            <div className="border-t border-border p-4 space-y-4">
              <p className="text-body-sm text-text-secondary">
                {algorithm.description}
              </p>

              {/* Inputs/Outputs */}
              <div className="flex flex-wrap gap-4">
                {algorithm.inputs.length > 0 && (
                  <div>
                    <span className="text-caption text-text-muted">Inputs:</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {algorithm.inputs.map((input, idx) => (
                        <span
                          key={idx}
                          className="px-2 py-0.5 rounded bg-bg-tertiary text-caption text-text-secondary"
                        >
                          {input}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                {algorithm.outputs.length > 0 && (
                  <div>
                    <span className="text-caption text-text-muted">Outputs:</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {algorithm.outputs.map((output, idx) => (
                        <span
                          key={idx}
                          className="px-2 py-0.5 rounded bg-accent-primary/20 text-caption text-accent-secondary"
                        >
                          {output}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Pseudocode */}
              {algorithm.pseudocode && (
                <div>
                  <h6 className="text-body-sm font-medium text-text-secondary mb-2">
                    Pseudocode
                  </h6>
                  <CodeBlock
                    code={algorithm.pseudocode}
                    language="python"
                    showLineNumbers={false}
                    maxHeight="300px"
                  />
                </div>
              )}
            </div>
          </Disclosure.Panel>
        </GlassCard>
      )}
    </Disclosure>
  )
}

export function PaperSummary({ paper, className }: PaperSummaryProps) {
  return (
    <div className={cn('space-y-6', className)}>
      {/* Header / Metadata */}
      <GlassCard>
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-lg bg-graph-paper/20 flex items-center justify-center shrink-0">
            <DocumentTextIcon className="h-6 w-6 text-graph-paper" />
          </div>
          <div className="flex-1 min-w-0">
            <h2 className="text-heading-2 text-text-primary mb-2">
              {paper.title}
            </h2>
            <p className="text-body-sm text-text-secondary mb-3">
              {paper.authors.slice(0, 5).join(', ')}
              {paper.authors.length > 5 && ` and ${paper.authors.length - 5} others`}
            </p>
            {paper.source_metadata.arxiv_id && (
              <span className="inline-flex items-center px-2 py-1 rounded bg-bg-tertiary text-caption text-text-muted">
                arXiv: {paper.source_metadata.arxiv_id}
              </span>
            )}
          </div>
        </div>
      </GlassCard>

      {/* Abstract */}
      <GlassCard title="Abstract" icon={<DocumentTextIcon className="h-5 w-5 text-text-muted" />}>
        <p className="text-body text-text-secondary leading-relaxed">
          {paper.abstract}
        </p>
      </GlassCard>

      {/* Key Concepts */}
      <GlassCard
        title="Key Concepts"
        subtitle={`${paper.key_concepts.length} concepts identified`}
      >
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {paper.key_concepts.map((concept, idx) => (
            <ConceptBadge key={idx} concept={concept} />
          ))}
        </div>
      </GlassCard>

      {/* Algorithms */}
      {paper.algorithms.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-heading-3 text-text-primary">
            Algorithms ({paper.algorithms.length})
          </h3>
          {paper.algorithms.map((algorithm, idx) => (
            <AlgorithmCard key={idx} algorithm={algorithm} />
          ))}
        </div>
      )}

      {/* Methodology */}
      <GlassCard title="Methodology">
        <div className="space-y-4">
          <div>
            <h5 className="text-body-sm font-medium text-text-secondary mb-1">
              Approach
            </h5>
            <p className="text-body-sm text-text-muted">
              {paper.methodology.approach}
            </p>
          </div>

          {paper.methodology.datasets.length > 0 && (
            <div>
              <h5 className="text-body-sm font-medium text-text-secondary mb-2">
                Datasets
              </h5>
              <div className="flex flex-wrap gap-2">
                {paper.methodology.datasets.map((dataset, idx) => (
                  <span
                    key={idx}
                    className="px-2 py-1 rounded bg-bg-tertiary text-body-sm text-text-secondary"
                  >
                    {dataset}
                  </span>
                ))}
              </div>
            </div>
          )}

          {paper.methodology.evaluation_metrics.length > 0 && (
            <div>
              <h5 className="text-body-sm font-medium text-text-secondary mb-2">
                Evaluation Metrics
              </h5>
              <div className="flex flex-wrap gap-2">
                {paper.methodology.evaluation_metrics.map((metric, idx) => (
                  <span
                    key={idx}
                    className="px-2 py-1 rounded bg-accent-primary/10 text-body-sm text-accent-secondary"
                  >
                    {metric}
                  </span>
                ))}
              </div>
            </div>
          )}

          {paper.methodology.baselines.length > 0 && (
            <div>
              <h5 className="text-body-sm font-medium text-text-secondary mb-2">
                Baselines
              </h5>
              <div className="flex flex-wrap gap-2">
                {paper.methodology.baselines.map((baseline, idx) => (
                  <span
                    key={idx}
                    className="px-2 py-1 rounded bg-status-info/10 text-body-sm text-status-info"
                  >
                    {baseline}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </GlassCard>

      {/* Reproducibility */}
      <GlassCard title="Reproducibility">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="flex items-center gap-3">
            {paper.reproducibility.code_available ? (
              <CheckCircleIcon className="h-5 w-5 text-status-success" />
            ) : (
              <XCircleIcon className="h-5 w-5 text-status-error" />
            )}
            <div>
              <p className="text-body-sm text-text-primary">Code Available</p>
              <p className="text-caption text-text-muted">
                {paper.reproducibility.code_available ? 'Yes' : 'No'}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {paper.reproducibility.data_available ? (
              <CheckCircleIcon className="h-5 w-5 text-status-success" />
            ) : (
              <XCircleIcon className="h-5 w-5 text-status-error" />
            )}
            <div>
              <p className="text-body-sm text-text-primary">Data Available</p>
              <p className="text-caption text-text-muted">
                {paper.reproducibility.data_available ? 'Yes' : 'No'}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <CpuChipIcon className="h-5 w-5 text-text-muted" />
            <div>
              <p className="text-body-sm text-text-primary">Hardware</p>
              <p className="text-caption text-text-muted">
                {paper.reproducibility.hardware_requirements || 'Not specified'}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <ClockIcon className="h-5 w-5 text-text-muted" />
            <div>
              <p className="text-body-sm text-text-primary">Est. Time</p>
              <p className="text-caption text-text-muted">
                {paper.reproducibility.estimated_time || 'Not specified'}
              </p>
            </div>
          </div>
        </div>
      </GlassCard>
    </div>
  )
}

export default PaperSummary
