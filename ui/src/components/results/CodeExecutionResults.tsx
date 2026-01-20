import { Disclosure } from '@headlessui/react'
import {
  ChevronDownIcon,
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  LightBulbIcon,
  CpuChipIcon,
  DocumentIcon,
} from '@heroicons/react/24/outline'
import { cn } from '@/lib/cn'
import { GlassCard } from '@/components/ui/GlassCard'
import { CodeBlock } from '@/components/code/CodeBlock'
import { formatDuration } from '@/lib/formatters'
import type { CodeResults, TestResult, GeneratedScript } from '@/api/types'

interface CodeExecutionResultsProps {
  codeResults: CodeResults
  className?: string
}

interface TestResultCardProps {
  result: TestResult
}

function TestResultCard({ result }: TestResultCardProps) {
  const hasOutput = result.stdout || result.stderr
  void (result.error || result.stderr) // hasError reserved

  return (
    <Disclosure>
      {({ open }) => (
        <div
          className={cn(
            'rounded-lg border overflow-hidden transition-colors',
            result.success
              ? 'border-status-success/30 hover:border-status-success/50'
              : 'border-status-error/30 hover:border-status-error/50'
          )}
        >
          <Disclosure.Button
            className={cn(
              'w-full flex items-center justify-between gap-4 p-4',
              'bg-bg-tertiary/30 hover:bg-bg-tertiary/50 transition-colors'
            )}
          >
            <div className="flex items-center gap-3 min-w-0">
              {result.success ? (
                <CheckCircleIcon className="h-5 w-5 text-status-success shrink-0" />
              ) : (
                <XCircleIcon className="h-5 w-5 text-status-error shrink-0" />
              )}
              <div className="text-left min-w-0">
                <h5 className="text-body-sm font-medium text-text-primary truncate">
                  {result.concept}
                </h5>
                <p className="text-caption text-text-muted font-mono truncate">
                  {result.code_element}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-4 shrink-0">
              <span className="flex items-center gap-1.5 text-caption text-text-muted">
                <ClockIcon className="h-3.5 w-3.5" />
                {formatDuration(result.execution_time * 1000)}
              </span>
              {hasOutput && (
                <ChevronDownIcon
                  className={cn(
                    'h-5 w-5 text-text-muted transition-transform',
                    open && 'rotate-180'
                  )}
                />
              )}
            </div>
          </Disclosure.Button>

          {hasOutput && (
            <Disclosure.Panel className="border-t border-border">
              <div className="p-4 space-y-4">
                {result.stdout && (
                  <div>
                    <h6 className="text-body-sm font-medium text-text-secondary mb-2">
                      Output (stdout)
                    </h6>
                    <CodeBlock
                      code={result.stdout}
                      language="shell"
                      showLineNumbers={false}
                      maxHeight="200px"
                    />
                  </div>
                )}

                {result.stderr && (
                  <div>
                    <h6 className="text-body-sm font-medium text-status-error mb-2">
                      Errors (stderr)
                    </h6>
                    <CodeBlock
                      code={result.stderr}
                      language="shell"
                      showLineNumbers={false}
                      maxHeight="200px"
                    />
                  </div>
                )}

                {result.error && !result.stderr && (
                  <div className="p-3 rounded bg-status-error/10 border border-status-error/20">
                    <p className="text-body-sm text-status-error">{result.error}</p>
                  </div>
                )}

                {result.output_files.length > 0 && (
                  <div>
                    <h6 className="text-body-sm font-medium text-text-secondary mb-2">
                      Output Files
                    </h6>
                    <div className="flex flex-wrap gap-2">
                      {result.output_files.map((file, idx) => (
                        <span
                          key={idx}
                          className="flex items-center gap-1.5 px-2 py-1 rounded bg-bg-tertiary text-caption text-text-muted"
                        >
                          <DocumentIcon className="h-3.5 w-3.5" />
                          {file}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                <div className="flex items-center gap-4 text-caption text-text-muted">
                  <span>Return code: {result.return_code}</span>
                  <span>Isolation: {result.isolation_level}</span>
                </div>
              </div>
            </Disclosure.Panel>
          )}
        </div>
      )}
    </Disclosure>
  )
}

interface ScriptCardProps {
  script: GeneratedScript
}

function ScriptCard({ script }: ScriptCardProps) {
  return (
    <Disclosure>
      {({ open }) => (
        <GlassCard noPadding className="overflow-hidden">
          <Disclosure.Button
            className={cn(
              'w-full flex items-center justify-between gap-4 p-4',
              'hover:bg-bg-tertiary/50 transition-colors'
            )}
          >
            <div className="flex items-center gap-3 min-w-0">
              <div
                className={cn(
                  'w-8 h-8 rounded-lg flex items-center justify-center',
                  script.syntax_valid && script.import_valid
                    ? 'bg-status-success/20'
                    : 'bg-status-warning/20'
                )}
              >
                <span
                  className={cn(
                    'text-caption font-bold',
                    script.syntax_valid && script.import_valid
                      ? 'text-status-success'
                      : 'text-status-warning'
                  )}
                >
                  {script.language.slice(0, 2).toUpperCase()}
                </span>
              </div>
              <div className="text-left min-w-0">
                <h5 className="text-body-sm font-medium text-text-primary truncate">
                  {script.concept}
                </h5>
                <p className="text-caption text-text-muted truncate">
                  {script.file_name}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3 shrink-0">
              <span className="text-body-sm text-accent-secondary">
                {Math.round(script.confidence * 100)}%
              </span>
              <ChevronDownIcon
                className={cn(
                  'h-5 w-5 text-text-muted transition-transform',
                  open && 'rotate-180'
                )}
              />
            </div>
          </Disclosure.Button>

          <Disclosure.Panel className="border-t border-border">
            <div className="p-4 space-y-3">
              <div className="flex items-center gap-4 text-caption text-text-muted">
                <span
                  className={cn(
                    'flex items-center gap-1',
                    script.syntax_valid ? 'text-status-success' : 'text-status-error'
                  )}
                >
                  {script.syntax_valid ? (
                    <CheckCircleIcon className="h-3.5 w-3.5" />
                  ) : (
                    <XCircleIcon className="h-3.5 w-3.5" />
                  )}
                  Syntax
                </span>
                <span
                  className={cn(
                    'flex items-center gap-1',
                    script.import_valid ? 'text-status-success' : 'text-status-error'
                  )}
                >
                  {script.import_valid ? (
                    <CheckCircleIcon className="h-3.5 w-3.5" />
                  ) : (
                    <XCircleIcon className="h-3.5 w-3.5" />
                  )}
                  Imports
                </span>
              </div>
              <CodeBlock
                code={script.code}
                language={script.language as 'python' | 'javascript'}
                maxHeight="300px"
              />
            </div>
          </Disclosure.Panel>
        </GlassCard>
      )}
    </Disclosure>
  )
}

export function CodeExecutionResults({ codeResults, className }: CodeExecutionResultsProps) {
  const { summary, results, scripts, resource_estimate } = codeResults

  const passRate = summary.total_tests > 0
    ? Math.round((summary.passed / summary.total_tests) * 100)
    : 0

  return (
    <div className={cn('space-y-6', className)}>
      {/* Summary Stats */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        <GlassCard className="text-center">
          <p className="text-caption text-text-muted mb-1">Total Tests</p>
          <p className="text-heading-2 text-text-primary">{summary.total_tests}</p>
        </GlassCard>

        <GlassCard className="text-center">
          <p className="text-caption text-text-muted mb-1">Passed</p>
          <p className="text-heading-2 text-status-success">{summary.passed}</p>
        </GlassCard>

        <GlassCard className="text-center">
          <p className="text-caption text-text-muted mb-1">Failed</p>
          <p className="text-heading-2 text-status-error">{summary.failed}</p>
        </GlassCard>

        <GlassCard className="text-center">
          <p className="text-caption text-text-muted mb-1">Skipped</p>
          <p className="text-heading-2 text-status-warning">{summary.skipped}</p>
        </GlassCard>

        <GlassCard className="text-center">
          <p className="text-caption text-text-muted mb-1">Pass Rate</p>
          <p
            className={cn(
              'text-heading-2',
              passRate >= 80
                ? 'text-status-success'
                : passRate >= 50
                  ? 'text-status-warning'
                  : 'text-status-error'
            )}
          >
            {passRate}%
          </p>
        </GlassCard>
      </div>

      {/* Resource Estimate */}
      {resource_estimate && (
        <GlassCard title="Resource Estimate" icon={<CpuChipIcon className="h-5 w-5 text-text-muted" />}>
          <div className="space-y-4">
            {/* Metrics */}
            <div className="grid gap-4 sm:grid-cols-3 lg:grid-cols-6">
              <div>
                <p className="text-caption text-text-muted">Compute Level</p>
                <p className="text-body font-medium text-text-primary capitalize">
                  {resource_estimate.compute_level}
                </p>
              </div>
              <div>
                <p className="text-caption text-text-muted">Memory</p>
                <p className="text-body font-medium text-text-primary">
                  {resource_estimate.memory_gb} GB
                </p>
              </div>
              <div>
                <p className="text-caption text-text-muted">GPU</p>
                <p className="text-body font-medium text-text-primary">
                  {resource_estimate.gpu_required
                    ? `${resource_estimate.gpu_memory_gb} GB`
                    : 'Not required'}
                </p>
              </div>
              <div>
                <p className="text-caption text-text-muted">Est. Time</p>
                <p className="text-body font-medium text-text-primary">
                  {resource_estimate.estimated_time_minutes} min
                </p>
              </div>
              <div>
                <p className="text-caption text-text-muted">Disk Space</p>
                <p className="text-body font-medium text-text-primary">
                  {resource_estimate.disk_space_gb} GB
                </p>
              </div>
              <div>
                <p className="text-caption text-text-muted">Complexity</p>
                <p className="text-body font-medium text-text-primary">
                  {resource_estimate.complexity_score}/10
                </p>
              </div>
            </div>

            {/* Warnings */}
            {resource_estimate.warnings.length > 0 && (
              <div className="p-3 rounded-lg bg-status-warning/10 border border-status-warning/20">
                <div className="flex items-start gap-2">
                  <ExclamationTriangleIcon className="h-5 w-5 text-status-warning shrink-0 mt-0.5" />
                  <div>
                    <p className="text-body-sm font-medium text-status-warning mb-1">
                      Warnings
                    </p>
                    <ul className="space-y-1">
                      {resource_estimate.warnings.map((warning, idx) => (
                        <li key={idx} className="text-body-sm text-text-secondary">
                          {warning}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}

            {/* Recommendations */}
            {resource_estimate.recommendations.length > 0 && (
              <div className="p-3 rounded-lg bg-status-info/10 border border-status-info/20">
                <div className="flex items-start gap-2">
                  <LightBulbIcon className="h-5 w-5 text-status-info shrink-0 mt-0.5" />
                  <div>
                    <p className="text-body-sm font-medium text-status-info mb-1">
                      Recommendations
                    </p>
                    <ul className="space-y-1">
                      {resource_estimate.recommendations.map((rec, idx) => (
                        <li key={idx} className="text-body-sm text-text-secondary">
                          {rec}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </div>
        </GlassCard>
      )}

      {/* Test Results */}
      {results.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-heading-3 text-text-primary">
            Test Results ({results.length})
          </h3>
          <div className="space-y-2">
            {results.map((result, idx) => (
              <TestResultCard key={idx} result={result} />
            ))}
          </div>
        </div>
      )}

      {/* Generated Scripts */}
      {scripts.length > 0 && (
        <Disclosure>
          {({ open }) => (
            <>
              <Disclosure.Button
                className={cn(
                  'w-full flex items-center justify-between gap-4 p-4 rounded-lg',
                  'bg-bg-tertiary hover:bg-bg-tertiary/80 transition-colors'
                )}
              >
                <span className="text-heading-3 text-text-primary">
                  Generated Scripts ({scripts.length})
                </span>
                <ChevronDownIcon
                  className={cn(
                    'h-5 w-5 text-text-muted transition-transform',
                    open && 'rotate-180'
                  )}
                />
              </Disclosure.Button>

              <Disclosure.Panel className="space-y-3 mt-3">
                {scripts.map((script, idx) => (
                  <ScriptCard key={idx} script={script} />
                ))}
              </Disclosure.Panel>
            </>
          )}
        </Disclosure>
      )}
    </div>
  )
}

export default CodeExecutionResults
