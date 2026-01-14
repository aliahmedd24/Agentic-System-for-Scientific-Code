import { Tab } from '@headlessui/react'
import {
  FolderIcon,
  DocumentIcon,
  CubeIcon,
  CommandLineIcon,
  ArrowTopRightOnSquareIcon,
  CpuChipIcon,
  ClockIcon,
} from '@heroicons/react/24/outline'
import { cn } from '@/lib/cn'
import { GlassCard } from '@/components/ui/GlassCard'
import type { RepoData, SetupComplexity } from '@/api/types'

interface RepoSummaryProps {
  repo: RepoData
  className?: string
}

const complexityColors: Record<SetupComplexity['level'], { bg: string; text: string }> = {
  easy: { bg: 'bg-status-success/20', text: 'text-status-success' },
  medium: { bg: 'bg-status-warning/20', text: 'text-status-warning' },
  hard: { bg: 'bg-status-error/20', text: 'text-status-error' },
  expert: { bg: 'bg-graph-function/20', text: 'text-graph-function' },
}

export function RepoSummary({ repo, className }: RepoSummaryProps) {
  const dependencyTabs = [
    { key: 'python', label: 'Python', deps: repo.dependencies.python },
    { key: 'javascript', label: 'JavaScript', deps: repo.dependencies.javascript },
    { key: 'julia', label: 'Julia', deps: repo.dependencies.julia },
    { key: 'r', label: 'R', deps: repo.dependencies.r },
    { key: 'system', label: 'System', deps: repo.dependencies.system },
  ].filter((t) => t.deps.length > 0)

  const complexityStyle = complexityColors[repo.setup_complexity.level]

  return (
    <div className={cn('space-y-6', className)}>
      {/* Header */}
      <GlassCard>
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-lg bg-graph-repository/20 flex items-center justify-center shrink-0">
            <FolderIcon className="h-6 w-6 text-graph-repository" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-3 mb-2">
              <h2 className="text-heading-2 text-text-primary">{repo.name}</h2>
              <a
                href={repo.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-accent-secondary hover:text-accent-primary transition-colors"
              >
                <ArrowTopRightOnSquareIcon className="h-5 w-5" />
              </a>
            </div>
            <p className="text-body-sm text-text-secondary mb-3">
              {repo.overview.purpose}
            </p>

            {/* Stats */}
            <div className="flex flex-wrap gap-4 text-body-sm text-text-muted">
              <span className="flex items-center gap-1.5">
                <DocumentIcon className="h-4 w-4" />
                {repo.stats.code_files} files
              </span>
              <span className="flex items-center gap-1.5">
                <CubeIcon className="h-4 w-4" />
                {repo.stats.classes} classes
              </span>
              <span className="flex items-center gap-1.5">
                <CommandLineIcon className="h-4 w-4" />
                {repo.stats.functions} functions
              </span>
            </div>
          </div>
        </div>
      </GlassCard>

      {/* Overview */}
      <GlassCard title="Overview">
        <div className="space-y-4">
          <div>
            <h5 className="text-body-sm font-medium text-text-secondary mb-1">
              Architecture
            </h5>
            <p className="text-body-sm text-text-muted">
              {repo.overview.architecture}
            </p>
          </div>

          {repo.overview.key_features.length > 0 && (
            <div>
              <h5 className="text-body-sm font-medium text-text-secondary mb-2">
                Key Features
              </h5>
              <ul className="space-y-1">
                {repo.overview.key_features.map((feature, idx) => (
                  <li
                    key={idx}
                    className="flex items-start gap-2 text-body-sm text-text-muted"
                  >
                    <span className="text-accent-primary mt-1">•</span>
                    <span>{feature}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </GlassCard>

      {/* Key Components */}
      <GlassCard
        title="Key Components"
        subtitle={`${repo.key_components.length} components identified`}
      >
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-2 px-3 text-body-sm font-medium text-text-muted">
                  Name
                </th>
                <th className="text-left py-2 px-3 text-body-sm font-medium text-text-muted">
                  Path
                </th>
                <th className="text-left py-2 px-3 text-body-sm font-medium text-text-muted">
                  Description
                </th>
                <th className="text-left py-2 px-3 text-body-sm font-medium text-text-muted">
                  Importance
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {repo.key_components.map((component, idx) => (
                <tr key={idx} className="hover:bg-bg-tertiary/30 transition-colors">
                  <td className="py-3 px-3 text-body-sm text-text-primary font-medium">
                    {component.name}
                  </td>
                  <td className="py-3 px-3 text-body-sm text-text-muted font-mono">
                    {component.path}
                  </td>
                  <td className="py-3 px-3 text-body-sm text-text-secondary max-w-xs truncate">
                    {component.description}
                  </td>
                  <td className="py-3 px-3">
                    <span className="text-caption text-text-muted capitalize">
                      {component.importance}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </GlassCard>

      {/* Entry Points */}
      {repo.entry_points.length > 0 && (
        <GlassCard title="Entry Points">
          <div className="space-y-3">
            {repo.entry_points.map((entry, idx) => (
              <div
                key={idx}
                className="p-3 rounded-lg border border-border hover:border-border-glow transition-colors"
              >
                <div className="flex items-center gap-2 mb-1">
                  <CommandLineIcon className="h-4 w-4 text-accent-primary" />
                  <span className="text-body-sm font-medium text-text-primary font-mono">
                    {entry.name}
                  </span>
                </div>
                <p className="text-caption text-text-muted mb-2 ml-6">
                  {entry.path}
                </p>
                <p className="text-body-sm text-text-secondary ml-6">
                  {entry.description}
                </p>
                {entry.arguments.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-2 ml-6">
                    {entry.arguments.map((arg, argIdx) => (
                      <span
                        key={argIdx}
                        className="px-2 py-0.5 rounded bg-bg-tertiary text-caption text-text-muted font-mono"
                      >
                        {arg}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </GlassCard>
      )}

      {/* Dependencies */}
      {dependencyTabs.length > 0 && (
        <GlassCard title="Dependencies">
          <Tab.Group>
            <Tab.List className="flex gap-2 mb-4">
              {dependencyTabs.map((tab) => (
                <Tab
                  key={tab.key}
                  className={({ selected }) =>
                    cn(
                      'px-3 py-1.5 rounded-lg text-body-sm font-medium transition-colors',
                      selected
                        ? 'bg-accent-primary text-white'
                        : 'bg-bg-tertiary text-text-muted hover:text-text-secondary'
                    )
                  }
                >
                  {tab.label}
                  <span className="ml-1.5 text-caption opacity-70">
                    ({tab.deps.length})
                  </span>
                </Tab>
              ))}
            </Tab.List>

            <Tab.Panels>
              {dependencyTabs.map((tab) => (
                <Tab.Panel key={tab.key}>
                  <div className="flex flex-wrap gap-2">
                    {tab.deps.map((dep, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-1 rounded bg-bg-tertiary text-body-sm text-text-secondary font-mono"
                      >
                        {dep}
                      </span>
                    ))}
                  </div>
                </Tab.Panel>
              ))}
            </Tab.Panels>
          </Tab.Group>
        </GlassCard>
      )}

      {/* Setup & Requirements */}
      <div className="grid gap-4 lg:grid-cols-2">
        {/* Setup Complexity */}
        <GlassCard title="Setup Complexity">
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <span
                className={cn(
                  'px-3 py-1 rounded-full text-body-sm font-medium capitalize',
                  complexityStyle.bg,
                  complexityStyle.text
                )}
              >
                {repo.setup_complexity.level}
              </span>
              {repo.setup_complexity.estimated_time && (
                <span className="flex items-center gap-1.5 text-body-sm text-text-muted">
                  <ClockIcon className="h-4 w-4" />
                  {repo.setup_complexity.estimated_time}
                </span>
              )}
            </div>

            {repo.setup_complexity.steps.length > 0 && (
              <ol className="space-y-2">
                {repo.setup_complexity.steps.map((step, idx) => (
                  <li key={idx} className="flex items-start gap-3 text-body-sm">
                    <span className="w-5 h-5 rounded-full bg-accent-primary/20 text-accent-secondary text-caption flex items-center justify-center shrink-0">
                      {idx + 1}
                    </span>
                    <span className="text-text-secondary">{step}</span>
                  </li>
                ))}
              </ol>
            )}
          </div>
        </GlassCard>

        {/* Compute Requirements */}
        <GlassCard title="Compute Requirements">
          <div className="grid grid-cols-2 gap-4">
            <div className="flex items-center gap-3">
              <CpuChipIcon className="h-5 w-5 text-text-muted" />
              <div>
                <p className="text-body-sm text-text-primary">CPU Cores</p>
                <p className="text-heading-3 text-accent-secondary">
                  {repo.compute_requirements.cpu_cores}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <CubeIcon className="h-5 w-5 text-text-muted" />
              <div>
                <p className="text-body-sm text-text-primary">Memory</p>
                <p className="text-heading-3 text-accent-secondary">
                  {repo.compute_requirements.memory_gb} GB
                </p>
              </div>
            </div>

            <div className="col-span-2 flex items-center gap-3">
              <div
                className={cn(
                  'w-5 h-5 rounded-full flex items-center justify-center',
                  repo.compute_requirements.gpu_required
                    ? 'bg-status-warning/20'
                    : 'bg-status-success/20'
                )}
              >
                <span
                  className={cn(
                    'text-caption font-bold',
                    repo.compute_requirements.gpu_required
                      ? 'text-status-warning'
                      : 'text-status-success'
                  )}
                >
                  G
                </span>
              </div>
              <div>
                <p className="text-body-sm text-text-primary">GPU</p>
                <p className="text-body-sm text-text-muted">
                  {repo.compute_requirements.gpu_required
                    ? `Required (${repo.compute_requirements.gpu_memory_gb} GB VRAM)`
                    : 'Not required'}
                </p>
              </div>
            </div>
          </div>
        </GlassCard>
      </div>
    </div>
  )
}

export default RepoSummary
