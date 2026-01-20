import {
  DocumentTextIcon,
  CodeBracketIcon,
  GlobeAltIcon,
  DocumentArrowDownIcon,
} from '@heroicons/react/24/outline'
import { cn } from '@/lib/cn'
import { GlassCard } from '@/components/ui/GlassCard'
import { Button } from '@/components/ui/Button'
import { Toggle } from '@/components/ui/Toggle'

export type ReportFormat = 'html' | 'json' | 'markdown'

export interface ReportOptions {
  includePaper: boolean
  includeRepo: boolean
  includeMappings: boolean
  includeTests: boolean
  includeGraph: boolean
}

interface ReportGeneratorProps {
  selectedFormat: ReportFormat
  onFormatChange: (format: ReportFormat) => void
  options: ReportOptions
  onOptionsChange: (options: ReportOptions) => void
  onGenerate: () => void
  isGenerating?: boolean
  className?: string
}

const formatOptions: {
  id: ReportFormat
  label: string
  description: string
  icon: React.ComponentType<{ className?: string }>
}[] = [
  {
    id: 'html',
    label: 'HTML Report',
    description: 'Interactive report with visualizations',
    icon: GlobeAltIcon,
  },
  {
    id: 'markdown',
    label: 'Markdown',
    description: 'Text format for documentation',
    icon: DocumentTextIcon,
  },
  {
    id: 'json',
    label: 'JSON Export',
    description: 'Structured data for programmatic access',
    icon: CodeBracketIcon,
  },
]

const sectionOptions: {
  key: keyof ReportOptions
  label: string
  description: string
}[] = [
  {
    key: 'includePaper',
    label: 'Paper Analysis',
    description: 'Include paper metadata, concepts, and algorithms',
  },
  {
    key: 'includeRepo',
    label: 'Repository Analysis',
    description: 'Include repository structure and components',
  },
  {
    key: 'includeMappings',
    label: 'Concept Mappings',
    description: 'Include all concept-to-code mappings with evidence',
  },
  {
    key: 'includeTests',
    label: 'Test Results',
    description: 'Include code execution and test results',
  },
  {
    key: 'includeGraph',
    label: 'Knowledge Graph',
    description: 'Include knowledge graph data (HTML only)',
  },
]

export function ReportGenerator({
  selectedFormat,
  onFormatChange,
  options,
  onOptionsChange,
  onGenerate,
  isGenerating = false,
  className,
}: ReportGeneratorProps) {
  const handleOptionToggle = (key: keyof ReportOptions) => {
    onOptionsChange({
      ...options,
      [key]: !options[key],
    })
  }

  return (
    <div className={cn('space-y-6', className)}>
      {/* Format Selection */}
      <div>
        <h3 className="text-heading-3 text-text-primary mb-4">Report Format</h3>
        <div className="grid gap-3 sm:grid-cols-3">
          {formatOptions.map((format) => {
            const isSelected = selectedFormat === format.id
            const Icon = format.icon
            return (
              <button
                key={format.id}
                onClick={() => onFormatChange(format.id)}
                className={cn(
                  'p-4 rounded-xl border text-left transition-all duration-200',
                  isSelected
                    ? 'border-accent-primary bg-accent-primary/10'
                    : 'border-border bg-bg-tertiary hover:border-border-glow'
                )}
              >
                <div className="flex items-center gap-3 mb-2">
                  <div
                    className={cn(
                      'w-10 h-10 rounded-lg flex items-center justify-center',
                      isSelected ? 'bg-accent-primary/20' : 'bg-bg-secondary'
                    )}
                  >
                    <Icon
                      className={cn(
                        'h-5 w-5',
                        isSelected ? 'text-accent-primary' : 'text-text-muted'
                      )}
                    />
                  </div>
                  <div
                    className={cn(
                      'w-4 h-4 rounded-full border-2 flex items-center justify-center',
                      isSelected ? 'border-accent-primary' : 'border-text-muted'
                    )}
                  >
                    {isSelected && (
                      <div className="w-2 h-2 rounded-full bg-accent-primary" />
                    )}
                  </div>
                </div>
                <h4
                  className={cn(
                    'text-body font-medium mb-1',
                    isSelected ? 'text-text-primary' : 'text-text-secondary'
                  )}
                >
                  {format.label}
                </h4>
                <p className="text-caption text-text-muted">{format.description}</p>
              </button>
            )
          })}
        </div>
      </div>

      {/* Section Options */}
      <GlassCard title="Include Sections">
        <div className="space-y-4">
          {sectionOptions.map((section) => {
            const isDisabled =
              section.key === 'includeGraph' && selectedFormat !== 'html'
            return (
              <Toggle
                key={section.key}
                label={section.label}
                description={
                  isDisabled
                    ? 'Only available for HTML format'
                    : section.description
                }
                checked={options[section.key] && !isDisabled}
                onChange={() => handleOptionToggle(section.key)}
                disabled={isDisabled}
              />
            )
          })}
        </div>
      </GlassCard>

      {/* Generate Button */}
      <Button
        variant="primary"
        size="lg"
        onClick={onGenerate}
        loading={isGenerating}
        leftIcon={<DocumentArrowDownIcon className="h-5 w-5" />}
        className="w-full"
      >
        {isGenerating ? 'Generating Report...' : 'Generate Report'}
      </Button>
    </div>
  )
}

export default ReportGenerator
