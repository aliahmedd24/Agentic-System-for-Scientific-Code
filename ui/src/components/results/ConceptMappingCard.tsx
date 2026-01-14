import { useState } from 'react'
import { ChevronDownIcon, ChevronUpIcon, CodeBracketIcon } from '@heroicons/react/24/outline'
import { cn } from '@/lib/cn'
import { GlassCard } from '@/components/ui/GlassCard'
import { ConfidenceMeter } from '@/components/data-display/ConfidenceMeter'
import type { MappingResult } from '@/api/types'

interface ConceptMappingCardProps {
  mapping: MappingResult
  expanded?: boolean
  onToggle?: () => void
  className?: string
}

interface SignalBarProps {
  label: string
  value: number
  color: string
}

function SignalBar({ label, value, color }: SignalBarProps) {
  const percentage = Math.round(value * 100)
  return (
    <div className="flex items-center gap-2">
      <span className="text-caption text-text-muted w-20 shrink-0">{label}</span>
      <div className="flex-1 h-1.5 bg-bg-tertiary rounded-full overflow-hidden">
        <div
          className={cn('h-full rounded-full transition-all duration-300', color)}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="text-caption text-text-muted w-8 text-right">{percentage}%</span>
    </div>
  )
}

export function ConceptMappingCard({
  mapping,
  expanded: controlledExpanded,
  onToggle,
  className,
}: ConceptMappingCardProps) {
  const [internalExpanded, setInternalExpanded] = useState(false)
  const isControlled = controlledExpanded !== undefined
  const isExpanded = isControlled ? controlledExpanded : internalExpanded

  const handleToggle = () => {
    if (isControlled) {
      onToggle?.()
    } else {
      setInternalExpanded(!internalExpanded)
    }
  }

  const confidenceColor = mapping.confidence >= 0.8
    ? 'text-status-success'
    : mapping.confidence >= 0.6
      ? 'text-status-warning'
      : mapping.confidence >= 0.4
        ? 'text-orange-500'
        : 'text-status-error'

  return (
    <GlassCard
      variant="interactive"
      noPadding
      className={cn('overflow-hidden', className)}
    >
      {/* Header - always visible */}
      <button
        onClick={handleToggle}
        className={cn(
          'w-full p-4 text-left',
          'flex items-start justify-between gap-4',
          'hover:bg-bg-tertiary/50 transition-colors duration-200'
        )}
      >
        <div className="flex-1 min-w-0">
          {/* Concept name and code element */}
          <div className="flex items-center gap-2 mb-1">
            <h4 className="text-body font-medium text-text-primary truncate">
              {mapping.concept_name}
            </h4>
            <span className="text-text-muted">→</span>
            <span className="text-body-sm text-accent-secondary font-mono truncate">
              {mapping.code_element}
            </span>
          </div>

          {/* Code file path */}
          <div className="flex items-center gap-1.5 text-caption text-text-muted">
            <CodeBracketIcon className="h-3.5 w-3.5" />
            <span className="truncate">{mapping.code_file}</span>
          </div>
        </div>

        {/* Confidence and expand button */}
        <div className="flex items-center gap-3">
          <div className="text-right">
            <span className={cn('text-heading-3 font-semibold', confidenceColor)}>
              {Math.round(mapping.confidence * 100)}%
            </span>
          </div>
          {isExpanded ? (
            <ChevronUpIcon className="h-5 w-5 text-text-muted" />
          ) : (
            <ChevronDownIcon className="h-5 w-5 text-text-muted" />
          )}
        </div>
      </button>

      {/* Expanded content */}
      {isExpanded && (
        <div className="px-4 pb-4 space-y-4 border-t border-border">
          {/* Concept description */}
          <div className="pt-4">
            <h5 className="text-body-sm font-medium text-text-secondary mb-1">
              Concept Description
            </h5>
            <p className="text-body-sm text-text-muted">
              {mapping.concept_description}
            </p>
          </div>

          {/* Match signals */}
          <div>
            <h5 className="text-body-sm font-medium text-text-secondary mb-2">
              Match Signals
            </h5>
            <div className="space-y-1.5">
              <SignalBar
                label="Lexical"
                value={mapping.match_signals.lexical}
                color="bg-status-info"
              />
              <SignalBar
                label="Semantic"
                value={mapping.match_signals.semantic}
                color="bg-accent-primary"
              />
              <SignalBar
                label="Documentary"
                value={mapping.match_signals.documentary}
                color="bg-status-success"
              />
            </div>
          </div>

          {/* Evidence */}
          {mapping.evidence.length > 0 && (
            <div>
              <h5 className="text-body-sm font-medium text-text-secondary mb-2">
                Evidence
              </h5>
              <ul className="space-y-1">
                {mapping.evidence.map((item, idx) => (
                  <li
                    key={idx}
                    className="text-body-sm text-text-muted flex items-start gap-2"
                  >
                    <span className="text-accent-primary mt-1.5">•</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Reasoning */}
          {mapping.reasoning && (
            <div>
              <h5 className="text-body-sm font-medium text-text-secondary mb-1">
                Reasoning
              </h5>
              <p className="text-body-sm text-text-muted">{mapping.reasoning}</p>
            </div>
          )}
        </div>
      )}
    </GlassCard>
  )
}

export default ConceptMappingCard
