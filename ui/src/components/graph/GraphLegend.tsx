import { cn } from '@/lib/cn'
import { NODE_TYPES } from '@/lib/constants'

interface GraphLegendProps {
  activeTypes: Set<string>
  compact?: boolean
  className?: string
}

// Group node types by category for organized display
const NODE_TYPE_GROUPS = {
  'Paper': ['PAPER', 'SECTION', 'CONCEPT', 'ALGORITHM', 'EQUATION', 'FIGURE', 'CITATION'],
  'Code': ['REPOSITORY', 'FILE', 'CLASS', 'FUNCTION', 'DEPENDENCY', 'MODULE'],
  'Execution': ['TEST', 'RESULT', 'VISUALIZATION', 'ERROR'],
  'Meta': ['AGENT', 'INSIGHT', 'MAPPING'],
} as const

export function GraphLegend({
  activeTypes,
  compact = false,
  className,
}: GraphLegendProps) {
  // Get only active types for display
  const activeEntries = Object.entries(NODE_TYPES)
    .filter(([key]) => activeTypes.has(key))

  if (compact) {
    // Compact view - horizontal list
    return (
      <div
        className={cn(
          'flex flex-wrap gap-3 p-3 rounded-lg',
          'bg-bg-glass/90 backdrop-blur-[20px] border border-border',
          className
        )}
      >
        {activeEntries.slice(0, 8).map(([key, config]) => (
          <div key={key} className="flex items-center gap-1.5">
            <span
              className="w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: config.color }}
            />
            <span className="text-caption text-text-muted">{config.label}</span>
          </div>
        ))}
        {activeEntries.length > 8 && (
          <span className="text-caption text-text-muted">
            +{activeEntries.length - 8} more
          </span>
        )}
      </div>
    )
  }

  // Full view - grouped list
  return (
    <div
      className={cn(
        'p-4 rounded-xl',
        'bg-bg-glass/90 backdrop-blur-[20px] border border-border',
        className
      )}
    >
      <h4 className="text-body-sm font-medium text-text-primary mb-3">Legend</h4>

      <div className="space-y-4">
        {Object.entries(NODE_TYPE_GROUPS).map(([groupName, types]) => {
          const activeInGroup = types.filter(t => activeTypes.has(t))
          if (activeInGroup.length === 0) return null

          return (
            <div key={groupName}>
              <div className="text-caption text-text-muted uppercase tracking-wider mb-2">
                {groupName}
              </div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                {activeInGroup.map((typeKey) => {
                  const config = NODE_TYPES[typeKey as keyof typeof NODE_TYPES]
                  if (!config) return null

                  return (
                    <div key={typeKey} className="flex items-center gap-2">
                      <span
                        className="w-3 h-3 rounded-full flex-shrink-0"
                        style={{ backgroundColor: config.color }}
                      />
                      <span className="text-body-sm text-text-secondary truncate">
                        {config.label}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default GraphLegend
