import { cn } from '@/lib/cn'
import { NODE_TYPES } from '@/lib/constants'
import { Button } from '@/components/ui/Button'
import {
  CheckIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline'

interface NodeTypeFilterProps {
  activeTypes: Set<string>
  onToggleType: (type: string) => void
  onSelectAll: () => void
  onClearAll: () => void
  nodeCounts?: Record<string, number>
  className?: string
}

// Group node types by category
const NODE_TYPE_GROUPS = {
  'Paper': ['PAPER', 'SECTION', 'CONCEPT', 'ALGORITHM', 'EQUATION', 'FIGURE', 'CITATION'],
  'Code': ['REPOSITORY', 'FILE', 'CLASS', 'FUNCTION', 'DEPENDENCY', 'MODULE'],
  'Execution': ['TEST', 'RESULT', 'VISUALIZATION', 'ERROR'],
  'Meta': ['AGENT', 'INSIGHT', 'MAPPING'],
} as const

export function NodeTypeFilter({
  activeTypes,
  onToggleType,
  onSelectAll,
  onClearAll,
  nodeCounts,
  className,
}: NodeTypeFilterProps) {
  const allSelected = Object.keys(NODE_TYPES).every(type => activeTypes.has(type))
  const noneSelected = activeTypes.size === 0

  return (
    <div className={cn('space-y-4', className)}>
      {/* Quick Actions */}
      <div className="flex gap-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={onSelectAll}
          disabled={allSelected}
          className="flex-1"
          leftIcon={<CheckIcon className="h-4 w-4" />}
        >
          All
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={onClearAll}
          disabled={noneSelected}
          className="flex-1"
          leftIcon={<XMarkIcon className="h-4 w-4" />}
        >
          Clear
        </Button>
      </div>

      {/* Grouped Node Types */}
      <div className="space-y-4">
        {Object.entries(NODE_TYPE_GROUPS).map(([groupName, types]) => {
          const activeInGroup = types.filter(t => activeTypes.has(t)).length
          const totalInGroup = types.length

          return (
            <div key={groupName}>
              {/* Group Header */}
              <div className="flex items-center justify-between mb-2">
                <span className="text-caption text-text-muted uppercase tracking-wider">
                  {groupName}
                </span>
                <span className="text-caption text-text-muted">
                  {activeInGroup}/{totalInGroup}
                </span>
              </div>

              {/* Types in Group */}
              <div className="space-y-1">
                {types.map((typeKey) => {
                  const config = NODE_TYPES[typeKey as keyof typeof NODE_TYPES]
                  if (!config) return null

                  const isActive = activeTypes.has(typeKey)
                  const count = nodeCounts?.[typeKey] || nodeCounts?.[typeKey.toLowerCase()] || 0

                  return (
                    <label
                      key={typeKey}
                      className={cn(
                        'flex items-center gap-3 p-2 rounded-lg cursor-pointer transition-colors',
                        'hover:bg-bg-tertiary/50',
                        isActive ? 'bg-bg-tertiary/30' : 'opacity-60'
                      )}
                    >
                      {/* Checkbox */}
                      <input
                        type="checkbox"
                        checked={isActive}
                        onChange={() => onToggleType(typeKey)}
                        className="sr-only"
                      />
                      <div
                        className={cn(
                          'w-4 h-4 rounded border-2 flex items-center justify-center transition-colors',
                          isActive
                            ? 'bg-accent-primary border-accent-primary'
                            : 'border-border bg-transparent'
                        )}
                      >
                        {isActive && <CheckIcon className="h-3 w-3 text-white" />}
                      </div>

                      {/* Color Indicator */}
                      <span
                        className="w-3 h-3 rounded-full flex-shrink-0"
                        style={{ backgroundColor: config.color }}
                      />

                      {/* Label */}
                      <span className="text-body-sm text-text-primary flex-1">
                        {config.label}
                      </span>

                      {/* Count Badge */}
                      {nodeCounts && (
                        <span className="text-caption text-text-muted bg-bg-tertiary px-1.5 py-0.5 rounded">
                          {count}
                        </span>
                      )}
                    </label>
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

export default NodeTypeFilter
