import { cn } from '@/lib/cn'
import { PIPELINE_STAGES, type PipelineStageId } from '@/lib/constants'
import {
  CheckIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline'

interface PipelineStepsProps {
  currentStage: string
  progress: number
  status?: 'running' | 'completed' | 'failed' | 'pending'
  compact?: boolean
  className?: string
}

// Icons for each stage (optional)
const stageIcons: Record<string, string> = {
  parsing_paper: '📄',
  analyzing_repo: '🔍',
  mapping_concepts: '🔗',
  generating_code: '⚙️',
  setting_up_env: '🛠️',
  executing_code: '▶️',
  generating_report: '📊',
}

export function PipelineSteps({
  currentStage,
  progress,
  status = 'running',
  compact = false,
  className,
}: PipelineStepsProps) {
  // Filter out initialized and completed - show only the main stages
  const displayStages = PIPELINE_STAGES.filter(
    (stage) => !['initialized', 'completed'].includes(stage.id)
  )

  const getCurrentStageIndex = () => {
    return displayStages.findIndex((s) => s.id === currentStage)
  }

  const currentIndex = getCurrentStageIndex()

  const getStageStatus = (stageIndex: number) => {
    if (status === 'failed' && stageIndex === currentIndex) return 'failed'
    if (status === 'completed' || stageIndex < currentIndex) return 'completed'
    if (stageIndex === currentIndex) return 'current'
    return 'pending'
  }

  if (compact) {
    return (
      <div className={cn('flex items-center gap-1', className)}>
        {displayStages.map((stage, idx) => {
          const stageStatus = getStageStatus(idx)
          return (
            <div
              key={stage.id}
              className={cn(
                'w-2 h-2 rounded-full transition-all duration-300',
                stageStatus === 'completed' && 'bg-status-success',
                stageStatus === 'current' && 'bg-accent-primary animate-pulse',
                stageStatus === 'failed' && 'bg-status-error',
                stageStatus === 'pending' && 'bg-text-muted/30'
              )}
              title={stage.label}
            />
          )
        })}
      </div>
    )
  }

  return (
    <div className={cn('w-full', className)}>
      <div className="flex items-center justify-between">
        {displayStages.map((stage, idx) => {
          const stageStatus = getStageStatus(idx)
          const isLast = idx === displayStages.length - 1

          return (
            <div key={stage.id} className="flex items-center flex-1">
              {/* Step indicator */}
              <div className="flex flex-col items-center">
                <div
                  className={cn(
                    'flex items-center justify-center w-10 h-10 rounded-full border-2 transition-all duration-300',
                    stageStatus === 'completed' &&
                      'bg-status-success/20 border-status-success text-status-success',
                    stageStatus === 'current' &&
                      'bg-accent-primary/20 border-accent-primary text-accent-primary animate-pulse shadow-glow-sm',
                    stageStatus === 'failed' &&
                      'bg-status-error/20 border-status-error text-status-error',
                    stageStatus === 'pending' && 'border-border text-text-muted'
                  )}
                >
                  {stageStatus === 'completed' && (
                    <CheckIcon className="h-5 w-5" />
                  )}
                  {stageStatus === 'failed' && (
                    <XMarkIcon className="h-5 w-5" />
                  )}
                  {stageStatus === 'current' && (
                    <span className="text-lg">{stageIcons[stage.id] || '⏳'}</span>
                  )}
                  {stageStatus === 'pending' && (
                    <span className="text-body-sm font-medium">{idx + 1}</span>
                  )}
                </div>
                <span
                  className={cn(
                    'mt-2 text-caption font-medium text-center whitespace-nowrap',
                    stageStatus === 'completed' && 'text-status-success',
                    stageStatus === 'current' && 'text-accent-primary',
                    stageStatus === 'failed' && 'text-status-error',
                    stageStatus === 'pending' && 'text-text-muted'
                  )}
                >
                  {stage.label}
                </span>
              </div>

              {/* Connector line */}
              {!isLast && (
                <div className="flex-1 mx-2 h-0.5 -mt-6">
                  <div
                    className={cn(
                      'h-full rounded-full transition-all duration-500',
                      stageStatus === 'completed' || idx < currentIndex
                        ? 'bg-status-success'
                        : 'bg-border'
                    )}
                  />
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default PipelineSteps
