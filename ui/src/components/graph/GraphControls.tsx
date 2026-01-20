import { cn } from '@/lib/cn'
import { Button } from '@/components/ui/Button'
import {
  MagnifyingGlassPlusIcon,
  MagnifyingGlassMinusIcon,
  ArrowPathIcon,
  ArrowsPointingOutIcon,
  ArrowsPointingInIcon,
  ViewfinderCircleIcon,
} from '@heroicons/react/24/outline'

interface GraphControlsProps {
  onZoomIn: () => void
  onZoomOut: () => void
  onResetView: () => void
  onToggleFullscreen: () => void
  onFitToView: () => void
  isFullscreen: boolean
  zoomLevel: number
  className?: string
}

export function GraphControls({
  onZoomIn,
  onZoomOut,
  onResetView,
  onToggleFullscreen,
  onFitToView,
  isFullscreen,
  zoomLevel,
  className,
}: GraphControlsProps) {
  const zoomPercentage = Math.round(zoomLevel * 100)

  return (
    <div
      className={cn(
        'flex flex-col gap-1.5 p-2 rounded-xl bg-bg-glass/90 backdrop-blur-[20px] border border-border',
        className
      )}
    >
      {/* Zoom In */}
      <Button
        variant="icon"
        size="sm"
        onClick={onZoomIn}
        title="Zoom In"
        aria-label="Zoom In"
      >
        <MagnifyingGlassPlusIcon className="h-4 w-4" />
      </Button>

      {/* Zoom Level Indicator */}
      <div
        className="flex items-center justify-center text-caption text-text-muted py-1"
        title={`Zoom: ${zoomPercentage}%`}
      >
        {zoomPercentage}%
      </div>

      {/* Zoom Out */}
      <Button
        variant="icon"
        size="sm"
        onClick={onZoomOut}
        title="Zoom Out"
        aria-label="Zoom Out"
      >
        <MagnifyingGlassMinusIcon className="h-4 w-4" />
      </Button>

      {/* Divider */}
      <div className="h-px bg-border my-1" />

      {/* Fit to View */}
      <Button
        variant="icon"
        size="sm"
        onClick={onFitToView}
        title="Fit to View"
        aria-label="Fit to View"
      >
        <ViewfinderCircleIcon className="h-4 w-4" />
      </Button>

      {/* Reset View */}
      <Button
        variant="icon"
        size="sm"
        onClick={onResetView}
        title="Reset View"
        aria-label="Reset View"
      >
        <ArrowPathIcon className="h-4 w-4" />
      </Button>

      {/* Divider */}
      <div className="h-px bg-border my-1" />

      {/* Fullscreen Toggle */}
      <Button
        variant="icon"
        size="sm"
        onClick={onToggleFullscreen}
        title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
        aria-label={isFullscreen ? 'Exit Fullscreen' : 'Enter Fullscreen'}
      >
        {isFullscreen ? (
          <ArrowsPointingInIcon className="h-4 w-4" />
        ) : (
          <ArrowsPointingOutIcon className="h-4 w-4" />
        )}
      </Button>
    </div>
  )
}

export default GraphControls
