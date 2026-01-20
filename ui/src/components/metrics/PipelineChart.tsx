import { useRef, useEffect, useState, useCallback } from 'react'
import * as d3 from 'd3'
import { cn } from '@/lib/cn'
import { GlassCard } from '@/components/ui/GlassCard'
import { formatDuration } from '@/lib/formatters'
import type { PipelineStageMetric } from '@/api/types'

interface PipelineChartProps {
  stages: PipelineStageMetric[]
  height?: number
  animate?: boolean
  className?: string
}

// Color scale based on success rate
const getBarColor = (successRate: number): string => {
  if (successRate >= 0.9) return '#22c55e' // status-success
  if (successRate >= 0.7) return '#f59e0b' // status-warning
  return '#ef4444' // status-error
}

export function PipelineChart({
  stages,
  height = 300,
  animate = true,
  className,
}: PipelineChartProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 600, height })

  // Margins for the chart
  const margin = { top: 20, right: 120, bottom: 40, left: 140 }
  const chartWidth = dimensions.width - margin.left - margin.right
  const chartHeight = dimensions.height - margin.top - margin.bottom

  // Resize observer
  useEffect(() => {
    if (!containerRef.current) return

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (entry) {
        const { width } = entry.contentRect
        setDimensions({ width, height })
      }
    })

    observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [height])

  // Show tooltip
  const showTooltip = useCallback((event: MouseEvent, stage: PipelineStageMetric) => {
    if (!tooltipRef.current) return

    const tooltip = tooltipRef.current
    tooltip.innerHTML = `
      <div class="font-medium text-text-primary mb-1">${stage.stage.replace(/_/g, ' ')}</div>
      <div class="text-caption text-text-secondary">
        <div>Duration: ${formatDuration(stage.avg_duration_ms)}</div>
        <div>Success Rate: ${(stage.success_rate * 100).toFixed(1)}%</div>
        <div>Executions: ${stage.count}</div>
      </div>
    `
    tooltip.style.opacity = '1'
    tooltip.style.left = `${event.pageX + 10}px`
    tooltip.style.top = `${event.pageY - 10}px`
  }, [])

  // Hide tooltip
  const hideTooltip = useCallback(() => {
    if (!tooltipRef.current) return
    tooltipRef.current.style.opacity = '0'
  }, [])

  // Render chart with D3
  useEffect(() => {
    if (!svgRef.current || stages.length === 0) return

    const svg = d3.select(svgRef.current)

    // Clear existing content
    svg.selectAll('*').remove()

    // Create main group
    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // X scale (duration)
    const maxDuration = d3.max(stages, (d) => d.avg_duration_ms) || 1000
    const xScale = d3.scaleLinear().domain([0, maxDuration]).range([0, chartWidth])

    // Y scale (stages)
    const yScale = d3
      .scaleBand()
      .domain(stages.map((d) => d.stage))
      .range([0, chartHeight])
      .padding(0.3)

    // Add X axis
    const xAxis = d3.axisBottom(xScale).ticks(5).tickFormat((d) => formatDuration(d as number))

    g.append('g')
      .attr('transform', `translate(0,${chartHeight})`)
      .attr('class', 'x-axis')
      .call(xAxis)
      .selectAll('text')
      .attr('fill', '#94a3b8') // text-text-muted
      .attr('font-size', '11px')

    g.selectAll('.x-axis path, .x-axis line').attr('stroke', '#334155') // border color

    // Add Y axis
    const yAxis = d3.axisLeft(yScale).tickFormat((d) => d.replace(/_/g, ' '))

    g.append('g')
      .attr('class', 'y-axis')
      .call(yAxis)
      .selectAll('text')
      .attr('fill', '#e2e8f0') // text-text-primary
      .attr('font-size', '12px')
      .style('text-transform', 'capitalize')

    g.selectAll('.y-axis path, .y-axis line').attr('stroke', '#334155')

    // Add grid lines
    g.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0,${chartHeight})`)
      .call(
        d3.axisBottom(xScale)
          .ticks(5)
          .tickSize(-chartHeight)
          .tickFormat(() => '')
      )
      .selectAll('line')
      .attr('stroke', '#1e293b')
      .attr('stroke-opacity', 0.5)

    g.selectAll('.grid path').attr('stroke', 'none')

    // Add bars with animation
    const bars = g
      .selectAll('.bar')
      .data(stages)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('y', (d) => yScale(d.stage) || 0)
      .attr('height', yScale.bandwidth())
      .attr('fill', (d) => getBarColor(d.success_rate))
      .attr('rx', 4)
      .attr('cursor', 'pointer')

    if (animate) {
      bars
        .attr('x', 0)
        .attr('width', 0)
        .transition()
        .duration(800)
        .delay((_, i) => i * 100)
        .attr('width', (d) => xScale(d.avg_duration_ms))
    } else {
      bars.attr('x', 0).attr('width', (d) => xScale(d.avg_duration_ms))
    }

    // Add value labels at end of bars
    const labels = g
      .selectAll('.label')
      .data(stages)
      .enter()
      .append('text')
      .attr('class', 'label')
      .attr('y', (d) => (yScale(d.stage) || 0) + yScale.bandwidth() / 2)
      .attr('dy', '0.35em')
      .attr('fill', '#e2e8f0')
      .attr('font-size', '11px')
      .attr('font-weight', '500')
      .text((d) => formatDuration(d.avg_duration_ms))

    if (animate) {
      labels
        .attr('x', 8)
        .attr('opacity', 0)
        .transition()
        .duration(400)
        .delay((_, i) => i * 100 + 600)
        .attr('x', (d) => xScale(d.avg_duration_ms) + 8)
        .attr('opacity', 1)
    } else {
      labels.attr('x', (d) => xScale(d.avg_duration_ms) + 8)
    }

    // Add interactivity
    bars
      .on('mouseenter', function (event, d) {
        d3.select(this)
          .transition()
          .duration(150)
          .attr('opacity', 0.8)
        showTooltip(event, d)
      })
      .on('mousemove', function (event, d) {
        showTooltip(event, d)
      })
      .on('mouseleave', function () {
        d3.select(this)
          .transition()
          .duration(150)
          .attr('opacity', 1)
        hideTooltip()
      })

  }, [stages, chartWidth, chartHeight, margin, animate, showTooltip, hideTooltip])

  if (stages.length === 0) {
    return (
      <GlassCard title="Stage Timing Distribution" className={className}>
        <div className="py-12 text-center text-text-muted">
          No pipeline data available
        </div>
      </GlassCard>
    )
  }

  return (
    <GlassCard title="Stage Timing Distribution" className={className}>
      <div
        ref={containerRef}
        className="relative"
        role="img"
        aria-label="Pipeline stage timing distribution chart"
      >
        <svg
          ref={svgRef}
          width={dimensions.width}
          height={dimensions.height}
          className="overflow-visible"
        >
          <title>Pipeline Stage Timing Distribution</title>
          <desc>
            A horizontal bar chart showing the average duration of each pipeline stage.
          </desc>
        </svg>

        {/* Tooltip */}
        <div
          ref={tooltipRef}
          className={cn(
            'fixed z-50 px-3 py-2 rounded-lg pointer-events-none transition-opacity duration-150',
            'bg-bg-glass/95 backdrop-blur-[20px] border border-border shadow-lg'
          )}
          style={{ opacity: 0 }}
        />
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-4 pt-4 border-t border-border">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-sm bg-status-success" />
          <span className="text-caption text-text-muted">≥90% Success</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-sm bg-status-warning" />
          <span className="text-caption text-text-muted">70-89% Success</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-sm bg-status-error" />
          <span className="text-caption text-text-muted">&lt;70% Success</span>
        </div>
      </div>
    </GlassCard>
  )
}

export default PipelineChart
