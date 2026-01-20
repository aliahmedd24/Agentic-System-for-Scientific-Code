import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { PageHeader } from '@/components/layout/PageHeader'
import { Tabs, TabPanel } from '@/components/ui/Tabs'
import { Button } from '@/components/ui/Button'
import {
  ReportGenerator,
  ReportPreview,
  type ReportFormat,
  type ReportOptions,
} from '@/components/reports'
import { getReportUrl } from '@/api/endpoints'
import { useToastStore } from '@/components/ui/Toast'
import {
  Cog6ToothIcon,
  EyeIcon,
} from '@heroicons/react/24/outline'

const tabItems = [
  { label: 'Generate', icon: Cog6ToothIcon },
  { label: 'Preview', icon: EyeIcon },
]

const defaultOptions: ReportOptions = {
  includePaper: true,
  includeRepo: true,
  includeMappings: true,
  includeTests: true,
  includeGraph: true,
}

export default function Reports() {
  const { jobId } = useParams<{ jobId: string }>()
  const addToast = useToastStore((s) => s.addToast)

  const [selectedFormat, setSelectedFormat] = useState<ReportFormat>('html')
  const [options, setOptions] = useState<ReportOptions>(defaultOptions)
  const [isGenerating, setIsGenerating] = useState(false)
  const [activeTab, setActiveTab] = useState(0)

  const handleGenerate = async () => {
    if (!jobId) return

    setIsGenerating(true)

    try {
      // For now, we just open the report URL since the backend generates on-demand
      // In a more complete implementation, this could POST options to customize the report
      const url = getReportUrl(jobId, selectedFormat)

      // Simulate generation delay for UX
      await new Promise((resolve) => setTimeout(resolve, 500))

      window.open(url, '_blank')

      addToast({
        type: 'success',
        title: 'Report generated',
        message: `Your ${selectedFormat.toUpperCase()} report is ready`,
      })

      // Switch to preview tab
      setActiveTab(1)
    } catch (error) {
      addToast({
        type: 'error',
        title: 'Failed to generate report',
        message: error instanceof Error ? error.message : 'An error occurred',
      })
    } finally {
      setIsGenerating(false)
    }
  }

  if (!jobId) {
    return (
      <div className="animate-in">
        <PageHeader title="Reports" />
        <div className="py-16 text-center text-text-muted">
          No job ID provided
        </div>
      </div>
    )
  }

  return (
    <div className="animate-in">
      <PageHeader
        title="Reports"
        breadcrumbs={[
          { label: 'Dashboard', href: '/' },
          { label: 'Jobs', href: '/jobs' },
          { label: jobId.slice(0, 8), href: `/jobs/${jobId}` },
          { label: 'Reports' },
        ]}
        actions={
          <div className="flex items-center gap-3">
            <Link to={`/jobs/${jobId}/results`}>
              <Button variant="secondary">View Results</Button>
            </Link>
            <Link to={`/jobs/${jobId}/graph`}>
              <Button variant="secondary">Knowledge Graph</Button>
            </Link>
          </div>
        }
      />

      <Tabs
        items={tabItems}
        variant="default"
        selectedIndex={activeTab}
        onChange={setActiveTab}
      >
        {/* Generate Tab */}
        <TabPanel>
          <ReportGenerator
            selectedFormat={selectedFormat}
            onFormatChange={setSelectedFormat}
            options={options}
            onOptionsChange={setOptions}
            onGenerate={handleGenerate}
            isGenerating={isGenerating}
          />
        </TabPanel>

        {/* Preview Tab */}
        <TabPanel>
          <ReportPreview jobId={jobId} format={selectedFormat} />
        </TabPanel>
      </Tabs>
    </div>
  )
}
