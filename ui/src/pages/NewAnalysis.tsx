import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { PageHeader } from '@/components/layout/PageHeader'
import { GlassCard } from '@/components/ui/GlassCard'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Select, type SelectOption } from '@/components/ui/Select'
import { Toggle } from '@/components/ui/Toggle'
import { FileUpload } from '@/components/ui/FileUpload'
import { useJobsStore } from '@/stores/jobsStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { toast } from '@/components/ui/Toast'
import { LLM_PROVIDERS } from '@/lib/constants'
import {
  DocumentTextIcon,
  CodeBracketIcon,
  Cog6ToothIcon,
  ArrowRightIcon,
  ArrowLeftIcon,
  CheckIcon,
  LinkIcon,
  DocumentArrowUpIcon,
} from '@heroicons/react/24/outline'

type Step = 1 | 2 | 3
type InputMethod = 'arxiv' | 'url' | 'upload'

const steps = [
  { id: 1, name: 'Paper Source', icon: DocumentTextIcon },
  { id: 2, name: 'Repository', icon: CodeBracketIcon },
  { id: 3, name: 'Settings', icon: Cog6ToothIcon },
]

const inputMethods: { id: InputMethod; label: string; icon: typeof DocumentTextIcon }[] = [
  { id: 'arxiv', label: 'arXiv ID', icon: DocumentTextIcon },
  { id: 'url', label: 'Paper URL', icon: LinkIcon },
  { id: 'upload', label: 'Upload PDF', icon: DocumentArrowUpIcon },
]

const llmOptions: SelectOption[] = LLM_PROVIDERS.map((p) => ({
  id: p.id,
  label: p.label,
  description: p.description,
}))

export default function NewAnalysis() {
  const navigate = useNavigate()
  const { startAnalysis, startAnalysisWithUpload, isSubmitting } = useJobsStore()
  const { defaultLLMProvider, autoExecute } = useSettingsStore()

  // Wizard state
  const [currentStep, setCurrentStep] = useState<Step>(1)

  // Step 1: Paper input
  const [inputMethod, setInputMethod] = useState<InputMethod>('arxiv')
  const [arxivId, setArxivId] = useState('')
  const [paperUrl, setPaperUrl] = useState('')
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)

  // Step 2: Repository
  const [repoUrl, setRepoUrl] = useState('')
  const [repoError, setRepoError] = useState<string | null>(null)

  // Step 3: Settings
  const [llmProvider, setLlmProvider] = useState(defaultLLMProvider)
  const [shouldAutoExecute, setShouldAutoExecute] = useState(autoExecute)

  // Computed values
  const getPaperSource = useCallback(() => {
    switch (inputMethod) {
      case 'arxiv':
        return arxivId.trim()
      case 'url':
        return paperUrl.trim()
      case 'upload':
        return uploadedFile ? 'upload' : ''
    }
  }, [inputMethod, arxivId, paperUrl, uploadedFile])

  const canProceedStep1 = (() => {
    switch (inputMethod) {
      case 'arxiv':
        return arxivId.trim().length > 0
      case 'url':
        return paperUrl.trim().length > 0 && paperUrl.includes('http')
      case 'upload':
        return uploadedFile !== null
    }
  })()

  const validateRepoUrl = (url: string) => {
    if (!url.trim()) {
      setRepoError('Repository URL is required')
      return false
    }
    if (!url.includes('github.com')) {
      setRepoError('Please enter a valid GitHub URL')
      return false
    }
    setRepoError(null)
    return true
  }

  const canProceedStep2 = repoUrl.trim().length > 0 && !repoError

  const handleNext = () => {
    if (currentStep === 2) {
      if (!validateRepoUrl(repoUrl)) return
    }
    if (currentStep < 3) {
      setCurrentStep((currentStep + 1) as Step)
    }
  }

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep((currentStep - 1) as Step)
    }
  }

  const handleSubmit = async () => {
    try {
      let response
      if (inputMethod === 'upload' && uploadedFile) {
        response = await startAnalysisWithUpload(
          uploadedFile,
          repoUrl,
          llmProvider,
          shouldAutoExecute
        )
      } else {
        response = await startAnalysis({
          paper_source: getPaperSource(),
          repo_url: repoUrl,
          llm_provider: llmProvider,
          auto_execute: shouldAutoExecute,
        })
      }

      toast.success('Analysis Started', `Job ${response.job_id.slice(0, 8)} created successfully`)
      navigate(`/jobs/${response.job_id}`)
    } catch (err) {
      toast.error('Failed to start analysis', err instanceof Error ? err.message : 'Unknown error')
    }
  }

  const handleInputMethodChange = (method: InputMethod) => {
    setInputMethod(method)
    // Clear other inputs when switching
    if (method !== 'arxiv') setArxivId('')
    if (method !== 'url') setPaperUrl('')
    if (method !== 'upload') setUploadedFile(null)
  }

  return (
    <div className="animate-in">
      <PageHeader
        title="New Analysis"
        subtitle="Analyze a scientific paper and map concepts to code implementations"
        breadcrumbs={[
          { label: 'Dashboard', href: '/' },
          { label: 'New Analysis' },
        ]}
      />

      {/* Step Indicator */}
      <div className="mb-8">
        <div className="flex items-center justify-between max-w-2xl mx-auto">
          {steps.map((step, idx) => (
            <div key={step.id} className="flex items-center flex-1">
              <div className="flex flex-col items-center">
                <button
                  onClick={() => {
                    if (step.id < currentStep) setCurrentStep(step.id as Step)
                  }}
                  disabled={step.id > currentStep}
                  className={`flex items-center justify-center w-12 h-12 rounded-full border-2 transition-all duration-300 ${
                    currentStep > step.id
                      ? 'bg-status-success border-status-success text-white cursor-pointer hover:bg-status-success/80'
                      : currentStep === step.id
                      ? 'bg-accent-primary/20 border-accent-primary text-accent-primary shadow-glow-sm'
                      : 'border-border text-text-muted cursor-not-allowed'
                  }`}
                >
                  {currentStep > step.id ? (
                    <CheckIcon className="h-6 w-6" />
                  ) : (
                    <step.icon className="h-6 w-6" />
                  )}
                </button>
                <span
                  className={`mt-2 text-body-sm font-medium ${
                    currentStep >= step.id ? 'text-text-primary' : 'text-text-muted'
                  }`}
                >
                  {step.name}
                </span>
              </div>
              {idx < steps.length - 1 && (
                <div
                  className={`flex-1 h-0.5 mx-4 transition-colors duration-300 ${
                    currentStep > step.id ? 'bg-status-success' : 'bg-border'
                  }`}
                />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Step Content */}
      <div className="max-w-2xl mx-auto">
        <GlassCard>
          {/* Step 1: Paper Source */}
          {currentStep === 1 && (
            <div className="space-y-6">
              <div>
                <h3 className="text-heading-3 text-text-primary mb-2">Paper Source</h3>
                <p className="text-body-sm text-text-secondary">
                  Choose how you want to provide the scientific paper for analysis
                </p>
              </div>

              {/* Input Method Tabs */}
              <div className="flex gap-2 p-1 bg-bg-tertiary rounded-lg">
                {inputMethods.map((method) => (
                  <button
                    key={method.id}
                    onClick={() => handleInputMethodChange(method.id)}
                    className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-md text-body-sm font-medium transition-all ${
                      inputMethod === method.id
                        ? 'bg-accent-primary text-white shadow-md'
                        : 'text-text-secondary hover:text-text-primary hover:bg-bg-secondary'
                    }`}
                  >
                    <method.icon className="h-5 w-5" />
                    {method.label}
                  </button>
                ))}
              </div>

              {/* Input Fields based on method */}
              <div className="min-h-[180px]">
                {inputMethod === 'arxiv' && (
                  <Input
                    label="arXiv ID"
                    placeholder="e.g., 1706.03762 or 2301.00001"
                    value={arxivId}
                    onChange={(e) => setArxivId(e.target.value)}
                    hint="Enter the arXiv paper ID (numbers after 'arxiv.org/abs/')"
                  />
                )}

                {inputMethod === 'url' && (
                  <Input
                    label="Paper URL"
                    placeholder="https://arxiv.org/abs/1706.03762"
                    value={paperUrl}
                    onChange={(e) => setPaperUrl(e.target.value)}
                    hint="Direct link to the paper (arXiv, PDF, or other supported sources)"
                    leftIcon={<LinkIcon className="h-5 w-5" />}
                  />
                )}

                {inputMethod === 'upload' && (
                  <FileUpload
                    label="Upload PDF"
                    accept="application/pdf"
                    maxSize={50 * 1024 * 1024}
                    file={uploadedFile}
                    onFile={setUploadedFile}
                    hint="Maximum file size: 50MB"
                  />
                )}
              </div>
            </div>
          )}

          {/* Step 2: Repository */}
          {currentStep === 2 && (
            <div className="space-y-6">
              <div>
                <h3 className="text-heading-3 text-text-primary mb-2">Code Repository</h3>
                <p className="text-body-sm text-text-secondary">
                  Provide the GitHub repository containing the paper's implementation
                </p>
              </div>

              <Input
                label="GitHub Repository URL"
                placeholder="https://github.com/username/repository"
                value={repoUrl}
                onChange={(e) => {
                  setRepoUrl(e.target.value)
                  if (repoError) validateRepoUrl(e.target.value)
                }}
                onBlur={() => repoUrl && validateRepoUrl(repoUrl)}
                error={repoError || undefined}
                leftIcon={<CodeBracketIcon className="h-5 w-5" />}
                hint="The repository should contain the implementation of concepts from the paper"
              />

              <div className="p-4 rounded-lg bg-bg-tertiary/50 border border-border">
                <h4 className="text-body-sm font-medium text-text-primary mb-2">
                  What we'll analyze:
                </h4>
                <ul className="space-y-1 text-body-sm text-text-secondary">
                  <li className="flex items-center gap-2">
                    <CheckIcon className="h-4 w-4 text-status-success" />
                    Repository structure and file organization
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckIcon className="h-4 w-4 text-status-success" />
                    Classes, functions, and their relationships
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckIcon className="h-4 w-4 text-status-success" />
                    Dependencies and setup requirements
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckIcon className="h-4 w-4 text-status-success" />
                    Entry points and execution patterns
                  </li>
                </ul>
              </div>
            </div>
          )}

          {/* Step 3: Settings */}
          {currentStep === 3 && (
            <div className="space-y-6">
              <div>
                <h3 className="text-heading-3 text-text-primary mb-2">Analysis Settings</h3>
                <p className="text-body-sm text-text-secondary">
                  Configure how the analysis should be performed
                </p>
              </div>

              <Select
                label="Language Model"
                options={llmOptions}
                value={llmProvider}
                onChange={(value) => setLlmProvider(value as typeof llmProvider)}
                hint="Choose the AI model for analysis"
              />

              <div className="p-4 rounded-lg border border-border">
                <Toggle
                  label="Auto-Execute Generated Code"
                  description="Automatically run test code to validate concept mappings"
                  checked={shouldAutoExecute}
                  onChange={setShouldAutoExecute}
                />
              </div>

              {/* Summary */}
              <div className="p-4 rounded-lg bg-accent-primary/10 border border-accent-primary/30">
                <h4 className="text-body-sm font-semibold text-accent-secondary mb-3">
                  Analysis Summary
                </h4>
                <dl className="space-y-2 text-body-sm">
                  <div className="flex justify-between">
                    <dt className="text-text-secondary">Paper Source:</dt>
                    <dd className="text-text-primary font-medium truncate max-w-[250px]">
                      {inputMethod === 'upload' ? uploadedFile?.name : getPaperSource() || '-'}
                    </dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-text-secondary">Repository:</dt>
                    <dd className="text-text-primary font-medium truncate max-w-[250px]">
                      {repoUrl || '-'}
                    </dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-text-secondary">LLM Provider:</dt>
                    <dd className="text-text-primary font-medium">
                      {LLM_PROVIDERS.find((p) => p.id === llmProvider)?.label}
                    </dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-text-secondary">Auto-Execute:</dt>
                    <dd className="text-text-primary font-medium">
                      {shouldAutoExecute ? 'Yes' : 'No'}
                    </dd>
                  </div>
                </dl>
              </div>
            </div>
          )}

          {/* Navigation */}
          <div className="flex items-center justify-between mt-8 pt-6 border-t border-border">
            <Button
              variant="ghost"
              onClick={handleBack}
              disabled={currentStep === 1}
              leftIcon={<ArrowLeftIcon className="h-5 w-5" />}
            >
              Back
            </Button>

            {currentStep < 3 ? (
              <Button
                onClick={handleNext}
                disabled={
                  (currentStep === 1 && !canProceedStep1) ||
                  (currentStep === 2 && !canProceedStep2)
                }
                rightIcon={<ArrowRightIcon className="h-5 w-5" />}
              >
                Continue
              </Button>
            ) : (
              <Button
                onClick={handleSubmit}
                loading={isSubmitting}
                disabled={!canProceedStep1 || !canProceedStep2}
              >
                Start Analysis
              </Button>
            )}
          </div>
        </GlassCard>
      </div>
    </div>
  )
}
