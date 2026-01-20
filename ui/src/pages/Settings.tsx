import { useState } from 'react'
import { PageHeader } from '@/components/layout/PageHeader'
import { GlassCard } from '@/components/ui/GlassCard'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { useSettingsStore } from '@/stores/settingsStore'
import { toast } from '@/components/ui/Toast'
import { LLM_PROVIDERS, LLM_MODELS } from '@/lib/constants'
import type { LLMProvider } from '@/api/types'
import {
  Cog6ToothIcon,
  CpuChipIcon,
  BeakerIcon,
  ArrowPathIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  AdjustmentsHorizontalIcon,
} from '@heroicons/react/24/outline'

export default function Settings() {
  const [showAdvanced, setShowAdvanced] = useState(false)

  const {
    defaultLLMProvider,
    defaultModel,
    defaultMaxTokens,
    defaultTemperature,
    topP,
    frequencyPenalty,
    presencePenalty,
    autoExecute,
    executionTimeout,
    setDefaultLLMProvider,
    setDefaultModel,
    setDefaultMaxTokens,
    setDefaultTemperature,
    setTopP,
    setFrequencyPenalty,
    setPresencePenalty,
    setAutoExecute,
    setExecutionTimeout,
    resetToDefaults,
  } = useSettingsStore()

  const handleProviderChange = (provider: LLMProvider) => {
    setDefaultLLMProvider(provider)
    // Set default model for the new provider
    const providerModels = LLM_MODELS[provider]
    const defaultModelForProvider = providerModels.find(m => 'default' in m && m.default)?.id || providerModels[0].id
    setDefaultModel(provider, defaultModelForProvider)
  }

  const handleReset = () => {
    if (confirm('Are you sure you want to reset all settings to defaults?')) {
      resetToDefaults()
      toast.success('Settings Reset', 'All settings have been restored to defaults')
    }
  }

  const handleSave = () => {
    toast.success('Settings Saved', 'Your preferences have been saved')
  }

  const currentModels = LLM_MODELS[defaultLLMProvider] || []
  const currentModel = defaultModel[defaultLLMProvider]

  return (
    <div className="animate-in">
      <PageHeader
        title="Settings"
        subtitle="Configure system preferences"
        actions={
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              onClick={handleReset}
              leftIcon={<ArrowPathIcon className="h-5 w-5" />}
              aria-label="Reset all settings to default values"
            >
              Reset to Defaults
            </Button>
            <Button onClick={handleSave} aria-label="Save current settings">
              Save Changes
            </Button>
          </div>
        }
      />

      <div className="space-y-6 max-w-3xl">
        {/* LLM Configuration */}
        <GlassCard
          title="LLM Configuration"
          subtitle="Configure the default language model settings"
          icon={<CpuChipIcon className="h-5 w-5" />}
        >
          <div className="space-y-6">
            {/* Provider Selection */}
            <div>
              <label
                id="provider-label"
                className="block mb-3 text-body-sm font-medium text-text-primary"
              >
                Default Provider
              </label>
              <div
                className="grid grid-cols-3 gap-3"
                role="radiogroup"
                aria-labelledby="provider-label"
              >
                {LLM_PROVIDERS.map((provider) => (
                  <button
                    key={provider.id}
                    onClick={() => handleProviderChange(provider.id as LLMProvider)}
                    role="radio"
                    aria-checked={defaultLLMProvider === provider.id}
                    className={`p-4 rounded-lg border text-left transition-colors ${
                      defaultLLMProvider === provider.id
                        ? 'border-accent-primary bg-accent-primary/20'
                        : 'border-border hover:border-accent-primary/50'
                    }`}
                  >
                    <p className="text-body-sm font-medium text-text-primary">{provider.label}</p>
                    <p className="text-caption text-text-muted">{provider.description}</p>
                  </button>
                ))}
              </div>
            </div>

            {/* Model Selection */}
            <div>
              <label
                htmlFor="model-select"
                className="block mb-2 text-body-sm font-medium text-text-primary"
              >
                Model
              </label>
              <select
                id="model-select"
                value={currentModel}
                onChange={(e) => setDefaultModel(defaultLLMProvider, e.target.value)}
                className="w-full px-4 py-2.5 rounded-lg bg-bg-secondary border border-border text-text-primary focus:outline-none focus:ring-2 focus:ring-accent-primary/50 focus:border-accent-primary transition-colors"
                aria-describedby="model-hint"
              >
                {currentModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.label}
                  </option>
                ))}
              </select>
              <p id="model-hint" className="mt-1.5 text-caption text-text-muted">
                Select the model to use for {LLM_PROVIDERS.find(p => p.id === defaultLLMProvider)?.label}
              </p>
            </div>

            {/* Basic Parameters */}
            <div className="grid grid-cols-2 gap-6">
              <Input
                label="Max Tokens"
                type="number"
                value={defaultMaxTokens}
                onChange={(e) => setDefaultMaxTokens(parseInt(e.target.value) || 8192)}
                hint="Maximum tokens for LLM responses"
                aria-describedby="max-tokens-hint"
              />
              <Input
                label="Temperature"
                type="number"
                step="0.1"
                min="0"
                max="2"
                value={defaultTemperature}
                onChange={(e) => setDefaultTemperature(parseFloat(e.target.value) || 0.7)}
                hint="Creativity level (0-2)"
                aria-describedby="temperature-hint"
              />
            </div>

            {/* Advanced Parameters Toggle */}
            <div>
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center gap-2 text-body-sm text-accent-primary hover:text-accent-secondary transition-colors"
                aria-expanded={showAdvanced}
                aria-controls="advanced-settings"
              >
                <AdjustmentsHorizontalIcon className="h-4 w-4" />
                Advanced Parameters
                {showAdvanced ? (
                  <ChevronUpIcon className="h-4 w-4" />
                ) : (
                  <ChevronDownIcon className="h-4 w-4" />
                )}
              </button>

              {/* Advanced Parameters */}
              {showAdvanced && (
                <div
                  id="advanced-settings"
                  className="mt-4 p-4 rounded-lg bg-bg-tertiary/50 space-y-4"
                >
                  <div className="grid grid-cols-3 gap-4">
                    <Input
                      label="Top P"
                      type="number"
                      step="0.05"
                      min="0"
                      max="1"
                      value={topP}
                      onChange={(e) => setTopP(parseFloat(e.target.value) || 1.0)}
                      hint="Nucleus sampling (0-1)"
                      inputSize="sm"
                    />
                    <Input
                      label="Frequency Penalty"
                      type="number"
                      step="0.1"
                      min="0"
                      max="2"
                      value={frequencyPenalty}
                      onChange={(e) => setFrequencyPenalty(parseFloat(e.target.value) || 0)}
                      hint="Penalize repetition (0-2)"
                      inputSize="sm"
                    />
                    <Input
                      label="Presence Penalty"
                      type="number"
                      step="0.1"
                      min="0"
                      max="2"
                      value={presencePenalty}
                      onChange={(e) => setPresencePenalty(parseFloat(e.target.value) || 0)}
                      hint="Encourage novelty (0-2)"
                      inputSize="sm"
                    />
                  </div>
                  <p className="text-caption text-text-muted">
                    These parameters fine-tune the model's output behavior. Use defaults unless you have specific requirements.
                  </p>
                </div>
              )}
            </div>
          </div>
        </GlassCard>

        {/* Execution Settings */}
        <GlassCard
          title="Execution Settings"
          subtitle="Configure code execution behavior"
          icon={<BeakerIcon className="h-5 w-5" />}
        >
          <div className="space-y-6">
            <div className="flex items-center justify-between p-4 rounded-lg border border-border">
              <div>
                <p className="text-body font-medium text-text-primary">Auto-Execute Code</p>
                <p className="text-body-sm text-text-muted">Automatically run generated test code</p>
              </div>
              <button
                onClick={() => setAutoExecute(!autoExecute)}
                role="switch"
                aria-checked={autoExecute}
                aria-label="Toggle auto-execute code"
                className={`relative w-12 h-6 rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-accent-primary/50 ${
                  autoExecute ? 'bg-accent-primary' : 'bg-bg-tertiary'
                }`}
              >
                <span
                  className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
                    autoExecute ? 'translate-x-7' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>

            <Input
              label="Execution Timeout (seconds)"
              type="number"
              value={executionTimeout}
              onChange={(e) => setExecutionTimeout(parseInt(e.target.value) || 300)}
              hint="Maximum time for code execution"
            />
          </div>
        </GlassCard>

        {/* System Info */}
        <GlassCard
          title="System Information"
          subtitle="View system configuration"
          icon={<Cog6ToothIcon className="h-5 w-5" />}
        >
          <div className="space-y-3" role="list" aria-label="System information">
            <div
              className="flex items-center justify-between py-2 border-b border-border"
              role="listitem"
            >
              <span className="text-body-sm text-text-secondary">UI Version</span>
              <span className="text-body-sm font-mono text-text-primary">1.0.0</span>
            </div>
            <div
              className="flex items-center justify-between py-2 border-b border-border"
              role="listitem"
            >
              <span className="text-body-sm text-text-secondary">API Endpoint</span>
              <span className="text-body-sm font-mono text-text-primary">/api</span>
            </div>
            <div
              className="flex items-center justify-between py-2 border-b border-border"
              role="listitem"
            >
              <span className="text-body-sm text-text-secondary">WebSocket</span>
              <span className="text-body-sm font-mono text-text-primary">/ws</span>
            </div>
            <div
              className="flex items-center justify-between py-2"
              role="listitem"
            >
              <span className="text-body-sm text-text-secondary">Framework</span>
              <span className="text-body-sm font-mono text-text-primary">React + Vite + Tailwind</span>
            </div>
          </div>
        </GlassCard>
      </div>
    </div>
  )
}
