import { PageHeader } from '@/components/layout/PageHeader'
import { GlassCard } from '@/components/ui/GlassCard'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { useSettingsStore } from '@/stores/settingsStore'
import { toast } from '@/components/ui/Toast'
import { LLM_PROVIDERS } from '@/lib/constants'
import {
  Cog6ToothIcon,
  CpuChipIcon,
  BeakerIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline'

export default function Settings() {
  const {
    defaultLLMProvider,
    defaultMaxTokens,
    defaultTemperature,
    autoExecute,
    executionTimeout,
    setDefaultLLMProvider,
    setDefaultMaxTokens,
    setDefaultTemperature,
    setAutoExecute,
    setExecutionTimeout,
    resetToDefaults,
  } = useSettingsStore()

  const handleReset = () => {
    if (confirm('Are you sure you want to reset all settings to defaults?')) {
      resetToDefaults()
      toast.success('Settings Reset', 'All settings have been restored to defaults')
    }
  }

  const handleSave = () => {
    toast.success('Settings Saved', 'Your preferences have been saved')
  }

  return (
    <div className="animate-in">
      <PageHeader
        title="Settings"
        subtitle="Configure system preferences"
        actions={
          <div className="flex items-center gap-3">
            <Button variant="ghost" onClick={handleReset} leftIcon={<ArrowPathIcon className="h-5 w-5" />}>
              Reset to Defaults
            </Button>
            <Button onClick={handleSave}>Save Changes</Button>
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
            <div>
              <label className="block mb-3 text-body-sm font-medium text-text-primary">
                Default Provider
              </label>
              <div className="grid grid-cols-3 gap-3">
                {LLM_PROVIDERS.map((provider) => (
                  <button
                    key={provider.id}
                    onClick={() => setDefaultLLMProvider(provider.id)}
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

            <div className="grid grid-cols-2 gap-6">
              <Input
                label="Max Tokens"
                type="number"
                value={defaultMaxTokens}
                onChange={(e) => setDefaultMaxTokens(parseInt(e.target.value) || 8192)}
                hint="Maximum tokens for LLM responses"
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
              />
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
                className={`relative w-12 h-6 rounded-full transition-colors ${
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
          <div className="space-y-3">
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-body-sm text-text-secondary">UI Version</span>
              <span className="text-body-sm font-mono text-text-primary">1.0.0</span>
            </div>
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-body-sm text-text-secondary">API Endpoint</span>
              <span className="text-body-sm font-mono text-text-primary">/api</span>
            </div>
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-body-sm text-text-secondary">WebSocket</span>
              <span className="text-body-sm font-mono text-text-primary">/ws</span>
            </div>
            <div className="flex items-center justify-between py-2">
              <span className="text-body-sm text-text-secondary">Framework</span>
              <span className="text-body-sm font-mono text-text-primary">React + Vite + Tailwind</span>
            </div>
          </div>
        </GlassCard>
      </div>
    </div>
  )
}
