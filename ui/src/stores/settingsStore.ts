import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { LLMProvider } from '@/api/types'

interface SettingsState {
  // LLM settings
  defaultLLMProvider: LLMProvider
  defaultModel: Record<LLMProvider, string>
  defaultMaxTokens: number
  defaultTemperature: number

  // Advanced LLM settings
  topP: number
  frequencyPenalty: number
  presencePenalty: number

  // Execution settings
  autoExecute: boolean
  executionTimeout: number

  // UI settings
  sidebarCollapsed: boolean

  // Actions
  setDefaultLLMProvider: (provider: LLMProvider) => void
  setDefaultModel: (provider: LLMProvider, model: string) => void
  setDefaultMaxTokens: (tokens: number) => void
  setDefaultTemperature: (temp: number) => void
  setTopP: (topP: number) => void
  setFrequencyPenalty: (penalty: number) => void
  setPresencePenalty: (penalty: number) => void
  setAutoExecute: (auto: boolean) => void
  setExecutionTimeout: (timeout: number) => void
  setSidebarCollapsed: (collapsed: boolean) => void
  resetToDefaults: () => void
}

const defaultSettings = {
  defaultLLMProvider: 'gemini' as LLMProvider,
  defaultModel: {
    gemini: 'gemini-2.0-flash',
    anthropic: 'claude-sonnet-4-20250514',
    openai: 'gpt-4o',
  } as Record<LLMProvider, string>,
  defaultMaxTokens: 8192,
  defaultTemperature: 0.7,
  topP: 1.0,
  frequencyPenalty: 0,
  presencePenalty: 0,
  autoExecute: true,
  executionTimeout: 300,
  sidebarCollapsed: false,
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      // Initial state
      ...defaultSettings,

      // Actions
      setDefaultLLMProvider: (provider) => set({ defaultLLMProvider: provider }),
      setDefaultModel: (provider, model) =>
        set((state) => ({
          defaultModel: { ...state.defaultModel, [provider]: model },
        })),
      setDefaultMaxTokens: (tokens) => set({ defaultMaxTokens: tokens }),
      setDefaultTemperature: (temp) => set({ defaultTemperature: temp }),
      setTopP: (topP) => set({ topP }),
      setFrequencyPenalty: (penalty) => set({ frequencyPenalty: penalty }),
      setPresencePenalty: (penalty) => set({ presencePenalty: penalty }),
      setAutoExecute: (auto) => set({ autoExecute: auto }),
      setExecutionTimeout: (timeout) => set({ executionTimeout: timeout }),
      setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
      resetToDefaults: () => set(defaultSettings),
    }),
    {
      name: 'scientific-agent-settings',
    }
  )
)

export default useSettingsStore
