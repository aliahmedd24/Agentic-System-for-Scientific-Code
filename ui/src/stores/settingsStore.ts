import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { LLMProvider } from '@/api/types'

interface SettingsState {
  // LLM settings
  defaultLLMProvider: LLMProvider
  defaultMaxTokens: number
  defaultTemperature: number

  // Execution settings
  autoExecute: boolean
  executionTimeout: number

  // UI settings
  sidebarCollapsed: boolean

  // Actions
  setDefaultLLMProvider: (provider: LLMProvider) => void
  setDefaultMaxTokens: (tokens: number) => void
  setDefaultTemperature: (temp: number) => void
  setAutoExecute: (auto: boolean) => void
  setExecutionTimeout: (timeout: number) => void
  setSidebarCollapsed: (collapsed: boolean) => void
  resetToDefaults: () => void
}

const defaultSettings = {
  defaultLLMProvider: 'gemini' as LLMProvider,
  defaultMaxTokens: 8192,
  defaultTemperature: 0.7,
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
      setDefaultMaxTokens: (tokens) => set({ defaultMaxTokens: tokens }),
      setDefaultTemperature: (temp) => set({ defaultTemperature: temp }),
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
