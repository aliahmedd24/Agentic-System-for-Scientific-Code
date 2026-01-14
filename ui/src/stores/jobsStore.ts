import { create } from 'zustand'
import type { Job, JobStatusResponse, JobResult, AnalysisRequest, JobResponse } from '@/api/types'
import * as api from '@/api/endpoints'

interface JobsState {
  // Data
  jobs: Job[]
  currentJob: JobStatusResponse | null
  currentResult: JobResult | null

  // Loading states
  isLoadingJobs: boolean
  isLoadingJob: boolean
  isLoadingResult: boolean
  isSubmitting: boolean

  // Error states
  error: string | null

  // Actions
  fetchJobs: (limit?: number) => Promise<void>
  fetchJob: (jobId: string) => Promise<void>
  fetchResult: (jobId: string) => Promise<void>
  startAnalysis: (request: AnalysisRequest) => Promise<JobResponse>
  startAnalysisWithUpload: (file: File, repoUrl: string, llmProvider?: string, autoExecute?: boolean) => Promise<JobResponse>
  cancelJob: (jobId: string) => Promise<void>
  clearError: () => void
  clearCurrentJob: () => void
}

export const useJobsStore = create<JobsState>((set, get) => ({
  // Initial state
  jobs: [],
  currentJob: null,
  currentResult: null,
  isLoadingJobs: false,
  isLoadingJob: false,
  isLoadingResult: false,
  isSubmitting: false,
  error: null,

  // Fetch all jobs
  fetchJobs: async (limit = 20) => {
    set({ isLoadingJobs: true, error: null })
    try {
      const jobs = await api.getJobs(limit)
      set({ jobs, isLoadingJobs: false })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch jobs'
      set({ error: message, isLoadingJobs: false })
    }
  },

  // Fetch single job status
  fetchJob: async (jobId: string) => {
    set({ isLoadingJob: true, error: null })
    try {
      const job = await api.getJobStatus(jobId)
      set({ currentJob: job, isLoadingJob: false })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch job'
      set({ error: message, isLoadingJob: false })
    }
  },

  // Fetch job result
  fetchResult: async (jobId: string) => {
    set({ isLoadingResult: true, error: null })
    try {
      const result = await api.getJobResult(jobId)
      set({ currentResult: result, isLoadingResult: false })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch result'
      set({ error: message, isLoadingResult: false })
    }
  },

  // Start new analysis
  startAnalysis: async (request: AnalysisRequest) => {
    set({ isSubmitting: true, error: null })
    try {
      const response = await api.startAnalysis(request)
      set({ isSubmitting: false })
      return response
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start analysis'
      set({ error: message, isSubmitting: false })
      throw err
    }
  },

  // Start analysis with file upload
  startAnalysisWithUpload: async (file: File, repoUrl: string, llmProvider = 'gemini', autoExecute = true) => {
    set({ isSubmitting: true, error: null })
    try {
      const response = await api.startAnalysisWithUpload(file, repoUrl, llmProvider, autoExecute)
      set({ isSubmitting: false })
      return response
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start analysis'
      set({ error: message, isSubmitting: false })
      throw err
    }
  },

  // Cancel job
  cancelJob: async (jobId: string) => {
    try {
      await api.cancelJob(jobId)
      // Refresh the job list
      get().fetchJobs()
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to cancel job'
      set({ error: message })
    }
  },

  // Clear error
  clearError: () => set({ error: null }),

  // Clear current job
  clearCurrentJob: () => set({ currentJob: null, currentResult: null }),
}))

export default useJobsStore
