import { useMutation, useQueryClient } from '@tanstack/react-query'
import { cancelJob, startAnalysis, startAnalysisWithUpload } from '@/api/endpoints'
import { useToastStore } from '@/components/ui/Toast'
import type { LLMProvider } from '@/api/types'

/**
 * React Query mutations for job operations
 */
export function useJobMutations() {
  const queryClient = useQueryClient()
  const addToast = useToastStore((s) => s.addToast)

  // Cancel a running job
  const cancelMutation = useMutation({
    mutationFn: cancelJob,
    onSuccess: (_, jobId) => {
      // Invalidate queries to refresh job list
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
      queryClient.invalidateQueries({ queryKey: ['jobStatus', jobId] })
      addToast({
        type: 'success',
        title: 'Job cancelled',
        message: `Job ${jobId.slice(0, 8)} has been cancelled`,
      })
    },
    onError: (error: Error) => {
      addToast({
        type: 'error',
        title: 'Failed to cancel job',
        message: error.message,
      })
    },
  })

  // Start a new analysis (arXiv/URL)
  const startAnalysisMutation = useMutation({
    mutationFn: startAnalysis,
    onSuccess: (response) => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
      addToast({
        type: 'success',
        title: 'Analysis started',
        message: `Job ${response.job_id.slice(0, 8)} has been created`,
      })
    },
    onError: (error: Error) => {
      addToast({
        type: 'error',
        title: 'Failed to start analysis',
        message: error.message,
      })
    },
  })

  // Start analysis with file upload
  const uploadAnalysisMutation = useMutation({
    mutationFn: ({
      file,
      repoUrl,
      llmProvider,
      autoExecute,
    }: {
      file: File
      repoUrl: string
      llmProvider?: LLMProvider
      autoExecute?: boolean
    }) => startAnalysisWithUpload(file, repoUrl, llmProvider, autoExecute),
    onSuccess: (response) => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
      addToast({
        type: 'success',
        title: 'Analysis started',
        message: `Job ${response.job_id.slice(0, 8)} has been created from uploaded file`,
      })
    },
    onError: (error: Error) => {
      addToast({
        type: 'error',
        title: 'Failed to upload and start analysis',
        message: error.message,
      })
    },
  })

  return {
    // Cancel job
    cancelJob: cancelMutation.mutate,
    cancelJobAsync: cancelMutation.mutateAsync,
    isCancelling: cancelMutation.isPending,

    // Start analysis (arXiv/URL)
    startAnalysis: startAnalysisMutation.mutate,
    startAnalysisAsync: startAnalysisMutation.mutateAsync,
    isStartingAnalysis: startAnalysisMutation.isPending,

    // Upload and start analysis
    uploadAnalysis: uploadAnalysisMutation.mutate,
    uploadAnalysisAsync: uploadAnalysisMutation.mutateAsync,
    isUploading: uploadAnalysisMutation.isPending,
  }
}

export default useJobMutations
