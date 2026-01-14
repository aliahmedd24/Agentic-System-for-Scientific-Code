import axios, { AxiosError, AxiosInstance } from 'axios'

// Create axios instance with default config
const apiClient: AxiosInstance = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    // Extract error message from response
    const message =
      (error.response?.data as { detail?: string })?.detail ||
      error.message ||
      'An unexpected error occurred'

    console.error('API Error:', message)

    return Promise.reject({
      message,
      status: error.response?.status,
      data: error.response?.data,
    })
  }
)

export interface ApiError {
  message: string
  status?: number
  data?: unknown
}

export default apiClient
