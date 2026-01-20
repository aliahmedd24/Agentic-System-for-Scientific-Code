import { useState, useCallback, useRef, useEffect } from 'react'
import * as api from '@/api/endpoints'
import type { UseGraphSearchReturn, SearchResult } from '../types'

const DEBOUNCE_MS = 300
const MIN_QUERY_LENGTH = 2

export function useGraphSearch(jobId: string | undefined): UseGraphSearchReturn {
  const [results, setResults] = useState<SearchResult>({
    nodes: [],
    total: 0,
    highlightedIds: new Set(),
  })
  const [isSearching, setIsSearching] = useState(false)

  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
    }
  }, [])

  const search = useCallback(async (query: string, nodeType?: string) => {
    // Clear previous debounce timer
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current)
    }

    // Abort previous request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    // Clear results for empty query
    if (!query.trim() || query.length < MIN_QUERY_LENGTH) {
      setResults({
        nodes: [],
        total: 0,
        highlightedIds: new Set(),
      })
      setIsSearching(false)
      return
    }

    if (!jobId) return

    // Debounce the search
    debounceTimerRef.current = setTimeout(async () => {
      setIsSearching(true)
      abortControllerRef.current = new AbortController()

      try {
        const response = await api.searchKnowledgeGraph(jobId, query, nodeType, 50)

        // Create set of highlighted IDs
        const highlightedIds = new Set(response.nodes.map(n => n.id))

        setResults({
          nodes: response.nodes,
          total: response.total,
          highlightedIds,
        })
      } catch (err) {
        // Ignore abort errors
        if (err instanceof Error && err.name === 'AbortError') {
          return
        }
        console.error('Search failed:', err)
        setResults({
          nodes: [],
          total: 0,
          highlightedIds: new Set(),
        })
      } finally {
        setIsSearching(false)
      }
    }, DEBOUNCE_MS)
  }, [jobId])

  const clearSearch = useCallback(() => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current)
    }
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    setResults({
      nodes: [],
      total: 0,
      highlightedIds: new Set(),
    })
    setIsSearching(false)
  }, [])

  return {
    search,
    results,
    isSearching,
    clearSearch,
  }
}

export default useGraphSearch
