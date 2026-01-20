import { useState, useCallback, useEffect, RefObject } from 'react'

interface UseFullscreenReturn {
  isFullscreen: boolean
  enterFullscreen: () => Promise<void>
  exitFullscreen: () => Promise<void>
  toggleFullscreen: () => Promise<void>
}

export function useFullscreen(elementRef: RefObject<HTMLElement>): UseFullscreenReturn {
  const [isFullscreen, setIsFullscreen] = useState(false)

  // Check if fullscreen is currently active
  const checkFullscreen = useCallback(() => {
    const isFs = Boolean(
      document.fullscreenElement ||
      (document as unknown as { webkitFullscreenElement?: Element }).webkitFullscreenElement ||
      (document as unknown as { mozFullScreenElement?: Element }).mozFullScreenElement ||
      (document as unknown as { msFullscreenElement?: Element }).msFullscreenElement
    )
    setIsFullscreen(isFs)
  }, [])

  // Listen for fullscreen changes
  useEffect(() => {
    document.addEventListener('fullscreenchange', checkFullscreen)
    document.addEventListener('webkitfullscreenchange', checkFullscreen)
    document.addEventListener('mozfullscreenchange', checkFullscreen)
    document.addEventListener('MSFullscreenChange', checkFullscreen)

    return () => {
      document.removeEventListener('fullscreenchange', checkFullscreen)
      document.removeEventListener('webkitfullscreenchange', checkFullscreen)
      document.removeEventListener('mozfullscreenchange', checkFullscreen)
      document.removeEventListener('MSFullscreenChange', checkFullscreen)
    }
  }, [checkFullscreen])

  const enterFullscreen = useCallback(async () => {
    const element = elementRef.current
    if (!element) return

    try {
      if (element.requestFullscreen) {
        await element.requestFullscreen()
      } else if ((element as unknown as { webkitRequestFullscreen?: () => Promise<void> }).webkitRequestFullscreen) {
        await (element as unknown as { webkitRequestFullscreen: () => Promise<void> }).webkitRequestFullscreen()
      } else if ((element as unknown as { mozRequestFullScreen?: () => Promise<void> }).mozRequestFullScreen) {
        await (element as unknown as { mozRequestFullScreen: () => Promise<void> }).mozRequestFullScreen()
      } else if ((element as unknown as { msRequestFullscreen?: () => Promise<void> }).msRequestFullscreen) {
        await (element as unknown as { msRequestFullscreen: () => Promise<void> }).msRequestFullscreen()
      }
    } catch (err) {
      console.error('Failed to enter fullscreen:', err)
    }
  }, [elementRef])

  const exitFullscreen = useCallback(async () => {
    try {
      if (document.exitFullscreen) {
        await document.exitFullscreen()
      } else if ((document as unknown as { webkitExitFullscreen?: () => Promise<void> }).webkitExitFullscreen) {
        await (document as unknown as { webkitExitFullscreen: () => Promise<void> }).webkitExitFullscreen()
      } else if ((document as unknown as { mozCancelFullScreen?: () => Promise<void> }).mozCancelFullScreen) {
        await (document as unknown as { mozCancelFullScreen: () => Promise<void> }).mozCancelFullScreen()
      } else if ((document as unknown as { msExitFullscreen?: () => Promise<void> }).msExitFullscreen) {
        await (document as unknown as { msExitFullscreen: () => Promise<void> }).msExitFullscreen()
      }
    } catch (err) {
      console.error('Failed to exit fullscreen:', err)
    }
  }, [])

  const toggleFullscreen = useCallback(async () => {
    if (isFullscreen) {
      await exitFullscreen()
    } else {
      await enterFullscreen()
    }
  }, [isFullscreen, enterFullscreen, exitFullscreen])

  return {
    isFullscreen,
    enterFullscreen,
    exitFullscreen,
    toggleFullscreen,
  }
}

export default useFullscreen
