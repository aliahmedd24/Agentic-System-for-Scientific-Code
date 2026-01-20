import { Fragment } from 'react'
import { Transition } from '@headlessui/react'
import { cn } from '@/lib/cn'
import {
  CheckCircleIcon,
  ExclamationCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline'
import { create } from 'zustand'

// Toast types
export type ToastType = 'success' | 'error' | 'warning' | 'info'

export interface Toast {
  id: string
  type: ToastType
  title: string
  message?: string
  duration?: number
}

// Toast store
interface ToastStore {
  toasts: Toast[]
  addToast: (toast: Omit<Toast, 'id'>) => void
  removeToast: (id: string) => void
}

export const useToastStore = create<ToastStore>((set) => ({
  toasts: [],
  addToast: (toast) => {
    const id = Math.random().toString(36).substring(2, 9)
    set((state) => ({
      toasts: [...state.toasts, { ...toast, id }],
    }))

    // Auto-remove after duration
    if (toast.duration !== 0) {
      setTimeout(() => {
        set((state) => ({
          toasts: state.toasts.filter((t) => t.id !== id),
        }))
      }, toast.duration || 5000)
    }
  },
  removeToast: (id) =>
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id),
    })),
}))

// Helper function to show toasts
export const toast = {
  success: (title: string, message?: string, duration?: number) =>
    useToastStore.getState().addToast({ type: 'success', title, message, duration }),
  error: (title: string, message?: string, duration?: number) =>
    useToastStore.getState().addToast({ type: 'error', title, message, duration }),
  warning: (title: string, message?: string, duration?: number) =>
    useToastStore.getState().addToast({ type: 'warning', title, message, duration }),
  info: (title: string, message?: string, duration?: number) =>
    useToastStore.getState().addToast({ type: 'info', title, message, duration }),
}

// Toast item config
const toastConfig: Record<
  ToastType,
  { icon: React.ComponentType<{ className?: string }>; iconColor: string; borderColor: string }
> = {
  success: {
    icon: CheckCircleIcon,
    iconColor: 'text-status-success',
    borderColor: 'border-l-status-success',
  },
  error: {
    icon: ExclamationCircleIcon,
    iconColor: 'text-status-error',
    borderColor: 'border-l-status-error',
  },
  warning: {
    icon: ExclamationTriangleIcon,
    iconColor: 'text-status-warning',
    borderColor: 'border-l-status-warning',
  },
  info: {
    icon: InformationCircleIcon,
    iconColor: 'text-status-info',
    borderColor: 'border-l-status-info',
  },
}

interface ToastItemProps {
  toast: Toast
  onDismiss: () => void
}

function ToastItem({ toast: t, onDismiss }: ToastItemProps) {
  const config = toastConfig[t.type]
  const Icon = config.icon

  return (
    <Transition
      appear
      show={true}
      as={Fragment}
      enter="transform transition duration-300 ease-out"
      enterFrom="translate-x-full opacity-0"
      enterTo="translate-x-0 opacity-100"
      leave="transform transition duration-200 ease-in"
      leaveFrom="translate-x-0 opacity-100"
      leaveTo="translate-x-full opacity-0"
    >
      <div
        className={cn(
          'flex items-start gap-3 w-80 p-4',
          'bg-bg-secondary/95 backdrop-blur-xl rounded-lg shadow-xl',
          'border border-border border-l-4',
          config.borderColor
        )}
      >
        <Icon className={cn('h-5 w-5 flex-shrink-0 mt-0.5', config.iconColor)} />
        <div className="flex-1 min-w-0">
          <p className="text-body-sm font-medium text-text-primary">{t.title}</p>
          {t.message && (
            <p className="mt-1 text-body-sm text-text-secondary">{t.message}</p>
          )}
        </div>
        <button
          onClick={onDismiss}
          className="flex-shrink-0 p-1 rounded hover:bg-bg-tertiary transition-colors"
        >
          <XMarkIcon className="h-4 w-4 text-text-muted" />
        </button>
      </div>
    </Transition>
  )
}

export function ToastContainer() {
  const { toasts, removeToast } = useToastStore()

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
      {toasts.map((t) => (
        <ToastItem key={t.id} toast={t} onDismiss={() => removeToast(t.id)} />
      ))}
    </div>
  )
}

export default ToastContainer
