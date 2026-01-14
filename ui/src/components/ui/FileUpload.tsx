import { useCallback, useState } from 'react'
import { cn } from '@/lib/cn'
import {
  CloudArrowUpIcon,
  DocumentIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline'

interface FileUploadProps {
  accept?: string
  maxSize?: number // in bytes
  onFile: (file: File | null) => void
  file?: File | null
  label?: string
  hint?: string
  error?: string
  disabled?: boolean
  className?: string
}

export function FileUpload({
  accept = 'application/pdf',
  maxSize = 50 * 1024 * 1024, // 50MB default
  onFile,
  file,
  label,
  hint,
  error,
  disabled = false,
  className,
}: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)

  const displayError = error || localError

  const validateFile = useCallback(
    (file: File): string | null => {
      // Check file type
      if (accept) {
        const acceptedTypes = accept.split(',').map((t) => t.trim())
        const fileType = file.type
        const fileExtension = `.${file.name.split('.').pop()?.toLowerCase()}`

        const isValidType = acceptedTypes.some(
          (type) =>
            type === fileType ||
            type === fileExtension ||
            (type.endsWith('/*') && fileType.startsWith(type.slice(0, -1)))
        )

        if (!isValidType) {
          return `Invalid file type. Accepted: ${accept}`
        }
      }

      // Check file size
      if (maxSize && file.size > maxSize) {
        const maxSizeMB = (maxSize / (1024 * 1024)).toFixed(1)
        return `File too large. Maximum size: ${maxSizeMB}MB`
      }

      return null
    },
    [accept, maxSize]
  )

  const handleFile = useCallback(
    (file: File) => {
      const error = validateFile(file)
      if (error) {
        setLocalError(error)
        onFile(null)
      } else {
        setLocalError(null)
        onFile(file)
      }
    },
    [validateFile, onFile]
  )

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      setIsDragging(false)

      if (disabled) return

      const droppedFile = e.dataTransfer.files[0]
      if (droppedFile) {
        handleFile(droppedFile)
      }
    },
    [disabled, handleFile]
  )

  const handleDragOver = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      if (!disabled) {
        setIsDragging(true)
      }
    },
    [disabled]
  )

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selectedFile = e.target.files?.[0]
      if (selectedFile) {
        handleFile(selectedFile)
      }
      // Reset input value to allow re-selecting the same file
      e.target.value = ''
    },
    [handleFile]
  )

  const handleRemove = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation()
      setLocalError(null)
      onFile(null)
    },
    [onFile]
  )

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  return (
    <div className={cn('w-full', className)}>
      {label && (
        <label className="block mb-2 text-body-sm font-medium text-text-primary">
          {label}
        </label>
      )}

      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={cn(
          'relative border-2 border-dashed rounded-xl transition-all duration-200',
          'flex flex-col items-center justify-center',
          file ? 'p-4' : 'p-8',
          isDragging && 'border-accent-primary bg-accent-primary/10 scale-[1.02]',
          displayError && 'border-status-error',
          !isDragging && !displayError && 'border-border hover:border-accent-primary/50',
          disabled && 'opacity-50 cursor-not-allowed',
          !disabled && 'cursor-pointer'
        )}
      >
        <input
          type="file"
          accept={accept}
          onChange={handleChange}
          disabled={disabled}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
        />

        {file ? (
          <div className="flex items-center gap-4 w-full">
            <div className="flex items-center justify-center w-12 h-12 rounded-lg bg-accent-primary/20">
              <DocumentIcon className="h-6 w-6 text-accent-primary" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-body-sm font-medium text-text-primary truncate">
                {file.name}
              </p>
              <p className="text-caption text-text-muted">
                {formatFileSize(file.size)}
              </p>
            </div>
            <button
              type="button"
              onClick={handleRemove}
              disabled={disabled}
              className={cn(
                'p-2 rounded-lg transition-colors',
                'hover:bg-status-error/20 text-text-muted hover:text-status-error',
                'focus:outline-none focus:ring-2 focus:ring-status-error',
                disabled && 'pointer-events-none'
              )}
            >
              <XMarkIcon className="h-5 w-5" />
            </button>
          </div>
        ) : (
          <>
            <CloudArrowUpIcon
              className={cn(
                'h-12 w-12 mb-3 transition-colors',
                isDragging ? 'text-accent-primary' : 'text-text-muted'
              )}
            />
            <p className="text-body font-medium text-text-primary mb-1">
              {isDragging ? 'Drop file here' : 'Drag and drop or click to browse'}
            </p>
            <p className="text-body-sm text-text-muted">
              {accept === 'application/pdf' ? 'PDF files only' : accept}
              {maxSize && ` • Max ${(maxSize / (1024 * 1024)).toFixed(0)}MB`}
            </p>
          </>
        )}
      </div>

      {(displayError || hint) && (
        <p
          className={cn(
            'mt-1.5 text-body-sm',
            displayError ? 'text-status-error' : 'text-text-muted'
          )}
        >
          {displayError || hint}
        </p>
      )}
    </div>
  )
}

export default FileUpload
