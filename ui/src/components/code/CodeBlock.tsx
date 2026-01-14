import { useState } from 'react'
import { Highlight, themes } from 'prism-react-renderer'
import { ClipboardDocumentIcon, CheckIcon } from '@heroicons/react/24/outline'
import { cn } from '@/lib/cn'
import { useToastStore } from '@/components/ui/Toast'

type Language = 'python' | 'javascript' | 'typescript' | 'shell' | 'bash' | 'json' | 'jsx' | 'tsx'

interface CodeBlockProps {
  code: string
  language?: Language
  showLineNumbers?: boolean
  maxHeight?: string
  copyButton?: boolean
  title?: string
  className?: string
}

const languageMap: Record<Language, string> = {
  python: 'python',
  javascript: 'javascript',
  typescript: 'typescript',
  shell: 'bash',
  bash: 'bash',
  json: 'json',
  jsx: 'jsx',
  tsx: 'tsx',
}

const customTheme = {
  ...themes.nightOwl,
  plain: {
    ...themes.nightOwl.plain,
    backgroundColor: '#12121a',
  },
}

export function CodeBlock({
  code,
  language = 'python',
  showLineNumbers = true,
  maxHeight = '400px',
  copyButton = true,
  title,
  className,
}: CodeBlockProps) {
  const [copied, setCopied] = useState(false)
  const addToast = useToastStore((s) => s.addToast)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      addToast({
        type: 'success',
        title: 'Copied to clipboard',
      })
      setTimeout(() => setCopied(false), 2000)
    } catch {
      addToast({
        type: 'error',
        title: 'Failed to copy',
        message: 'Could not copy code to clipboard',
      })
    }
  }

  const trimmedCode = code.trim()
  const prismLanguage = languageMap[language] || 'python'

  return (
    <div
      className={cn(
        'relative rounded-lg overflow-hidden',
        'bg-bg-secondary border border-border',
        className
      )}
    >
      {(title || copyButton) && (
        <div
          className={cn(
            'flex items-center justify-between px-4 py-2',
            'border-b border-border bg-bg-tertiary'
          )}
        >
          <div className="flex items-center gap-2">
            {title && (
              <span className="text-body-sm font-medium text-text-secondary">
                {title}
              </span>
            )}
            <span className="text-caption text-text-muted uppercase">
              {language}
            </span>
          </div>
          {copyButton && (
            <button
              onClick={handleCopy}
              className={cn(
                'p-1.5 rounded-md transition-colors duration-200',
                'text-text-muted hover:text-text-primary hover:bg-bg-secondary',
                'focus:outline-none focus:ring-2 focus:ring-accent-primary'
              )}
              aria-label="Copy code"
            >
              {copied ? (
                <CheckIcon className="h-4 w-4 text-status-success" />
              ) : (
                <ClipboardDocumentIcon className="h-4 w-4" />
              )}
            </button>
          )}
        </div>
      )}

      <div
        className="overflow-auto"
        style={{ maxHeight }}
      >
        <Highlight
          theme={customTheme}
          code={trimmedCode}
          language={prismLanguage}
        >
          {({ className: hlClassName, style, tokens, getLineProps, getTokenProps }) => (
            <pre
              className={cn(hlClassName, 'p-4 text-body-sm font-mono m-0')}
              style={{ ...style, backgroundColor: 'transparent' }}
            >
              {tokens.map((line, i) => {
                const lineProps = getLineProps({ line, key: i })
                return (
                  <div
                    key={i}
                    {...lineProps}
                    className={cn(lineProps.className, 'table-row')}
                  >
                    {showLineNumbers && (
                      <span className="table-cell pr-4 text-text-muted/50 select-none text-right w-8">
                        {i + 1}
                      </span>
                    )}
                    <span className="table-cell">
                      {line.map((token, key) => (
                        <span key={key} {...getTokenProps({ token, key })} />
                      ))}
                    </span>
                  </div>
                )
              })}
            </pre>
          )}
        </Highlight>
      </div>
    </div>
  )
}

export default CodeBlock
