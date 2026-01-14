import { Fragment } from 'react'
import { Listbox, Transition } from '@headlessui/react'
import { CheckIcon, ChevronUpDownIcon } from '@heroicons/react/24/outline'
import { cn } from '@/lib/cn'

export interface SelectOption {
  id: string
  label: string
  description?: string
  disabled?: boolean
}

interface SelectProps {
  label?: string
  options: SelectOption[]
  value: string
  onChange: (value: string) => void
  placeholder?: string
  error?: string
  hint?: string
  disabled?: boolean
  className?: string
}

export function Select({
  label,
  options,
  value,
  onChange,
  placeholder = 'Select an option',
  error,
  hint,
  disabled = false,
  className,
}: SelectProps) {
  const selectedOption = options.find((opt) => opt.id === value)

  return (
    <div className={cn('w-full', className)}>
      {label && (
        <label className="block mb-2 text-body-sm font-medium text-text-primary">
          {label}
        </label>
      )}
      <Listbox value={value} onChange={onChange} disabled={disabled}>
        <div className="relative">
          <Listbox.Button
            className={cn(
              'relative w-full cursor-pointer rounded-lg py-3 pl-4 pr-10 text-left',
              'bg-bg-tertiary border transition-colors duration-200',
              'focus:outline-none focus:ring-2 focus:ring-accent-primary focus:border-accent-primary',
              error
                ? 'border-status-error focus:ring-status-error'
                : 'border-border hover:border-accent-primary/50',
              disabled && 'opacity-50 cursor-not-allowed'
            )}
          >
            <span className={cn('block truncate', !selectedOption && 'text-text-muted')}>
              {selectedOption?.label || placeholder}
            </span>
            <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3">
              <ChevronUpDownIcon className="h-5 w-5 text-text-muted" aria-hidden="true" />
            </span>
          </Listbox.Button>

          <Transition
            as={Fragment}
            leave="transition ease-in duration-100"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <Listbox.Options
              className={cn(
                'absolute z-50 mt-1 max-h-60 w-full overflow-auto rounded-lg py-1',
                'bg-bg-secondary border border-border shadow-xl',
                'focus:outline-none'
              )}
            >
              {options.map((option) => (
                <Listbox.Option
                  key={option.id}
                  value={option.id}
                  disabled={option.disabled}
                  className={({ active, selected }) =>
                    cn(
                      'relative cursor-pointer select-none py-3 pl-10 pr-4',
                      'transition-colors duration-100',
                      active && 'bg-accent-primary/20',
                      selected && 'bg-accent-primary/10',
                      option.disabled && 'opacity-50 cursor-not-allowed'
                    )
                  }
                >
                  {({ selected }) => (
                    <>
                      <div>
                        <span
                          className={cn(
                            'block truncate text-body-sm',
                            selected ? 'font-medium text-accent-secondary' : 'text-text-primary'
                          )}
                        >
                          {option.label}
                        </span>
                        {option.description && (
                          <span className="block truncate text-caption text-text-muted">
                            {option.description}
                          </span>
                        )}
                      </div>
                      {selected && (
                        <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-accent-primary">
                          <CheckIcon className="h-5 w-5" aria-hidden="true" />
                        </span>
                      )}
                    </>
                  )}
                </Listbox.Option>
              ))}
            </Listbox.Options>
          </Transition>
        </div>
      </Listbox>

      {(error || hint) && (
        <p className={cn('mt-1.5 text-body-sm', error ? 'text-status-error' : 'text-text-muted')}>
          {error || hint}
        </p>
      )}
    </div>
  )
}

export default Select
