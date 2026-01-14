import { Disclosure } from '@headlessui/react'
import { ChevronDownIcon, ExclamationCircleIcon } from '@heroicons/react/24/outline'
import { cn } from '@/lib/cn'
import { GlassCard } from '@/components/ui/GlassCard'
import type { UnmappedItem } from '@/api/types'

interface UnmappedSectionProps {
  concepts: UnmappedItem[]
  code: UnmappedItem[]
  className?: string
}

interface UnmappedItemRowProps {
  item: UnmappedItem
}

function UnmappedItemRow({ item }: UnmappedItemRowProps) {
  return (
    <div className="py-3 px-4 hover:bg-bg-tertiary/50 transition-colors duration-200">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <h5 className="text-body-sm font-medium text-text-primary">
            {item.name}
          </h5>
          {item.description && (
            <p className="text-caption text-text-muted mt-0.5 line-clamp-2">
              {item.description}
            </p>
          )}
        </div>
        <span
          className={cn(
            'shrink-0 px-2 py-0.5 rounded-full text-caption',
            'bg-status-warning/20 text-status-warning'
          )}
        >
          {item.reason}
        </span>
      </div>
    </div>
  )
}

interface DisclosureSectionProps {
  title: string
  items: UnmappedItem[]
  icon: React.ReactNode
  emptyText: string
}

function DisclosureSection({ title, items, icon, emptyText }: DisclosureSectionProps) {
  return (
    <Disclosure defaultOpen={items.length > 0 && items.length <= 5}>
      {({ open }) => (
        <GlassCard noPadding>
          <Disclosure.Button
            className={cn(
              'w-full flex items-center justify-between gap-3 p-4',
              'hover:bg-bg-tertiary/50 transition-colors duration-200'
            )}
          >
            <div className="flex items-center gap-3">
              {icon}
              <span className="text-body font-medium text-text-primary">
                {title}
              </span>
              <span
                className={cn(
                  'px-2 py-0.5 rounded-full text-caption',
                  items.length > 0
                    ? 'bg-status-warning/20 text-status-warning'
                    : 'bg-status-success/20 text-status-success'
                )}
              >
                {items.length}
              </span>
            </div>
            <ChevronDownIcon
              className={cn(
                'h-5 w-5 text-text-muted transition-transform duration-200',
                open && 'rotate-180'
              )}
            />
          </Disclosure.Button>

          <Disclosure.Panel>
            <div className="border-t border-border">
              {items.length > 0 ? (
                <div className="divide-y divide-border">
                  {items.map((item, idx) => (
                    <UnmappedItemRow key={idx} item={item} />
                  ))}
                </div>
              ) : (
                <div className="py-8 text-center">
                  <p className="text-body-sm text-text-muted">{emptyText}</p>
                </div>
              )}
            </div>
          </Disclosure.Panel>
        </GlassCard>
      )}
    </Disclosure>
  )
}

export function UnmappedSection({ concepts, code, className }: UnmappedSectionProps) {
  const totalUnmapped = concepts.length + code.length

  return (
    <div className={cn('space-y-4', className)}>
      {/* Summary */}
      {totalUnmapped > 0 && (
        <div className="flex items-center gap-2 p-3 rounded-lg bg-status-warning/10 border border-status-warning/20">
          <ExclamationCircleIcon className="h-5 w-5 text-status-warning shrink-0" />
          <p className="text-body-sm text-text-secondary">
            {totalUnmapped} item{totalUnmapped !== 1 ? 's' : ''} could not be mapped.
            This may indicate missing implementations or incomplete documentation.
          </p>
        </div>
      )}

      {/* Unmapped concepts */}
      <DisclosureSection
        title="Unmapped Concepts"
        items={concepts}
        icon={
          <span className="w-8 h-8 rounded-lg bg-graph-concept/20 flex items-center justify-center">
            <span className="text-body-sm font-medium text-graph-concept">C</span>
          </span>
        }
        emptyText="All concepts were successfully mapped!"
      />

      {/* Unmapped code */}
      <DisclosureSection
        title="Unmapped Code Elements"
        items={code}
        icon={
          <span className="w-8 h-8 rounded-lg bg-graph-function/20 flex items-center justify-center">
            <span className="text-body-sm font-medium text-graph-function">{"</>"}</span>
          </span>
        }
        emptyText="All code elements were successfully mapped!"
      />
    </div>
  )
}

export default UnmappedSection
