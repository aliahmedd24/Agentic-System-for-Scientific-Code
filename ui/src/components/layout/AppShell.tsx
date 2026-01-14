import { cn } from '@/lib/cn'
import { Sidebar } from './Sidebar'

interface AppShellProps {
  children: React.ReactNode
  className?: string
}

export function AppShell({ children, className }: AppShellProps) {
  return (
    <div className="min-h-screen bg-bg-primary">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <main
        className={cn(
          'ml-64 min-h-screen',
          'px-8 py-6',
          className
        )}
      >
        <div className="max-w-7xl mx-auto relative z-10">
          {children}
        </div>
      </main>
    </div>
  )
}

export default AppShell
