/**
 * Small presentational primitives shared by App, ModalityToggle and ResultPanel.
 *
 * These exist so the same surface, pill and skeleton markup is not repeated
 * across components — and so colours only ever come from theme tokens.
 */
import type { ReactNode } from 'react'

type Tone = 'neutral' | 'accent' | 'success' | 'warning' | 'danger' | 'info'

const TONE_SOFT: Record<Tone, string> = {
  neutral: 'bg-white/[0.06] text-content-secondary',
  accent: 'bg-accent-soft text-accent-hover',
  success: 'bg-success-soft text-success-text',
  warning: 'bg-warning-soft text-warning-text',
  danger: 'bg-danger-soft text-danger-text',
  info: 'bg-info-soft text-info-text',
}

const TONE_BORDER: Record<Tone, string> = {
  neutral: 'border-line',
  accent: 'border-accent/40',
  success: 'border-success/40',
  warning: 'border-warning/40',
  danger: 'border-danger/40',
  info: 'border-info/40',
}

export function Badge({
  tone = 'neutral',
  children,
  className = '',
}: {
  tone?: Tone
  children: ReactNode
  className?: string
}) {
  return (
    <span
      className={`pill ${TONE_BORDER[tone]} ${TONE_SOFT[tone]} ${className}`}
    >
      {children}
    </span>
  )
}

export function Panel({
  title,
  icon,
  action,
  children,
  className = '',
}: {
  title: string
  icon?: ReactNode
  action?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section className={`card overflow-hidden ${className}`}>
      <header className="flex items-center justify-between gap-3 border-b border-line px-5 py-3">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-content-primary">
          {icon ? <span className="text-content-muted">{icon}</span> : null}
          {title}
        </h2>
        {action}
      </header>
      <div className="p-5">{children}</div>
    </section>
  )
}

/** Shimmering placeholder used while a request is in flight. */
export function Skeleton({ className = '' }: { className?: string }) {
  return (
    <div
      aria-hidden="true"
      className={`animate-pulse rounded bg-white/[0.06] ${className}`}
    />
  )
}

/** Loading placeholder mirroring the result layout, so nothing jumps. */
export function ResultSkeleton() {
  return (
    <div className="space-y-4" role="status" aria-live="polite" aria-busy="true">
      <span className="sr-only">Analysing the URL</span>
      <div className="card space-y-4 p-5">
        <div className="flex items-end justify-between gap-4">
          <div className="space-y-2">
            <Skeleton className="h-2.5 w-14" />
            <Skeleton className="h-7 w-32" />
          </div>
          <div className="space-y-2 text-right">
            <Skeleton className="h-2.5 w-28" />
            <Skeleton className="ml-auto h-7 w-24" />
          </div>
        </div>
        <Skeleton className="h-1.5 w-full rounded-full" />
        <Skeleton className="h-3 w-48" />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {[0, 1].map((i) => (
          <div key={i} className="card space-y-3 p-5">
            <Skeleton className="h-3.5 w-32" />
            {[0, 1, 2].map((r) => (
              <div key={r} className="flex items-center justify-between gap-3">
                <Skeleton className="h-3 w-28" />
                <Skeleton className="h-3 w-16" />
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

/** Centred placeholder for a section with nothing to show. */
export function EmptyState({
  icon,
  title,
  children,
}: {
  icon?: ReactNode
  title: string
  children?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-4 py-8 text-center">
      {icon ? (
        <span className="mb-1 flex h-9 w-9 items-center justify-center rounded-full bg-white/[0.05] text-content-muted">
          {icon}
        </span>
      ) : null}
      <p className="text-sm font-medium text-content-secondary">{title}</p>
      {children ? <div className="max-w-sm hint">{children}</div> : null}
    </div>
  )
}
