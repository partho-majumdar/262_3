import { useEffect, useState } from 'react'
import { analyzeUrl, fetchHealth } from './lib/api'
import type { AnalyzeResponse, HealthResponse, ModalityName } from './lib/types'
import { Icon } from './components/Icon'
import { ModalityToggle } from './components/ModalityToggle'
import { ResultPanel } from './components/ResultPanel'
import { Badge, ResultSkeleton } from './components/ui'

/**
 * Demo fixtures, each verified against the shipped URL checkpoint.
 *
 * These are held-out PhiUSIIL test-split URLs, chosen because the model scores
 * them as their dataset label does. Do NOT swap in hand-written phishing URLs
 * such as `secure-login-verify.apple-id.account-check.top`: the corpus contains
 * no phishing on free-hosting platforms, so those score ~1e-5 (legitimate) and
 * the demo appears broken. See reports/adversarial_robustness.md.
 */
const EXAMPLES: { url: string; kind: 'legitimate' | 'phishing' }[] = [
  { url: 'https://www.google.com/', kind: 'legitimate' },
  { url: 'https://www.wikipedia.org/', kind: 'legitimate' },
  { url: 'https://github.com/anthropics', kind: 'legitimate' },
  { url: 'https://www.acnesupport.org.uk', kind: 'phishing' },
  { url: 'https://www.latestupdate9.com', kind: 'phishing' },
  { url: 'https://www.qusecure.com', kind: 'phishing' },
  { url: 'https://www.parisupdate.com', kind: 'phishing' },
]

function healthTone(health: HealthResponse | null) {
  if (!health) return { tone: 'neutral' as const, label: 'connecting to backend…' }
  const service = health.screenshot_service.replace('_', ' ')
  if (health.status === 'ok') {
    return { tone: 'success' as const, label: `models ${health.status} · screenshots ${service}` }
  }
  return { tone: 'warning' as const, label: `models ${health.status} · screenshots ${service}` }
}

export default function App() {
  const [url, setUrl] = useState('')
  const [selected, setSelected] = useState<ModalityName[]>(['url', 'html', 'vision'])
  const [res, setRes] = useState<AnalyzeResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [health, setHealth] = useState<HealthResponse | null>(null)

  useEffect(() => {
    const ctl = new AbortController()
    fetchHealth(ctl.signal)
      .then(setHealth)
      .catch(() => setHealth(null))
    return () => ctl.abort()
  }, [])

  const disabled: ModalityName[] = health
    ? ([
        !health.url_model && 'url',
        !health.html_model && 'html',
        !health.vision_model && 'vision',
      ].filter(Boolean) as ModalityName[])
    : []

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    if (!url.trim() || busy) return
    setBusy(true)
    setError(null)
    setRes(null)
    try {
      setRes(await analyzeUrl(url.trim(), selected))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const status = healthTone(health)

  return (
    <div className="min-h-screen bg-base">
      {/* one soft gradient, behind the header only */}
      <div aria-hidden="true" className="bg-app-glow pointer-events-none absolute inset-x-0 top-0 h-72" />

      <header className="relative border-b border-line bg-base/80 backdrop-blur">
        <div className="mx-auto flex max-w-app flex-wrap items-center justify-between gap-4 px-5 py-4 sm:px-8">
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-card bg-accent-soft text-accent-hover">
              <Icon name="shield" className="h-5 w-5" />
            </span>
            <div>
              <h1 className="text-base font-semibold tracking-tight text-content-primary">
                Multimodal Phishing URL Detector
              </h1>
              <p className="hint mt-0.5 hidden sm:block">
                URL string, page HTML and a rendered screenshot, fused into one calibrated score.
              </p>
            </div>
          </div>

          <Badge tone={status.tone}>
            <span
              aria-hidden="true"
              className={`h-1.5 w-1.5 rounded-full ${
                status.tone === 'success'
                  ? 'bg-success'
                  : status.tone === 'warning'
                    ? 'bg-warning'
                    : 'animate-pulse bg-content-muted'
              }`}
            />
            {status.label}
          </Badge>
        </div>
      </header>

      <main className="relative mx-auto max-w-app space-y-4 px-5 py-8 sm:px-8">
        {/* ---------- the single primary action ---------- */}
        <form onSubmit={submit} className="space-y-4">
          <section className="card p-5">
            <label htmlFor="url" className="label">
              URL to check
            </label>

            <div className="mt-2 flex flex-col gap-2 sm:flex-row">
              <div className="relative flex-1">
                <Icon
                  name="globe"
                  className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-content-muted"
                />
                <input
                  id="url"
                  type="text"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://example.com/login"
                  spellCheck={false}
                  autoComplete="off"
                  className="field pl-9"
                />
              </div>
              <button type="submit" disabled={busy || !url.trim()} className="btn-primary sm:w-36">
                {busy ? (
                  <>
                    <span
                      aria-hidden="true"
                      className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/30 border-t-white"
                    />
                    Analysing…
                  </>
                ) : (
                  <>
                    <Icon name="search" className="h-4 w-4" />
                    Analyse
                  </>
                )}
              </button>
            </div>

            <div className="mt-4 flex flex-wrap items-center gap-2">
              <span className="hint">try:</span>
              {EXAMPLES.map((ex) => (
                <button
                  key={ex.url}
                  type="button"
                  onClick={() => setUrl(ex.url)}
                  title={`${ex.url} (held-out test split — expected: ${ex.kind})`}
                  className="max-w-full truncate rounded-md border border-line bg-elevated px-2 py-1 font-mono text-xs text-content-secondary transition hover:border-accent/50 hover:text-content-primary"
                >
                  <span
                    aria-hidden="true"
                    className={`mr-1.5 inline-block h-1.5 w-1.5 rounded-full align-middle ${
                      ex.kind === 'phishing' ? 'bg-danger' : 'bg-success'
                    }`}
                  />
                  {ex.url.length > 46 ? `${ex.url.slice(0, 46)}…` : ex.url}
                </button>
              ))}
            </div>
          </section>

          <ModalityToggle selected={selected} disabled={disabled} onChange={setSelected} />
        </form>

        {/* ---------- loading ---------- */}
        {busy && (
          <>
            <p className="hint" role="status" aria-live="polite">
              Fetching the page and rendering it in a browser. This can take several seconds.
            </p>
            <ResultSkeleton />
          </>
        )}

        {/* ---------- error ---------- */}
        {error && (
          <div
            role="alert"
            className="card border-l-[3px] border-l-danger bg-danger-soft/40 p-5"
          >
            <p className="flex items-center gap-2 text-sm font-semibold text-danger-text">
              <Icon name="alert" className="h-4 w-4" />
              Request failed
            </p>
            <p className="mt-1.5 text-sm text-content-secondary">{error}</p>
          </div>
        )}

        {/* ---------- result ---------- */}
        {res && <ResultPanel res={res} />}
      </main>
    </div>
  )
}
