import type { AnalyzeResponse, ModalityScore } from '../lib/types'
import { MODALITY_LABELS, SIGNAL_FEATURE_HIGHLIGHTS, SIGNAL_LABELS } from '../lib/types'
import { Icon, MODALITY_ICON } from './Icon'
import { Badge, EmptyState } from './ui'

const pct = (v: number | null | undefined) =>
  v === null || v === undefined ? '—' : `${(v * 100).toFixed(2)}%`

type Tone = 'danger' | 'success' | 'neutral'

function Verdict({ res }: { res: AnalyzeResponse }) {
  const p = res.probability_phishing
  const tone: Tone =
    res.verdict === 'phishing' ? 'danger' : res.verdict === 'legitimate' ? 'success' : 'neutral'

  const accentBar =
    tone === 'danger' ? 'bg-danger' : tone === 'success' ? 'bg-success' : 'bg-white/30'
  const textTone =
    tone === 'danger'
      ? 'text-danger-text'
      : tone === 'success'
        ? 'text-success-text'
        : 'text-content-primary'

  const label =
    res.verdict === 'phishing'
      ? 'Phishing'
      : res.verdict === 'legitimate'
        ? 'Legitimate'
        : 'No verdict'

  return (
    <section
      className={`card border-l-[3px] p-5 ${
        tone === 'danger'
          ? 'border-l-danger'
          : tone === 'success'
            ? 'border-l-success'
            : 'border-l-white/20'
      }`}
    >
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="min-w-0">
          <p className="label">Verdict</p>
          <p className={`mt-1 text-3xl font-semibold tracking-tight ${textTone}`}>{label}</p>
          <p className="mt-2 flex items-center gap-1.5 text-xs text-content-muted">
            <Icon name="external" className="h-3.5 w-3.5" />
            <span className="truncate" title={res.url}>
              {res.url}
            </span>
          </p>
        </div>

        <div className="text-right">
          <p className="label">Phishing probability</p>
          <p className={`numeric mt-1 text-3xl font-semibold tracking-tight ${textTone}`}>
            {pct(p)}
          </p>
        </div>
      </div>

      <div className="mt-5">
        <div
          className="h-1.5 w-full overflow-hidden rounded-full bg-white/[0.07]"
          role="img"
          aria-label={`Phishing probability ${(p * 100).toFixed(1)} percent`}
        >
          <div
            className={`h-full rounded-full transition-[width] duration-500 ease-out ${accentBar}`}
            style={{ width: `${Math.max(Math.round(p * 100), 1)}%` }}
          />
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2">
          <Badge tone={res.fused ? 'accent' : 'neutral'}>
            <Icon name={res.fused ? 'shield' : 'info'} className="h-3 w-3" />
            {res.fused ? 'fused across modalities' : 'single modality'}
          </Badge>
          {/* An uncalibrated score is a raw sigmoid output, not a frequency.
              Saying so here stops the number being read as a real-world rate. */}
          <Badge tone={res.probability_is_calibrated ? 'success' : 'neutral'}>
            <Icon name={res.probability_is_calibrated ? 'check' : 'alert'} className="h-3 w-3" />
            {res.probability_is_calibrated ? 'calibrated score' : 'raw score, not calibrated'}
          </Badge>
          {/* One text node, so the confidence figure is not confused with the
              headline probability when reading the value out of the DOM. */}
          <span className="numeric hint">Confidence {pct(res.confidence)}</span>
        </div>

        {!res.probability_is_calibrated && (
          <p className="mt-2 flex items-start gap-1.5 text-xs text-content-muted">
            <Icon name="info" className="mt-px h-3 w-3 shrink-0" />
            <span>
              No calibration artifact is loaded, so this probability is an uncalibrated model
              output. It ranks URLs but is not a calibrated estimate of real-world prevalence.
            </span>
          </p>
        )}
      </div>
    </section>
  )
}

function ModalityRow({ m }: { m: ModalityScore }) {
  return (
    <li className="flex items-start justify-between gap-4 border-b border-line px-5 py-3 transition-colors last:border-0 hover:bg-white/[0.02]">
      <div className="flex min-w-0 items-start gap-3">
        <span
          className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg ${
            m.available ? 'bg-accent-soft text-accent-hover' : 'bg-white/[0.05] text-content-muted'
          }`}
        >
          <Icon name={MODALITY_ICON[m.name]} className="h-3.5 w-3.5" />
        </span>
        <div className="min-w-0">
          <p className="text-sm font-medium text-content-primary">{MODALITY_LABELS[m.name]}</p>
          {!m.available && m.reason && (
            <p className="mt-0.5 flex items-start gap-1 text-xs text-warning-text">
              <Icon name="alert" className="mt-px h-3 w-3 shrink-0" />
              <span>unavailable: {m.reason}</span>
            </p>
          )}
        </div>
      </div>

      <div className="shrink-0 text-right">
        <p
          className={`numeric text-sm ${
            m.probability === null || m.probability === undefined
              ? 'text-content-muted'
              : 'text-content-primary'
          }`}
        >
          {pct(m.probability)}
        </p>
        {m.weight !== null && (
          <p className="numeric text-xs text-content-muted">
            weight {(m.weight ?? 0).toFixed(3)}
          </p>
        )}
      </div>
    </li>
  )
}

/**
 * Leave-one-modality-out counterfactual.
 *
 * This is a measured ablation, not a gate weight: each available modality is
 * withheld from the fusion and the score is recomputed. It answers "what would
 * the verdict have been without this evidence", which is the question a user
 * reviewing a verdict actually wants answered.
 */
function ModalityInfluencePanel({ res }: { res: AnalyzeResponse }) {
  const influence = res.modality_influence
  if (!influence || influence.method === 'unavailable') return null

  const entries = Object.entries(
    (influence.per_modality ?? {}) as Record<string, number>,
  ).filter(([, v]) => typeof v === 'number' && Number.isFinite(v))

  if (entries.length === 0) return null

  const scale = Math.max(...entries.map(([, v]) => Math.abs(v)), 0.01)

  return (
    <section className="card overflow-hidden">
      <header className="flex items-center justify-between border-b border-line px-5 py-3">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-content-primary">
          <Icon name="info" className="h-4 w-4 text-content-muted" />
          Modality influence
          <span className="text-xs font-normal text-content-muted">
            leave-one-out ablation
          </span>
        </h2>
      </header>

      <ul className="space-y-2.5 p-5">
        {entries.map(([name, delta]) => {
          const label = MODALITY_LABELS[name as ModalityScore['name']] ?? name
          const pushesPhishing = delta > 0
          const mag = Math.min(Math.abs(delta) / scale, 1)
          return (
            <li key={name} className="flex items-center gap-3">
              <span className="w-36 shrink-0 truncate text-xs text-content-secondary">{label}</span>
              <span className="relative h-2.5 flex-1 rounded-full bg-white/[0.05]">
                <span
                  aria-hidden="true"
                  className="absolute left-1/2 top-0 h-2.5 w-px -translate-x-1/2 bg-white/20"
                />
                <span
                  className={`absolute top-0 h-2.5 rounded-full ${
                    pushesPhishing ? 'bg-danger/80' : 'bg-success/80'
                  }`}
                  style={
                    pushesPhishing
                      ? { left: '50%', width: `${(mag * 50).toFixed(1)}%` }
                      : { right: '50%', width: `${(mag * 50).toFixed(1)}%` }
                  }
                />
              </span>
              <span className="numeric w-16 shrink-0 text-right text-xs text-content-secondary">
                {delta > 0 ? '+' : ''}
                {delta.toFixed(3)}
              </span>
            </li>
          )
        })}
      </ul>

      <p className="border-t border-line px-5 py-3 hint">
        Change in phishing probability when that modality is withheld. Red means removing it
        lowered the score, so it was the evidence carrying the verdict.
      </p>
    </section>
  )
}

/**
 * Live-measured signals.
 *
 * These are deliberately separated from the modality scores above. The URL,
 * HTML and vision rows are model outputs that feed the verdict; everything in
 * this panel is a direct observation of the target. Presenting them beside the
 * modality scores without that distinction would imply they were scored, which
 * they are not -- only the graph branch has a trained head.
 */
function SignalsPanel({ res }: { res: AnalyzeResponse }) {
  const signals = res.signals ?? {}
  const names = Object.keys(signals)
  if (names.length === 0) return null

  return (
    <section className="card overflow-hidden">
      <header className="flex items-center justify-between border-b border-line px-5 py-3">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-content-primary">
          <Icon name="shield" className="h-4 w-4 text-content-muted" />
          Live signals
          <span className="text-xs font-normal text-content-muted">observed, not scored</span>
        </h2>
      </header>

      <ul className="divide-y divide-line">
        {names.map((name) => {
          const block = signals[name]
          const highlights = (SIGNAL_FEATURE_HIGHLIGHTS[name] ?? []).filter(
            (k) => block.features && k in block.features,
          )
          return (
            <li key={name} className="px-5 py-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-sm font-medium text-content-primary">
                  {SIGNAL_LABELS[name] ?? name}
                </span>
                {block.probability !== null && block.probability !== undefined ? (
                  <span className="numeric text-sm text-content-primary">
                    {pct(block.probability)}
                  </span>
                ) : (
                  <Badge tone={block.available ? 'success' : 'neutral'}>
                    {block.available ? 'observed' : 'unavailable'}
                  </Badge>
                )}
              </div>

              {block.reason && !block.available && (
                <p className="mt-1 flex items-start gap-1 text-xs text-warning-text">
                  <Icon name="alert" className="mt-px h-3 w-3 shrink-0" />
                  <span>{block.reason}</span>
                </p>
              )}

              {highlights.length > 0 && (
                <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 sm:grid-cols-3">
                  {highlights.map((k) => (
                    <div key={k} className="flex items-baseline justify-between gap-2">
                      <dt className="truncate font-mono text-[11px] text-content-muted" title={k}>
                        {k}
                      </dt>
                      <dd className="numeric text-xs text-content-secondary">
                        {(block.features as Record<string, number>)[k].toFixed(2)}
                      </dd>
                    </div>
                  ))}
                </dl>
              )}
            </li>
          )
        })}
      </ul>

      <p className="border-t border-line px-5 py-3 hint">
        Measured directly from the target: TLS certificate metadata, live JavaScript behaviour,
        page geometry and brand-image impersonation. These are reported for inspection and are
        not inputs to the verdict -- only the graph row is a trained model output.
      </p>
    </section>
  )
}

export function ResultPanel({ res }: { res: AnalyzeResponse }) {
  const features = res.top_features?.url ?? []
  const hasAcquisition = Object.keys(res.acquisition).length > 0

  return (
    <div className="space-y-4">
      <Verdict res={res} />

      <ModalityInfluencePanel res={res} />

      <SignalsPanel res={res} />

      <div className="grid gap-4 lg:grid-cols-2">
        {/* ---------- per-modality ---------- */}
        <section className="card overflow-hidden">
          <header className="flex items-center justify-between border-b border-line px-5 py-3">
            <h2 className="flex items-center gap-2 text-sm font-semibold text-content-primary">
              <Icon name="shield" className="h-4 w-4 text-content-muted" />
              Per-modality scores
            </h2>
          </header>
          <ul>
            {res.modalities.map((m) => (
              <ModalityRow key={m.name} m={m} />
            ))}
          </ul>
          <p className="border-t border-line px-5 py-3 hint">
            A blank score means the evidence was not available, not that the model scored it
            zero.
          </p>
        </section>

        {/* ---------- attribution ---------- */}
        <section className="card overflow-hidden">
          <header className="flex items-center justify-between border-b border-line px-5 py-3">
            <h2 className="flex items-center gap-2 text-sm font-semibold text-content-primary">
              <Icon name="search" className="h-4 w-4 text-content-muted" />
              URL attribution
              <span className="text-xs font-normal text-content-muted">
                integrated gradients
              </span>
            </h2>
          </header>

          {features.length === 0 ? (
            <EmptyState icon={<Icon name="info" />} title="No attribution returned for this URL." />
          ) : (
            <ul className="space-y-2.5 p-5">
              {features.map((f) => {
                const mag = Math.min(Math.abs(f.contribution) / 0.5, 1)
                const positive = f.contribution > 0
                return (
                  <li key={f.label} className="flex items-center gap-3">
                    <span
                      className="w-36 shrink-0 truncate font-mono text-xs text-content-secondary"
                      title={f.label}
                    >
                      {f.label}
                    </span>
                    <span className="relative h-2.5 flex-1 rounded-full bg-white/[0.05]">
                      {/* centre tick so the sign is readable at a glance */}
                      <span
                        aria-hidden="true"
                        className="absolute left-1/2 top-0 h-2.5 w-px -translate-x-1/2 bg-white/20"
                      />
                      <span
                        className={`absolute top-0 h-2.5 rounded-full transition-[width] duration-300 ${
                          positive ? 'bg-danger/80' : 'bg-success/80'
                        }`}
                        style={
                          positive
                            ? { left: '50%', width: `${(mag * 50).toFixed(1)}%` }
                            : { right: '50%', width: `${(mag * 50).toFixed(1)}%` }
                        }
                      />
                    </span>
                    <span className="numeric w-16 shrink-0 text-right text-xs text-content-secondary">
                      {f.contribution.toFixed(3)}
                    </span>
                  </li>
                )
              })}
            </ul>
          )}

          <p className="border-t border-line px-5 py-3 hint">
            Red pushes the score toward phishing, green toward legitimate.
          </p>
        </section>
      </div>

      {/* ---------- acquisition ---------- */}
      {(res.warnings.length > 0 || hasAcquisition) && (
        <section className="card overflow-hidden">
          <header className="flex items-center justify-between border-b border-line px-5 py-3">
            <h2 className="flex items-center gap-2 text-sm font-semibold text-content-primary">
              <Icon name="info" className="h-4 w-4 text-content-muted" />
              Acquisition detail
            </h2>
          </header>

          {hasAcquisition ? (
            <dl className="divide-y divide-line">
              {Object.entries(res.acquisition).map(([k, v]) => (
                <div
                  key={k}
                  className="flex items-center justify-between gap-4 px-5 py-2 transition-colors hover:bg-white/[0.02]"
                >
                  <dt className="text-xs text-content-secondary">{k}</dt>
                  <dd className="numeric truncate text-xs text-content-primary">{String(v)}</dd>
                </div>
              ))}
            </dl>
          ) : null}

          {res.warnings.length > 0 && (
            <ul className="space-y-1.5 border-t border-line px-5 py-4">
              {res.warnings.map((w) => (
                <li key={w} className="flex items-start gap-2 text-xs text-warning-text">
                  <Icon name="alert" className="mt-px h-3.5 w-3.5 shrink-0" />
                  <span>{w}</span>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}
    </div>
  )
}
