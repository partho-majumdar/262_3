import { MODALITY_BLURB, MODALITY_LABELS, type ModalityName } from '../lib/types'
import { Icon, MODALITY_ICON } from './Icon'

const ORDER: ModalityName[] = ['url', 'html', 'vision']

interface Props {
  selected: ModalityName[]
  disabled: ModalityName[]
  onChange: (next: ModalityName[]) => void
}

/**
 * Modality toggle bar.
 *
 * The URL branch is intentionally not disable-able in a way that can leave the
 * request with zero modalities: turning every switch off would ask the backend
 * to predict from nothing, and it answers 0.5 with no evidence. Instead the last
 * remaining switch cannot be switched off, and the control says why.
 */
export function ModalityToggle({ selected, disabled, onChange }: Props) {
  const toggle = (name: ModalityName) => {
    const isOn = selected.includes(name)
    if (isOn && selected.length === 1) return
    onChange(isOn ? selected.filter((m) => m !== name) : [...selected, name])
  }

  return (
    <section className="card">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-5 py-3">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-content-primary">
          <Icon name="shield" className="h-4 w-4 text-content-muted" />
          Modalities to analyse
        </h2>
        <span className="text-xs text-content-muted">
          {selected.length === 1
            ? 'single modality - no fusion'
            : `${selected.length} selected - fused`}
        </span>
      </header>

      <div className="p-5">
        <p className="hint mb-4">
          Switch off a modality to exclude it. The fused score is recomputed from only the
          modalities you leave on.
        </p>

        <div
          role="group"
          aria-label="Modalities"
          className="grid gap-3 sm:grid-cols-3"
        >
          {ORDER.map((name) => {
            const on = selected.includes(name)
            const isLast = on && selected.length === 1
            const unavailable = disabled.includes(name)
            return (
              <button
                key={name}
                type="button"
                role="switch"
                aria-checked={on}
                aria-label={MODALITY_LABELS[name]}
                disabled={unavailable}
                onClick={() => toggle(name)}
                title={
                  unavailable
                    ? 'No model is loaded for this modality on the server.'
                    : MODALITY_BLURB[name]
                }
                className={[
                  'group relative flex flex-col gap-3 rounded-card border p-4 text-left',
                  'transition-colors duration-200',
                  unavailable
                    ? 'cursor-not-allowed border-line bg-base/40 opacity-60'
                    : on
                      ? 'border-accent/50 bg-accent-soft'
                      : 'border-line bg-elevated hover:border-white/20 hover:bg-white/[0.04]',
                ].join(' ')}
              >
                <span className="flex items-center justify-between gap-2">
                  <span className="flex items-center gap-2 text-sm font-medium text-content-primary">
                    <Icon
                      name={MODALITY_ICON[name]}
                      className={`h-4 w-4 ${on ? 'text-accent-hover' : 'text-content-muted'}`}
                    />
                    {MODALITY_LABELS[name]}
                  </span>

                  {/* Switch track */}
                  <span
                    aria-hidden="true"
                    className={[
                      'relative inline-flex h-5 w-9 shrink-0 items-center rounded-full',
                      'transition-colors duration-200',
                      on ? 'bg-accent' : 'bg-white/15',
                    ].join(' ')}
                  >
                    <span
                      className={[
                        'block h-3.5 w-3.5 rounded-full bg-white shadow-card',
                        'transition-transform duration-200',
                        on ? 'translate-x-[1.125rem]' : 'translate-x-[0.1875rem]',
                      ].join(' ')}
                    />
                  </span>
                </span>

                <span className="hint">
                  {unavailable
                    ? 'model not loaded'
                    : isLast
                      ? 'last one on - at least one is required'
                      : MODALITY_BLURB[name]}
                </span>
              </button>
            )
          })}
        </div>
      </div>
    </section>
  )
}
