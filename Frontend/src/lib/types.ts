/** Types mirroring the backend's `/analyze` response. */

export type ModalityName = 'url' | 'html' | 'vision'

export interface ModalityScore {
  name: ModalityName
  available: boolean
  probability: number | null
  weight: number | null
  reason: string | null
}

export interface ExplainItem {
  label: string
  contribution: number
}

/**
 * Leave-one-modality-out counterfactual from the backend's `explain_fusion`.
 *
 * `per_modality` maps a modality to the change in fused phishing probability
 * when that modality alone is withheld. A positive value means the modality was
 * pushing the verdict toward phishing; withholding it lowers the score. The
 * value is a measured counterfactual, not the fusion gate weight.
 */
export interface ModalityInfluence {
  method: 'leave_one_modality_out' | 'unavailable' | string
  note?: string
  per_modality: Partial<Record<ModalityName, number>>
}

/**
 * One measured signal group from the backend.
 *
 * `features` is *observed, not scored*: these come from a live inspection of
 * the target and no trained classifier head consumes them yet, so the UI must
 * not present them as contributing to the verdict. Only `graph` carries a real
 * model probability.
 */
export interface SignalBlock {
  available: boolean
  reason?: string | null
  probability?: number | null
  features?: Record<string, number>
}

export interface AnalyzeResponse {
  url: string
  verdict: 'phishing' | 'legitimate' | 'unknown'
  probability_phishing: number
  confidence: number
  /**
   * True when temperature scaling was applied to the reported probability.
   * False means the number is a raw sigmoid output and is not a calibrated
   * frequency estimate, so it must not be read as "92% of these are phishing".
   */
  probability_is_calibrated: boolean
  fused: boolean
  modalities: ModalityScore[]
  top_features: Record<string, ExplainItem[]>
  modality_influence: ModalityInfluence | Record<string, never>
  signals?: Record<string, SignalBlock>
  acquisition: Record<string, unknown>
  warnings: string[]
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  url_model: boolean
  html_model: boolean
  vision_model: boolean
  fusion_model: boolean
  graph_model?: boolean
  screenshot_service: 'in_process' | 'container' | 'unavailable'
  calibrated?: boolean
  notes: string[]
}

export const SIGNAL_LABELS: Record<string, string> = {
  certificate: 'TLS certificate',
  page: 'Page probe (JS · layout · logos)',
  graph: 'Domain-IP graph',
}

/**
 * Features worth surfacing first, per group. The backend returns dozens; showing
 * them all would bury the two or three that actually indicate impersonation.
 * Order is the display order.
 */
export const SIGNAL_FEATURE_HIGHLIGHTS: Record<string, string[]> = {
  certificate: ['verify_failed', 'self_signed_guess', 'days_to_expiry', 'is_expired', 'hostname_mismatch'],
  page: [
    'brand_impersonation_any',
    'n_brand_host_mismatch_images',
    'form_action_cross_origin',
    'keyboard_event_intercepts',
    'keystroke_suppressions',
    'window_open_calls',
    'eval_calls',
    'obfuscated_script_hits',
    'timer_redirects',
    'n_password_fields',
    'password_vertical_position',
    'form_center_deviation',
  ],
  graph: [],
}

export const MODALITY_LABELS: Record<ModalityName, string> = {
  url: 'URL string',
  html: 'Page HTML',
  vision: 'Screenshot',
}

export const MODALITY_BLURB: Record<ModalityName, string> = {
  url: 'Character model over the raw URL. Always available.',
  html: 'DOM and text of the fetched page. Needs the page to load.',
  vision: 'Rendered screenshot. Needs a live browser render.',
}
