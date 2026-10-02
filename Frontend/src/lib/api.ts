import type { AnalyzeResponse, HealthResponse, ModalityName } from './types'

/**
 * The backend is reached through the Vite dev proxy at /api, so the browser
 * stays same-origin and no CORS preflight or credential is involved.
 */
const BASE = '/api'

async function readError(res: Response): Promise<string> {
  try {
    const body = await res.json()
    if (body?.detail) {
      return typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    }
  } catch {
    /* fall through to the status text */
  }
  return `${res.status} ${res.statusText}`
}

export async function analyzeUrl(
  url: string,
  modalities: ModalityName[],
  signal?: AbortSignal,
  includeSignals = true,
): Promise<AnalyzeResponse> {
  const res = await fetch(`${BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    // includeSignals opens a TLS socket and a browser visit against the target,
    // so it is opt-out rather than opt-in for the dashboard.
    body: JSON.stringify({ url, modalities, explain: true, include_signals: includeSignals }),
    signal,
  })
  if (!res.ok) throw new Error(await readError(res))
  return (await res.json()) as AnalyzeResponse
}

export async function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`, { signal })
  if (!res.ok) throw new Error(await readError(res))
  return (await res.json()) as HealthResponse
}

export async function fetchMetrics(signal?: AbortSignal): Promise<Record<string, unknown>> {
  const res = await fetch(`${BASE}/metrics`, { signal })
  if (!res.ok) throw new Error(await readError(res))
  return (await res.json()) as Record<string, unknown>
}
