import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ModalityToggle } from '../components/ModalityToggle'
import { ResultPanel } from '../components/ResultPanel'
import type { AnalyzeResponse } from '../lib/types'

describe('ModalityToggle', () => {
  it('renders one switch per modality', () => {
    render(<ModalityToggle selected={['url']} disabled={[]} onChange={() => {}} />)
    expect(screen.getAllByRole('switch')).toHaveLength(3)
  })

  it('reflects the selected state', () => {
    render(<ModalityToggle selected={['url', 'vision']} disabled={[]} onChange={() => {}} />)
    expect(screen.getByRole('switch', { name: 'URL string' })).toBeChecked()
    expect(screen.getByRole('switch', { name: 'Screenshot' })).toBeChecked()
    expect(screen.getByRole('switch', { name: 'Page HTML' })).not.toBeChecked()
  })

  it('turns a modality on when clicked', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<ModalityToggle selected={['url']} disabled={[]} onChange={onChange} />)
    await user.click(screen.getByRole('switch', { name: 'Page HTML' }))
    expect(onChange).toHaveBeenCalledWith(['url', 'html'])
  })

  it('turns a modality off when clicked', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<ModalityToggle selected={['url', 'html']} disabled={[]} onChange={onChange} />)
    await user.click(screen.getByRole('switch', { name: 'Page HTML' }))
    expect(onChange).toHaveBeenCalledWith(['url'])
  })

  it('refuses to switch off the last remaining modality', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<ModalityToggle selected={['url']} disabled={[]} onChange={onChange} />)
    await user.click(screen.getByRole('switch', { name: 'URL string' }))
    expect(onChange).not.toHaveBeenCalled()
  })

  it('says why the last modality cannot be turned off', () => {
    render(<ModalityToggle selected={['url']} disabled={[]} onChange={() => {}} />)
    expect(screen.getByText(/last one on/i)).toBeInTheDocument()
  })

  it('reports single-modality mode', () => {
    render(<ModalityToggle selected={['url']} disabled={[]} onChange={() => {}} />)
    expect(screen.getByText(/single modality - no fusion/i)).toBeInTheDocument()
  })

  it('reports fusion mode when several are selected', () => {
    render(<ModalityToggle selected={['url', 'html']} disabled={[]} onChange={() => {}} />)
    expect(screen.getByText(/2 selected - fused/i)).toBeInTheDocument()
  })

  it('disables a modality whose model is not loaded', () => {
    render(<ModalityToggle selected={['url']} disabled={['vision']} onChange={() => {}} />)
    expect(screen.getByRole('switch', { name: 'Screenshot' })).toBeDisabled()
    expect(screen.getByText('model not loaded')).toBeInTheDocument()
  })
})

const SAMPLE: AnalyzeResponse = {
  url: 'https://paypal-secure.top/login',
  verdict: 'phishing',
  probability_phishing: 0.973,
  confidence: 0.973,
  probability_is_calibrated: true,
  fused: true,
  modalities: [
    { name: 'url', available: true, probability: 0.98, weight: 0.7, reason: null },
    { name: 'html', available: true, probability: 0.99, weight: 0.2, reason: null },
    { name: 'vision', available: false, probability: null, weight: 0.1, reason: 'screenshot timed out' },
  ],
  top_features: {
    url: [
      { label: 'char 12', contribution: 0.42 },
      { label: 'char 30', contribution: -0.11 },
    ],
  },
  modality_influence: {
    method: 'leave_one_modality_out',
    note: 'change in fused phishing probability when this modality is withheld',
    per_modality: { url: 0.41, html: 0.22 },
  },
  acquisition: { http_status: 200 },
  warnings: ['vision embedding failed'],
}

describe('ResultPanel', () => {
  it('shows the verdict and probability', () => {
    render(<ResultPanel res={SAMPLE} />)
    expect(screen.getByText('Phishing')).toBeInTheDocument()
    expect(screen.getByText('97.30%')).toBeInTheDocument()
  })

  it('shows a legitimate verdict', () => {
    render(<ResultPanel res={{ ...SAMPLE, verdict: 'legitimate', probability_phishing: 0.02 }} />)
    expect(screen.getByText('Legitimate')).toBeInTheDocument()
  })

  it('shows per-modality scores', () => {
    render(<ResultPanel res={SAMPLE} />)
    expect(screen.getByText('98.00%')).toBeInTheDocument()
    expect(screen.getByText('99.00%')).toBeInTheDocument()
  })

  it('shows a dash rather than a zero for an unavailable modality', () => {
    render(<ResultPanel res={SAMPLE} />)
    expect(screen.getByText('—')).toBeInTheDocument()
  })

  it('explains why a modality was unavailable', () => {
    render(<ResultPanel res={SAMPLE} />)
    expect(screen.getByText(/unavailable: screenshot timed out/i)).toBeInTheDocument()
  })

  it('notes that a blank score is not a zero score', () => {
    render(<ResultPanel res={SAMPLE} />)
    expect(screen.getByText(/not that the model scored it zero/i)).toBeInTheDocument()
  })

  it('renders attribution contributions', () => {
    render(<ResultPanel res={SAMPLE} />)
    expect(screen.getByText('char 12')).toBeInTheDocument()
    expect(screen.getByText('0.420')).toBeInTheDocument()
  })

  it('reports that the score is fused', () => {
    render(<ResultPanel res={SAMPLE} />)
    expect(screen.getByText(/fused across modalities/i)).toBeInTheDocument()
  })

  it('reports single-modality scoring', () => {
    render(<ResultPanel res={{ ...SAMPLE, fused: false }} />)
    expect(screen.getByText(/single modality/i)).toBeInTheDocument()
  })

  it('surfaces warnings', () => {
    render(<ResultPanel res={SAMPLE} />)
    expect(screen.getByText('vision embedding failed')).toBeInTheDocument()
  })

  it('handles a response with no attribution', () => {
    render(<ResultPanel res={{ ...SAMPLE, top_features: {} }} />)
    expect(screen.getByText(/no attribution returned/i)).toBeInTheDocument()
  })

  it('marks the score as calibrated', () => {
    render(<ResultPanel res={SAMPLE} />)
    expect(screen.getByText(/calibrated score/i)).toBeInTheDocument()
    expect(screen.queryByText(/not a calibrated estimate/i)).not.toBeInTheDocument()
  })

  // The headline number is only meaningful as a frequency if it was calibrated.
  // Showing an uncalibrated sigmoid output as a percentage invites the reader to
  // treat it as prevalence, so the UI has to say otherwise.
  it('warns that an uncalibrated score is not a prevalence estimate', () => {
    render(<ResultPanel res={{ ...SAMPLE, probability_is_calibrated: false }} />)
    expect(screen.getByText(/raw score, not calibrated/i)).toBeInTheDocument()
    expect(screen.getByText(/not a calibrated estimate/i)).toBeInTheDocument()
  })

  it('renders leave-one-out modality influence', () => {
    render(<ResultPanel res={SAMPLE} />)
    expect(screen.getByText(/modality influence/i)).toBeInTheDocument()
    expect(screen.getByText('+0.410')).toBeInTheDocument()
    expect(screen.getByText('+0.220')).toBeInTheDocument()
  })

  it('omits the influence panel when the backend reports it unavailable', () => {
    render(
      <ResultPanel
        res={{ ...SAMPLE, modality_influence: { method: 'unavailable', per_modality: {} } }}
      />,
    )
    expect(screen.queryByText(/modality influence/i)).not.toBeInTheDocument()
  })

  it('tolerates a missing influence block', () => {
    render(<ResultPanel res={{ ...SAMPLE, modality_influence: {} }} />)
    expect(screen.queryByText(/modality influence/i)).not.toBeInTheDocument()
  })

  it('omits the signals panel when the backend sent none', () => {
    render(<ResultPanel res={SAMPLE} />)
    expect(screen.queryByText(/live signals/i)).not.toBeInTheDocument()
  })

  // The signal rows are observations, not model outputs. Labelling them as
  // scored would overstate how much of the verdict they explain.
  it('labels signals as observed rather than scored', () => {
    render(
      <ResultPanel
        res={{
          ...SAMPLE,
          signals: {
            certificate: { available: true, features: { verify_failed: 1, days_to_expiry: 3.5 } },
            page: { available: false, reason: 'navigation timeout' },
          },
        }}
      />,
    )
    expect(screen.getByText(/live signals/i)).toBeInTheDocument()
    expect(screen.getByText(/observed, not scored/i)).toBeInTheDocument()
    expect(screen.getByText('TLS certificate')).toBeInTheDocument()
    expect(screen.getByText('verify_failed')).toBeInTheDocument()
  })

  it('shows the reason a signal group is unavailable', () => {
    render(
      <ResultPanel
        res={{ ...SAMPLE, signals: { page: { available: false, reason: 'navigation timeout' } } }}
      />,
    )
    expect(screen.getByText('navigation timeout')).toBeInTheDocument()
  })

  it('shows a probability instead of a badge for the graph branch', () => {
    render(
      <ResultPanel
        res={{ ...SAMPLE, signals: { graph: { available: true, probability: 0.5413 } } }}
      />,
    )
    expect(screen.getByText('54.13%')).toBeInTheDocument()
  })

  it('does not show highlight features the backend did not return', () => {
    render(
      <ResultPanel
        res={{ ...SAMPLE, signals: { page: { available: true, features: { n_forms: 2 } } } }}
      />,
    )
    expect(screen.queryByText('eval_calls')).not.toBeInTheDocument()
  })
})
