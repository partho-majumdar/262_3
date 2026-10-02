export type IconName =
  | 'shield'
  | 'globe'
  | 'code'
  | 'image'
  | 'search'
  | 'check'
  | 'alert'
  | 'info'
  | 'x'
  | 'external'

const PATHS: Record<IconName, JSX.Element> = {
  shield: <path d="M12 3l7 3v6c0 4.5-3 7.7-7 9-4-1.3-7-4.5-7-9V6l7-3z" />,
  globe: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M3 12h18M12 3c2.5 2.7 2.5 15.3 0 18M12 3c-2.5 2.7-2.5 15.3 0 18" />
    </>
  ),
  code: <path d="M9 8l-4 4 4 4M15 8l4 4-4 4" />,
  image: (
    <>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <circle cx="8.5" cy="9.5" r="1.5" />
      <path d="M21 16l-5-5-6 6-2-2-5 5" />
    </>
  ),
  search: (
    <>
      <circle cx="11" cy="11" r="7" />
      <path d="M20 20l-3.5-3.5" />
    </>
  ),
  check: <path d="M5 13l4 4L19 7" />,
  alert: (
    <>
      <path d="M12 4l9 16H3l9-16z" />
      <path d="M12 10v4M12 17.5v.5" />
    </>
  ),
  info: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5M12 8v.5" />
    </>
  ),
  x: <path d="M6 6l12 12M18 6L6 18" />,
  external: <path d="M14 4h6v6M20 4l-8 8M18 14v5a1 1 0 01-1 1H5a1 1 0 01-1-1V7a1 1 0 011-1h5" />,
}

const FILLED: IconName[] = ['shield', 'info']

interface Props {
  name: IconName
  className?: string
  /** Decorative by default; pass a title to expose it to assistive tech. */
  title?: string
}

export function Icon({ name, className = 'h-4 w-4', title }: Props) {
  const filled = FILLED.includes(name)
  return (
    <svg
      viewBox="0 0 24 24"
      className={className}
      fill={filled ? 'currentColor' : 'none'}
      stroke={filled ? 'none' : 'currentColor'}
      strokeWidth={filled ? undefined : 1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      role={title ? 'img' : undefined}
      aria-hidden={title ? undefined : true}
      focusable="false"
    >
      {title ? <title>{title}</title> : null}
      {PATHS[name]}
    </svg>
  )
}

/** Icon per modality, used by the toggle and the score list. */
export const MODALITY_ICON: Record<string, IconName> = {
  url: 'globe',
  html: 'code',
  vision: 'image',
}
