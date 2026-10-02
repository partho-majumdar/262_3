/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Layered dark surfaces. `base` is the page, `surface` cards,
        // `elevated` overlays and inputs.
        base: '#0B0D12',
        surface: '#12151C',
        elevated: '#181C25',

        line: {
          DEFAULT: 'rgba(255,255,255,0.08)',
          strong: 'rgba(255,255,255,0.14)',
        },

        content: {
          primary: '#E6E8EC',
          secondary: '#9AA1AE',
          muted: '#6B7280',
        },

        accent: {
          DEFAULT: '#6366F1',
          hover: '#7C7FF5',
          active: '#5457D6',
          soft: 'rgba(99,102,241,0.12)',
          ring: 'rgba(99,102,241,0.45)',
        },

        success: {
          DEFAULT: '#10B981',
          soft: 'rgba(16,185,129,0.12)',
          text: '#5DDBA9',
        },
        warning: {
          DEFAULT: '#F59E0B',
          soft: 'rgba(245,158,11,0.12)',
          text: '#F7C15C',
        },
        danger: {
          DEFAULT: '#EF4444',
          soft: 'rgba(239,68,68,0.12)',
          text: '#F58B8B',
        },
        info: {
          DEFAULT: '#3B82F6',
          soft: 'rgba(59,130,246,0.12)',
          text: '#7EB0F8',
        },
      },

      fontFamily: {
        sans: [
          'Inter',
          'Geist',
          'system-ui',
          '-apple-system',
          'Segoe UI',
          'Roboto',
          'Helvetica Neue',
          'sans-serif',
        ],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },

      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],
        xs: ['0.75rem', { lineHeight: '1.125rem' }],
        sm: ['0.875rem', { lineHeight: '1.375rem' }],
        base: ['1rem', { lineHeight: '1.5rem' }],
        lg: ['1.125rem', { lineHeight: '1.75rem' }],
        xl: ['1.25rem', { lineHeight: '1.75rem' }],
        '2xl': ['1.5rem', { lineHeight: '2rem' }],
        '3xl': ['2rem', { lineHeight: '2.25rem' }],
      },

      borderRadius: {
        DEFAULT: '8px',
        card: '12px',
        modal: '16px',
      },

      boxShadow: {
        // Low-opacity, defined by layering rather than glow.
        card: '0 1px 2px rgba(0,0,0,0.28), 0 1px 3px rgba(0,0,0,0.16)',
        raised: '0 4px 12px rgba(0,0,0,0.34)',
        overlay: '0 16px 48px rgba(0,0,0,0.48)',
        'focus-accent': '0 0 0 3px rgba(99,102,241,0.28)',
      },

      transitionDuration: {
        DEFAULT: '180ms',
      },
      transitionTimingFunction: {
        DEFAULT: 'cubic-bezier(0.2, 0, 0.2, 1)',
      },

      maxWidth: {
        app: '80rem',
      },
    },
  },
  plugins: [],
}
