import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // Backgrounds
        bg: {
          primary: '#0a0a0f',
          secondary: '#12121a',
          tertiary: '#1a1a25',
          glass: 'rgba(18, 18, 26, 0.8)',
        },
        // Accent colors
        accent: {
          primary: '#6366f1',
          secondary: '#818cf8',
          glow: 'rgba(99, 102, 241, 0.3)',
        },
        // Text colors
        text: {
          primary: '#f8fafc',
          secondary: '#94a3b8',
          muted: '#64748b',
        },
        // Status colors
        status: {
          success: '#22c55e',
          warning: '#f59e0b',
          error: '#ef4444',
          info: '#3b82f6',
        },
        // Border colors
        border: {
          DEFAULT: 'rgba(255, 255, 255, 0.1)',
          glow: 'rgba(99, 102, 241, 0.5)',
        },
        // Knowledge graph node colors
        graph: {
          paper: '#6366f1',
          concept: '#22c55e',
          algorithm: '#f59e0b',
          repository: '#06b6d4',
          function: '#ec4899',
          class: '#ec4899',
          mapping: '#8b5cf6',
          file: '#64748b',
        },
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      fontSize: {
        'display': ['2.5rem', { lineHeight: '1.2', fontWeight: '700' }],
        'heading-1': ['2rem', { lineHeight: '1.25', fontWeight: '700' }],
        'heading-2': ['1.5rem', { lineHeight: '1.3', fontWeight: '600' }],
        'heading-3': ['1.25rem', { lineHeight: '1.4', fontWeight: '600' }],
        'body-lg': ['1.125rem', { lineHeight: '1.6', fontWeight: '400' }],
        'body': ['1rem', { lineHeight: '1.6', fontWeight: '400' }],
        'body-sm': ['0.875rem', { lineHeight: '1.5', fontWeight: '400' }],
        'caption': ['0.8125rem', { lineHeight: '1.4', fontWeight: '400' }],
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem',
      },
      backdropBlur: {
        glass: '20px',
      },
      boxShadow: {
        glow: '0 0 30px rgba(99, 102, 241, 0.3)',
        'glow-sm': '0 0 15px rgba(99, 102, 241, 0.2)',
      },
      animation: {
        'slide-in': 'slideIn 0.3s ease-out',
        'fade-in': 'fadeIn 0.2s ease-out',
        'pulse-glow': 'pulseGlow 2s ease-in-out infinite',
      },
      keyframes: {
        slideIn: {
          '0%': { transform: 'translateX(100%)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        pulseGlow: {
          '0%, 100%': { boxShadow: '0 0 15px rgba(99, 102, 241, 0.3)' },
          '50%': { boxShadow: '0 0 30px rgba(99, 102, 241, 0.5)' },
        },
      },
    },
  },
  plugins: [],
}

export default config
