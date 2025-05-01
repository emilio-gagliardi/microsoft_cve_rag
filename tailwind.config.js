module.exports = {
  // 1. Configure Content Scanning Paths
  content: [
    './microsoft_cve_rag/application/data/templates/**/*.html',
    './microsoft_cve_rag/application/data/templates/quarterly_deep_dive/**/*.html',
    './microsoft_cve_rag/application/data/templates/quarterly_deep_dive/_components/**/*.html',
    './microsoft_cve_rag/application/data/templates/quarterly_deep_dive/_partials/**/*.html',
    './microsoft_cve_rag/application/static/js/**/*.js'
  ],

  // Keep safelist if dynamic class generation is a concern
  safelist: [
    // Brand Colors (keep if dynamically used)
    'bg-brand-blue', 'text-brand-blue', 'border-brand-blue',
    'bg-brand-green', 'text-brand-green', 'border-brand-green',
    'bg-brand-teal', 'text-brand-teal', 'border-brand-teal',
    'bg-brand-orange', 'text-brand-orange', 'border-brand-orange',
    'bg-brand-pink', 'text-brand-pink', 'border-brand-pink',
    'bg-brand-red', 'text-brand-red', 'border-brand-red',
    // Report UI (keep if dynamically used)
    'bg-report-background', 'text-report-text',
    'bg-report-background-dark', 'text-report-text-dark',
    'bg-report-surface', 'bg-report-surface-dark',
    'text-report-text-muted', 'text-report-text-muted-dark',
    'text-report-text-contrast', // Added
    'border-report-border', 'border-report-border-dark',
    'bg-report-accent', 'text-report-accent', 'border-report-accent',
    'bg-report-accent-dark', 'text-report-accent-dark', 'border-report-accent-dark',
    'bg-report-light-accent', // Added
    // Semantic Status Colors (useful for dynamic status messages/callouts)
    'bg-status-info-bg', 'text-status-info-text', 'border-status-info',
    'dark:bg-status-info-bg-dark', 'dark:text-status-info-text-dark', 'dark:border-status-info-border-dark',
    'bg-status-success-bg', 'text-status-success-text', 'border-status-success',
    'dark:bg-status-success-bg-dark', 'dark:text-status-success-text-dark', 'dark:border-status-success-border-dark',
    'bg-status-warning-bg', 'text-status-warning-text', 'border-status-warning',
    'dark:bg-status-warning-bg-dark', 'dark:text-status-warning-text-dark', 'dark:border-status-warning-border-dark',
    'bg-status-danger-bg', 'text-status-danger-text', 'border-status-danger',
    'dark:bg-status-danger-bg-dark', 'dark:text-status-danger-text-dark', 'dark:border-status-danger-border-dark',
    // Component classes (Keep if generated dynamically, otherwise content scanning is better)
    'stat-card', 'stat-card-value', 'stat-card-label',
    'callout', 'callout-info', 'callout-warning', 'callout-danger', 'callout-success', 'callout-quote', 'callout-takeaway',
    'data-table-container', 'data-table', 'table-actions', 'table-action-btn', 'edit', 'delete', 'view',
    'section-title',
    'chart-figure', 'plotly-chart-container',
  ],

  // 2. Configure Dark Mode Strategy
  darkMode: 'class',

  // 3. Define Theme Customizations
  theme: {
    fontFamily: {
      sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'Noto Sans', 'sans-serif', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji'],
      mono: ['Roboto Mono', "SFMono-Regular", "Consolas", "Liberation Mono", "Menlo", "monospace"],
    },
    extend: {
      colors: {
        // --- Base Brand Colors ---
        'brand': {
          'pink':   'var(--color-brand-pink)',
          'red':    'var(--color-brand-red)',
          'orange': 'var(--color-brand-orange)',
          'yellow': 'var(--color-brand-yellow)',
          'lime':   'var(--color-brand-lime)',
          'green':  'var(--color-brand-green)',
          'teal':   'var(--color-brand-teal)',
          'blue':   'var(--color-brand-blue)',
          'black':  'var(--color-brand-black)',
        },
        // --- Report UI Palette ---
        'report': {
          'background': 'var(--color-report-background)',
          'background-dark': 'var(--color-report-background-dark)',
          'surface': 'var(--color-report-surface)',
          'surface-dark': 'var(--color-report-surface-dark)',
          'border': 'var(--color-report-border)',
          'border-dark': 'var(--color-report-border-dark)',
          'text': 'var(--color-report-text)',
          'text-dark': 'var(--color-report-text-dark)',
          'text-muted': 'var(--color-report-text-muted)',
          'text-muted-dark': 'var(--color-report-text-muted-dark)',
          'text-contrast': 'var(--color-report-text-contrast)', // Added
          'accent': 'var(--color-report-accent)',
          'accent-dark': 'var(--color-report-accent-dark)',
          'light-accent': 'var(--color-report-light-accent)', // Added
           // Add 'alabaster' if used outside prose, otherwise prose handles it
          'alabaster': 'var(--color-report-alabaster)',
        },
        // --- Semantic Status Colors ---
        'status': {
          'info':         'var(--color-status-info)',
          'info-bg':      'var(--color-status-info-bg)',
          'info-bg-dark': 'var(--color-status-info-bg-dark)', // Define dark explicitly if needed by JS/logic
          'info-text':    'var(--color-status-info-text)',
          'info-text-dark':'var(--color-status-info-text-dark)',
          'info-border':  'var(--color-status-info-border)',
          'info-border-dark':'var(--color-status-info-border-dark)',

          'success':      'var(--color-status-success)',
          'success-bg':   'var(--color-status-success-bg)',
          'success-bg-dark':'var(--color-status-success-bg-dark)',
          'success-text': 'var(--color-status-success-text)',
          'success-text-dark':'var(--color-status-success-text-dark)',
          'success-border':'var(--color-status-success-border)',
          'success-border-dark':'var(--color-status-success-border-dark)',

          'warning':      'var(--color-status-warning)',
          'warning-bg':   'var(--color-status-warning-bg)',
          'warning-bg-dark':'var(--color-status-warning-bg-dark)',
          'warning-text': 'var(--color-status-warning-text)',
          'warning-text-dark':'var(--color-status-warning-text-dark)',
          'warning-border':'var(--color-status-warning-border)',
          'warning-border-dark':'var(--color-status-warning-border-dark)',

          'danger':       'var(--color-status-danger)',
          'danger-bg':    'var(--color-status-danger-bg)',
          'danger-bg-dark':'var(--color-status-danger-bg-dark)',
          'danger-text':  'var(--color-status-danger-text)',
          'danger-text-dark':'var(--color-status-danger-text-dark)',
          'danger-border':'var(--color-status-danger-border)',
          'danger-border-dark':'var(--color-status-danger-border-dark)',
        },
        // --- Misc UI Colors ---
        'footer': {
            'background': 'var(--color-footer-background)',
            'text': 'var(--color-footer-text)',
            'link': 'var(--color-footer-link)',
        },
      },
      // Extend other theme sections like spacing, borderRadius if needed
      // spacing: { ... },
      // borderRadius: { ... },
    },
  },

  // 4. Configure Plugins
  plugins: [
    require('@tailwindcss/typography'),
    // require('@tailwindcss/forms'),
    // require('@tailwindcss/aspect-ratio'),
  ],
};
