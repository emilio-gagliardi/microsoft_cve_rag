module.exports = {
  // 1. Configure Content Scanning Paths
  content: [
    './microsoft_cve_rag/application/data/templates/**/*.html',
    './microsoft_cve_rag/application/data/templates/quarterly_deep_dive/**/*.html',
    './microsoft_cve_rag/application/data/templates/quarterly_deep_dive/_components/**/*.html',
    './microsoft_cve_rag/application/data/templates/quarterly_deep_dive/_partials/**/*.html',
    './microsoft_cve_rag/application/static/js/**/*.js',
    './microsoft_cve_rag/application/reports/**/*.py', // Added to scan Python files for classes
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
    'dark:bg-report-background-dark', 'dark:text-report-text-dark',
    'bg-report-surface', 'bg-report-surface-dark',
    'text-report-text-muted', 'dark:text-report-text-muted-dark',
    'text-report-text-contrast',
    'border-report-border', 'dark:border-report-border-dark',
    'bg-report-accent', 'text-report-accent', 'border-report-accent',
    'dark:bg-report-accent-dark', 'dark:text-report-accent-dark', 'dark:border-report-accent-dark',
    'bg-report-light-accent',
    // Semantic Status Colors (useful for dynamic status messages/callouts)
    'bg-status-info-bg', 'text-status-info-text', 'border-status-info',
    'dark:bg-status-info-bg-dark', 'dark:text-status-info-text-dark', 'dark:border-status-info-border-dark',
    'bg-status-success-bg', 'text-status-success-text', 'border-status-success',
    'dark:bg-status-success-bg-dark', 'dark:text-status-success-text-dark', 'dark:border-status-success-border-dark',
    'bg-status-warning-bg', 'text-status-warning-text', 'border-status-warning',
    'dark:bg-status-warning-bg-dark', 'dark:text-status-warning-text-dark', 'dark:border-status-warning-border-dark',
    'bg-status-danger-bg', 'text-status-danger-text', 'border-status-danger',
    'dark:bg-status-danger-bg-dark', 'dark:text-status-danger-text-dark', 'dark:border-status-danger-border-dark',
    'bg-test-direct-danger', // <<< NEW SAFELIST ENTRY

    // === NEW: Stat Card Palette Color Patterns ===
    {
      pattern: /bg-stat-card-bg-(teal|indigo|amber|neutral)/,
      variants: ['dark'],
    },
    {
      pattern: /text-stat-card-text-(teal|indigo|amber|neutral)/,
      variants: ['dark'],
    },
    // Ensure semantic stat card classes are also safelisted if not covered by simpler patterns above
    // (The status-*-bg/text patterns above might cover these, but explicit is safer)
    'bg-stat-card-danger-bg', 'text-stat-card-danger-text-light', 'dark:text-stat-card-danger-text-dark',
    'bg-stat-card-warning-bg', 'text-stat-card-warning-text-light', 'dark:text-stat-card-warning-text-dark',
    'bg-stat-card-success-bg', 'text-stat-card-success-text-light', 'dark:text-stat-card-success-text-dark',
    'bg-stat-card-info-bg', 'text-stat-card-info-text-light', 'dark:text-stat-card-info-text-dark',
    // ============================================

    // NEW: Charting Color Patterns
    {
      pattern: /bg-chart-(blue|green|orange|red|teal|gray)-(1|2)/,
      variants: ['dark'],
    },
    {
      pattern: /border-chart-(blue|green|orange|red|teal|gray)-border/,
      variants: ['dark'],
    },
    {
      pattern: /text-chart-(blue|green|orange|red|teal|gray)-(1|2)/,
      variants: ['dark'], // This will safelist text-chart-... AND dark:text-chart-...
    },
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
      fontSize: {
        'xxxs': '0.5rem', // 8px
        'xxs': '0.625rem', // 10px
        'xs': '0.75rem',      // 12px - Override default to remove bundled line-height
        'tiny': '0.6875rem', // 11px
        // To ensure leading-* utilities work for all standard text sizes,
        // you might want to override other defaults here as well, for example:
        // 'sm': '0.875rem',
        // 'base': '1rem',
        // 'lg': '1.125rem',
        // 'xl': '1.25rem',
      },
      colors: {
        // --- Base Brand Colors ---
        'brand': {
          'pink': {
            400: 'var(--color-brand-pink-400)',
            600: 'var(--color-brand-pink-600)'
          },
          'red': {
            500: 'var(--color-brand-red-500)',
            700: 'var(--color-brand-red-700)'
          },
          'orange': {
            400: 'var(--color-brand-orange-400)',
            600: 'var(--color-brand-orange-600)'
          },
          'green': {
            400: 'var(--color-brand-green-400)',
            600: 'var(--color-brand-green-600)'
          },
          'blue': {
            400: 'var(--color-brand-blue-400)',
            600: 'var(--color-brand-blue-600)'
          },
          'yellow': 'var(--color-brand-yellow)',
          'lime':   'var(--color-brand-lime)',
          // 'green':  'var(--color-brand-green)',
          'teal':   'var(--color-brand-teal)',
          // 'blue':   'var(--color-brand-blue)', // Original brand-blue
          // 'blue': 'rgb(var(--rgb-brand-blue) / <alpha-value>)', // New for opacity          'black':  'var(--color-brand-black)',
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
          'text-contrast': 'var(--color-report-text-contrast)',
          'accent': 'var(--color-report-accent)',
          'accent-dark': 'var(--color-report-accent-dark)',
          'light-accent': 'var(--color-report-light-accent)',
          'light-accent-dark': 'var(--color-report-light-accent-dark)',
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
        // NEW Charting Colors
        'chart-blue-1': 'var(--color-chart-blue-1)',
        'chart-blue-2': 'var(--color-chart-blue-2)',
        'chart-blue-border': 'var(--color-chart-blue-border)',
        'chart-blue-1-dark': 'var(--color-chart-blue-1-dark)',
        'chart-blue-2-dark': 'var(--color-chart-blue-2-dark)',
        'chart-blue-border-dark': 'var(--color-chart-blue-border-dark)',

        'chart-green-1': 'var(--color-chart-green-1)',
        'chart-green-2': 'var(--color-chart-green-2)',
        'chart-green-border': 'var(--color-chart-green-border)',
        'chart-green-1-dark': 'var(--color-chart-green-1-dark)',
        'chart-green-2-dark': 'var(--color-chart-green-2-dark)',
        'chart-green-border-dark': 'var(--color-chart-green-border-dark)',

        'chart-orange-1': 'var(--color-chart-orange-1)',
        'chart-orange-2': 'var(--color-chart-orange-2)',
        'chart-orange-border': 'var(--color-chart-orange-border)',
        'chart-orange-1-dark': 'var(--color-chart-orange-1-dark)',
        'chart-orange-2-dark': 'var(--color-chart-orange-2-dark)',
        'chart-orange-border-dark': 'var(--color-chart-orange-border-dark)',

        'chart-red-1': 'var(--color-chart-red-1)',
        'chart-red-2': 'var(--color-chart-red-2)',
        'chart-red-border': 'var(--color-chart-red-border)',
        'chart-red-1-dark': 'var(--color-chart-red-1-dark)',
        'chart-red-2-dark': 'var(--color-chart-red-2-dark)',
        'chart-red-border-dark': 'var(--color-chart-red-border-dark)',

        'chart-teal-1': 'var(--color-chart-teal-1)',
        'chart-teal-2': 'var(--color-chart-teal-2)',
        'chart-teal-border': 'var(--color-chart-teal-border)',
        'chart-teal-1-dark': 'var(--color-chart-teal-1-dark)',
        'chart-teal-2-dark': 'var(--color-chart-teal-2-dark)',
        'chart-teal-border-dark': 'var(--color-chart-teal-border-dark)',

        'chart-gray-1': 'var(--color-chart-gray-1)',
        'chart-gray-2': 'var(--color-chart-gray-2)',
        'chart-gray-border': 'var(--color-chart-gray-border)',
        'chart-gray-1-dark': 'var(--color-chart-gray-1-dark)',
        'chart-gray-2-dark': 'var(--color-chart-gray-2-dark)',
        'chart-gray-border-dark': 'var(--color-chart-gray-border-dark)',

        // Map brand colors
        'brand-primary': 'var(--color-brand-primary)',
        'brand-secondary': 'var(--color-brand-secondary)',
        'brand-accent': 'var(--color-brand-accent)',
        'brand-neutral': 'var(--color-brand-neutral)',

        // Map status colors from company-base.css for semantic use
        'status-danger-bg': 'var(--color-status-danger-bg)',
        'status-danger-text-light': 'var(--color-status-danger-text-light)',
        'status-danger-text-dark': 'var(--color-status-danger-text-dark)',
        'status-warning-bg': 'var(--color-status-warning-bg)',
        'status-warning-text-light': 'var(--color-status-warning-text-light)',
        'status-warning-text-dark': 'var(--color-status-warning-text-dark)',
        'status-success-bg': 'var(--color-status-success-bg)',
        'status-success-text-light': 'var(--color-status-success-text-light)',
        'status-success-text-dark': 'var(--color-status-success-text-dark)',
        'status-info-bg': 'var(--color-status-info-bg)',
        'status-info-text-light': 'var(--color-status-info-text-light)',
        'status-info-text-dark': 'var(--color-status-info-text-dark)',

        // Map Charting colors from report-styles.css
        'chart-blue-1': 'var(--color-chart-blue-1)',
        'chart-blue-2': 'var(--color-chart-blue-2)',
        'chart-blue-border': 'var(--color-chart-blue-border)',
        'chart-blue-1-dark': 'var(--color-chart-blue-1-dark)',
        'chart-blue-2-dark': 'var(--color-chart-blue-2-dark)',
        'chart-blue-border-dark': 'var(--color-chart-blue-border-dark)',

        'chart-green-1': 'var(--color-chart-green-1)',
        'chart-green-2': 'var(--color-chart-green-2)',
        'chart-green-border': 'var(--color-chart-green-border)',
        'chart-green-1-dark': 'var(--color-chart-green-1-dark)',
        'chart-green-2-dark': 'var(--color-chart-green-2-dark)',
        'chart-green-border-dark': 'var(--color-chart-green-border-dark)',

        'chart-orange-1': 'var(--color-chart-orange-1)',
        'chart-orange-2': 'var(--color-chart-orange-2)',
        'chart-orange-border': 'var(--color-chart-orange-border)',
        'chart-orange-1-dark': 'var(--color-chart-orange-1-dark)',
        'chart-orange-2-dark': 'var(--color-chart-orange-2-dark)',
        'chart-orange-border-dark': 'var(--color-chart-orange-border-dark)',

        'chart-red-1': 'var(--color-chart-red-1)',
        'chart-red-2': 'var(--color-chart-red-2)',
        'chart-red-border': 'var(--color-chart-red-border)',
        'chart-red-1-dark': 'var(--color-chart-red-1-dark)',
        'chart-red-2-dark': 'var(--color-chart-red-2-dark)',
        'chart-red-border-dark': 'var(--color-chart-red-border-dark)',

        'chart-teal-1': 'var(--color-chart-teal-1)',
        'chart-teal-2': 'var(--color-chart-teal-2)',
        'chart-teal-border': 'var(--color-chart-teal-border)',
        'chart-teal-1-dark': 'var(--color-chart-teal-1-dark)',
        'chart-teal-2-dark': 'var(--color-chart-teal-2-dark)',
        'chart-teal-border-dark': 'var(--color-chart-teal-border-dark)',

        'chart-gray-1': 'var(--color-chart-gray-1)',
        'chart-gray-2': 'var(--color-chart-gray-2)',
        'chart-gray-border': 'var(--color-chart-gray-border)',
        'chart-gray-1-dark': 'var(--color-chart-gray-1-dark)',
        'chart-gray-2-dark': 'var(--color-chart-gray-2-dark)',
        'chart-gray-border-dark': 'var(--color-chart-gray-border-dark)',

        // Define semantic colors specifically for Stat Cards using status variables
        'stat-card-danger-bg': 'var(--color-status-danger-bg)',
        'stat-card-danger-text-light': 'var(--color-status-danger-text-light)',
        'stat-card-danger-text-dark': 'var(--color-status-danger-text-dark)',
        'stat-card-warning-bg': 'var(--color-status-warning-bg)',
        'stat-card-warning-text-light': 'var(--color-status-warning-text-light)',
        'stat-card-warning-text-dark': 'var(--color-status-warning-text-dark)',
        'stat-card-success-bg': 'var(--color-status-success-bg)',
        'stat-card-success-text-light': 'var(--color-status-success-text-light)',
        'stat-card-success-text-dark': 'var(--color-status-success-text-dark)',
        'stat-card-info-bg': 'var(--color-status-info-bg)',
        'stat-card-info-text-light': 'var(--color-status-info-text-light)',
        'stat-card-info-text-dark': 'var(--color-status-info-text-dark)',
        // Add neutral/other semantic mappings if needed, e.g., using brand-neutral or grays
        'stat-card-neutral-bg': 'var(--color-brand-neutral-light)', // Example: using light neutral
        'stat-card-neutral-text-light': 'var(--color-brand-neutral-dark)', // Example: using dark neutral
        'stat-card-neutral-text-dark': 'var(--color-brand-neutral-light)', // Example: using light neutral

        // === NEW: Map Stat Card Palette Colors ===
        'stat-card-bg-teal':         'var(--color-stat-card-teal-bg)',
        'stat-card-bg-teal-dark':    'var(--color-stat-card-teal-bg-dark)',
        'stat-card-text-teal':       'var(--color-stat-card-teal-text)',
        'stat-card-text-teal-dark':  'var(--color-stat-card-teal-text-dark)',

        'stat-card-bg-indigo':         'var(--color-stat-card-indigo-bg)',
        'stat-card-bg-indigo-dark':    'var(--color-stat-card-indigo-bg-dark)',
        'stat-card-text-indigo':       'var(--color-stat-card-indigo-text)',
        'stat-card-text-indigo-dark':  'var(--color-stat-card-indigo-text-dark)',

        'stat-card-bg-amber':         'var(--color-stat-card-amber-bg)',
        'stat-card-bg-amber-dark':    'var(--color-stat-card-amber-bg-dark)',
        'stat-card-text-amber':       'var(--color-stat-card-amber-text)',
        'stat-card-text-amber-dark':  'var(--color-stat-card-amber-text-dark)',

        'stat-card-bg-neutral':         'var(--color-stat-card-neutral-bg)',
        'stat-card-bg-neutral-dark':    'var(--color-stat-card-neutral-bg-dark)',
        'stat-card-text-neutral':       'var(--color-stat-card-neutral-text)',
        'stat-card-text-neutral-dark':  'var(--color-stat-card-neutral-text-dark)',
        // ========================================
        'test-direct-danger': 'var(--color-status-danger-bg)', // <<< NEW TEST COLOR DEFINITION
        // 'brand-blue-dark': 'var(--color-status-info-bg-dark)', // Original source for the color value
        'brand-blue-dark': 'rgb(var(--rgb-brand-blue-dark) / <alpha-value>)', // New for opacity
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
