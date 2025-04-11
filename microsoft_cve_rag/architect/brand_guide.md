# 1. Color Palette Selection & Recommendations

From the provided palette, here are 5 versatile choices and recommended variations for UI use:

Brand Blue (#22A5DD): Excellent primary action color (buttons, links). Good contrast.

--brand-blue-dark: #158AB8 (For hover/active states)

--brand-blue-light: #E9F6FD (For light backgrounds, tags)

--brand-blue-text: #145C7C (Darker version for text on light blue backgrounds)

Brand Green (#00A33E): Good secondary accent or success indicator.

--brand-green-dark: #007A2F (For hover/active states)

--brand-green-light: #E6F6EE (For light backgrounds, tags)

--brand-green-text: #004D26 (Darker version for text on light green backgrounds)

Brand Teal (#67C2C0): Softer accent, useful for secondary info or subtle backgrounds.

--brand-teal-dark: #4CA7A3

--brand-teal-light: #EFF9F9

--brand-teal-text: #307B78

Brand Orange (#EB8B06): Strong warm accent, good for warnings or highlighting specific elements (use sparingly).

--brand-orange-dark: #C57104

--brand-orange-light: #FEF4E6

--brand-orange-text: #8A4F03

Neutral Grays (Refined): Keeping neutrals is essential for readability. Let's refine the existing ones slightly for better harmony.

--text-color: #1F2937 (Dark Gray - keep)

--secondary-text-color: #4B5563 (Medium Gray - keep)

--muted-text-color: #6B7280 (Lighter Medium Gray - added)

--divider-color: #E5E7EB (Light Gray Border - keep)

--card-background: #ffffff (White - keep)

--content-background: #F8F9FA (Very Light Gray - background for modal content, slight off-white)

--body-background: #F9FAFB (Slightly different very light gray for main body)

--dark-footer-background: #212529 (Very Dark Gray, less harsh than pure black)

--dark-footer-text: #CED4DA (Light Gray for footer text)

--dark-footer-link: #90CAF9 (Light Blue for footer links)

# 2. There is a CSS stylesheet with these colors already defined. When building UI elements, use the CSS variables instead of the hex colors. Or if you need to create new colors, use the CSS variables as a reference.

It is located at: C:\Users\emili\PycharmProjects\microsoft_cve_rag\microsoft_cve_rag\application\data\templates\weekly_kb_report\css\stylesheet.css

# 3. PortalFuse Report Brand Guide (Material Design Adaptation)

This guide outlines how to apply PortalFuse branding and Material Design principles consistently within reports.

## Color Palette

Color is fundamental to branding and user experience. We use distinct palettes for overall branding (PortalFuse) and report-specific content.

*   **PortalFuse Primary:** `var(--portalfuse-primary-color)`
    *   **Usage:** Main brand identity, primary actions (e.g., Subscribe button), report headers/footers, key section titles. Creates brand recognition.
    *   **Example (Button):**
        ```html
        <button style="background-color: var(--portalfuse-primary-color); color: var(--portalfuse-text-on-primary); padding: 8px 16px; border: none; border-radius: 8px; cursor: pointer;">
            PortalFuse Action
        </button>
        ```
*   **Report Primary / Secondary Action:** `var(--report-primary-color)`
    *   **Usage:** Key actions *within* the report content (e.g., "View CVEs" button), prominent links, data visualization accents. Distinguishes report actions from global brand actions.
    *   **Example (Button):**
        ```html
        <button style="background-color: var(--report-primary-color); color: white; padding: 8px 16px; border: none; border-radius: 8px; cursor: pointer;">
            Report Action (View CVEs)
        </button>
        ```
*   **Surface / Card Background:** `var(--card-background)`
    *   **Usage:** Primary background for content containers like cards. Provides a clean base.
*   **Page Background:** `var(--background-color)`
    *   **Usage:** Overall page background. Creates subtle contrast with white cards.
*   **Text Colors:**
    *   Primary Text: `var(--text-color)` Use for body copy, main titles.
    *   Secondary Text: `var(--text-secondary)` Use for meta-data (dates), labels, less important descriptions.
    *   Muted Text: `var(--text-muted)` Often interchangeable with secondary, used for placeholders, disabled text hints.
    *   **Example:**
        ```html
        <div>
            <h3 style="color: var(--text-color); margin-bottom: 4px;">Article Title</h3>
            <p style="color: var(--text-secondary); font-size: 0.9rem;">Published: October 26, 2023</p>
            <p style="color: var(--text-color);">This is the main body text content.</p>
        </div>
        ```
*   **Borders / Dividers:** `var(--divider-color)`
    *   **Usage:** Subtle separation within cards (headers/footers), between sections, or as component borders (inputs, cards).
    *   **Example (Card Border):**
        ```html
        <div style="background-color: white; border: 1px solid var(--divider-color); padding: 16px; border-radius: 8px;">
            Card content with a border.
        </div>
        ```
*   **Accent Colors (Use Sparingly):**
    *   PortalFuse Secondary: `var(--portalfuse-secondary-color)` Use for subtle backgrounds (hover states, alerts, badges).
    *   Report Accent: `var(--report-accent-color)` Reserved for high-emphasis highlights if needed, but generally avoid overuse.

## Elevation & Surfaces

Material Design uses elevation to convey hierarchy and focus. We achieve this primarily through `box-shadow`.

*   **Standard Elevation (Cards):** `var(--card-shadow)` (`0 1px 3px rgba(0, 0, 0, 0.1)`)
    *   **Usage:** Default shadow for cards and distinct surface elements.
    *   **Example:**
        ```html
        <div style="background-color: white; border-radius: 8px; padding: 16px; box-shadow: var(--card-shadow);">
            Standard Card Elevation
        </div>
        ```
*   **Raised Elevation (Modals, Hover):** `var(--card-shadow-raised)` (`0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1)`) or modal-specific shadows.
    *   **Usage:** Modals, dropdowns, or optional hover states on interactive cards to lift them visually.

## Typography

Clear and consistent typography is crucial for readability and a professional feel.

*   **Font Families:**
    *   UI / Body: `var(--font-family)` ('Inter', sans-serif)
    *   Monospace / Code: `var(--font-family-mono)` ('Roboto Mono', monospace)
*   **Hierarchy:** Use size and weight to differentiate elements.
    *   `H1` (Report Title): Large, prominent.
    *   `H2` (Section Title): Uses brand color, clear separation.
    *   `H3` (Card/Article Title): Standard heading weight/size.
    *   Body Text (`p`): Base size, good line-height (1.5-1.6).
    *   Meta/Caption Text (e.g., dates, labels): Slightly smaller, often uses secondary text color.
    *   **Example:**
        ```html
        <h2 style="font-size: 1.75rem; font-weight: 600; color: var(--portalfuse-primary-color); border-bottom: 2px solid var(--portalfuse-secondary-color); padding-bottom: 8px; margin-bottom: 24px;">Section Title</h2>
        <div style="background-color: white; border-radius: 8px; padding: 16px; box-shadow: var(--card-shadow);">
            <h3 style="font-size: 1.25rem; font-weight: 600; margin-bottom: 4px;">Article Title</h3>
            <p style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 16px;">Published Date / Meta Info</p>
            <p style="font-size: 1rem; line-height: 1.6;">This is the main paragraph text for content readability.</p>
            <code>Code example: var(--example)</code>
        </div>
        ```

## Component States

Interactive elements must provide clear visual feedback for different states.

*   **Hover:** Indicates an element is interactive. Often involves a subtle background change (e.g., light primary/secondary color) or darkening/lightening the main color.
*   **Focus:** Indicates an element is selected via keyboard navigation. Typically shown with an outline (`outline: 2px solid ...`) or a distinct focus ring (`box-shadow`). Use `:focus-visible` for modern browsers.
*   **Active:** Indicates an element is being pressed/activated. Often involves a slightly darker background or an inset shadow.
*   **Disabled:** Indicates an element is not interactive. Usually achieved with reduced opacity (`opacity: 0.6`) and a `cursor: not-allowed` style.
*   **Example (Primary Button States - CSS):**
    ```css
    .my-primary-button {
        background-color: var(--portalfuse-primary-color);
        color: white;
        /* ... other base styles ... */
        transition: background-color 0.2s, opacity 0.2s, box-shadow 0.2s;
    }

    .my-primary-button:hover:not(:disabled) {
        background-color: #512DA8; /* Darker purple */
        box-shadow: var(--button-shadow-hover);
    }

    .my-primary-button:focus-visible {
        outline: 2px solid var(--portalfuse-primary-color);
        outline-offset: 2px;
    }

    .my-primary-button:active:not(:disabled) {
         background-color: #4527A0; /* Even darker purple */
         box-shadow: inset 0 1px 2px rgba(0,0,0,0.1);
    }

    .my-primary-button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
        box-shadow: none;
    }
    ```
    ```html
    <button class="my-primary-button">Interact With Me</button>
    <button class="my-primary-button" disabled>I Am Disabled</button>
    ```

By adhering to these guidelines, we can ensure PortalFuse reports maintain a consistent, professional, and user-friendly appearance based on Material Design principles.
