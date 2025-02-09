# Project Description: Jinja2 Static Report Template

## Overview

This project involves developing a static report generation system using Python and Jinja2. The generated reports will incorporate modern HTML, CSS, and JavaScript best practices, drawing inspiration from shadcn and Tailwind CSS for functionality and styling—without relying on frameworks such as React or Node.js.

## Technical Specifications

### 1. Core Dependencies
- Python 3.11+
- Jinja2 3.0+
- TailwindCSS 3.4+
- Alpine.js 3.x (for interactive components)
- WeasyPrint (for PDF generation)

### 2. Template Architecture
- **Base Template Structure:**
  ```jinja2
  base_report.html.j2      # Core layout and includes
  ├── components/          # Reusable UI components
  │   ├── cards.html.j2
  │   ├── tables.html.j2
  │   └── popups.html.j2
  ├── layouts/            # Page layout variations
  │   └── default.html.j2
  └── partials/          # Common sections
      ├── header.html.j2
      └── footer.html.j2
  ```

### 3. Data Structure
```python
class ReportContext(BaseModel):
    """Report data structure with validation."""
    title: str
    generated_at: datetime
    kb_articles: List[KBArticle]
    cve_data: Dict[str, CVEInfo]
    metadata: Dict[str, Any]

class KBArticle(BaseModel):
    """KB Article structure."""
    id: str
    title: str
    description: str
    severity: str
    cves: List[str]
```

## Objectives

- **Static HTML Generation:** Create a Jinja2 template to produce static HTML files for deployment on a web server.
- **Modern Web Standards:** Ensure the template adheres to the latest standards in web development, including current best practices for HTML, CSS, and JavaScript.
- **Scalability & Maintainability:** Centralize core styling and theme information to maintain consistency and enable easy updates.
- **Responsiveness:** Design the layout so that typography and UI elements adjust fluidly across various screen sizes.

## Data Handling Strategy

- **Preprocessing:** Data extraction and preprocessing will be managed separately in Python.
- **Data Structure:**
  - **KB Articles DataFrame:** Contains the primary records (KB Articles).
  - **CVEs DataFrame:** Serves as a lookup table for CVE information linked to each KB Article. This structure facilitates the ranking and grouping of CVEs based on specific properties.

## UI/UX Components

- **CVE Popup Component:**
  - Trigger: Data attribute based click handlers
  - Animation: CSS transitions (300ms ease-in-out)
  - Backdrop: rgba(0, 0, 0, 0.5) with blur
  - Z-index management for proper layering

### 1. Core Components
- **CVE Popup Component:**
  - Each record will include a button that, when clicked, triggers a popup displaying all related CVEs.
  - **Implementation Options:**
    - **Dynamic Data Structure:** Store all CVE data in a JavaScript-accessible data structure that the popup script can reference.
    - **Pre-populated Hidden Divs:** Alternatively, pre-build hidden or off-canvas divs containing the necessary CVE data and use JavaScript to transfer the relevant data into the popup upon activation.
  - The popup should feature smooth opening and closing transitions, employing modern JavaScript techniques.

## Styling & Theming

- **Inspiration:** The visual design and interactive elements should take cues from shadcn and Tailwind CSS, focusing on clean, modular styling without using React or Node.js.
- **Centralized Styles:** Define all core style information in a single location (or file) to act as the theme source, ensuring consistent application of styles across the entire report.
- **Responsiveness:** Utilize responsive design techniques to guarantee that elements stack properly and adjust in size based on the device or viewport dimensions.

### 2. Styling & Theme Configuration
```yaml
theme:
  colors:
    primary: "#0284c7"
    secondary: "#7c3aed"
    background: "#ffffff"
    text: "#1f2937"
  typography:
    base_size: "16px"
    scale: 1.25
    font_family: "Inter var, system-ui"
  spacing:
    base_unit: "0.25rem"
    container_padding: "2rem"
  breakpoints:
    sm: "640px"
    md: "768px"
    lg: "1024px"
```

## Development Guidelines

1. **Jinja2 Template Construction:**
   - Develop a template that dynamically populates static HTML based on the preprocessed data.
   - Ensure template syntax is clean and modular, facilitating easy updates and maintenance.

2. **Integration of Modern Web Practices:**
   - Write semantic HTML.
   - Leverage CSS for layout and styling, ensuring the separation of concerns.
   - Implement JavaScript for interactive components (e.g., the CVE popup) using unobtrusive and progressive enhancement techniques.

3. **Responsive Design:**
   - Use CSS media queries and flexible layout techniques (e.g., flexbox or grid) to create a fluid, responsive interface.

4. **Mockup Guidance:**
   - Utilize the annotated mockup images as a visual and functional guide.
   - **Important:** The red annotations on the mockups are for guidance only and should not be included in the final rendered template.

### 1. Code Organization
- Follow PEP-8 and project style guidelines
- Type hints required for all Python functions
- Document all functions with docstrings
- Separate concerns: template logic vs. data processing

### 2. Performance Requirements
- First Contentful Paint < 1.5s
- Largest Contentful Paint < 2.5s
- Total Blocking Time < 300ms
- Cumulative Layout Shift < 0.1

### 3. Accessibility Standards
- WCAG 2.1 Level AA compliance
- Semantic HTML structure
- ARIA labels for interactive elements
- Keyboard navigation support
- Color contrast ratio ≥ 4.5:1

### 4. Security Measures
- Content Security Policy headers
- XSS protection via input sanitization
- CSRF protection for any forms
- Secure content embedding

## Folder Structure
```
microsoft_cve_rag/
└── application/
    └── data/
        ├── templates/
        │   └── weekly_kb_report/
        │       ├── base.html.j2
        │       ├── components/
        │       └── layouts/
        └── reports/
            └── weekly_kb_report/
                ├── html/
                ├── md/
                ├── pdf/
                ├── assets/
                └── media/
```

## Implementation Workflow

1. **Setup Phase**
   - Initialize project structure
   - Configure TailwindCSS
   - Set up template inheritance

2. **Development Phase**
   - Implement base templates
   - Build component library
   - Create data models
   - Develop rendering pipeline

3. **Testing & Validation**
   - Unit tests for Python code
   - Visual regression testing
   - Accessibility testing
   - Performance benchmarking

## Final Notes

This document outlines the technical specifications and implementation details for developing a modern, responsive static report template. The approach emphasizes maintainability, performance, and adherence to web standards while providing a rich user experience.

For implementation questions or clarifications, refer to the technical specifications sections above. All code should follow the project's established patterns and guidelines.
