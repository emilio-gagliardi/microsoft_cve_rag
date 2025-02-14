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
   - add data extraction and preprocessing steps
   - integrate data extraction with template rendering
   - build a fastapi endpoint to serve the rendered report that takes two datetime parameters: start and end.

3. **Testing & Validation**
   - Unit tests for Python code
   - Visual regression testing
   - Accessibility testing
   - Performance benchmarking

## Final Notes

This document outlines the technical specifications and implementation details for developing a modern, responsive static report template. The approach emphasizes maintainability, performance, and adherence to web standards while providing a rich user experience.

For implementation questions or clarifications, refer to the technical specifications sections above. All code should follow the project's established patterns and guidelines.


## Mongo queries and data extraction
```python
import pandas as pd
from pymongo import MongoClient
from datetime import datetime

def choose_score(doc):
    """
    Selects the higher score between CNA and NIST scores.
    If both exist and are equal, defaults to the CNA score.
    Falls back to ADP if neither CNA nor NIST is available.
    """
    cna_num = doc.get("cna_base_score_num")
    nist_num = doc.get("nist_base_score_num")
    adp_num = doc.get("adp_base_score_num")

    if cna_num is not None and nist_num is not None:
        if cna_num > nist_num:
            return {"score_num": cna_num, "score_rating": doc.get("cna_base_score_rating")}
        elif nist_num > cna_num:
            return {"score_num": nist_num, "score_rating": doc.get("nist_base_score_rating")}
        else:
            # Equal scores: default to CNA
            return {"score_num": cna_num, "score_rating": doc.get("cna_base_score_rating")}
    elif cna_num is not None:
        return {"score_num": cna_num, "score_rating": doc.get("cna_base_score_rating")}
    elif nist_num is not None:
        return {"score_num": nist_num, "score_rating": doc.get("nist_base_score_rating")}
    elif adp_num is not None:
        return {"score_num": adp_num, "score_rating": doc.get("adp_base_score_rating")}
    else:
        return {"score_num": None, "score_rating": None}

# Connect to MongoDB
client = MongoClient("mongodb://your_mongo_uri")
db = client["your_db"]

# Define the date range for KB articles (adjust as needed)
start_date = datetime(2023, 12, 1)
end_date = datetime(2023, 12, 31)

# --- Query 1: Fetch KB articles ---
kb_cursor = db.microsoft_kb_articles.find({
    "published": {"$gte": start_date, "$lte": end_date}
})
kb_articles = list(kb_cursor)
kb_df = pd.DataFrame(kb_articles)

# --- Extract unique CVE IDs from KB articles ---
unique_cve_ids = {
    cve
    for record in kb_articles
    if record.get("cve_ids")
    for cve in record["cve_ids"]
}

# --- Query 2: Fetch CVE details with projection ---
cve_projection = {
    "metadata.id": 1,
    "metadata.revision": 1,
    "metadata.published": 1,
    "metadata.source": 1,
    "metadata.post_id": 1,
    "metadata.cve_category": 1,
    "metadata.adp_base_score_num": 1,
    "metadata.adp_base_score_rating": 1,
    "metadata.cna_base_score_num": 1,
    "metadata.cna_base_score_rating": 1,
    "metadata.nist_base_score_num": 1,
    "metadata.nist_base_score_rating": 1
}

cve_cursor = db.docstore.find(
    {"cve_mentions": {"$in": list(unique_cve_ids)}},
    cve_projection
)
cve_details = list(cve_cursor)

# --- Flatten the metadata and choose the appropriate score pair ---
for doc in cve_details:
    # Remove and retrieve the nested metadata dictionary.
    metadata = doc.pop("metadata", {})
    # Merge metadata keys into the root-level of the document.
    doc.update(metadata)
    # Choose the best score pair based on CNA vs. NIST (or fallback to ADP).
    score = choose_score(doc)
    doc["score_num"] = score["score_num"]
    doc["score_rating"] = score["score_rating"]

# --- Build a CVE lookup dictionary keyed by the unique identifier (post_id) ---
cve_lookup = {doc.get("post_id"): doc for doc in cve_details if doc.get("post_id")}

# --- Integrate CVE details into KB articles ---
def attach_cve_details(record):
    """
    For each KB record, attach a list of detailed CVE documents
    by looking up each CVE ID (from record['cve_ids']) in the CVE lookup.
    """
    return [cve_lookup[cve] for cve in record.get("cve_ids", []) if cve in cve_lookup]

kb_df["cve_details"] = kb_df.apply(attach_cve_details, axis=1)
```
