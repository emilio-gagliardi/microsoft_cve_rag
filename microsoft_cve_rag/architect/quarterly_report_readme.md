# Quarterly CVE Report Generator: System Documentation

## 1. Overview

This document details the architecture, technologies, features, and workflow of the Quarterly CVE Report Generator system.

**Purpose:** To automatically generate a sophisticated, branded, interactive, and visually engaging web-based report analyzing Microsoft CVE (Common Vulnerabilities and Exposures) trends over a specified period (typically quarterly).

**Output:** The primary output is a self-contained directory containing:
    *   An interactive **HTML report** (`index.html`) with text, Plotly charts, LLM-generated insights, and custom styling.
    *   A **Markdown version** (`report.md`) of the report content.
    *   Supporting **CSS** (`output.css`) generated via Tailwind CSS.
    *   Supporting **JavaScript** (`main.js`) for chart loading and interactivity.
    *   **JSON data files** (`*.json`) for each Plotly chart, decoupling data from presentation.

**Workflow:** The generation process is triggered via a FastAPI API endpoint. It fetches CVE data (or uses synthetic data for development), processes it, generates text and visualizations using Plotly and an LLM, renders HTML and Markdown using Jinja2 templates, styles the HTML using Tailwind CSS, and saves all assets locally to a structured directory. Uploading these assets to hosting (SFTP, Azure Storage) is handled externally by separate services that consume the output of this generator.

## 2. Core Technologies

*   **Backend Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Python) - Used for creating the API endpoints that trigger report generation. Chosen for its speed, async capabilities, automatic documentation, and Pydantic integration.
*   **Data Validation:** [Pydantic](https://docs.pydantic.dev/) (Python) - Used extensively for defining data models for API requests (`QuarterlyReportRequest`), configuration (`ReportConfig`), internal data structures (`ReportContext`), and the final output manifest (`GeneratedReportAssets`). Ensures data integrity and provides clear data contracts.
*   **Data Manipulation:** [Pandas](https://pandas.pydata.org/) (Python) - The primary tool for cleaning, transforming, aggregating, and analyzing the CVE data fetched from the source or generated synthetically.
*   **Charting Library:** [Plotly (Python Library)](https://plotly.com/python/) - Used to create rich, interactive, web-native charts and graphs (bars, lines, areas, potentially CDFs, etc.). Exports chart data and layout to JSON format.
*   **Templating Engine:** [Jinja2](https://jinja.palletsprojects.com/) (Python) - Used for:
    *   Rendering the final HTML report structure from templates.
    *   Generating structured LLM prompts using template inheritance.
    *   Potentially generating the Markdown report from a dedicated template (or converting HTML).
*   **Styling Framework:** [Tailwind CSS](https://tailwindcss.com/) (JavaScript/Node.js) - A utility-first CSS framework used for styling the HTML report. Requires a Node.js build process (`npm run build:css`/`watch:css`) to compile utility classes into a final CSS file based on template usage (purging) and customizations defined in `tailwind.config.js`.
*   **CSS Processing:** [PostCSS](https://postcss.org/), [Autoprefixer](https://github.com/postcss/autoprefixer) (Node.js) - Used by the Tailwind CSS build process for CSS transformations and adding vendor prefixes for browser compatibility.
*   **HTML to Markdown Conversion:** [Markdownify](https://github.com/matthewwithanm/python-markdownify) (Python) - Used to convert the generated HTML report content into a Markdown format.
*   **Frontend JavaScript:** Vanilla JavaScript (or potentially a micro-framework like Alpine.js) - Used in `main.js` primarily to dynamically load Plotly charts from their JSON data files after the HTML page loads, and potentially for handling UI interactions like the floating navigation.
*   **LLM Integration:** External Large Language Model API (e.g., OpenAI GPT, Anthropic Claude) - Accessed via a Python client (likely using `requests` or a dedicated library) to generate textual insights (callouts, summaries) based on processed data and Jinja-templated prompts.
*   **Data Source:** MongoDB (via `DocumentService`) - The primary source for raw CVE data.
*   **Development Environment:** Node.js & npm (or yarn) - Required for installing and running the Tailwind CSS build process.
*   **Language:** Python 3.x

## 3. Project Structure (Key Locations)

*(Adjust paths based on actual project)*

*   `PROJECT_ROOT/`
    *   `src/`
        *   `main.py`: FastAPI application entry point (imports routers).
        *   `config.py` / `initialize_environment_and_paths()`: Defines global paths like `REPORTS_DIR`, `APP_DIR`.
        *   `reporting/`
            *   `quarterly_deep_dive_generator.py`: Contains `QuarterlyDeepDiveReportGenerator` class and `generate_synthetic_cve_data` utility.
        *   `routes/`
            *   `report_routes.py`: Contains FastAPI APIRouter with `/generate` and `/generate-dev-synthetic` endpoints.
        *   `services/`
            *   `document_service.py`: Class for interacting with MongoDB.
            *   `llm_client.py`: (Assumed) Client for LLM API interaction.
            *   `sftp_service.py` / `azure_service.py`: (Assumed) External services for uploading generated assets.
        *   `models/`
            *   `report_models.py`: Contains Pydantic models (`ReportConfig`, `QuarterlyReportRequest`, `GeneratedReportAssets`, etc.).
    *   `templates/` (Base directory for Jinja environment)
        *   `quarterly_deep_dive/`: Templates for the HTML report.
            *   `_base.html`, `_layouts/`, `_partials/`, `_components/`, `report.html`
        *   `llm_prompts/`: Jinja templates for generating LLM prompts.
            *   `base_prompt.j2`, `volume_severity_callout.j2`, etc.
    *   `reports/` (Base output directory defined by `REPORTS_DIR`)
        *   `quarterly_deep_dive/` (Created by generator, named by `ReportConfig.report_name`)
            *   `report.md`: Generated Markdown report.
            *   `html/` (Created by generator, named by `ReportConfig.html_subdir`)
                *   `index.html`: Generated HTML report.
                *   `css/output.css`: Copied finalized Tailwind CSS.
                *   `js/main.js`: Copied frontend JavaScript.
                *   `data/`: (Created by generator, named by `ReportConfig.data_subdir`)
                    *   `volume-severity-chart.json`, ... : Exported Plotly chart data.
    *   `application/static/dist/` (Example location for *built* static assets - ADJUST AS NEEDED)
        *   `css/output.css`: The CSS file generated by `npm run build:css`.
        *   `js/main.js`: The frontend JS file.
    *   `src/css/input.css`: Source file for Tailwind directives and custom CSS.
    *   `tailwind.config.js`: Tailwind configuration file.
    *   `postcss.config.js`: PostCSS configuration file.
    *   `package.json`: Node.js project configuration and dependencies.
    *   `requirements.txt`: Python project dependencies.

### Updated April 14, 2025
microsoft_cve_rag/
├── application/
│   ├── data/
│   │   └── templates/
│   │       └── quarterly_deep_dive/
│   │           ├── _base.html
│   │           ├── _components/
│   │           ├── _layouts/
│   │           ├── _partials/
│   │           └── report.html
│   ├── static/
│   │   ├── css/
│   │   │   └── output.css  <-- ❗ FINAL BUILT CSS GOES HERE
│   │   ├── js/
│   │   │   └── main.js     <-- Your custom JS (optional)
│   │   └── images/         <-- Your images (optional)
│   └── __init__.py         <-- Your Flask/Python app files
│   └── ... (other python files)
│
├── src/                   <-- ✨ NEW FOLDER FOR SOURCE CSS
│   ├── company-base.css   <-- ✨ Store company base styles here (optional)
│   ├── report-styles.css  <-- ✨ Store report-specific styles/components here
│   └── input.css          <-- ✨ Main CSS entry point for Tailwind build
│
├── tailwind.config.js     <-- ✨ Tailwind config HERE (Project Root)
├── postcss.config.js      <-- ✨ PostCSS config HERE (Generated by init -p)
├── package.json           <-- ✨ Node.js project file (Manages dev dependencies)
├── package-lock.json      <-- ✨ Generated by npm
└── node_modules/          <-- ✨ Generated by npm install (add to .gitignore)
└── ... (other project files like README.md, .gitignore, etc.)

## 4. Key Components & Workflow

1.  **Triggering (FastAPI):**
    *   A POST request is made to `/reports/quarterly-deep-dive/generate` (for real data) or `/reports/quarterly-deep-dive/generate-dev-synthetic` (for testing).
    *   The request body includes `start_date` and `end_date` (`QuarterlyReportRequest` model). Optional config overrides can be included.
    *   FastAPI uses Dependency Injection (`Depends`) to provide instances of:
        *   `DocumentService` (for the production route).
        *   `QuarterlyDeepDiveReportGenerator` (injecting Jinja env, LLM client).
        *   (Optionally) SFTP/Azure upload services (for the production route's upload step).

2.  **Data Acquisition:**
    *   **Production Route:** Calls the injected `DocumentService` to fetch raw CVE documents from MongoDB for the specified date range. The result (list of dicts) is converted to a Pandas DataFrame.
    *   **Dev Route:** Calls the `generate_synthetic_cve_data` utility function to create a mock Pandas DataFrame.

3.  **Report Generation (`QuarterlyDeepDiveReportGenerator.generate_report`):**
    *   Receives the DataFrame, `ReportConfig`, `start_date`, `end_date`.
    *   Creates the necessary output directories based on `REPORTS_DIR` and `ReportConfig`.
    *   Calls `_prepare_input_dataframe` to validate, clean, and standardize the input DataFrame (add `published_month`, rename columns, etc.).
    *   Iterates through a predefined `chart_configs` dictionary:
        *   Calls `_prepare_chart_data` to aggregate/transform data for the specific chart.
        *   Calls `_generate_plotly_figure` to create a Plotly `Figure` object using the prepared data and applying the custom theme.
        *   If a figure is generated, calls `_export_chart_json` to save the figure's data/layout to a `.json` file in the `html/data/` output directory.
        *   Calls `_get_llm_insight` (see step 4).
    *   Calls `_generate_appendix_tables` to create HTML string(s) for appendix tables.
    *   Constructs the `ReportContext` Pydantic model containing all data needed for templates (config, stats, chart info, insights, table HTML).
    *   Renders the main HTML template (`report.html`) using Jinja2 and the `ReportContext`, saving the output to `html/index.html`.
    *   Converts the generated HTML content to Markdown using `markdownify` and saves it to `report.md` in the report's base directory.
    *   Calls `_place_static_assets` to copy the built `output.css` and `main.js` from their source location (e.g., `application/static/dist/`) into the `html/css/` and `html/js/` output directories.
    *   Returns a `GeneratedReportAssets` object containing `pathlib.Path` objects for all created files.

4.  **LLM Integration (`_get_llm_insight`):**
    *   Uses Jinja2 to render a specific prompt template from `templates/llm_prompts/` (e.g., `volume_severity_callout.j2`).
    *   The prompt template inherits from `base_prompt.j2` for common instructions.
    *   Context passed to the prompt template includes report metadata (title, dates) and formatted data relevant to the insight (`chart_data_str`).
    *   Calls the injected LLM client with the rendered prompt.
    *   Returns the cleaned text response from the LLM.

5.  **Styling (Tailwind CSS Build):**
    *   This is an *external* process run via `npm run watch:css` (development) or `npm run build:css` (production).
    *   Tailwind scans template files specified in `tailwind.config.js` (`content` array).
    *   It processes `src/css/input.css`, applying utilities, base styles, components, and processing `@tailwind` directives.
    *   It uses PostCSS and Autoprefixer.
    *   It purges unused styles based on template scanning.
    *   The final, optimized CSS is written to `application/static/dist/css/output.css` (or configured path).
    *   The `_place_static_assets` step in the Python generator *copies* this pre-built file into the specific report's output directory.

6.  **Frontend Interaction (`main.js`):**
    *   When `index.html` is loaded in the browser, `main.js` executes.
    *   It finds all chart container divs (`<div id="..." class="plotly-chart-container" data-chart-id="...">`).
    *   For each container, it uses the `data-chart-id` to `fetch` the corresponding `data/*.json` file.
    *   It uses `Plotly.react()` or `Plotly.newPlot()` to render the interactive chart inside the container div using the fetched JSON data.
    *   It may also handle interactivity for elements like the floating navigation bar.

7.  **Output & Upload (External):**
    *   The FastAPI route receives the `GeneratedReportAssets` object from the generator.
    *   This object contains paths to all generated files (`index.html`, `report.md`, `css/output.css`, `js/main.js`, `data/*.json`).
    *   The FastAPI route can then pass this object or the list of files to separate, injected services (`sftp_service`, `azure_service`) which handle the actual upload logic to the desired hosting destination(s).

## 5. Design Decisions & Rationale

*   **Decoupled Generator:** The `QuarterlyDeepDiveReportGenerator` focuses *only* on generating the report assets locally. It does not handle data fetching or uploading. This promotes Separation of Concerns, making the generator more testable and reusable. Data fetching logic belongs in the `DocumentService`, and upload logic belongs in dedicated upload services.
*   **Plotly JSON Export:** Exporting chart data to JSON instead of embedding chart HTML directly into the main report HTML keeps the primary HTML file smaller and cleaner. It allows the frontend JS to load charts asynchronously and potentially allows for easier updates to chart data without re-rendering the entire HTML report structure.
*   **Tailwind CSS Build Process:** Using the Node.js build process (instead of the CDN or Play CDN) is necessary for:
    *   **Customization:** Defining brand colors, fonts, etc., in `tailwind.config.js`.
    *   **Purging:** Generating significantly smaller production CSS files by removing unused styles.
    *   **Plugins:** Enabling the use of Tailwind plugins.
*   **Jinja for LLM Prompts:** Storing prompts in `.j2` files with inheritance makes them highly maintainable, reusable, and separates prompt engineering from Python code. Base templates enforce consistency.
*   **Pydantic Models:** Used throughout for clear data contracts, validation, and improved developer experience (auto-completion, type checking).
*   **Synthetic Data Route:** Provides a crucial workflow for frontend development (HTML/CSS/JS/Templating) without requiring a live database connection or complex data setup, accelerating the development of the report's look and feel.
*   **HTML to Markdown Conversion:** Using `markdownify` leverages the single source of truth (the main HTML template) to generate the Markdown version, ensuring content consistency rather than maintaining two separate content templates.
*   **`_place_static_assets` Step:** Copying the built CSS/JS ensures the generated report directory is fully self-contained and doesn't rely on external paths to static assets once deployed.

## 6. Features & Capabilities

*   Generates interactive HTML reports with Plotly charts.
*   Generates a corresponding Markdown version of the report.
*   Integrates LLM-generated textual insights (callouts, summaries).
*   Utilizes a highly customizable Tailwind CSS theme reflecting brand identity.
*   Features responsive design for desktop and mobile viewing (via Tailwind classes).
*   Supports various Plotly chart types (configured in Python).
*   Decouples chart data (JSON) from presentation (HTML/JS).
*   Uses structured Jinja templates for maintainable HTML and LLM prompts.
*   Provides a development workflow using synthetic data.
*   Outputs a self-contained directory with all necessary assets, ready for deployment/upload.
*   Configuration-driven via Pydantic models.

## 7. Dependencies & Setup

*   **Python Environment (`requirements.txt`):**
    *   `fastapi`, `uvicorn` (for running the server)
    *   `pydantic`
    *   `pandas`
    *   `numpy`
    *   `plotly`
    *   `jinja2`
    *   `pymongo` or `motor` (for `DocumentService`)
    *   `requests` or LLM-specific SDK (for `llm_client`)
    *   `markdownify`
    *   `python-dotenv` (for environment variable loading)
    *   `Faker` (for synthetic data generation)
    *   (Optional) `paramiko` (for SFTP), `azure-storage-blob` (for Azure)
*   **Node.js Environment (`package.json`):**
    *   `tailwindcss`
    *   `postcss`
    *   `autoprefixer`
*   **External Services:**
    *   MongoDB Database accessible by `DocumentService`.
    *   LLM API endpoint accessible by `llm_client` (requires API key, likely via environment variable).
*   **Configuration:**
    *   Environment variables for API keys, database connection strings, etc. (loaded via `.env` files).
    *   `config.yaml` (or similar) defining `PROJECT_ROOT`.
    *   `tailwind.config.js` configured with project paths and theme.
    *   `initialize_environment_and_paths()` function correctly setting global path variables (`REPORTS_DIR`, `APP_DIR`).

## 8. Development Workflow

1.  **Install Dependencies:** Run `pip install -r requirements.txt` and `npm install`.
2.  **Configure Environment:** Ensure `.env` files and `config.yaml` are set up correctly.
3.  **Start Tailwind Watcher:** Open a terminal in `PROJECT_ROOT` and run `npm run watch:css`. Leave this running.
4.  **Start FastAPI Server:** Open a *second* terminal in `PROJECT_ROOT` and run `uvicorn src.main:app --reload` (or your specific command).
5.  **Develop Templates/Styles:** Modify files in `templates/quarterly_deep_dive/` and `src/css/input.css`. Save changes. The Tailwind watcher will automatically rebuild the CSS to `application/static/dist/css/output.css`.
6.  **Test Generation:** Send a POST request (e.g., using Postman) to the `/reports/quarterly-deep-dive/generate-dev-synthetic` endpoint, providing `start_date` and `end_date` in the body.
7.  **View Output:** The generator will run, using synthetic data, rendering templates, generating JSON, and copying the *latest* built CSS/JS. Open the generated `reports/quarterly_deep_dive/html/index.html` file locally in your browser to review the appearance and chart loading.
8.  **Iterate:** Repeat steps 5-7 until the report structure, styling, and frontend behavior are correct.
9.  **Integrate Real Data/LLM:** Switch to testing the `/reports/quarterly-deep-dive/generate` endpoint after implementing the data fetching (`DocumentService`) and LLM call logic (`_get_llm_insight`).

## 9. Future Considerations / Extensions

*   **Background Tasks:** For longer-running reports, use FastAPI's `BackgroundTasks` for the `/generate` endpoint to avoid request timeouts.
*   **Error Handling:** Implement more robust error handling in data fetching, LLM calls, and file operations.
*   **More Chart Types:** Add more sophisticated visualizations (e.g., Sankey diagrams, heatmaps) via Plotly.
*   **Infographics:** Integrate more infographic-style elements, potentially combining small Plotly charts with custom HTML/CSS/SVG.
*   **Caching:** Implement caching for data fetching or LLM responses if applicable.
*   **User Interface:** Build a simple frontend UI to trigger reports instead of using API calls directly.
*   **Report Parameterization:** Allow more configuration options via the API request (e.g., specific CVE categories to focus on).
*   **Advanced Markdown:** Use a more powerful HTML-to-Markdown converter or dedicated Markdown Jinja templates if complex MD features are needed.
*   **Asynchronous Operations:** Convert potentially blocking operations (LLM calls, maybe extensive Pandas processing) to use `async`/`await` if FastAPI is run with async workers.
