Project startup
# Create and activate a new environment
conda create -n myenv python=3.9
conda activate myenv

https://anaconda.org/main/pymongo
conda install main::pymongo
conda install conda-forge::neo4j-python-driver
conda install qdrant-client
conda install main::pydantic-core
conda install main::cython==0.29.37
conda install main::uvicorn
conda install main::sqlalchemy

pip install protobuf==4.25.4
# grpcio-tools 1.62.2 has requirement protobuf<5.0dev,>=4.21.6
# opentelemetry-proto 1.16.0 has requirement protobuf<5.0,>=3.19

# FastEmbed is a lightweight, fast, Python library built for embedding generation. Used # by Qdrant and others.
# Models - https://qdrant.github.io/fastembed/examples/Supported_Models/
# snowflake/snowflake-arctic-embed-l (1024)
# BAAI/bge-large-en-v1.5 (1024)
# nomic-ai/nomic-embed-text-v1.5 (768)

## GPU Support Setup
To enable GPU support for FastEmbed (via ONNX Runtime), install dependencies in this order:
```bash
# 1. Install CUDA Toolkit
conda install -c nvidia cuda-toolkit=12.8.1

# 2. Install cuDNN
conda install -c conda-forge cudnn=9.8.0.87

# 3. Install ONNX Runtime GPU
pip install onnxruntime-gpu==1.21.0
```

Note: The specific versions above are known to work together. After installation, verify GPU support:
```python
import onnxruntime as ort
ort.get_device()  # Should return 'GPU'
```

FastEmbed will automatically use GPU acceleration when available through ONNX Runtime.

pip install fastembed-gpu

# Install JupyterLab and Jupyter Notebook
conda install -c conda-forge jupyterlab
conda install -c conda-forge notebook

# Install IPython widgets
conda install -c conda-forge ipywidgets

# Install Jupyter Notebook extensions
conda install -c conda-forge jupyter_contrib_nbextensions
conda install -c conda-forge jupyter_nbextensions_configurator

# Install and enable the nbextensions
jupyter contrib nbextension install --sys-prefix
jupyter nbextensions_configurator enable --sys-prefix

# Enable IPython widgets extension
jupyter nbextension enable --py widgetsnbextension --sys-prefix
jupyter nbextension install --py widgetsnbextension --sys-prefix

install requirements.txt

Build Dockerfile and docker-compose.yml


Going to production
remove Jupyter resources from environment
conda remove --name myenv jupyter jupyterlab notebook ipywidgets jupyter_contrib_nbextensions jupyter_nbextensions_configurator

update to test git connection


## TODO
### Refactor relationships for PatchManagementPosts
- Current - all posts within a thread are connected to the CVE or KB referenced in the original post
- Refactor- only the original post is connected to the CVE or KB referenced. But if a post within the thread explicitly references a CVE or KB then create an explicit relationship for it.
- Notes:
Adjusted Approach
1. Thread-Level to CVE Relationship:
- Maintain a single relationship from the originating post (thread level) to the CVE. This simplifies queries about the general topics discussed in the thread.
2. Selective Post-Level Relationships:
- Relate individual posts to the CVE only if they contain a Cause, Fix, or other significant information relevant to understanding or resolving the CVE. This helps filter out irrelevant posts and creates a more focused view of the discussion.
- Include key metadata, like timestamps, sentiment, or authority, to track the evolution of the discussion.
3. Tagging Posts with Significance:
    Posts should be tagged with properties like:
    contains_fix: true/false
    contains_cause: true/false
    contains_symptom: true/false
    This allows you to filter for posts that add substantive value without creating unnecessary edges.
4. Thread Navigation:
- Use PREVIOUS_MESSAGE edges to capture the order of the discussion. This helps reconstruct the flow of information and identifies where a solution (if any) originated.
5. Summarized Relationships:
- Instead of relating every post to the CVE, summarize the thread's key findings (e.g., causes, fixes) at the thread level:
Use properties or sub-nodes connected to the originating post to capture aggregated insights like causes_found or fixes_proposed.

#### Why This Works:
- Efficiency: Avoids excessive edges while still preserving valuable relationships.
- Granularity: Enables post-level insights when needed, without cluttering the graph with redundant connections.
- Flexibility: Supports both thread-level analysis (broad overview) and post-level analysis (detailed insight).
#### Suggested Schema Enhancements:
- Add a SIGNIFICANT edge type for posts containing critical information (e.g., cause, fix).
- Use properties on SIGNIFICANT edges to capture additional details like severity, confidence, or source reliability.
#### Example Query Patterns:
1. Identify All Fixes for a CVE
- Query posts with contains_fix: true linked to a specific CVE.
2. Track Key Post Flow
- Traverse PREVIOUS_MESSAGE edges backward from a post containing a Fix to understand how the solution evolved.
This approach balances scalability with granularity and makes the graph both useful and manageable.

End Refactor relationships for PatchManagementPosts


LlamaIndex
https://llamahub.ai/llama_index/quickstart

# When storing:
LlamaIndex Vector Index
├── Takes document
├── Chunks it into nodes
├── Sends embeddings to Qdrant (vector store)
├── Stores original text in docstore
└── Maintains mapping between them

# When retrieving:
LlamaIndex Vector Index
├── Takes query
├── Gets similar vectors from Qdrant
├── Maps vectors back to original content
└── Synthesizes final response


## KB Report API Endpoint

### Generate KB Report
Endpoint to generate a report of KB articles and their associated CVEs for a specified date range.

**Endpoint:** `/api/v1/reports/kb`
**Method:** POST
**Content-Type:** application/json

#### Request Format
```bash
curl -X 'POST' \
  'http://127.0.0.1:7501/api/v1/reports/kb' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "start_date": "2024-09-01T00:00:00",
    "end_date": "2024-09-15T23:59:59"
}'
```

#### Response Format
```json
{
    "status": "success",
    "data": [...],  // Array of KB articles with CVE details
    "message": "Successfully processed N KB articles"
}
```

## KB Articles and Product Family Relationships

### Current Implementation
- KB Articles in MongoDB Atlas are stored in `microsoft_kb_articles` collection
- Product family information is stored separately in `microsoft_product_builds` collection
- Current ETL pipeline creates KB articles by joining:
  - `microsoft_kb_articles`
  - `windows_10`
  - `windows_11` collections

### Data Structure
- `microsoft_product_builds` collection contains `product` property
- This property indicates which product the KB Article applies to
- Required for product family-based reporting

### Known Limitation
- `microsoft_kb_articles` collection doesn't directly support filtering by product family
- This impacts reporting capabilities where product family filtering is needed

### Recommended Solution
- Create an aggregation pipeline to:
  - Match documents between `microsoft_product_builds` and `microsoft_kb_articles`
  - Feature engineer the `product` column onto KB article records
  - This enables proper product family-based filtering and reporting

## Recent Updates [2024-12-24]

### Patch Post Transformer Enhancements
1. Historical Document Integration
   - Improved thread linking between historical and in-batch documents
   - Preserved existing thread_id and previous_id from historical documents
   - Updated next_id of last historical to point to first in-batch document
   - Updated previous_id of first in-batch to point to last historical document

2. Code Documentation
   - Added detailed docstrings with complete workflow documentation
   - Enhanced type hints for function signatures
   - Improved code organization and readability

3. Pipeline Optimization
   - Removed async wrapper for patch post transformation
   - Direct synchronous execution of transform_patch_posts_v2

4. Function Refinements
   - Enhanced group_by_normalized_subject with proper type hints
   - Improved documentation for subject normalization and fuzzy matching

### Next Steps
- Validate thread linking behavior with historical documents
- Monitor performance of synchronous patch post transformation
- Consider adding metrics for thread linking success rate

## Alpine.js CVE Popup Implementation Notes [2025-02-12]

### Key Issues and Solutions
1. **Data Structure Mismatch**
   - Issue: Initial implementation assumed flat CVE ID array but received nested objects
   - Solution: Processed nested structure to extract CVE IDs and details

2. **Severity Badge Styling**
   - Issue: Badges showed 'Unknown' due to incorrect data access path
   - Solution: Fixed data access path to `item.score.score_rating`

3. **Sorting by Severity**
   - Issue: CVEs appeared in random order instead of severity ranking
   - Solution: Added severity ranking system and sorting by:
     - Primary: Severity rank (Critical > High > Medium > Low > Unknown)
     - Secondary: Numeric score for same-severity CVEs

4. **Initial Display Count**
   - Issue: Showing too many CVEs initially made popup too tall
   - Solution: Limited to 6 CVEs per category with scroll for remainder

### Implementation Best Practices
1. **Data Processing**
   - Parse raw data on modal open
   - Create lookup maps for efficient CVE detail retrieval
   - Sort CVEs before display

2. **State Management**
   - Use Alpine.js store for modal state
   - Clear state on modal close

3. **Styling**
   - Use Tailwind CSS utility classes
   - Ensure responsive layout
   - Add scroll for overflow

4. **Error Handling**
   - Default to 'Unknown' for missing data
   - Add fallback styling for unknown severity

### Key Components
1. **Alpine.js Store**
   - Manages modal state and CVE data
   - Handles data processing and sorting

2. **Modal Template**
   - Responsive grid layout
   - Category sections with CVE lists
   - Severity badges with color coding

3. **Helper Functions**
   - getSeverityRank: Converts severity text to numeric rank
   - getCveDetails: Retrieves CVE details from lookup map

### Future Improvements
1. Add loading state during data processing
2. Implement search/filter within modal
3. Add CVE detail expansion panels
4. Support keyboard navigation

## Conda Dependency Management [2025-03-11]

### Adding New Dependencies
1. Add the package to `environment.yml` with version specification
2. Update the environment using:
   ```bash
   conda env update -f environment.yml --prune
   ```

### Package Version Format
- Format: `package=version=build_string`
- Example: `colorlog=6.9.0=pyh7428d3b_1`
  - `pyh7428d3b_1` is the build string, which can be found by:
  1. Search package on https://anaconda.org
  2. Select the appropriate channel (conda-forge/main)
  3. Find the specific version's build string

Note: The build string ensures exact package reproduction across environments.

## VSCode Settings Configuration
Place the following in `.vscode/settings.json` to ensure consistent code formatting and linting:

```json
{
    "diffEditor.experimental.showMoves": true,
    "editor.minimap.maxColumn": 105,
    "editor.inlineSuggest.syntaxHighlightingEnabled": true,
    "editor.quickSuggestions": {
        "comments": "on"
    },
    "editor.quickSuggestionsDelay": 3,
    "files.autoSave": "off",
    "files.insertFinalNewline": true,
    "files.trimTrailingWhitespace": true,
    "workbench.externalBrowser": "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe",
    "workbench.list.horizontalScrolling": true,
    "workbench.editor.highlightModifiedTabs": true,
    "workbench.editor.labelFormat": "short",
    "workbench.editor.pinnedTabsOnSeparateRow": true,
    "workbench.editor.titleScrollbarSizing": "large",
    "editor.formatOnSave": true,
    "editor.formatOnType": true,
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter"
    },
    "editor.codeActionsOnSave": {
        "source.organizeImports": "explicit"
    },
    "black-formatter.args": [
        "--line-length",
        "79",
        "--skip-string-normalization",
        "--preview"
    ],
    "isort.args": [
        "--profile",
        "black",
        "--line-length",
        "79"
    ],
    "flake8.args": [
        "--max-line-length=79",
        "--extend-ignore=W503",
        "--per-file-ignores=__init__.py:F401"
    ],
    "autoDocstring.startOnNewLine": false,
    "autoDocstring.docstringFormat": "google"
}
```

This configuration:
1. Sets up Black as the default Python formatter with 79 character line length
2. Configures isort to match Black's style for import organization
3. Aligns Flake8 with our .flake8 configuration
4. Sets up proper docstring formatting rules
5. Enables format on save and format on type for better code consistency

After updating settings.json, restart VSCode for changes to take effect.

## Tailwind CSS Setup and Configuration

### Installation and Setup (Latest Version: 4.1.4)

1. Initialize a new npm project (if not already done):
   ```bash
   npm init -y
   ```

2. Install TailwindCSS and its dependencies:
   ```bash
   npm install -D tailwindcss@latest postcss autoprefixer @tailwindcss/typography
   ```

3. Generate configuration files:
   ```bash
   npx tailwindcss init -p
   ```

4. Configure your `tailwind.config.js`:
   ```javascript
   module.exports = {
     content: [
       './application/data/templates/**/*.html',
       './application/static/js/**/*.js'
     ],
     darkMode: 'class',
     theme: {
       extend: {
         // Your theme extensions here
       }
     },
     plugins: [
       require('@tailwindcss/typography')
     ]
   }
   ```

5. Create your CSS entry point (e.g., `src/input.css`):
   ```css
   @tailwind base;
   @tailwind components;
   @tailwind utilities;
   ```

6. Add build scripts to `package.json`:
   ```json
   {
     "scripts": {
       "build:css": "tailwindcss -i src/input.css -o ./application/static/css/QuarterlyReportStylesheet.css",
       "watch:css": "tailwindcss -i src/input.css -o ./application/static/css/QuarterlyReportStylesheet.css --watch",
       "build:prod": "tailwindcss -i src/input.css -o ./application/static/css/QuarterlyReportStylesheet.css --minify"
     }
   }
   ```

### Building CSS

- Development build: `npm run build:css`
- Watch mode: `npm run watch:css`
- Production build: `npm run build:prod`

### Documentation Links

- [TailwindCSS Documentation](https://tailwindcss.com/docs)
- [Upgrade Guide](https://tailwindcss.com/docs/upgrade-guide)
- [@tailwindcss/typography Plugin](https://tailwindcss.com/docs/typography-plugin)

### Important Notes

1. Always use the local installation of TailwindCSS (via npm scripts or npx) rather than global installation
2. The project uses TailwindCSS v4.1.4, which includes:
   - Improved performance
   - Enhanced JIT engine
   - Better dark mode support
   - Updated typography plugin compatibility

3. Common Issues and Solutions:
   - If the CLI isn't recognized, ensure you're using npm scripts or npx
   - If styles aren't applying, check the content paths in tailwind.config.js
   - For optimal performance, use the --minify flag in production builds

## Tailwind CSS Implementation Notes (v4+)

1. **No tailwind.config.js**
   - Since upgrading to Tailwind CSS v4+, the project no longer uses a `tailwind.config.js` file. All configuration is handled via conventions and direct imports.

2. **Brand Styles in :root**
   - Company brand styles (color palette, typography, spacing) are defined in a `:root` selector in `company-base.css` to keep brand tokens modular and separate from utility/component styles.
   - `company-base.css` is imported at the top of `input.css`.

3. **Component Styles in report-styles.css**
   - `report-styles.css` contains only component definitions, all wrapped in `@layer components { ... }` for proper Tailwind layering and maintainability.

4. **input.css Structure**
   - The main entrypoint, `input.css`, imports in this order:
     1. `company-base.css` (brand tokens)
     2. `@theme { ... }` for theme variables if needed
     3. `@import 'tailwindcss' layer(base, components, utilities) variants(responsive);`
     4. Base styles
     5. `report-styles.css` (component styles)

5. **Future UI Tweaks**
   - To adjust the UI, update `report-styles.css` for component-level changes and `input.css` for global or layer ordering changes.

---

**Note:** This approach is tailored for Tailwind v4+ where config files are optional and most customization is handled via CSS layers and imports. For more details, see Tailwind v4+ documentation.

## Code Block Styling with Pygments

The project uses two separate stylesheets for code block styling:

1. **Container Styling (`stylesheet.css`)**:
   - Handles the outer container appearance (borders, padding, headers)
   - Follows the page theme and design system
   - Defines classes like `.code-block-wrapper`, `.code-block-header`

2. **Syntax Highlighting (`pygments-xcode.css`)**:
   - Generated using Pygments: `pygmentize -S xcode -f html > pygments-xcode.css`
   - Provides syntax highlighting styles for code content
   - Must match the `pygments_style` setting in the markdown configuration:
     ```python
     markdown.markdown(
         text,
         extension_configs={
             'codehilite': {
                 'pygments_style': 'xcode'  # Must match generated CSS
             }
         }
     )
     ```

For available Pygments themes, see [Pygments Built-in Styles](https://pygments.org/styles/).

## Lessons Learned

## Conda Dependency Management Best Practices

**Issue Date:** April 4, 2025

**Problem:**
- Inconsistent package versions across environments
- Confusion about Python-specific vs. generic package versions
- Uncertainty about when to pin exact versions

**Resolution: Multi-Step Dependency Workflow**
1. Search conda-forge for available packages and versions
2. Test installation with dry run:
   ```bash
   conda install <package> --dry-run
   ```
3. Note the exact version and build hash that conda would install
4. Update `environment.yml` based on package maturity:
   - For newer/startup packages: Pin exact versions to ensure stability
   - For mature packages: Allow conda to resolve latest compatible version unless:
     * Major version updates are expected
     * Known deprecations are coming
     * Specific version is required for compatibility

**Package Version Selection Guidelines**
- Python-specific versions (e.g., `py311haa95532_0`):
  * Compiled specifically for that Python version
  * May have optimizations for that version
  * Example: `azure-storage-blob-12.19.0-py311haa95532_0`

- Generic versions (e.g., `pyhd8ed1ab_0`):
  * Pure Python packages that work across versions
  * More portable across environments
  * Example: `azure-storage-blob-12.24.1-pyhd8ed1ab_0`

Choose generic versions when available unless specific Python version optimizations are needed.

## Initialization Strategies: `__init__` vs. `@property` (Lazy Initialization)

**Issue Date:** April 5, 2025

**Context:**
When designing classes, particularly those managing external resources or expensive objects (like database connections or API clients), deciding *when* to initialize these resources is important. Two common patterns are direct initialization in the `__init__` method and lazy initialization using the `@property` decorator.

**Explanation:**

*   **Direct Initialization (`__init__`)**: The resource or object is created immediately when an instance of the class is instantiated.
    ```python
    class DirectInitExample:
        def __init__(self, connection_string):
            print("Initializing resource in __init__...")
            self.resource = setup_expensive_resource(connection_string) # Happens immediately
            print("Resource ready.")
    ```

*   **Lazy Initialization (`@property`)**: The resource or object creation is deferred until the corresponding attribute is accessed for the *first time*. The `@property` decorator turns a method into a managed attribute (accessed without `()`), and the method's code handles the on-demand creation.
    ```python
    class LazyInitExample:
        def __init__(self, connection_string):
            self._connection_string = connection_string
            self._resource = None # Placeholder
            print("LazyInitExample instance created, resource not yet initialized.")

        @property
        def resource(self):
            if self._resource is None:
                print("First access: Initializing resource via @property...")
                self._resource = setup_expensive_resource(self._connection_string) # Happens only now
                print("Resource ready.")
            return self._resource
    ```

**Comparison:**

| Feature             | Direct Initialization (`__init__`)                     | Lazy Initialization (`@property`)                       |
| :------------------ | :----------------------------------------------------- | :------------------------------------------------------ |
| **Timing**          | Immediately on object creation                         | On first access of the property                         |
| **Initial Cost**    | Higher (initialization cost paid upfront)              | Lower (initialization cost deferred)                    |
| **Error Discovery** | Early (errors occur during object creation)            | Late (errors occur on first use)                        |
| **First Use Speed** | Fast (resource already available)                      | Slower (includes initialization time on first access) |
| **Resource Usage**  | Resource always created, even if never used            | Resource only created if actually needed                |
| **Code Complexity** | Simpler `__init__`, simpler getter (if needed)         | More complex getter logic (check, create, store)        |
| **Thread Safety**   | Generally simpler                                      | Requires explicit locking for thread-safe lazy init     |

**When to Use Which:**

*   **Use `__init__` when:**
    *   The resource is essential for the object's core functionality and likely always needed.
    *   Early failure detection (e.g., bad configuration) is critical.
    *   The initialization cost is acceptable upfront.
    *   Simpler code is preferred.
*   **Use `@property` (Lazy Init) when:**
    *   The resource is expensive to create and might not always be used.
    *   Deferring startup cost is important.
    *   The resource is optional or used in specific scenarios only.
    *   You can tolerate potential errors occurring later, upon first use.

*   **Use `__enter__` (Context Manager) when:**
    *   Resource management

<!-- brief overview of use of __enter__ and __exit__ in context managers -->
The SFTPService class is a service abstraction intended for interactions with an SFTP server—specifically for file uploads.
One of its most notable features is the implementation of context management methods (__enter__ and __exit__), which is increasingly
considered a modern design strategy when working with resources that must be explicitly managed and cleaned up (e.g., network
connections, file handles). By requiring the user to employ a with block, the class guarantees that connections are properly
established and later closed, reducing the likelihood of resource leaks. This design not only enhances code safety but also
communicates to developers that they must respect the intended usage pattern.

Implementing __enter__ and __exit__ implies that the SFTPService manages a critical resource lifecycle. The __enter__ method
invokes a private _connect method that sets up both the SSH and SFTP connections. The _connect method is carefully designed:
it checks for an active SSH connection before attempting to reconnect, thereby avoiding redundant network calls. It also handles
various exceptions that might occur during connection establishment, from file-not-found issues (if the private key is missing) to
authentication and network errors. Each exception is logged with context-specific messages, and then re-raised as a ConnectionError.
This strategy ensures that any failures are explicitly signaled and logged, facilitating debugging and error handling in higher
layers of the application.

In addition to establishing connections, the SFTPService class encapsulates cleanup logic in its _close method. This method gracefully
shuts down the SFTP session and SSH connection. Its design is resilient: even if one part of the connection closure fails, it logs
the error and attempts to close the remaining connection. This robust error handling ensures that the system remains in a predictable
state even in the face of exceptions, aligning with the principle of graceful degradation.

Beyond connection management, the class adheres to several other critical design principles. It encapsulates its configuration settings
within a dedicated SFTPSettings object, thereby promoting the separation of configuration from business logic. This makes the class
more maintainable and testable because changes in the configuration do not directly affect the core logic.

The class also practices defensive programming. For example, before attempting operations like file uploads, it checks whether
the SFTP client is connected using the is_connected property and the _ensure_connection method. These checks prevent operations on
an uninitialized or closed connection, ensuring that errors are caught early and reported clearly.

When it comes to uploading files, the SFTPService provides both single file and batch upload methods. The single file upload method
(upload_file) first validates the existence of the local file and normalizes the remote file path. It then ensures that the remote
directory structure exists before attempting to transfer the file. If the directory does not exist, it recursively creates it.
The batch upload method (upload_files) leverages the single file upload function and logs the success or failure of each file upload.
This modular approach demonstrates sound separation of concerns, as each method is responsible for a distinct part of the overall functionality.

Key design aspects include:

- Context Management Enforcement:
  - Implements __enter__ and __exit__ to automate resource setup and teardown.
  - Enforces usage within a with block to ensure connections are properly established and closed.

- Robust Error Handling and Logging:
  - Detailed logging is integrated throughout the connection, file operations, and exception handling routines.
  - Catches specific exceptions (e.g., FileNotFoundError, AuthenticationException) and re-raises them as ConnectionError.

- Encapsulation and Modularity:
  - Separates configuration (SFTPSettings) from connection logic.
  - Uses private methods (prefixed with _) to encapsulate internal implementation details.

- Defensive Programming:
  - Validates prerequisites (such as checking if the connection is active and verifying file existence) before proceeding with operations.
  - Ensures that resources are only used when in a valid state.

- Resource Lifecycle Management:
  - Manages the SSH and SFTP clients in a way that ensures proper allocation and cleanup.
  - Uses context management to enforce a strict order of operations.

When considering such design strategies for your own classes, ask:
- Does my class manage resources that require explicit allocation and cleanup?
- Is there a risk of resource leaks if cleanup code isn’t invoked?
- Would explicit initialization and teardown improve clarity and maintainability?
- How critical is robust error handling and detailed logging for my operations?
- Do I need to enforce a strict order of operations (e.g., connect before uploading)?

Overall, the SFTPService class provides a robust template for managing network connections, emphasizing clear resource management, modular design, and proactive error handling.

Key Design Principles:
- **Context Management:**
  - Implements __enter__ and __exit__ for automatic resource handling.
  - Enforces proper usage via the with statement.

- **Error Handling:**
  - Catches and logs specific exceptions.
  - Re-raises errors with clear context for higher-level management.

- **Encapsulation & Modularity:**
  - Isolates configuration from business logic.
  - Uses private methods to hide internal details.

- **Defensive Programming:**
  - Checks resource states before proceeding.
  - Prevents operations on invalid or closed connections.

- **Resource Lifecycle Management:**
  - Ensures proper setup and teardown of SSH and SFTP connections.
  - Leverages context management to avoid resource leaks.

<!-- brief overview of use __enter__ and __exit__ in context managers -->

### Additional notes
Ah, that's an excellent point and highlights the difference between channel sources and package build types. Let's break down those azure-storage-blob examples:

Build Types:

py310haa95532_0, py311haa95532_0, etc. (from pkgs/main): These are Python-version-specific builds. They are compiled/packaged specifically for Python 3.10, 3.11, etc. This is often necessary if the package includes compiled C/C++/Fortran extensions that interact directly with the Python C API, which can change between Python versions.
pyh... builds (like the newer ones on conda-forge): These are noarch: python builds. The h in the build string often indicates this. These packages are typically pure Python code (no compiled extensions) and are designed to work on any operating system and any Python version that meets the package's minimum requirements (e.g., Python >= 3.8). They don't need to be rebuilt for every specific Python version.
Channel Differences & Version Lag:

pkgs/main (Anaconda's default channel) has version 12.19.0, built specifically for different Python versions.
conda-forge (Community channel) has newer versions (up to 12.25.0) packaged as noarch: python.
It's very common for conda-forge to have newer versions, especially for packages that can be distributed as noarch. Packaging noarch is often easier for maintainers as they only need one build for many Python versions.
Which to Prefer? (Sonnet's Advice vs. Reality):

The advice to "prefer generic (noarch) unless specific optimizations exist" is generally sound if the versions were the same. noarch packages are simpler and work broadly.
However, Conda prioritizes channel order and then the highest version number. Since conda-forge is listed first in your environment.yml channels section, and it has newer versions (12.25.0 > 12.19.0), Conda will almost certainly choose the 12.25.0 pyhd8ed1ab_0 build from conda-forge when you run conda install azure-storage-blob.
In this case, the newer version from conda-forge is the better choice anyway, and it happens to be a noarch build. You don't need to worry about the older, Python-specific builds on pkgs/main unless conda-forge was unavailable or had issues.
How to Instruct Conda:

You usually don't need to explicitly instruct it. Because conda-forge is your highest priority channel and has the newest version, running conda install azure-storage-blob will automatically select azure-storage-blob 12.25.0 pyhd8ed1ab_0 from conda-forge.
If you wanted the older pkgs/main version (not recommended here), you could force it: conda install pkgs/main::azure-storage-blob=12.19.0.
If you wanted to ensure it came from conda-forge (redundant here, but for illustration), you could use: conda install conda-forge::azure-storage-blob.
Recommendation:

Stick with the simplest approach. Add - azure-storage-blob to your environment.yml under dependencies, then run conda env update --file environment.yml --prune. Conda will use your channel priority (conda-forge first) and version preference (highest) to install the latest version (12.25.0 noarch build) from conda-forge, which is exactly what you want.


That's a great observation about the different builds and channels!

py311... builds (like on pkgs/main) are specific to a Python version, often because they contain compiled code.
pyh... builds (like the newer ones on conda-forge) are usually noarch: python packages. They are pure Python and work on any compatible Python version. conda-forge often updates these faster.
Which does Conda choose? It prioritizes:

Channel Order: Your environment.yml lists conda-forge first.
Highest Version: It looks for the highest version number compatible with your environment (Python 3.11).
Since conda-forge is listed first and has the newer azure-storage-blob version (12.25.0), Conda will automatically pick that noarch version when you run conda install azure-storage-blob. You don't need to explicitly tell it to prefer noarch or conda-forge in this case; the default behavior works in your favor.

So, the advice to prefer noarch is less relevant here than the fact that the newest version is noarch and available on your preferred channel (conda-forge).

Recommendation: Just add - azure-storage-blob to your environment.yml and run conda env update --file environment.yml --prune. Conda will install the desired 12.25.0 version from conda-forge.

## MCP Server Installation Issues

### Exa MCP Server Installation and Configuration Challenges

**Issue Date:** March 12, 2025

**Problem:**
- Default npm installation method (`npm install -g exa-mcp-server`) succeeded but server failed to connect
- Using `npx` to run the server as suggested in documentation didn't work
- Server connection errors persisted after initial configuration

**Impact:**
- MCP server was not accessible for web searches
- Received "Not connected" errors when attempting to use the server's tools

**Root Cause:**
- Documentation assumed `npx` would work for running the server
- Path resolution issues with globally installed npm packages
- Incomplete configuration instructions for Cline/WindSurf integration

**Resolution Steps:**
1. Installed server globally:
   ```bash
   npm install -g exa-mcp-server
   ```

2. Located actual executable path:
   ```bash
   npm list -g exa-mcp-server
   # Found at C:\Users\[user]\AppData\Roaming\npm\node_modules\exa-mcp-server
   ```

3. Modified Cline MCP settings to use full path:
   ```json
   {
     "mcpServers": {
       "github.com/exa-labs/exa-mcp-server": {
         "command": "node",
         "args": ["C:\\Users\\[user]\\AppData\\Roaming\\npm\\node_modules\\exa-mcp-server\\build\\index.js"],
         "env": {
           "EXA_API_KEY": "your-api-key"
         },
         "disabled": false,
         "autoApprove": []
       }
     }
   }
   ```

**Prevention:**
- Always use full paths in MCP server configurations
- Test server connection immediately after installation
- Document specific configuration requirements for different environments (Claude Desktop vs Cline/WindSurf)

**Related Components:**
- Exa MCP Server
- Cline Extension
- WindSurf
- npm Global Packages

## Docker and Container Issues

### Docker Container Log Corruption Causing Vector Database Unavailability

**Issue Date:** January 30, 2025

**Problem:**
- Docker container for Qdrant vector database became unavailable due to log file corruption
- Log file contained invalid characters (`\x00`) causing errors in log processing
- Python application received `qdrant_client.http.exceptions.ResponseHandlingException`
- Error occurred in `qdrant_client.http.api_client.py` during request handling

**Impact:**
- Vector database became inaccessible
- ETL pipeline processing was blocked
- Application unable to perform vector operations

**Root Cause:**
- Log file corruption with null characters (`\x00`)
- Invalid character sequence in Docker container logs preventing proper log processing

**Resolution Steps:**
1. Map WSL Docker data directory to local drive:
   ```powershell
   net use Z: \\wsl$\docker-desktop-data
   ```

2. Navigate to container logs directory:
   ```powershell
   cd /d Z:\data\docker\containers\[container_id]
   ```

3. Open logs in VSCode:
   ```powershell
   code .
   ```

4. Use VSCode regex search to locate corrupted lines:
   - Search pattern: `\x00`
   - Delete affected lines from log file

5. Restart Docker container to resume normal operations

**Prevention:**
- Monitor Docker container logs for corruption
- Consider implementing log rotation to prevent large log files
- Add error handling in application code for vector database connection issues

**Related Components:**
- Qdrant Vector Database
- Docker Container Logs
- WSL2 Integration
- ETL Pipeline

## LiteLLM Completion Methods [2025-04-21]

LiteLLM provides a unified interface to interact with various Large Language Models (LLMs). It offers both synchronous (`completion`) and asynchronous (`acompletion`) methods for generating chat completions.

### Core Methods

1.  **`litellm.completion(...)` (Synchronous)**
    *   Blocks execution until the LLM response is received.
    *   Suitable for simple scripts or applications where concurrency isn't critical.
    *   Example: Used in `chat_service.py` for direct responses.

2.  **`litellm.acompletion(...)` (Asynchronous)**
    *   Uses `async`/`await` syntax for non-blocking execution.
    *   Ideal for web servers (like FastAPI/Uvicorn) or applications needing to handle multiple requests concurrently without waiting for each LLM call to finish.
    *   Requires an `asyncio` event loop to run.

Both methods accept similar parameters to control the generation process.

### Key Parameters Explained

*   **`model`** (str, required):
    *   Specifies the LLM to use (e.g., `"gpt-3.5-turbo"`, `"claude-3-opus-20240229"`, `"azure/your-deployment-name"`).
    *   LiteLLM routes the request based on the model name and configured API keys/bases.

*   **`messages`** (List[Dict], required):
    *   A list representing the conversation history, where each dictionary has `"role"` (`"system"`, `"user"`, `"assistant"`) and `"content"`.
    *   Example: `[{"role": "user", "content": "Explain quantum physics simply."}]`

*   **`temperature`** (float, optional, 0.0-2.0, default varies):
    *   Controls the randomness of the output. Lower values (e.g., 0.2) make the output more deterministic and focused, while higher values (e.g., 0.8) make it more creative and diverse.
    *   Example: Use `0.2` for factual summaries, `0.7` for creative writing.

*   **`max_tokens`** (int, optional):
    *   Limits the maximum number of tokens (words/subwords) the model can generate in its response.
    *   Example: Set to `100` for a brief summary, or `1000` for a detailed explanation.

*   **`top_p`** (float, optional, 0.0-1.0):
    *   Nucleus sampling: Considers only the most probable tokens whose cumulative probability exceeds `top_p`. It's an alternative to `temperature`.
    *   Example: `top_p=0.9` means the model considers the smallest set of tokens that make up 90% of the probability mass, leading to less random but still diverse outputs compared to very low temperatures. Don't use with `temperature` typically.

*   **`stream`** (bool, optional, default=False):
    *   If `True`, the response is returned as a stream of chunks (tokens) as they are generated, rather than waiting for the full response.
    *   Example: Use `True` in a chatbot UI to display the response word-by-word for a better user experience.

*   **`stop`** (str or List[str], optional):
    *   Specifies sequences where the API should stop generating further tokens.
    *   Example: `stop=["\n", " Human:"]` tells the model to stop if it generates a newline or the start of a human turn.

*   **`presence_penalty`** (float, optional, -2.0 to 2.0):
    *   Penalizes new tokens based on whether they have already appeared in the text, encouraging the model to talk about new topics. Higher values increase this effect.
    *   Example: Use a positive value (e.g., `0.5`) to reduce repetition in longer generated texts.

*   **`frequency_penalty`** (float, optional, -2.0 to 2.0):
    *   Penalizes new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. Higher values increase this effect.
    *   Example: Use a positive value (e.g., `0.3`) to make the model use less common words or phrases.

*   **`response_format`** (dict, optional):
    *   Forces the output to be in a specific format, currently supporting JSON mode.
    *   Example: `response_format={"type": "json_object"}` requires the model to output valid JSON. You *must* also instruct the model via the prompt (e.g., "Provide your answer in JSON format.").

*   **`seed`** (int, optional):
    *   If specified, the model provider will attempt to make the output deterministic (same input + seed = same output). Not guaranteed across all models/providers.
    *   Example: Use `seed=42` during testing to get reproducible results for the same prompt and parameters.

*   **`tools`** and **`tool_choice`** (List/str, optional):
    *   Used for function calling/tool use, allowing the model to request external tools be called.
    *   Example: Provide a list of available functions (like `get_current_weather`) in `tools`, and use `tool_choice="auto"` to let the model decide if/when to call one.

*   **`logprobs`** (bool, optional, default=False):
    *   If `True`, the response includes the log probabilities (logarithm of the probability) for each output token. This tells you how confident the model was about choosing each specific token in the sequence.
    *   Example: Setting `logprobs=True` might show that the model assigned a high probability (low negative logprob) to "Paris" when asked "Capital of France?", but lower probabilities to alternatives like "London". Useful for analyzing model confidence or implementing advanced techniques like beam search.

*   **`top_logprobs`** (int, optional, 0-5):
    *   Requires `logprobs=True`. Returns the log probabilities for the specified number of most likely tokens at each position.
    *   Example: `logprobs=True, top_logprobs=3` would show the top 3 token choices and their log probabilities for each step in the generation. This helps understand *why* the model chose a specific token and what alternatives it considered.

*   **`api_key`**, **`api_base`**, **`api_version`**, **`deployment_id`** (str, optional):
    *   Allow overriding environment variables or global settings for specific calls, useful for multi-provider setups or Azure deployments.
    *   Example: `litellm.completion(model="azure/my-gpt4", api_key="...", api_base="...", api_version="...")`

*   **`timeout`** (int/float, optional, default=600):
    *   Sets the maximum time (in seconds) to wait for a response from the LLM API.
    *   Example: Set `timeout=30` for quick user interactions where a long wait is unacceptable.

*   **`metadata`** (dict, optional):
    *   A dictionary of custom key-value pairs passed through to logging and callback functions. Does not affect the LLM call itself.
    *   Example: `metadata={"user_id": "123", "session_id": "abc"}` to track request origins in logs.

Understanding these parameters allows for fine-tuning LLM behavior for specific tasks, balancing creativity vs. determinism, controlling response length, enforcing structure, and analyzing model outputs.

## Metrics Database

A dedicated metrics database has been implemented to store and manage application performance and usage metrics.

**Key Components:**

*   **Database:** Uses DuckDB (`metrics.db`).
*   **Location:** `C:\Users\emili\PycharmProjects\microsoft_cve_rag\microsoft_cve_rag\application\data\metrics_db\metrics.db`
*   **Management:** Managed via a separate Docker appliance incorporating Alembic for migrations.
*   **Data Models:** Defined using SQLModel within a shared package (`microsoft_cve_report_models`).
*   **Service Layer:** A `metrics_service` class encapsulates all CRUD operations for the metrics data.

**Shared Package Installation:**

The SQLModel definitions are maintained in a separate repository. To use the metrics feature, this package must be installed:

```bash
pip install git+https://github.com/emilio-gagliardi/microsoft_cve_report_models.git#egg=microsoft_cve_report_models
```

**Note:** The DuckDB Docker appliance is primarily meant to handle the initial set up of the duckdb database with the correct tables and schema. Then the appliance is used to generate migrations and update the database schema.

## MCP Servers

### MotherDuck MCP Server (`mcp-server-motherduck`)

Configuring the `mcp-server-motherduck` to connect to a local DuckDB file (`metrics.db`) involved several troubleshooting steps. The goal was to enable querying the local database via Cascade (Windsurf client in PyCharm).

**Initial Configuration & Challenges:**

1.  **`uvx` Command:** The initial approach involved using `uvx` (from the `uv` package manager) in `mcp_config.json` to run the `mcp-server-motherduck` package:
    ```json
    "mcp-server-motherduck": {
      "command": "uvx",
      "args": ["mcp-server-motherduck", "--db-path", "<path_to_db>"],
      "env": {"HOME": "<project_home>", "PYTHONUNBUFFERED": "1"},
      "disabled": false
    }
    ```
2.  **Server Termination Issues:** This configuration led to the server process terminating immediately upon launch by the Windsurf client. This manifested as:
    *   Cascade reporting "server process has ended."
    *   PyCharm's "Output" tab for the MCP server showing "Got EOF from server."
    *   Manual attempts to run the server sometimes revealed an `IOException: File is already open`, suggesting a file locking issue or an environment incompatibility when launched via `uvx` by the IDE.

**Troubleshooting Steps & Evolution:**

1.  **Direct Python Module Execution (Attempt 1):**
    *   To align with the project's Conda-based environment (`cve_support_rag_venv`), `mcp-server-motherduck` was installed directly into the Conda environment using `pip install mcp-server-motherduck`.
    *   `mcp_config.json` was updated to run the server as a Python module:
        ```json
        "command": "C:\\Users\\emili\\anaconda3\\envs\\cve_support_rag_venv\\python.exe",
        "args": ["-m", "mcp_server_motherduck", "--db-path", "..."],
        ```
    *   **Outcome:** This failed with the error: `No module named mcp_server_motherduck.__main__; 'mcp_server_motherduck' is a package and cannot be directly executed.` This indicated the package wasn't designed to be run directly with `python -m`.

2.  **Direct Executable Execution (Successful Approach):**
    *   **Discovery:** After `pip install mcp-server-motherduck` into the `cve_support_rag_venv` Conda environment, an executable wrapper `mcp-server-motherduck.exe` was found in the environment's `Scripts` directory (`C:\Users\emili\anaconda3\envs\cve_support_rag_venv\Scripts\`).
    *   **Final `mcp_config.json` Configuration:**
        ```json
        "mcp-server-motherduck": {
          "command": "C:\\Users\\emili\\anaconda3\\envs\\cve_support_rag_venv\\Scripts\\mcp-server-motherduck.exe",
          "args": [
            "--db-path",
            "C:\\Users\\emili\\PycharmProjects\\microsoft_cve_rag\\microsoft_cve_rag\\application\\data\\metrics_db\\metrics.db",
            "--home-dir", // Explicitly set for clarity and DuckDB's needs
            "C:\\Users\\emili\\PycharmProjects\\microsoft_cve_rag"
          ],
          "env": {
            "PYTHONUNBUFFERED": "1"
          },
          "disabled": false
        }
        ```
    *   **Critical Step:** Restarting PyCharm after any changes to `mcp_config.json` was essential for the Windsurf client to pick up the new configuration.
    *   **Outcome:** This configuration successfully launched the server, allowing queries like `SHOW TABLES;` to be executed via Cascade.

**Key Learnings:**

*   For MCP servers that are Python packages, if direct `uvx` or `python -m` approaches fail (especially with file access or environment issues within an IDE-launched context), installing the package into the primary project environment (e.g., Conda) and then targeting the `pip`-generated executable script directly in `mcp_config.json` can be a robust solution.
*   The `HOME` directory can be important for DuckDB; passing it explicitly as `--home-dir` to the server is a good practice.
*   Always restart the IDE (PyCharm) after modifying `mcp_config.json` for changes to take effect.

## Plotly Layout and Coordinates

### Figure Layout and Margins

Plotly uses a normalized coordinate system (0-1) for positioning elements within a figure. This system is particularly useful for consistent placement of annotations, shapes, and other elements across different figure sizes.

#### Normalized Page Coordinates (Paper Coordinates)
```
  y=1   ┌───────────────────────────────────────────────┐
        │              FIGURE BOUNDING BOX              │
        │                                               │
 margin │   y=1-margin.top                              │
  top   │    ┌───────────────────────────────────┐      │
        │    │             PLOT AREA             │      │
        │    │                                   │      │
        │    │                                   │      │
 margin │    │              …                    │      │
  bot   │    │                                   │      │
        │    └───────────────────────────────────┘      │
        │ y=margin.bottom                               │
  y=0   └───────────────────────────────────────────────┘
        x=0         x=1-margin.right    x=1       (Paper X)
```

#### Key Points:
- The entire figure exists in a coordinate space where (0,0) is the bottom-left corner and (1,1) is the top-right corner
- Margins are specified in normalized coordinates (0-1) relative to the figure size
- The plot area is automatically calculated based on the specified margins
- Common margin settings:
  - `margin=dict(l=50, r=50, t=80, b=50)`  # Left, Right, Top, Bottom in pixels
  - `margin=dict(autoexpand=True)`  # Let Plotly automatically determine margins

#### Example Usage:
```python
import plotly.graph_objects as go

fig = go.Figure()

# Add your traces here
# fig.add_trace(...)


# Set layout with custom margins
fig.update_layout(
    margin=dict(l=50, r=50, t=80, b=50),  # Left, Right, Top, Bottom margins
    paper_bgcolor='white',
    plot_bgcolor='white',
    height=600,
    width=800,
    # Add legend with paper coordinates
    legend=dict(
        x=1,               # X at right‐edge of paper
        y=0,               # Y at bottom of paper
        xanchor="right",   # anchor the legend's right side at x
        yanchor="bottom",  # anchor the legend's bottom at y
        orientation="h"    # horizontal orientation
    )
)

# Add annotations using normalized coordinates
fig.add_annotation(
    x=0.5,  # Center of x-axis (0-1)
    y=1.05,  # Just above the top of the plot area
    text="Figure Title",
    showarrow=False,
    xref="paper",
    yref="paper",
    font=dict(size=16)
)

fig.show()
```

### Legend Positioning in Paper Coordinates

When positioning a legend using paper coordinates, Plotly follows these steps:
1. Computes the pixel location of the specified (x, y) coordinates based on the figure's width, height, and margins
2. Places the legend's anchor point (determined by `xanchor` and `yanchor`) at that exact position

#### Example: Bottom-Right Anchored Legend
```python
legend=dict(
    x=1,               # X at right‐edge of paper
    y=0,               # Y at bottom of paper
    xanchor="right",   # anchor the legend's right side at x
    yanchor="bottom",  # anchor the legend's bottom at y
    orientation="h"    # horizontal orientation
)
```

#### Visual Representation:
```
Legend Anchored to Bottom‐Right
  y=1 ┌───────────────────────────────────────────────────────┐
      │                                                       │
      │            [ Plot Area — axes, bars, etc. ]          │
 margin│                                                     │
  top  │                                                     │
      │                                                       │
      │   ┌───────────────────────────────────────────────┐   │
      │   │                                               │   │
      │   │                                               │   │
 margin│   │             (x=1, y=0) ⬇                     │   │
  bot  │   │                         ┌───────────────┐     │   │
      │   │                         │ Legend Box    │     │   │
      │   │                         │ [A][B][C]      │     │   │
  y=0 └───└─────────────────────────┴───────────────┴─────┘───┘
      x=0             x=1-margin.right      x=1
```

#### Common Anchor Combinations:
- Top-right: `x=1, y=1, xanchor="right", yanchor="top"`
- Top-left: `x=0, y=1, xanchor="left", yanchor="top"`
- Bottom-left: `x=0, y=0, xanchor="left", yanchor="bottom"`
- Center-right: `x=1, y=0.5, xanchor="right", yanchor="middle"`

#### Additional Legend Options:
- `bgcolor`: Set the legend background color
- `bordercolor`: Color of the legend border
- `borderwidth`: Width of the legend border
- `font`: Font properties for the legend text
- `itemsizing`: Size of the legend items ("constant" or "trace")
- `itemwidth`: Width of the legend item symbols in pixels

### Margins and Title Offsets

This diagram illustrates how Plotly's paper coordinates, margins, and title offsets work together:

```
Paper Coordinates (0-1 in both x and y)
y=1 ┌───────────────────────────────────────────────────────────────────┐
    ┆   ↑ margin.top (pixels)                                           ┆
    ┆   ┆                                                               ┆
    │   ┌───────────────────────────────────────────────────────────┐   │
    ┆   ┆┄┄← margin.left (pixels)                                   ┆   ┆
    ┆   ┆  ┆                                                        ┆   ┆
    ┆   ┆  ┆  ↑ title_standoff_y (pixels)                           ┆   ┆
    ┆   ┆  ┆  ┆  ┌┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┐      ┆   ┆
    ┆   ┆  ┆  ┆  ┆       Y-axis title ("CVE Count")          ┆      ┆   ┆
    ┆   ┆  ┆  ┆  ┆  *                                        ┆      ┆   ┆
    │   │  │  │  │  │  ┌───────────────────────────────┐     │      │   │
    │   │  │  │  │  │  │  Data Region                  │     │      │   │
    │   │  │  │  │  │  │  (bars, lines, etc.)          │     │      │   │
    │   │  │  │  │  │  │                               │     │      │   │
    │   │  │  │  │  │  │  * * * * * * * * * * * * * * *│*    │      │   │
    │   │  │  │  │  │  │  * * * * * * * * * * * * * * *│*    │      │   │
    │   │  │  │  │  │  │  * * * * * * * * * * * * * * *│*    │      │   │
    │   │  │  │  │  │  └───┬───────────────────────┬───┘     │      │   │
    ┆   ┆  ┆  ┆  │  │      ┆  X-axis title_standoff┆         ┆      ┆   ┆
    ┆   ┆  ┆  ┆  └┄┄┄┄┄┄┴┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┘           ┆      ┆   ┆
    │   │  │  │  └──────────────────────────────────────────┘       │   │
    │   │  │  └──────────────────────────────────────────────────┘      │
    │   └──┴───────────────────────────────────────────────────────┘    │
    ┆   ↑ title_standoff_x (pixels)   <┄─ X-axis title ("Attack Vector") ┆
    ┆   ┆  *                                                           ┆
    │   │  ───────────────────────────────────────────────────────┬───┘
    ┆   ┆      tick labels ("Web", "Local")                  ┆        ┆
    ┆   ┆                                                      ┆        ┆
    ┆   ↓ margin.bottom (pixels)                                     ┆
y=0 └───────────────────────────────────────────────────────────────────┘
     x=0                                                      x=1
```

#### Key Components:

1.  **Paper Coordinates**:
    *   The entire figure is normalized to (0,0) bottom-left to (1,1) top-right
    *   All margins and positioning are relative to these coordinates

2.  **Margins** (l, r, t, b):
    *   Specified in pixels
    *   Create padding around the plot area
    *   Ensure content doesn't get cut off

3.  **Plot Area**:
    *   The white space inside the margins
    *   Contains all data visualizations and axes

4.  **Title Standoffs**:
    *   `title_standoff_x`: Distance between x-axis and its title
    *   `title_standoff_y`: Distance between y-axis and its title
    *   Both are measured in pixels from the axis line

#### Common Adjustments:

```python
# Increase left margin for y-axis title
fig.update_layout(
    margin=dict(l=100),  # Increase left margin
    yaxis=dict(
        title="CVE Count",
        title_standoff=15  # Space between y-axis and its title
    ),
    xaxis=dict(
        title="Attack Vector",
        title_standoff=15  # Space between x-axis and its title
    )
)
```

#### Troubleshooting:

*   If titles overlap with tick labels:
    *   Increase the corresponding margin (`margin.l` for y-axis, `margin.b` for x-axis)
    *   Or increase the `title_standoff` value

*   If the plot looks too small:
    *   Decrease margins
    *   Or increase the overall figure size

## Quarterly Report

### Risk score calculation

Risk Score Calculation Methodology

This risk score prioritizes vulnerabilities based on their exploitability characteristics
rather than just severity. The weights are derived from:

1. Industry analysis of exploit prevalence in the wild
2. Microsoft's own security response team prioritization guidelines
3. The relative importance of each factor in automated attacks

Formula:
risk_score = (
    (AV_weight * AV_value) +
    (AC_weight * AC_value) +
    (PR_weight * PR_value) +
    (UI_weight * UI_value)
)

Where:
- AV_weight = 0.35 (Attack Vector has highest impact on exploitability)
- AC_weight = 0.25 (Attack Complexity is second most important)
- PR_weight = 0.20 (Privileges Required)
- UI_weight = 0.20 (User Interaction)

And values are normalized between 0-1 based on CVSS v3 standards
