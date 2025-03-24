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

# Lessons Learned

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
