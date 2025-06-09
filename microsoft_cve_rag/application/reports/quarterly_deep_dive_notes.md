# Quarterly Deep Dive Report Generator - Data Flow and Context Management

## Key Data Structures

### 1. `generated_charts_context: Dict[str, ChartExport]`
- **Purpose**: Stores metadata and paths for successfully generated chart images and JSON files
- **Usage**: Primarily used for rendering the final HTML/Markdown output

### 2. `all_chart_data_for_llm: Dict[str, str]`
- **Purpose**: Stores formatted data (as stringified tables) for all successfully generated charts
- **Usage**: Provides chart data to LLM prompts for analysis and insight generation

### 3. `llm_section_content: Dict[str, Dict[str, str]]`
- **Structure**: Nested dictionary with format `{section_key: {prompt_identifier: llm_generated_text}}`
- **Purpose**: Stores LLM-generated text outputs including:
  - Chart insights
  - Callouts
  - Section narratives
- **Organization**: Content is organized by report section and content type

### 4. `base_llm_context_from_python: Dict[str, Any]`
- **Purpose**: Contains foundational data common to all LLM prompts
- **Typical Contents**:
  - Report name
  - Date ranges
  - Quarter information
  - Other global report metadata

## Report Generation Process Flow

The generation follows a hierarchical approach, processing each section defined in the report configuration (`config.sections`).

### 1. Overall Process
1. Iterate through each section in the configuration
2. For each section, process in this order:
   - Generate charts
   - Generate chart insights
   - Process callouts
   - Generate section narrative (if applicable)

### 2. Chart Generation
- **Process**:
  1. For each section, iterate through defined charts
  2. Generate chart data (`chart_df`)
  3. Create visualization (e.g., Altair chart)
  4. Export to image/JSON formats
  5. Store paths and metadata in `generated_charts_context`
  6. Format `chart_df` as string (e.g., markdown table)
  7. Store in `all_chart_data_for_llm[chart_key]`

### 3. Chart Insight Generation
- **Timing**: Generated immediately after corresponding chart is created
- **Context Available**:
  - `base_llm_context_from_python`
  - Specific chart data from `all_chart_data_for_llm[current_chart_key]`
  - Chart metadata (title, description from `chart_config`)
- **Storage**: Results stored in `llm_section_content[current_section_key]["{current_chart_key}_insight"]`

### 4. Callout Generation
- **Timing**: Processed after all charts and insights for a section
- **Context Available**:
  - `base_llm_context_from_python`
  - Report metrics (`calculated_metrics`)
  - Relevant chart data from `all_chart_data_for_llm`
  - Chart insights from `llm_section_content`
- **Storage**: Results stored in `llm_section_content[current_section_key][callout_key]`

### 5. Section Narrative Generation
- **Timing**: Generated after all other section components (charts, insights, callouts)
- **Context Available**:
  - All section-specific chart data
  - All section-specific insights
  - All section callouts
  - Global metrics and metadata
- **Storage**: Results typically stored in `llm_section_content[current_section_key]["narrative"]`

## Data Flow Summary

1. **Progressive Context Building**:
   - Chart data is added to `all_chart_data_for_llm` as it's generated
   - LLM outputs are stored in `llm_section_content` for reference by subsequent prompts

2. **Context Isolation**:
   - Each section's components have access to relevant context
   - Global context is always available
   - Later components can reference earlier outputs

3. **Dependency Management**:
   - Charts must be generated before their insights
   - Section narratives have access to all section components
   - Global summary (if any) has access to all report data

## Summary Answers to Your Questions

### 1. "if all charts are generated, then all the chart insights before processing the callout prompts."
Not exactly globally. It's more likely a sequential process within each section:
- Charts for the section are generated one by one.
- As each chart is made, its data is stored in `all_chart_data_for_llm`.
- Immediately after a chart is made and its data stored, if it has an associated insight prompt, that insight is generated using that specific chart's data.
- Once all charts and their insights for the current section are done, then the callouts for that section are processed.
So, it's not: (all charts for report) -> (all insights for report) -> (all callouts for report). It's more granular and section-oriented.

### 2. "Does the narrative data context get all chart data tables and chart insights and callouts?"
For a section-specific narrative: Yes, it would receive all chart data tables, chart insights, and callouts that belong to that specific section, plus global metrics.
For a report-wide summary narrative (if one exists): Yes, it would be the culmination and would receive everything: all chart data tables from all sections, all insights from all sections, all callouts from all sections, and all previously generated section-specific narratives.

The system is designed to build context progressively. `all_chart_data_for_llm` acts as a growing repository of chart data tables, and `llm_section_content` collects all textual LLM outputs, which can then be selectively passed to subsequent LLM prompts that require broader context.

This explanation is based on the provided code snippet and common best practices for generating complex, multi-part reports. The exact details depend on the loop structure and context assembly logic that follows the snippet you've shown.
