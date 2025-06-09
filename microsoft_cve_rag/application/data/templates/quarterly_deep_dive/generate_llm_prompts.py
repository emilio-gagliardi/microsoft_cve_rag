#!/usr/bin/env python3
"""Generate Jinja2 stub templates for Quarterly Deep Dive LLM prompts."""
import os


def main() -> None:
    """Create llm_prompts directory and stub Jinja templates for sections 1-10."""
    base_dir = os.path.join(
        os.path.dirname(__file__), 'llm_prompts'
    )
    os.makedirs(base_dir, exist_ok=True)

    templates = ['base_prompt.j2']
    for sec in range(1, 11):
        for name in [
            'chart_1_insight', 'chart_2_insight',
            'callout_1', 'callout_2',
            'narrative', 'summary'
        ]:
            templates.append(f'section{sec}_{name}.j2')

    # Populate callout and summary templates with actual Jinja content
    for sec in range(1, 11):
        section_title = f"Section {sec}"
        # Generate insight prompts
        for idx in [1, 2]:
            name = f'section{sec}_chart_{idx}_insight.j2'
            path = os.path.join(base_dir, name)
            content = f"""{{% extends "llm_prompts/base_prompt.j2" %}}

{{# Specific context for this prompt #}}
{{% set section_title = "{section_title}" %}}
{{% set output_format_description = "a concise insight (2-3 sentences) highlighting the main trend or anomaly from chart {idx} of the {section_title} section" %}}

{{% block task_instructions %}}
Analyze the provided "{{{{ section_title }}}}" chart {idx} data. Identify the key trend, pattern, or anomaly, and articulate a concise insight in 2-3 sentences.
{{% endblock task_instructions %}}

{{% block data_context %}}
{{{{ formatted_data | default("Data not available.") }}}}
{{% endblock data_context %}}
"""
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
        # Generate narrative prompt
        name = f'section{sec}_narrative.j2'
        path = os.path.join(base_dir, name)
        content = f"""{{% extends "llm_prompts/base_prompt.j2" %}}

{{# Specific context for this prompt #}}
{{% set section_title = "{section_title}" %}}
{{% set output_format_description = "one or two paragraphs (approx. 3-5 sentences total) providing narrative context for the {section_title} section" %}}

{{% block task_instructions %}}
Write a narrative for the "{{{{ section_title }}}}" section. Provide context, explain the significance of the data, and guide the reader through the key findings of the charts.
{{% endblock task_instructions %}}

{{% block data_context %}}
{{{{ formatted_data | default("Data not available.") }}}}
{{% endblock data_context %}}
"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        # Generate callout prompts
        for idx in [1, 2]:
            name = f'section{sec}_callout_{idx}.j2'
            path = os.path.join(base_dir, name)
            content = f"""{{% extends "llm_prompts/base_prompt.j2" %}}

{{# Specific context for this prompt #}}
{{% set section_title = "{section_title}" %}}
{{% set output_format_description = "a brief callout (1-2 sentences) highlighting a key insight from the {section_title} data, suitable for a callout box" %}}

{{% block task_instructions %}}
Identify a standout insight from the "{{{{ section_title }}}}" data. Craft a concise callout (1-2 sentences) that highlights this insight clearly and engages the reader.
{{% endblock task_instructions %}}

{{% block data_context %}}
{{{{ formatted_data | default("Data not available.") }}}}
{{% endblock data_context %}}
"""
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
        # summary prompt
        name = f'section{sec}_summary.j2'
        path = os.path.join(base_dir, name)
        content = f"""{{% extends "llm_prompts/base_prompt.j2" %}}

{{# Specific context for this prompt #}}
{{% set section_title = "{section_title}" %}}
{{% set output_format_description = "a concise summary (3-4 sentences) of the main takeaways from the {section_title} section" %}}

{{% block task_instructions %}}
Summarize the key findings and trends from the "{{{{ section_title }}}}" section in 3-4 sentences. Provide clear, high-level takeaways that can be included in the report summary.
{{% endblock task_instructions %}}

{{% block data_context %}}
{{{{ formatted_data | default("Data not available.") }}}}
{{% endblock data_context %}}
"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    for tpl in templates:
        path = os.path.join(base_dir, tpl)
        if not os.path.exists(path):
            with open(path, 'w', encoding='utf-8') as f:
                f.write(f"{{# TODO: Stub for {tpl} #}}\n")


if __name__ == '__main__':
    main()
