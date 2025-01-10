import json
import os

# Define input and output paths
input_path = r"C:\Users\emili\PycharmProjects\microsoft_cve_support_report\data\08_reporting\periodic_report_CVE_WEEKLY_v1\json\periodic_report_CVE_WEEKLY_v1_2024_12_24.json"
output_path = r"C:\Users\emili\PycharmProjects\microsoft_cve_support_report\data\08_reporting\periodic_report_CVE_WEEKLY_v1\md\periodic_report_CVE_WEEKLY_v1_2024_12_24.md"

# Keys to extract from each dict
keys_to_extract = ["id", "post_id", "published", "revision", "post_type", "title", "description", "summary", "kb_article_pairs"]

# Load the JSON file
with open(input_path, 'r') as file:
    data = json.load(file)

# Extract relevant section from JSON
section_1_data = data.get("section_1_data", [])

# Create Markdown content
markdown_content = "# PortalFuse Weekly Security Update Report\n\n"
markdown_content += f"**Report Dates:** {data.get('report_start_date')} - {data.get('report_end_date')}\n\n"
markdown_content += f"**Title:** {data.get('title')}\n\n"
markdown_content += "**Description:**\n\n"
markdown_content += f"{data.get('description')}\n\n"

markdown_content += "## MSRC Posts\n\n"

# Iterate through each CVE and extract required keys
for cve in section_1_data:
    cve_data = {key: cve.get(key) for key in keys_to_extract}
    markdown_content += f"### {cve_data.get('post_id')}\n\n"
    markdown_content += f"- **ID:** {cve_data.get('id')}\n"
    markdown_content += f"- **Post ID:** {cve_data.get('post_id')}\n"
    markdown_content += f"- **Published:** {cve_data.get('published')}\n"
    markdown_content += f"- **Revision:** {cve_data.get('revision')}\n"
    markdown_content += f"- **Post Type:** {cve_data.get('post_type')}\n"
    markdown_content += f"- **Title:** {cve_data.get('title')}\n"
    markdown_content += f"- **Description:** {cve_data.get('description')}\n"
    markdown_content += f"- **Summary:**\n  {cve_data.get('summary')}\n\n"
    kb_article_pairs = cve_data.get('kb_article_pairs', [])
    if kb_article_pairs:
        markdown_content += "- **KB Article Pairs:**\n"
        for kb in kb_article_pairs:
            markdown_content += f"  - [{kb}](https://support.microsoft.com/help/{kb})\n"
    markdown_content += "\n"

# Write Markdown content to output file
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as file:
    file.write(markdown_content)

print(f"Markdown file generated at: {output_path}")
