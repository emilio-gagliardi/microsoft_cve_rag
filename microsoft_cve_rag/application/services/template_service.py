"""Handle Jinja2 template rendering operations."""

import os
import re
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, TemplateNotFound, TemplateSyntaxError

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def strftime_filter(date_str: str, format_str: str) -> str:
    """Convert date string to formatted date string.

    Args:
        date_str: Date string in ISO format
        format_str: Format string for strftime

    Returns:
        Formatted date string
    """
    try:
        if isinstance(date_str, str):
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        elif isinstance(date_str, datetime):
            date = date_str
        else:
            return date_str
        return date.strftime(format_str)
    except (ValueError, TypeError) as e:
        logger.error(f"Error formatting date {date_str}: {e}")
        return date_str


class TemplateService:
    """Service for rendering Jinja2 templates."""

    def __init__(
        self,
        template_dir: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> None:
        """Initialize the template service.

        Args:
            template_dir: Base directory for templates. Defaults to application
                templates directory.
            output_dir: Directory for rendered output. Defaults to reports
                directory.
        """
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.template_dir = template_dir or os.path.join(
            base_dir, "application", "data", "templates"
        )
        self.output_dir = output_dir or os.path.join(
            base_dir, "application", "data", "reports"
        )

        logger.info(f"Template directory: {self.template_dir}")
        logger.info(f"Output directory: {self.output_dir}")

        if not os.path.exists(self.template_dir):
            raise ValueError(f"Template directory not found: {self.template_dir}")

        try:
            self.env = Environment(
                loader=FileSystemLoader([
                    self.template_dir,
                    os.path.join(self.template_dir, "weekly_kb_report")
                ]),
                autoescape=True,
                trim_blocks=True,
                lstrip_blocks=True
            )
            # Add custom filters
            self.env.filters['parse_markdown'] = self._parse_markdown
            self.env.filters['to_json'] = self._to_json
            self.env.filters['strftime'] = strftime_filter
            # Test if we can list templates
            available_templates = self.env.list_templates()
            logger.debug(f"Available templates: {available_templates}")
        except Exception as e:
            logger.error(f"Failed to initialize Jinja environment: {str(e)}")
            raise

    def _parse_markdown(self, text: str) -> str:
        """Parse markdown text to HTML.

        Currently supports:
        - Bold text with ** markers (converted to h4 if at start of line)
        - Bullet lists (- markers)
        - Numbered lists (1. markers)
        - Code blocks (``` markers)
        - Paragraph spacing
        """
        if not text:
            return text

        logger.debug(f"Processing markdown text: {text[:100]}...")

        # First preserve paragraph spacing by converting double newlines
        text = text.replace('\n\n', '</p><p>')

        # Handle code blocks with triple backticks
        def replace_code_blocks(match):
            code = match.group(2).strip()
            lang = match.group(1) if match.group(1) else ''
            return f'<pre><code class="language-{lang}">{code}</code></pre>'

        text = re.sub(r'```(\w+)?\n(.*?)```', replace_code_blocks, text, flags=re.DOTALL)

        # First process headers and split content into sections
        sections = []
        current_section = []
        lines = text.split('\n')

        for line in lines:
            # Check if this is a header line
            header_match = re.match(r'^\s*\*\*([^*]+?)\*\*\s*:?\s*$', line)
            if header_match:
                # If we have content in the current section, save it
                if current_section:
                    sections.append(('content', '\n'.join(current_section)))
                    current_section = []

                # Add the header, removing any trailing colon
                header_text = re.sub(r'\s*:\s*$', '', header_match.group(1))
                sections.append(('header', header_text))
            else:
                current_section.append(line)

        # Add the last section if it has content
        if current_section:
            sections.append(('content', '\n'.join(current_section)))

        # Process each section
        def process_content(content: str) -> str:
            if not content.strip():
                return ''

            # Process lists
            def convert_lists(text: str) -> str:
                lines = text.split('\n')
                result = []
                current_list = []
                in_list = False
                list_type = None

                for line in lines:
                    stripped = line.strip()

                    # Skip empty lines and standalone colons
                    if not stripped or stripped == ':':
                        if in_list and current_list:
                            # End the current list
                            list_class = 'list-disc' if list_type == 'ul' else 'list-decimal'
                            result.append(f'<div class="mb-8"><{list_type} class="{list_class} list-inside space-y-2">')
                            result.extend(current_list)
                            result.append(f'</{list_type}></div>')
                            current_list = []
                            in_list = False
                        continue

                    # Check for bullet points
                    if stripped.startswith('- '):
                        content = stripped[2:].strip()
                        if content:  # Only process non-empty items
                            if not in_list or list_type != 'ul':
                                if in_list and current_list:
                                    # End previous list if it exists
                                    list_class = 'list-disc' if list_type == 'ul' else 'list-decimal'
                                    result.append(f'<div class="mb-8"><{list_type} class="{list_class} list-inside space-y-2">')
                                    result.extend(current_list)
                                    result.append(f'</{list_type}></div>')
                                    current_list = []
                                list_type = 'ul'
                                in_list = True
                            current_list.append(f'<li>{content}</li>')
                            continue

                    # Check for numbered lists
                    num_match = re.match(r'^\d+\.\s+(.+)$', stripped)
                    if num_match:
                        content = num_match.group(1).strip()
                        if content:  # Only process non-empty items
                            if not in_list or list_type != 'ol':
                                if in_list and current_list:
                                    # End previous list if it exists
                                    list_class = 'list-disc' if list_type == 'ul' else 'list-decimal'
                                    result.append(f'<div class="mb-8"><{list_type} class="{list_class} list-inside space-y-2">')
                                    result.extend(current_list)
                                    result.append(f'</{list_type}></div>')
                                    current_list = []
                                list_type = 'ol'
                                in_list = True
                            current_list.append(f'<li>{content}</li>')
                            continue

                    # Not a list item
                    if in_list and current_list:
                        # End the current list
                        list_class = 'list-disc' if list_type == 'ul' else 'list-decimal'
                        result.append(f'<div class="mb-8"><{list_type} class="{list_class} list-inside space-y-2">')
                        result.extend(current_list)
                        result.append(f'</{list_type}></div>')
                        current_list = []
                        in_list = False

                    # Add non-list content
                    result.append(line)

                # Handle any remaining list
                if in_list and current_list:
                    list_class = 'list-disc' if list_type == 'ul' else 'list-decimal'
                    result.append(f'<div class="mb-8"><{list_type} class="{list_class} list-inside space-y-2">')
                    result.extend(current_list)
                    result.append(f'</{list_type}></div>')

                return '\n'.join(result)

            # Process the content
            content = convert_lists(content)

            # Handle any remaining bold text (non-headers)
            content = re.sub(r'\*\*([^*]+?)\*\*', r'<strong>\1</strong>', content)

            return content.strip()

        # Build the final HTML
        html_parts = []
        for section_type, section_content in sections:
            if section_type == 'header':
                html_parts.append(f'<h4 class="text-lg font-medium mb-6">{section_content}</h4>')
            else:
                processed = process_content(section_content)
                if processed:
                    html_parts.append(processed)

        # Join all parts and clean up
        text = '\n'.join(html_parts)

        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)

        logger.info("Final HTML: %s", text[:200])

        return text

    def _to_json(self, value: Any) -> str:
        """Convert a value to JSON string with proper escaping.

        Args:
            value: Any Python object that can be serialized to JSON.

        Returns:
            str: JSON string with proper escaping for HTML and JavaScript.
        """
        return json.dumps(value, ensure_ascii=False)

    def render_kb_report(
        self,
        kb_data: list[Dict[str, Any]],
        report_date: datetime,
        report_title: str
    ) -> str:
        """Render the KB report template with provided data.

        Args:
            kb_data: List of KB article data to render
            report_date: Date of report generation
            report_title: Title of the report

        Returns:
            str: Path to the rendered HTML file

        Raises:
            TemplateNotFound: If template files cannot be found
            TemplateSyntaxError: If there are syntax errors in the template.
            Exception: For other rendering errors
        """
        if not kb_data:
            raise ValueError("KB data is required for report generation.")
        logger.info(f"Rendering template with {len(kb_data)} articles...")
        try:
            logger.info("Loading report template...")
            template = self.env.get_template("weekly_kb_report/report.html.j2")

            # # Create output directory if it doesn't exist
            # report_dir = os.path.join(
            #     self.output_dir,
            #     "weekly_kb_report",
            #     "html"
            # )
            # os.makedirs(report_dir, exist_ok=True)

            # # Generate output filename
            # output_file = os.path.join(
            #     report_dir,
            #     f"kb_report_{report_date.strftime('%Y%m%d')}.html"
            # )
            # report_date_format = '%B %d, %Y'

            # if not report_date or not isinstance(report_date, datetime):
            #     if isinstance(report_date, str):
            #         report_date = datetime.strptime(report_date, report_date_format)
            #     else:
            #         report_date = datetime.now()
            #         logger.warning("Report date not provided or invalid, using current date.")
            # else:
            #     report_date = report_date.strftime(report_date_format)

            # Build CVE data lookup
            all_kb_cve_data = {}
            for article in kb_data:
                if 'kb_id' in article and 'cve_details' in article:
                    all_kb_cve_data[article['kb_id']] = article['cve_details']

            # Render template with data
            html_content = template.render(
                kb_articles=kb_data,
                title=report_title,
                generated_at=report_date,
                all_kb_cve_data=all_kb_cve_data
            )

            # logger.info(f"Writing output to: {output_file}")
            # # Write to file
            # with open(output_file, "w", encoding="utf-8") as f:
            #     f.write(html_content)

            return html_content if html_content else None

        except TemplateNotFound as e:
            logger.error(f"Template not found: {str(e)}")
            logger.error(f"Template search path: {self.template_dir}")
            raise
        except TemplateSyntaxError as e:
            logger.error(f"Template syntax error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error rendering template: {str(e)}")
            logger.error(f"Template dir contents: {os.listdir(self.template_dir)}")
            raise
