"""Service for generating KB report from templates."""
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel


class CVEInfo(BaseModel):
    """CVE information structure."""
    id: str
    description: str
    severity: str


class KBArticle(BaseModel):
    """KB Article structure with all necessary information."""
    id: str
    title: str
    os_version: str
    published_date: str
    url: str
    os_builds: List[str]
    cves: List[CVEInfo]
    new_features: Optional[List[str]] = None
    bug_fixes: Optional[List[str]] = None
    known_issues: Optional[List[str]] = None
    summary: Optional[str] = None


class ReportContext(BaseModel):
    """Complete report context with all necessary data."""
    title: str
    report_date: str
    total_kb_articles: int
    multi_os_updates: int
    known_issues: int
    windows_10_count: int
    windows_11_count: int
    windows_server_count: int
    kb_articles: List[KBArticle]


class ReportGenerator:
    """Service for generating KB reports from templates."""
    
    def __init__(self, template_dir: str):
        """Initialize the report generator with template directory.
        
        Args:
            template_dir: Path to the directory containing Jinja2 templates.
        """
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=True
        )
        
        # Add custom filters
        self.env.filters['tojson'] = self._to_json
    
    def generate_report(self, context: ReportContext, output_path: str) -> None:
        """Generate an HTML report using the provided context.
        
        Args:
            context: ReportContext containing all necessary data.
            output_path: Path where the generated HTML should be saved.
        """
        template = self.env.get_template('report.html.j2')
        html_content = template.render(**context.dict())
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html_content, encoding='utf-8')
    
    @staticmethod
    def _to_json(value: Any) -> str:
        """Convert a value to a JSON string for use in JavaScript.
        
        Args:
            value: Any Python value that needs to be converted to JSON.
        
        Returns:
            JSON string representation of the value.
        """
        import json
        return json.dumps(value)
