import asyncio
import html
import inspect
import json
import logging
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import markdown
import markdownify
import numpy as np
import pandas as pd
import plotly.express as px  # Often useful even if using go
import plotly.graph_objects as go
import plotly.io as pio
import tiktoken
from bs4 import BeautifulSoup
from faker import Faker  # For synthetic data
from jinja2 import Environment as JinjaEnvironment  # Type hint
from pydantic import BaseModel, Field

try:
    from application.app_utils import APP_DIR, DATA_DIR, REPORTS_DIR

    TEMPLATES_BASE_DIR = DATA_DIR / "templates"
except ImportError:
    logging.warning("Unable to import app_utils. Using relative paths.")
    APP_DIR = Path("microsoft_cve_rag/application")
    REPORTS_DIR = APP_DIR / "reports"
    DATA_DIR = APP_DIR / "data"
    TEMPLATES_BASE_DIR = DATA_DIR / "templates"
from application.etl.NVDDataExtractor import NVDDataExtractor
from application.reports.parameters.card_properties import (
    CARD_STYLES,
    CVE_CATEGORY_COLOR_MAP,
    STAT_CARD_LABELS,
    STAT_CARD_SUBTITLES,
)
from application.reports.parameters.cwe_lookup import (
    CWE_DETAILS,
    CWE_ROOT_CAUSE,
)
from application.reports.parameters.icon_map import (
    ICON_SVG_STRINGS,
    STAT_CARD_ICONS,
)
from application.reports.parameters.system_prompts import SYSTEM_PROMPTS
from application.services.chat_service import LLMClient
from application.services.metrics_service import DuckDBMetricsService

get_nvd_columns = NVDDataExtractor.get_all_possible_columns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class ReportConfig(BaseModel):
    """
    Configuration defining relative paths and names for a specific report type.
    Provides methods to resolve these into absolute paths based on global base directories.
    """

    report_name: str = Field(
        default="quarterly_deep_dive",
        description=(
            "Unique identifier and subdirectory name for this report type"
            " under the global reports directory."
        ),
    )
    report_title: str = Field(
        default="Quarterly Microsoft CVE Deep Dive",
        description="Display title for the report.",
    )

    # --- Relative Directory Names (within the specific report's output) ---
    output_html_subdir: str = Field(
        default="html",
        description=(
            "Subdirectory name within the report's base output directory for"
            " HTML."
        ),
    )
    output_data_subdir: str = Field(
        default="data",
        description=(
            "Subdirectory name within the report's base output directory for"
            " storing chart JSON data."
        ),
    )
    output_css_subdir: str = Field(
        default="css",
        description=(
            "Subdirectory name within the report's base output directory for"
            " CSS files."
        ),
    )
    output_js_subdir: str = Field(
        default="js",
        description=(
            "Subdirectory name within the report's base output directory for"
            " JavaScript files."
        ),
    )
    output_image_subdir: str = Field(
        default="images",
        description=(
            "Subdirectory name within the report's base output directory for"
            " static images (if any)."
        ),
    )
    output_markdown_subdir: str = Field(
        default="markdown",
        description=(
            "Subdirectory name within the report's base output directory for"
            " Markdown files."
        ),
    )
    # --- Relative Template Directory Name (within the global templates base) ---
    template_subdir: str = Field(
        default="quarterly_deep_dive",
        description=(
            "Subdirectory name within the global application templates"
            " directory containing Jinja templates for this report."
        ),
    )
    template_html_filename: str = Field(
        default="report.html",
        description="Filename for the main Jinja HTML template file.",
    )
    # --- Output Filenames ---
    output_css_filename: str = Field(
        default="QuarterlyReportStylesheet.css",
        description="Filename for the primary CSS stylesheet.",
    )
    output_js_filename: str = Field(
        default="QuarterlyReport.js",
        description="Filename for the primary JavaScript file.",
    )

    # --- Methods to Resolve Absolute Paths ---

    def get_report_base_dir(self, global_reports_dir: Path) -> Path:
        """Returns the absolute path to the base directory for this specific report instance."""
        if not global_reports_dir or not isinstance(global_reports_dir, Path):
            raise ValueError("global_reports_dir must be a valid Path object.")
        return global_reports_dir / self.report_name

    def get_html_output_dir(self, global_reports_dir: Path) -> Path:
        """Returns the absolute path to the HTML assets directory for this report instance."""
        return (
            self.get_report_base_dir(global_reports_dir)
            / self.output_html_subdir
        )

    def get_data_output_dir(self, global_reports_dir: Path) -> Path:
        """Returns the absolute path to the chart data directory."""
        return (
            self.get_html_output_dir(global_reports_dir)
            / self.output_data_subdir
        )

    def get_css_output_dir(self, global_reports_dir: Path) -> Path:
        """Returns the absolute path to the CSS directory."""
        return (
            self.get_html_output_dir(global_reports_dir)
            / self.output_css_subdir
        )

    def get_js_output_dir(self, global_reports_dir: Path) -> Path:
        """Returns the absolute path to the JS directory."""
        return (
            self.get_html_output_dir(global_reports_dir)
            / self.output_js_subdir
        )

    def get_image_output_dir(self, global_reports_dir: Path) -> Path:
        """Returns the absolute path to the image directory."""
        return (
            self.get_html_output_dir(global_reports_dir)
            / self.output_image_subdir
        )

    def get_markdown_output_dir(self, global_reports_dir: Path) -> Path:
        """Returns the absolute path to the Markdown output directory."""
        return (
            self.get_report_base_dir(global_reports_dir)
            / self.output_markdown_subdir
        )

    def get_css_filepath(self, global_reports_dir: Path) -> Path:
        """Returns the absolute path to the output CSS file."""
        return (
            self.get_css_output_dir(global_reports_dir)
            / self.output_css_filename
        )

    def get_js_filepath(self, global_reports_dir: Path) -> Path:
        """Returns the absolute path to the output JS file."""
        return (
            self.get_js_output_dir(global_reports_dir)
            / self.output_js_filename
        )

    def get_template_dir(self, global_templates_dir: Path) -> Path:
        """Returns the absolute path to the Jinja template directory for this report."""
        if not global_templates_dir or not isinstance(
            global_templates_dir, Path
        ):
            raise ValueError(
                "global_templates_dir must be a valid Path object."
            )
        return global_templates_dir / self.template_subdir

    def get_template_filepath(self, global_templates_dir: Path) -> Path:
        """Returns the absolute path to the main Jinja template file."""
        return (
            self.get_template_dir(global_templates_dir)
            / self.template_html_filename
        )


class GeneratedReportAssets(BaseModel):
    """Holds the paths to all assets generated for a single report run."""

    report_config: ReportConfig
    base_directory: Path  # The root directory for this specific report run
    html_file: Path
    markdown_file: Path
    css_file: Path
    js_file: Path
    chart_data_files: List[Path] = Field(default_factory=list)
    image_files: List[Path] = Field(default_factory=list)
    calculated_metrics: Dict[str, Any] = Field(default_factory=dict)

    def get_html_file(self) -> Path:
        return self.html_file

    def get_markdown_file(self) -> Path:
        return self.markdown_file

    def get_chart_data_files(self) -> List[Path]:
        return self.chart_data_files

    def get_image_files(self) -> List[Path]:
        return self.image_files

    def get_static_assets(self) -> List[Path]:
        """Returns paths to CSS and JS files."""
        return [self.css_file, self.js_file]

    def get_all_files(self) -> List[Path]:
        """Returns a list of all generated file paths."""
        all_files = [
            self.html_file,
            self.markdown_file,
            self.css_file,
            self.js_file,
            *self.chart_data_files,
            *self.image_files,
        ]
        return [
            f for f in all_files if f is not None
        ]  # Filter out potential Nones


class ChartExport(BaseModel):
    chart_id: str
    caption: Optional[str] = None


class ReportContext(BaseModel):
    # Data passed directly to the Jinja template
    report_title: str
    generation_date: str
    start_date: datetime  # Add dates for potential display
    end_date: datetime
    config: ReportConfig  # Pass config for potential use in template
    charts: Dict[str, ChartExport] = Field(default_factory=dict)
    llm_insights: Dict[str, str] = Field(default_factory=dict)
    appendix_tables: Dict[str, Dict[str, str]] = Field(
        default_factory=dict
    )  # Each entry: {'title': str, 'html': str}
    report_structure: Dict[str, Any] = Field(default_factory=dict)
    section_stats_data: Dict[str, List[Dict]] = Field(default_factory=dict)


class QuarterlyDeepDiveReportGenerator:

    REPORT_STRUCTURE = {
        "report_title": "Quarterly Microsoft CVE Deep Dive",
        "explicit_sections": {
            "executive_summary": {
                "id_html": "executive-summary",
                "heading": "Executive Summary & Key Statistics",
                "prompts": {
                    "executive_summary": "executive_summary.j2",
                },
                "charts": [],
                "tables": [],
                "stats_definitions": [],
            },
            "toc": {
                "id_html": "table-of-contents",
                "heading": "Table of Contents",
                "prompts": {},
                "charts": [],
                "tables": [],
                "stats_definitions": [],
            },
            "report_conclusion": {
                "id_html": "report-conclusion",
                "heading": "Conclusion",
                "prompts": {
                    "report_conclusion": "report_conclusion.j2",
                },
                "charts": [],
                "tables": [],
                "stats_definitions": [],
            },
            "appendix": {
                "id_html": "appendix",
                "heading": "Appendix",
                "prompts": {},
                "charts": [],
                "tables": [
                    {
                        "id": "appendix_cve_list",
                        "data_prep_key": "appendix_cve_list",
                        "table_gen_key": "appendix_cve_list",
                        "caption": (
                            "Table A1. CVEs Included in This Report"
                            " (Windows-Specific)"
                        ),
                        "row_click_config": {
                            "enabled": True,
                            "id_column": "cve_id",
                            "url_template": "https://msrc.microsoft.com/update-guide/vulnerability/{cve_id}",
                        },
                        "columns": {
                            'cve_id': 'CVE ID',
                            'title': 'Title',
                            'cve_category': 'CVE Category',
                            'severity_type': 'Severity',
                            'cvss_score': 'CVSS Score',
                            'published': 'Published',
                        },
                    },
                    {
                        "id": "appendix_outlier_list",
                        "data_prep_key": "appendix_outlier_list",
                        "table_gen_key": "appendix_outlier_list",
                        "caption": (
                            "Table A2. CVEs with CVSS scores that are outliers"
                            " compared to their vulnerability category."
                        ),
                        "row_click_config": {
                            "enabled": True,
                            "id_column": "cve_id",
                            "url_template": "https://msrc.microsoft.com/update-guide/vulnerability/{cve_id}",
                        },
                        "columns": {
                            'outlier_type': "Outlier Type",
                            'cvss_score': 'CVSS Score',
                            'cve_id': 'CVE ID',
                            'title': 'Title',
                            'cve_category': 'CVE Category',
                            'severity_type': 'Severity',
                            'published': 'Published',
                        },
                    },
                    {
                        "id": "appendix_all_metrics",
                        "data_prep_key": "appendix_all_metrics",
                        "table_gen_key": "appendix_all_metrics",
                        "caption": "Table A3. All Calculated Metrics",
                        "columns": {
                            'metric_id': 'Metric ID',
                            'metric_name': 'Metric Name',
                            'metric_value': 'Metric Value',
                            'metric_unit': 'Metric Unit',
                        },
                    },
                ],
            },
        },
        "looped_sections": {
            "section_3_cve_landscape": {
                "id_html": (
                    "section-3-landscape-overview"
                ),  # Updated for TOC link
                "heading": (
                    "Quarterly CVE Landscape: Volume, Severity, and"
                    " Categorization"
                ),
                "section_blurb": (
                    "Explore the quarter's CVE landscape, from the volume of"
                    " new and updated threats to their severity and"
                    " categorization. Key metrics and charts reveal overall"
                    " trends in vulnerability reporting and classification."
                ),
                "prompts": {
                    "narrative": "section3_narrative.j2",
                    "callout_1": "section3_callout_1.j2",
                    "callout_2": "section3_callout_2.j2",
                    "callout_3": "section3_callout_3.j2",
                    "chart_insight_1": (
                        "section3_chart_1_insight.j2"
                    ),  # For new_vs_updated_monthly
                    "chart_insight_2": (
                        "section3_chart_2_insight.j2"
                    ),  # For volume_by_severity_monthly
                    "chart_insight_3": (
                        "section3_chart_3_insight.j2"
                    ),  # For overall_category_distribution_chart
                    "section_summary": "section3_summary.j2",
                },
                "charts": [
                    {
                        "id": "new_vs_updated_monthly",
                        "data_prep_key": "new_vs_updated_monthly",
                        "figure_gen_key": "new_vs_updated_monthly",
                        "title": "Monthly New vs. Updated CVEs",
                        "caption": (
                            "Figure 1. Trend of New vs. Updated CVEs Published"
                            " Monthly."
                        ),
                        "insight_prompt": "section3_chart_1_insight.j2",
                    },
                    {
                        "id": "volume_by_severity_monthly",
                        "data_prep_key": "volume_by_severity_monthly",
                        "figure_gen_key": "volume_by_severity_monthly",
                        "title": "Monthly CVE Volume by Severity",
                        "caption": (
                            "Figure 2. Monthly CVE Volume by Assessed"
                            " Severity."
                        ),
                        "insight_prompt": "section3_chart_2_insight.j2",
                    },
                    {
                        "id": "overall_category_distribution_chart",
                        "data_prep_key": "overall_category_distribution_chart",
                        "figure_gen_key": (
                            "overall_category_distribution_chart"
                        ),
                        "title": "Overall Vulnerability Category Distribution",
                        "caption": (
                            "Figure 3. Distribution of CVEs by Vulnerability"
                            " Category for the Quarter."
                        ),
                        "insight_prompt": "section3_chart_3_insight.j2",
                    },
                ],
                "callouts_config": [
                    {
                        "key_suffix": "1",
                        "position": "right",
                        "style": "info",
                        "title": "",
                    },
                    {
                        "key_suffix": "2",
                        "position": "left",
                        "style": "info",
                        "title": "",
                    },
                    {
                        "key_suffix": "3",
                        "position": "right",
                        "style": "info",
                        "title": "",
                    },
                ],
                "stats_definitions": [
                    {
                        "metric_id": "total_cves",
                        "card_style": "total_cves_card",
                    },
                    {
                        "metric_id": "critical_count",
                        "card_style": "critical_count_card",
                    },
                    {
                        "metric_id": "important_count",
                        "card_style": "important_count_card",
                    },
                    {
                        "metric_id": "critical_high_pct",
                        "card_style": "critical_high_pct_card",
                    },
                    {
                        "metric_id": "top_impact_type",
                        "card_style": "top_impact_type_card",
                    },
                ],
            },
            "section_4_attack_surface": {
                "id_html": "section-4-attack-surface",
                "heading": "Attack Surface Analysis: Exploitability & Access",
                "section_blurb": (
                    "Analyze the evolving attack surface by examining"
                    " vulnerability exploitability, including attack vectors,"
                    " privilege requirements, user interaction, and overall"
                    " complexity. Discover how these factors shape the risk"
                    " profile and highlight common exploitation pathways."
                ),
                "prompts": {
                    "narrative": "section4_narrative.j2",
                    "callout_1": "section4_callout_1.j2",
                    "callout_2": "section4_callout_2.j2",
                    "callout_3": "section4_callout_3.j2",
                    "chart_insight_1": (
                        "section4_chart_1_insight.j2"
                    ),  # For attack_vectors_by_privileges_chart
                    "chart_insight_2": (
                        "section4_chart_2_insight.j2"
                    ),  # For interaction_vs_complexity_chart
                    "chart_insight_3": (
                        "section4_chart_3_insight.j2"
                    ),  # For overall_attack_vector_dist
                    "chart_insight_4": (
                        "section4_chart_4_insight.j2"
                    ),  # For cat_av_pr_chart
                    "chart_insight_5": (
                        "section4_chart_5_insight.j2"
                    ),  # For cat_ui_ac_chart
                    "section_summary": "section4_summary.j2",
                },
                "charts": [
                    {
                        "id": "overall_attack_vector_dist",
                        "data_prep_key": "overall_attack_vector_dist",
                        "figure_gen_key": "overall_attack_vector_dist",
                        "title": "Distribution of Attack Vectors",
                        "caption": "Figure 4. Distribution of Attack Vectors.",
                        "insight_prompt": "section4_chart_1_insight.j2",
                    },
                    {
                        "id": "attack_vectors_by_privileges_chart",
                        "data_prep_key": "attack_vectors_by_privileges_chart",
                        "figure_gen_key": "attack_vectors_by_privileges_chart",
                        "title": "Attack Vectors by Privileges Required",
                        "caption": (
                            "Figure 5. Breakdown of Attack Vectors by"
                            " Privilege Requirements."
                        ),
                        "insight_prompt": "section4_chart_2_insight.j2",
                    },
                    {
                        "id": "interaction_vs_complexity_chart",
                        "data_prep_key": "interaction_vs_complexity_chart",
                        "figure_gen_key": "interaction_vs_complexity_chart",
                        "title": "User Interaction vs. Attack Complexity",
                        "caption": (
                            "Figure 6. CVE Count by User Interaction and"
                            " Attack Complexity."
                        ),
                        "insight_prompt": "section4_chart_3_insight.j2",
                    },
                    {
                        "id": "cat_av_pr_chart",
                        "data_prep_key": "cat_av_pr_chart",
                        "figure_gen_key": "cat_av_pr_chart",
                        "title": (
                            "Exploitability: Category vs. Attack Vector &"
                            " Privileges"
                        ),
                        "caption": (
                            "Figure 7. Privilege Requirements for Attack"
                            " Vectors within each Vulnerability Category."
                        ),
                        "insight_prompt": "section4_chart_4_insight.j2",
                    },
                    {
                        "id": "cat_ui_ac_chart",
                        "data_prep_key": "cat_ui_ac_chart",
                        "figure_gen_key": "cat_ui_ac_chart",
                        "title": (
                            "Exploitability: Category vs. User Interaction &"
                            " Complexity"
                        ),
                        "caption": (
                            "Figure 8. Attack Complexity by User Interaction"
                            " within each Vulnerability Category."
                        ),
                        "insight_prompt": "section4_chart_5_insight.j2",
                    },
                ],
                "callouts_config": [
                    {
                        "key_suffix": "1",
                        "position": "right",
                        "style": "info",
                        "title": "",
                    },
                    {
                        "key_suffix": "2",
                        "position": "left",
                        "style": "info",
                        "title": "",
                    },
                    {
                        "key_suffix": "3",
                        "position": "right",
                        "style": "info",
                        "title": "",
                    },
                ],
                "stats_definitions": [
                    {
                        "metric_id": "network_vector_pct",
                        "card_style": "network_vector_pct_card",
                    },
                    {
                        "metric_id": "adjacent_vector_pct",
                        "card_style": "adjacent_vector_pct_card",
                    },
                    {
                        "metric_id": "local_vector_pct",
                        "card_style": "local_vector_pct_card",
                    },
                    {
                        "metric_id": "physical_vector_pct",
                        "card_style": "physical_vector_pct_card",
                    },
                    {
                        "metric_id": "user_action_req_pct",
                        "card_style": "user_action_req_pct_card",
                    },
                    {
                        "metric_id": "low_privs_required_pct",
                        "card_style": "low_privs_required_pct_card",
                    },
                    {
                        "metric_id": "high_privs_required_pct",
                        "card_style": "high_privs_required_pct_card",
                    },
                    {
                        "metric_id": "low_attack_comp_pct",
                        "card_style": "low_attack_comp_pct_card",
                    },
                    {
                        "metric_id": "high_attack_comp_pct",
                        "card_style": "high_attack_comp_pct_card",
                    },
                    {
                        "metric_id": "worst_case_cves",
                        "card_style": "worst_case_cves_card",
                    },
                    # {"metric_id": "exploited_pct_kev", "card_style": "exploited_pct_kev_card"},
                ],
            },
            "section-5-cvss-weakness": {
                "id_html": "section-5-cvss-weakness",
                "heading": (
                    "Deep Dive: CVSS Insights & Prioritization Hotspots"
                ),
                "section_blurb": (
                    "Gain deeper insights into CVSS scores and their"
                    " distribution across vulnerability categories,"
                    " identifying common weaknesses (CWEs) linked to high-risk"
                    " CVEs. Pinpoint prioritization hotspots by understanding"
                    " the correlation between CVSS metrics and underlying"
                    " vulnerability types."
                ),
                "prompts": {
                    "narrative": "section5_narrative.j2",
                    "callout_1": "section5_callout_1.j2",
                    "callout_2": "section5_callout_2.j2",
                    "callout_3": "section5_callout_3.j2",
                    "chart_insight_1": (
                        "section5_chart_1_insight.j2"
                    ),  # For cvss_score_by_category_chart
                    "chart_insight_2": (
                        "section5_chart_2_insight.j2"
                    ),  # For top_cwe_in_high_risk_cves_chart
                    "chart_insight_3": (
                        "section5_chart_3_insight.j2"
                    ),  # For top_cwe_av_cvss_chart
                    "section_summary": "section5_summary.j2",
                },
                "charts": [
                    {
                        "id": "cvss_score_by_category_chart",
                        "data_prep_key": "cvss_score_by_category_chart",
                        "figure_gen_key": "cvss_score_by_category_chart",
                        "title": (
                            "CVSS Score Distribution by Vulnerability Category"
                        ),
                        "caption": (
                            "Figure 9. Box Plots of CVSS Scores per Category."
                        ),
                        "insight_prompt": "section5_chart_1_insight.j2",
                    },
                    {
                        "id": "top_cwe_av_cvss_chart",
                        "data_prep_key": "top_cwe_av_cvss_chart",
                        "figure_gen_key": "top_cwe_av_cvss_chart",
                        "title": (
                            "CVSS Distribution by Attack Vector for Top CWEs"
                        ),
                        "caption": (
                            "Figure 10. CVSS Distribution by Attack Vector for"
                            " Top CWEs."
                        ),
                        "insight_prompt": "section5_chart_2_insight.j2",
                    },
                ],
                "callouts_config": [
                    {
                        "key_suffix": "1",
                        "position": "right",
                        "style": "info",
                        "title": "",
                    },
                    {
                        "key_suffix": "2",
                        "position": "left",
                        "style": "info",
                        "title": "",
                    },
                    {
                        "key_suffix": "3",
                        "position": "right",
                        "style": "info",
                        "title": "",
                    },
                ],
                "stats_definitions": [
                    {
                        "metric_id": "median_cvss",
                        "card_style": "median_cvss_card",
                    },
                    {"metric_id": "p90_cvss", "card_style": "p90_cvss_card"},
                    {
                        "metric_id": "highest_cvss",
                        "card_style": "highest_cvss_card",
                    },
                    {
                        "metric_id": "top_cwe_id",
                        "card_style": "top_cwe_id_card",
                    },
                    {
                        "metric_id": "top_3_cwe_coverage_pct",
                        "card_style": "top_3_cwe_coverage_pct_card",
                    },
                    {
                        "metric_id": "memory_safety_vs_logic_bugs_pct",
                        "card_style": "memory_safety_vs_logic_bugs_pct_card",
                    },
                ],
            },
            "section-6-product-timeliness": {
                "id_html": (
                    "section-6-product-timeliness"
                ),  # Updated for TOC link
                "heading": "Product Impact & Remediation Timeliness",
                "section_blurb": (
                    "Assess the impact of vulnerabilities on specific products"
                    " and the timeliness of remediation, highlighting the most"
                    " affected software by CVE criticality. Understand"
                    " patching and disclosure timelines through an analysis of"
                    " NVD publication delays."
                ),
                "prompts": {
                    "narrative": "section6_narrative.j2",
                    "callout_1": "section6_callout_1.j2",
                    "callout_2": "section6_callout_2.j2",
                    "callout_3": "section6_callout_3.j2",
                    "chart_insight_1": (
                        "section6_chart_1_insight.j2"
                    ),  # For top_windows_versions_by_criticality_chart
                    "chart_insight_2": (
                        "section6_chart_2_insight.j2"
                    ),  # For days_to_nvd_by_severity_boxplots
                    "section_summary": "section6_summary.j2",
                },
                "charts": [
                    {
                        "id": "top_windows_versions_by_criticality_chart",
                        "data_prep_key": (
                            "top_windows_versions_by_criticality_chart"
                        ),
                        "figure_gen_key": (
                            "top_top_windows_versions_by_criticality_chart"
                        ),
                        "title": "Top Affected Products by CVE Criticality",
                        "caption": (
                            "Figure 11. Products with Highest Counts of"
                            " Critical/High CVEs."
                        ),
                        "insight_prompt": "section6_chart_1_insight.j2",
                    },
                    {
                        "id": "days_to_nvd_by_severity_boxplots",
                        "data_prep_key": "days_to_nvd_by_severity_boxplots",
                        "figure_gen_key": (
                            "days_to_nvd_by_severity_boxplots_faceted"
                        ),
                        "title": "NVD Publication Delay by CVE Severity",
                        "caption": (
                            "Figure 12. Days from Vendor Patch to NVD"
                            " Publication, by Severity."
                        ),
                        "insight_prompt": "section6_chart_2_insight.j2",
                    },
                ],
                "callouts_config": [
                    {
                        "key_suffix": "1",
                        "position": "right",
                        "style": "info",
                        "title": "",
                    },
                    {
                        "key_suffix": "2",
                        "position": "left",
                        "style": "info",
                        "title": "",
                    },
                    {
                        "key_suffix": "3",
                        "position": "right",
                        "style": "info",
                        "title": "",
                    },
                ],
                "stats_definitions": [
                    {
                        "metric_id": "most_affected_product",
                        "card_style": "most_affected_product_card",
                    },
                    {
                        "metric_id": "median_days_to_patch",
                        "card_style": "median_days_to_patch_card",
                    },
                    {
                        "metric_id": "patched_le_7d_pct",
                        "card_style": "patched_le_7d_pct_card",
                    },
                    {
                        "metric_id": "patched_le_30d_pct",
                        "card_style": "patched_le_30d_pct_card",
                    },
                    # {"metric_id": "zero_day_count", "card_style": "zero_day_count_card"},  # no ETL pipeline for this metric yet
                ],
            },
            "section-7-critical-spotlight": {
                "id_html": (
                    "section-7-critical-spotlight"
                ),  # Updated for TOC link
                "heading": (
                    "Spotlight: This Quarter's Most Critical Vulnerabilities"
                ),
                "section_blurb": (
                    "Spotlight the quarter's most critical vulnerabilities,"
                    " offering detailed insights and CVSS scores for"
                    " high-impact threats. Visualizations help uncover"
                    " distinct patterns and shared characteristics among these"
                    " significant issues, aiding in a deeper understanding of"
                    " their nature."
                ),
                "prompts": {
                    "narrative": "section7_narrative.j2",
                    "callout_1": "section7_callout_1.j2",
                    "callout_2": "section7_callout_2.j2",
                    "callout_3": "section7_callout_3.j2",
                    "chart_insight_1": (
                        "section7_chart_1_insight.j2"
                    ),  # For spotlight_cves_cvss_scores_bar
                    "chart_insight_2": (
                        "section7_chart_2_insight.j2"
                    ),  # For tsne_cve_clusters_plot
                    "section_summary": "section7_summary.j2",
                },
                "charts": [
                    {
                        "id": "spotlight_cves_cvss_chart",
                        "data_prep_key": "spotlight_cves_cvss_chart",
                        "figure_gen_key": "spotlight_cves_cvss_chart",
                        "title": (
                            "CVSS Scores of This Quarter's Most Critical CVEs"
                        ),
                        "caption": (
                            "Figure 13. CVSS scores of this quarter's most"
                            " critical CVEs."
                        ),
                        "insight_prompt": "section7_chart_1_insight.j2",
                    },
                    {
                        "id": "tsne_cve_profile_chart",
                        "data_prep_key": "tsne_cve_profile_chart",
                        "figure_gen_key": "tsne_cve_profile_chart",
                        "title": "CVE Profile Clusters (t-SNE Visualization)",
                        "caption": (
                            "Figure 14. t-SNE projection of CVEs showing"
                            " distinct vulnerability profiles."
                        ),
                        "insight_prompt": "section7_chart_2_insight.j2",
                    },
                ],
                "tables": [{
                    "id": "spotlight_cve_table",
                    "data_prep_key": "spotlight_cves_details_table",
                    "table_gen_key": "spotlight_cve_table_html",
                    "caption": (
                        "Table S1. Details of Spotlighted Critical"
                        " Vulnerabilities."
                    ),
                    "row_click_config": {
                        "column_name": "CVE ID",
                        "url_template": "https://cve.mitre.org/cgi-bin/cvename.cgi?name={CVE_ID}",
                        "tooltip": "View CVE Details",
                    },
                }],
                "callouts_config": [
                    {
                        "key_suffix": "1",
                        "position": "right",
                        "style": "info",
                        "title": "",
                    },
                    {
                        "key_suffix": "2",
                        "position": "left",
                        "style": "info",
                        "title": "",
                    },
                    {
                        "key_suffix": "3",
                        "position": "right",
                        "style": "info",
                        "title": "",
                    },
                ],
                "stats_definitions": [
                    {
                        "metric_id": "worst_case_cves",
                        "card_style": "worst_case_cves_card",
                    },
                ],
            },
        },
    }

    LLM_TASK_MODELS = {
        "narrative": (
            "gpt-4.1-mini"
        ),  # cheap | fast | middle smart @ $0.40/1M tokens - 1M context window
        "callout": (
            "gpt-4o-mini"
        ),  # cheap | fast | low smart @ $0.15/1M tokens - 128K context window
        "chart_insight": (
            "gpt-4o"
        ),  # cheap | fast | smart @ $0.40/1M tokens - 1M context window
        "section_summary": (
            "o3-mini"
        ),  # expensive | slow | very smart @ $1.10/1M tokens - 200K context window
        "executive_summary": (
            "o3-mini"
        ),  # expensive | slow | very smart @ $1.10/1M tokens - 200K context window
        "report_conclusion": (
            "o3-mini"
        ),  # expensive | slow | very smart @ $1.10/1M tokens - 200K context window
        "default": (  # cheap | fast | middle smart @ $0.40/1M tokens - 1M context window
            "gpt-4.1-mini"
        ),
    }

    def __init__(
        self,
        jinja_env: JinjaEnvironment,
        llm_client: Optional[LLMClient] = None,
        global_reports_dir: Optional[Path] = None,
        global_templates_dir: Optional[Path] = None,
    ):
        """Initializes the generator with essential dependencies."""
        if not global_reports_dir or not global_reports_dir.is_dir():
            raise ValueError(
                f"global_reports_dir ('{global_reports_dir}') must be a valid"
                " Path object and directory."
            )
        if not global_templates_dir or not global_templates_dir.is_dir():
            raise ValueError(
                f"global_templates_dir ('{global_templates_dir}') must be a"
                " valid Path object and directory."
            )
        # Ensure APP_DIR is also initialized before using it here
        if APP_DIR is None or not APP_DIR.is_dir():
            raise ValueError(
                f"APP_DIR ('{APP_DIR}') is not initialized or not a valid"
                " directory."
            )
        self.start_date = None
        self.end_date = None
        self.jinja_env = jinja_env
        self.llm = llm_client
        self.reports_base_dir = global_reports_dir
        self.templates_base_dir = global_templates_dir
        self.source_static_dir = global_templates_dir / "static" / "dist"
        self.severity_color_map = {
            'low': '#25B556',  # Lime
            'medium': '#4BAEEA',  # Yellow/Orange
            'high': '#F0981F',  # Report Main Orange
            'critical': '#EF4444',  # Report Dark Accent Red
            'none': '#9CA3AF',  # Report Light Accent
            'unknown': '#6B7280',  # Gray
        }
        self.severity_display_name_map = {
            'critical': 'Critical',
            'high': 'High',
            'medium': 'Medium',
            'low': 'Low',
            'none': 'Unspecified',
        }
        self.category_color_map = {
            'remote_code_execution': '#EF4444',
            'privilege_elevation': '#F0981F',
            'denial_of_service': '#4BAEEA',
            'disclosure': '#67C2C0',
            'spoofing': '#25B556',
            'tampering': '#89CFF0',
            'feature_bypass': '#F87171',
            'none': '#9CA3AF',
        }
        self.category_color_map_suggestion = {
            'remote_code_execution': 'var(--color-chart-red-1)',  # #EF4444
            'privilege_elevation': 'var(--color-chart-orange-1)',  # #F0981F
            'denial_of_service': 'var(--color-chart-blue-1)',  # #4BAEEA
            'disclosure': 'var(--color-chart-teal-1)',  # #67C2C0
            'spoofing': 'var(--color-chart-green-1)',  # #25B556
            'tampering': (
                'var(--color-chart-blue-2)'
            ),  # #89CFF0 (Lighter Blue for distinction)
            'feature_bypass': (
                'var(--color-chart-red-2)'
            ),  # #F87171 (Lighter Red for distinction)
            'none': 'var(--color-chart-gray-1)',  # #9CA3AF
        }
        self.heatmap_custom_colorscale_blue = [
            [0.0, '#E0F2FE'],  # Very light custom blue
            [0.3, '#89CFF0'],  # Your --color-chart-blue-2
            [0.7, '#4BAEEA'],  # Your --color-chart-blue-1
            [1.0, '#1A8BCD'],  # Your --color-chart-blue-border (or darker)
        ]
        self.heatmap_custom_colorscale_blue_green = [
            [0.0, '#89CFF0'],  # Very Light Custom Blue (for low counts)
            [0.5, '#6EBCA8'],  # Your --color-chart-blue-2
            [
                1.0,
                '#25B556',
            ],  # Your --color-chart-green-1 (or darker green for high counts)
        ]
        self.category_display_name_map = {
            "remote_code_execution": "Remote Code Execution",
            "privilege_elevation": "Privilege Elevation",
            "spoofing": "Spoofing",
            "disclosure": "Information Disclosure",
            "feature_bypass": "Security Feature Bypass",
            "denial_of_service": "Denial of Service",
            "tampering": "Tampering",
            "none": "None",
        }
        self.attack_vector_color_map = {
            'network': '#EF4444',
            'adjacent_network': '#F0981F',
            'local': '#4BAEEA',
            'physical': '#25B556',
            'none': '#9CA3AF',
        }
        # Optional: for nicer x-axis labels if keys are snake_case
        self.attack_vector_display_map = {
            'network': 'Network',
            'adjacent_network': 'Adjacent',
            'local': 'Local',
            'physical': 'Physical',
            'none': 'None',  # Or 'Unspecified'
        }
        self.attack_vector_display_map = {
            'network': 'Network',
            'adjacent': 'Adjacent',
            'local': 'Local',
            'physical': 'Physical',
            'none': 'None',
        }
        self.privileges_required_color_map = {
            'none': '#9CA3AF',  # Gray
            'low': '#4BAEEA',  # Blue
            'high': (  # Orange (or '#EF4444' for Red if 'high' means more critical)
                '#F0981F'
            ),
        }
        self.privileges_required_display_map = {
            'none': 'No Privileges',
            'low': 'Low Privileges',
            'high': 'High Privileges',
        }
        self.user_interaction_display_map = {
            'none': 'No User Action',
            'required': 'User Action Required',
        }
        self.attack_complexity_display_map = {
            'low': 'Low Complexity',
            'high': 'High Complexity',
            # 'none' key is not needed here as it will be filtered
        }
        self.attack_complexity_color_map = (
            {  # Colors for the bars (Low/High Complexity)
                'low': '#4BAEEA',  # Blue
                'high': '#F0981F',  # Orange
            }
        )
        self.spotlight_plot_color = '#EF4444'
        self.cluster_colors_list = px.colors.qualitative.Plotly
        self.cluster_descriptive_names = {
            # EXAMPLE - YOU MUST DEFINE THESE BASED ON THE PRESENT ANALYSIS. CANNOT BE PREDEFINED.
        }
        self.MANUAL_CLUSTER_NAMES = {
            # EXAMPLE - YOU MUST DEFINE THESE BASED ON THE PRESENT ANALYSIS. CANNOT BE PREDEFINED.
        }
        # Define orderings
        self.severity_order = ['low', 'medium', 'high', 'critical']
        self.category_order = [
            'remote_code_execution',
            'privilege_elevation',
            'feature_bypass',
            'denial_of_service',
            'disclosure',
            'spoofing',
            'tampering',
        ]
        self.attack_vector_order = ['network', 'adjacent', 'local', 'physical']
        self.attack_complexity_order = ['low', 'high']
        self.user_interaction_order = ['required', 'none']
        self.privileges_required_order = ['low', 'high']
        self.scope_order = ['unchanged', 'changed']
        self.trend_order = ['up', 'down', 'neutral', 'none']
        self.STAT_CARD_LABELS = STAT_CARD_LABELS
        self.STAT_CARD_SUBTITLES = STAT_CARD_SUBTITLES
        self.ICON_SVG_STRINGS = ICON_SVG_STRINGS
        self.CARD_STYLES = CARD_STYLES
        self.CVE_CATEGORY_COLOR_MAP = CVE_CATEGORY_COLOR_MAP
        self.STAT_CARD_ICONS = STAT_CARD_ICONS
        self.CWE_ROOT_CAUSE = CWE_ROOT_CAUSE
        self.CWE_DETAILS = CWE_DETAILS
        self.METRICS_ELIGIBLE_FOR_TREND = {
            "total_cves",
            "critical_count",
            "important_count",
            "moderate_count",
            "low_count",
            "critical_high_pct",
            "high_to_total_pct",
            "median_cvss",
            "p90_cvss",
            "avg_cvss_score",
            "score_iqr",
            "score_std_dev",
            "rce_pct",
            "eop_pct",
            "dos_pct",
            "info_disclosure_pct",
            "spoofing_pct",
            "tampering_pct",
            "feature_bypass_pct",
            "network_vector_pct",
            "local_vector_pct",
            "adjacent_vector_pct",
            "physical_vector_pct",
            "no_user_action_pct",
            "user_action_req_pct",
            "no_privs_required_pct",
            "low_privs_required_pct",
            "high_privs_required_pct",
            "low_attack_comp_pct",
            "high_attack_comp_pct",
            "worst_case_cves",
            "exploited_count_kev",
            "exploited_pct_kev",
            "top_3_cwe_coverage_pct",
            "median_days_to_patch",
            "patched_le_7d_pct",
            "patched_le_30d_pct",
            "zero_day_count",
            "null_field_pct",
            "rows_analysed",
        }
        self.SYSTEM_PROMPTS = SYSTEM_PROMPTS
        self.CHART_DATA_HANDLERS = {
            # Aligned with REPORT_STRUCTURE chart IDs (looped sections)
            "new_vs_updated_monthly": (
                self._prepare_data_for_new_vs_updated_monthly
            ),
            "volume_by_severity_monthly": (
                self._prepare_data_for_volume_by_severity_monthly
            ),
            "overall_category_distribution_chart": (
                self._prepare_data_for_overall_category_distribution_chart
            ),
            "attack_vectors_by_privileges_chart": (
                self._prepare_data_for_attack_vectors_by_privileges_chart
            ),
            "interaction_vs_complexity_chart": (
                self._prepare_data_for_interaction_vs_complexity_chart
            ),
            "cvss_score_by_category_chart": (
                self._prepare_data_for_cvss_score_by_category_chart
            ),
            "top_cwe_in_high_risk_cves_chart": (
                self._prepare_data_for_top_cwe_in_high_risk_cves_chart
            ),
            "top_windows_versions_by_criticality_chart": (
                self._prepare_data_for_top_windows_versions_by_criticality_chart
            ),
            "days_to_nvd_by_severity_boxplots": (
                self._prepare_data_for_days_to_nvd_by_severity_boxplots
            ),
            "spotlight_cves_cvss_chart": (
                self._prepare_data_for_spotlight_cves_base
            ),
            "cat_av_pr_chart": self._prepare_data_for_cat_av_pr_chart,
            "cat_ui_ac_chart": self._prepare_data_for_cat_ui_ac_chart,
            "top_cwe_av_cvss_chart": (
                self._prepare_data_for_top_cwe_av_cvss_chart
            ),
            "tsne_cve_profile_chart": (
                self._prepare_data_for_tsne_cve_profile_chart
            ),
            # Retaining other handlers that might be used elsewhere or for non-looped charts/tables
            # Review these if they are confirmed to be legacy or unused for looped section charts.
            "cvss_distribution": self._prepare_data_for_cvss_distribution,
            "rce_proportion_monthly": (
                self._prepare_data_for_rce_proportion_monthly
            ),
            "rce_cvss_distribution": (
                self._prepare_data_for_rce_cvss_distribution
            ),
            "eop_proportion_monthly": (
                self._prepare_data_for_eop_proportion_monthly
            ),
            "eop_by_attack_vector": (
                self._prepare_data_for_eop_by_attack_vector
            ),
            "dos_infodisc_monthly": (
                self._prepare_data_for_dos_infodisc_monthly
            ),
            "user_interaction_proportion_monthly": (
                self._prepare_data_for_user_interaction_proportion_monthly
            ),
            "overall_attack_vector_dist": (
                self._prepare_data_for_overall_attack_vector_dist
            ),
            "overall_attack_complexity_dist": (
                self._prepare_data_for_overall_attack_complexity_dist
            ),
            "patch_delay_risk_scatter": (
                self._prepare_data_for_patch_delay_risk_scatter
            ),
            "kev_by_category": self._prepare_data_for_kev_by_category,
            "spotlight_cves_details_table": (
                self._prepare_data_for_spotlight_cves_base
            ),  # For appendix table
        }
        self.PLOTLY_FIGURE_HANDLERS = {
            # Aligned with REPORT_STRUCTURE chart IDs and their figure_gen_key
            "new_vs_updated_monthly": self._plot_new_vs_updated_monthly,
            "volume_by_severity_monthly": (
                self._plot_volume_by_severity_monthly
            ),
            "overall_category_distribution_chart": (
                self._plot_overall_category_distribution_chart
            ),  # Maps to figure_gen_key 'overall_category_treemap'
            "attack_vectors_by_privileges_chart": (
                self._plot_attack_vectors_by_privileges_chart
            ),  # Maps to 'attack_vectors_by_privileges_stacked_bar'
            "interaction_vs_complexity_chart": (
                self._plot_interaction_vs_complexity_chart
            ),  # Maps to 'interaction_vs_complexity_heatmap'
            "cvss_score_by_category_chart": (
                self._plot_cvss_score_by_category_chart
            ),  # Maps to 'cvss_score_by_category_chart_faceted'
            "top_cwe_in_high_risk_cves_chart": (
                self._plot_top_cwe_in_high_risk_cves_chart
            ),  # Maps to 'top_cwe_in_high_risk_dot_plot'
            "top_windows_versions_by_criticality_chart": (
                self._plot_top_windows_versions_by_criticality_chart
            ),  # Maps to 'top_windows_versions_by_criticality_bar'
            "days_to_nvd_by_severity_boxplots": (
                self._plot_days_to_nvd_by_severity_boxplots
            ),  # Maps to 'days_to_nvd_by_severity_boxplots_faceted'
            "spotlight_cves_cvss_chart": (
                self._plot_spotlight_cves_cvss_scores_dot_plot
            ),  # Maps to 'spotlight_cves_cvss_scores_bar'
            "cat_av_pr_chart": self._plot_cat_av_pr_chart,
            "cat_ui_ac_chart": self._plot_cat_ui_ac_chart,
            "top_cwe_av_cvss_chart": self._plot_top_cwe_av_cvss_chart,
            "tsne_cve_profile_chart": self._plot_tsne_cve_profile_chart,
            # Retaining other handlers that might be used elsewhere or for non-looped charts.
            # Review these if they are confirmed to be legacy or unused for looped section charts.
            # Note: Some original keys might now be redundant if their functionality is covered by the new keys above.
            # For example, "cvss_distribution" pointed to _plot_cvss_score_by_category_chart, which is now handled by "cvss_score_by_category_chart".
            # I'm keeping them for now but they are candidates for cleanup.
            # "cvss_distribution": self._plot_cvss_score_by_category_chart, # Potentially redundant
            "rce_proportion_monthly": self._plot_rce_proportion_monthly,
            "rce_cvss_distribution": self._plot_rce_cvss_distribution,
            "eop_proportion_monthly": self._plot_eop_proportion_monthly,
            "eop_by_attack_vector": self._plot_eop_by_attack_vector,
            "dos_infodisc_monthly": self._plot_dos_infodisc_monthly,
            "user_interaction_proportion_monthly": (
                self._plot_user_interaction_proportion_monthly
            ),
            "overall_attack_vector_dist": (
                self._plot_overall_attack_vector_dist
            ),
            "overall_attack_complexity_dist": (
                self._plot_overall_attack_complexity_dist
            ),
            "patch_delay_risk_scatter": self._plot_patch_delay_risk_scatter,
            "kev_by_category_bar": self._plot_kev_by_category_bar,
            # The following seem to be direct mappings already or specific plot types not directly chart IDs from the task list
            # "overall_category_treemap": self._plot_overall_category_treemap, # Now covered by overall_category_distribution_chart
            # "attack_vectors_by_privileges_stacked_bar": self._plot_attack_vectors_by_privileges_stacked_bar, # Now covered
            # "interaction_vs_complexity_heatmap": self._plot_interaction_vs_complexity_chart, # Now covered
            # "cvss_score_by_category_chart_faceted": self._plot_cvss_score_by_category_chart, # Now covered
            # "top_cwe_in_high_risk_dot_plot": self._plot_top_cwe_in_high_risk_cves_chart, # Now covered
            # "top_windows_versions_by_criticality_bar": self._plot_top_windows_versions_by_criticality_chart, # Now covered
            # "days_to_nvd_by_severity_boxplots_faceted": self._plot_days_to_nvd_by_severity_boxplots, # Now covered
        }
        self.cwe_id_to_category_map = {}
        for category, data in CWE_ROOT_CAUSE.items():
            for cwe_id in data.get("ids", []):
                self.cwe_id_to_category_map[cwe_id] = category
        self.llm_token_usage_log: List[Dict[str, Any]] = []
        self.TOKEN_PRICING = {
            # --- Models directly used by LLM_TASK_MODELS ---
            # Model: "gpt-4o" (Used for 'chart_insight')
            "gpt-4o": {"input_1k": 0.005, "output_1k": 0.02},
            # Model: "gpt-4.1-mini" (Used for 'narrative', 'default')
            "gpt-4.1-mini": {"input_1k": 0.0004, "output_1k": 0.0016},
            # Model: "gpt-4o-mini" (Used for 'callout')
            # Note: Added based on standard pricing as it was in LLM_TASK_MODELS but not your new list.
            "gpt-4o-mini": {"input_1k": 0.00015, "output_1k": 0.0006},
            # Model: "o3-mini" (Used for 'section_summary', 'executive_summary', 'report_conclusion')
            # Note: Mapped to your "gpt-3.5-turbo" pricing.
            "o3-mini": {"input_1k": 0.0015, "output_1k": 0.002},
            # --- Other models from your list (potentially used elsewhere or as fallbacks) ---
            "gpt-4.1": {"input_1k": 0.002, "output_1k": 0.008},
            "gpt-3.5-turbo": {
                "input_1k": 0.0015,
                "output_1k": 0.002,
            },  # Explicit gpt-3.5-turbo entry
            "o4-mini-high": {"input_1k": 0.0011, "output_1k": 0.0044},
            # --- Provider-prefixed entries (can serve as specific fallbacks if models are called with these full names) ---
            "openai/o3": {"input_1k": 0.01, "output_1k": 0.04},
            "openai/o4-mini": {"input_1k": 0.0011, "output_1k": 0.0044},
            # --- Optional third-party providers ---
            "openrouter/google/gemini-2.5-pro-preview": {
                "input_1k": 0.000125,
                "output_1k": 0.001,
            },
            # --- Default fallback for unlisted models ---
            "default_llm_pricing": {"input_1k": 0.001, "output_1k": 0.002},
        }
        self._create_plotly_themes()
        logger.info(
            "Initialized QuarterlyDeepDiveReportGenerator with base_dir:"
            f" {self.reports_base_dir}"
        )

    @staticmethod
    def _count_tokens(model_name: str, text: str) -> int:
        """
        Returns the number of tokens in the text for the specified model.
        Handles OpenAI GPT models and Google Gemini models (via OpenRouter).
        Logs all mapping and error situations.
        """
        logger = logging.getLogger(__name__)
        try:
            # Normalize model name (handles OpenRouter prefixes and case)
            normalized_model_name = model_name.split('/')[-1].lower()
            logger.info(
                f"Original model_name: '{model_name}', normalized:"
                f" '{normalized_model_name}'"
            )

            # Default tiktoken model key, used if no specific mapping is found
            tiktoken_model_key = "cl100k_base"

            # Mappings based on TOKEN_PRICING keys and common patterns.
            # Order is important: most specific matches should come first.
            if (
                "gpt-4o-mini" == normalized_model_name
                or "gpt-o4-mini-high" == normalized_model_name
            ):
                tiktoken_model_key = "gpt-4o-mini"
            elif "gpt-4o" == normalized_model_name:
                tiktoken_model_key = "gpt-4o"
            elif "gpt-o3" == normalized_model_name:
                tiktoken_model_key = "gpt-3.5-turbo"
            elif "gpt-3.5" in normalized_model_name:
                tiktoken_model_key = "gpt-3.5-turbo"
            elif (
                "gemini" in normalized_model_name
                or "google/gemini" in model_name.lower()
            ):
                logger.info(
                    "No specific tiktoken encoding for Gemini model"
                    f" '{model_name}'. Using '{tiktoken_model_key}'"
                    " (cl100k_base)."
                )
                # tiktoken_model_key is already "cl100k_base" as per default

            logger.info(
                f"Using tiktoken model key: '{tiktoken_model_key}' for model"
                f" '{model_name}'"
            )

            try:
                encoding = tiktoken.encoding_for_model(tiktoken_model_key)
                logger.info(
                    f"Successfully got encoding for '{tiktoken_model_key}'"
                )
            except KeyError:
                logger.warning(
                    "Tiktoken encoding not found for derived key "
                    f"'{tiktoken_model_key}' (from original model_name "
                    f"'{model_name}'). Using cl100k_base as a fallback."
                )
                encoding = tiktoken.get_encoding("cl100k_base")

            try:
                tokens = encoding.encode(text)
                logger.info(
                    f"Token count for model '{model_name}': {len(tokens)}"
                )
                return len(tokens)
            except Exception as e:
                logger.error(
                    f"Error encoding text for model '{model_name}' "
                    f"with encoding '{tiktoken_model_key}': {e}. "
                    "Falling back to cl100k_base."
                )
                fallback_encoding = tiktoken.get_encoding("cl100k_base")
                return len(fallback_encoding.encode(text))

        except Exception as e:
            logger.error(f"Unexpected error in _count_tokens: {e}")
            fallback_encoding = tiktoken.get_encoding("cl100k_base")
            return len(fallback_encoding.encode(text))

    def _create_plotly_themes(self) -> None:
        """Creates a custom Plotly theme based on brand guidelines."""
        report_colors = {
            'light_bg': '#F4F6F3',
            'light_accent': '#B9A696',
            'main': '#EB8B06',
            'dark_accent': '#D84749',
            'dark_bg': '#1B222D',
            'text_light': '#1B222D',
            'text_dark': '#FCFCFC',
            'border_light': '#d1cdc7',
            'border_dark': '#3a4250',
            'surface_light': '#FFFFFF',
            'surface_dark': '#2a3140',
            'grid_light': '#e5e7eb',  # Tailwind gray-200
            'grid_dark': '#4b5563',  # Tailwind gray-600
            # Define semantic colors if needed for specific traces
            'info': '#22A5DD',
            'success': '#00A33E',
            'warning': '#EB8B06',  # report-main
            'danger': '#D84749',  # report-dark-accent
            # Add more core colors for sequences if needed
            'color_sequence': [
                '#EB8B06',
                '#D84749',
                '#B9A696',
                '#22A5DD',
                '#00A33E',
                '#A0C722',
            ],  # Main, Dark Accent, Light Accent, Brand Blue, Brand Green, Brand Lime
        }
        # --- Light Theme Template ---
        plotly_template_light = go.layout.Template(
            layout=go.Layout(
                font=dict(
                    family="Inter, sans-serif",
                    size=12,
                    color=report_colors['text_light'],
                ),
                title=dict(
                    font=dict(size=18, color=report_colors['text_light']),
                    x=0.05,
                ),  # Title left-aligned
                paper_bgcolor=report_colors[
                    'surface_light'
                ],  # Chart background
                plot_bgcolor=report_colors[
                    'surface_light'
                ],  # Plot area background
                xaxis=dict(
                    gridcolor=report_colors['grid_light'],
                    linecolor=report_colors['border_light'],
                    zerolinecolor=report_colors['grid_light'],
                    tickfont=dict(color=report_colors['text_light']),
                    title=dict(font=dict(color=report_colors['text_light'])),
                ),
                yaxis=dict(
                    gridcolor=report_colors['grid_light'],
                    linecolor=report_colors['border_light'],
                    zerolinecolor=report_colors['grid_light'],
                    tickfont=dict(color=report_colors['text_light']),
                    title=dict(font=dict(color=report_colors['text_light'])),
                ),
                legend=dict(
                    bgcolor='rgba(255,255,255,0.7)',  # Slightly transparent background
                    bordercolor=report_colors['border_light'],
                    font=dict(color=report_colors['text_light']),
                ),
                colorway=report_colors[
                    'color_sequence'
                ],  # Default color sequence for traces
                margin=dict(l=50, r=50, t=80, b=50),
            )
        )

        # --- Dark Theme Template ---
        plotly_template_dark = go.layout.Template(
            layout=go.Layout(
                font=dict(
                    family="Inter, sans-serif",
                    size=12,
                    color=report_colors['text_dark'],
                ),
                title=dict(
                    font=dict(size=18, color=report_colors['text_dark']),
                    x=0.05,
                ),
                paper_bgcolor=report_colors['surface_dark'],
                plot_bgcolor=report_colors['surface_dark'],
                xaxis=dict(
                    gridcolor=report_colors['grid_dark'],
                    linecolor=report_colors['border_dark'],
                    zerolinecolor=report_colors['grid_dark'],
                    tickfont=dict(color=report_colors['text_dark']),
                    title=dict(font=dict(color=report_colors['text_dark'])),
                ),
                yaxis=dict(
                    gridcolor=report_colors['grid_dark'],
                    linecolor=report_colors['border_dark'],
                    zerolinecolor=report_colors['grid_dark'],
                    tickfont=dict(color=report_colors['text_dark']),
                    title=dict(font=dict(color=report_colors['text_dark'])),
                ),
                legend=dict(
                    bgcolor='rgba(42, 49, 64, 0.7)',  # Slightly transparent dark background
                    bordercolor=report_colors['border_dark'],
                    font=dict(color=report_colors['text_dark']),
                ),
                colorway=report_colors['color_sequence'],
                margin=dict(l=50, r=50, t=80, b=50),
            )
        )

        # --- Register the templates ---
        pio.templates['report_light'] = plotly_template_light
        pio.templates['report_dark'] = plotly_template_dark
        pio.templates.default = (  # Set default to light + some plotly defaults
            'report_light+plotly_white'
        )

        # Apply this theme globally within Plotly for this instance or per fig
        # pio.templates['custom_brand'] = theme
        # pio.templates.default = 'plotly_white+custom_brand'
        self._set_plotly_theme('light')

    def _set_plotly_theme(self, theme_name='light'):
        """Sets the active Plotly theme for the report."""
        if theme_name == 'dark':
            pio.templates.default = 'report_dark+plotly_dark'
        else:
            pio.templates.default = 'report_light+plotly_white'

    # ------------------------------------------------------------------------------
    # --- BEGIN DATA PROCESSING FUNCTIONS ------------------------------------------
    # ------------------------------------------------------------------------------
    @staticmethod
    def _extract_metadata_columns(
        df: pd.DataFrame, metadata_col: str, keys: List[str]
    ) -> pd.DataFrame:
        """
        Extracts specified keys from a dictionary column and adds them as new columns.

        Args:
            df: Input DataFrame.
            metadata_col: Name of the column containing dictionaries.
            keys: List of keys to extract.

        Returns:
            DataFrame with new columns for each key.
        """
        df = df.copy()
        for key in keys:
            df[key] = df[metadata_col].apply(
                lambda x: x.get(key, pd.NA) if isinstance(x, dict) else pd.NA
            )
        return df

    @staticmethod
    def _extract_highest_cvss_score(df: pd.DataFrame) -> pd.Series:
        """
        For each row, select the highest available CVSS base score from cna, adp, or nist.
        If multiple are present, use the highest. If none are present, result is NaN.
        """
        score_cols = [
            'cna_base_score_num',
            'adp_base_score_num',
            'nist_base_score_num',
        ]
        # Ensure columns exist and convert to numeric (None, strings -> np.nan)
        for col in score_cols:
            if col not in df.columns:
                df[col] = np.nan
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Use pandas max with skipna=True (default) for row-wise max
        return df[score_cols].max(axis=1, skipna=True)

    @staticmethod
    def _extract_attack_complexity(df: pd.DataFrame) -> pd.Series:
        """
        Extracts the highest attack complexity value from columns containing '_attack_complexity'.

        For each row, returns 'high' if any value is 'high', 'low' if any value is 'low', else None.

        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.Series: Series of highest attack complexity values per row.
        """
        complexity_cols = [
            col for col in df.columns if '_attack_complexity' in col
        ]

        def highest_complexity(row) -> str | None:
            values = [
                str(row[col]).lower() if pd.notna(row[col]) else None
                for col in complexity_cols
            ]
            if 'high' in values:
                return 'high'
            if 'low' in values:
                return 'low'
            return None

        return df.apply(highest_complexity, axis=1)

    @staticmethod
    def _extract_attack_vector(df: pd.DataFrame) -> pd.Series:
        """
        Extracts the highest attack vector value from columns containing '_attack_vector'.

        For each row, returns the value with the highest precedence:
        'network' > 'adjacent network' > 'local' > 'physical' > None.

        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.Series: Series of highest attack vector values per row.
        """
        vector_cols = [col for col in df.columns if '_attack_vector' in col]
        precedence = {'network': 4, 'adjacent': 3, 'local': 2, 'physical': 1}
        reverse_precedence = {v: k for k, v in precedence.items()}

        def highest_vector(row) -> str | None:
            values = [
                str(row[col]).lower() if pd.notna(row[col]) else None
                for col in vector_cols
            ]
            max_rank = 0
            for val in values:
                if val in precedence and precedence[val] > max_rank:
                    max_rank = precedence[val]
            return reverse_precedence[max_rank] if max_rank > 0 else None

        return df.apply(highest_vector, axis=1)

    @staticmethod
    def _extract_privileges_required(df: pd.DataFrame) -> pd.Series:
        """
        Extracts the highest privileges required value from columns containing '_privileges_required'.

        For each row, returns the value with the highest precedence:
        'high' > 'low' > None.

        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.Series: Series of highest privileges required values per row.
        """
        priv_cols = [
            col for col in df.columns if '_privileges_required' in col
        ]
        precedence = {'high': 2, 'low': 1}
        reverse_precedence = {v: k for k, v in precedence.items()}

        def highest_priv(row) -> str | None:
            values = [
                str(row[col]).lower() if pd.notna(row[col]) else None
                for col in priv_cols
            ]
            max_rank = 0
            for val in values:
                if val in precedence and precedence[val] > max_rank:
                    max_rank = precedence[val]
            return reverse_precedence[max_rank] if max_rank > 0 else None

        return df.apply(highest_priv, axis=1)

    @staticmethod
    def _extract_user_interaction(df: pd.DataFrame) -> pd.Series:
        """
        Extracts the user interaction value from columns containing '_user_interaction'.

        For each row, returns the value with the highest precedence:
        'Required' > 'None' > None.

        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.Series: Series of user interaction values per row.
        """
        ui_cols = [col for col in df.columns if '_user_interaction' in col]
        precedence = {'required': 2, 'none': 1}
        reverse_precedence = {v: k for k, v in precedence.items()}

        def highest_ui(row) -> str | None:
            values = [
                str(row[col]).lower() if pd.notna(row[col]) else None
                for col in ui_cols
            ]
            max_rank = 0
            for val in values:
                if val in precedence and precedence[val] > max_rank:
                    max_rank = precedence[val]
            return reverse_precedence[max_rank] if max_rank > 0 else None

        return df.apply(highest_ui, axis=1)

    @staticmethod
    def _extract_confidentiality_impact(df: pd.DataFrame) -> pd.Series:
        """
        Extracts the confidentiality impact value from columns containing '_confidentiality_impact'.

        For each row, returns the value with the highest precedence:
        'high' > 'low' > 'none' > None.

        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.Series: Series of confidentiality impact values per row.
        """
        conf_cols = [col for col in df.columns if '_confidentiality' in col]
        precedence = {'high': 3, 'low': 2, 'none': 1}
        reverse_precedence = {v: k for k, v in precedence.items()}

        def highest_conf(row) -> str | None:
            values = [
                str(row[col]).lower() if pd.notna(row[col]) else None
                for col in conf_cols
            ]
            max_rank = 0
            for val in values:
                if val in precedence and precedence[val] > max_rank:
                    max_rank = precedence[val]
            return reverse_precedence[max_rank] if max_rank > 0 else None

        return df.apply(highest_conf, axis=1)

    @staticmethod
    def _extract_integrity_impact(df: pd.DataFrame) -> pd.Series:
        """
        Extracts the integrity impact value from columns containing '_integrity_impact'.

        For each row, returns the value with the highest precedence:
        'high' > 'low' > 'none' > None.

        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.Series: Series of integrity impact values per row.
        """
        integ_cols = [col for col in df.columns if '_integrity' in col]
        precedence = {'high': 3, 'low': 2, 'none': 1}
        reverse_precedence = {v: k for k, v in precedence.items()}

        def highest_integ(row) -> str | None:
            values = [
                str(row[col]).lower() if pd.notna(row[col]) else None
                for col in integ_cols
            ]
            max_rank = 0
            for val in values:
                if val in precedence and precedence[val] > max_rank:
                    max_rank = precedence[val]
            return reverse_precedence[max_rank] if max_rank > 0 else None

        return df.apply(highest_integ, axis=1)

    @staticmethod
    def _extract_availability_impact(df: pd.DataFrame) -> pd.Series:
        """
        Extracts the availability impact value from columns containing '_availability_impact'.

        For each row, returns the value with the highest precedence:
        'high' > 'low' > 'none' > None.

        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.Series: Series of availability impact values per row.
        """
        avail_cols = [col for col in df.columns if '_availability' in col]
        precedence = {'high': 3, 'low': 2, 'none': 1}
        reverse_precedence = {v: k for k, v in precedence.items()}

        def highest_avail(row) -> str | None:
            values = [
                str(row[col]).lower() if pd.notna(row[col]) else None
                for col in avail_cols
            ]
            max_rank = 0
            for val in values:
                if val in precedence and precedence[val] > max_rank:
                    max_rank = precedence[val]
            return reverse_precedence[max_rank] if max_rank > 0 else None

        return df.apply(highest_avail, axis=1)

    @staticmethod
    def _extract_impact_score(df: pd.DataFrame) -> pd.Series:
        """
        Extracts the highest impact score from columns containing '_impact_score'.

        For each row, returns the highest numeric value (float) or None if all are NaN.

        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.Series: Series of highest impact scores per row.
        """
        impact_cols = [col for col in df.columns if '_impact_score' in col]

        def max_impact(row) -> float | None:
            values = [row[col] for col in impact_cols if pd.notna(row[col])]
            if values:
                return float(max(values))
            return None

        return df.apply(max_impact, axis=1)

    @staticmethod
    def _extract_exploitability_score(df: pd.DataFrame) -> pd.Series:
        """
        Extracts the highest exploitability score from columns containing '_exploitability_score'.

        For each row, returns the highest numeric value (float) or None if all are NaN.

        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.Series: Series of highest exploitability scores per row.
        """
        expl_cols = [
            col for col in df.columns if '_exploitability_score' in col
        ]

        def max_exploit(row) -> float | None:
            values = [row[col] for col in expl_cols if pd.notna(row[col])]
            if values:
                return float(max(values))
            return None

        return df.apply(max_exploit, axis=1)

    @staticmethod
    def _extract_base_score_rating(df: pd.DataFrame) -> pd.Series:
        """
        Extracts the single best available CVSS base score rating string (e.g., 'critical', 'high')
        for each row, checking columns in a defined priority order.

        Prioritization (Example - Adjust as needed):
        1. cna_base_score_rating
        2. nist_base_score_rating
        3. adp_base_score_rating

        Returns the first valid, standard rating found according to the priority.
        Cleans the output to lowercase.

        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.Series: Series containing the single best rating string per row (or None if none found).
        """
        # --- Define the columns to check, IN ORDER OF PREFERENCE ---
        rating_cols_priority = [
            'cna_base_score_rating',
            'nist_base_score_rating',
            'adp_base_score_rating',
        ]

        available_rating_cols = [
            col for col in rating_cols_priority if col in df.columns
        ]

        if not available_rating_cols:
            logger.warning(
                "No recognized base score rating columns found. Cannot extract"
                " severity."
            )
            return pd.Series([None] * len(df), index=df.index)

        valid_ratings = {'critical', 'high', 'medium', 'low'}

        # Helper returns single string or None
        def get_best_rating(row) -> str | None:
            """Finds the first valid rating in the priority list for the row."""
            for col in available_rating_cols:  # Iterate in priority order
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    rating = str(value).strip().lower()
                    if rating in valid_ratings:
                        return (
                            rating  # *** RETURN the first valid one found ***
                        )
            # If loop completes without finding a valid rating
            return None  # *** RETURN None for this row ***

        # logger.info(f"Extracting single best rating using columns (priority): {available_rating_cols}")
        # df.apply calls get_best_rating once per row, building a Series of results
        extracted_rating_series = df.apply(get_best_rating, axis=1)
        # num_extracted = extracted_rating_series.notna().sum()
        # logger.info(f"Successfully extracted {num_extracted} ratings out of {len(df)} rows.")
        # The result is a pd.Series, where each element is a single string or None
        return extracted_rating_series

    @staticmethod
    def _normalise_cwe_column(
        df: pd.DataFrame, col: str = "cwe_id"
    ) -> pd.Series:
        """Explode multi-ID strings like 'CWE-416; CWE-362' → Series of single IDs."""
        pat = re.compile(r"CWE-\d+")
        exploded = (
            df[col]
            .dropna()
            .apply(lambda s: pat.findall(str(s)))  # list per row
            .explode()
            .str.strip()
        )
        return exploded

    @staticmethod
    def _map_root_cause(cwe_id: str) -> str:
        for cause, meta in CWE_ROOT_CAUSE.items():
            if cwe_id in meta["ids"]:
                return cause
        return "Other / Unmapped"

    @staticmethod
    def _compute_cwe_metrics(
        df: pd.DataFrame, prior_df: pd.DataFrame | None = None
    ):
        """Computes various metrics related to Common Weakness Enumerations (CWEs).

        Args:
            df: DataFrame containing the CVE data, must include 'cwe_id' column.
            prior_df: Optional DataFrame for the previous period, used for calculating new CWE types.

        Returns:
            Dictionary containing computed CWE metrics:
                - top_cwe_id (str): The most frequent CWE ID.
                - top_3_cwe_coverage_pct (float): Percentage of CVEs covered by the top 3 CWEs.
                - memory_safety_vs_logic_bugs_pct (str | None): Ratio of memory safety vs logic bugs.
                - new_cwe_types_this_qtr (int): Count of new CWE types this quarter.
                - cwe_category_distribution (dict): Count of CVEs per defined root cause category.
        """
        logger.debug("Computing CWE metrics...")
        cwe_metrics_results = {
            "top_cwe_id": "N/A",
            "top_3_cwe_coverage_pct": 0,
            "memory_safety_vs_logic_bugs_pct": None,
            "new_cwe_types_this_qtr": (
                0
            ),  # Placeholder - requires historical data comparison
            "cwe_category_distribution": {},
        }
        total_cves_with_cwe = 0

        if "cwe_id" not in df.columns or df["cwe_id"].isna().all():
            logger.warning(
                "CWE metrics calculation skipped: 'cwe_id' column missing or"
                " empty."
            )
            return cwe_metrics_results

        # Extract primary CWE ID (e.g., 'CWE-79' from 'CWE-79, CWE-80') and count occurrences
        # Handle potential multiple CWEs per entry by splitting and exploding
        # Make a copy to avoid SettingWithCopyWarning
        df_cwe_expanded = df.dropna(subset=['cwe_id']).copy()
        df_cwe_expanded['cwe_id_list'] = df_cwe_expanded['cwe_id'].str.findall(
            r'(CWE-\d+)'
        )
        df_cwe_expanded = df_cwe_expanded.explode('cwe_id_list').rename(
            columns={'cwe_id_list': 'primary_cwe_id'}
        )

        if (
            df_cwe_expanded.empty
            or 'primary_cwe_id' not in df_cwe_expanded.columns
            or df_cwe_expanded['primary_cwe_id'].isna().all()
        ):
            logger.warning("No valid primary CWE IDs found after extraction.")
            return cwe_metrics_results

        cwe_counts = df_cwe_expanded['primary_cwe_id'].value_counts()
        # Count unique CVEs that have at least one *valid* primary CWE extracted
        total_cves_with_cwe = df_cwe_expanded.dropna(
            subset=['primary_cwe_id']
        )['cve_id'].nunique()

        if cwe_counts.empty or total_cves_with_cwe == 0:
            logger.warning(
                "No CWE counts found after processing or no CVEs with valid"
                " CWEs."
            )
            return cwe_metrics_results

        # --- Calculate Basic CWE Metrics ---
        cwe_metrics_results["top_cwe_id"] = cwe_counts.idxmax()
        top_3_count = cwe_counts.head(3).sum()
        # Base percentage on unique CVEs that *have* a valid CWE ID assigned
        cwe_metrics_results["top_3_cwe_coverage_pct"] = (
            round((top_3_count / total_cves_with_cwe * 100), 1)
            if total_cves_with_cwe
            else 0
        )

        # --- Calculate Category-Based Metrics ---
        category_counts = {category: 0 for category in CWE_ROOT_CAUSE.keys()}
        unknown_cwe_count = 0

        # Iterate through unique primary CWE IDs found in the data
        for cwe_id, count in cwe_counts.items():
            category = QuarterlyDeepDiveReportGenerator._map_root_cause(cwe_id)
            if category:
                # Sum the counts for all occurrences of this CWE ID
                category_counts[category] += count
            else:
                unknown_cwe_count += count

        cwe_metrics_results["cwe_category_distribution"] = category_counts

        if unknown_cwe_count > 0:
            logger.warning(
                f"{unknown_cwe_count} occurrences of CWE IDs not found in"
                " defined categories."
            )

        # --- Memory Safety vs Logic Bugs Ratio ---
        memory_bugs_count = category_counts.get("memory_safety", 0)
        logic_bugs_count = category_counts.get("logic_state_errors", 0)
        # Consider adding other categories to 'logic' if appropriate, e.g., input validation? For now, just direct logic errors.

        if memory_bugs_count > 0 and logic_bugs_count > 0:
            ratio = round(memory_bugs_count / logic_bugs_count, 2)
            cwe_metrics_results["memory_safety_vs_logic_bugs_pct"] = (
                f"{ratio}:1"
            )
        elif memory_bugs_count > 0:
            cwe_metrics_results["memory_safety_vs_logic_bugs_pct"] = (
                "Inf:1 (No Logic)"  # Indicate only memory bugs found
            )
        elif logic_bugs_count > 0:
            cwe_metrics_results["memory_safety_vs_logic_bugs_pct"] = (
                "1:Inf (No Memory)"  # Indicate only logic bugs found
            )
        else:
            cwe_metrics_results["memory_safety_vs_logic_bugs_pct"] = (
                "N/A (None Found)"
            )

        # Placeholder for New CWE Types - requires comparison logic with previous data
        # cwe_metrics_results["new_cwe_types_this_qtr"] = self._compare_cwe_types(current_cwe_set, previous_cwe_set)

        logger.debug(f"Computed CWE metrics: {cwe_metrics_results}")
        return cwe_metrics_results

    def _get_exploded_standardized_products(
        self, df: pd.DataFrame, product_column_name: str = 'products'
    ) -> Optional[pd.DataFrame]:
        """
        Explodes a list-like product column and standardizes product names.

        Args:
            df: DataFrame containing product data.
            product_column_name: Name of the column containing lists of products.
                                 Defaults to 'products'.

        Returns:
            A DataFrame with an exploded 'product_column_name' and a new
            'standardized_product' column, or None if input is invalid.
            All original columns from the input df are preserved in the output.
        """
        if df is None or df.empty:
            logger.warning(
                "Input DataFrame is None or empty for product explosion."
            )
            return None
        if product_column_name not in df.columns:
            logger.warning(
                f"Product column '{product_column_name}' not found in "
                "DataFrame for explosion."
            )
            return None

        df_copy = df.copy()

        # Ensure product column contains lists, handle NaNs and single strings
        df_copy[product_column_name] = df_copy[product_column_name].apply(
            lambda x: (
                x if isinstance(x, list) else ([]) if pd.isna(x) else [str(x)]
            )
        )

        # Explode the DataFrame based on the product column
        # This will duplicate rows for each product in the list
        df_exploded = df_copy.explode(product_column_name)

        if (
            df_exploded.empty
            or df_exploded[product_column_name].isnull().all()
        ):
            logger.warning(
                "DataFrame became empty or product column"
                f" '{product_column_name}' is all null after exploding."
            )
            # Return an empty DataFrame with expected columns if needed by callers,
            # or None if that's handled. For now, let's return it as is.
            # Add 'standardized_product' column even if empty to maintain schema.
            df_exploded['standardized_product'] = pd.Series(dtype='object')
            return df_exploded

        # Standardize the exploded product names
        # The apply function will operate on each individual product name string
        df_exploded['standardized_product'] = df_exploded[
            product_column_name
        ].apply(self._standardize_windows_product_name)

        return df_exploded

    def _prepare_input_dataframe(
        self, report_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Validates and prepares the input DataFrame."""
        logger.debug("Preparing input DataFrame...")
        if not isinstance(report_data, pd.DataFrame):
            raise TypeError("report_data must be a Pandas DataFrame.")
        if report_data.empty:
            logger.warning("Warning: Input DataFrame is empty.")
            return report_data  # Return empty df

        df = report_data.copy()
        # we need to pull out the metadata key-value pairs before proceeding
        # --- Column Validation ---
        nvd_meta_cols = get_nvd_columns()
        document_meta_cols = [
            'id',
            'post_id',
            'revision',
            'published',
            'title',
            'description',
            'source',
            'cve_category',
            'post_type',  # one of 'Information only', 'Solution provided', 'Critical'
            'severity_type',
            'summary',
            'build_numbers',
            'products',
        ]
        # flatten the metadata Object into columns
        all_metadata_keys = nvd_meta_cols + document_meta_cols
        df = QuarterlyDeepDiveReportGenerator._extract_metadata_columns(
            df, 'metadata', all_metadata_keys
        )
        df = df.drop(columns=['metadata'])
        document_root_cols = ['id_', 'text', 'kb_ids']
        required_cols = nvd_meta_cols + document_meta_cols + document_root_cols
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Input DataFrame is missing required columns: {missing_cols}"
            )

        # --- Type Conversions & Derived Columns ---
        # Handle Revision for New/Updated Status
        def get_status(revision_str):
            if pd.isna(revision_str):
                return 'none'
            try:
                # Extract the major version number before the decimal
                major_version = float(str(revision_str).split('.')[0])
                return 'new' if major_version == 1.0 else 'updated'
            except (ValueError, IndexError):
                return 'none'  # Handle parsing errors

        if 'revision' in df.columns:
            df['status'] = df['revision'].apply(get_status)
        else:
            df['status'] = 'none'  # Default if column is missing

        # Handle Build Numbers

        # Handle NVD Published Date as Patch Date Proxy
        if 'nvd_published_date' in df.columns:
            df['patch_date'] = pd.to_datetime(
                df['nvd_published_date'], errors='coerce'
            )
        else:
            df['patch_date'] = pd.NaT  # No proxy available

        # Handle CVE Category Standardization
        category_map = {
            "privilege_elevation": "privilege_elevation",
            "spoofing": "spoofing",
            "remote_code_execution": "remote_code_execution",
            "tampering": "tampering",
            "disclosure": "disclosure",
            "feature_bypass": "feature_bypass",
            "denial_of_service": "denial_of_service",
            pd.NA: 'none',
            None: 'none',
            "": 'none',
        }
        if 'cve_category' in df.columns:
            # Apply mapping, fill remaining NaNs/unmapped with 'other'
            df['cve_category'] = (
                df['cve_category'].map(category_map).fillna('none')
            )
        else:
            df['cve_category'] = 'none'  # Default if column is missing

        # --- Derived Columns ---
        if 'published' in df.columns and pd.api.types.is_datetime64_any_dtype(
            df['published']
        ):
            df['published_month_dt'] = df['published'].dt.to_period('M')
            df['published_month'] = df['published_month_dt'].astype(
                str
            )  # String for grouping/axis
        else:
            # Create placeholder columns if 'published' is missing or invalid
            df['published_month_dt'] = pd.NaT
            df['published_month'] = 'none'
        patterns_to_lower = [
            "attack_vector",
            "attack_complexity",
            "user_interaction",
            "privileges_required",
            "cve_category",
            "severity_type",
            "confidentiality",
            "integrity",
            "availability",
            "scope",
        ]
        for pattern in patterns_to_lower:
            for col in df.columns:
                if pattern in col:
                    df[col] = df[col].astype(str).str.lower()

        df['cvss_score'] = (
            QuarterlyDeepDiveReportGenerator._extract_highest_cvss_score(df)
        )
        df['attack_vector'] = (
            QuarterlyDeepDiveReportGenerator._extract_attack_vector(df)
        )
        df['attack_complexity'] = (
            QuarterlyDeepDiveReportGenerator._extract_attack_complexity(df)
        )
        df['privileges_required'] = (
            QuarterlyDeepDiveReportGenerator._extract_privileges_required(df)
        )
        df['user_interaction'] = (
            QuarterlyDeepDiveReportGenerator._extract_user_interaction(df)
        )
        df['confidentiality_impact'] = (
            QuarterlyDeepDiveReportGenerator._extract_confidentiality_impact(
                df
            )
        )
        df['integrity_impact'] = (
            QuarterlyDeepDiveReportGenerator._extract_integrity_impact(df)
        )
        df['availability_impact'] = (
            QuarterlyDeepDiveReportGenerator._extract_availability_impact(df)
        )
        df['impact_score'] = (
            QuarterlyDeepDiveReportGenerator._extract_impact_score(df)
        )
        df['exploitability_score'] = (
            QuarterlyDeepDiveReportGenerator._extract_exploitability_score(df)
        )
        df['severity_type'].fillna(
            QuarterlyDeepDiveReportGenerator._extract_base_score_rating(df)
        )
        df = df.rename(columns={'post_id': 'cve_id'})

        logger.debug("Input DataFrame prepared successfully.")
        return df

    # -----------------------------------------------------------------------------
    # --- END DATA PROCESSING FUNCTIONS -------------------------------------------
    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # --- BEGIN METRICS PROCESSING FUNCTIONS --------------------------------------
    # -----------------------------------------------------------------------------
    def _get_previous_period_key(
        self, current_start_date: datetime.date
    ) -> str:
        """
        Calculates the period_key for the quarter immediately preceding the
        quarter of the given current_start_date.
        """
        current_year = current_start_date.year
        # Calculate current quarter (1-4)
        current_quarter_num = (current_start_date.month - 1) // 3 + 1

        if current_quarter_num == 1:
            previous_year = current_year - 1
            previous_quarter_num = 4
        else:
            previous_year = current_year
            previous_quarter_num = current_quarter_num - 1

        return f"{previous_year}Q{previous_quarter_num}"

    async def _fetch_previous_period_metrics(
        self,
        current_start_date: datetime.date,
        metrics_service: DuckDBMetricsService,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetches, re-assembles, and returns all metrics from the database for the
        period immediately preceding the current_start_date.

        Args:
            current_start_date: The start date of the current reporting period.
            metrics_service: An instance of DuckDBMetricsService.

        Returns:
            A dictionary of metrics from the previous period (similar to
            _calculate_all_metrics output), or None if not found or an error occurs.
        """
        if not metrics_service:
            logger.warning(
                "Metrics service is not available. Cannot fetch previous"
                " period metrics."
            )
            return None

        previous_period_key = self._get_previous_period_key(current_start_date)
        logger.debug(
            "Attempting to fetch metrics for previous period:"
            f" {previous_period_key}"
        )

        try:
            # This returns List[FactMetricValue] or similar based on your service
            # Ensure FactMetricValue is imported or correctly type-hinted
            previous_metrics_raw = (
                await metrics_service.get_metrics_for_period(
                    previous_period_key
                )
            )

            if not previous_metrics_raw:
                logger.warning(
                    "No metrics found for previous period:"
                    f" {previous_period_key}"
                )
                return None

            previous_metrics_dict: Dict[str, Any] = {}
            for metric_obj in previous_metrics_raw:
                key = (
                    metric_obj.metric_key
                )  # Assuming FactMetricValue has 'metric_key'
                value_to_store: Any = None

                # Reconstruct the original value (numeric or JSON structure)
                if metric_obj.numeric_val is not None:
                    value_to_store = metric_obj.numeric_val
                elif metric_obj.json_val is not None:
                    # Attempt to unwrap if _save_all_metrics wrapped it
                    # (e.g. strings as {"value": "..."}, lists as {"data": [...]})
                    json_data = metric_obj.json_val
                    if isinstance(json_data, dict):
                        if (
                            "data" in json_data
                            and len(json_data) == 1
                            and isinstance(json_data["data"], list)
                        ):
                            value_to_store = json_data["data"]  # Unwrap list
                        elif "value" in json_data and len(json_data) == 1:
                            value_to_store = json_data[
                                "value"
                            ]  # Unwrap string or other single value
                        else:
                            value_to_store = json_data  # Store as dict
                    else:  # Should not happen if saved correctly, but good to handle
                        value_to_store = json_data
                else:
                    # This case means the metric was stored as None (both num_val and json_val are None)
                    value_to_store = None

                previous_metrics_dict[key] = value_to_store

            logger.debug(
                "Successfully fetched and re-assembled"
                f" {len(previous_metrics_dict)} metrics for previous period:"
                f" {previous_period_key}"
            )
            return previous_metrics_dict

        except Exception as e:
            logger.error(
                "Error fetching or processing metrics for previous period"
                f" {previous_period_key}: {e}",
                exc_info=True,
            )
            return None

    def _calculate_all_metrics(
        self,
        df: pd.DataFrame,
        previous_period_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates ALL potentially available metrics based on the input DataFrame.
        Calculation is based on availability of required columns, not REPORT_STRUCTURE.

        Returns:
            A dictionary where keys are metric_ids and values are the raw
            calculated metric values (float, int, str, list, dict, None).
        """
        if df.empty:
            logger.warning(
                "DataFrame is empty, skipping all metric calculations."
            )
            return {}
        if previous_period_metrics:
            logger.debug(
                "Previous period metrics available with"
                f" {len(previous_period_metrics)} keys."
            )
        else:
            logger.debug(
                "No previous period metrics available for trend calculation."
            )

        calculated_values = {}
        logger.debug("Calculating all available metrics...")

        # --- Basic Counts / Shares ---
        calculated_values["total_cves"] = len(df)
        total_cves_val = calculated_values[
            "total_cves"
        ]  # Use this consistent variable

        if 'severity_type' in df.columns:
            # Assuming severity_type column was standardized (e.g., to lowercase) in preprocessing
            calculated_values["critical_count"] = len(
                df[df['severity_type'] == 'critical']
            )
            calculated_values["important_count"] = len(
                df[df['severity_type'] == 'high']
            )
            calculated_values["moderate_count"] = len(
                df[df['severity_type'] == 'medium']
            )
            calculated_values["low_count"] = len(
                df[df['severity_type'] == 'low']
            )

            if total_cves_val > 0:
                crit_high_count = len(
                    df[df['severity_type'].isin(['critical', 'high'])]
                )
                calculated_values["critical_high_pct"] = (
                    crit_high_count / total_cves_val * 100
                )
                high_count = calculated_values.get("important_count", 0)
                calculated_values["high_to_total_pct"] = (
                    high_count / total_cves_val * 100
                )
            else:
                calculated_values["critical_high_pct"] = 0.0
                calculated_values["high_to_total_pct"] = 0.0
        else:
            logger.warning(
                "Missing 'severity_type' column, cannot calculate"
                " severity-based metrics."
            )
            # Set relevant metrics to None or default
            calculated_values["critical_count"] = None
            calculated_values["important_count"] = None
            calculated_values["moderate_count"] = None
            calculated_values["low_count"] = None
            calculated_values["critical_high_pct"] = None
            calculated_values["high_to_total_pct"] = None

        # --- Trend / Volatility Metrics ---
        if "published" in df.columns:
            if df["published"].notna().any():
                monthly_periods = df["published"].dropna().dt.to_period('M')
                if not monthly_periods.empty:
                    monthly_counts = monthly_periods.value_counts()
                    if not monthly_counts.empty:
                        most_frequent_period = monthly_counts.idxmax()
                        calculated_values["most_volatile_month"] = str(
                            most_frequent_period
                        )
                    else:
                        calculated_values["most_volatile_month"] = "N/A"
                else:
                    calculated_values["most_volatile_month"] = "N/A"
            else:
                calculated_values["most_volatile_month"] = "N/A"
        else:
            calculated_values["most_volatile_month"] = (
                None  # Indicate column missing
            )

        # --- CVSS Stats ---
        cvss_col = 'cvss_score'
        if cvss_col in df.columns and df[cvss_col].notna().any():
            valid_cvss = df[cvss_col].dropna()  # Already handles NaN
            if not valid_cvss.empty:
                calculated_values["median_cvss"] = valid_cvss.median()
                calculated_values["p90_cvss"] = valid_cvss.quantile(0.9)
                calculated_values["highest_cvss"] = valid_cvss.max()
                calculated_values["lowest_cvss"] = valid_cvss.min()
                q75, q25 = valid_cvss.quantile(0.75), valid_cvss.quantile(0.25)
                calculated_values["score_iqr"] = (
                    q75 - q25 if pd.notna(q75) and pd.notna(q25) else None
                )
                calculated_values["score_std_dev"] = valid_cvss.std()
                # Calculate average too, often requested
                calculated_values["avg_cvss_score"] = valid_cvss.mean()
            else:  # Column exists but all values are NaN
                calculated_values["median_cvss"] = None
                calculated_values["p90_cvss"] = None
                calculated_values["highest_cvss"] = None
                calculated_values["lowest_cvss"] = None
                calculated_values["score_iqr"] = None
                calculated_values["score_std_dev"] = None
                calculated_values["avg_cvss_score"] = None

            # CVSS Stats by Category (Depends on CVSS and Category columns)
            if (
                'cve_category' in df.columns
                and df['cve_category'].notna().any()
            ):
                category_stats_dict = {}
                try:
                    df_cleaned_for_grouping = df.copy()
                    df_cleaned_for_grouping[cvss_col] = pd.to_numeric(
                        df_cleaned_for_grouping[cvss_col], errors='coerce'
                    )
                    # Ensure cve_category is also string for consistent grouping keys
                    df_cleaned_for_grouping['cve_category'] = (
                        df_cleaned_for_grouping['cve_category'].astype(str)
                    )
                    df_cleaned_for_grouping = df_cleaned_for_grouping.dropna(
                        subset=[cvss_col, 'cve_category']
                    )

                    if not df_cleaned_for_grouping.empty:
                        grouped = df_cleaned_for_grouping.groupby(
                            'cve_category'
                        )[cvss_col]
                        aggregated_stats = grouped.agg(
                            Q1=lambda x: x.quantile(0.25),
                            Median=lambda x: x.median(),
                            Q3=lambda x: x.quantile(0.75),
                            Count=lambda x: x.count(),
                            Mean=lambda x: x.mean(),
                            StdDev=lambda x: x.std(),
                        ).reset_index()

                        for _, row in aggregated_stats.iterrows():
                            category = row['cve_category']
                            # Ensure category name is clean (e.g., handle "nan" string if it slipped through)
                            if category.lower() == 'nan':
                                continue

                            q1, median, q3 = (
                                row['Q1'],
                                row['Median'],
                                row['Q3'],
                            )
                            count, mean, std_dev = (
                                row['Count'],
                                row['Mean'],
                                row['StdDev'],
                            )
                            iqr = (
                                q3 - q1
                                if pd.notna(q1) and pd.notna(q3)
                                else None
                            )
                            lower_fence = (
                                q1 - 1.5 * iqr if pd.notna(iqr) else None
                            )
                            upper_fence = (
                                q3 + 1.5 * iqr if pd.notna(iqr) else None
                            )

                            # Get color hint - assumes category names match keys in CATEGORY_COLOR_CLASSES or preprocessed to match
                            color_hint_key = category.lower().replace(
                                ' ', '_'
                            )  # Example preprocessing

                            category_stats_dict[category] = {
                                "Q1": round(q1, 2) if pd.notna(q1) else None,
                                "Median": (
                                    round(median, 2)
                                    if pd.notna(median)
                                    else None
                                ),
                                "Q3": round(q3, 2) if pd.notna(q3) else None,
                                "IQR": (
                                    round(iqr, 2) if pd.notna(iqr) else None
                                ),
                                "Lower_Fence": (
                                    round(lower_fence, 2)
                                    if pd.notna(lower_fence)
                                    else None
                                ),
                                "Upper_Fence": (
                                    round(upper_fence, 2)
                                    if pd.notna(upper_fence)
                                    else None
                                ),
                                "Count": int(count) if pd.notna(count) else 0,
                                "Mean": (
                                    round(mean, 2) if pd.notna(mean) else None
                                ),
                                "StdDev": (
                                    round(std_dev, 2)
                                    if pd.notna(std_dev)
                                    else None
                                ),
                                "id_for_color": (
                                    color_hint_key  # Added hint for formatter
                                ),
                            }
                        calculated_values["cvss_iqr_by_category"] = (
                            category_stats_dict
                        )
                    else:
                        calculated_values["cvss_iqr_by_category"] = {}
                except Exception as e:
                    logger.error(
                        f"Error calculating CVSS stats by category: {e}",
                        exc_info=True,
                    )
                    calculated_values["cvss_iqr_by_category"] = {}
            else:
                calculated_values["cvss_iqr_by_category"] = (
                    {}
                )  # Indicate category column missing
        else:  # CVSS column missing
            calculated_values["median_cvss"] = None
            calculated_values["p90_cvss"] = None
            calculated_values["highest_cvss"] = None
            calculated_values["lowest_cvss"] = None
            calculated_values["score_iqr"] = None
            calculated_values["score_std_dev"] = None
            calculated_values["avg_cvss_score"] = None
            calculated_values["cvss_iqr_by_category"] = {}

        # --- Impact Shares ---
        if "cve_category" in df.columns and total_cves_val > 0:
            # Standardize category names if needed (e.g., lowercase)
            category_counts = df["cve_category"].value_counts()
            calculated_values["cve_category_distribution_raw_counts"] = (
                category_counts.to_dict()
            )  # Store raw counts

            # Calculate individual shares
            calculated_values["rce_pct"] = (
                category_counts.get("remote_code_execution", 0)
                / total_cves_val
                * 100
            )
            calculated_values["eop_pct"] = (
                category_counts.get("privilege_elevation", 0)
                / total_cves_val
                * 100
            )
            calculated_values["dos_pct"] = (
                category_counts.get("denial_of_service", 0)
                / total_cves_val
                * 100
            )
            # Ensure your category names match exactly or use .get with default 0
            calculated_values["info_disclosure_pct"] = (
                category_counts.get("disclosure", 0) / total_cves_val * 100
            )
            calculated_values["spoofing_pct"] = (
                category_counts.get("spoofing", 0) / total_cves_val * 100
            )
            calculated_values["tampering_pct"] = (
                category_counts.get("tampering", 0) / total_cves_val * 100
            )
            calculated_values["feature_bypass_pct"] = (
                category_counts.get("feature_bypass", 0) / total_cves_val * 100
            )

            calculated_values["top_impact_type"] = (
                category_counts.idxmax() if not category_counts.empty else "nc"
            )

            # Prepare data for the comparison bar card (list of dicts)
            comparison_bar_data = []
            for cat_name, count in category_counts.items():
                color_hint_key = cat_name.lower().replace(
                    ' ', '_'
                )  # Example preprocessing
                comparison_bar_data.append({
                    'name': cat_name.replace(
                        "_", " "
                    ).title(),  # Make name readable
                    'count': count,
                    'percentage': count / total_cves_val * 100,
                    'id_for_color': color_hint_key,
                })
            # Sort by count desc for the bar chart display order
            calculated_values["cve_category_distribution"] = sorted(
                comparison_bar_data, key=lambda x: x['count'], reverse=True
            )

        else:  # Handle missing category or zero CVEs
            calculated_values["cve_category_distribution_raw_counts"] = {}
            calculated_values["rce_pct"] = None
            calculated_values["eop_pct"] = None
            calculated_values["dos_pct"] = None
            calculated_values["info_disclosure_pct"] = None
            calculated_values["spoofing_pct"] = None
            calculated_values["tampering_pct"] = None
            calculated_values["feature_bypass_pct"] = None
            calculated_values["top_impact_type"] = None
            calculated_values["cve_category_distribution"] = []

        # --- Exploitability Metrics ---
        exploitability_cols = [
            'attack_vector',
            'user_interaction',
            'privileges_required',
            'attack_complexity',
        ]
        if all(col in df.columns for col in exploitability_cols):
            if total_cves_val > 0:
                # Assuming lowercase standardization in preprocessing
                calculated_values["network_vector_pct"] = (
                    df['attack_vector'].eq('network').sum()
                    / total_cves_val
                    * 100
                )
                calculated_values["local_vector_pct"] = (
                    df['attack_vector'].eq('local').sum()
                    / total_cves_val
                    * 100
                )
                calculated_values["adjacent_vector_pct"] = (
                    df['attack_vector'].eq('adjacent').sum()
                    / total_cves_val
                    * 100
                )
                calculated_values["physical_vector_pct"] = (
                    df['attack_vector'].eq('physical').sum()
                    / total_cves_val
                    * 100
                )

                calculated_values["no_user_action_pct"] = (
                    df['user_interaction'].eq('none').sum()
                    / total_cves_val
                    * 100
                )
                calculated_values["user_action_req_pct"] = (
                    df['user_interaction'].eq('required').sum()
                    / total_cves_val
                    * 100
                )

                calculated_values["no_privs_required_pct"] = (
                    df['privileges_required'].eq('none').sum()
                    / total_cves_val
                    * 100
                )
                calculated_values["low_privs_required_pct"] = (
                    df['privileges_required'].eq('low').sum()
                    / total_cves_val
                    * 100
                )
                calculated_values["high_privs_required_pct"] = (
                    df['privileges_required'].eq('high').sum()
                    / total_cves_val
                    * 100
                )

                calculated_values["low_attack_comp_pct"] = (
                    df['attack_complexity'].eq('low').sum()
                    / total_cves_val
                    * 100
                )
                calculated_values["high_attack_comp_pct"] = (
                    df['attack_complexity'].eq('high').sum()
                    / total_cves_val
                    * 100
                )

                # Worst case CVEs (example definition)
                try:
                    if total_cves_val > 0:
                        # Calculate normalized values for each factor (0-1 scale)
                        df['av_score'] = (
                            df['attack_vector']
                            .map({
                                'network': 0.85,
                                'adjacent': 0.62,
                                'local': 0.55,
                                'physical': 0.2,
                            })
                            .fillna(0)
                        )

                        df['ac_score'] = (
                            df['attack_complexity']
                            .map({'low': 0.77, 'high': 0.44})
                            .fillna(0)
                        )

                        df['pr_score'] = (
                            df['privileges_required']
                            .map({'none': 0.85, 'low': 0.62, 'high': 0.27})
                            .fillna(0)
                        )

                        df['ui_score'] = (
                            df['user_interaction']
                            .map({'none': 0.85, 'required': 0.62})
                            .fillna(0)
                        )

                        # Calculate weighted risk score
                        df['risk_score'] = (
                            (0.35 * df['av_score'])
                            + (0.25 * df['ac_score'])
                            + (0.20 * df['pr_score'])
                            + (0.20 * df['ui_score'])
                        )

                        # Define high-risk threshold (calibrated to identify ~2-5% of CVEs)
                        high_risk_threshold = 0.75  # This can be adjusted based on your data distribution

                        # Count high-risk CVEs and store in the existing worst_case_cves key
                        calculated_values["worst_case_cves"] = int(
                            (df['risk_score'] >= high_risk_threshold).sum()
                        )

                        # Create a list of high-risk CVE details
                        high_risk_cves_list = []
                        if calculated_values["worst_case_cves"] > 0:
                            high_risk_df = df[
                                df['risk_score'] >= high_risk_threshold
                            ]
                            for _, row in high_risk_df.iterrows():
                                high_risk_cves_list.append({
                                    'cve_id': row.get('post_id', 'Unknown'),
                                    'cvss': row.get('cvss_score', None),
                                    'title': row.get('title', 'Unknown'),
                                })

                        # Store the list of high-risk CVEs
                        calculated_values["high_risk_cves"] = (
                            high_risk_cves_list
                        )
                    else:
                        calculated_values["worst_case_cves"] = 0
                        calculated_values["high_risk_cves"] = []
                except Exception as e:
                    logger.error(
                        "Error calculating worst_case_cves with risk"
                        f" score: {e}",
                        exc_info=True,
                    )
                    calculated_values["worst_case_cves"] = None
                    calculated_values["high_risk_cves"] = []

            else:  # total_cves_val is 0
                for key in [
                    "network_vector_pct",
                    "local_vector_pct",
                    "adjacent_vector_pct",
                    "physical_vector_pct",
                    "no_user_action_pct",
                    "user_action_req_pct",
                    "no_privs_required_pct",
                    "low_privs_required_pct",
                    "high_privs_required_pct",
                    "low_attack_comp_pct",
                    "high_attack_comp_pct",
                    "worst_case_cves",
                ]:
                    calculated_values[key] = (
                        0.0 if "pct" in key else (0 if "cves" in key else None)
                    )
        else:
            missing_cols = [
                col for col in exploitability_cols if col not in df.columns
            ]
            logger.warning(
                f"Missing one or more exploitability columns ({missing_cols}),"
                " cannot calculate related metrics."
            )
            # Set relevant metrics to None or default
            for key in [
                "network_vector_pct",
                "local_vector_pct",
                "adjacent_vector_pct",
                "physical_vector_pct",
                "no_user_action_pct",
                "user_action_req_pct",
                "no_privs_required_pct",
                "low_privs_required_pct",
                "high_privs_required_pct",
                "low_attack_comp_pct",
                "high_attack_comp_pct",
                "worst_case_cves",
            ]:
                calculated_values[key] = None

        # --- CISA KEV ---
        # TODO: There is no scraping/collection of this data currently.
        if 'cisa_kev' in df.columns:  # Check if KEV data is present
            calculated_values["exploited_count_kev"] = int(
                df['cisa_kev'].astype(bool).sum()
            )
            if total_cves_val > 0:
                calculated_values["exploited_pct_kev"] = (
                    calculated_values["exploited_count_kev"]
                    / total_cves_val
                    * 100
                )
            else:
                calculated_values["exploited_pct_kev"] = 0.0
        else:
            calculated_values["exploited_count_kev"] = None
            calculated_values["exploited_pct_kev"] = None

        # --- Top 3 Affected Builds ---
        # This calculation is quite specific, keep it conditional on column presence
        if "build_numbers" in df.columns and df["build_numbers"].notna().any():
            # ... (keep existing calculation logic for top_3_affected_builds) ...
            # Ensure it calculates and adds 'top_3_affected_builds' to calculated_values
            try:
                build_series = df.dropna(subset=["build_numbers"])[
                    "build_numbers"
                ]

                def build_to_string(build):
                    if isinstance(build, list) and len(build) == 4:
                        try:
                            return '.'.join(map(str, build))
                        except TypeError:
                            return None
                    return None

                if not build_series.empty and isinstance(
                    build_series.iloc[0], list
                ):
                    exploded_builds = build_series.explode()
                    build_strings = exploded_builds.apply(
                        build_to_string
                    ).dropna()
                    if not build_strings.empty:
                        build_counts = build_strings.value_counts()
                        top_3_list = (
                            build_counts.head(3).reset_index().values.tolist()
                        )
                        calculated_values["top_3_affected_builds"] = [
                            {"build": item[0], "count": item[1]}
                            for item in top_3_list
                        ]
                    else:
                        calculated_values["top_3_affected_builds"] = []
                else:
                    calculated_values["top_3_affected_builds"] = []
            except Exception as e:
                logger.error(
                    "Error processing build_numbers for"
                    f" top_3_affected_builds: {e}",
                    exc_info=True,
                )
                calculated_values["top_3_affected_builds"] = [
                    {"build": "Error Parsing", "count": 0}
                ]
        else:
            calculated_values["top_3_affected_builds"] = (
                []
            )  # Or None if preferred

        # --- CWE Weaknesses ---
        if "cwe_id" in df.columns and df["cwe_id"].notna().any():
            try:
                cwe_metrics = self._compute_cwe_metrics(
                    df
                )  # Assumes this helper exists
                calculated_values["top_cwe_id"] = cwe_metrics.get(
                    "top_cwe_id", "N/A"
                )
                calculated_values["top_3_cwe_coverage_pct"] = cwe_metrics.get(
                    "top_3_cwe_coverage_pct", 0.0
                )
                calculated_values["memory_safety_vs_logic_bugs_pct"] = (
                    cwe_metrics.get("memory_safety_vs_logic_bugs_pct", None)
                )
                calculated_values["new_cwe_types_this_qtr"] = cwe_metrics.get(
                    "new_cwe_types_this_qtr", 0
                )
            except Exception as e:
                logger.error(
                    f"Error computing CWE metrics: {e}", exc_info=True
                )
                # Set defaults if computation fails
                calculated_values["top_cwe_id"] = "Error"
                calculated_values["top_3_cwe_coverage_pct"] = None
                calculated_values["memory_safety_vs_logic_bugs_pct"] = None
                calculated_values["new_cwe_types_this_qtr"] = None
        else:
            calculated_values["top_cwe_id"] = None
            calculated_values["top_3_cwe_coverage_pct"] = None
            calculated_values["memory_safety_vs_logic_bugs_pct"] = None
            calculated_values["new_cwe_types_this_qtr"] = None

        # --- Patch Timeliness ---
        if "published" in df.columns and "patch_date" in df.columns:
            # Assuming datetime conversion happened in preprocessing
            df_valid_dates = df.dropna(
                subset=['published', 'patch_date']
            ).copy()
            if not df_valid_dates.empty:
                df_valid_dates['patch_delta_days'] = (
                    df_valid_dates['published'] - df_valid_dates['patch_date']
                ).dt.days
                positive_delta_df = df_valid_dates[
                    df_valid_dates['patch_delta_days'] >= 0
                ]
                calculated_values["median_days_to_patch"] = (
                    positive_delta_df['patch_delta_days'].median()
                    if not positive_delta_df.empty
                    else 0.0
                )

                total_comparable = len(df_valid_dates)
                if total_comparable > 0:
                    count_le_7d = (
                        df_valid_dates['patch_delta_days'] <= 7
                    ).sum()
                    calculated_values["patched_le_7d_pct"] = (
                        count_le_7d / total_comparable * 100
                    )
                    count_le_30d = (
                        df_valid_dates['patch_delta_days'] <= 30
                    ).sum()
                    calculated_values["patched_le_30d_pct"] = (
                        count_le_30d / total_comparable * 100
                    )
                else:
                    calculated_values["patched_le_7d_pct"] = 0.0
                    calculated_values["patched_le_30d_pct"] = 0.0

                # Zero-day - depends on 'exploited' column definition and presence
                if "exploited" in df.columns:
                    # Example definition: Exploited before or on the day of the *vendor* patch
                    zero_day_df = df_valid_dates[
                        (df_valid_dates['exploited'] == True)
                        & (
                            df_valid_dates['published']
                            >= df_valid_dates['patch_date']
                        )  # NVD Published >= Vendor Patch Date
                        # Potentially refine based on specific 'exploited' data meaning
                    ]
                    calculated_values["zero_day_count"] = len(zero_day_df)
                else:
                    calculated_values["zero_day_count"] = (
                        None  # Indicate exploited column missing
                    )
            else:  # Not enough valid date pairs
                calculated_values["median_days_to_patch"] = None
                calculated_values["patched_le_7d_pct"] = None
                calculated_values["patched_le_30d_pct"] = None
                calculated_values["zero_day_count"] = None
        else:  # Date columns missing
            calculated_values["median_days_to_patch"] = None
            calculated_values["patched_le_7d_pct"] = None
            calculated_values["patched_le_30d_pct"] = None
            calculated_values["zero_day_count"] = None

        # --- Other Potential Metrics ---
        # --- Most Affected Product ---
        calculated_values['most_affected_product'] = "N/A"  # Default

        # Check if the 'products' column (your source of product lists)
        # and 'cve_id' (your CVE identifier) exist in the DataFrame 'df'.
        # Adjust 'cve_id' if your CVE ID column is named differently (e.g., 'id').
        if "products" in df.columns and "cve_id" in df.columns:
            # Use the helper method to get exploded and standardized products.
            # The 'product_column_name' should match your actual column name.
            processed_products_df = self._get_exploded_standardized_products(
                df, product_column_name='products'
            )

            if (
                processed_products_df is not None
                and not processed_products_df.empty
                and 'standardized_product' in processed_products_df.columns
            ):

                # Group by the standardized product name and count unique CVE IDs.
                # Ensure 'cve_id' here matches your actual CVE ID column name.
                product_counts = (
                    processed_products_df.groupby('standardized_product')[
                        'cve_id'
                    ]
                    .nunique()
                    .sort_values(ascending=False)
                )

                if not product_counts.empty:
                    calculated_values['most_affected_product'] = (
                        product_counts.index[0]
                    )
                else:
                    logger.info(
                        "No product counts found after processing for"
                        " 'most_affected_product'."
                    )
            else:
                logger.info(
                    "Processed products DataFrame is None, empty, or missing"
                    " 'standardized_product' for 'most_affected_product'."
                )
        else:
            logger.warning(
                "Required columns ('affected_products' or 'cve_id') not found"
                " in df for 'most_affected_product' calculation."
            )
            calculated_values['most_affected_product'] = (
                None  # Or "N/A" if preferred when columns are missing
            )

        # Placeholder - requires specific logic
        # calculated_values["quarter_risk_index"] = self._calculate_risk_index(calculated_values)
        # calculated_values["next_patch_tuesday_date"] = self._calculate_next_patch_tuesday()
        # calculated_values["data_freshness"] = self.data_load_timestamp

        # General quality metric
        key_cols = [
            'cvss_score',
            'cwe_id',
            'cve_category',
            'severity_type',
            'attack_vector',
            'patch_date',
        ]
        existing_key_cols = [col for col in key_cols if col in df.columns]
        if existing_key_cols and not df.empty:
            total_cells = len(df) * len(existing_key_cols)
            null_cells = df[existing_key_cols].isnull().sum().sum()
            calculated_values["null_field_pct"] = (
                (null_cells / total_cells * 100) if total_cells > 0 else 0.0
            )
        else:
            calculated_values["null_field_pct"] = (
                None  # Indicate issue calculating
            )

        calculated_values["rows_analysed"] = len(df)

        # Placeholder - requires previous period data
        calculated_values["total_cves_qoq_delta"] = None

        logger.info(
            "Finished calculating metrics. Total metrics calculated:"
            f" {len(calculated_values)}"
        )
        # logger.debug(f"Calculated values: {calculated_values}") # Optional debug

        # --- SECTION: Calculate Quarter-over-Quarter (QoQ) Trends ---
        if previous_period_metrics:
            logger.debug(
                "Calculating QoQ trends using previous period metrics..."
            )
            for metric_id in self.METRICS_ELIGIBLE_FOR_TREND:
                current_value = calculated_values.get(metric_id)
                previous_value = previous_period_metrics.get(
                    metric_id
                )  # Assumes keys are the same

                # Ensure both values are numeric and not None for calculation
                if (
                    isinstance(current_value, (int, float))
                    and isinstance(previous_value, (int, float))
                    and pd.notna(current_value)
                    and pd.notna(previous_value)
                ):

                    delta_abs = current_value - previous_value
                    calculated_values[f"{metric_id}_qoq_delta_abs"] = delta_abs

                    if previous_value != 0:
                        delta_pct = (delta_abs / previous_value) * 100
                        calculated_values[f"{metric_id}_qoq_delta_pct"] = (
                            delta_pct
                        )
                    else:
                        # Handle division by zero if previous value was 0
                        # If current is also 0, change is 0%. If current is non-zero, change is infinite/undefined.
                        if current_value == 0:
                            calculated_values[f"{metric_id}_qoq_delta_pct"] = (
                                0.0
                            )
                        else:
                            # Using a large placeholder or None, as percentage change is technically infinite
                            # An alternative is to show the absolute change only.
                            calculated_values[f"{metric_id}_qoq_delta_pct"] = (
                                None  # Or some indicator like float('inf') if you handle it
                            )
                            logger.debug(
                                f"QoQ % change for '{metric_id}' is undefined"
                                " (previous was 0, current is"
                                f" {current_value})."
                            )

                    # Determine trend direction based on the metric's nature (higher is better/worse)
                    # This is a simplified direction; some metrics are "lower is better" (e.g., median_days_to_patch)
                    # You might need a more sophisticated way to define "good" vs "bad" trends per metric
                    if delta_abs > 0:
                        calculated_values[
                            f"{metric_id}_qoq_trend_direction"
                        ] = "up"
                    elif delta_abs < 0:
                        calculated_values[
                            f"{metric_id}_qoq_trend_direction"
                        ] = "down"
                    else:
                        calculated_values[
                            f"{metric_id}_qoq_trend_direction"
                        ] = "neutral"
                else:
                    # If current or previous value is missing or not numeric, can't calculate trend
                    calculated_values[f"{metric_id}_qoq_delta_abs"] = None
                    calculated_values[f"{metric_id}_qoq_delta_pct"] = None
                    calculated_values[f"{metric_id}_qoq_trend_direction"] = (
                        None
                    )
                    if (
                        metric_id in calculated_values
                    ):  # Only log if current value was attempted
                        logger.debug(
                            f"Could not calculate QoQ trend for '{metric_id}'."
                            f" Current: {current_value}, Previous:"
                            f" {previous_value}"
                        )
        else:
            # If no previous metrics, set all trend fields to None for eligible metrics
            for metric_id in self.METRICS_ELIGIBLE_FOR_TREND:
                calculated_values[f"{metric_id}_qoq_delta_abs"] = None
                calculated_values[f"{metric_id}_qoq_delta_pct"] = None
                calculated_values[f"{metric_id}_qoq_trend_direction"] = None

        logger.info(
            "Finished calculating all metrics. Total metrics:"
            f" {len(calculated_values)}"
        )
        # logger.info(f"FINAL calculated_values dictionary (keys: {len(calculated_values.keys())}):")
        # For better readability, print keys sorted or one per line if many
        # for key in sorted(calculated_values.keys()):
        #     logger.info(f"  - {key}: {calculated_values[key]}")
        return calculated_values

    def _format_metrics_for_report_cards(
        self, calculated_metrics: Dict[str, Any]
    ) -> Dict[str, List[Dict]]:
        """
        Formats the raw calculated metrics into dictionaries suitable for
        rendering stat cards in the Jinja template.
        Includes handling for simple and complex card types.
        """
        all_section_stats_data = defaultdict(list)

        sections_to_process = {
            **self.REPORT_STRUCTURE.get("explicit_sections", {}),
            **self.REPORT_STRUCTURE.get("looped_sections", {}),
        }

        for section_key, section_config in sections_to_process.items():
            section_card_data_list = []
            logger.info(f"Processing section: {section_key}")
            for stat_def in section_config.get("stats_definitions", []):
                metric_id = stat_def["metric_id"]
                raw_value = calculated_metrics.get(metric_id)
                # This 'card_style_from_config' determines WHICH Jinja macro branch to use
                card_style_key = stat_def.get("card_style", "default_standard")
                card_style_definition = self.CARD_STYLES.get(
                    card_style_key,
                    self.CARD_STYLES.get("default_standard", {}),
                )
                logger.info(
                    f"Processing metric: {metric_id} with style_key:"
                    f" {card_style_key}"
                )
                logger.debug(
                    f"Raw value type: {type(raw_value)} -> {raw_value}"
                )
                # Defaults for simple cards
                formatted_value_simple = "N/A"
                unit_simple = None

                label_simple = stat_def.get(
                    "title",
                    self.STAT_CARD_LABELS.get(
                        metric_id, metric_id.replace("_", " ").title()
                    ),
                )
                subtitle_simple = self.STAT_CARD_SUBTITLES.get(metric_id, "")
                icon_name_key = self.STAT_CARD_ICONS.get(metric_id, "default")
                icon_svg_simple = self.ICON_SVG_STRINGS.get(
                    icon_name_key,
                    self.ICON_SVG_STRINGS.get(
                        "default", "<svg><!-- fallback SVG --></svg>"
                    ),
                )
                trend_value_display = (
                    None  # For formatted trend string e.g., "+5.2%"
                )
                trend_direction_final = (
                    None  # 'up', 'down', 'neutral', or None
                )

                # --- Trend Data Processing (if eligible and available) ---
                if metric_id in self.METRICS_ELIGIBLE_FOR_TREND:
                    # Fetch the pre-calculated trend components from calculated_metrics
                    # These were computed in _calculate_all_metrics
                    delta_abs = calculated_metrics.get(
                        f"{metric_id}_qoq_delta_abs"
                    )
                    delta_pct = calculated_metrics.get(
                        f"{metric_id}_qoq_delta_pct"
                    )
                    trend_direction_final = calculated_metrics.get(
                        f"{metric_id}_qoq_trend_direction"
                    )

                    if (
                        trend_direction_final is not None
                    ):  # A valid trend direction was determined
                        # Prioritize showing percentage change if available and numeric
                        if isinstance(delta_pct, (int, float)) and pd.notna(
                            delta_pct
                        ):
                            # We want to show the magnitude, arrow shows direction
                            # For neutral, the existing "±0.0%" or "±0" from _calculate_all_metrics might be fine,
                            # or we can re-format here for consistency.
                            if trend_direction_final == "neutral":
                                # If _calculate_all_metrics already set delta_pct to 0.0 for neutral, this is fine.
                                # Let's assume we want to be explicit with "±" for neutral if value is ~0
                                if abs(delta_pct) < 0.001:
                                    trend_value_display = "±0.0%"
                                else:  # Should ideally be 0.0 if neutral
                                    trend_value_display = (  # Show absolute value
                                        f"{abs(delta_pct):.1f}%"
                                    )
                            else:  # Up or Down
                                trend_value_display = f"{abs(delta_pct):.1f}%"

                        # Fallback to absolute change if percentage is not available/numeric
                        elif isinstance(delta_abs, (int, float)) and pd.notna(
                            delta_abs
                        ):
                            if trend_direction_final == "neutral":
                                if (
                                    abs(delta_abs) < 0.001
                                ):  # Check if it's effectively zero
                                    trend_value_display = "±0"
                                    # If metric_id is a percentage itself, format as percent
                                    if "_pct" in metric_id:
                                        trend_value_display += "%"
                                else:  # Should be 0 if neutral
                                    trend_value_display = (  # Show absolute value
                                        f"{abs(delta_abs):.1f}"
                                    )
                                    if "_pct" in metric_id:
                                        trend_value_display += (  # Add % if original metric was a pct
                                            "%"
                                        )
                            else:  # Up or Down
                                if isinstance(
                                    calculated_metrics.get(metric_id), int
                                ) or (
                                    isinstance(delta_abs, float)
                                    and delta_abs.is_integer()
                                ):
                                    trend_value_display = (
                                        f"{abs(int(delta_abs)):,}"
                                    )
                                else:
                                    trend_value_display = (
                                        f"{abs(delta_abs):,.1f}"
                                    )
                                if (
                                    "_pct" in metric_id
                                    and not trend_value_display.endswith('%')
                                ):  # If base metric is %, and we show abs delta
                                    # This case is tricky: if base is 50% and abs delta is 5, is it "5" or "5%"?
                                    # For simplicity, if showing absolute delta of a percentage metric, maybe don't add trailing %
                                    pass

                        # If trend_direction_final is "neutral" but display is still None (e.g. both deltas were None)
                        # this means previous_value might have been None or current_value was None,
                        # but _calculate_all_metrics decided direction is "neutral" because they are both None.
                        # In such case, an empty trend value might be best.
                        if (
                            trend_direction_final == "neutral"
                            and trend_value_display is None
                        ):
                            # If _calculate_all_metrics correctly set deltas to 0 for this neutral case,
                            # one of the blocks above would have formatted it.
                            # If it's still None, it means both actual values were None.
                            trend_value_display = (  # Or perhaps "~" if you want an indicator without a value
                                ""
                            )

                # This will hold the structured data for complex cards (e.g., a list of dicts)
                # It will become the 'value' attribute of the final_card_data_for_jinja
                complex_card_structured_value = None

                # --- Special handling for complex card types based on metric_id AND card_style_from_config ---
                # We check card_style_from_config to ensure we only format complex data if the template intends to use that style.
                if (
                    card_style_key == "category_comparison_bar_style"
                    and metric_id == "cve_category_distribution"
                ):
                    if isinstance(raw_value, list):
                        complex_card_structured_value_temp = []
                        for item in raw_value:
                            color_hint_key = item.get(
                                'id_for_color',
                                item.get('name', 'default')
                                .lower()
                                .replace(' ', '_'),
                            )
                            color_class = CVE_CATEGORY_COLOR_MAP.get(
                                color_hint_key
                            )
                            complex_card_structured_value_temp.append({
                                'name': item.get('name'),
                                'count': item.get('count'),
                                'percentage': item.get('percentage'),
                                'color_class': color_class,
                            })

                        # Sort: put "None" (or "Unknown", "NC") at the end, others by percentage desc
                        none_like_categories = [
                            "none",
                            "unknown",
                            "nc",
                            "other",
                        ]  # Define what counts as "None"

                        items_none = [
                            item
                            for item in complex_card_structured_value_temp
                            if item['name'].lower() in none_like_categories
                        ]
                        items_other = [
                            item
                            for item in complex_card_structured_value_temp
                            if item['name'].lower() not in none_like_categories
                        ]

                        items_other_sorted = sorted(
                            items_other,
                            key=lambda x: x['percentage'],
                            reverse=True,
                        )
                        items_none_sorted = sorted(
                            items_none,
                            key=lambda x: x['percentage'],
                            reverse=True,
                        )  # Also sort "None" items if multiple

                        complex_card_structured_value = (
                            items_other_sorted + items_none_sorted
                        )  # "None" items append at the end

                        label_simple = self.STAT_CARD_LABELS.get(
                            metric_id, "Vulnerability Categories"
                        )
                    else:
                        # Data not in expected format for this complex card, fallback or log error
                        complex_card_structured_value = [{
                            'name': 'Error',
                            'count': 0,
                            'percentage': 0.0,
                            'color_class': 'bg-red-500',
                        }]
                        label_simple = self.STAT_CARD_LABELS.get(
                            metric_id, "Category Data Error"
                        )

                elif (
                    card_style_key == "cvss_iqr_by_category_style"
                    and metric_id == "cvss_iqr_by_category"
                ):
                    # logger.info(f"Processing complex card: {metric_id} with style {card_style_key}")
                    logger.info(
                        f"Raw value type for {metric_id}: {type(raw_value)}"
                    )
                    if isinstance(
                        raw_value, dict
                    ):  # Expecting a dict where keys are category names
                        temp_list_for_sorting = []
                        complex_card_structured_value = []
                        for cat_name, stats in raw_value.items():
                            # Assume stats dict is like:
                            # {'Q1': ..., 'Median': ..., 'Q3': ..., 'IQR': ..., 'id_for_color': 'remote_code_execution'}
                            color_hint_key = stats.get(
                                'id_for_color',
                                cat_name.lower().replace(' ', '_'),
                            )
                            color_class = CVE_CATEGORY_COLOR_MAP.get(
                                color_hint_key
                            )
                            if cat_name.lower() == "remote_code_execution":
                                cat_name = "Remote Code Execution"
                            elif cat_name.lower() == "privilege_elevation":
                                cat_name = "Privilege Elevation"
                            elif cat_name.lower() == "denial_of_service":
                                cat_name = "Denial of Service"
                            elif cat_name.lower() == "disclosure":
                                cat_name = "Information Disclosure"
                            elif cat_name.lower() == "feature_bypass":
                                cat_name = "Security Feature Bypass"
                            elif cat_name.lower() == "none":
                                cat_name = "Unknown"
                            else:
                                cat_name = cat_name.title()

                            temp_list_for_sorting.append({
                                'name': cat_name,
                                'q1': stats.get('Q1'),
                                'median': stats.get('Median'),
                                'q3': stats.get('Q3'),
                                'iqr': stats.get('IQR'),
                                'color_class': color_class,
                            })
                        # Sort by category name for consistent display
                        complex_card_structured_value = sorted(
                            temp_list_for_sorting,
                            key=lambda x: (
                                x['median'] if x['median'] is not None else -1
                            ),  # Sort Nones last (as -1)
                            reverse=True,  # Higher median = higher severity = comes first
                        )

                        label_simple = self.STAT_CARD_LABELS.get(metric_id)
                        logger.debug(
                            "Built complex_card_structured_value for"
                            f" {metric_id}: {complex_card_structured_value}"
                        )
                    else:
                        logger.error(
                            f"ERROR: For {metric_id}, raw_value was NOT a dict"
                            f" as expected! Type: {type(raw_value)}"
                        )
                        complex_card_structured_value = [{
                            'name': 'Error',
                            'q1': 0,
                            'median': 0,
                            'q3': 0,
                            'iqr': 0,
                            'color_class': 'bg-red-500',
                        }]
                        label_simple = self.STAT_CARD_LABELS.get(
                            metric_id, "CVSS Stats Error"
                        )

                # --- Formatting for simple card types (if not a complex card processed above) ---
                if complex_card_structured_value is None:
                    if raw_value is not None:
                        if (
                            metric_id == 'most_volatile_month'
                            and raw_value
                            and isinstance(raw_value, str)
                            and '-' in raw_value
                        ):
                            try:
                                year, month_num = raw_value.split('-')
                                # Create a datetime object to easily get month name
                                month_dt = datetime(
                                    int(year), int(month_num), 1
                                )
                                formatted_value_simple = month_dt.strftime(
                                    '%B'
                                )  # "March 2024"
                                # Or for just month name: formatted_value_simple = month_dt.strftime('%B') # "March"
                            except ValueError:
                                formatted_value_simple = raw_value

                        elif (
                            metric_id == 'median_days_to_patch'
                            and raw_value
                            and isinstance(raw_value, str)
                        ):
                            unit_simple = "days"

                        elif (
                            metric_id == 'top_impact_type'
                            and raw_value
                            and isinstance(raw_value, str)
                        ):
                            if raw_value.lower() == "remote_code_execution":
                                formatted_value_simple = (
                                    "Remote Code Execution"
                                )
                            elif raw_value.lower() == "privilege_elevation":
                                formatted_value_simple = "Privilege Elevation"
                            elif raw_value.lower() == "denial_of_service":
                                formatted_value_simple = "Denial of Service"
                            elif raw_value.lower() == "disclosure":
                                formatted_value_simple = (
                                    "Information Disclosure"
                                )
                            elif raw_value.lower() == "feature_bypass":
                                formatted_value_simple = (
                                    "Security Feature Bypass"
                                )
                            elif raw_value.lower() == "tampering":
                                formatted_value_simple = "Tampering"
                            elif raw_value.lower() == "spoofing":
                                formatted_value_simple = "Spoofing"
                            elif raw_value.lower() == "none":
                                formatted_value_simple = "Unknown"
                            else:
                                formatted_value_simple = raw_value.title()
                            unit_simple = ""

                        elif metric_id == "most_affected_product":
                            formatted_value_simple = "N/A"  # Default
                            unit_simple = ""

                            # Determine the text size class based on the number of tied products
                            text_size_class_to_add = 'text-lg'  # Default

                            if (
                                isinstance(raw_value, dict)
                                and "products" in raw_value
                                and "count" in raw_value
                            ):
                                products_list = raw_value["products"]
                                cve_count = raw_value["count"]
                                num_tied_products = len(products_list)

                                if num_tied_products == 0:
                                    formatted_value_simple = "N/A"
                                    subtitle_simple = "No product data"
                                    # text_size_class_to_add remains 'text-lg'
                                elif num_tied_products == 1:
                                    formatted_value_simple = products_list[
                                        0
                                    ].title()
                                    subtitle_simple = f"{cve_count} CVEs"
                                    # text_size_class_to_add remains 'text-lg'
                                elif num_tied_products == 2:
                                    formatted_value_simple = (
                                        f"{products_list[0].title()} &"
                                        f" {products_list[1].title()}"
                                    )
                                    subtitle_simple = (
                                        f"{num_tied_products} Products Tied"
                                        f" ({cve_count} CVEs each)"
                                    )
                                    text_size_class_to_add = (  # Slightly smaller
                                        'text-md'
                                    )
                                elif num_tied_products == 3:
                                    formatted_value_simple = ", ".join(
                                        [p.title() for p in products_list[:3]]
                                    )
                                    subtitle_simple = (
                                        f"{num_tied_products} Products Tied"
                                        f" ({cve_count} CVEs each)"
                                    )
                                    text_size_class_to_add = (  # Even smaller
                                        'text-sm'
                                    )
                                else:  # More than 3 tied products
                                    formatted_value_simple = (
                                        f"{num_tied_products} Products Tied"
                                    )
                                    example_prods = ", ".join(
                                        [p.title() for p in products_list[:2]]
                                    )
                                    subtitle_simple = (
                                        f"e.g., {example_prods}, etc."
                                        f" ({cve_count} CVEs each)"
                                    )
                                    text_size_class_to_add = (  # Reset to a readable size for summary
                                        'text-base'
                                    )

                            elif (
                                raw_value is None
                            ):  # Handles the case where calculation resulted in None
                                formatted_value_simple = "N/A"
                                subtitle_simple = "Data unavailable"
                                # text_size_class_to_add remains 'text-lg'
                            else:
                                # A single product was the highest count
                                formatted_value_simple = raw_value.title()
                                text_size_class_to_add = 'text-md font-bold'
                            # Modify the 'value' style in card_style_definition
                            # First, ensure 'value' key exists and get its current classes
                            current_value_classes = card_style_definition.get(
                                'value', ''
                            )

                            # Optional: Remove any existing text-size classes to avoid conflicts
                            # This is a simple way; a more robust way would use regex or parse classes
                            existing_text_sizes = [
                                'text-xs',
                                'text-sm',
                                'text-base',
                                'text-lg',
                                'text-xl',
                                'text-2xl',
                                'text-3xl',
                                'text-4xl',
                                'text-5xl',
                                'text-6xl',
                                'text-7xl',
                                'text-8xl',
                                'text-9xl',
                            ]
                            cleaned_value_classes = ' '.join(
                                c
                                for c in current_value_classes.split()
                                if c not in existing_text_sizes
                            )

                            # Add the new text size class
                            card_style_definition['value'] = (
                                f"{cleaned_value_classes} {text_size_class_to_add}"
                                .strip()
                            )

                        elif metric_id.startswith("spotlight_cwe_slot_"):
                            cwe_item_data = calculated_metrics.get(metric_id)
                            if cwe_item_data and isinstance(
                                cwe_item_data, dict
                            ):
                                cwe_id_raw = cwe_item_data[
                                    'cwe_id_raw'
                                ]  # e.g., "CWE-79"

                                # Construct keys for predefined labels, subtitles, styles using the raw CWE ID
                                predefined_label_key = (
                                    f"spotlight_cwe_{cwe_id_raw}"
                                )
                                predefined_subtitle_key = (  # Or a generic one
                                    f"spotlight_cwe_{cwe_id_raw}"
                                )
                                predefined_style_key = (  # For per-CWE specific style
                                    f"spotlight_cwe_{cwe_id_raw}_card"
                                )

                                # Get card style: try specific, then default for slot, then overall default
                                base_card_style = self.CARD_STYLES.get(
                                    predefined_style_key
                                )
                                card_style_for_item = base_card_style.copy()
                                label = self.STAT_CARD_LABELS.get(
                                    predefined_label_key,
                                    cwe_item_data['display_cwe_label'],
                                )
                                # Primary subtitle from STAT_CARD_SUBTITLES for the specific CWE, if defined
                                subtitle = self.STAT_CARD_SUBTITLES.get(
                                    predefined_subtitle_key, ""
                                )

                                formatted_value = str(cwe_item_data['count'])
                                # This will be passed as an extra field for the Jinja template to use
                                predominant_category_for_card = cwe_item_data[
                                    'display_predominant_category'
                                ]
                                avg_cvss_for_card = cwe_item_data['avg_cvss']

                                icon_name_key = self.STAT_CARD_ICONS.get(
                                    predefined_label_key
                                )
                                icon_svg = self.ICON_SVG_STRINGS.get(
                                    icon_name_key, None
                                )

                                category_key_for_color_map = (
                                    predominant_category_for_card.lower().replace(
                                        " ", "_"
                                    )
                                    if predominant_category_for_card
                                    else 'unknown'
                                )
                                raw_bg_color_class = self.CVE_CATEGORY_COLOR_MAP.get(
                                    category_key_for_color_map,
                                    self.CVE_CATEGORY_COLOR_MAP.get(
                                        'default',
                                        'bg-chart-gray-2'
                                        ' dark:bg-chart-gray-2-dark',
                                    ),
                                )
                                text_color_class_for_category = (
                                    raw_bg_color_class.replace('bg-', 'text-')
                                    if raw_bg_color_class
                                    else (
                                        'text-report-text-muted'
                                        ' dark:text-report-text-muted-dark'
                                    )
                                )
                                existing_predominant_style = (
                                    card_style_for_item.get(
                                        "predominant_category_text"
                                    )
                                )
                                card_style_for_item[
                                    "predominant_category_text"
                                ] = (
                                    f"{existing_predominant_style} {text_color_class_for_category}"
                                )
                                final_card_data_for_jinja = {
                                    "id": (
                                        metric_id
                                    ),  # Use the slot_id, e.g. "spotlight_cwe_slot_1"
                                    "value": (
                                        formatted_value
                                    ),  # This is the count
                                    "unit": "CVEs",
                                    "label": label,  # "CWE-XXX: Name"
                                    "subtitle": (
                                        subtitle
                                    ),  # Main subtitle (if any predefined)
                                    "icon_svg": icon_svg,
                                    "predominant_category": (
                                        predominant_category_for_card
                                    ),
                                    "avg_cvss": avg_cvss_for_card,
                                    "cwe_id_raw_for_styling": (
                                        cwe_id_raw
                                    ),  # Potentially for template logic
                                    "card_layout_style": stat_def.get(
                                        "card_style"
                                    ),  # e.g., "spotlight_cwe_default_style"
                                    "styles": card_style_for_item,
                                }
                                section_card_data_list.append(
                                    final_card_data_for_jinja
                                )
                                continue
                            else:
                                # This slot is empty (fewer than N top CWEs found)
                                # Optionally, create a placeholder "empty" card or just skip
                                logger.info(
                                    "No data for CWE spotlight slot:"
                                    f" {metric_id}"
                                )
                                continue
                            # end CWE card handling
                        elif isinstance(raw_value, (int, float)):
                            if "pct" in metric_id or "_pct" in metric_id:
                                formatted_value_simple = f"{raw_value:.1f}"
                                unit_simple = "%"
                            elif isinstance(raw_value, float):
                                if pd.isna(raw_value):
                                    formatted_value_simple = "N/A"
                                else:
                                    formatted_value_simple = f"{raw_value:.1f}"
                            else:  # Integer
                                formatted_value_simple = f"{raw_value:,}"

                        elif isinstance(raw_value, list):
                            if (
                                metric_id == "top_3_affected_builds"
                                and raw_value
                                and isinstance(raw_value[0], dict)
                            ):
                                formatted_value_simple = (
                                    raw_value[0].get("build", "N/A")
                                    if raw_value
                                    else "N/A"
                                )
                                subtitle_simple = (
                                    f"{len(raw_value)} builds in list"
                                    if raw_value
                                    else "No build data"
                                )
                            else:
                                formatted_value_simple = (
                                    f"{len(raw_value)} items"
                                )
                        elif isinstance(raw_value, dict):
                            formatted_value_simple = (  # Fallback for other dicts
                                "See Appendix"
                            )
                        elif isinstance(raw_value, str):
                            formatted_value_simple = raw_value
                        else:
                            try:
                                formatted_value_simple = str(raw_value)
                            except Exception:
                                formatted_value_simple = "Error Formatting"
                    # else raw_value is None, formatted_value_simple remains "N/A"

                # --- Construct the final card dictionary for Jinja ---
                final_card_data_for_jinja = {
                    "id": metric_id,
                    "value": (
                        complex_card_structured_value
                        if complex_card_structured_value is not None
                        else formatted_value_simple
                    ),
                    "unit": (
                        unit_simple
                        if complex_card_structured_value is None
                        else None
                    ),
                    "label": label_simple,
                    "subtitle": subtitle_simple,
                    "icon_svg": icon_svg_simple,
                    "trend_value_display": (
                        trend_value_display
                    ),  # The formatted string like "+5.2%"
                    "trend_direction": (
                        trend_direction_final
                    ),  # 'up', 'down', 'neutral' or None
                    "card_layout_style": card_style_key,
                    "styles": card_style_definition,
                }
                section_card_data_list.append(final_card_data_for_jinja)

            if section_card_data_list:
                all_section_stats_data[section_key] = section_card_data_list
        return dict(all_section_stats_data)

    async def _save_all_metrics(
        self,
        metrics_data: Dict[str, Any],
        period_key: str,
        report_name: str,  # e.g., "Quarterly_Deep_Dive_Security_Report"
        metrics_service: DuckDBMetricsService,  # Forward reference if service is in another module
        time_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Saves all calculated metrics to the analytics database using DuckDBMetricsService.

        Args:
            metrics_data: A dictionary of calculated metrics, where keys are
                          metric_ids and values are the metric values.
            period_key: The key for the time period (e.g., "2024Q1").
            report_name: The name of the report originating these metrics.
            metrics_service: An instance of DuckDBMetricsService.
            time_info: Optional dictionary with data for DimTime if it needs
                       to be created (e.g., {'start_date': ..., 'end_date': ..., 'label': ...}).
        """
        if not metrics_data:
            logger.warning("No metrics data provided to save.")
            return

        if not metrics_service:
            logger.error(
                "Metrics service is not available. Cannot save metrics."
            )
            return

        logger.debug(
            f"Starting to save {len(metrics_data)} metrics for period"
            f" '{period_key}' from report '{report_name}'."
        )

        tasks = []
        for metric_key, metric_value in metrics_data.items():
            num_val: Optional[float] = None
            js_val: Optional[Dict[str, Any]] = None

            if isinstance(metric_value, (int, float)):
                num_val = float(metric_value)
            elif isinstance(metric_value, str):
                # Store standalone strings as a JSON object with a 'value' key
                js_val = {"value": metric_value}
            elif isinstance(metric_value, dict):
                js_val = metric_value
            elif isinstance(metric_value, list):
                # For lists (e.g., list of dicts), wrap them in a dict under a 'data' key
                # This ensures compatibility if FactMetricValue.json_val expects a Dict
                js_val = {"data": metric_value}
            elif metric_value is None:
                # Value is None, both num_val and js_val will remain None.
                # The upsert_metric_value method should handle this.
                pass
            else:
                logger.warning(
                    f"Unsupported metric type for key '{metric_key}' in period"
                    f" '{period_key}': {type(metric_value)}. Skipping this"
                    " metric."
                )
                continue

            # Prepare data for DimMetric if it needs to be created/updated
            # You can customize the description further if needed.
            metric_info_for_dim = {
                "description": (
                    f"Metric '{metric_key}' from report '{report_name}' for"
                    f" period '{period_key}'."
                ),
                "originating_report_name": report_name,
            }

            # Create a task for each upsert operation
            task = metrics_service.upsert_metric_value(
                period_key=period_key,
                metric_key=metric_key,
                numeric_val=num_val,
                json_val=js_val,
                time_data=time_info,  # Pass time_info for DimTime creation/lookup
                metric_data=metric_info_for_dim,  # Pass metric_info for DimMetric creation/lookup
            )
            tasks.append(task)

        if not tasks:
            logger.warning("No valid metrics found to save after processing.")
            return

        # Run all upsert tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = 0
        failure_count = 0
        for i, result in enumerate(results):
            metric_key = list(metrics_data.keys())[
                i
            ]  # Get corresponding metric_key, not ideal but works for logging
            if isinstance(result, Exception):
                logger.error(
                    f"Failed to save metric '{metric_key}' for period"
                    f" '{period_key}': {result}"
                )
                failure_count += 1
            elif (
                result is False
            ):  # Assuming upsert_metric_value returns False on logical failure
                logger.warning(
                    f"Saving metric '{metric_key}' for period '{period_key}'"
                    " reported failure (returned False)."
                )
                failure_count += 1
            else:
                success_count += 1

        logger.info(
            f"Finished saving metrics for period '{period_key}'. "
            f"Successfully saved: {success_count}, Failed: {failure_count}."
        )

    # -----------------------------------------------------------------------------
    # --- END METRICS PROCESSING FUNCTIONS ----------------------------------------
    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # --- BEGIN CHART DATA PROCESSING FUNCTIONS -----------------------------------
    # -----------------------------------------------------------------------------

    def _prepare_data_for_new_vs_updated_monthly(
        self, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """
        Prepare DataFrame for 'new_vs_updated' chart.
        'published_month' (e.g., "2024-01", "none") is converted to datetime.
        Valid, unique, sorted months are mapped to "January", "February", etc.

        Args:
            df (pd.DataFrame): Input DataFrame with 'published_month' (string) and 'status'.

        Returns:
            pd.DataFrame | None: Processed DataFrame with 'month_display_label' (Categorical),
                                 'status', and 'count'. Returns None if data is insufficient.
        """
        required_cols = ['published_month', 'status']
        if not all(c in df.columns for c in required_cols):
            logger.warning(
                f"Missing required columns (need {required_cols}) for"
                " 'new_vs_updated' chart."
            )
            return None

        for col in required_cols:
            if (
                df[col].empty or df[col].isna().all()
            ):  # Check if series itself is empty or all NaNs
                logger.warning(
                    f"Required column '{col}' is empty or contains all NaNs"
                    " for 'new_vs_updated' chart."
                )
                return None

        df_copy = df.copy()

        # Convert 'published_month' string to datetime objects.
        # "YYYY-MM" becomes YYYY-MM-01. "none" or "NaT" (string) become pd.NaT.
        df_copy['published_month_dt_obj'] = pd.to_datetime(
            df_copy['published_month'], errors='coerce'
        )

        # Filter out rows where 'published_month' could not be converted to a valid date (is NaT).
        # These rows cannot be meaningfully assigned to "January", "February", etc.
        df_copy = df_copy.dropna(subset=['published_month_dt_obj'])

        if df_copy.empty:
            logger.warning(
                "No valid 'published_month' data remains after filtering NaT"
                " values. Cannot generate chart data."
            )
            return None

        df_copy['status'] = pd.Categorical(
            df_copy['status'], categories=['new', 'updated'], ordered=False
        )

        # Group by the datetime object for correct chronological grouping
        agg_data = (
            df_copy.groupby(
                ['published_month_dt_obj', 'status'], observed=False
            )
            .size()
            .reset_index(name='count')
        )

        if agg_data.empty:
            logger.warning(
                "No data after aggregation for 'new_vs_updated' chart (post"
                " NaT filter)."
            )
            return None

        # Sort by the datetime objects to ensure chronological order of periods
        agg_data = agg_data.sort_values('published_month_dt_obj')

        # Get unique, sorted datetime objects. These now only contain valid dates.
        unique_dates = sorted(agg_data['published_month_dt_obj'].unique())

        date_to_label_map = {}
        month_display_names = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

        for i, date_val in enumerate(
            unique_dates
        ):  # date_val is a datetime object
            if i < len(month_display_names):
                date_to_label_map[date_val] = month_display_names[i]
            else:
                # Fallback for more than 12 periods (e.g., "Jan 2025")
                label = date_val.strftime('%b %Y')
                date_to_label_map[date_val] = label
                logger.warning(
                    f"More than {len(month_display_names)} unique periods for"
                    f" 'new_vs_updated' chart. Period {i+1} (date:"
                    f" {date_val.strftime('%Y-%m-%d')}) labeled as '{label}'."
                )

        if (
            not date_to_label_map
        ):  # Should not happen if unique_dates is populated
            logger.error(
                "Internal error: No unique dates found to create display"
                " labels, though aggregated data was present."
            )
            return None

        agg_data['month_display_label'] = agg_data[
            'published_month_dt_obj'
        ].map(date_to_label_map)

        # Ensure 'month_display_label' is categorical and ordered correctly for Plotly Express
        ordered_labels = [
            date_to_label_map[date]
            for date in unique_dates
            if date in date_to_label_map
        ]
        agg_data['month_display_label'] = pd.Categorical(
            agg_data['month_display_label'],
            categories=ordered_labels,
            ordered=True,
        )

        return agg_data[
            ['month_display_label', 'status', 'count']
        ]  # Return only necessary columns

    def _prepare_data_for_volume_by_severity_monthly(
        self, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """
        Prepare DataFrame for 'volume_by_severity_monthly' chart.
        'published_month' (e.g., "2024-01") is converted to datetime.
        Valid, unique, sorted months are mapped to "January", "February", etc.
        'severity_type' is made categorical based on self.severity_order.

        Args:
            df (pd.DataFrame): Input DataFrame with 'published_month' (string) and 'severity_type'.

        Returns:
            pd.DataFrame | None: DataFrame named `agg_data` with 'month_display_label',
                                 'severity_type', and 'count'. Returns None if data is insufficient.
        """
        required_cols = ['published_month', 'severity_type']
        if not all(c in df.columns for c in required_cols):
            logger.warning(
                f"Missing required columns (need {required_cols}) for"
                " 'volume_by_severity_monthly' chart."
            )
            return None

        for col in required_cols:
            if df[col].empty or df[col].isna().all():
                logger.warning(
                    f"Required column '{col}' is empty or contains all NaNs"
                    " for 'volume_by_severity_monthly' chart."
                )
                return None

        df_copy = df.copy()

        # Convert 'published_month' string to datetime objects
        df_copy['published_month_dt_obj'] = pd.to_datetime(
            df_copy['published_month'], errors='coerce'
        )
        df_copy = df_copy.dropna(subset=['published_month_dt_obj'])

        if df_copy.empty:
            logger.warning(
                "No valid 'published_month' data remains after filtering NaT"
                " values for 'volume_by_severity_monthly'."
            )
            return None

        # Ensure severity_type is categorical and ordered, handling potential NaNs
        df_copy['severity_type'] = pd.Categorical(
            df_copy['severity_type'].fillna('none'),
            categories=self.severity_order,  # Relies on self.severity_order
            ordered=True,
        )

        # Group by the datetime object and ordered severity
        agg_data = (
            df_copy.groupby(
                ['published_month_dt_obj', 'severity_type'], observed=False
            )
            .size()
            .reset_index(name='count')
        )

        if agg_data.empty:
            logger.warning(
                "No data after aggregation for 'volume_by_severity_monthly'"
                " chart."
            )
            return None

        # Sort by datetime then by the categorical severity_type
        agg_data = agg_data.sort_values(
            ['published_month_dt_obj', 'severity_type']
        )

        unique_dates = sorted(agg_data['published_month_dt_obj'].unique())
        date_to_label_map = {}
        month_display_names = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

        for i, date_val in enumerate(unique_dates):
            if i < len(month_display_names):
                date_to_label_map[date_val] = month_display_names[i]
            else:
                label = date_val.strftime('%b %Y')
                date_to_label_map[date_val] = label
                logger.warning(
                    f"More than {len(month_display_names)} unique periods for"
                    f" 'volume_by_severity_monthly'. Period {i+1} (date:"
                    f" {date_val.strftime('%Y-%m-%d')}) labeled as '{label}'."
                )

        if not date_to_label_map:
            logger.error(
                "Internal error: No unique dates found to create display"
                " labels for 'volume_by_severity_monthly'."
            )
            return None

        agg_data['month_display_label'] = agg_data[
            'published_month_dt_obj'
        ].map(date_to_label_map)

        ordered_labels = [
            date_to_label_map[date]
            for date in unique_dates
            if date in date_to_label_map
        ]
        agg_data['month_display_label'] = pd.Categorical(
            agg_data['month_display_label'],
            categories=ordered_labels,
            ordered=True,
        )

        # Ensure consistency in returned variable name and columns
        agg_data = agg_data[['month_display_label', 'severity_type', 'count']]
        return agg_data

    def _prepare_data_for_cvss_distribution(
        self, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """
        Prepare the DataFrame for aggregation into 'cvss_distribution' chart data.

        Args:
            df (pd.DataFrame): Input DataFrame containing 'cve_category' and 'cvss_score' columns.

        Returns:
            pd.DataFrame | None: DataFrame with 'cve_category', 'cvss_score', and 'count' columns.
                'cve_category' is a categorical column with 'kernel', 'usermode', 'firmware', and 'other' values.
                'cvss_score' is a categorical column with CVSS score ranges (e.g., '0-3', '4-6', etc.).
                'count' is the number of CVEs for each combination of 'cve_category' and 'cvss_score'.
                The DataFrame is sorted by 'cve_category' and 'cvss_score'.
                If the required columns are missing or empty, None is returned instead.
        """
        required_cols = ['cve_category', 'cvss_score']
        if not all(
            c in df.columns and df[c].notna().any() for c in required_cols
        ):
            logger.warning(
                f"Missing or empty required columns {required_cols} for"
                " 'cvss_distribution'."
            )
            return None
        df_copy = df.copy()
        # Ensure cve_category column is present and handle fillna before categorizing
        if 'cve_category' not in df_copy.columns:
            df_copy['cve_category'] = 'none'
        df_copy['cve_category'] = pd.Categorical(
            df_copy['cve_category'].fillna('none'),
            categories=self.category_order,
            ordered=False,
        )

        chart_data = df_copy[['cve_category', 'cvss_score']].dropna(
            subset=['cvss_score']
        )
        if chart_data.empty:
            logger.warning(
                "No valid CVSS scores found for 'cvss_distribution'."
            )
            return None
        chart_data = chart_data.sort_values(by='cve_category')
        return chart_data

    def _prepare_data_for_rce_proportion_monthly(
        self, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """
        Prepare the DataFrame for aggregation into 'rce_proportion_monthly' chart data.

        Args:
            df (pd.DataFrame): Input DataFrame containing 'published_month' and 'cve_category' columns.

        Returns:
            pd.DataFrame | None: DataFrame with 'published_month', 'cve_category', 'total_count', and 'rce_count' columns.
                'published_month' is the month of publication.
                'cve_category' is a categorical column with 'kernel', 'usermode', 'firmware', and 'other' values.
                'total_count' is the total number of CVEs for each 'published_month'.
                'rce_count' is the number of CVEs with 'cve_category' == 'remote_code_execution' for each 'published_month'.
                The DataFrame is sorted by 'published_month' and 'cve_category'.
                If the required columns are missing or empty, None is returned instead.
        """
        required_cols = ['published_month', 'cve_category']
        if not all(c in df.columns for c in required_cols):
            logger.warning(
                "Missing required columns {required_cols} for"
                " 'rce_proportion_monthly'."
            )
            return None
        total_monthly = (
            df.groupby('published_month', observed=False)
            .size()
            .rename('total_count')
        )
        rce_monthly = (
            df[df['cve_category'] == 'remote_code_execution']
            .groupby('published_month', observed=False)
            .size()
            .rename('rce_count')
        )
        agg_data = pd.merge(
            total_monthly, rce_monthly, on='published_month', how='left'
        ).fillna(0)
        agg_data['proportion'] = (
            agg_data['rce_count']
            / agg_data['total_count'].replace(0, np.nan)
            * 100
        ).fillna(
            0
        )  # Handle division by zero
        return agg_data.reset_index().sort_values('published_month')

    def _prepare_data_for_rce_cvss_distribution(
        self, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """
        Prepare the DataFrame for aggregation into 'rce_cvss_distribution' chart data.

        Args:
            df (pd.DataFrame): Input DataFrame containing 'cve_category' and 'cvss_score' columns.

        Returns:
            pd.DataFrame | None: DataFrame with 'cve_category', 'cvss_score', and 'count' columns.
                'cve_category' is a categorical column with 'kernel', 'usermode', 'firmware', and 'other' values.
                'cvss_score' is a categorical column with CVSS score ranges (e.g., '0-3', '4-6', etc.).
                'count' is the number of CVEs for each combination of 'cve_category' and 'cvss_score'.
                The DataFrame is sorted by 'cve_category' and 'cvss_score'.
                If the required columns are missing or empty, None is returned instead.
        """
        required_cols = ['cve_category', 'cvss_score']
        if not all(c in df.columns for c in required_cols):
            logger.warning(
                "Missing required columns {required_cols} for"
                " 'rce_cvss_distribution'."
            )
            return None
        chart_data = (
            df[df['cve_category'] == 'remote_code_execution'][['cvss_score']]
            .dropna()
            .copy()
        )
        if chart_data.empty:
            logger.warning(
                "No valid RCE CVSS scores found for 'rce_cvss_distribution'."
            )
            return None
        chart_data['category_label'] = 'Remote Code Execution'
        return chart_data

    def _prepare_data_for_eop_proportion_monthly(
        self, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """
        Prepare the DataFrame for aggregation into 'eop_proportion_monthly' chart data.

        Args:
            df (pd.DataFrame): Input DataFrame containing 'published_month' and 'cve_category' columns.

        Returns:
            pd.DataFrame | None: DataFrame with 'published_month', 'cve_category', 'total_count', and 'eop_count' columns.
                'published_month' is the month of publication.
                'cve_category' is a categorical column with 'kernel', 'usermode', 'firmware', and 'other' values.
                'total_count' is the total number of CVEs for each 'published_month'.
                'eop_count' is the number of CVEs with 'cve_category' == 'privilege_elevation' for each 'published_month'.
                The DataFrame is sorted by 'published_month' and 'cve_category'.
                If the required columns are missing or empty, None is returned instead.
        """
        required_cols = ['published_month', 'cve_category']
        if not all(c in df.columns for c in required_cols):
            logger.warning(
                "Missing required columns {required_cols} for"
                " 'eop_proportion_monthly'."
            )
            return None
        total_monthly = (
            df.groupby('published_month', observed=False)
            .size()
            .rename('total_count')
        )
        eop_monthly = (
            df[df['cve_category'] == 'privilege_elevation']
            .groupby('published_month', observed=False)
            .size()
            .rename('eop_count')
        )
        agg_data = pd.merge(
            total_monthly, eop_monthly, on='published_month', how='left'
        ).fillna(0)
        agg_data['proportion'] = (
            agg_data['eop_count']
            / agg_data['total_count'].replace(0, np.nan)
            * 100
        ).fillna(0)
        return agg_data.reset_index().sort_values('published_month')

    def _prepare_data_for_eop_by_attack_vector(
        self, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """
        Prepare the DataFrame for aggregation into 'eop_by_attack_vector' chart data.

        Args:
            df (pd.DataFrame): Input DataFrame containing 'cve_category' and 'attack_vector' columns.

        Returns:
            pd.DataFrame | None: DataFrame with 'cve_category', 'attack_vector', and 'count' columns.
                'cve_category' is a categorical column with 'kernel', 'usermode', 'firmware', and 'other' values.
                'attack_vector' is a categorical column with attack vector values.
                'count' is the number of CVEs for each combination of 'cve_category' and 'attack_vector'.
                The DataFrame is sorted by 'cve_category' and 'attack_vector'.
                If the required columns are missing or empty, None is returned instead.
        """
        required_cols = ['cve_category', 'attack_vector']
        if not all(c in df.columns for c in required_cols):
            logger.warning(
                "Missing required columns {required_cols} for"
                " 'eop_by_attack_vector'."
            )
            return None
        df_eop = df[df['cve_category'] == 'privilege_elevation'].copy()
        if df_eop.empty:
            logger.warning("No EoP CVEs found for 'eop_by_attack_vector'.")
            return None
        df_eop['attack_vector'] = pd.Categorical(
            df_eop['attack_vector'].fillna('none'),
            categories=self.attack_vector_order,
            ordered=True,
        )
        agg_data = (
            df_eop.groupby('attack_vector', observed=False)
            .size()
            .reset_index(name='count')
        )
        agg_data = (
            agg_data.set_index('attack_vector')
            .reindex(self.attack_vector_order, fill_value=0)
            .reset_index()
        )
        return agg_data

    def _prepare_data_for_dos_infodisc_monthly(
        self, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """
        Prepare the DataFrame for aggregation into 'dos_infodisc_monthly' chart data.

        Args:
            df (pd.DataFrame): Input DataFrame containing 'published_month' and 'cve_category' columns.

        Returns:
            pd.DataFrame | None: DataFrame with 'published_month', 'cve_category', 'total_count', and 'dos_infodisc_count' columns.
                'published_month' is the month of publication.
                'cve_category' is a categorical column with 'kernel', 'usermode', 'firmware', and 'other' values.
                'total_count' is the total number of CVEs for each 'published_month'.
                'dos_infodisc_count' is the number of CVEs with 'cve_category' in ['denial_of_service', 'disclosure'] for each 'published_month'.
                The DataFrame is sorted by 'published_month' and 'cve_category'.
                If the required columns are missing or empty, None is returned instead.
        """
        required_cols = ['published_month', 'cve_category']
        categories_to_plot = [
            'denial_of_service',
            'disclosure',
        ]  # disclosure was 'information_disclosure' earlier, ensure consistency
        if not all(c in df.columns for c in required_cols):
            logger.warning(
                "Missing required columns {required_cols} for"
                " 'dos_infodisc_monthly'."
            )
            return None
        df_filtered = df[df['cve_category'].isin(categories_to_plot)].copy()
        if df_filtered.empty:
            logger.warning(
                "No DoS or Info Disclosure CVEs found for"
                " 'dos_infodisc_monthly'."
            )
            return None
        df_filtered['category_display'] = df_filtered['cve_category'].replace(
            {'denial_of_service': 'DoS', 'disclosure': 'Info Disclosure'}
        )
        agg_data = (
            df_filtered.groupby(
                ['published_month', 'category_display'], observed=False
            )
            .size()
            .reset_index(name='count')
        )
        return agg_data.sort_values(['published_month', 'category_display'])

    def _prepare_data_for_user_interaction_proportion_monthly(
        self, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """
        Prepare the DataFrame for aggregation into 'user_interaction_proportion_monthly' chart data.

        Args:
            df (pd.DataFrame): Input DataFrame containing 'published_month' and 'user_interaction' columns.

        Returns:
            pd.DataFrame | None: DataFrame with 'published_month', 'user_interaction', 'total_count', and 'required_count' columns.
                'published_month' is the month of publication.
                'user_interaction' is a categorical column with 'required', 'none', and 'unknown' values.
                'total_count' is the total number of CVEs for each 'published_month'.
                'required_count' is the number of CVEs with 'user_interaction' == 'required' for each 'published_month'.
                The DataFrame is sorted by 'published_month' and 'user_interaction'.
                If the required columns are missing or empty, None is returned instead.
        """
        required_cols = ['published_month', 'user_interaction']
        if not all(c in df.columns for c in required_cols):
            logger.warning(
                "Missing required columns {required_cols} for"
                " 'user_interaction_proportion_monthly'."
            )
            return None
        total_monthly = (
            df.groupby('published_month', observed=False)
            .size()
            .rename('total_count')
        )
        required_monthly = (
            df[df['user_interaction'] == 'required']
            .groupby('published_month', observed=False)
            .size()
            .rename('required_count')
        )
        agg_data = pd.merge(
            total_monthly, required_monthly, on='published_month', how='left'
        ).fillna(0)
        agg_data['proportion_required'] = (
            agg_data['required_count']
            / agg_data['total_count'].replace(0, np.nan)
            * 100
        ).fillna(0)
        return agg_data.reset_index().sort_values('published_month')

    def _prepare_data_for_overall_attack_vector_dist(
        self, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """
        Prepare DataFrame for 'overall_attack_vector_dist' chart.
        Calculates percentage for each attack vector.

        Args:
            df (pd.DataFrame): Input DataFrame with 'attack_vector' column.

        Returns:
            pd.DataFrame | None: DataFrame named `agg_data` with 'attack_vector' (original key),
                                 'display_attack_vector' (for x-axis labels), and 'percentage'.
                                 Returns None if data is insufficient.
        """
        chart_key = "overall_attack_vector_dist"
        required_cols = ['attack_vector']
        if not all(
            c in df.columns for c in required_cols
        ):  # Check if col exists
            logger.warning(
                f"Missing required columns {required_cols} for '{chart_key}'."
            )
            return None

        # Explicitly copy to avoid SettingWithCopyWarning if df is a slice
        df_copy = df.copy()

        # Fill NaNs in 'attack_vector' with 'none' before counting, assuming 'none' is a defined category
        df_copy['attack_vector'] = df_copy['attack_vector'].fillna('none')

        # Ensure the column actually has non-NA data after potential fillna if all were NA
        if df_copy['attack_vector'].isna().all() or df_copy.empty:
            logger.warning(
                "Column 'attack_vector' is empty or all NaN after attempting"
                f" to fill for '{chart_key}'."
            )
            return None

        # Calculate percentage
        vc = df_copy['attack_vector'].value_counts(normalize=True).mul(100)

        # Reindex to ensure all categories from self.attack_vector_order are present, fill missing with 0
        # This uses self.attack_vector_order for the keys
        vc = vc.reindex(self.attack_vector_order, fill_value=0.0)

        agg_data = vc.reset_index()
        agg_data.columns = ['attack_vector', 'percentage']

        # Create display names for x-axis labels
        # Fallback to original key if not in display_map
        mapped_av = agg_data['attack_vector'].map(
            self.attack_vector_display_map
        )
        agg_data['display_attack_vector'] = mapped_av.fillna(
            agg_data['attack_vector'].astype(object)
        )

        # Make 'attack_vector' (original key) categorical for sorting and color mapping
        agg_data['attack_vector'] = pd.Categorical(
            agg_data['attack_vector'],
            categories=self.attack_vector_order,
            ordered=True,
        )
        # Make 'display_attack_vector' categorical based on the order of original keys for consistent x-axis plotting
        ordered_display_labels = [
            self.attack_vector_display_map.get(key, key)
            for key in self.attack_vector_order
        ]
        agg_data['display_attack_vector'] = pd.Categorical(
            agg_data['display_attack_vector'],
            categories=ordered_display_labels,
            ordered=True,
        )

        agg_data = agg_data.sort_values(
            'attack_vector'
        )  # Sort by the original key's categorical order

        return agg_data[
            ['attack_vector', 'display_attack_vector', 'percentage']
        ]

    def _prepare_data_for_overall_attack_complexity_dist(
        self, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """
        Prepare the DataFrame for aggregation into 'overall_attack_complexity_dist' chart data.

        Args:
            df (pd.DataFrame): Input DataFrame containing 'attack_complexity' column.

        Returns:
            pd.DataFrame | None: DataFrame with 'attack_complexity' and 'percentage' columns.
                'attack_complexity' is a categorical column with attack complexity values.
                'percentage' is the percentage of CVEs for each 'attack_complexity'.
                The DataFrame is sorted by 'attack_complexity'.
                If the required columns are missing or empty, None is returned instead.
        """
        required_cols = ['attack_complexity']
        if not all(
            c in df.columns and df[c].notna().any() for c in required_cols
        ):
            logger.warning(
                "Missing or empty required columns {required_cols} for"
                " 'overall_attack_complexity_dist'."
            )
            return None
        df_filtered = df.dropna(subset=['attack_complexity'])
        if df_filtered.empty:
            return None
        vc = df_filtered['attack_complexity'].value_counts(normalize=True)
        vc = vc.reindex(self.attack_complexity_order, fill_value=0) * 100
        agg_data = vc.reset_index()
        agg_data.columns = ['attack_complexity', 'percentage']
        agg_data['attack_complexity'] = pd.Categorical(
            agg_data['attack_complexity'],
            categories=self.attack_complexity_order,
            ordered=True,
        )
        return agg_data.sort_values('attack_complexity')

    def _prepare_data_for_patch_delay_risk_scatter(
        self, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """
        Prepare the DataFrame for aggregation into 'patch_delay_risk_scatter' chart data.

        Args:
            df (pd.DataFrame): Input DataFrame containing 'published', 'patch_date', 'cvss_score', 'cve_category', and 'cve_id' columns.

        Returns:
            pd.DataFrame | None: DataFrame with 'cve_id', 'days_to_patch', 'cvss_score', and 'cve_category' columns.
                'cve_id' is the CVE ID.
                'days_to_patch' is the number of days between 'published' and 'patch_date'.
                'cvss_score' is the CVSS score of the CVE.
                'cve_category' is the category of the CVE.
                If the required columns are missing or empty, None is returned instead.
        """
        required_cols = [
            'published',
            'patch_date',
            'cvss_score',
            'cve_category',
            'cve_id',
        ]
        if not all(
            c in df.columns and df[c].notna().any() for c in required_cols
        ):
            logger.warning(
                "Missing or empty required columns {required_cols} for"
                " 'patch_delay_risk_scatter'."
            )
            return None
        df_copy = (
            df.copy()
        )  # Work on a copy to avoid modifying original df passed around

        # Check for essential columns, allow others to be potentially missing but handle them
        if (
            'published' not in df_copy.columns
            or 'patch_date' not in df_copy.columns
        ):
            logger.warning(
                "Missing 'published' or 'patch_date' for"
                " 'patch_delay_risk_scatter'. Cannot calculate delay."
            )
            return None

        for col in ['cvss_score', 'cve_category', 'cve_id']:
            if col not in df_copy.columns:
                logger.warning(
                    "Column '{col}' missing for 'patch_delay_risk_scatter'."
                    " Filling with defaults."
                )
                if col == 'cvss_score':
                    df_copy[col] = np.nan
                elif col == 'cve_category':
                    df_copy[col] = 'none'  # Not Categorized
                elif col == 'cve_id':
                    df_copy[col] = 'cve-idx-' + pd.Series(
                        df_copy.index
                    ).astype(str)

        valid_dates_mask = (
            df_copy['patch_date'].notna() & df_copy['published'].notna()
        )
        df_copy['days_to_patch'] = np.nan
        if valid_dates_mask.any():
            df_copy.loc[valid_dates_mask, 'days_to_patch'] = (
                df_copy.loc[valid_dates_mask, 'patch_date']
                - df_copy.loc[valid_dates_mask, 'published']
            ).dt.days
        df_copy.loc[df_copy['days_to_patch'] < 0, 'days_to_patch'] = 0

        chart_data = df_copy[
            ['cve_id', 'days_to_patch', 'cvss_score', 'cve_category']
        ].copy()
        chart_data.dropna(subset=['days_to_patch', 'cvss_score'], inplace=True)

        if chart_data.empty:
            logger.warning(
                "No valid data points found for 'patch_delay_risk_scatter'."
            )
            return None
        chart_data['cve_category'] = pd.Categorical(
            chart_data['cve_category'].fillna('none'),
            categories=self.category_order,
            ordered=False,
        )
        return chart_data

    def _prepare_data_for_overall_category_distribution_chart(
        self, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """
        Prepares data for a treemap of overall CVE category distribution.
        Uses self.category_display_name_map to create display labels.

        Args:
            df (pd.DataFrame): Input DataFrame containing 'cve_category' column.

        Returns:
            pd.DataFrame | None: DataFrame named `agg_data` with 'cve_category' (original),
                                 'display_category' (user-friendly), and 'count'.
                                 Returns None if data is insufficient.
        """
        chart_key = "overall_category_distribution"
        required_cols = ['cve_category']
        if not all(
            c in df.columns and df[c].notna().any() for c in required_cols
        ):  # Check if col exists and has at least one non-NA value
            logger.warning(
                f"Missing or all-NaN required columns {required_cols} for"
                f" '{chart_key}'."
            )
            return None

        df_copy = df.copy()

        # Ensure 'cve_category' exists, fill NaNs with 'none' before counting
        # This assumes 'none' is a valid category key in your maps
        df_copy['cve_category'] = df_copy['cve_category'].fillna('none')

        # Count CVEs per category
        category_counts = df_copy['cve_category'].value_counts().reset_index()
        category_counts.columns = ['cve_category', 'count']

        # Map to display names. Use original if no mapping exists.
        mapped_display_cat = category_counts['cve_category'].map(
            self.category_display_name_map
        )
        category_counts['display_category'] = mapped_display_cat.fillna(
            category_counts['cve_category'].astype(object)
        )

        # Make 'cve_category' categorical for sorting and consistent color mapping, using original keys
        # This ensures that color_discrete_map uses the original snake_case keys
        category_counts['cve_category'] = pd.Categorical(
            category_counts['cve_category'],
            categories=self.category_order,  # Use self.category_order for consistent treemap layout
            ordered=True,
        )

        # Sort by the categorical 'cve_category' to influence treemap block order somewhat,
        # though treemap algorithm primarily uses 'count'.
        # Or sort by 'count' descending if you want largest blocks first regardless of category order.
        category_counts = category_counts.sort_values(
            by='count', ascending=False
        )
        # category_counts = category_counts.sort_values(by='cve_category') # Alternative sorting

        if category_counts.empty or category_counts['count'].sum() == 0:
            logger.warning(
                f"No category data with counts found for '{chart_key}'."
            )
            return None

        # Ensure consistent return variable name
        agg_data = category_counts[
            ['cve_category', 'display_category', 'count']
        ]
        return agg_data

    def _prepare_data_for_attack_vectors_by_privileges_chart(
        self, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """
        Prepares data for stacked bar: Attack Vectors segmented by Privileges Required.
        Uses display maps for categorical labels if available.

        Args:
            df (pd.DataFrame): Input DataFrame with 'attack_vector' and 'privileges_required'.

        Returns:
            pd.DataFrame | None: DataFrame `agg_data` with 'attack_vector' (original key),
                                 'display_attack_vector', 'privileges_required' (original key),
                                 'display_privileges_required', and 'count'.
        """
        chart_key = "attack_vectors_by_privileges"
        required_cols = ['attack_vector', 'privileges_required']

        if not all(c in df.columns for c in required_cols):
            logger.warning(
                f"Missing required columns {required_cols} for '{chart_key}'."
            )
            return None

        df_copy = df.copy()

        # Fill NaNs before converting to categorical
        df_copy['attack_vector'] = df_copy['attack_vector'].fillna('none')
        df_copy['privileges_required'] = df_copy['privileges_required'].fillna(
            'none'
        )

        if (
            df_copy['attack_vector'].isna().all()
            or df_copy['privileges_required'].isna().all()
        ):
            logger.warning(
                "One or more required columns are all NaN after fillna for"
                f" '{chart_key}'."
            )
            return None

        # Ensure categorical for ordering and complete categories
        df_copy['attack_vector'] = pd.Categorical(
            df_copy['attack_vector'],
            categories=self.attack_vector_order,
            ordered=True,
        )
        df_copy['privileges_required'] = pd.Categorical(
            df_copy['privileges_required'],
            categories=self.privileges_required_order,
            ordered=True,  # Use defined order
        )

        agg_data = (
            df_copy.groupby(
                ['attack_vector', 'privileges_required'], observed=False
            )
            .size()
            .reset_index(name='count')
        )

        if agg_data.empty or agg_data['count'].sum() == 0:
            logger.warning(
                f"No data with counts after aggregation for '{chart_key}'."
            )
            return None

        # Create display names for x-axis and legend
        mapped_av = agg_data['attack_vector'].map(
            self.attack_vector_display_map
        )
        agg_data['display_attack_vector'] = mapped_av.fillna(
            agg_data['attack_vector'].astype(object)
        )  # Ensure fillna value is not categorical
        mapped_pr = agg_data['privileges_required'].map(
            self.privileges_required_display_map
        )
        agg_data['display_privileges_required'] = mapped_pr.fillna(
            agg_data['privileges_required'].astype(object)
        )  # or .astype(str)
        # Make display columns categorical based on the order of their original keys
        ordered_av_display_labels = [
            self.attack_vector_display_map.get(key, key)
            for key in self.attack_vector_order
        ]
        agg_data['display_attack_vector'] = pd.Categorical(
            agg_data['display_attack_vector'],
            categories=ordered_av_display_labels,
            ordered=True,
        )
        ordered_pr_display_labels = [
            self.privileges_required_display_map.get(key, key)
            for key in self.privileges_required_order
        ]
        agg_data['display_privileges_required'] = pd.Categorical(
            agg_data['display_privileges_required'],
            categories=ordered_pr_display_labels,
            ordered=True,
        )

        # Sort by the original categorical order
        agg_data = agg_data.sort_values(
            ['attack_vector', 'privileges_required']
        )

        return agg_data[[
            'attack_vector',
            'display_attack_vector',
            'privileges_required',
            'display_privileges_required',
            'count',
        ]]

    def _prepare_data_for_interaction_vs_complexity_chart(
        self, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """
        Prepares data for heatmap: User Interaction vs. Attack Complexity.
        Filters out 'none' for attack_complexity.
        """
        chart_key = "interaction_vs_complexity"
        required_cols = [
            'user_interaction',
            'attack_complexity',
            'cvss_score',
            'cve_id',
        ]

        if not all(c in df.columns for c in required_cols):
            logger.warning(
                f"Missing required columns {required_cols} for '{chart_key}'."
            )
            return None

        df_copy = df.copy()

        # --- Filter out 'none' for attack_complexity ---
        df_copy = df_copy[df_copy['attack_complexity'] != 'none']
        if df_copy.empty:
            logger.info(
                "No data remains after filtering 'none' attack_complexity for"
                f" '{chart_key}'."
            )
            return None

        # Fill NaNs before converting to categorical
        # Assuming 'none' for user_interaction is valid and 'low'/'high' for complexity
        df_copy['user_interaction'] = df_copy['user_interaction'].fillna(
            'none'
        )
        df_copy['attack_complexity'] = df_copy['attack_complexity'].fillna(
            'low'
        )  # Default if NaN after filtering 'none'

        # Ensure all required columns have some non-NA data after filtering and filling
        if not all(
            df_copy[c].notna().any()
            for c in required_cols
            if c in df_copy.columns
        ):
            logger.warning(
                "One or more required columns are all NaN after processing"
                f" for '{chart_key}'."
            )
            return None

        df_copy['user_interaction'] = pd.Categorical(
            df_copy['user_interaction'],
            categories=self.user_interaction_order,
            ordered=True,
        )
        df_copy['attack_complexity'] = pd.Categorical(
            df_copy['attack_complexity'],
            categories=self.attack_complexity_order,
            ordered=True,  # Only 'low', 'high'
        )

        agg_data = (
            df_copy.groupby(
                ['user_interaction', 'attack_complexity'], observed=False
            )
            .agg(count=('cve_id', 'size'), avg_cvss=('cvss_score', 'mean'))
            .reset_index()
        )

        agg_data['avg_cvss'] = agg_data['avg_cvss'].fillna(0).round(1)

        if agg_data.empty or agg_data['count'].sum() == 0:
            logger.warning(
                f"No data with counts after aggregation for '{chart_key}'."
            )
            return None

        # Add display columns for axis labels

        mapped_ui = agg_data['user_interaction'].map(
            self.user_interaction_display_map
        )
        agg_data['display_user_interaction'] = mapped_ui.fillna(
            agg_data['user_interaction'].astype(object)
        )

        mapped_ac = agg_data['attack_complexity'].map(
            self.attack_complexity_display_map
        )
        agg_data['display_attack_complexity'] = mapped_ac.fillna(
            agg_data['attack_complexity'].astype(object)
        )

        return agg_data  # user_interaction, attack_complexity, display_user_interaction, display_attack_complexity, count, avg_cvss

    def _prepare_data_for_kev_by_category(
        self, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        """
        Prepares data for bar chart: Known Exploited Vulns by Category.

        Args:
            df (pd.DataFrame): Input DataFrame containing 'cve_category' and 'cisa_kev' columns.

        Returns:
            pd.DataFrame | None: DataFrame with 'cve_category' and 'count' columns.
                'cve_category' is a categorical column with category values.
                'count' is the number of CVEs for each category.
                The DataFrame is sorted by 'cve_category'.
                If the required columns are missing or empty, None is returned instead.
        """
        chart_key = "kev_by_category"
        required_cols = [
            'cve_category',
            'cisa_kev',
        ]  # Assumes 'cisa_kev' is boolean or 0/1
        if not all(
            c in df.columns for c in required_cols
        ):  # Not checking notna for cisa_kev, as it might be sparse
            logger.warning(
                f"Missing required columns {required_cols} for '{chart_key}'."
            )
            return None
        if df['cisa_kev'].notna().sum() == 0:  # No KEV data points
            logger.warning(f"No CISA KEV data points found for '{chart_key}'.")
            return pd.DataFrame(
                columns=['cve_category', 'kev_count']
            )  # Return empty DF with correct columns

        df_copy = df[
            df['cisa_kev'] == True
        ].copy()  # Filter for exploited CVEs
        if df_copy.empty:
            logger.warning(f"No CVEs marked as KEV for '{chart_key}'.")
            return pd.DataFrame(columns=['cve_category', 'kev_count'])

        df_copy['cve_category'] = pd.Categorical(
            df_copy['cve_category'].fillna('none'),
            categories=self.category_order,
            ordered=True,
        )
        agg_data = df_copy['cve_category'].value_counts().reset_index()
        agg_data.columns = ['cve_category', 'kev_count']

        # Ensure all categories are present for consistent chart axis
        agg_data = (
            agg_data.set_index('cve_category')
            .reindex(self.category_order, fill_value=0)
            .reset_index()
        )
        return agg_data.sort_values(by='kev_count', ascending=False)

    def _prepare_data_for_cvss_score_by_category_chart(
        self, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        chart_key = (  # Renamed for clarity from generic key
            "cvss_score_by_category_box"
        )
        required_cols = ['cve_category', 'cvss_score']
        if not all(c in df.columns for c in required_cols):
            logger.warning(
                f"Missing required columns {required_cols} for '{chart_key}'."
            )
            return None

        df_copy = df.copy()

        df_copy['cve_category'] = df_copy['cve_category'].fillna('none')
        # Ensure CVSS scores are numeric, coercing errors
        df_copy['cvss_score'] = pd.to_numeric(
            df_copy['cvss_score'], errors='coerce'
        )

        # Drop rows where cvss_score is NaN after coercion, or cve_category is NaN (though filled)
        chart_data = df_copy.dropna(subset=['cve_category', 'cvss_score'])

        if chart_data.empty:
            logger.warning(
                f"No valid data after processing for '{chart_key}'."
            )
            return None

        # Add display_cve_category
        mapped_cve_category = chart_data['cve_category'].map(
            self.category_display_name_map
        )
        chart_data['display_cve_category'] = mapped_cve_category.fillna(
            chart_data['cve_category'].astype(object)
        )

        # Make display_cve_category categorical for faceting order
        ordered_cat_display = [
            self.category_display_name_map.get(k, k)
            for k in self.category_order
        ]
        chart_data['display_cve_category'] = pd.Categorical(
            chart_data['display_cve_category'],
            categories=ordered_cat_display,
            ordered=True,
        )
        # Also make original cve_category categorical if it's used for any logic prior to display mapping
        chart_data['cve_category'] = pd.Categorical(
            chart_data['cve_category'],
            categories=self.category_order,
            ordered=True,
        )

        return chart_data[
            ['cve_category', 'display_cve_category', 'cvss_score']
        ].sort_values(by='display_cve_category')

    def _get_cwe_display_label(
        self, cwe_id_str: str, max_name_len: int = 40
    ) -> str:  # Increased max_name_len
        """Helper to get a formatted CWE label (ID: Name)."""
        details = self.CWE_DETAILS.get(str(cwe_id_str).strip(), {})
        name = details.get('name', 'Unknown CWE Name')
        if len(name) > max_name_len:
            name = name[: max_name_len - 3] + "..."
        return f"{str(cwe_id_str).strip()}: {name}"

    def _prepare_data_for_top_cwe_spotlight_cards(
        self, df: pd.DataFrame, top_n: int = 6
    ) -> list[dict] | None:
        """
        Prepares a list of dictionaries for Top N CWEs, suitable for rendering as 'stat cards'.
        Focuses on Critical/Important vulnerabilities across ALL categories.
        """
        chart_key = "top_cwe_spotlight_cards_data"
        required_cols = [
            'cwe_id',
            'severity_type',
            'cve_category',
            'cvss_score',
            'cve_id',
        ]

        if not all(c in df.columns for c in required_cols):
            missing_cols = [
                col for col in required_cols if col not in df.columns
            ]
            logger.warning(
                f"Missing required columns {missing_cols} for '{chart_key}'."
            )
            return None

        df_copy = df[required_cols].copy()

        high_risk_filter = df_copy['severity_type'].isin(
            ['critical', 'important']
        )
        df_high_risk = df_copy[high_risk_filter]

        if df_high_risk.empty:
            logger.warning(
                f"No Critical/Important CVEs found for '{chart_key}'."
            )
            return []

        # --- CWE ID Processing (assuming cwe_id can be comma/semicolon separated string) ---
        # Explode and take the first CWE ID as primary if multiple are listed
        df_cwe_processed = df_high_risk.assign(
            primary_cwe_id=df_high_risk['cwe_id']
            .astype(str)
            .str.split('[,;]\s*')
            .str[0]
            .str.strip()
        )
        df_cwe_processed = df_cwe_processed[
            df_cwe_processed['primary_cwe_id'] != ''
        ]
        df_cwe_processed.dropna(subset=['primary_cwe_id'], inplace=True)
        # --- End CWE ID Processing ---

        if df_cwe_processed.empty:
            logger.warning(
                "No valid primary CWE IDs in Critical/Important CVEs for"
                f" '{chart_key}'."
            )
            return []

        # Ensure cvss_score is numeric for aggregation
        df_cwe_processed['cvss_score'] = pd.to_numeric(
            df_cwe_processed['cvss_score'], errors='coerce'
        )

        cwe_agg = (
            df_cwe_processed.groupby('primary_cwe_id')
            .agg(
                count=('cve_id', 'nunique'),  # Count unique CVEs per CWE
                avg_cvss=('cvss_score', 'mean'),
                # Get the mode for cve_category. If ties, mode() returns multiple; take first.
                cve_category_predominant=(
                    'cve_category',
                    lambda x: x.mode()[0] if not x.mode().empty else 'none',
                ),
            )
            .reset_index()
        )

        # Sort by count (desc) then by avg_cvss (desc) to break ties for top N
        top_n_cwes_df = cwe_agg.sort_values(
            by=['count', 'avg_cvss'], ascending=[False, False]
        ).head(top_n)

        if top_n_cwes_df.empty:
            logger.warning(
                f"No CWEs to display after aggregation for '{chart_key}'."
            )
            return []

        # Prepare list of dictionaries for stat cards
        spotlight_cards_data = []
        for _, row in top_n_cwes_df.iterrows():
            cwe_id = str(row['primary_cwe_id'])
            pred_cat_raw = row['cve_category_predominant']

            card = {
                "cwe_id_raw": cwe_id,
                "display_cwe_label": self._get_cwe_display_label(
                    cwe_id
                ),  # e.g., "CWE-79: Cross-site Scripting"
                "count": int(row['count']),
                "avg_cvss": (
                    f"{row['avg_cvss']:.1f}"
                    if pd.notna(row['avg_cvss'])
                    else "N/A"
                ),
                "predominant_category_raw": pred_cat_raw,
                "display_predominant_category": (
                    self.category_display_name_map.get(
                        pred_cat_raw, pred_cat_raw.replace('_', ' ').title()
                    )
                ),
            }
            spotlight_cards_data.append(card)

        return spotlight_cards_data

    def _prepare_data_for_top_cwe_in_high_risk_cves_chart(
        self, df: pd.DataFrame, top_n: int = 6
    ) -> list[dict] | None:
        """
        Prepares data for Top N CWEs in Critical/Important vulnerabilities (all categories).
        Not used currently, but left in case it's needed in the future.
        """
        chart_key = "top_cwe_in_high_risk_all_cat"  # Updated key
        required_cols = [
            'cwe_id',
            'severity_type',
            'cve_category',
            'cvss_score',
            'cve_id',
        ]  # cve_id for unique count
        if not all(c in df.columns for c in required_cols):
            logger.warning(
                f"Missing required columns {required_cols} for '{chart_key}'."
            )
            return None

        df_copy = df.copy()

        # Filter for Critical/Important CVEs (adjust severity terms if yours are different)
        high_risk_filter = df_copy['severity_type'].isin(['critical', 'high'])
        df_high_risk = df_copy[high_risk_filter]

        if df_high_risk.empty:
            logger.warning(
                f"No Critical/Important CVEs found for '{chart_key}'."
            )
            # Return empty DataFrame with expected columns for graceful failure in plotting
            return pd.DataFrame(
                columns=[
                    'primary_cwe_id',
                    'display_cwe_label',
                    'count',
                    'avg_cvss',
                    'cve_category_predominant',
                    'display_cve_category_predominant',
                ]
            )

        # --- CWE ID Processing (assuming cwe_id can be comma/semicolon separated string) ---
        # Create a series of lists from the cwe_id string
        df_cwe_exploded = df_high_risk.assign(
            primary_cwe_id_list=df_high_risk['cwe_id'].str.split('[,;]\s*')
        ).explode('primary_cwe_id_list')

        # Use the first CWE ID from the exploded list as the 'primary_cwe_id'
        # or handle cases where 'cwe_id' might not be a string or not splittable
        # For simplicity, let's assume if it's not a string list, it's a single ID.
        # A more robust way is to ensure cwe_id is always a list first, then explode.
        # Here, let's take the first one if it was a list, or the original if not.
        df_cwe_exploded['primary_cwe_id'] = df_cwe_exploded[
            'primary_cwe_id_list'
        ].str.strip()
        # --- End CWE ID Processing ---

        df_cwe_exploded = df_cwe_exploded.dropna(subset=['primary_cwe_id'])
        df_cwe_exploded = df_cwe_exploded[
            df_cwe_exploded['primary_cwe_id'] != ''
        ]  # Remove empty strings

        if df_cwe_exploded.empty:
            logger.warning(
                "No valid primary CWE IDs in Critical/Important CVEs for"
                f" '{chart_key}'."
            )
            return pd.DataFrame(
                columns=[
                    'primary_cwe_id',
                    'display_cwe_label',
                    'count',
                    'avg_cvss',
                    'cve_category_predominant',
                    'display_cve_category_predominant',
                ]
            )

        cwe_agg = (
            df_cwe_exploded.groupby('primary_cwe_id')
            .agg(
                count=('cve_id', 'nunique'),  # Count unique CVE IDs per CWE
                avg_cvss=('cvss_score', 'mean'),
                cve_category_predominant=(
                    'cve_category',
                    lambda x: x.mode()[0] if not x.mode().empty else 'none',
                ),
            )
            .reset_index()
        )

        top_n_cwes_df = cwe_agg.sort_values(by='count', ascending=False).head(
            top_n
        )

        if top_n_cwes_df.empty:
            logger.warning(
                f"No CWEs to display after aggregation for '{chart_key}'."
            )
            return pd.DataFrame(
                columns=[
                    'primary_cwe_id',
                    'display_cwe_label',
                    'count',
                    'avg_cvss',
                    'cve_category_predominant',
                    'display_cve_category_predominant',
                ]
            )

        # Create a more readable label for Y-axis (CWE ID: Name)
        def get_cwe_display_label(cwe_id_str):
            details = self.CWE_DETAILS.get(str(cwe_id_str).strip(), {})
            name = details.get('name', 'Unknown Name')
            # Keep it relatively short for y-axis
            return f"{str(cwe_id_str).strip()}: {name.split('(')[0].strip()}"

        top_n_cwes_df['display_cwe_label'] = top_n_cwes_df[
            'primary_cwe_id'
        ].apply(get_cwe_display_label)

        # Add display name for predominant category
        top_n_cwes_df['display_cve_category_predominant'] = (
            top_n_cwes_df['cve_category_predominant']
            .map(self.category_display_name_map)
            .fillna(top_n_cwes_df['cve_category_predominant'])
        )

        # Make display columns categorical for consistent ordering in plot
        # Order Y-axis by count descending (achieved by sorting then passing list to category_orders)
        ordered_cwe_display_labels = top_n_cwes_df.sort_values(
            'count', ascending=True
        )[
            'display_cwe_label'
        ].tolist()  # Ascending for y-axis order
        top_n_cwes_df['display_cwe_label'] = pd.Categorical(
            top_n_cwes_df['display_cwe_label'],
            categories=ordered_cwe_display_labels,
            ordered=True,
        )

        ordered_predominant_cat_display = [
            self.category_display_name_map.get(k, k)
            for k in self.category_order
        ]
        top_n_cwes_df['display_cve_category_predominant'] = pd.Categorical(
            top_n_cwes_df['display_cve_category_predominant'],
            categories=ordered_predominant_cat_display,
            ordered=True,
        )

        return top_n_cwes_df.sort_values(by='count', ascending=False)

    def _prepare_data_for_top_windows_versions_by_criticality_chart(
        self, df: pd.DataFrame, top_n: int = 10
    ) -> pd.DataFrame | None:
        """Prepares data for Top Affected Windows Versions, stacked by severity.
        Handles exploding 'products' list internally.
        """
        chart_key = "top_windows_versions_by_criticality"
        # Required columns in the input 'df'
        required_cols = ['products', 'severity_type', 'cve_id']

        if not all(c in df.columns for c in required_cols):
            missing_cols = [
                col for col in required_cols if col not in df.columns
            ]
            logger.warning(
                f"Missing columns {missing_cols} from input df for"
                f" '{chart_key}'."
            )
            return None

        # Ensure severity_type is present and has some data
        if (
            'severity_type' not in df.columns
            or df['severity_type'].isna().all()
        ):
            logger.warning(
                "Column 'severity_type' is missing or all NaN for"
                f" '{chart_key}'."
            )
            return None

        df_input_for_helper = df[required_cols].copy()

        # --- Step 1 & 2: Use helper to get exploded and standardized products ---
        # The helper function handles explosion and standardization.
        # It expects 'products' as the column name by default.
        df_exploded = self._get_exploded_standardized_products(
            df_input_for_helper, product_column_name='products'
        )

        if df_exploded is None or df_exploded.empty:
            logger.warning(
                "Helper '_get_exploded_standardized_products' returned None"
                f" or empty DataFrame for '{chart_key}'."
            )
            return (
                pd.DataFrame()
            )  # Or return pd.DataFrame() if an empty DF is expected by callers

        # Rename the 'standardized_product' column from the helper's output
        # to 'windows_edition_version_arch' for compatibility with subsequent logic.
        if 'standardized_product' in df_exploded.columns:
            df_exploded.rename(
                columns={
                    'standardized_product': 'windows_edition_version_arch'
                },
                inplace=True,
            )
        else:
            logger.warning(
                "Expected 'standardized_product' column not found in"
                f" DataFrame returned by helper for '{chart_key}'."
            )
            return pd.DataFrame()

        # Filter out any rows where standardization might have resulted in a generic "Unknown Product Version" if desired,
        # or keep them to see how many CVEs affect unknown/unparsed products. For now, keep them.
        # df_exploded = df_exploded[df_exploded['windows_edition_version_arch'] != "Unknown Product Version"]

        if (
            df_exploded.empty
            or 'windows_edition_version_arch' not in df_exploded.columns
            or df_exploded['windows_edition_version_arch'].isna().all()
        ):
            logger.warning(
                "No valid 'windows_edition_version_arch' after processing"
                f" with helper for '{chart_key}'."
            )
            return pd.DataFrame()

        # Now df_exploded contains 'windows_edition_version_arch', 'severity_type', 'cve_id' (and original 'products')

        # Map severities for display and grouping
        severity_map = {
            'critical': 'Critical',
            'high': 'High',
            'medium': 'Medium/Low',
            'low': 'Medium/Low',
            'none': 'Other/None',
        }
        df_exploded['display_severity'] = (
            df_exploded['severity_type'].map(severity_map).fillna('Other/None')
        )

        display_severity_order = [
            'Critical',
            'High',
            'Medium/Low',
            'Other/None',
        ]
        df_exploded['display_severity'] = pd.Categorical(
            df_exploded['display_severity'],
            categories=display_severity_order,
            ordered=True,
        )

        # Group by the standardized windows_edition_version_arch and display_severity
        # Count unique CVEs for each product version - severity combination
        product_severity_counts_df = (
            df_exploded.groupby(
                ['windows_edition_version_arch', 'display_severity'],
                observed=False,
            )['cve_id']
            .nunique()
            .unstack(fill_value=0)
        )

        if product_severity_counts_df.empty:
            logger.warning(
                f"No product data after grouping for '{chart_key}'."
            )
            return None

        for sev_col in display_severity_order:
            if sev_col not in product_severity_counts_df.columns:
                product_severity_counts_df[sev_col] = 0

        product_severity_counts_df['critical_high_sum'] = (
            product_severity_counts_df.get('Critical', 0)
            + product_severity_counts_df.get('High', 0)
        )

        top_n_products_df = product_severity_counts_df.sort_values(
            by=[
                'critical_high_sum',
                'Critical',
                'High',
            ],  # Secondary sort by Critical, then High
            ascending=[False, False, False],
        ).head(top_n)

        if top_n_products_df.empty:
            logger.warning(
                f"No top N product data to display for '{chart_key}'."
            )
            return None

        columns_to_melt = [
            col
            for col in display_severity_order
            if col in top_n_products_df.columns
        ]
        chart_data = (
            top_n_products_df[columns_to_melt]
            .reset_index()
            .melt(
                id_vars='windows_edition_version_arch',
                var_name='severity_group',
                value_name='count',
            )
        )

        chart_data['severity_group'] = pd.Categorical(
            chart_data['severity_group'],
            categories=display_severity_order,
            ordered=True,
        )

        ordered_y_labels = top_n_products_df.index.tolist()[::-1]
        chart_data['windows_edition_version_arch'] = pd.Categorical(
            chart_data['windows_edition_version_arch'],
            categories=ordered_y_labels,
            ordered=True,
        )

        return chart_data.sort_values(
            ['windows_edition_version_arch', 'severity_group']
        )

    def _prepare_data_for_days_to_nvd_by_severity_boxplots(
        self, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        chart_key = "days_to_nvd_by_severity_vbox"
        required_cols = ['published', 'patch_date', 'severity_type']
        if not all(c in df.columns for c in required_cols):
            logger.warning(
                f"Missing one or more required columns {required_cols} for"
                f" '{chart_key}'."
            )
            return None

        df_copy = df.copy()

        # Convert to datetime, coercing errors
        df_copy['published'] = pd.to_datetime(
            df_copy['published'], errors='coerce'
        )
        df_copy['patch_date'] = pd.to_datetime(
            df_copy['patch_date'], errors='coerce'
        )

        # Calculate days_to_nvd. Allow negative values.
        valid_dates_mask = (
            df_copy['patch_date'].notna() & df_copy['published'].notna()
        )
        if not valid_dates_mask.any():
            logger.warning(
                "No valid date pairs (published, patch_date) for"
                f" '{chart_key}'."
            )
            return None

        df_copy['days_to_nvd'] = np.nan
        df_copy.loc[valid_dates_mask, 'days_to_nvd'] = (
            df_copy.loc[valid_dates_mask, 'patch_date']
            - df_copy.loc[valid_dates_mask, 'published']
        ).dt.days

        # Fill NaN severity_type with 'none' before categorization
        df_copy['severity_type'] = df_copy['severity_type'].fillna('none')

        # Drop rows where days_to_nvd could not be calculated or severity is still NaN
        chart_data = df_copy.dropna(subset=['days_to_nvd', 'severity_type'])

        if chart_data.empty:
            logger.warning(
                f"No data points for '{chart_key}' after processing dates and"
                " NaNs."
            )
            return None

        # Add display_severity_type
        chart_data['display_severity_type'] = (
            chart_data['severity_type']
            .map(self.severity_display_name_map)
            .fillna(chart_data['severity_type'])
        )

        # Make display_severity_type categorical for X-axis ordering
        ordered_severity_display = [
            self.severity_display_name_map.get(k, k)
            for k in self.severity_order
        ]
        chart_data['display_severity_type'] = pd.Categorical(
            chart_data['display_severity_type'],
            categories=ordered_severity_display,
            ordered=True,
        )
        # Also make original severity_type categorical if used for any intermediate logic
        chart_data['severity_type'] = pd.Categorical(
            chart_data['severity_type'],
            categories=self.severity_order,
            ordered=True,
        )

        return chart_data[
            ['severity_type', 'display_severity_type', 'days_to_nvd']
        ].sort_values(by='display_severity_type')

    def _prepare_data_for_spotlight_cves_base(
        self, df: pd.DataFrame, top_n: int = 10
    ) -> pd.DataFrame | None:
        """Selects Top N most critical CVEs for spotlight section (data for table & chart)."""
        chart_key = "spotlight_cves_details"
        # Initial required columns for filtering and sorting
        # cve_id and cvss_score are essential for the plot. title is good for hover.
        required_for_plot = ['cve_id', 'cvss_score', 'title']
        # Columns needed for the sorting logic itself
        required_for_sort = [
            'attack_vector',
            'attack_complexity',
            'privileges_required',
            'user_interaction',
        ]
        # Optional for sorting
        optional_for_sort = ['cisa_kev']

        all_possible_cols = (
            required_for_plot + required_for_sort + optional_for_sort
        )

        df_copy = df.copy()  # Work on a copy

        # Ensure essential columns for plotting exist, even if just as placeholders initially
        for col in required_for_plot:
            if col not in df_copy.columns:
                if col == 'cvss_score':
                    df_copy[col] = 0.0
                elif col == 'cve_id':
                    df_copy[col] = "CVE-UNKNOWN"  # Placeholder
                else:
                    df_copy[col] = "N/A"

        # For sorting columns, fill NA more robustly before mapping
        for col in required_for_sort:
            if col not in df_copy.columns:
                df_copy[col] = 'n/a'  # Default for missing sort factor
            else:
                df_copy[col] = df_copy[col].fillna('n/a')

        if 'cisa_kev' in df_copy.columns:
            df_copy['cisa_kev'] = df_copy['cisa_kev'].fillna(False)

        # Define sorting criteria
        sort_columns = ['cvss_score']
        ascending_order = [False]  # CVSS descending

        if (
            'cisa_kev' in df_copy.columns
        ):  # Check if it actually exists after potential creation
            df_copy['cisa_kev_sort'] = df_copy['cisa_kev'].astype(bool)
            sort_columns.append('cisa_kev_sort')
            ascending_order.append(False)  # KEV=True first

        av_map = {
            'network': 0,
            'adjacent_network': 1,
            'local': 2,
            'physical': 3,
            'none': 4,
            'n/a': 4,
        }
        ac_map = {'low': 0, 'high': 1, 'n/a': 2}
        pr_map = {'none': 0, 'low': 1, 'high': 2, 'n/a': 3}
        ui_map = {'none': 0, 'required': 1, 'n/a': 2}

        df_copy['av_sort'] = (
            df_copy['attack_vector']
            .str.lower()
            .map(av_map)
            .fillna(max(av_map.values()))
        )
        df_copy['ac_sort'] = (
            df_copy['attack_complexity']
            .str.lower()
            .map(ac_map)
            .fillna(max(ac_map.values()))
        )
        df_copy['pr_sort'] = (
            df_copy['privileges_required']
            .str.lower()
            .map(pr_map)
            .fillna(max(pr_map.values()))
        )
        df_copy['ui_sort'] = (
            df_copy['user_interaction']
            .str.lower()
            .map(ui_map)
            .fillna(max(ui_map.values()))
        )

        sort_columns.extend(['av_sort', 'ac_sort', 'pr_sort', 'ui_sort'])
        ascending_order.extend([True, True, True, True])

        # Perform sort and select top_n
        # Ensure cvss_score is numeric before sorting
        df_copy['cvss_score'] = pd.to_numeric(
            df_copy['cvss_score'], errors='coerce'
        ).fillna(0.0)

        spotlight_df = df_copy.sort_values(
            by=sort_columns, ascending=ascending_order
        ).head(top_n)

        if spotlight_df.empty:
            logger.warning(f"No CVEs found for spotlight in '{chart_key}'.")
            return None

        # Select only columns needed for the plot + hover to simplify
        # The full set of display_cols might be for a table elsewhere
        plot_cols = ['cve_id', 'cvss_score', 'title']  # title for hover
        final_spotlight_df = spotlight_df[
            plot_cols
        ].copy()  # .copy() to avoid warnings on next step

        # Ensure cve_id is string for y-axis category
        final_spotlight_df['cve_id'] = final_spotlight_df['cve_id'].astype(str)

        return final_spotlight_df.sort_values(
            by='cvss_score', ascending=False
        )  # Final sort for plot order

    def _prepare_data_for_cat_av_pr_chart(
        self, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        chart_key = "cat_av_pr_chart"
        required_cols = [
            'cve_category',
            'attack_vector',
            'privileges_required',
            'cve_id',
        ]
        if not all(
            c in df.columns for c in required_cols
        ):  # Simpler check for column existence
            logger.warning(
                f"Missing required columns {required_cols} for '{chart_key}'."
            )
            return None

        df_copy = df.copy()

        # Fill NaNs before converting to categorical
        df_copy['cve_category'] = df_copy['cve_category'].fillna('none')
        df_copy['attack_vector'] = df_copy['attack_vector'].fillna('none')
        df_copy['privileges_required'] = df_copy['privileges_required'].fillna(
            'none'
        )

        # Check for all-NaN columns after filling, which indicates no actual data
        if not all(df_copy[c].notna().any() for c in required_cols):
            logger.warning(
                "One or more required columns are all NaN after fillna for"
                f" '{chart_key}'."
            )
            return None

        # Ensure categorical types for consistent ordering and all categories present
        df_copy['cve_category'] = pd.Categorical(
            df_copy['cve_category'],
            categories=self.category_order,
            ordered=True,
        )
        df_copy['attack_vector'] = pd.Categorical(
            df_copy['attack_vector'],
            categories=self.attack_vector_order,
            ordered=True,
        )
        df_copy['privileges_required'] = pd.Categorical(
            df_copy['privileges_required'],
            categories=self.privileges_required_order,
            ordered=True,
        )

        agg_data = (
            df_copy.groupby(
                ['cve_category', 'attack_vector', 'privileges_required'],
                observed=False,
            )
            .size()
            .reset_index(name='count')
        )

        if agg_data.empty or agg_data['count'].sum() == 0:
            logger.warning(
                f"No data with counts after aggregation for '{chart_key}'."
            )
            return None

        # Add display name columns for use in plotting
        mapped_cve_category = agg_data['cve_category'].map(
            self.category_display_name_map
        )
        agg_data['display_cve_category'] = mapped_cve_category.fillna(
            agg_data['cve_category'].astype(object)
        )

        mapped_av = agg_data['attack_vector'].map(
            self.attack_vector_display_map
        )
        agg_data['display_attack_vector'] = mapped_av.fillna(
            agg_data['attack_vector'].astype(object)
        )

        mapped_pr = agg_data['privileges_required'].map(
            self.privileges_required_display_map
        )
        agg_data['display_privileges_required'] = mapped_pr.fillna(
            agg_data['privileges_required'].astype(object)
        )

        # Make display columns categorical based on the order of their original keys
        # This ensures Plotly plots them in the desired order when used for axes/facets/colors
        ordered_cat_display = [
            self.category_display_name_map.get(k, k)
            for k in self.category_order
        ]
        agg_data['display_cve_category'] = pd.Categorical(
            agg_data['display_cve_category'],
            categories=ordered_cat_display,
            ordered=True,
        )

        ordered_av_display = [
            self.attack_vector_display_map.get(k, k)
            for k in self.attack_vector_order
        ]
        agg_data['display_attack_vector'] = pd.Categorical(
            agg_data['display_attack_vector'],
            categories=ordered_av_display,
            ordered=True,
        )

        ordered_pr_display = [
            self.privileges_required_display_map.get(k, k)
            for k in self.privileges_required_order
        ]
        agg_data['display_privileges_required'] = pd.Categorical(
            agg_data['display_privileges_required'],
            categories=ordered_pr_display,
            ordered=True,
        )
        logger.info(f"Aggregated data for '{chart_key}': {agg_data.head()}")
        return agg_data.sort_values(
            ['cve_category', 'attack_vector', 'privileges_required']
        )

    def _prepare_data_for_cat_ui_ac_chart(
        self, df: pd.DataFrame
    ) -> pd.DataFrame | None:
        chart_key = "cat_ui_ac_chart"
        required_cols = [
            'cve_category',
            'user_interaction',
            'attack_complexity',
            'cve_id',
        ]
        if not all(c in df.columns for c in required_cols):
            logger.warning(
                f"Missing required columns {required_cols} for '{chart_key}'."
            )
            return None

        df_copy = df.copy()

        # Fill NaNs appropriately before categorization
        df_copy['cve_category'] = df_copy['cve_category'].fillna('none')
        df_copy['user_interaction'] = df_copy['user_interaction'].fillna(
            'none'
        )
        # For attack_complexity, if 'none' exists and should be filtered:
        # df_copy = df_copy[df_copy['attack_complexity'] != 'none']
        # Then fillna for remaining NaNs, e.g. with 'low' if that's the default assumption
        df_copy['attack_complexity'] = df_copy['attack_complexity'].fillna(
            'low'
        )

        # Filter attack_complexity to only include 'low' and 'high' if other values might exist
        df_copy = df_copy[
            df_copy['attack_complexity'].isin(self.attack_complexity_order)
        ]

        if not all(
            df_copy[c].notna().any()
            for c in required_cols
            if c in df_copy.columns
        ):
            logger.warning(
                "One or more required columns are all NaN after"
                f" fillna/filtering for '{chart_key}'."
            )
            return None
        if df_copy.empty:
            logger.warning(
                f"DataFrame is empty after pre-processing for '{chart_key}'."
            )
            return None

        df_copy['cve_category'] = pd.Categorical(
            df_copy['cve_category'],
            categories=self.category_order,
            ordered=True,
        )
        df_copy['user_interaction'] = pd.Categorical(
            df_copy['user_interaction'],
            categories=self.user_interaction_order,
            ordered=True,
        )
        df_copy['attack_complexity'] = pd.Categorical(
            df_copy['attack_complexity'],
            categories=self.attack_complexity_order,
            ordered=True,
        )

        agg_data = (
            df_copy.groupby(
                ['cve_category', 'user_interaction', 'attack_complexity'],
                observed=False,
            )
            .size()
            .reset_index(name='count')
        )

        if agg_data.empty or agg_data['count'].sum() == 0:
            logger.warning(
                f"No data with counts after aggregation for '{chart_key}'."
            )
            return None

        mapped_cve_category = agg_data['cve_category'].map(
            self.category_display_name_map
        )
        agg_data['display_cve_category'] = mapped_cve_category.fillna(
            agg_data['cve_category'].astype(object)
        )
        mapped_ui = agg_data['user_interaction'].map(
            self.user_interaction_display_map
        )
        agg_data['display_user_interaction'] = mapped_ui.fillna(
            agg_data['user_interaction'].astype(object)
        )
        mapped_ac = agg_data['attack_complexity'].map(
            self.attack_complexity_display_map
        )
        agg_data['display_attack_complexity'] = mapped_ac.fillna(
            agg_data['attack_complexity'].astype(object)
        )

        ordered_cat_display = [
            self.category_display_name_map.get(k, k)
            for k in self.category_order
        ]
        agg_data['display_cve_category'] = pd.Categorical(
            agg_data['display_cve_category'],
            categories=ordered_cat_display,
            ordered=True,
        )

        ordered_ui_display = [
            self.user_interaction_display_map.get(k, k)
            for k in self.user_interaction_order
        ]
        agg_data['display_user_interaction'] = pd.Categorical(
            agg_data['display_user_interaction'],
            categories=ordered_ui_display,
            ordered=True,
        )

        ordered_ac_display = [
            self.attack_complexity_display_map.get(k, k)
            for k in self.attack_complexity_order
        ]
        agg_data['display_attack_complexity'] = pd.Categorical(
            agg_data['display_attack_complexity'],
            categories=ordered_ac_display,
            ordered=True,
        )

        logger.info(f"Aggregated data for '{chart_key}': {agg_data.head()}")
        return agg_data.sort_values(
            ['cve_category', 'user_interaction', 'attack_complexity']
        )

    def _prepare_data_for_top_cwe_av_cvss_chart(
        self, df: pd.DataFrame, top_n_cwes_val: int = 5
    ) -> pd.DataFrame | None:
        chart_key = "top_cwe_av_cvss_boxplot"
        required_cols = [
            'cwe_id',
            'attack_vector',
            'cvss_score',
            'cve_id',
        ]  # cve_id for counting top CWEs
        if not all(c in df.columns for c in required_cols):
            logger.warning(
                f"Missing required columns {required_cols} for '{chart_key}'."
            )
            return None

        df_copy = df.copy()

        # --- Determine Top N CWEs from the entire dataset (or relevant subset) ---
        # First, correctly parse/explode cwe_id if it can be a list
        # This assumes cwe_id might be like "CWE-79, CWE-89"
        if (
            df_copy['cwe_id'].dtype == 'object'
            and df_copy['cwe_id'].str.contains('[,;]', na=False).any()
        ):
            cwe_counts_df = df_copy.assign(
                parsed_cwe=df_copy['cwe_id'].str.split('[,;]\s*')
            ).explode('parsed_cwe')
            cwe_counts_df['parsed_cwe'] = cwe_counts_df[
                'parsed_cwe'
            ].str.strip()
            # Filter out empty strings that might result from splitting
            cwe_counts_df = cwe_counts_df[cwe_counts_df['parsed_cwe'] != '']
        else:  # Assume cwe_id is already a single, clean ID per row or needs no splitting
            cwe_counts_df = df_copy.rename(columns={'cwe_id': 'parsed_cwe'})

        cwe_counts_df = cwe_counts_df.dropna(subset=['parsed_cwe'])
        if cwe_counts_df.empty:
            logger.warning(
                "No CWE data available to determine top CWEs for"
                f" '{chart_key}'."
            )
            return None

        top_cwe_ids = (
            cwe_counts_df['parsed_cwe']
            .value_counts()
            .nlargest(top_n_cwes_val)
            .index.tolist()
        )
        if not top_cwe_ids:
            logger.warning(
                f"Could not determine Top {top_n_cwes_val} CWEs for"
                f" '{chart_key}'."
            )
            return None
        # --- End Top N CWEs Determination ---

        # Filter the original (or pre-processed) DataFrame to include only these top CWEs
        # and then process for the chart data.
        # We need to re-process the main df_copy to get individual CVEs for boxplots,
        # focusing on rows where *any* of their listed CWEs is a top CWE.

        # Create a boolean mask for rows where 'cwe_id' (potentially multi-valued string) contains any of the top_cwe_ids
        def contains_top_cwe(cwe_string):
            if pd.isna(cwe_string):
                return False
            current_cwes = {
                c.strip() for c in str(cwe_string).split('[,;]\s*')
            }
            return any(top_cwe in current_cwes for top_cwe in top_cwe_ids)

        df_filtered_for_top_cwes = df_copy[
            df_copy['cwe_id'].apply(contains_top_cwe)
        ].copy()  # .copy() to avoid SettingWithCopyWarning

        if df_filtered_for_top_cwes.empty:
            logger.warning(
                "No CVEs found associated with the top CWEs for"
                f" '{chart_key}'."
            )
            return None

        # Now, from this filtered DataFrame, we need to prepare for faceting.
        # Each row will contribute to potentially multiple facets if it has multiple top CWEs.
        # So, we "explode" again, but this time only keeping rows that match a top CWE.

        chart_data_list = []
        for index, row in df_filtered_for_top_cwes.iterrows():
            cvss_score = row['cvss_score']
            attack_vector = row['attack_vector']
            if pd.isna(cvss_score) or pd.isna(attack_vector):
                continue

            current_row_cwes = {
                c.strip() for c in str(row['cwe_id']).split('[,;]\s*')
            }
            for top_cwe_id_str in top_cwe_ids:
                if top_cwe_id_str in current_row_cwes:
                    chart_data_list.append({
                        'primary_cwe_id': (
                            top_cwe_id_str
                        ),  # This will be our facet
                        'attack_vector': attack_vector,
                        'cvss_score': cvss_score,
                    })

        if not chart_data_list:
            logger.warning(
                f"No data points generated for top CWEs for '{chart_key}'."
            )
            return None

        chart_data = pd.DataFrame(chart_data_list)

        chart_data['attack_vector'] = chart_data['attack_vector'].fillna(
            'none'
        )
        chart_data = chart_data.dropna(
            subset=['primary_cwe_id', 'attack_vector', 'cvss_score']
        )
        # Ensure cvss_score is numeric
        chart_data['cvss_score'] = pd.to_numeric(
            chart_data['cvss_score'], errors='coerce'
        ).dropna()

        if chart_data.empty:
            logger.warning(
                f"No valid data points after final cleaning for '{chart_key}'."
            )
            return None

        # Add display names
        chart_data['display_primary_cwe_id'] = chart_data[
            'primary_cwe_id'
        ].apply(self._get_cwe_display_label)
        mapped_av = chart_data['attack_vector'].map(
            self.attack_vector_display_map
        )
        chart_data['display_attack_vector'] = mapped_av.fillna(
            chart_data['attack_vector'].astype(object)
        )

        # Categorical ordering
        # Order facets by the original top_cwe_ids order (which was by frequency)
        ordered_cwe_display_labels = [
            self._get_cwe_display_label(cwe_id) for cwe_id in top_cwe_ids
        ]
        chart_data['display_primary_cwe_id'] = pd.Categorical(
            chart_data['display_primary_cwe_id'],
            categories=ordered_cwe_display_labels,
            ordered=True,
        )

        ordered_av_display_labels = [
            self.attack_vector_display_map.get(k, k)
            for k in self.attack_vector_order
        ]
        chart_data['display_attack_vector'] = pd.Categorical(
            chart_data['display_attack_vector'],
            categories=ordered_av_display_labels,
            ordered=True,
        )

        # Sort for consistent plot generation if needed, though category_orders handles it mostly
        return chart_data.sort_values(
            ['display_primary_cwe_id', 'display_attack_vector']
        )

    def _get_cluster_characterization(
        self, df_cluster: pd.DataFrame, categorical_features: list
    ) -> dict:
        """Calculates descriptive statistics for a single cluster."""
        if df_cluster.empty:
            return {"size": 0, "cvss_profile": {}, "categorical_profiles": {}}

        char_dict = {"size": len(df_cluster)}

        # CVSS Profile
        if (
            'cvss_score' in df_cluster.columns
            and df_cluster['cvss_score'].notna().any()
        ):
            cvss_desc = df_cluster['cvss_score'].describe().to_dict()
            char_dict["cvss_profile"] = {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in cvss_desc.items()
            }
        else:
            char_dict["cvss_profile"] = {"mean": "N/A"}

        # Categorical Features Profile
        cat_profiles = {}
        for feat in categorical_features:
            if feat in df_cluster.columns and df_cluster[feat].notna().any():
                counts = (
                    df_cluster[feat]
                    .value_counts(normalize=True)
                    .mul(100)
                    .round(1)
                )
                # Get display name for the feature values using the maps
                display_counts = {}
                if feat == 'cve_category':
                    mapper = self.category_display_name_map
                elif feat == 'attack_vector':
                    mapper = self.attack_vector_display_map
                elif feat == 'attack_complexity':
                    mapper = self.attack_complexity_display_map
                elif feat == 'privileges_required':
                    mapper = self.privileges_required_display_map
                elif feat == 'user_interaction':
                    mapper = self.user_interaction_display_map
                else:
                    mapper = {}

                for val, perc in counts.items():
                    display_val = mapper.get(
                        val, str(val)
                    )  # Fallback to original value
                    display_counts[display_val] = perc
                cat_profiles[feat] = dict(
                    sorted(
                        display_counts.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:3]
                )  # Top 3
            else:
                cat_profiles[feat] = {"top_value": "N/A"}
        char_dict["categorical_profiles"] = cat_profiles
        return char_dict

    def _attempt_automated_cluster_name(
        self, char_dict: dict, cluster_numeric_id: int
    ) -> str:
        """Attempts to generate a preliminary name. Uses full display names from char_dict."""
        display_id = cluster_numeric_id + 1

        if char_dict['size'] == 0:
            return f"Profile Group {display_id} (Empty)"

        parts = []
        # CVSS
        cvss_median = char_dict.get("cvss_profile", {}).get("median", "N/A")
        if isinstance(cvss_median, (int, float)):
            if cvss_median >= 8.0:
                parts.append("V.High CVSS")
            elif cvss_median >= 7.0:
                parts.append("High CVSS")
            elif cvss_median >= 4.0:
                parts.append("Med CVSS")
            else:
                parts.append("Low CVSS")

        # Top Category (already a display name from _get_cluster_characterization)
        cat_profile = char_dict.get("categorical_profiles", {}).get(
            "cve_category", {}
        )
        if cat_profile:
            top_cat_display_name = list(cat_profile.keys())[0]
            # --- Local Shortening for Auto-Name String ---
            if "Remote Code Execution" in top_cat_display_name:
                short_cat_name = "RCE"
            elif "Privilege Elevation" in top_cat_display_name:
                short_cat_name = "EoP"
            elif "Information Disclosure" in top_cat_display_name:
                short_cat_name = "Info. Disc."
            elif "Denial of Service" in top_cat_display_name:
                short_cat_name = "DoS"
            else:
                short_cat_name = top_cat_display_name  # Use full display name if no short version
            parts.append(short_cat_name)

        # Top Attack Vector (already a display name)
        av_profile = char_dict.get("categorical_profiles", {}).get(
            "attack_vector", {}
        )
        if av_profile:
            top_av_display_name = list(av_profile.keys())[0]
            if (
                "Unknown" not in top_av_display_name
                and "None" not in top_av_display_name
            ):
                parts.append(f"{top_av_display_name} Vector")
            else:
                parts.append(
                    top_av_display_name
                )  # e.g. "Unknown AV" or "None AV"

        if not parts or len(parts) < 2:
            return f"Profile Group {display_id}"

        name_base = ", ".join(parts[:3])  # Max 3 parts
        return f"{name_base} (Auto)"

    def _prepare_data_for_tsne_cve_profile_chart(
        self,
        df: pd.DataFrame,
        data_output_dir: str = None,  # Parameter for output directory
        n_clusters: int = 4,
    ) -> pd.DataFrame | None:
        # from sklearn... (imports as in your version)
        from sklearn.cluster import KMeans
        from sklearn.compose import ColumnTransformer
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        chart_key = "tsne_cve_profile_chart"
        base_features = ['cvss_score']
        categorical_features_raw = [
            'attack_vector',
            'attack_complexity',
            'privileges_required',
            'user_interaction',
            'cve_category',
        ]
        required_cols = base_features + categorical_features_raw + ['cve_id']

        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            logger.warning(
                f"Missing required columns {missing} for '{chart_key}'. Cannot"
                " proceed."
            )
            return None

        df_original_context = df[required_cols].copy()
        # Ensure cvss_score is numeric before dropping NaNs based on it
        df_original_context['cvss_score'] = pd.to_numeric(
            df_original_context['cvss_score'], errors='coerce'
        )
        df_original_context.dropna(subset=['cvss_score'], inplace=True)
        for col in categorical_features_raw:
            # Using 'unknown' as fillna consistent with my _get_cluster_characterization example.
            # If your _get_cluster_characterization expects 'none', use df_original_context[col].fillna('none', inplace=True)
            df_original_context[col] = df_original_context[col].fillna(
                'unknown'
            )

        min_samples_for_tsne = max(
            n_clusters + 1, 15
        )  # Smallest reasonable sample size
        if (
            df_original_context.empty
            or len(df_original_context) < min_samples_for_tsne
        ):
            logger.warning(
                f"Not enough data points ({len(df_original_context)}, need at"
                f" least {min_samples_for_tsne}) for t-SNE/clustering for"
                f" '{chart_key}'."
            )
            return None

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), base_features),
                (
                    'cat',
                    OneHotEncoder(
                        handle_unknown='ignore', sparse_output=False
                    ),
                    categorical_features_raw,
                ),
            ],
            remainder='drop',
        )

        feature_matrix = preprocessor.fit_transform(
            df_original_context.drop(columns=['cve_id'])
        )

        if feature_matrix.shape[0] == 0:
            logger.warning(
                "Feature matrix is empty after preprocessing for"
                f" '{chart_key}'."
            )
            return None

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels_numeric = kmeans.fit_predict(
            feature_matrix
        )  # These are 0, 1, 2...

        # --- Characterization, Naming, and Saving ---
        actual_stats_to_save = (
            {}
        )  # Will hold detailed stats for JSON, keyed by "Profile Group X (Numeric ID Y)"
        final_names_for_plot = (
            {}
        )  # Maps numeric_id (0,1,2..) to final display name for the plot

        # Use a fresh copy of df_original_context for adding cluster_numeric,
        # as df_original_context is used later for joining features.
        temp_df_for_char_analysis = df_original_context.copy()
        temp_df_for_char_analysis['cluster_numeric'] = cluster_labels_numeric

        for i in range(n_clusters):  # i is the numeric cluster ID: 0, 1, 2...
            df_cluster_subset = temp_df_for_char_analysis[
                temp_df_for_char_analysis['cluster_numeric'] == i
            ]

            characterization_data_for_cluster_i = (
                self._get_cluster_characterization(
                    df_cluster_subset, categorical_features_raw
                )
            )

            # Key for JSON: User-friendly 1-indexed, plus original 0-indexed for clarity
            json_key = f"Profile Group {i+1} (Internal ID {i})"
            actual_stats_to_save[json_key] = (
                characterization_data_for_cluster_i
            )

            # Determine the display name for this cluster for the plot
            manual_name = self.MANUAL_CLUSTER_NAMES.get(
                i
            )  # Check manual overrides (0-indexed key)
            if manual_name:
                current_cluster_plot_name = manual_name
            else:
                current_cluster_plot_name = (
                    self._attempt_automated_cluster_name(
                        characterization_data_for_cluster_i, i
                    )
                )

            final_names_for_plot[i] = (
                current_cluster_plot_name  # Store final name for numeric ID i
            )

        # Save the DETAILED CHARACTERIZATION STATISTICS
        if (
            actual_stats_to_save and data_output_dir
        ):  # Use passed data_output_dir
            try:
                dynamic_filename = "cluster_characterization.json"  # Default
                if hasattr(self, 'run_identifier') and self.run_identifier:
                    run_id_parts = self.run_identifier.split('_')
                    if (
                        len(run_id_parts) >= 7
                    ):  # Expecting at least 7 parts for the full date range
                        start_month = run_id_parts[3]
                        start_year = run_id_parts[4]
                        end_month = run_id_parts[5]
                        end_year = run_id_parts[6]
                        # Construct a string like "jan2024_mar2024"
                        date_timestamp_str = (
                            f"{start_month}{start_year}_{end_month}{end_year}"
                        )
                        dynamic_filename = f"cluster_characterization_{date_timestamp_str}.json"
                    # --- END CORRECTION ---
                    else:
                        logger.warning(
                            f"self.run_identifier ('{self.run_identifier}')"
                            " does not have the expected number of parts"
                            f" (found {len(run_id_parts)}, expected >= 7) for"
                            " detailed date extraction. Using default"
                            " filename for cluster characterization."
                        )

                # Ensure the data_output_dir itself exists
                os.makedirs(data_output_dir, exist_ok=True)

                char_file_path = os.path.join(
                    data_output_dir, dynamic_filename
                )

                with open(char_file_path, 'w') as f:
                    json.dump(actual_stats_to_save, f, indent=2)

                logger.info(
                    "Cluster characterization statistics saved to"
                    f" {char_file_path}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to save cluster characterization statistics: {e}"
                )
        # --- End Characterization, Naming, Saving ---

        # --- t-SNE Transformation ---
        # Your existing perplexity calculation - ensure it's robust
        tsne_perplexity = min(
            30, max(1, len(df_original_context) - 2, n_clusters)
        )  # Max perplexity is n_samples - 1. Ensure > 0.
        if tsne_perplexity >= len(df_original_context):
            tsne_perplexity = max(
                1, len(df_original_context) - 1
            )  # Clamp if too high
        if (
            tsne_perplexity == 0 and len(df_original_context) > 1
        ):  # Handle edge case if samples = 1
            tsne_perplexity = 1

        if (
            len(df_original_context) <= tsne_perplexity
        ):  # Scikit-learn constraint
            tsne_perplexity = max(1, len(df_original_context) - 1)
            if tsne_perplexity == 0:  # if only 1 sample
                logger.warning(
                    f"Only {len(df_original_context)} sample(s), t-SNE cannot"
                    " run with perplexity. Skipping t-SNE."
                )
                # Create dummy t-SNE results or return None
                df_tsne_plot = pd.DataFrame(index=df_original_context.index)
                df_tsne_plot['tsne_x'] = 0
                df_tsne_plot['tsne_y'] = 0
            else:
                tsne = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=tsne_perplexity,
                    n_iter=250,
                    init='pca',
                    learning_rate='auto',
                )  # Reduced n_iter
                tsne_results = tsne.fit_transform(feature_matrix)
                df_tsne_plot = pd.DataFrame(
                    tsne_results,
                    columns=['tsne_x', 'tsne_y'],
                    index=df_original_context.index,
                )
        else:
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=tsne_perplexity,
                n_iter=250,
                init='pca',
                learning_rate='auto',
            )
            tsne_results = tsne.fit_transform(feature_matrix)
            df_tsne_plot = pd.DataFrame(
                tsne_results,
                columns=['tsne_x', 'tsne_y'],
                index=df_original_context.index,
            )

        df_tsne_plot['cluster_numeric'] = cluster_labels_numeric
        df_tsne_plot['cluster_label'] = df_tsne_plot['cluster_numeric'].map(
            final_names_for_plot
        )

        # Join back ALL columns from df_original_context (which includes cve_id, cvss_score, and raw categorical features)
        df_tsne_plot = df_tsne_plot.join(df_original_context)

        return df_tsne_plot

    def _prepare_chart_data(
        self,
        df: pd.DataFrame,
        chart_key: str,
        data_output_dir: Optional[str] = None,
        **chart_specific_params,
    ) -> Any:
        """
        Generates data formatted for a specific chart type using a dispatch dictionary.
        """
        logger.info(f"Preparing data for chart: {chart_key}")
        if df.empty:
            logger.warning(
                f"DataFrame is empty for chart '{chart_key}'. Returning None."
            )
            return None

        handler_method: Optional[Callable[..., Any]] = (
            self.CHART_DATA_HANDLERS.get(chart_key)
        )

        if handler_method:
            try:
                logger.info(
                    f"Using handler: {handler_method.__name__} for chart"
                    f" '{chart_key}'"
                )

                # Inspect the handler's signature to pass only relevant arguments
                handler_signature = inspect.signature(handler_method)
                handler_params = handler_signature.parameters

                # Prepare arguments to pass to the handler
                handler_args = {}
                if 'df' in handler_params:  # All handlers should take 'df'
                    handler_args['df'] = df

                # Pass data_output_dir if the handler accepts it
                if (
                    'data_output_dir' in handler_params
                    and data_output_dir is not None
                ):
                    handler_args['data_output_dir'] = data_output_dir

                # Pass any chart_specific_params if the handler accepts them
                if chart_specific_params:
                    for (
                        param_name,
                        param_value,
                    ) in chart_specific_params.items():
                        if param_name in handler_params:
                            handler_args[param_name] = param_value

                # Log the arguments being passed for debugging
                # logger.debug(f"Calling {handler_method.__name__} with args: {list(handler_args.keys())}")

                return handler_method(**handler_args)

            except Exception as e:
                logger.exception(
                    f"Error executing handler '{handler_method.__name__}' for"
                    f" chart '{chart_key}': {e}"
                )
                return None
        else:
            logger.warning(
                "No data preparation handler defined for chart key"
                f" '{chart_key}'. Returning None."
            )
            return None

    # ------------------------------------------------------------------------------
    # --- END DATA PREPARATION FUNCTIONS -------------------------------------------
    # ------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------
    # --- BEGIN PLOTTING FUNCTIONS -------------------------------------------------
    # ------------------------------------------------------------------------------

    def _plot_new_vs_updated_monthly(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure | None:
        """
        Plots a bar chart for new vs. updated CVEs monthly.

        Args:
            chart_data (pd.DataFrame): Data prepared by _prepare_data_for_new_vs_updated_monthly.
            title (str): Title for the chart.

        Returns:
            go.Figure | None: The Plotly figure object, or None if plotting fails.
        """
        if chart_data is None or chart_data.empty:
            logger.info(
                "No data provided to _plot_new_vs_updated_monthly for title"
                f" '{title}'. Cannot generate chart."
            )
            return None

        if (
            'month_display_label' not in chart_data.columns
            or chart_data['month_display_label'].isna().all()
        ):
            logger.error(
                "Critical: 'month_display_label' column is missing or all NaN"
                " in chart_data for _plot_new_vs_updated_monthly. Cannot"
                " generate chart correctly."
            )
            return None

        fig = px.bar(
            chart_data,
            x='month_display_label',  # Use the ordered categorical column for x-axis
            y='count',
            color='status',
            labels={  # Defines hover labels and can influence default axis titles
                'month_display_label': "Month",
                'count': 'Number of CVEs',
                'status': 'Status',
            },
            barmode='group',
            text_auto=True,  # Automatically display text (bar values)
            title=title,  # Set the chart title
            color_discrete_map={  # Apply custom colors from your theme
                'new': '#4BAEEA',
                'updated': '#25B556',
            },
        )

        fig.update_traces(
            textposition='outside',  # Position text labels outside the bars
            cliponaxis=False,  # Prevent text labels from being clipped at the chart/axis edge
        )

        max_y_value = chart_data['count'].max()
        # Ensure a sensible y-axis range even if max_y_value is 0 or NaN
        if pd.isna(max_y_value) or max_y_value <= 0:
            y_axis_upper_limit = 10  # Default small upper limit if no positive data or all zeros
        else:
            y_axis_upper_limit = (
                max_y_value * 1.15
            )  # Add 15% padding for text labels

        fig.update_layout(
            title_x=0.5,  # Center the main chart title
            xaxis_title="Month",  # Explicitly set X-axis title
            yaxis_title="Number of CVEs",  # Explicitly set Y-axis title
            yaxis_range=[
                0,
                y_axis_upper_limit,
            ],  # Apply calculated y-axis range
            legend_title_text='Status',  # Set a title for the legend
        )

        return fig

    def _plot_volume_by_severity_monthly(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure | None:
        """
        Plots a bar chart for CVE volume by severity, monthly.

        Args:
            chart_data (pd.DataFrame): Data prepared by _prepare_data_for_volume_by_severity_monthly.
                                       Expected columns: 'month_display_label', 'severity_type', 'count'.
            title (str): Title for the chart.

        Returns:
            go.Figure | None: The Plotly figure object, or None if plotting fails.
        """
        if chart_data is None or chart_data.empty:
            logger.info(
                "No data provided to _plot_volume_by_severity_monthly for"
                f" title '{title}'."
            )
            return None

        required_plot_cols = ['month_display_label', 'severity_type', 'count']
        if not all(c in chart_data.columns for c in required_plot_cols):
            logger.error(
                "Chart data is missing required columns"
                f" {required_plot_cols} for _plot_volume_by_severity_monthly."
            )
            return None

        if chart_data['month_display_label'].isna().all():
            logger.error(
                "Critical: 'month_display_label' column is all NaN in"
                " chart_data for _plot_volume_by_severity_monthly."
            )
            return None

        fig = px.bar(
            chart_data,
            x='month_display_label',
            y='count',
            color='severity_type',
            labels={
                'month_display_label': 'Month',
                'count': 'Number of CVEs',
                'severity_type': (  # This will be used by legend unless overridden
                    'Severity'
                ),
            },
            category_orders={
                "severity_type": self.severity_order
            },  # Uses self.severity_order
            color_discrete_map=self.severity_color_map,  # Uses self.severity_color_map
            barmode='group',
            text_auto=True,
            title=title,
        )

        fig.update_traces(
            textposition='outside',
            cliponaxis=False,  # Prevent text labels from being clipped at the chart/axis edge
        )

        max_y_value = chart_data['count'].max()
        if pd.isna(max_y_value) or max_y_value <= 0:
            y_axis_upper_limit = 10
        else:
            y_axis_upper_limit = max_y_value * 1.15  # Add 15% padding

        fig.update_layout(
            title_x=0.5,  # Center the main chart title
            xaxis_title="Month",
            yaxis_title="Number of CVEs",
            yaxis_range=[0, y_axis_upper_limit],
            legend_title_text='Severity',  # Explicit title for the legend
            margin=dict(
                t=60, b=50, l=50, r=20
            ),  # Adjusted top margin for title, right for legend
        )

        return fig

    def _plot_cvss_distribution(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure:
        fig = px.violin(
            chart_data,
            x='cve_category',
            y='cvss_score',
            labels={
                'cve_category': 'Vulnerability Category',
                'cvss_score': 'CVSS Base Score',
            },
            category_orders={"cve_category": self.category_order},
            color="cve_category",
            color_discrete_map=self.category_color_map,
            box=True,
            points='outliers',
        )
        fig.update_layout(showlegend=False)
        return fig

    def _plot_rce_proportion_monthly(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure:
        fig = px.line(
            chart_data,
            x='published_month',
            y='proportion',
            labels={
                'published_month': 'Month',
                'proportion': '% of Total CVEs',
            },
            markers=True,
        )
        fig.update_layout(
            yaxis_range=[
                0,
                max(
                    10,
                    (
                        chart_data['proportion'].max() * 1.1
                        if not chart_data.empty
                        and chart_data['proportion'].max() > 0
                        else 10
                    ),
                ),
            ],
            yaxis_ticksuffix="%",
        )
        return fig

    def _plot_rce_cvss_distribution(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure:
        fig = px.violin(
            chart_data,
            x='category_label',
            y='cvss_score',
            labels={'category_label': '', 'cvss_score': 'CVSS Base Score'},
            box=True,
            points='all',
            color_discrete_sequence=[
                self.category_color_map.get('remote_code_execution', '#D84749')
            ],
        )
        fig.update_layout(xaxis_title=None)
        return fig

    def _plot_eop_proportion_monthly(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure:
        fig = px.line(
            chart_data,
            x='published_month',
            y='proportion',
            labels={
                'published_month': 'Month',
                'proportion': '% of Total CVEs',
            },
            markers=True,
        )
        fig.update_layout(
            yaxis_range=[
                0,
                max(
                    10,
                    (
                        chart_data['proportion'].max() * 1.1
                        if not chart_data.empty
                        and chart_data['proportion'].max() > 0
                        else 10
                    ),
                ),
            ],
            yaxis_ticksuffix="%",
        )
        return fig

    def _plot_eop_by_attack_vector(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure:
        fig = px.bar(
            chart_data,
            x='attack_vector',
            y='count',
            labels={
                'attack_vector': 'Attack Vector',
                'count': 'Number of EoP CVEs',
            },
            category_orders={"attack_vector": self.attack_vector_order},
            text_auto=True,
        )
        fig.update_traces(
            marker_color=self.category_color_map.get(
                'privilege_elevation', '#EB8B06'
            )
        )
        return fig

    def _plot_dos_infodisc_monthly(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure:
        fig = px.bar(
            chart_data,
            x='published_month',
            y='count',
            color='category_display',
            labels={
                'published_month': 'Month',
                'count': 'Number of CVEs',
                'category_display': 'Category',
            },
            color_discrete_map={
                'DoS': self.category_color_map.get(
                    'denial_of_service', '#22A5DD'
                ),
                'Info Disclosure': self.category_color_map.get(
                    'disclosure', '#67C2C0'
                ),  # Ensure 'disclosure' key exists in map
            },
            barmode='group',
            text_auto=True,
        )
        fig.update_traces(textposition='outside')
        return fig

    def _plot_user_interaction_proportion_monthly(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure:
        fig = px.line(
            chart_data,
            x='published_month',
            y='proportion_required',
            labels={
                'published_month': 'Month',
                'proportion_required': '% Requiring User Interaction',
            },
            markers=True,
        )
        fig.update_layout(yaxis_range=[0, 100], yaxis_ticksuffix="%")
        return fig

    def _plot_overall_attack_vector_dist(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure | None:
        """
        Plots a bar chart for overall attack vector distribution.

        Args:
            chart_data (pd.DataFrame): Data from _prepare_data_for_overall_attack_vector_dist.
                                       Expected columns: 'attack_vector' (for color),
                                       'display_attack_vector' (for x-axis), 'percentage'.
            title (str): Title for the chart.

        Returns:
            go.Figure | None: The Plotly figure object, or None if plotting fails.
        """
        if chart_data is None or chart_data.empty:
            logger.info(
                "No data provided to _plot_overall_attack_vector_dist for"
                f" title '{title}'."
            )
            return None

        required_plot_cols = [
            'attack_vector',
            'display_attack_vector',
            'percentage',
        ]
        if not all(c in chart_data.columns for c in required_plot_cols):
            logger.error(
                "Chart data is missing required columns"
                f" {required_plot_cols} for attack vector distribution."
            )
            return None

        fig = px.bar(
            chart_data,
            x='display_attack_vector',  # Use display name for x-axis
            y='percentage',
            color='attack_vector',  # Color by original key to use the color map
            color_discrete_map=self.attack_vector_color_map,
            labels={
                'display_attack_vector': (
                    'Attack Vector'
                ),  # Label for the x-axis itself
                'percentage': '% of Total CVEs',
                'attack_vector': (  # Label for legend/hover if color is shown
                    'Attack Vector Type'
                ),
            },
            # category_orders={"display_attack_vector": [self.attack_vector_display_map.get(k,k) for k in self.attack_vector_order]}, # Order x-axis
            text_auto=False,
        )

        # Add '%' suffix to bar text labels and y-axis ticks
        fig.update_traces(
            width=0.4,
            texttemplate='%{y:.1f}%',  # Add % suffix to the text displayed on bars
            textposition='outside',
            cliponaxis=False,
        )
        fig.update_layout(
            yaxis_ticksuffix="%",
            title_text=title,
            title_x=0.5,
            xaxis_title="Attack Vector",  # Explicit x-axis title
            yaxis_title="% of Total CVEs",  # Explicit y-axis title
            showlegend=False,  # Hide legend as colors are distinct and x-axis is categorical
        )

        # Adjust y-axis range to prevent clipping of text labels
        max_y_value = chart_data['percentage'].max()
        if pd.isna(max_y_value) or max_y_value <= 0:
            y_axis_upper_limit = 10  # Default if no positive data
        else:
            # Add a bit more padding for percentage values
            y_axis_upper_limit = max_y_value * 1.20

        fig.update_layout(yaxis_range=[0, y_axis_upper_limit])

        return fig

    def _plot_overall_attack_complexity_dist(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure:
        fig = px.bar(
            chart_data,
            x='attack_complexity',
            y='percentage',
            labels={
                'attack_complexity': 'Attack Complexity',
                'percentage': '% of Total CVEs',
            },
            category_orders={
                "attack_complexity": self.attack_complexity_order
            },
            text_auto='.1f',
        )
        fig.update_layout(yaxis_ticksuffix="%")
        fig.update_traces(textposition='outside')
        return fig

    def _plot_patch_delay_risk_scatter(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure:
        fig = px.scatter(
            chart_data,
            x='days_to_patch',
            y='cvss_score',
            labels={
                'days_to_patch': 'Days Until NVD Publication (Proxy)',
                'cvss_score': 'CVSS Base Score',
                'cve_category': 'Category',
            },
            color='cve_category',
            color_discrete_map=self.category_color_map,
            category_orders={"cve_category": self.category_order},
            hover_data=['cve_id'],
        )
        return fig

    def _plot_overall_category_distribution_chart(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure | None:
        """
        Generates a treemap for overall CVE category distribution with improved labels and colors.

        Args:
            chart_data (pd.DataFrame): Expected columns: 'cve_category' (for color mapping),
                                       'display_category' (for labels), 'count'.
            title (str): Title for the chart.

        Returns:
            go.Figure | None: The Plotly figure object, or None if plotting fails.
        """
        if chart_data is None or chart_data.empty:
            logger.info(
                "No data provided to"
                " _plot_overall_category_distribution_chart for title"
                f" '{title}'."
            )
            return None

        required_plot_cols = ['cve_category', 'display_category', 'count']
        if not all(c in chart_data.columns for c in required_plot_cols):
            logger.error(
                f"Chart data missing required columns {required_plot_cols} for"
                " overall category distribution."
            )
            return None

        fig = px.treemap(
            chart_data,
            path=[
                px.Constant("All Categories"),
                'display_category',
            ],  # Use display_category for path
            values='count',
            color='cve_category',  # Color by original 'cve_category' to use the map keys
            color_discrete_map=self.category_color_map,
            custom_data=[
                'display_category',
                'count',
            ],  # Add data for hover/text
        )

        # Customize text formatting inside treemap tiles
        # We want:
        # Display Category (Bold)
        # Count (Larger)
        # Percentage (Larger)
        fig.update_traces(
            texttemplate=(
                "<b>%{label}</b><br>%{customdata[1]}<br>%{percentRoot:.0%}"
            ),
            textfont_size=12,  # Base size, can be overridden by HTML tags if needed
            textposition='middle center',
            insidetextfont=dict(
                size=14, color='white'
            ),  # Increase size of text inside the tiles
            # Adjust text color for better contrast if needed, e.g., based on tile color.
            # This is complex; often a single contrasting color (like white or black) is chosen.
            # If tiles are dark, white text is good. If light, black.
            # Plotly doesn't auto-adjust text color for contrast on treemaps easily.
            # Consider adding a border to the text or a slight shadow if contrast is an issue across various colors.
            marker=dict(cornerradius=5),  # Slightly rounded corners for tiles
        )

        fig.update_layout(
            title_text=title,
            title_x=0.5,  # Center title
            margin=dict(t=50, l=25, r=25, b=25),  # Adjust margins
            # Uniform text for the root ("All Categories") - can be styled further if needed
            # treemapcolorway = [self.category_color_map.get(cat, '#CCCCCC') for cat in chart_data['cve_category']] # Ensures color consistency
        )

        # The 'All Categories' box color - often a neutral or derived color.
        # Plotly takes the first color of the colorscale or colorway by default for the root.
        # To explicitly set it, you might need to manipulate the figure structure or use a specific color for it.
        # For simplicity, we'll let Plotly handle it, but it often picks the first color from the map.
        # If 'All Categories' is pink in your image, it might be that the first category in your data was 'feature_bypass'
        # and its color was pink. The treemap root color often reflects the dominant child or the first one.

        return fig

    def _plot_attack_vectors_by_privileges_chart(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure | None:
        """Generates a stacked bar chart for Attack Vectors by Privileges Required."""
        if chart_data is None or chart_data.empty:
            logger.info(
                "No data provided to _plot_attack_vectors_by_privileges_chart"
                f" for '{title}'."
            )
            return None

        # Ensure all required display columns for plotting are present
        required_plot_cols = [
            'display_attack_vector',
            'display_privileges_required',
            'count',
        ]
        if not all(c in chart_data.columns for c in required_plot_cols):
            missing_cols = [
                col
                for col in required_plot_cols
                if col not in chart_data.columns
            ]
            logger.error(
                f"Chart data for '{title}' missing required plot columns:"
                f" {missing_cols}"
            )
            # Return an error figure or None
            fig = go.Figure()
            fig.update_layout(title_text=title, title_x=0.5, height=300)
            fig.add_annotation(
                text=(
                    "Plot data misconfigured. Missing:"
                    f" {', '.join(missing_cols)}"
                ),
                showarrow=False,
            )
            return fig

        color_map_for_display_privs = {
            self.privileges_required_display_map.get(key, key): color
            for key, color in self.privileges_required_color_map.items()
        }

        # Dynamic height based on number of attack vectors (x-axis categories)
        # For vertical bars, width is more of an issue; height is for overall plot.
        # num_x_categories = chart_data['display_attack_vector'].nunique()
        plot_height = 500  # Fixed height, or adjust dynamically if many x-categories cause label issues

        fig = px.bar(
            chart_data,
            x='display_attack_vector',
            y='count',
            color='display_privileges_required',
            labels={
                'display_attack_vector': 'Attack Vector',
                'count': 'Number of CVEs',
                'display_privileges_required': 'Privileges Required',
            },
            category_orders={
                "display_attack_vector": [
                    self.attack_vector_display_map.get(k, k)
                    for k in self.attack_vector_order
                ],
                "display_privileges_required": [
                    self.privileges_required_display_map.get(k, k)
                    for k in self.privileges_required_order
                ],
            },
            color_discrete_map=color_map_for_display_privs,
            barmode='stack',  # This is a stacked bar chart
            text_auto=True,
            height=plot_height,
        )

        # --- Calculate Y-axis range with padding ---
        # For stacked bars, we need the sum of 'count' for each 'display_attack_vector'
        max_stacked_height = 0
        if not chart_data.empty:
            max_stacked_height = (
                chart_data.groupby('display_attack_vector')['count']
                .sum()
                .max()
            )
        logger.info(f"Max stacked height: {max_stacked_height}")
        if pd.isna(max_stacked_height) or max_stacked_height <= 0:
            y_upper_limit = 10  # Default small range if no positive data
        else:
            # Add padding (e.g., 15-20% or fixed points)
            y_upper_limit = np.ceil(max_stacked_height * 1.20)
            if y_upper_limit < 10:  # Ensure a minimum sensible upper limit
                y_upper_limit = max(10, np.ceil(max_stacked_height + 2))
        logger.info(f"Y upper limit: {y_upper_limit}")
        # --- Standardized Layout ---
        fig.update_layout(
            title_text=title,
            title_x=0.5,
            xaxis_title="Attack Vector",
            yaxis_title="Number of CVEs",
            legend_title_text="<b>Privileges Required</b>",  # Bolded legend title
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.98,  # Position below plot
                xanchor="center",
                x=0.5,
                bordercolor="LightGrey",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.85)",
                font=dict(size=10),
            ),
            margin=dict(
                l=70, r=40, t=80, b=150
            ),  # Increased bottom margin for legend
            # --- Apply calculated Y-axis range ---
            yaxis_range=[0, y_upper_limit],
        )

        # Adjust text on bars for readability in stacks
        fig.update_traces(
            textposition='inside',
            insidetextanchor='middle',
            textfont_size=9,
            # cliponaxis=False # Not typically needed for 'inside' text
        )

        # X-axis tick label font and standoff
        fig.update_xaxes(
            showticklabels=True,
            tickfont=dict(size=10),
            title_standoff=10,
            ticks='outside',
            ticklen=5,
            tickwidth=1,
            tickcolor='LightGrey',
        )
        # Y-axis tick label font and standoff
        fig.update_yaxes(
            showticklabels=True,
            tickfont=dict(size=10),
            title_standoff=10,
            ticks='outside',
            ticklen=5,
            tickwidth=1,
            tickcolor='LightGrey',
        )

        return fig

    def _plot_interaction_vs_complexity_chart(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure | None:
        """Generates a heatmap for User Interaction vs. Attack Complexity with improved text and layout."""
        if chart_data is None or chart_data.empty:
            # ... (error handling) ...
            logger.info(
                "No data for _plot_interaction_vs_complexity_chart with title"
                f" '{title}'."
            )
            fig = go.Figure()
            fig.update_layout(title_text=title, title_x=0.5, height=300)
            fig.add_annotation(
                text="No data to display for heatmap.", showarrow=False
            )
            return fig

        try:
            ordered_y_display = [
                self.user_interaction_display_map.get(k, k)
                for k in self.user_interaction_order
            ]
            ordered_x_display = [
                self.attack_complexity_display_map.get(k, k)
                for k in self.attack_complexity_order
            ]

            heatmap_data_count = chart_data.pivot_table(
                index='display_user_interaction',
                columns='display_attack_complexity',
                values='count',
                aggfunc='sum',
            ).reindex(
                index=ordered_y_display,
                columns=ordered_x_display,
                fill_value=0,
            )

            heatmap_data_avg_cvss = chart_data.pivot_table(
                index='display_user_interaction',
                columns='display_attack_complexity',
                values='avg_cvss',
                aggfunc='mean',
            ).reindex(
                index=ordered_y_display,
                columns=ordered_x_display,
                fill_value=0,
            )

        except Exception as e:
            # ... (error handling) ...
            logger.error(
                "Error pivoting data for heatmap"
                f" 'interaction_vs_complexity': {e}"
            )
            fig = go.Figure()
            fig.update_layout(title_text=title, title_x=0.5)
            fig.add_annotation(
                text=f"Error preparing heatmap data: {e}",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
            )
            return fig

        custom_text_matrix = []
        for user_inter_display in ordered_y_display:
            row_text = []
            for att_comp_display in ordered_x_display:
                count_val = heatmap_data_count.loc[
                    user_inter_display, att_comp_display
                ]
                avg_cvss_val = heatmap_data_avg_cvss.loc[
                    user_inter_display, att_comp_display
                ]

                text_content = f"Count: {int(count_val)}"
                if (
                    count_val > 0 and avg_cvss_val > 0
                ):  # Only show avg_cvss if count > 0
                    text_content += f"<br>Avg CVSS: {avg_cvss_val:.1f}"
                elif count_val == 0:  # If count is 0, don't show avg_cvss line
                    text_content = f"Count: {int(count_val)}"

                row_text.append(text_content)
            custom_text_matrix.append(row_text)

        flat_avg_cvss = heatmap_data_avg_cvss.values.flatten()

        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data_count.values,
                x=ordered_x_display,
                y=ordered_y_display,
                colorscale=self.heatmap_custom_colorscale_blue_green,
                text=custom_text_matrix,
                texttemplate="%{text}",
                textfont=dict(color="white", size=11),
                hovertemplate=(
                    "<b>User Interaction:</b> %{y}<br>"
                    "<b>Attack Complexity:</b> %{x}<br>"
                    "<b>CVE Count:</b> %{z}<br>"
                    "<b>Avg CVSS:</b> %{customdata[0]:.1f}<extra></extra>"
                ),
                customdata=np.dstack((flat_avg_cvss,))[0],
                xgap=3,
                ygap=3,
                colorbar=dict(title='CVE Count'),  # Add title to colorbar
            )
        )

        fig.update_layout(
            title_text=title,
            title_x=0.5,
            xaxis_title="Attack Complexity",
            yaxis_title="User Interaction Requirement",
            xaxis_side="bottom",
            yaxis=dict(tickangle=-30, automargin=True),
            xaxis=dict(automargin=True),
            margin=dict(l=180, r=50, b=100, t=80, pad=10),
            plot_bgcolor='rgba(0,0,0,0)',
        )

        # --- Attempt to make text color dynamic (more robust) ---
        # This is an advanced technique if simple textfont.color isn't enough.
        # It creates a separate scatter trace just for text, allowing per-point color.
        # Note: This will overlay text; you might need to set text=[] in go.Heatmap above.
        # For this to work well, we need the textfont_color_matrix from previous attempts.

        # Re-calculate textfont_color_matrix based on z-values (counts)
        text_colors_for_scatter = []
        min_val = heatmap_data_count.values.min()
        max_val = heatmap_data_count.values.max()
        # Threshold for deciding black vs white text. (Midpoint of the Z values)
        # If z value is less than this threshold, cell is lighter, use black text. Else, white.
        threshold = min_val + (max_val - min_val) / 2
        if max_val == min_val:  # If all cells have same value
            threshold = (
                max_val + 1
            )  # Ensure all are considered 'lighter' if max_val is low, or 'darker' if max_val is high

        for r_idx, y_label in enumerate(ordered_y_display):
            for c_idx, x_label in enumerate(ordered_x_display):
                z_val = heatmap_data_count.iloc[r_idx, c_idx]
                if z_val < threshold:  # Lighter cell background
                    text_colors_for_scatter.append('black')
                else:  # Darker cell background
                    text_colors_for_scatter.append('white')

        # Reshape for scatter plot if needed, or apply directly if text is flat
        # text attribute in scatter takes a flat list corresponding to x and y

        # If you use this scatter overlay for text, clear text from heatmap:
        fig.update_traces(
            selector=dict(type='heatmap'), texttemplate="", text=[]
        )  # Clear text from heatmap trace

        text_x_coords, text_y_coords, cell_texts_for_scatter_flat = [], [], []
        for r_idx, y_val in enumerate(ordered_y_display):
            for c_idx, x_val in enumerate(ordered_x_display):
                text_x_coords.append(x_val)
                text_y_coords.append(y_val)
                cell_texts_for_scatter_flat.append(
                    custom_text_matrix[r_idx][c_idx].replace("<br>", "\n")
                )

        fig.add_trace(
            go.Scatter(
                x=text_x_coords,
                y=text_y_coords,
                text=cell_texts_for_scatter_flat,
                mode='text',
                textfont=dict(
                    size=11, color=text_colors_for_scatter  # Array of colors
                ),
                hoverinfo='skip',  # Don't want hover for this text layer
                showlegend=False,
            )
        )
        # --- End dynamic text color attempt ---

        return fig

    def _plot_cvss_score_by_category_chart(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure | None:
        if (
            chart_data is None
            or chart_data.empty
            or 'display_cve_category' not in chart_data.columns
        ):
            logger.info(
                "No data or missing 'display_cve_category' for consolidated"
                f" CVSS box plot, title '{title}'"
            )
            fig = go.Figure()
            fig.update_layout(title_text=title, title_x=0.5, height=300)
            fig.add_annotation(
                text="No CVSS score data by category to display.",
                showarrow=False,
            )
            return fig

        color_map_for_display_categories = {}
        # Ensure all categories present in chart_data get a color
        for display_name in chart_data['display_cve_category'].cat.categories:
            raw_cat_key_found = None
            for r_key, d_name in self.category_display_name_map.items():
                if d_name == display_name:
                    raw_cat_key_found = r_key
                    break
            if not raw_cat_key_found:
                raw_cat_key_found = display_name.lower().replace(' ', '_')
            color_map_for_display_categories[display_name] = (
                self.category_color_map.get(raw_cat_key_found, '#CCCCCC')
            )

        num_display_categories = chart_data['display_cve_category'].nunique()
        pixels_per_category_bar = 38  # Increased for more space per box plot
        base_height_for_titles_margins = (
            160  # Increased for more top/bottom margin
        )
        plot_height = max(
            450,
            (num_display_categories * pixels_per_category_bar)
            + base_height_for_titles_margins,
        )

        fig = px.box(
            chart_data,
            x='cvss_score',
            y='display_cve_category',
            orientation='h',
            color='display_cve_category',
            color_discrete_map=color_map_for_display_categories,
            category_orders={
                "display_cve_category": chart_data[
                    'display_cve_category'
                ].cat.categories.tolist()
            },
            labels={
                'cvss_score': 'CVSS Base Score',
                'display_cve_category': 'Vulnerability Category',
            },
            points="outliers",  # Show all points, including outliers. Could also be "outliers" or False.
            height=plot_height,
        )

        fig.update_layout(
            title_text=title,
            title_x=0.5,
            showlegend=False,
            xaxis_title="CVSS Base Score",
            # --- X-axis range to include all data points ---
            # Let Plotly auto-range x based on data (including outliers), then add a bit of padding.
            # Or, if you always want 0-10, ensure outliers beyond 10 are still conceptually plotted.
            # For now, let's ensure 0-10.5 is visible, Plotly will expand if data goes beyond.
            xaxis=dict(
                range=[0, 10.5],
                showline=False,
                linewidth=1,
                linecolor='rgb(204,204,204)',  # Make axis line visible
                mirror=True,  # Draw axis line on top and right edges too if desired
            ),
            yaxis_title="Vulnerability Category",
            yaxis=dict(
                automargin=True,
                tickfont=dict(size=10),
                title_standoff=25,  # << INCREASED Y-AXIS TITLE STANDOFF
                showline=True,
                linewidth=1,
                linecolor='rgb(204,204,204)',  # Make Y-axis line visible
                mirror=False,  # Usually not needed for Y if X has it
            ),
            margin=dict(
                l=260, r=40, t=80, b=80, pad=5
            ),  # Increased left margin, added right padding
        )

        # Adjust box plot visual thickness and other trace properties
        fig.update_traces(
            width=0.65,  # Makes horizontal boxes "taller"
            marker=dict(
                size=5, opacity=0.7
            ),  # Style for outlier points if shown
            line=dict(width=1),  # Thinner lines for box plot whiskers/median
        )

        return fig

    def _plot_kev_by_category_bar(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure:
        """Generates a horizontal bar chart for KEVs by Category."""
        # chart_data expects: 'cve_category', 'kev_count' (sorted by kev_count desc in data prep)
        fig = px.bar(
            chart_data,
            y='cve_category',  # Categories on Y for horizontal bars
            x='kev_count',
            orientation='h',
            labels={
                'cve_category': 'Vulnerability Category',
                'kev_count': 'Number of Known Exploited CVEs',
            },
            category_orders={
                "cve_category": chart_data['cve_category'].tolist()
            },  # Preserve sort order from data prep
            color='cve_category',
            color_discrete_map=self.category_color_map,
            text_auto=True,
        )
        fig.update_layout(
            showlegend=False, yaxis={'categoryorder': 'total ascending'}
        )  # Keep sorted order
        return fig

    def _plot_top_cwe_in_high_risk_cves_chart(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure | None:
        """Generates a horizontal bar chart for Top CWEs in Critical/Important CVEs."""

        if chart_data is None or chart_data.empty:
            # Return an empty figure or one with a message
            fig = go.Figure()
            fig.update_layout(title_text=title, title_x=0.5, height=200)
            fig.add_annotation(
                text="No Top CWE data to display.", showarrow=False
            )
            return fig

        plot_df = chart_data.copy()
        # Sort for plotting (highest count at top of horizontal bar chart)
        plot_df = plot_df.sort_values(by='count', ascending=True)

        fig = px.bar(
            plot_df,
            y='display_cwe_label',
            x='count',
            orientation='h',
            text='count',
            labels={
                'display_cwe_label': 'CWE',
                'count': 'Count in High-Risk CVEs',
            },
        )
        fig.update_layout(
            title_text=title,
            title_x=0.5,
            showlegend=False,
            yaxis_title="Common Weakness (CWE)",
            xaxis_title="Number of Occurrences in High-Risk CVEs",
            height=max(300, len(plot_df) * 35 + 100),  # Dynamic height
            margin=dict(
                l=300, t=80, b=50, r=30
            ),  # l=300 for potentially long CWE labels
        )
        fig.update_traces(marker_color='#4BAEEA', textposition='outside')
        return fig

    def _plot_top_windows_versions_by_criticality_chart(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure | None:
        """Generates a stacked horizontal bar chart for top Windows versions by CVE criticality with layout refinements."""
        if chart_data is None or chart_data.empty:
            logger.info(
                "No data for _plot_top_windows_versions_by_criticality_chart,"
                f" title '{title}'"
            )
            return None

        severity_group_order_for_plot = [
            'Critical',
            'High',
            'Medium/Low',
            'Other/None',
        ]

        plot_color_map = {
            'Critical': self.severity_color_map.get(
                'critical', '#E13B30'
            ),  # Fallbacks slightly darker
            'High': self.severity_color_map.get('important', '#F48C06'),
            'Medium/Low': self.severity_color_map.get('moderate', '#3498DB'),
            'Other/None': self.severity_color_map.get('none', '#BDC3C7'),
        }
        logger.info(f"chart_data sample: {chart_data.head()}")
        # Determine dynamic height based on number of products to ensure y-axis labels are not too cramped
        num_products = chart_data['windows_edition_version_arch'].nunique()
        # Allocate more pixels per bar if product names are long
        pixels_per_bar = 35  # Adjust this based on typical length of your product names and font size
        base_plot_height = 150  # For title, legend, x-axis
        plot_height = max(
            400, (num_products * pixels_per_bar) + base_plot_height
        )

        fig = px.bar(
            chart_data,
            y='windows_edition_version_arch',
            x='count',
            color='severity_group',
            orientation='h',
            labels={
                'windows_edition_version_arch': (
                    'Windows Edition / Version / Architecture'
                ),
                'count': 'Number of Unique CVEs',
                'severity_group': 'Severity Group',
            },
            category_orders={
                "severity_group": severity_group_order_for_plot,
                "windows_edition_version_arch": chart_data[
                    'windows_edition_version_arch'
                ].cat.categories.tolist(),
            },
            color_discrete_map=plot_color_map,
            barmode='stack',
            text_auto=True,  # Display count on each segment
            height=plot_height,  # Apply dynamic height
        )

        # --- Layout Refinements ---
        fig.update_layout(
            title_text=title,
            title_x=0.5,
            # --- Y-axis Title and Tick Label Spacing ---
            yaxis_title_text="Windows Edition / Version / Arch.",  # Slightly shorter for space
            yaxis=dict(
                automargin=True,  # Allow Plotly to try and make space for y-axis labels
                title_standoff=20,  # Increase space between y-axis title and y-axis tick labels
            ),
            xaxis_title_text="Number of Unique CVEs",
            # --- Legend Placement and Styling ---
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,  # Position below plot; adjust y and margin.b
                xanchor="center",
                x=0.5,
                bordercolor="LightGrey",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.85)",  # Slightly more opaque background
                title=dict(text="<b>Severity Group</b>", font=dict(size=11)),
                font=dict(size=10),
            ),
            # --- Margins to accommodate labels and legend ---
            # Increased left margin for potentially long Y-axis tick labels (product names)
            # Increased bottom margin for X-axis title and legend
            margin=dict(l=320, r=30, t=80, b=150, pad=5),
        )

        # --- Text on Bars ---
        # Adjust text position for stacked bars (inside, centered).
        # Smaller font if numbers are large or bars thin.
        fig.update_traces(
            textposition='inside', insidetextanchor='middle', textfont_size=9
        )

        return fig

    def _plot_days_to_nvd_by_severity_boxplots(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure | None:
        """Generates vertical box plots of 'Days to NVD Publication' by severity."""
        if chart_data is None or chart_data.empty:
            logger.info(
                "No data for _plot_days_to_nvd_by_severity_boxplots, title"
                f" '{title}'"
            )
            return None

        # Map colors for severity using display names as keys
        color_map_for_display_severity = {
            self.severity_display_name_map.get(key, key): color
            for key, color in self.severity_color_map.items()
        }

        fig = px.box(
            chart_data,
            x='display_severity_type',  # Categories on X-axis
            y='days_to_nvd',  # Values on Y-axis (vertical box plot)
            color='display_severity_type',  # Color by display name
            color_discrete_map=color_map_for_display_severity,
            category_orders={
                "display_severity_type": [
                    self.severity_display_name_map.get(k, k)
                    for k in self.severity_order
                ]
            },
            labels={
                'days_to_nvd': (
                    'Days to NVD Publication (from Vendor Patch)'
                ),  # Y-axis title
                'display_severity_type': 'CVE Severity',  # X-axis title
            },
            points="outliers",  # Show outliers, common for box plots. Use "all" for all points, False to hide.
            # orientation='v' is default, can be explicit
        )

        fig.update_layout(
            title_text=title,
            title_x=0.5,
            showlegend=False,  # Legend is redundant as X-axis categories are colored
            xaxis_title="CVE Severity",
            yaxis_title="Days to NVD Publication",
            margin=dict(l=50, r=20, t=80, b=80),  # Adjust margins
        )

        # Consider Y-axis range if extreme outliers skew the view too much,
        # or if negative values are very prevalent and you want to ensure 0 is visible.
        # Example: fig.update_yaxes(range=[-10, chart_data['days_to_nvd'].max() * 1.1 + 5])
        # This is highly data-dependent. Start with auto-range.

        return fig

    def _plot_spotlight_cves_cvss_scores_dot_plot(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure | None:
        """Generates a horizontal dot plot of CVSS scores for spotlighted CVEs with layout refinements."""
        if chart_data is None or chart_data.empty:
            logger.info(
                "No data for _plot_spotlight_cves_cvss_scores_dot_plot, title"
                f" '{title}'"
            )
            return None

        # Sort for y-axis order (highest score at the top)
        plot_data_sorted = chart_data.sort_values(
            by='cvss_score', ascending=True
        ).copy()

        # --- Prepare Y-axis labels with bolding ---
        plot_data_sorted['cve_id_bold'] = (
            '<b>' + plot_data_sorted['cve_id'].astype(str) + '</b>'
        )

        # --- Prepare text labels for scores with bolding ---
        plot_data_sorted['cvss_score_text_bold'] = plot_data_sorted[
            'cvss_score'
        ].apply(lambda x: f"<b>{x:.1f}</b>")

        fig = go.Figure()

        # Lollipop lines
        for index, row in plot_data_sorted.iterrows():
            fig.add_shape(
                type='line',
                x0=0,
                y0=row['cve_id_bold'],  # Use bolded CVE ID for y0/y1
                x1=row['cvss_score'],
                y1=row['cve_id_bold'],
                line=dict(
                    color=self.spotlight_plot_color, width=1, dash='dash'
                ),
            )  # Thinner, dashed line

        # Scatter trace for markers and text
        fig.add_trace(
            go.Scatter(
                x=plot_data_sorted['cvss_score'],
                y=plot_data_sorted[
                    'cve_id_bold'
                ],  # Use bolded CVE ID for y-axis categories
                mode='markers+text',
                marker=dict(
                    color=self.spotlight_plot_color, size=9
                ),  # Slightly smaller marker
                text=plot_data_sorted[
                    'cvss_score_text_bold'
                ],  # Use bolded score text
                textposition="middle right",
                textfont=dict(
                    size=10, color=self.spotlight_plot_color
                ),  # Text color same as marker
                hoverinfo='text',
                customdata=plot_data_sorted[
                    ['title', 'cve_id', 'cvss_score']
                ].values,
                hovertemplate=(
                    "<b>CVE ID:</b> %{customdata[1]}<br>"
                    "<b>Title:</b> %{customdata[0]}<br>"
                    "<b>CVSS Score:</b> %{customdata[2]:.1f}<extra></extra>"
                ),
            )
        )

        # Determine dynamic height based on number of CVEs
        num_cves = len(plot_data_sorted)
        pixels_per_cve_row = (
            35  # Adjust for desired spacing between y-axis labels
        )
        base_height_for_titles_margins = (
            120  # Space for title, x-axis, margins
        )
        plot_height = max(
            350,
            (num_cves * pixels_per_cve_row) + base_height_for_titles_margins,
        )

        fig.update_layout(
            title_text=title,
            title_x=0.5,
            xaxis_title="CVSS Base Score",  # Simpler x-axis title
            yaxis_title="Spotlighted CVEs",
            xaxis=dict(
                range=[
                    -0.5,
                    10.5,
                ],  # Start slightly before 0 for lollipop line origin
                zeroline=True,
                zerolinecolor='lightgrey',
                zerolinewidth=1,
                showgrid=True,
                gridcolor='whitesmoke',
                tickmode='array',  # Ensure specific ticks are shown
                tickvals=[0, 2, 4, 6, 8, 10],
                ticktext=['0', '2', '4', '6', '8', '10'],
            ),
            yaxis=dict(
                categoryorder='array',
                categoryarray=plot_data_sorted[
                    'cve_id_bold'
                ].tolist(),  # Use bolded list for order
                # tickfont=dict(size=10), # Control y-axis tick label font size if needed
                automargin=True,  # Helps fit y-axis labels
                title_standoff=15,  # Space between y-axis title and labels
            ),
            showlegend=False,
            height=plot_height,
            # --- Adjust margins and plot domain for Y-axis space ---
            margin=dict(
                l=200, r=60, t=80, b=80, pad=5
            ),  # Significantly increased left margin, increased right for text
            # paper_bgcolor='rgba(0,0,0,0)',  # Optional: transparent background for figure
            # plot_bgcolor='rgba(0,0,0,0)',  # Optional: transparent background for plot area
            # --- Optional: Constrain plot area if margins aren't enough ---
            # xaxis_domain=[0.25, 0.9],  # Example: X-axis takes 25% to 90% of figure width
            # This gives 25% on left for y-axis, 10% on right
        )

        return fig

    def _plot_cat_av_pr_chart(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure | None:
        if chart_data is None or chart_data.empty:
            logger.info(f"No data for _plot_cat_av_pr_chart title '{title}'")
            fig = go.Figure()
            fig.update_layout(title_text=title, title_x=0.5, height=300)
            fig.add_annotation(
                text="No data available for this chart.", showarrow=False
            )
            return fig

        color_map_for_display_privs = {
            self.privileges_required_display_map.get(key, key): color
            for key, color in self.privileges_required_color_map.items()
        }

        num_categories = chart_data['display_cve_category'].nunique()
        facet_col_wrap = 3
        num_rows = (num_categories + facet_col_wrap - 1) // facet_col_wrap

        row_height_alloc = 280
        plot_height = max(600, (row_height_alloc * num_rows) + 150)
        facet_row_spacing_val = 0.09  # Keep increased row spacing
        if num_rows <= 1:
            facet_row_spacing_val = 0.05

        # --- Horizontal Facet Spacing ---
        facet_col_spacing_val = (
            0.05  # Adjust this value (0 to 1) for space between facet columns
        )
        # Default is often small, e.g., 0.02 or 0.03

        fig = px.bar(
            chart_data,
            x='display_attack_vector',
            y='count',
            color='display_privileges_required',
            facet_col='display_cve_category',
            facet_col_wrap=facet_col_wrap,
            facet_row_spacing=facet_row_spacing_val,
            facet_col_spacing=facet_col_spacing_val,  # Apply column spacing
            labels={
                'display_attack_vector': 'Attack Vector',
                'count': 'CVE Count',
                'display_privileges_required': 'Privileges',
                'display_cve_category': 'Vulnerability Category',
            },
            category_orders={
                "display_attack_vector": [
                    self.attack_vector_display_map.get(k, k)
                    for k in self.attack_vector_order
                ],
                "display_privileges_required": [
                    self.privileges_required_display_map.get(k, k)
                    for k in self.privileges_required_order
                ],
                "display_cve_category": [
                    self.category_display_name_map.get(k, k)
                    for k in self.category_order
                ],
            },
            color_discrete_map=color_map_for_display_privs,
            barmode='group',
            text_auto=True,
            height=plot_height,
        )

        fig.update_layout(
            title_text=title,
            title_x=0.5,
            legend=dict(
                orientation="h",  # Horizontal items
                x=0.9,  # paper’s right edge
                xanchor="right",  # legend’s right edge ↦ x=1
                y=0.27,  # 2% up from paper bottom
                yanchor="top",  # legend’s top edge ↦ y=0.02
                bordercolor="LightGrey",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.85)",
                font=dict(size=10),
                title=dict(
                    text=" <b>Privileges Required</b>",  # Explicit legend title, can be bolded
                    font=dict(
                        size=11
                    ),  # Slightly larger/different font for title if desired
                ),
            ),
            margin=dict(
                l=130, r=40, t=100, b=200
            ),  # Keep increased bottom margin
            # give more room on the left edge 'l=120'
        )

        fig.for_each_annotation(
            lambda a: a.update(
                text=f"<b>{a.text.split('=')[-1]}</b>", font=dict(size=11)
            )
        )

        # --- X-axis and Y-axis updates for ALL facets ---
        fig.update_xaxes(
            title_text="Attack Vector",
            title_font=dict(size=11),
            title_standoff=15,
            showticklabels=True,
            ticks="outside",
            ticklen=5,
            tickfont=dict(size=9),
            # matches=None  # Decouple x-axes (already done)
        )

        # --- Y-axis: Ensure shared scale and control title visibility ---
        fig.update_yaxes(
            # title_text="CVE Count", # Global title applied below or per first-col facet
            title_font=dict(size=11),
            title_standoff=20,  # push the title farther left
            showticklabels=True,
            ticks='',
            tickfont=dict(size=9),
            # matches='y'           # CRITICAL: Ensure all Y-axes share the same scale (0 to max)
        )

        # Loop to set Y-axis title "CVE Count" ONLY on the first facet of each row
        # AND to ensure all y-axes actually use the same range (if px didn't do it perfectly)
        max_y_val_overall = 0
        if 'count' in chart_data.columns and not chart_data['count'].empty:
            max_y_val_overall = chart_data['count'].max()

        if pd.isna(max_y_val_overall) or max_y_val_overall <= 0:
            # If no positive counts, set a default small range like 0-5 for visual structure
            y_upper_limit = 5
        else:
            # Add padding: either a percentage or a fixed number of points
            # Option 1: Percentage padding (e.g., 15% of max value)
            # y_upper_limit = max_y_val_overall * 1.15

            # Option 2: Fixed points padding (e.g., 2 to 4 points above max)
            # Ensure it's an integer if your counts are integers for cleaner ticks
            padding_points = 3
            y_upper_limit = np.ceil(
                max_y_val_overall + padding_points
            )  # Using ceil for integer upper limit

            # Ensure a minimum range if max_y_val_overall is very small, e.g. 1 or 2
            if (
                y_upper_limit < 5
            ):  # if max bar is 1, padding makes it 4, still small.
                y_upper_limit = max(
                    5, np.ceil(max_y_val_overall + 1)
                )  # Ensure at least a bit of space or min height of 5

        common_yrange = [0, y_upper_limit]

        # Loop to set Y-axis title "CVE Count" ONLY on the first facet of each row.
        for i in range(
            1, num_categories + 1
        ):  # num_categories from your function
            is_first_col_facet = (
                i - 1
            ) % facet_col_wrap == 0  # facet_col_wrap from your function
            yaxis_name = f"yaxis{i if i > 1 else ''}"
            xaxis_name = (  # For bottom row x-tick logic
                f"xaxis{i if i > 1 else ''}"
            )

            if yaxis_name in fig.layout:
                if is_first_col_facet:
                    fig.layout[yaxis_name].title.text = "<b>CVE Count</b>"
                    fig.layout[yaxis_name].title.standoff = 20
                else:
                    fig.layout[yaxis_name].title.text = ""
                # Ensure all y-axes have the same range for comparability
                if common_yrange:
                    fig.layout[yaxis_name].range = common_yrange

            # Logic for ensuring x-axis tick labels on bottom row facets
            if xaxis_name in fig.layout:
                current_row_idx = (i - 1) // facet_col_wrap
                if current_row_idx == (
                    num_rows - 1
                ):  # num_rows from your function
                    fig.layout[xaxis_name].showticklabels = True
                # else: # If you wanted to hide x-ticks on non-bottom rows and x-axes are NOT matched
                #    fig.layout[xaxis_name].showticklabels = Fals

        fig.update_traces(
            textfont_size=9,
            textangle=0,
            textposition="outside",
            cliponaxis=False,
        )
        # fig.show(config={"responsive": True})
        # logger.info(f"legend positioning: {fig.layout.legend}")
        return fig

    def _plot_cat_ui_ac_chart(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure | None:
        logger.info(
            f"Plotting _plot_cat_ui_ac_chart, type: {type(chart_data)}. Chart"
            f" data: {chart_data.head()}"
        )
        if chart_data is None or chart_data.empty:
            # ... (no data handling) ...
            logger.info(f"No data for _plot_cat_ui_ac_chart, title '{title}'")
            fig = go.Figure()
            fig.update_layout(title_text=title, title_x=0.5, height=300)
            fig.add_annotation(
                text="No data available for this chart.", showarrow=False
            )
            return fig

        color_map_for_display_ac = {  # ... (color map definition) ...
            self.attack_complexity_display_map.get(key, key): color
            for key, color in self.attack_complexity_color_map.items()
        }

        num_categories = chart_data['display_cve_category'].nunique()
        facet_col_wrap = 3
        num_rows = (num_categories + facet_col_wrap - 1) // facet_col_wrap

        row_height_alloc = 280
        plot_height = max(600, (row_height_alloc * num_rows) + 150)
        facet_row_spacing_val = 0.085
        if num_rows <= 1:
            facet_row_spacing_val = 0.05
        facet_col_spacing_val = 0.05

        fig = px.bar(
            chart_data,
            x='display_user_interaction',
            y='count',
            color='display_attack_complexity',
            facet_col='display_cve_category',
            facet_col_wrap=facet_col_wrap,
            facet_row_spacing=facet_row_spacing_val,
            facet_col_spacing=facet_col_spacing_val,
            labels={
                'display_user_interaction': 'User Interaction',
                'count': 'CVE Count',
                'display_attack_complexity': 'Complexity',
                'display_cve_category': 'Vulnerability Category',
            },
            category_orders={  # ... (category_orders definition) ...
                "display_user_interaction": [
                    self.user_interaction_display_map.get(k, k)
                    for k in self.user_interaction_order
                ],
                "display_attack_complexity": [
                    self.attack_complexity_display_map.get(k, k)
                    for k in self.attack_complexity_order
                ],
                "display_cve_category": [
                    self.category_display_name_map.get(k, k)
                    for k in self.category_order
                ],
            },
            color_discrete_map=color_map_for_display_ac,
            barmode='group',
            text_auto=True,
            height=plot_height,
        )

        fig.update_layout(  # ... (title, legend, margin as before) ...
            title_text=title,
            title_x=0.5,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.27,
                xanchor="right",
                x=0.9,
                bordercolor="LightGrey",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.85)",
                font=dict(size=10),
                title=dict(
                    text=" <b>Attack Complexity</b>",  # Explicit legend title, can be bolded
                    font=dict(
                        size=11
                    ),  # Slightly larger/different font for title if desired
                ),
            ),
            margin=dict(l=70, r=40, t=100, b=180),
        )

        fig.for_each_annotation(
            lambda a: a.update(
                text=f"<b>{a.text.split('=')[-1]}</b>", font=dict(size=11)
            )
        )

        fig.update_xaxes(
            title_text="User Interaction",
            title_font=dict(size=11),
            title_standoff=10,  # Adjusted for closer X-axis title to ticks
            showticklabels=True,
            ticks="outside",
            ticklen=5,
            tickfont=dict(size=9),
            matches=None,  # Ensure x-axes are independent for labeling
        )

        fig.update_yaxes(
            title_font=dict(size=11),  # Default for y-axis titles
            title_standoff=15,
            showticklabels=True,  # Default to show y-tick labels
            tickfont=dict(size=9),
            # NO global matches='y' here
        )

        # Loop to set Y-axis title "CVE Count" ONLY on the first facet of each row
        # AND explicitly set common y-axis range.
        max_y_val_overall = 0
        if 'count' in chart_data.columns and not chart_data['count'].empty:
            max_y_val_overall = chart_data['count'].max()

        if pd.isna(max_y_val_overall) or max_y_val_overall <= 0:
            # If no positive counts, set a default small range like 0-5 for visual structure
            y_upper_limit = 5
        else:
            # Add padding: either a percentage or a fixed number of points
            # Option 1: Percentage padding (e.g., 15% of max value)
            # y_upper_limit = max_y_val_overall * 1.15

            # Option 2: Fixed points padding (e.g., 2 to 4 points above max)
            # Ensure it's an integer if your counts are integers for cleaner ticks
            padding_points = 3
            y_upper_limit = np.ceil(
                max_y_val_overall + padding_points
            )  # Using ceil for integer upper limit

            # Ensure a minimum range if max_y_val_overall is very small, e.g. 1 or 2
            if (
                y_upper_limit < 5
            ):  # if max bar is 1, padding makes it 4, still small.
                y_upper_limit = max(
                    5, np.ceil(max_y_val_overall + 1)
                )  # Ensure at least a bit of space or min height of 5

        common_yrange = [0, y_upper_limit]

        for i in range(1, num_categories + 1):
            is_first_col_facet = (i - 1) % facet_col_wrap == 0
            current_row_idx = (
                i - 1
            ) // facet_col_wrap  # For x-axis bottom row check

            yaxis_name = f"yaxis{i if i > 1 else ''}"
            xaxis_name = (  # For x-axis bottom row check
                f"xaxis{i if i > 1 else ''}"
            )

            if yaxis_name in fig.layout:
                if is_first_col_facet:
                    fig.layout[yaxis_name].title.text = "CVE Count"
                else:
                    fig.layout[yaxis_name].title.text = ""
                fig.layout[yaxis_name].range = (
                    common_yrange  # Enforce common y-axis range
                )

            # Ensure x-axis tick labels are shown on bottom row facets
            if xaxis_name in fig.layout:
                if current_row_idx == (
                    num_rows - 1
                ):  # num_rows calculated earlier
                    fig.layout[xaxis_name].showticklabels = True
                # else: # If x-axes are decoupled (matches=None), other rows will show ticks by default from update_xaxes
                #    pass # No need to hide them unless specifically desired

        fig.update_traces(
            textfont_size=9,
            textangle=0,
            textposition="outside",
            cliponaxis=False,
        )
        return fig

    def _plot_top_cwe_av_cvss_chart(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure | None:
        if (
            chart_data is None
            or chart_data.empty
            or 'display_primary_cwe_id' not in chart_data.columns
        ):
            logger.info(
                "No data or missing columns for _plot_top_cwe_av_cvss_chart,"
                f" title '{title}'"
            )
            fig = go.Figure()
            fig.update_layout(title_text=title, title_x=0.5, height=300)
            fig.add_annotation(
                text="No Top CWE data by Attack Vector to display.",
                showarrow=False,
            )
            return fig

        color_map_for_display_av = {
            self.attack_vector_display_map.get(key, key): color
            for key, color in self.attack_vector_color_map.items()
        }

        num_facets = chart_data['display_primary_cwe_id'].nunique()
        facet_col_wrap = min(3, num_facets) if num_facets > 0 else 1
        num_rows = (
            (num_facets + facet_col_wrap - 1) // facet_col_wrap
            if num_facets > 0
            else 1
        )

        # --- Standardized Height and Spacing ---
        row_height_alloc = 200
        base_plot_height_for_titles_etc = 150
        plot_height = max(
            400,
            (row_height_alloc * num_rows) + base_plot_height_for_titles_etc,
        )

        facet_row_spacing_val = (
            0.12  # Standardized vertical spacing between facet rows
        )
        if num_rows <= 1:
            facet_row_spacing_val = 0.1
        facet_col_spacing_val = 0.06  # Standardized horizontal facet spacing
        # --- End Spacing ---

        fig = px.box(
            chart_data,
            x='cvss_score',
            y='display_attack_vector',
            color='display_attack_vector',
            color_discrete_map=color_map_for_display_av,
            facet_col='display_primary_cwe_id',
            facet_col_wrap=facet_col_wrap,
            facet_row_spacing=facet_row_spacing_val,
            facet_col_spacing=facet_col_spacing_val,
            orientation='h',
            labels={
                'cvss_score': 'CVSS Base Score',
                'display_attack_vector': 'Attack Vector',
                'display_primary_cwe_id': 'CWE',
            },
            category_orders={
                "display_attack_vector": [
                    self.attack_vector_display_map.get(k, k)
                    for k in self.attack_vector_order
                ],
                "display_primary_cwe_id": chart_data[
                    'display_primary_cwe_id'
                ].cat.categories.tolist(),
            },
            points="outliers",
            height=plot_height,
        )

        # Box plot "thickness"
        fig.update_traces(width=0.33)

        # --- Standardized Layout Refinements ---
        fig.update_layout(
            title_text=title,
            title_x=0.5,
            legend_title_text=" <b>Attack Vector</b>",  # Bolded legend title
            # --- Standardized Legend Placement (Bottom Center, Horizontal) ---
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.3,  # Position legend below plot area
                xanchor="center",
                x=0.85,
                bordercolor="rgba(200,200,200,0.7)",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.9)",
                font=dict(size=10),
                # title=dict(font=dict(size=11)) # If legend_title_text not enough
            ),
            # Margins: Increased left for Y-axis elements, Increased bottom for X-axis titles & legend
            margin=dict(l=160, r=30, t=100, b=120, pad=5),
            # --- Define X-axis domain to leave space for Y-axis elements ---
            # This will apply to all x-axes created by faceting.
            # It means the plot area for x values will start at 20% from the left of the figure.
            # xaxis_domain=[x_domain_start, x_domain_end],
        )

        # Bold facet titles & adjust font
        fig.for_each_annotation(
            lambda a: a.update(
                text=f"<b>{a.text.split('=')[-1]}</b>", font=dict(size=10)
            )
        )

        # X-axis: Title, standoff, ticks. Shared scale (matches='x') is good for CVSS.
        fig.update_xaxes(
            title_text="<b>CVSS Base Score</b>",
            title_font=dict(size=10),
            title_standoff=15,  # Standardized standoff
            showticklabels=True,
            ticks="",
            ticklen=0,
            tickfont=dict(size=9),
            range=[0, 10.5],  # Consistent 0-10 CVSS scale
            tickvals=[0, 2, 4, 6, 8, 10],
            matches='x',  # Share x-axis scale across facets
        )

        # Y-axis (within facets): Title and ticks only on first column. Shared categories.
        fig.update_yaxes(
            # No global title_text here, set in loop
            title_font=dict(size=10),
            tickfont=dict(size=9),
            ticklen=0,
            # matches='y' # Let Plotly match y-axes if categories are identical
        )

        # Loop to manage Y-axis titles/ticks per facet & ensure X-ticks on bottom row
        for i in range(1, num_facets + 1):
            is_first_col_facet = (i - 1) % facet_col_wrap == 0
            current_row_idx = (i - 1) // facet_col_wrap

            y_axis_name = f"yaxis{i if i > 1 else ''}"
            x_axis_name = f"xaxis{i if i > 1 else ''}"

            if y_axis_name in fig.layout:
                fig.layout[y_axis_name].showticklabels = is_first_col_facet
                if is_first_col_facet:
                    fig.layout[y_axis_name].title.text = "<b>Attack Vector</b>"
                    fig.layout[y_axis_name].title.standoff = (
                        14  # Ample standoff for Y-title
                    )
                    fig.layout[y_axis_name].automargin = True
                else:
                    fig.layout[y_axis_name].title.text = ""

            if x_axis_name in fig.layout:
                # If x-axes are matched (matches='x'), Plotly shows ticks/title only on bottom-most of the FIGURE.
                # To show on all bottom-row FACETS, we need to be explicit if not matched,
                # or this loop ensures it for the matched group too.
                if current_row_idx == (num_rows - 1):
                    fig.layout[x_axis_name].showticklabels = True
                    fig.layout[
                        x_axis_name
                    ].title.text = (  # Ensure title on bottom row facets
                        "<b>CVSS Base Score</b>"
                    )
                elif not fig.layout[
                    x_axis_name
                ].matches:  # If x-axes somehow became un-matched
                    fig.layout[x_axis_name].showticklabels = (
                        False  # Hide on inner rows
                    )
                    fig.layout[x_axis_name].title.text = ""

        return fig

    def _plot_tsne_cve_profile_chart(
        self, chart_data: pd.DataFrame, title: str
    ) -> go.Figure | None:
        if (
            chart_data is None
            or chart_data.empty
            or 'cluster_label' not in chart_data.columns
        ):
            logger.info(
                "No data or missing 'cluster_label' for"
                f" _plot_tsne_cve_profile_chart, title '{title}'"
            )
            fig = go.Figure()
            fig.update_layout(title_text=title, title_x=0.5, height=300)
            fig.add_annotation(
                text="Not enough data or clusters to display t-SNE plot.",
                showarrow=False,
            )
            return fig

        # Ensure 'cluster_label' is categorical for consistent color mapping and legend order
        # Order clusters by their numeric part if names are like "Cluster X (Unnamed)" or by a predefined order if all are named
        # For now, let's sort unique labels alphabetically for consistent legend, or you can define a specific order.
        def sort_key_cluster_labels(label):  # Helper for sorting legend
            if "(Auto)" in label:
                return (2, label)
            if "Profile Group" in label and "(Empty)" not in label:
                return (1, label)
            return (0, label)

        unique_cluster_labels = sorted(
            chart_data['cluster_label'].unique(), key=sort_key_cluster_labels
        )
        chart_data['cluster_label'] = pd.Categorical(
            chart_data['cluster_label'],
            categories=unique_cluster_labels,
            ordered=True,
        )

        # Create a color map for the cluster labels present in the data
        # Ensure we have enough colors. If more clusters than colors, colors will repeat.
        # num_unique_clusters = len(unique_cluster_labels)
        dynamic_cluster_color_map = {
            label: self.cluster_colors_list[i % len(self.cluster_colors_list)]
            for i, label in enumerate(unique_cluster_labels)
        }

        # Prepare hover text with more readable feature values
        hover_text_lines = [
            "<b>CVE ID:</b> %{customdata[0]}",
            "<b>CVSS:</b> %{customdata[1]:.1f}",
            "<b>Profile:</b> %{customdata[2]}",  # Cluster Name
            "<br>--- Features ---",
            "Category: %{customdata[3]}",
            "Attack Vector: %{customdata[4]}",
            "Complexity: %{customdata[5]}",
            "Privs Req.: %{customdata[6]}",
            "User Interaction: %{customdata[7]}",
            "<extra></extra>",  # Removes trace info
        ]

        # Map original categorical values to display values for hover
        chart_data_hover = chart_data.copy()
        mapped_cve_cat_hover = chart_data_hover['cve_category'].map(
            self.category_display_name_map
        )
        chart_data_hover['cve_category_display'] = mapped_cve_cat_hover.fillna(
            chart_data_hover['cve_category'].astype(object)
        )

        mapped_av_hover = chart_data_hover['attack_vector'].map(
            self.attack_vector_display_map
        )
        chart_data_hover['attack_vector_display'] = mapped_av_hover.fillna(
            chart_data_hover['attack_vector'].astype(object)
        )

        mapped_ac_hover = chart_data_hover['attack_complexity'].map(
            self.attack_complexity_display_map
        )
        chart_data_hover['attack_complexity_display'] = mapped_ac_hover.fillna(
            chart_data_hover['attack_complexity'].astype(object)
        )

        mapped_pr_hover = chart_data_hover['privileges_required'].map(
            self.privileges_required_display_map
        )
        chart_data_hover['privileges_required_display'] = (
            mapped_pr_hover.fillna(
                chart_data_hover['privileges_required'].astype(object)
            )
        )

        mapped_ui_hover = chart_data_hover['user_interaction'].map(
            self.user_interaction_display_map
        )
        chart_data_hover['user_interaction_display'] = mapped_ui_hover.fillna(
            chart_data_hover['user_interaction'].astype(object)
        )

        fig = px.scatter(
            chart_data_hover,
            x='tsne_x',
            y='tsne_y',
            color='cluster_label',
            size='cvss_score',  # Keep size by CVSS, it adds a useful dimension
            opacity=0.75,
            custom_data=[
                'cve_id',
                'cvss_score',
                'cluster_label',  # cluster_label is for hover, not just color
                'cve_category_display',
                'attack_vector_display',
                'attack_complexity_display',
                'privileges_required_display',
                'user_interaction_display',
            ],
            color_discrete_map=dynamic_cluster_color_map,
            labels={
                'cluster_label': 'Identified CVE Profile'
            },  # For legend title
            size_max=18,
        )

        fig.update_traces(
            hovertemplate="<br>".join(hover_text_lines),
            marker=dict(line=dict(width=0.5, color='DarkSlateGrey')),
        )

        fig.update_layout(
            title_text=title,
            title_x=0.5,
            xaxis_title="t-SNE Component 1 (Abstract Dimension)",  # Clarify abstract nature
            yaxis_title="t-SNE Component 2 (Abstract Dimension)",
            legend_title_text='CVE Profile',  # Updated legend title
            legend=dict(
                orientation="v",  # Vertical stacking of legend items
                yanchor="top",
                y=1,  # Align top of legend with top of plot area
                xanchor="left",
                x=1.01,  # Position legend just to the right of plot area
                bordercolor="rgba(200,200,200,0.7)",  # Slightly more transparent border
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.9)",  # Background for legend
                font=dict(size=9),  # Slightly smaller legend font
                # itemwidth=30 # Can set if you want to control wrapping within legend items
            ),
            margin=dict(
                l=60, r=200, t=80, b=60, pad=5
            ),  # Adjusted top margin for legend
            height=550,
        )

        return fig

    def _generate_plotly_figure(
        self,
        chart_data: Any,
        chart_key: str,
        chart_title_override: Optional[str] = None,
    ) -> Optional[go.Figure]:
        """Creates a Plotly figure object using a dispatch dictionary."""
        logger.info(f"Generating Plotly figure for: {chart_key}")

        if chart_data is None or (
            isinstance(chart_data, pd.DataFrame) and chart_data.empty
        ):
            logger.warning(
                f"No data provided or empty DataFrame for chart '{chart_key}'."
                " Cannot generate figure."
            )
            return None

        # Determine the chart title
        final_chart_title = (
            chart_title_override  # 1. Prioritize override passed directly
        )

        if (
            final_chart_title is None
        ):  # 2. If no override, try REPORT_STRUCTURE
            # Flatten the search for the chart definition
            chart_definition_found = None
            for section_list_key in ["explicit_sections", "looped_sections"]:
                for section_config in self.REPORT_STRUCTURE.get(
                    section_list_key, {}
                ).values():
                    for chart_def in section_config.get('charts', []):
                        if chart_def.get('id') == chart_key:
                            chart_definition_found = chart_def
                            break
                    if chart_definition_found:
                        break
                if chart_definition_found:
                    break

            if chart_definition_found:
                # Use the "title" key from REPORT_STRUCTURE first, then "caption" as fallback
                final_chart_title = chart_definition_found.get(
                    'title', chart_definition_found.get('caption')
                )

            if final_chart_title is None:  # 3. Ultimate fallback
                final_chart_title = chart_key.replace('_', ' ').title()
                logger.debug(
                    f"Using default generated title for '{chart_key}':"
                    f" {final_chart_title}"
                )

        # Now 'final_chart_title' is the definitive title to use.
        # It will be passed to the handler (though the handler won't use it directly for px.title)
        # and then used in fig.update_layout(title_text=final_chart_title)

        fig_handler_method = self.PLOTLY_FIGURE_HANDLERS.get(chart_key)
        fig: Optional[go.Figure] = None

        if fig_handler_method:
            try:
                logger.info(
                    f"Using figure handler: {fig_handler_method.__name__} for"
                    f" chart '{chart_key}'"
                )
                # Pass the determined chart_title to the specific plotting function
                # chart_title is currently unused by _plot helpers
                fig = fig_handler_method(chart_data, final_chart_title)
            except Exception as e:
                logger.exception(
                    "Error executing figure handler"
                    f" '{fig_handler_method.__name__}' for chart"
                    f" '{chart_key}': {e}"
                )
                return None  # Return None if specific handler fails
        else:
            logger.warning(
                "No Plotly figure generation handler defined for chart key"
                f" '{chart_key}'."
            )
            return None

        # Apply common layout adjustments AFTER specific chart logic if figure was created
        if fig:
            try:
                # Ensure pio.templates['report_light'] exists or use a default
                active_theme_name = 'plotly_white'  # A safe default
                if (
                    'report_light' in pio.templates
                ):  # Check if your custom template is registered
                    active_theme_name = 'report_light'
                else:
                    logger.debug(
                        "Custom Plotly template 'report_light' not found,"
                        " using 'plotly_white'."
                    )

                common_layout_updates = dict(
                    template=pio.templates[active_theme_name],
                    title_text=final_chart_title,
                    title_x=0.05,
                    margin=dict(l=50, r=40, t=80, b=50),  # Standard margins
                )

                # Only apply a default legend style if the figure doesn't seem to have one specifically set.
                # This is a heuristic. A better way is if plot handlers could signal not to apply default.
                if fig.layout.legend is None or (
                    fig.layout.legend
                    and fig.layout.legend.orientation is None
                    and fig.layout.legend.x is None
                ):
                    common_layout_updates['legend_title_text'] = (
                        ''  # Default no legend title
                    )
                    common_layout_updates['legend'] = dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                    )
                elif (
                    fig.layout.legend and fig.layout.legend.title.text is None
                ):  # If legend exists but no title
                    common_layout_updates['legend_title_text'] = ''

                fig.update_layout(**common_layout_updates)

            except Exception as e:
                logger.exception(
                    "Error applying common layout updates for chart"
                    f" '{chart_key}': {e}"
                )
                # Decide if you want to return the fig without common layout or None
                # return None
        else:  # fig is None from handler
            logger.warning(
                f"Figure object is None after handler for '{chart_key}',"
                " cannot apply common layout."
            )
        logger.info(
            f"Generated figure for chart '{chart_key}' with"
            f" [{fig_handler_method.__name__}]"
        )
        return fig

    # ------------------------------------------------------------------------------
    # --- END PLOTTING FUNCTIONS ---------------------------------------------------
    # ------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------
    # --- BEGIN EXPORT FUNCTIONS ---------------------------------------------------
    # ------------------------------------------------------------------------------

    def _export_chart_json(
        self, fig: go.Figure, chart_id: str, data_output_dir: Path
    ) -> Optional[Path]:
        """
        Exports figure JSON to the specified data directory. Returns Path if successful.

        :param fig: Plotly figure to export
        :param chart_id: ID of the chart
        :param data_output_dir: Directory to save the JSON file
        :return: Path to the exported JSON file if successful, None otherwise
        """
        filepath = data_output_dir / f"{chart_id}.json"
        try:
            # Ensure directory exists (should be created by generate_report)
            pio.write_json(fig, filepath)
            logger.debug(f"Exported chart data to: {filepath}")
            return filepath
        except Exception as e:
            logger.warning(
                f"Error exporting chart JSON for '{chart_id}' to"
                f" '{filepath}': {e}"
            )
            return None

    def _export_chart_image(
        self,
        fig: go.Figure,
        chart_id: str,  # Use the HTML-friendly chart ID (e.g., chart-new-vs-updated)
        image_output_dir: Path,
        image_format: str = "png",  # Common format, supported by kaleido
        width: int = 800,  # Adjust resolution as needed
        height: int = 600,
        scale: int = 2,  # Increase scale for better resolution
    ) -> Optional[Path]:
        """
        Exports figure as a static image to the specified directory. Requires Kaleido.

        Args:
            fig: Plotly figure to export.
            chart_id: HTML-friendly ID of the chart (used for filename).
            image_output_dir: Directory to save the image file.
            image_format: Output format (png, jpeg, webp, svg, pdf).
            width: Image width in pixels.
            height: Image height in pixels.
            scale: Scale factor for higher resolution.

        Returns:
            Path to the exported image file if successful, None otherwise.
        """
        if fig is None:
            logger.warning(
                f"Cannot export image for '{chart_id}', figure is None."
            )
            return None
        # start_str = self.start_date.strftime("%b_%Y").lower()
        # end_str = self.end_date.strftime("%b_%Y").lower()
        filepath = image_output_dir / f"{chart_id}.{image_format}"
        try:
            # Ensure directory exists (should be created by generate_report)
            # Use write_image from plotly.io
            pio.write_image(
                fig,
                str(filepath),  # write_image expects a string path
                format=image_format,
                width=width,
                height=height,
                scale=scale,
            )
            logger.debug(f"Exported chart image to: {filepath}")
            return filepath
        except ValueError as ve:
            # Catch specific error if Kaleido is missing
            if "kaleido" in str(ve).lower():
                logger.error(
                    "Kaleido engine not found. Please install ('pip install -U"
                    " kaleido') to export static images."
                )
                # Optionally raise, or just return None to allow report generation without images
                return None
            else:
                logger.error(
                    f"Error exporting chart image for '{chart_id}' to"
                    f" '{filepath}': {ve}",
                    exc_info=True,
                )
                return None

    # ------------------------------------------------------------------------------
    # --- END EXPORTING FUNCTIONS --------------------------------------------------
    # ------------------------------------------------------------------------------

    def _format_data_for_llm(
        self, chart_data: Any, format_type: str = 'text'
    ) -> str:
        """Converts chart data into a string for the LLM prompt."""
        if chart_data is None:
            return "N/A"  # noqa E701
        try:
            if isinstance(chart_data, pd.DataFrame):
                if chart_data.empty:
                    return "N/A (DataFrame is empty)"  # noqa E701
                if format_type == 'markdown':
                    return chart_data.to_markdown(index=False)
                else:  # Default to text string
                    return chart_data.to_string(
                        index=False, max_rows=20
                    )  # Limit rows
            elif isinstance(chart_data, dict):
                # Simple key-value pairs often work well
                return "\n".join([
                    f"- {key.replace('_', ' ').title()}: {value}"
                    for key, value in chart_data.items()
                ])
            else:
                return str(chart_data)
        except Exception as e:
            logging.warning(f"Could not format data for LLM: {e}")
            return "Error formatting data."

    def _get_task_type_from_prompt_name(
        self, prompt_template_name: str
    ) -> str:
        """Infers the task type from the Jinja template filename."""
        # Simple inference based on keywords in the filename
        if (
            "chart" in prompt_template_name
            and "insight" in prompt_template_name
        ):
            return "chart_insight"
        elif "callout" in prompt_template_name:
            return "callout"
        elif "narrative" in prompt_template_name:
            return "narrative"
        elif (
            "summary" in prompt_template_name
            and "executive" not in prompt_template_name
        ):  # Avoid matching executive_summary again
            return "section_summary"
        elif (
            "summary" in prompt_template_name
            and "executive" in prompt_template_name
        ):  # Avoid matching executive_summary again
            return "executive_summary"
        elif "conclusion" in prompt_template_name:
            return "report_conclusion"
        else:
            logger.warning(
                "Could not determine specific task type for prompt"
                f" '{prompt_template_name}'. Using default."
            )
            return "default"

    async def _get_llm_response(
        self,
        prompt_template_name: str,  # e.g., "section3_callout_1.j2"
        prompt_context_data: Dict[str, Any],  # Contains base context + config
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.7,
        llm_max_tokens: int = 1024,
        image_paths: Optional[List[Path]] = None,
        llm_top_p: float = 0.95,
        llm_presence_penalty: float = 0.4,
        llm_frequency_penalty: float = 0.2,
        llm_stream: bool = False,
        llm_logprobs: bool = False,
        llm_top_logprobs: int = 0,
        llm_timeout: int = 120,
        llm_response_format: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """
        Generates an LLM insight using a specified Jinja prompt template.
        Can optionally include image paths for multimodal models.
        """
        request_timestamp = datetime.now(timezone.utc)
        rendered_prompt = ""
        system_prompt = ""
        task_type = self._get_task_type_from_prompt_name(prompt_template_name)
        input_tokens_rendered = 0
        input_tokens_system = 0
        text_response = (  # Default error response
            "Error: LLM response generation failed for"
            f" {prompt_template_name}."
        )
        usage_info_from_api: Optional[Dict[str, int]] = (
            None  # To store what API returns
        )

        logger.info(
            "Generating LLM insight using template:"
            f" {prompt_template_name} with model: {llm_model}"
        )
        if not self.llm:
            logger.warning(
                "LLM client not configured/injected. Returning placeholder."
            )
            # Log a dummy usage entry for attempts
            text_response, usage_info_from_api = (
                f"Placeholder: {prompt_template_name} - LLM client missing.",
                None,
            )

        try:
            # template_full_path = self.run_specific_template_dir / "llm_prompts" / prompt_template_name
            template_relative_path = (
                Path(self.config.template_subdir)
                / "llm_prompts"
                / prompt_template_name
            )
            template_path_str = str(template_relative_path).replace("\\", "/")
            logger.info(
                f"Loading LLM prompt template from: {template_path_str}"
            )
            template = self.jinja_env.get_template(template_path_str)

            # Add CWE_DETAILS to the prompt context if not already present
            if 'CWE_DETAILS' not in prompt_context_data:
                prompt_context_data['CWE_DETAILS'] = CWE_DETAILS

            rendered_prompt = template.render(prompt_context_data)

            system_prompt = self.SYSTEM_PROMPTS.get(
                task_type, self.SYSTEM_PROMPTS["default"]
            )
            logger.info(
                f"Selected Task Type: {task_type}, using associated system"
                " prompt."
            )
            # Count tokens for prompts (using the method from your class)
            input_tokens_rendered = self._count_tokens(
                text=rendered_prompt, model_name=llm_model
            )
            input_tokens_system = self._count_tokens(
                text=system_prompt, model_name=llm_model
            )
            kwargs_for_llm = {**kwargs, 'task_type': task_type}

            text_response, usage_info_from_api = (
                await self.llm.aget_completion(
                    model=llm_model,
                    prompt=rendered_prompt,
                    system_prompt=system_prompt,
                    temperature=llm_temperature,
                    max_tokens=llm_max_tokens,
                    top_p=llm_top_p,
                    presence_penalty=llm_presence_penalty,
                    frequency_penalty=llm_frequency_penalty,
                    stream=llm_stream,
                    logprobs=llm_logprobs,
                    top_logprobs=llm_top_logprobs,
                    timeout=llm_timeout,
                    response_format=llm_response_format,
                    image_paths=image_paths,
                    **kwargs_for_llm,
                )
            )
            # --- Token Logging Logic (Moved here, always executes) ---
            completion_tokens_final = 0
            total_input_tokens_final = (
                input_tokens_rendered + input_tokens_system
            )  # Our calculation
            status = "Success"

            if usage_info_from_api:  # If API provided usage
                completion_tokens_final = usage_info_from_api.get(
                    "completion_tokens", 0
                )
                # Prioritize API's prompt_tokens if available and makes sense
                if usage_info_from_api.get("prompt_tokens", 0) > 0:
                    total_input_tokens_final = usage_info_from_api.get(
                        "prompt_tokens", total_input_tokens_final
                    )
            elif (
                text_response
                and not text_response.startswith("Error:")
                and not text_response.startswith("Placeholder:")
                and not text_response.startswith("LiteLLM API Error:")
            ):
                # Fallback: estimate completion tokens if API didn't provide and response seems successful
                completion_tokens_final = self._count_tokens(
                    text=text_response, model_name=llm_model
                )
            else:  # Error or placeholder, no meaningful completion tokens
                status = (
                    f"Failed - Response: {text_response[:50]}..."
                    if not text_response.startswith("Placeholder:")
                    else "Skipped - Placeholder"
                )

            pricing_config = self.TOKEN_PRICING.get(
                llm_model, self.TOKEN_PRICING["default_llm_pricing"]
            )
            cost = (total_input_tokens_final / 1000) * pricing_config[
                "input_1k"
            ] + (completion_tokens_final / 1000) * pricing_config["output_1k"]

            usage_entry = {
                "timestamp": request_timestamp,
                "prompt_template_name": prompt_template_name,
                "task_type": task_type,
                "model_name": llm_model,
                "status": status,
                "system_prompt_tokens": input_tokens_system,
                "rendered_prompt_tokens": input_tokens_rendered,
                "total_input_tokens_calculated": (
                    input_tokens_system + input_tokens_rendered
                ),
                "total_input_tokens_api_reported": (
                    usage_info_from_api.get("prompt_tokens")
                    if usage_info_from_api
                    else None
                ),
                "completion_tokens_api_reported": (
                    usage_info_from_api.get("completion_tokens")
                    if usage_info_from_api
                    else None
                ),
                "final_input_tokens_for_cost": total_input_tokens_final,
                "final_completion_tokens_for_cost": completion_tokens_final,
                "cost_estimate": round(cost, 6),
            }
            self.llm_token_usage_log.append(usage_entry)
            if status == "Success":
                logger.info(
                    f"LLM call for '{prompt_template_name}' successful. Input:"
                    f" {total_input_tokens_final}, Output:"
                    f" {completion_tokens_final}, Cost: ${cost:.6f}"
                )
            else:
                logger.warning(
                    f"LLM call for '{prompt_template_name}' resulted in"
                    f" status: {status}. Input(calc):"
                    f" {usage_entry['total_input_tokens_calculated']},"
                    f" Output(est): {completion_tokens_final}, Cost:"
                    f" ${cost:.6f}"
                )

            return text_response  # Return only the text response

        except (
            Exception
        ) as e:  # Catch errors from template rendering, token counting, or unexpected issues
            logger.exception(
                "Outer error in _get_llm_response for template"
                f" {prompt_template_name}: {e}"
            )
            # Log failed usage
            cost_on_error = (
                ((input_tokens_rendered + input_tokens_system) / 1000)
                * self.TOKEN_PRICING.get(
                    llm_model, self.TOKEN_PRICING["default_llm_pricing"]
                )["input_1k"]
                if (input_tokens_rendered + input_tokens_system) > 0
                else 0.0
            )
            usage_entry = {
                "timestamp": request_timestamp,
                "prompt_template_name": prompt_template_name,
                "task_type": task_type,
                "model_name": llm_model,
                "status": f"Error - {type(e).__name__}",
                "system_prompt_tokens": input_tokens_system,
                "rendered_prompt_tokens": input_tokens_rendered,
                "total_input_tokens_calculated": (
                    input_tokens_system + input_tokens_rendered
                ),
                "total_input_tokens_api_reported": None,
                "completion_tokens_api_reported": None,
                "final_input_tokens_for_cost": (
                    input_tokens_system + input_tokens_rendered
                ),
                "final_completion_tokens_for_cost": 0,
                "cost_estimate": round(cost_on_error, 6),
            }
            self.llm_token_usage_log.append(usage_entry)
            return (
                "Error preparing/generating insight for"
                f" {prompt_template_name}. Details: {str(e)}"
            )

    def _generate_appendix_tables(
        self,
        df: pd.DataFrame,
        table_defs: List[Dict[str, str]],
        calculated_metrics: Dict[str, Any],
    ) -> Dict[str, Dict[str, str]]:
        """Generates HTML tables for the appendix based on definitions."""
        logger.info("Generating appendix tables...")
        appendix_tables_html: Dict[str, Dict[str, str]] = {}

        severity_order = ['critical', 'high', 'medium', 'low', 'unknown']
        severity_color_map = {
            'critical': (
                'text-brand-red-700 dark:text-brand-red-500 font-semibold'
            ),
            'high': (
                'text-brand-orange-600 dark:text-brand-orange-400'
                ' font-semibold'
            ),
            'medium': (
                'text-brand-blue-600 dark:text-brand-blue-400 font-semibold'
            ),
            'low': (
                'text-brand-green-600 dark:text-brand-green-400 font-semibold'
            ),
            'unknown': 'text-gray-500 dark:text-gray-400 font-semibold',
        }
        cve_category_map = {
            "denial_of_service": "Denial of Service",
            "privilege_elevation": "Privilege Elevation",
            "remote_code_execution": "Remote Code Execution",
            "disclosure": "Information Disclosure",
            "spoofing": "Spoofing",
            "tampering": "Tampering",
            "feature_bypass": "Security Feature Bypass",
            "none": "Unknown",
            "unknown": "Unknown",
        }
        # Initial check for completely empty inputs
        if df.empty and not calculated_metrics:
            logger.warning(
                "Main DataFrame and calculated metrics are empty, cannot"
                " generate appendix tables."
            )
            # Potentially return empty or minimal HTML for all table_defs
            for table_def in table_defs:
                table_id = table_def.get('id')
                table_title = table_def.get(
                    'caption', f"Data Table: {table_id}"
                )
                appendix_tables_html[table_id] = {
                    'title': table_title,
                    'html': (
                        f"<p class='text-sm text-gray-600 dark:text-gray-400"
                        f" mt-2'>No data available to generate this table.</p>"
                    ),
                }
            return appendix_tables_html

        for table_def in table_defs:
            table_id = table_def.get('id')
            table_title = table_def.get('caption', f"Data Table: {table_id}")
            cols_to_display_map = table_def.get('columns', {})
            df_source_metric = table_def.get('df_source_metric')
            row_click_config = table_def.get('row_click_config', {})

            table_html_content = ""
            current_df_for_processing: Optional[pd.DataFrame] = None

            # --- Try to get DF from calculated_metrics first ---
            if df_source_metric and calculated_metrics:
                metric_data = calculated_metrics.get(df_source_metric)
                if isinstance(metric_data, pd.DataFrame):
                    current_df_for_processing = metric_data.copy()
                elif (
                    isinstance(metric_data, dict)
                    and 'dataframe' in metric_data
                    and isinstance(metric_data['dataframe'], pd.DataFrame)
                ):
                    current_df_for_processing = metric_data['dataframe'].copy()
                else:
                    logger.warning(
                        f"DataFrame source metric '{df_source_metric}' for"
                        f" table '{table_id}' not found or not a DataFrame."
                    )

            # --- Fallback to main df if no specific source or if source was invalid ---
            if current_df_for_processing is None:
                if not df.empty:
                    current_df_for_processing = df.copy()
                else:  # If main df is also empty and no specific source found
                    logger.warning(
                        "No valid DataFrame source found for table"
                        f" '{table_id}' and main df is empty."
                    )
                    appendix_tables_html[table_id] = {
                        'title': table_title,
                        'html': (
                            "<p class='text-sm text-gray-600"
                            " dark:text-gray-400 mt-2'>No data available for"
                            f" table '{table_id}'.</p>"
                        ),
                    }
                    continue  # Next table_def

            # ========================================================================
            # START: LOGIC FOR STANDARD TABLES (NOT CVSS OUTLIERS OR ALL METRICS)
            # This is the branch you've primarily been updating.
            # ========================================================================
            if table_id not in [
                "appendix_outlier_list",
                "appendix_all_metrics",
            ]:
                if (
                    current_df_for_processing is None
                    or current_df_for_processing.empty
                ):
                    table_html_content = (
                        "<p class='text-sm text-gray-600 dark:text-gray-400"
                        f" mt-2'>No data available for table '{table_id}'.</p>"
                    )
                else:
                    cols_in_df = [
                        col
                        for col in cols_to_display_map.keys()
                        if col in current_df_for_processing.columns
                    ]
                    if not cols_in_df:
                        table_html_content = (
                            "<p class='text-sm text-gray-600"
                            " dark:text-gray-400 mt-2'>Error: No relevant"
                            f" columns found for table '{table_id}'.</p>"
                        )
                    else:
                        # --- Data Preparation (as you've implemented) ---
                        df_to_render = current_df_for_processing[
                            cols_in_df
                        ].copy()

                        if "cve_category" in df_to_render.columns:
                            df_to_render["cve_category"] = (
                                df_to_render["cve_category"]
                                .map(
                                    lambda x: cve_category_map.get(
                                        str(x).lower(), str(x)
                                    )
                                )
                                .fillna("Unknown")
                            )

                        if "severity_type" in df_to_render.columns:
                            df_to_render["severity_type_normalized"] = (
                                df_to_render["severity_type"]
                                .astype(str)
                                .str.lower()
                                .fillna('unknown')
                            )
                            df_to_render["severity_type"] = pd.Categorical(
                                df_to_render["severity_type_normalized"],
                                categories=severity_order,
                                ordered=True,
                            )

                        if 'published' in df_to_render.columns:
                            df_to_render['published'] = pd.to_datetime(
                                df_to_render['published'], errors='coerce'
                            )
                        if 'cvss_score' in df_to_render.columns:
                            df_to_render['cvss_score'] = pd.to_numeric(
                                df_to_render['cvss_score'], errors='coerce'
                            )

                        sort_by_columns = []
                        ascending_order = []
                        if "cve_category" in df_to_render.columns:
                            sort_by_columns.append("cve_category")
                            ascending_order.append(True)
                        if "severity_type" in df_to_render.columns:
                            sort_by_columns.append("severity_type")
                            ascending_order.append(True)
                        if "cvss_score" in df_to_render.columns:
                            sort_by_columns.append("cvss_score")
                            ascending_order.append(False)
                        if "published" in df_to_render.columns:
                            sort_by_columns.append("published")
                            ascending_order.append(False)

                        if sort_by_columns:
                            df_to_render = df_to_render.sort_values(
                                by=sort_by_columns, ascending=ascending_order
                            )

                        if 'published' in df_to_render.columns:
                            df_to_render['published'] = df_to_render[
                                'published'
                            ].dt.strftime('%Y-%m-%d')
                        if 'cvss_score' in df_to_render.columns:
                            df_to_render['cvss_score'] = df_to_render[
                                'cvss_score'
                            ].apply(lambda x: f"{x:.1f}" if pd.notna(x) else x)
                        # --- End Data Preparation ---

                        # --- HTML Generation for Standard Tables ---
                        html_parts_list = [
                            "<div class='overflow-x-auto"
                            " report-table-wrapper'>"
                        ]
                        html_parts_list.append(
                            "<table class='min-w-full table-fixed divide-y"
                            " divide-gray-200 dark:divide-gray-700'>"
                        )
                        html_parts_list.append(
                            "<thead class='bg-gray-50 dark:bg-gray-800'><tr>"
                        )
                        th_base_classes = (
                            "px-2 py-1.5 text-left text-sm font-semibold"
                            " text-gray-500 dark:text-gray-300 title-case"
                            " tracking-wider"
                        )
                        additional_th_classes = ""
                        for col_key in cols_in_df:
                            display_name = cols_to_display_map[col_key]
                            css_col_key = (
                                col_key.lower()
                                .replace('_type', '')
                                .replace('_', '-')
                            )
                            if col_key == 'cvss_score':
                                additional_th_classes = "w-8"
                            elif col_key.lower() == "title":
                                additional_th_classes = "w-full"
                            else:
                                additional_th_classes = ""
                            html_parts_list.append(
                                f"<th class='col-{css_col_key} {th_base_classes} {additional_th_classes}'>{display_name}</th>"
                            )
                        html_parts_list.append("</tr></thead>")
                        html_parts_list.append(
                            "<tbody class='bg-white dark:bg-gray-900 divide-y"
                            " divide-gray-200 dark:divide-gray-700'>"
                        )
                        for _, row_data in df_to_render.iterrows():
                            html_parts_list.append(
                                "<tr class='hover:bg-gray-100"
                                " dark:hover:bg-gray-700 transition-colors"
                                " duration-150 ease-in-out'>"
                            )
                            for col_key in cols_in_df:
                                cell_value = row_data.get(col_key)
                                display_value = (
                                    str(cell_value)
                                    if pd.notna(cell_value)
                                    else "N/A"
                                )

                                td_class_list = [
                                    (
                                        f"col-{col_key.lower().replace('_type', '').replace('_', '-')}"
                                    ),
                                    "px-2 py-1.5",
                                    "text-xs",
                                ]
                                if col_key == 'cve_id':
                                    td_class_list.append(
                                        "text-brand-pink-600"
                                        " dark:text-brand-pink-400"
                                        " font-semibold whitespace-nowrap"
                                    )
                                    if row_click_config.get(
                                        'enabled'
                                    ) and col_key == row_click_config.get(
                                        'id_column'
                                    ):
                                        url_template = row_click_config.get(
                                            'url_template'
                                        )
                                        id_val_for_url = row_data.get(
                                            row_click_config['id_column']
                                        )
                                        if url_template and pd.notna(
                                            id_val_for_url
                                        ):
                                            link_url = url_template.replace(
                                                f"{{{row_click_config['id_column']}}}",
                                                str(id_val_for_url),
                                            )
                                            display_value = (
                                                f"<a href='{link_url}'"
                                                " target='_blank'"
                                                " rel='noopener noreferrer'"
                                                " class='text-brand-pink-600"
                                                " dark:text-brand-pink-400"
                                                f" font-medium'>{display_value}</a>"
                                            )
                                elif (
                                    col_key == "severity_type"
                                ):  # Key in cols_to_display_map
                                    # Use 'severity_type_normalized' which exists on row_data if severity_type was processed
                                    severity_val_for_color = row_data.get(
                                        "severity_type_normalized", "unknown"
                                    )
                                    color_class = severity_color_map.get(
                                        severity_val_for_color,
                                        severity_color_map['unknown'],
                                    )
                                    td_class_list.append(color_class)
                                    # Display original severity value if it exists, otherwise the processed one
                                    original_severity_display = row_data.get(
                                        "severity_type"
                                    )  # Column is set to categorical above
                                    display_value = (
                                        str(original_severity_display)
                                        if pd.notna(original_severity_display)
                                        else "N/A"
                                    )
                                elif col_key == "title":
                                    td_class_list.extend(
                                        ["truncate", "max-w-sm"]
                                    )
                                else:
                                    td_class_list.append(
                                        "text-gray-700 dark:text-gray-300"
                                    )

                                html_parts_list.append(
                                    f"<td class='{' '.join(td_class_list)}'>{display_value}</td>"
                                )
                            html_parts_list.append("</tr>")
                        html_parts_list.append("</tbody></table></div>")
                        table_html_content = "".join(html_parts_list)

            # ========================================================================
            # START: LOGIC FOR CVSS OUTLIERS TABLE
            # ========================================================================
            elif table_id == "appendix_outlier_list":
                logger.debug("Generating 'outlier_list' appendix table...")
                # Assuming calculated_metrics['cvss_score_outliers']['dataframe'] holds the outliers
                if df.empty:
                    table_html_content = (
                        "<p class='text-sm text-gray-600 dark:text-gray-400"
                        " mt-2'>Main CVE data is empty, cannot calculate"
                        " outliers.</p>"
                    )
                    # Append to all_tables_html and continue to next table_def
                    appendix_tables_html[table_id] = {
                        'title': table_def.get(
                            'title', table_id.replace("_", " ").title()
                        ),
                        'html': table_html_content,
                    }
                    continue

                iqr_metrics_by_cat = calculated_metrics.get(
                    'cvss_iqr_by_category', {}
                )
                if not iqr_metrics_by_cat:
                    table_html_content = (
                        "<p class='text-sm text-gray-600 dark:text-gray-400"
                        " mt-2'>CVSS IQR metrics by category not"
                        " available.</p>"
                    )
                    appendix_tables_html[table_id] = {
                        'title': table_def.get(
                            'title', table_id.replace("_", " ").title()
                        ),
                        'html': table_html_content,
                    }
                    continue

                # Ensure necessary columns exist and are of correct type in the main df
                if (
                    'cvss_score' not in df.columns
                    or 'cve_category' not in df.columns
                ):
                    table_html_content = (
                        "<p class='text-sm text-gray-600 dark:text-gray-400"
                        " mt-2'>Required columns ('cvss_score',"
                        " 'cve_category') not in source data.</p>"
                    )
                    appendix_tables_html[table_id] = {
                        'title': table_def.get(
                            'title', table_id.replace("_", " ").title()
                        ),
                        'html': table_html_content,
                    }
                    continue

                temp_df = df.copy()  # Work on a copy
                temp_df['cvss_score'] = pd.to_numeric(
                    temp_df['cvss_score'], errors='coerce'
                )

                # Apply cve_category_map similar to other tables if it's not already done upstream
                # Assuming cve_category_map is defined earlier in the function
                if (
                    "cve_category" in temp_df.columns
                    and 'cve_category_map' in locals()
                ):
                    temp_df["cve_category_mapped"] = (
                        temp_df["cve_category"]
                        .astype(str)
                        .str.lower()
                        .map(
                            lambda x: cve_category_map.get(
                                x, x
                            )  # Use mapped value or original if not in map
                        )
                        .fillna("Unknown")
                    )
                else:
                    # If no map or column, use original or a placeholder
                    temp_df["cve_category_mapped"] = (
                        temp_df["cve_category"].astype(str).fillna("Unknown")
                    )

                outlier_rows = []
                for category_key, metrics in iqr_metrics_by_cat.items():
                    # Original calculated fences from metrics
                    calc_lower_fence = metrics.get('Lower_Fence')
                    calc_upper_fence = metrics.get('Upper_Fence')

                    # Initialize effective fences to be used for outlier detection
                    effective_lower_fence = None
                    effective_upper_fence = None

                    # Apply clamping if the calculated fence is a valid number
                    if pd.notna(calc_lower_fence):
                        effective_lower_fence = max(0.0, calc_lower_fence)

                    if pd.notna(calc_upper_fence):
                        effective_upper_fence = min(10.0, calc_upper_fence)

                    # If, after potential clamping, both fences are still None
                    # (e.g., category had no data for IQR, so calc_lower_fence and calc_upper_fence were NaN),
                    # then skip this category.
                    if (
                        effective_lower_fence is None
                        and effective_upper_fence is None
                    ):
                        continue

                    category_cves = temp_df[
                        temp_df['cve_category'] == category_key
                    ]

                    for index, row in category_cves.iterrows():
                        cvss = row['cvss_score']
                        row_dict = (
                            row.to_dict()
                        )  # Convert row to dict to append new key

                        if pd.notna(cvss):
                            # Default to False if a fence is not defined
                            is_low_outlier = False
                            if effective_lower_fence is not None:
                                is_low_outlier = cvss < effective_lower_fence

                            is_high_outlier = False
                            if effective_upper_fence is not None:
                                is_high_outlier = cvss > effective_upper_fence

                            if is_low_outlier:
                                row_dict['outlier_type'] = "Low Outlier"
                                outlier_rows.append(row_dict)
                            elif is_high_outlier:
                                row_dict['outlier_type'] = "High Outlier"
                                outlier_rows.append(row_dict)

                if not outlier_rows:
                    table_html_content = (
                        "<p class='text-sm text-gray-600 dark:text-gray-400"
                        " mt-2'>No CVSS score outliers found based on IQR"
                        " criteria.</p>"
                    )
                else:
                    final_outlier_df = pd.DataFrame(outlier_rows)
                    cols_to_display_map_outliers = table_def.get(
                        'columns',
                        {  # Default to empty if not defined in report_structure.yml
                            'cve_id': 'CVE ID',
                            'title': 'Title',
                            'outlier_type': 'Outlier Type',
                            'cvss_score': 'CVSS Score',
                            'cve_category': 'CVE Category',
                            'published': 'Published',
                        },
                    )
                    cols_to_use_outliers = [
                        col
                        for col in cols_to_display_map_outliers.keys()
                        if col in final_outlier_df.columns
                    ]

                    if not cols_to_use_outliers:
                        table_html_content = (
                            "<p class='text-sm text-gray-600"
                            " dark:text-gray-400 mt-2'>No relevant columns for"
                            " CVSS outliers table.</p>"
                        )
                    else:
                        df_to_render = final_outlier_df[
                            cols_to_use_outliers
                        ].copy()

                        # --- Data Preparation for Outliers ---
                        if "cve_category" in df_to_render.columns:
                            df_to_render["cve_category"] = (
                                df_to_render["cve_category"]
                                .map(
                                    lambda x: cve_category_map.get(
                                        str(x).lower(), str(x)
                                    )
                                )
                                .fillna("Unknown")
                            )

                        if "severity_type" in df_to_render.columns:
                            df_to_render["severity_type_normalized"] = (
                                df_to_render["severity_type"]
                                .astype(str)
                                .str.lower()
                                .fillna('unknown')
                            )
                            df_to_render["severity_type"] = pd.Categorical(
                                df_to_render["severity_type_normalized"],
                                categories=severity_order,
                                ordered=True,
                            )

                        if 'published' in df_to_render.columns:
                            df_to_render['published'] = pd.to_datetime(
                                df_to_render['published'], errors='coerce'
                            )
                        if 'cvss_score' in df_to_render.columns:
                            df_to_render['cvss_score'] = pd.to_numeric(
                                df_to_render['cvss_score'], errors='coerce'
                            )

                        sort_by_columns_ot = []
                        ascending_order_ot = []
                        if "cve_category" in df_to_render.columns:
                            sort_by_columns_ot.append("cve_category")
                            ascending_order_ot.append(True)
                        if "severity_type" in df_to_render.columns:
                            sort_by_columns_ot.append("severity_type")
                            ascending_order_ot.append(True)
                        if "cvss_score" in df_to_render.columns:
                            sort_by_columns_ot.append("cvss_score")
                            ascending_order_ot.append(False)
                        if "outlier_type" in df_to_render.columns:
                            sort_by_columns_ot.append("outlier_type")
                            ascending_order_ot.append(True)
                        if "published" in df_to_render.columns:
                            sort_by_columns_ot.append("published")
                            ascending_order_ot.append(False)

                        if sort_by_columns_ot:
                            df_to_render = df_to_render.sort_values(
                                by=sort_by_columns_ot,
                                ascending=ascending_order_ot,
                            )

                        if 'published' in df_to_render.columns:
                            df_to_render['published'] = df_to_render[
                                'published'
                            ].dt.strftime('%Y-%m-%d')
                        if 'cvss_score' in df_to_render.columns:
                            df_to_render['cvss_score'] = df_to_render[
                                'cvss_score'
                            ].apply(lambda x: f"{x:.1f}" if pd.notna(x) else x)
                        # --- End Data Preparation for Outliers ---

                        # --- HTML Generation for Outliers ---
                        html_parts_list = [
                            "<div class='overflow-x-auto"
                            " report-table-wrapper'>"
                        ]
                        html_parts_list.append(
                            "<table class='min-w-full table-fixed divide-y"
                            " divide-gray-200 dark:divide-gray-700'>"
                        )
                        html_parts_list.append(
                            "<thead class='bg-gray-50 dark:bg-gray-800'><tr>"
                        )
                        for col_key in cols_to_use_outliers:
                            display_name = cols_to_display_map_outliers[
                                col_key
                            ]
                            css_col_key = (
                                col_key.lower()
                                .replace('_type', '')
                                .replace('_', '-')
                            )
                            th_base_classes = (
                                "px-2 py-1.5 text-left text-sm font-semibold"
                                " text-gray-500 dark:text-gray-300 title-case"
                                " tracking-wider"
                            )
                            additional_th_classes = ""
                            if col_key == 'cvss_score':
                                additional_th_classes = "w-8"
                            elif col_key.lower() == "title":
                                additional_th_classes = "w-full"
                            html_parts_list.append(
                                f"<th class='col-{css_col_key} {th_base_classes} {additional_th_classes}'>{display_name}</th>"
                            )
                        html_parts_list.append("</tr></thead>")
                        html_parts_list.append(
                            "<tbody class='bg-white dark:bg-gray-900 divide-y"
                            " divide-gray-200 dark:divide-gray-700'>"
                        )
                        for _, row_data in df_to_render.iterrows():
                            html_parts_list.append(
                                "<tr class='hover:bg-gray-100"
                                " dark:hover:bg-gray-700 transition-colors"
                                " duration-150 ease-in-out'>"
                            )
                            for col_key in cols_to_use_outliers:
                                cell_value = row_data.get(col_key)
                                display_value = (
                                    str(cell_value)
                                    if pd.notna(cell_value)
                                    else "N/A"
                                )
                                # Start with base classes
                                td_class_list = [
                                    (
                                        f"col-{col_key.lower().replace('_type', '').replace('_', '-')}"
                                    ),
                                    "px-2 py-1.5",
                                    "text-xs",
                                ]
                                specific_color_applied = False
                                if col_key.lower() == "title":
                                    td_class_list.extend([
                                        "text-report-text",
                                        "dark:text-report-text-dark",
                                        "truncate",
                                        "max-w-sm",
                                    ])
                                    specific_color_applied = True
                                elif (
                                    col_key == "severity_type"
                                    and "severity_type_normalized" in row_data
                                ):
                                    severity_val_for_color = row_data.get(
                                        "severity_type_normalized", "unknown"
                                    )
                                    color_class = severity_color_map.get(
                                        severity_val_for_color,
                                        severity_color_map['unknown'],
                                    )
                                    td_class_list.append(
                                        color_class
                                    )  # This appends "text-brand-red-700 dark:text-brand-red-500" etc.

                                    # Also append font-semibold for severity text as seen in the screenshot
                                    td_class_list.append("font-semibold")

                                    original_severity_display = row_data.get(
                                        "severity_type"
                                    )
                                    display_value = (
                                        str(original_severity_display)
                                        if pd.notna(original_severity_display)
                                        else "N/A"
                                    )
                                    specific_color_applied = True

                                elif col_key == 'outlier_type':
                                    outlier_status = row_data.get(
                                        'outlier_type', ''
                                    )
                                    if outlier_status == "Low Outlier":
                                        td_class_list.append(
                                            "text-green-700"
                                            " dark:text-green-400 bg-green-50"
                                            " dark:bg-green-800/30 font-medium"
                                            " px-2 py-0.5 rounded-md"
                                        )
                                        specific_color_applied = True
                                    elif outlier_status == "High Outlier":
                                        # Styling for 'bad' outliers (e.g., CVSS score above upper fence)
                                        td_class_list.append(
                                            "text-red-700 dark:text-red-400"
                                            " bg-red-50 dark:bg-red-800/30"
                                            " font-medium px-2 py-0.5"
                                            " rounded-md"
                                        )
                                        specific_color_applied = True
                                    # The display_value itself (e.g., "Low Outlier") is already set correctly
                                elif col_key == "cvss_score":
                                    td_class_list.append("w-8")

                                if row_click_config.get(
                                    'enabled', True
                                ) and col_key == row_click_config.get(
                                    'id_column', 'cve_id'
                                ):
                                    td_class_list.append("whitespace-nowrap")
                                    url_template = row_click_config.get(
                                        'url_template',
                                        "https://www.cve.org/CVERecord?id={cve_id}",
                                    )  # Fallback URL template
                                    id_val_for_url = row_data.get(
                                        col_key
                                    )  # use col_key directly
                                    if pd.notna(id_val_for_url):
                                        link_url = url_template.replace(
                                            f"{{{col_key}}}",
                                            str(id_val_for_url),
                                        )  # use col_key
                                        display_value = (
                                            f"<a href='{link_url}'"
                                            " target='_blank' rel='noopener"
                                            " noreferrer'"
                                            " class='text-brand-pink-600"
                                            " dark:text-brand-pink-400"
                                            f" font-medium'>{display_value}</a>"
                                        )
                                if not specific_color_applied:
                                    td_class_list.extend(
                                        ["text-gray-700", "dark:text-gray-300"]
                                    )

                                html_parts_list.append(
                                    f"<td class='{' '.join(td_class_list)}'>{display_value}</td>"
                                )
                            html_parts_list.append("</tr>")
                        html_parts_list.append("</tbody></table></div>")
                        table_html_content = "".join(html_parts_list)

            # ========================================================================
            # START: LOGIC FOR ALL METRICS TABLE
            # ========================================================================
            elif table_id == "appendix_all_metrics":
                logger.debug("Generating 'all_metrics' appendix table...")
                all_metrics_records = []
                stat_card_labels = getattr(self, 'STAT_CARD_LABELS', {})
                row_counter = 0

                # Helper function to process metric values for data structure
                def _process_metric_display_value(
                    value,
                    original_metric_key_for_context: Optional[str] = None,
                    is_sub_item: bool = False,
                ):
                    if isinstance(value, float):
                        return round(value, 2)
                    if value is None:
                        return "N/A"
                    if isinstance(value, bool):
                        return "Yes" if value else "No"

                    # Specific handling for 'top_3_affected_builds' when value is a list
                    if (
                        original_metric_key_for_context
                        == "top_3_affected_builds"
                        and isinstance(value, list)
                    ):
                        if not value:  # Handle empty list case
                            return "N/A"

                        html_parts = []
                        for index, item_dict in enumerate(value):
                            if isinstance(item_dict, dict):
                                build_version = html.escape(
                                    str(item_dict.get("build", "N/A"))
                                )
                                count = html.escape(
                                    str(item_dict.get("count", "N/A"))
                                )

                                # "Build #{index + 1}" label
                                html_parts.append(
                                    '<div class="font-semibold text-sm mt-3'
                                    f' mb-1">Build #{index + 1}</div>'
                                )

                                # Inner UL for build and count details
                                inner_list_items = [
                                    (
                                        '<li class="text-xs"><strong'
                                        ' class="font-medium">Build'
                                        ' Version:</strong>'
                                        f' {build_version}</li>'
                                    ),
                                    (
                                        '<li class="text-xs"><strong'
                                        ' class="font-medium">Count:</strong>'
                                        f' {count}</li>'
                                    ),
                                ]
                                inner_list_html = "".join(inner_list_items)

                                html_parts.append(
                                    '<ul class="list-disc list-inside pl-5'
                                    f' text-xs mb-2">{inner_list_html}</ul>'
                                )
                            else:
                                # Fallback for unexpected item type in the list
                                html_parts.append(
                                    '<div class="font-semibold text-sm mt-3'
                                    f' mb-1">Item #{index + 1}</div>'
                                )
                                html_parts.append(
                                    '<ul class="list-disc list-inside pl-5'
                                    ' text-xs'
                                    f' mb-2"><li>{html.escape(str(item_dict))}</li></ul>'
                                )

                        return "".join(html_parts)

                    # General handling for lists (if not 'top_3_affected_builds')
                    if isinstance(value, list):
                        return [
                            _process_metric_display_value(
                                item, original_metric_key_for_context, True
                            )
                            for item in value
                        ]

                    # General handling for dictionaries
                    if isinstance(value, dict):
                        # Special handling for items within cve_category_distribution list
                        if (
                            original_metric_key_for_context
                            == "cve_category_distribution"
                            and is_sub_item
                        ):
                            return {
                                k: _process_metric_display_value(
                                    v, original_metric_key_for_context, True
                                )
                                for k, v in value.items()
                                if k not in ["name", "id_for_color"]
                            }
                        # Special handling for sub-items of cvss_iqr_by_category
                        elif (
                            original_metric_key_for_context
                            == "cvss_iqr_by_category"
                            and is_sub_item
                        ):
                            return {
                                k: _process_metric_display_value(
                                    v, original_metric_key_for_context, True
                                )
                                for k, v in value.items()
                                if k
                                not in [
                                    "id_for_color"
                                ]  # Exclude id_for_color here
                            }
                        # For other dictionaries (general case)
                        return {
                            k: _process_metric_display_value(
                                v, original_metric_key_for_context, True
                            )
                            for k, v in value.items()
                        }

                    # Fallback for any other data type
                    return str(value)

                # Helper function to format complex values (dicts/lists) into HTML
                def _format_complex_value_to_html(value, level=0) -> str:
                    if isinstance(value, dict):
                        if not value:
                            return "{ }"
                        items_html = ""
                        for k, v_item in value.items():
                            formatted_v = _format_complex_value_to_html(
                                v_item, level + 1
                            )
                            items_html += (
                                "<li><strong"
                                f" class='font-semibold'>{k}:</strong>"
                                f" {formatted_v}</li>"
                            )
                        return (
                            "<ul class='list-none pl-4"
                            f" text-left'>{items_html}</ul>"
                            if level == 0
                            else (
                                f"<ul class='list-none pl-4'>{items_html}</ul>"
                            )
                        )
                    elif isinstance(value, list):
                        if not value:
                            return "[ ]"
                        items_html = "".join([
                            f"<li>{_format_complex_value_to_html(item, level + 1)}</li>"
                            for item in value
                        ])
                        return (
                            "<ul class='list-disc list-inside pl-4"
                            f" text-left'>{items_html}</ul>"
                            if level == 0
                            else (
                                "<ul class='list-disc list-inside"
                                f" pl-4'>{items_html}</ul>"
                            )
                        )
                    else:
                        return str(value)

                if not calculated_metrics:
                    table_html_content = (
                        "<p class='text-sm text-gray-600 dark:text-gray-400"
                        " mt-2'>No calculated metrics provided to display.</p>"
                    )
                else:
                    for (
                        metric_key,
                        metric_value_original,
                    ) in calculated_metrics.items():
                        if (
                            "delta" in metric_key.lower()
                            or "qoq_trend_direction" in metric_key.lower()
                            or metric_key.endswith("_description")
                            or metric_key.endswith("_context")
                        ):
                            continue

                        main_metric_label = stat_card_labels.get(
                            metric_key, metric_key.replace("_", " ").title()
                        )

                        processed_value_structure = (
                            _process_metric_display_value(
                                metric_value_original, metric_key
                            )
                        )

                        if (
                            metric_key == "cve_category_distribution"
                            and isinstance(metric_value_original, list)
                        ):
                            if not metric_value_original:
                                row_counter += 1
                                all_metrics_records.append({
                                    'row_number': row_counter,
                                    'metric_name': main_metric_label,
                                    'metric_value': "No data available",
                                })
                            else:
                                for index, original_item_dict in enumerate(
                                    metric_value_original
                                ):
                                    row_counter += 1
                                    item_name_for_label = (
                                        original_item_dict.get(
                                            "name", f"Item {index + 1}"
                                        )
                                    )
                                    processed_item_value = (
                                        "Error processing item"
                                    )
                                    if isinstance(
                                        processed_value_structure, list
                                    ) and index < len(
                                        processed_value_structure
                                    ):
                                        processed_item_value = (
                                            processed_value_structure[index]
                                        )
                                    all_metrics_records.append({
                                        'row_number': row_counter,
                                        'metric_name': (
                                            f"{main_metric_label}:"
                                            f" {item_name_for_label}"
                                        ),
                                        'metric_value': processed_item_value,
                                    })
                        elif (
                            metric_key == "cvss_iqr_by_category"
                            and isinstance(processed_value_structure, dict)
                        ):
                            if not processed_value_structure:
                                row_counter += 1
                                all_metrics_records.append({
                                    'row_number': row_counter,
                                    'metric_name': main_metric_label,
                                    'metric_value': "No data available",
                                })
                            else:
                                for (
                                    sub_cat_key,
                                    sub_cat_value_processed,
                                ) in processed_value_structure.items():
                                    row_counter += 1
                                    sub_cat_display_name = sub_cat_key.replace(
                                        "_", " "
                                    ).title()
                                    all_metrics_records.append({
                                        'row_number': row_counter,
                                        'metric_name': (
                                            f"{main_metric_label}:"
                                            f" {sub_cat_display_name}"
                                        ),
                                        'metric_value': (
                                            sub_cat_value_processed
                                        ),
                                    })
                        else:
                            row_counter += 1
                            all_metrics_records.append({
                                'row_number': row_counter,
                                'metric_name': main_metric_label,
                                'metric_value': processed_value_structure,
                            })

                    if not all_metrics_records:
                        table_html_content = (
                            "<p class='text-sm text-gray-600"
                            " dark:text-gray-400 mt-2'>No metrics to display"
                            " after filtering.</p>"
                        )
                    else:
                        cols_to_display_map_metrics = {
                            'row_number': '#',
                            'metric_name': 'Metric Name',
                            'metric_value': 'Value',
                        }

                        html_parts_list = [
                            "<div class='overflow-x-auto"
                            " report-table-wrapper'>"
                        ]
                        html_parts_list.append(
                            "<table class='min-w-full divide-y divide-gray-200"
                            " dark:divide-gray-700'>"
                        )
                        html_parts_list.append(
                            "<thead class='bg-gray-50 dark:bg-gray-800'><tr>"
                        )
                        for col_key_map in cols_to_display_map_metrics.keys():
                            display_name = cols_to_display_map_metrics[
                                col_key_map
                            ]
                            css_col_key = col_key_map.lower().replace('_', '-')
                            width_class = ""
                            if col_key_map == 'row_number':
                                width_class = "w-12"
                            elif col_key_map == 'metric_name':
                                width_class = "w-1/3"
                            html_parts_list.append(
                                f"<th class='col-{css_col_key} {width_class} px-2"
                                " py-1.5 text-left text-sm font-semibold"
                                " text-gray-500 dark:text-gray-300 title-case"
                                f" tracking-wider'>{display_name}</th>"
                            )
                        html_parts_list.append("</tr></thead>")
                        html_parts_list.append(
                            "<tbody class='bg-white dark:bg-gray-900 divide-y"
                            " divide-gray-200 dark:divide-gray-700'>"
                        )

                        for record in all_metrics_records:
                            html_parts_list.append(
                                "<tr class='hover:bg-gray-100"
                                " dark:hover:bg-gray-700 transition-colors"
                                " duration-150 ease-in-out'>"
                            )
                            for (
                                col_key_record
                            ) in cols_to_display_map_metrics.keys():
                                cell_value = record.get(col_key_record)
                                display_value_html: str
                                if col_key_record == 'metric_value' and (
                                    isinstance(cell_value, dict)
                                    or isinstance(cell_value, list)
                                ):
                                    display_value_html = (
                                        _format_complex_value_to_html(
                                            cell_value
                                        )
                                    )
                                else:
                                    display_value_html = (
                                        str(cell_value)
                                        if pd.notna(cell_value)
                                        else "N/A"
                                    )

                                td_class_list = [
                                    (
                                        f"col-{col_key_record.lower().replace('_', '-')}"
                                    ),
                                    "px-2 py-1.5",
                                    "text-xs",
                                    "text-gray-700 dark:text-gray-300",
                                    "align-top",
                                ]
                                if col_key_record == 'metric_name':
                                    td_class_list.append("font-semibold")
                                if col_key_record == 'metric_value':
                                    td_class_list.append("whitespace-normal")
                                else:
                                    td_class_list.append("break-words")
                                html_parts_list.append(
                                    f"<td class='{' '.join(td_class_list)}'>{display_value_html}</td>"
                                )
                            html_parts_list.append("</tr>")

                        html_parts_list.append("</tbody></table></div>")
                        table_html_content = "".join(html_parts_list)

            # Always prepend the table title (h4) to the generated content
            appendix_tables_html[table_id] = {
                'title': table_title,
                'html': table_html_content,
            }

        return appendix_tables_html

    def _place_static_assets(
        self, css_output_dir: Path, js_output_dir: Path, config: ReportConfig
    ) -> tuple[Optional[Path], Optional[Path]]:
        """
        Copies the Tailwind-generated CSS file from the static directory to the report's output directory.
        The CSS file is copied to make the report self-contained.

        Args:
            css_output_dir: Path to the report's CSS output directory
            config: ReportConfig instance containing output paths

        Returns:
            tuple[Optional[Path], Optional[Path]]: Tuple of (css_path, js_path), where css_path is the
            destination path of the copied CSS file if successful, None otherwise
        """
        css_dest_path = None
        js_dest_path = None
        # Define source and destination paths
        source_css = (
            APP_DIR / "static" / "css" / "QuarterlyReportStylesheet.css"
        )
        dest_css = css_output_dir / config.output_css_filename.lstrip('/')
        source_js = APP_DIR / "static" / "js" / "QuarterlyReport.js"
        dest_js = js_output_dir / config.output_js_filename.lstrip('/')
        try:
            if source_css.is_file():
                # Ensure the destination directory exists
                dest_css.parent.mkdir(parents=True, exist_ok=True)

                # Copy the CSS file
                shutil.copy2(source_css, dest_css)
                css_dest_path = dest_css
                logger.debug(f"CSS file successfully copied to: {dest_css}")
            else:
                logger.warning(
                    "ERROR: Tailwind CSS file not found at expected location:"
                    f" {source_css}"
                )
                logger.warning(
                    "Please ensure the Tailwind build process has completed"
                    " successfully."
                )

        except Exception as e:
            logger.warning(f"Error copying CSS file: {e}")
            logger.warning(f"Source: {source_css}")
            logger.warning(f"Destination: {dest_css}")

        if source_js.is_file():
            # Ensure the destination directory exists
            dest_js.parent.mkdir(parents=True, exist_ok=True)

            # Copy the JS file
            shutil.copy2(source_js, dest_js)
            js_dest_path = dest_js
            logger.debug(f"JS file successfully copied to: {dest_js}")
        else:
            logger.warning(
                f"ERROR: JS file not found at expected location: {source_js}"
            )
            logger.warning(
                "Please ensure the JS file is present in the expected"
                " location."
            )

        return css_dest_path, js_dest_path

    async def generate_report(
        self,
        report_data: pd.DataFrame,
        config: ReportConfig,
        start_date: datetime.date,
        end_date: datetime.date,
        metrics_service: DuckDBMetricsService,
    ) -> GeneratedReportAssets:
        """
        Orchestrates the report generation process using provided data.
        """
        start_time = datetime.now()
        logger.info(
            "Starting report generation:"
            f" {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logger.info(
            f"Report Name: {config.report_name}, Period:"
            f" {start_date.strftime('%Y-%m-%d')} to"
            f" {end_date.strftime('%Y-%m-%d')}"
        )
        self.config = config
        self.start_date = start_date
        self.end_date = end_date
        start_str = start_date.strftime("%b_%Y").lower()
        end_str = end_date.strftime("%b_%Y").lower()
        self.run_identifier = f"{config.report_name}_{start_str}_{end_str}"
        base_filename = f"{self.run_identifier}"
        dynamic_html_filename = f"{base_filename}.html"
        dynamic_md_filename = f"{base_filename}.md"

        # --- 1. Determine Paths & Create Directories ---

        report_base_dir = config.get_report_base_dir(self.reports_base_dir)
        run_specific_output_dir = report_base_dir / self.run_identifier
        html_output_dir = run_specific_output_dir / config.output_html_subdir
        data_output_dir = run_specific_output_dir / config.output_data_subdir
        css_output_dir = run_specific_output_dir / config.output_css_subdir
        js_output_dir = run_specific_output_dir / config.output_js_subdir
        markdown_output_dir = (
            run_specific_output_dir / config.output_markdown_subdir
        )
        image_output_dir = run_specific_output_dir / config.output_image_subdir

        final_html_path = html_output_dir / dynamic_html_filename
        final_markdown_path = markdown_output_dir / dynamic_md_filename

        run_specific_template_dir = (
            self.templates_base_dir / config.template_subdir
        )
        self.run_specific_template_dir = run_specific_template_dir
        try:
            report_base_dir.mkdir(parents=True, exist_ok=True)
            run_specific_output_dir.mkdir(parents=True, exist_ok=True)
            html_output_dir.mkdir(parents=True, exist_ok=True)
            data_output_dir.mkdir(parents=True, exist_ok=True)
            image_output_dir.mkdir(parents=True, exist_ok=True)
            css_output_dir.mkdir(parents=True, exist_ok=True)
            js_output_dir.mkdir(parents=True, exist_ok=True)
            markdown_output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Output directories ensured: {html_output_dir}")
        except OSError as e:
            logger.error(
                f"FATAL: Error creating output directories: {e}"
            )  # Changed to error
            raise

        # === FETCH PREVIOUS PERIOD METRICS ===
        previous_period_metrics: Optional[Dict[str, Any]] = None
        if metrics_service:
            logger.debug(
                "Fetching metrics for the period prior to"
                f" {start_date.strftime('%Y-%m-%d')}"
            )
            previous_period_metrics = (
                await self._fetch_previous_period_metrics(
                    current_start_date=start_date,
                    metrics_service=metrics_service,
                )
            )
            if previous_period_metrics:
                logger.debug(
                    "Successfully fetched"
                    f" {len(previous_period_metrics)} metrics from the"
                    " previous period."
                )
            else:
                logger.warning(
                    "No previous period metrics found or an error occurred"
                    " while fetching."
                )
        else:
            logger.warning(
                "Metrics service not provided. Skipping fetch of previous"
                " period metrics."
            )
        # === END FETCH PREVIOUS PERIOD METRICS ===

        # --- 2. Prepare Input Data ---
        try:
            main_df = self._prepare_input_dataframe(report_data)
        except (TypeError, ValueError) as e:
            logger.error(
                f"FATAL: Input data preparation failed: {e}"
            )  # Changed to error
            raise

        if main_df.empty:
            logger.warning(
                "Prepared DataFrame is empty. Report generation will continue"
                " but may be incomplete."
            )
            # Handle empty df gracefully in subsequent steps
        calculated_metrics = self._calculate_all_metrics(
            df=main_df, previous_period_metrics=previous_period_metrics
        )
        top_cwe_spotlight_data_list = (
            self._prepare_data_for_top_cwe_spotlight_cards(main_df, top_n=4)
        )
        logger.info("Top CWE Spotlight Data calculated successfully.")
        for i, cwe_data in enumerate(top_cwe_spotlight_data_list):
            slot_metric_id = f"spotlight_cwe_slot_{i+1}"
            # Store the actual data for this CWE in the slot
            calculated_metrics[slot_metric_id] = cwe_data
        # logger.info("--- Metrics Dictionary Immediately After Calculation ---")
        # for key in sorted(calculated_metrics.keys()):
        #     logger.info(f"  FROM_CALC - {key}: {str(calculated_metrics[key])[:200]}")
        formatted_metrics_for_cards = self._format_metrics_for_report_cards(
            calculated_metrics
        )

        # --- 3. Generate Charts & LLM Insights ---
        generated_charts_context: Dict[str, ChartExport] = (
            {}
        )  # Holds context for charts successfully generated for final report
        generated_chart_files: List[Path] = (
            []
        )  # Holds paths to successfully exported chart JSON
        generated_image_files: List[Path] = (
            []
        )  # Holds paths to successfully exported chart images
        llm_section_content: Dict[str, Dict[str, str]] = defaultdict(
            dict
        )  # Stores final LLM text output {section_key: {prompt_type: text}}
        all_chart_data_for_llm: Dict[str, str] = (
            {}
        )  # Stores formatted data strings ONLY for successfully generated charts

        # Base context for ALL LLM prompts passed via Python
        base_llm_context_from_python = {
            "report_title": config.report_title,
            "start_date": start_date,
            "end_date": end_date,
            "total_cves": calculated_metrics.get("total_cves", 0),
            "config": (
                config
            ),  # Needed for finding template paths in _get_llm_response
        }

        # --- Iterate through Looped Sections Defined in Structure ---
        for section_key, section_config in self.REPORT_STRUCTURE.get(
            "looped_sections", {}
        ).items():
            logger.info(
                f"Processing section: {section_key} -"
                f" {section_config.get('heading', '')}"
            )

            section_charts_context: Dict[str, ChartExport] = (
                {}
            )  # Holds ChartExport for charts relevant *to this section*
            section_chart_insights: Dict[str, str] = (
                {}
            )  # Holds LLM insights for charts relevant *to this section*
            section_callouts: Dict[str, str] = (
                {}
            )  # Holds LLM callouts generated *for this section*

            # --- Generate Charts for the Section ---
            for chart_def in section_config.get("charts", []):
                chart_id = chart_def["id"]
                logger.info(f"--> Processing chart: {chart_id}")

                # Prepare data using the chart ID
                # Pass copy only if _prepare modifies df, otherwise pass main_df directly for efficiency
                chart_data = self._prepare_chart_data(
                    df=main_df if main_df.empty else main_df.copy(),
                    chart_key=chart_id,
                    data_output_dir=data_output_dir,
                )

                # Generate figure using the chart ID
                figure = self._generate_plotly_figure(
                    chart_data,
                    chart_id,
                    chart_title_override=chart_def.get("title"),
                )
                chart_image_path: Optional[Path] = None

                if figure:
                    html_chart_id = f"chart-{chart_id.replace('_', '-')}"
                    json_path = self._export_chart_json(
                        figure, html_chart_id, data_output_dir
                    )
                    chart_image_path = self._export_chart_image(
                        figure, html_chart_id, image_output_dir
                    )
                    if chart_image_path:
                        generated_image_files.append(
                            chart_image_path
                        )  # Track successful image exports

                    if json_path:
                        # Chart generated and exported successfully
                        generated_chart_files.append(json_path)
                        chart_export = ChartExport(
                            chart_id=html_chart_id,
                            caption=chart_def.get(
                                "caption", f"Chart: {chart_id}"
                            ),
                        )
                        generated_charts_context[chart_id] = (
                            chart_export  # Add to global context for final report rendering
                        )
                        section_charts_context[chart_id] = (
                            chart_export  # Add to section context for LLM prompts
                        )

                        formatted_data = self._format_data_for_llm(
                            chart_data, format_type='markdown'
                        )
                        all_chart_data_for_llm[chart_id] = (
                            formatted_data  # Store globally
                        )

                        # --- Get LLM Insight for Chart ---
                        insight_prompt_template = chart_def.get(
                            "insight_prompt"
                        )
                        if insight_prompt_template and self.llm:
                            prompt_context_data = {
                                **base_llm_context_from_python,
                                "section_title": section_config.get(
                                    'heading', 'Chart Analysis'
                                ),
                                "chart_caption": chart_def.get("caption"),
                                "data_context_payload": {
                                    "chart_data_summary": formatted_data,
                                },
                            }
                            logger.info(
                                "\nBuilding LLM prompt for chart"
                                " insight:\ntemplate"
                                f" {insight_prompt_template}\n data context"
                                f" {prompt_context_data}\n\n"
                            )
                            insight = await self._get_llm_response(
                                prompt_template_name=insight_prompt_template,
                                prompt_context_data=prompt_context_data,
                                llm_model=self.LLM_TASK_MODELS.get(
                                    "chart_insight"
                                ),
                                llm_max_tokens=1056,
                                llm_temperature=0.85,
                                image_paths=[chart_image_path],
                            )
                            insight_html = markdown.markdown(
                                insight,
                                extensions=[
                                    'fenced_code',
                                    'tables',
                                    'attr_list',
                                ],
                            )
                            llm_section_content[section_key][
                                f"chart_insight_{chart_id}"
                            ] = insight_html
                            section_chart_insights[chart_id] = (
                                insight_html  # Store for use in narrative/summary prompts
                            )
                        elif not self.llm:
                            logger.warning(
                                "LLM client not available, skipping insight"
                                f" for chart {chart_id}"
                            )
                            llm_section_content[section_key][
                                f"chart_insight_{chart_id}"
                            ] = "LLM insight skipped (client unavailable)."
                        else:
                            logger.warning(
                                "No insight_prompt defined for chart"
                                f" {chart_id}"
                            )

                    else:  # JSON export failed
                        logger.warning(
                            "Skipping LLM insight and context for chart"
                            f" '{chart_id}' due to JSON export error."
                        )
                        generated_charts_context[chart_id] = ChartExport(
                            chart_id=html_chart_id,
                            caption="(Chart JSON export failed)",
                        )
            # --- End of chart loop ---

            # --- Generate Section-Level LLM Content ---
            section_prompts = section_config.get("prompts", {})
            if not self.llm:
                logger.warning(
                    "LLM client not available, skipping section prompts for"
                    f" {section_key}"
                )
            else:
                # --- Generate Callouts ---
                for i in range(1, 4):
                    callout_key = f"callout_{i}"
                    callout_prompt_template = section_prompts.get(callout_key)
                    if callout_prompt_template:
                        prompt_context_data = {
                            **base_llm_context_from_python,
                            "section_title": section_config.get(
                                'heading', f'Section Callout {i}'
                            ),
                            "data_context_payload": {
                                "chart_insights_in_section": (
                                    section_chart_insights
                                ),  # Pass insights from charts *in this section*
                                "charts_data_summaries_in_section": {
                                    cid: all_chart_data_for_llm.get(cid, "N/A")
                                    for cid in section_charts_context
                                },
                            },
                        }
                        logger.info(
                            "\nBuilding LLM prompt for callout:\ntemplate"
                            f" {callout_prompt_template}\ndata context"
                            f" {prompt_context_data}\n\n"
                        )
                        callout_text = await self._get_llm_response(
                            prompt_template_name=callout_prompt_template,
                            prompt_context_data=prompt_context_data,
                            llm_model=self.LLM_TASK_MODELS.get("callout"),
                            image_paths=None,
                        )
                        llm_section_content[section_key][
                            callout_key
                        ] = callout_text
                        callout_html = markdown.markdown(
                            callout_text,
                            extensions=['fenced_code', 'tables', 'attr_list'],
                        )
                        section_callouts[callout_key] = (
                            callout_html  # Store for use in narrative/summary prompts
                        )

                        # # Ensure the main template context (self.llm_insights) gets the HTML version
                        # template_expected_key = f"{section_key}_callout_{str(i)}"
                        # llm_insights[template_expected_key] = callout_text

                # --- Generate Narrative ---
                stats_definitions = section_config.get("stats_definitions", [])
                current_section_stats_for_prompt = {}
                if stats_definitions:
                    for stat_def in stats_definitions:
                        metric_id = stat_def.get("metric_id")
                        if metric_id:
                            current_section_stats_for_prompt[metric_id] = (
                                calculated_metrics.get(metric_id)
                            )
                            # Add QoQ metrics automatically as discussed
                            for qoq_suffix in [
                                "_qoq_delta_abs",
                                "_qoq_delta_pct",
                                "_qoq_comparison_str",
                            ]:
                                qoq_metric_id = f"{metric_id}{qoq_suffix}"
                                if qoq_metric_id in calculated_metrics:
                                    current_section_stats_for_prompt[
                                        qoq_metric_id
                                    ] = calculated_metrics[qoq_metric_id]
                narrative_prompt_template = section_prompts.get("narrative")
                if narrative_prompt_template:
                    prompt_context_data = {
                        **base_llm_context_from_python,
                        "section_title": section_config.get(
                            'heading', 'Section Narrative'
                        ),
                        "data_context_payload": {
                            "chart_insights_in_section": (
                                section_chart_insights
                            ),  # Pass insights generated above
                            "callouts_in_section": (
                                section_callouts
                            ),  # Pass callouts generated above
                            "charts_data_summaries_in_section": {
                                cid: all_chart_data_for_llm.get(cid, "N/A")
                                for cid in section_charts_context
                            },
                            "section_specific_stats": (
                                current_section_stats_for_prompt
                            ),
                        },
                    }
                    logger.info(
                        "\nBuilding LLM prompt for narrative:\ntemplate"
                        f" {narrative_prompt_template}\ndata context"
                        f" {prompt_context_data}\n\n"
                    )
                    narrative_text = await self._get_llm_response(
                        prompt_template_name=narrative_prompt_template,
                        prompt_context_data=prompt_context_data,
                        llm_model=self.LLM_TASK_MODELS.get("narrative"),
                        llm_max_tokens=1024,
                        image_paths=None,
                    )
                    # Store the generated narrative in the main llm_section_content dict
                    llm_markdown_narrative = narrative_text
                    llm_section_content[section_key]["narrative"] = (
                        markdown.markdown(
                            llm_markdown_narrative,
                            extensions=['fenced_code', 'tables', 'attr_list'],
                        )
                    )
                else:
                    # Still add placeholder if no prompt, so key exists if needed later
                    llm_section_content[section_key][
                        "narrative"
                    ] = "Narrative generation skipped (no prompt defined)."

                # --- Generate Summary ---
                summary_prompt_template = section_prompts.get(
                    "section_summary"
                )
                if summary_prompt_template:
                    prompt_context_data = {
                        **base_llm_context_from_python,
                        "section_title": section_config.get(
                            'heading', 'Section Summary'
                        ),
                        "data_context_payload": {
                            "callouts_in_section": section_callouts,
                            "generated_narrative": llm_section_content[
                                section_key
                            ].get("narrative", "Narrative not available."),
                            "section_specific_stats": (
                                current_section_stats_for_prompt
                            ),
                        },
                    }
                    logger.info(
                        "\nBuilding LLM prompt for summary:\ntemplate"
                        f" {summary_prompt_template}\ndata context"
                        f" {prompt_context_data}\n\n"
                    )
                    summary_text = await self._get_llm_response(
                        prompt_template_name=summary_prompt_template,
                        prompt_context_data=prompt_context_data,
                        llm_model=self.LLM_TASK_MODELS.get("section_summary"),
                        image_paths=None,
                    )
                    llm_markdown_summary = summary_text
                    llm_section_content[section_key]["summary"] = (
                        markdown.markdown(
                            llm_markdown_summary,
                            extensions=['fenced_code', 'tables', 'attr_list'],
                        )
                    )
                else:
                    llm_section_content[section_key][
                        "summary"
                    ] = "Summary generation skipped (no prompt defined)."
        # --- End of looped sections ---

        # --- 4. Generate Appendix Tables ---
        appendix_section = self.REPORT_STRUCTURE.get(
            "explicit_sections", {}
        ).get("appendix", {})
        appendix_tables_defs = appendix_section.get("tables", [])
        appendix_tables_context = self._generate_appendix_tables(
            main_df, appendix_tables_defs, calculated_metrics
        )

        # --- 6. Generate Explicit Section LLM Content (e.g., Executive Summary, Conclusion) ---
        if self.llm:
            # Get keys for analytical sections (usually looped sections)
            analytical_section_keys = self.REPORT_STRUCTURE.get(
                "looped_sections", {}
            ).keys()

            for section_key, section_config in self.REPORT_STRUCTURE.get(
                "explicit_sections", {}
            ).items():
                # Skip sections that don't have LLM prompts defined (like toc, appendix)
                section_prompts = section_config.get("prompts", {})
                if not section_prompts:
                    logger.debug(
                        "Skipping LLM generation for explicit section"
                        f" '{section_key}': No prompts defined."
                    )
                    continue

                # Determine the primary prompt template for this explicit section
                # Allow using a key like 'executive_summary' or a generic 'narrative' key within prompts
                prompt_template = section_prompts.get(
                    section_key
                ) or section_prompts.get("narrative")

                if prompt_template:
                    logger.info(
                        "Preparing LLM context for explicit section:"
                        f" {section_key}"
                    )

                    # Default payload - can be overridden below for specific sections
                    data_payload_for_llm = {}

                    # --- Customize context based on the specific explicit section ---
                    if section_key == "executive_summary":
                        # 1. Select key overall stats for Exec Summary LLM
                        overall_stats_for_llm = {
                            # --- Select the most important high-level stats ---
                            "total_cves": calculated_metrics.get("total_cves"),
                            "critical_cves_count": calculated_metrics.get(
                                "critical_count"
                            ),
                            "important_cves_count": calculated_metrics.get(
                                "important_count"
                            ),
                            "critical_high_percentage": calculated_metrics.get(
                                "critical_high_pct"
                            ),
                            "cve_category_distribution_raw_counts": (
                                calculated_metrics.get(
                                    "cve_category_distribution_raw_counts"
                                )
                            ),
                            "rce_pct": calculated_metrics.get("rce_pct"),
                            "eop_pct": calculated_metrics.get("eop_pct"),
                            "median_cvss_overall": calculated_metrics.get(
                                "median_cvss"
                            ),
                            "exploited_percentage": calculated_metrics.get(
                                "exploited_pct_kev"
                            ),  # Using KEV %
                            "median_days_to_patch": calculated_metrics.get(
                                "median_days_to_patch"
                            ),
                            "worst_case_cves_count": calculated_metrics.get(
                                "worst_case_cves"
                            ),
                            "top_impact_type": calculated_metrics.get(
                                "top_impact_type"
                            ),
                        }
                        overall_stats_for_llm = {
                            k: v
                            for k, v in overall_stats_for_llm.items()
                            if v is not None
                        }

                        # 2. Prepare combined narrative/summary texts from ANALYTICAL sections
                        section_texts_for_exec_summary_prompt = {}

                        MAX_CHARS_PER_SECTION_CONTEXT = 700
                        MAX_TOKENS_PER_SECTION_CONTEXT = 500
                        for (
                            other_section_key,
                            other_section_data,
                        ) in llm_section_content.items():
                            if other_section_key in analytical_section_keys:
                                narrative_html = other_section_data.get(
                                    'narrative'
                                )
                                summary_html = other_section_data.get(
                                    'summary'
                                )
                                combined_text_for_section = ""
                                current_chars = 0
                                # current_tokens = 0 # if using tiktoken

                                # Priority 1: Narrative (all but last paragraph)
                                if narrative_html:
                                    narrative_paragraphs = (
                                        get_paragraphs_from_html(
                                            narrative_html
                                        )
                                    )
                                    if narrative_paragraphs:
                                        # Add "Narrative:" marker
                                        combined_text_for_section += (
                                            "Key points from section"
                                            " narrative:\n"
                                        )
                                        # Take all but the last paragraph of the narrative
                                        pars_to_take = (
                                            narrative_paragraphs[:-1]
                                            if len(narrative_paragraphs) > 1
                                            else narrative_paragraphs
                                        )
                                        for i, p_html in enumerate(
                                            pars_to_take
                                        ):
                                            # text_to_add = p_html # If passing HTML directly
                                            # Or strip tags for cleaner text for LLM (LLM will re-markdown)
                                            text_to_add = BeautifulSoup(
                                                p_html, "html.parser"
                                            ).get_text(
                                                separator=" ", strip=True
                                            )

                                            # Check budget (character-based example)
                                            if (
                                                current_chars
                                                + len(text_to_add)
                                                < MAX_TOKENS_PER_SECTION_CONTEXT
                                            ):
                                                combined_text_for_section += (
                                                    text_to_add + "\n"
                                                )  # Add newline for LLM readability
                                                current_chars += len(
                                                    text_to_add
                                                )
                                            else:
                                                # Add truncated part if possible
                                                remaining_chars = (
                                                    MAX_TOKENS_PER_SECTION_CONTEXT
                                                    - current_chars
                                                )
                                                if (
                                                    remaining_chars > 20
                                                ):  # Only add if meaningful amount
                                                    combined_text_for_section += (
                                                        text_to_add[
                                                            :remaining_chars
                                                        ]
                                                        + "...\n"
                                                    )
                                                current_chars = MAX_TOKENS_PER_SECTION_CONTEXT  # Mark as full
                                                break  # Stop adding from narrative
                                        combined_text_for_section += (  # Separator
                                            "\n"
                                        )

                                # Priority 2: Summary (first few paragraphs, if room)
                                if (
                                    summary_html
                                    and current_chars
                                    < MAX_TOKENS_PER_SECTION_CONTEXT
                                ):
                                    summary_paragraphs = (
                                        get_paragraphs_from_html(summary_html)
                                    )
                                    if summary_paragraphs:
                                        # Add "Summary:" marker if narrative was also added, or as main header
                                        if (
                                            narrative_html
                                            and narrative_paragraphs
                                        ):  # Check if narrative part was added
                                            combined_text_for_section += (
                                                "Key points from section"
                                                " summary:\n"
                                            )
                                        else:  # No narrative, summary is primary
                                            combined_text_for_section += (
                                                "Section summary:\n"
                                            )

                                        for i, p_html in enumerate(
                                            summary_paragraphs
                                        ):
                                            # text_to_add = p_html # If passing HTML directly
                                            text_to_add = BeautifulSoup(
                                                p_html, "html.parser"
                                            ).get_text(
                                                separator=" ", strip=True
                                            )

                                            if (
                                                current_chars
                                                + len(text_to_add)
                                                < MAX_CHARS_PER_SECTION_CONTEXT
                                            ):
                                                combined_text_for_section += (
                                                    text_to_add + "\n"
                                                )
                                                current_chars += len(
                                                    text_to_add
                                                )
                                            else:
                                                remaining_chars = (
                                                    MAX_CHARS_PER_SECTION_CONTEXT
                                                    - current_chars
                                                )
                                                if remaining_chars > 20:
                                                    combined_text_for_section += (
                                                        text_to_add[
                                                            :remaining_chars
                                                        ]
                                                        + "...\n"
                                                    )
                                                break  # Stop adding from summary

                                if combined_text_for_section.strip():
                                    section_texts_for_exec_summary_prompt[
                                        other_section_key
                                    ] = combined_text_for_section.strip()

                        # 3. Set the specific payload for the Exec Summary prompt
                        data_payload_for_llm = {
                            "overall_report_stats_for_llm": (
                                overall_stats_for_llm
                            ),
                            "section_main_texts": (
                                section_texts_for_exec_summary_prompt
                            ),
                        }
                        # Define model parameters specifically for Exec Summary
                        model_key = section_key  # Use 'executive_summary'
                        llm_model = self.LLM_TASK_MODELS.get(
                            model_key, self.LLM_TASK_MODELS.get("narrative")
                        )  # Fallback
                        max_tokens = (
                            1500  # More tokens for exec summary usually
                        )

                    elif section_key == "report_conclusion":
                        # 1. For the conclusion, we might want similar overall stats as context
                        overall_stats_for_llm_conclusion = {
                            # Select stats relevant for concluding remarks
                            "total_cves": calculated_metrics.get("total_cves"),
                            "critical_high_percentage": calculated_metrics.get(
                                "critical_high_pct"
                            ),
                            "median_cvss_overall": calculated_metrics.get(
                                "median_cvss"
                            ),
                            "exploited_percentage": calculated_metrics.get(
                                "exploited_pct_kev"
                            ),
                            "median_days_to_patch": calculated_metrics.get(
                                "median_days_to_patch"
                            ),
                            "worst_case_cves_count": calculated_metrics.get(
                                "worst_case_cves"
                            ),
                            "top_impact_type": calculated_metrics.get(
                                "top_impact_type"
                            ),
                            "cve_category_distribution_raw_counts": (
                                calculated_metrics.get(
                                    "cve_category_distribution_raw_counts"
                                )
                            ),
                        }
                        overall_stats_for_llm_conclusion = {
                            k: v
                            for k, v in overall_stats_for_llm_conclusion.items()
                            if v is not None
                        }

                        # 2. We also need the generated Executive Summary itself (if available) and maybe section summaries
                        # Assuming exec summary was generated before conclusion if order matters
                        executive_summary_text = llm_section_content.get(
                            "executive_summary", {}
                        ).get("narrative", "Executive Summary not available.")

                        section_summaries = {}
                        for (
                            other_section_key,
                            other_section_data,
                        ) in llm_section_content.items():
                            if other_section_key in analytical_section_keys:
                                summary_text = other_section_data.get(
                                    'summary'
                                )
                                if summary_text:
                                    section_summaries[other_section_key] = (
                                        summary_text  # Pass the raw summary text (Markdown?)
                                    )

                        # 3. Set the specific payload for the Conclusion prompt
                        data_payload_for_llm = {
                            "overall_report_stats_for_llm": (
                                overall_stats_for_llm_conclusion
                            ),
                            "executive_summary_text": (
                                executive_summary_text
                            ),  # Pass the generated summary text/HTML
                            "section_summaries": (
                                section_summaries
                            ),  # Pass summaries from analytical sections
                        }
                        # Define model parameters specifically for Conclusion
                        model_key = section_key  # Use 'report_conclusion'
                        llm_model = self.LLM_TASK_MODELS.get(
                            model_key, self.LLM_TASK_MODELS.get("narrative")
                        )
                        max_tokens = (
                            1000  # Adjust as needed for conclusion length
                        )

                    else:
                        # --- Default Context for other Explicit Sections (if any need LLM) ---
                        # Passes all metrics and all previously generated content (less efficient)
                        # Consider if other explicit sections need tailored context too.
                        logger.warning(
                            "Using default context for explicit section"
                            f" '{section_key}'. Consider tailoring."
                        )
                        data_payload_for_llm = {
                            "overall_report_stats": (
                                calculated_metrics
                            ),  # Passes everything
                            "all_section_content": (
                                llm_section_content
                            ),  # Passes everything generated so far
                        }
                        # Default model parameters
                        model_key = (
                            section_key
                            if section_key in self.LLM_TASK_MODELS
                            else "narrative"
                        )
                        llm_model = self.LLM_TASK_MODELS.get(model_key)
                        max_tokens = 1000  # Default length

                    # --- Prepare the final prompt context data ---
                    prompt_context_data = {
                        **base_llm_context_from_python,  # Includes dates, title etc.
                        "section_title": section_config.get(
                            'heading', section_key.replace('_', ' ').title()
                        ),
                        # Add section_key itself to the payload if prompts need it internally
                        "data_context_payload": {
                            "section_key": (
                                section_key
                            ),  # Make section_key available inside payload
                            **data_payload_for_llm,  # Add the specific payload determined above
                        },
                    }

                    # --- Call LLM ---
                    logger.info(
                        "Generating content for explicit section:"
                        f" {section_key} using model {llm_model}"
                    )
                    logger.info(
                        "\nBuilding LLM prompt for explicit"
                        f" section:\ntemplate {prompt_template}\ndata context"
                        f" {prompt_context_data}\n\n"
                    )
                    narrative_markdown = await self._get_llm_response(
                        prompt_template_name=prompt_template,
                        prompt_context_data=prompt_context_data,
                        llm_model=llm_model,
                        llm_max_tokens=max_tokens,
                        image_paths=None,
                    )

                    # --- Store the result (Convert Markdown to HTML) ---
                    if narrative_markdown:
                        # Convert the LLM's Markdown output to HTML
                        narrative_html = markdown.markdown(
                            narrative_markdown,
                            extensions=['fenced_code', 'tables', 'attr_list'],
                        )
                        # Store the HTML in the results dictionary
                        llm_section_content[section_key][
                            "narrative"
                        ] = narrative_html
                        logger.info(
                            "Successfully generated and stored HTML content"
                            f" for {section_key}"
                        )
                    else:
                        logger.warning(
                            f"LLM returned empty content for {section_key}"
                        )
                        llm_section_content[section_key]["narrative"] = (
                            "<p>Content generation failed for"
                            f" {section_key}.</p>"
                        )

                else:
                    # Create placeholder if no prompt template was found for this section
                    logger.debug(
                        "Skipping LLM generation for explicit section"
                        f" '{section_key}': No primary prompt template found."
                    )
                    # Check if content already exists (e.g., from a previous run or default)
                    if "narrative" not in llm_section_content.get(
                        section_key, {}
                    ):
                        llm_section_content[section_key]["narrative"] = (
                            f"<p>Narrative not generated for {section_key} (no"
                            " prompt template defined).</p>"
                        )

        # --- 7. Prepare Final Jinja Context ---
        final_llm_insights = {}
        for section_key, content_dict in llm_section_content.items():
            for prompt_type, text in content_dict.items():
                final_llm_insights[f"{section_key}_{prompt_type}"] = text

        report_context = ReportContext(
            report_title=config.report_title,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            start_date=start_date,
            end_date=end_date,
            config=config,
            charts=generated_charts_context,
            llm_insights=final_llm_insights,
            appendix_tables=appendix_tables_context,
            report_structure=self.REPORT_STRUCTURE,
            section_stats_data=formatted_metrics_for_cards,
        )

        # --- 8. Render & Save HTML ---
        html_content = ""
        try:
            # Construct path relative to the *base* template directory Jinja knows
            template_relative_path = (
                Path(config.template_subdir) / config.template_html_filename
            )
            logger.debug(
                "Template path:"
                f" {type(template_relative_path)} {template_relative_path}"
            )
            template = self.jinja_env.get_template(
                template_relative_path.as_posix()
            )
            # Pass the Pydantic model's dict representation to render
            context = report_context.model_dump()
            context["report_structure"] = self.REPORT_STRUCTURE
            # logger.info("---- Context after model_dump ----")
            # # Find the card for cvss_iqr_by_category in section_4_cvss_dist to inspect
            # if 'section_stats_data' in context and 'section_4_cvss_dist' in context['section_stats_data']: # Corrected key
            #     s4_data = context['section_stats_data']['section_4_cvss_dist'] # Corrected key
            #     if isinstance(s4_data, list):
            #         for card_item_dict_dumped in s4_data:
            #             if card_item_dict_dumped.get('id') == 'cvss_iqr_by_category':
            #                 logger.info(f"DUMPED Card 'cvss_iqr_by_category' data.value type: {type(card_item_dict_dumped.get('value'))}")
            #                 logger.info(f"DUMPED Card 'cvss_iqr_by_category' data.value content: {card_item_dict_dumped.get('value')}")
            #                 if isinstance(card_item_dict_dumped.get('value'), list) and card_item_dict_dumped.get('value'):
            #                     logger.info(f"Type of first item IN DUMPED data.value list: {type(card_item_dict_dumped.get('value')[0])}")
            #                 break
            # logger.info("---- End Context after model_dump ----")
            html_content = template.render(context)
            final_html_path.write_text(html_content, encoding="utf-8")
            logger.info(f"Report HTML saved to: {final_html_path}")
        except Exception as e:
            logger.error(
                f"FATAL: Error rendering or saving HTML report: {e}",
                exc_info=True,
            )  # Add exc_info
            raise

        # --- 9. Generate & Save Markdown ---

        try:
            if html_content:
                logger.debug("Converting generated HTML to Markdown...")
                markdown_output = markdownify.markdownify(
                    html_content, heading_style="ATX"
                )
                final_markdown_path.write_text(
                    markdown_output, encoding="utf-8"
                )
                logger.info(f"Report Markdown saved to: {final_markdown_path}")
            else:
                logger.warning(
                    "Skipping Markdown generation because HTML content is"
                    " empty."
                )
                final_markdown_path = None
        except Exception as e:
            logger.error(
                f"Error generating or saving Markdown report: {e}",
                exc_info=True,
            )  # Add exc_info
            final_markdown_path = None

        # --- 10. Copy Static Assets ---
        css_path, js_path = self._place_static_assets(
            css_output_dir, js_output_dir, config
        )

        # --- 11. Collate Asset Paths ---
        generated_assets = GeneratedReportAssets(
            report_config=config,
            base_directory=report_base_dir,
            html_file=(
                final_html_path if html_content else None
            ),  # Pass None if HTML failed
            markdown_file=final_markdown_path,
            css_file=css_path,
            js_file=js_path,  # js_path likely None based on _place_static_assets stub
            chart_data_files=generated_chart_files,
            image_files=generated_image_files,
            calculated_metrics=calculated_metrics,
        )
        # --- 12.Count tokens and estimate cost of report

        llm_usage_report_data = self.get_llm_token_usage_report_data()

        # Log the formatted summary text
        logger.info("\n" + llm_usage_report_data["summary_text"])

        # Optionally, save the summary text to a file
        summary_file_path = (
            DATA_DIR
            / "llm_usage"
            / "quarterly_deep_dive"
            / f"llm_usage_summary_{end_date.strftime('%Y-%m-%d')}.txt"
        )
        try:
            with open(summary_file_path, "w", encoding="utf-8") as f:
                f.write(llm_usage_report_data["summary_text"])
            logger.debug(f"LLM usage summary saved to: {summary_file_path}")
        except Exception as e:
            logger.error(f"Failed to save LLM usage summary: {e}")

        # logger.debug("--- Metrics Dictionary Just Before Saving Loop ---")

        # for key in sorted(calculated_metrics.keys()):  # Use the actual variable name
        #     logger.info(f"  TO_SAVE - {key}: {str(calculated_metrics[key])[:200]}")
        # Save metrics to the database
        if metrics_service:
            period_key = f"{start_date.year}Q{(start_date.month - 1) // 3 + 1}"
            time_details_for_db = {
                "start_date": start_date,
                "end_date": end_date,
                "label": (
                    f"Q{(start_date.month - 1) // 3 + 1} {start_date.year} -"
                    f" {config.report_title}"
                ),
            }
            try:
                logger.debug(
                    "Attempting to save calculated metrics for period"
                    f" {period_key} using report name '{config.report_title}'."
                )
                await self._save_all_metrics(
                    metrics_data=calculated_metrics,
                    period_key=period_key,
                    report_name=config.report_title,  # Using report_title for clarity
                    metrics_service=metrics_service,
                    time_info=time_details_for_db,
                )
                logger.debug(
                    "Successfully initiated saving of metrics for period"
                    f" {period_key}."
                )
            except Exception as e:
                logger.error(
                    "Error occurred while calling _save_all_metrics for"
                    f" period {period_key}: {e}",
                    exc_info=True,
                )
                # Decide if this error should halt report generation or just be logged.
                # For now, it's logged, and report generation continues.
        else:
            logger.warning(
                "Metrics service not provided to generate_report. Skipping"
                " saving metrics."
            )
        # --- 14. Finish ---
        end_time = datetime.now()
        logger.info(
            "Report generation finished:"
            f" {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logger.info(f"Total time: {end_time - start_time}")

        return generated_assets

    def _standardize_windows_product_name(
        self, product_string: str | None
    ) -> str:
        """Helper to standardize Windows product name strings."""
        if not isinstance(product_string, str) or not product_string.strip():
            return "Unknown Product Version"  # Consistent fallback

        name = product_string.lower()
        name = name.replace("-based_systems", "").strip()
        name = name.replace(" version ", " ")
        name = name.replace("32-bit_systems", "32-bit")

        if "amd64" in name:
            name = name.replace("amd64", "x64")
        # Add more specific normalizations based on your data's exact variations
        # e.g., to group server editions or handle "enterprise" vs "pro" if needed

        parts = name.split()
        if parts:
            # Capitalize "Windows"
            if (
                parts[0].lower() == "windows" and len(parts) > 1
            ):  # Avoid capitalizing if it's just "windows"
                parts[0] = "Windows"
            # Capitalize version identifiers like 22H2
            for i, part in enumerate(parts):
                if (
                    len(part) > 1 and "h" in part and part[:-1].isdigit()
                ):  # e.g. 21h2, 22H2
                    parts[i] = part.upper()
            name = " ".join(parts)

        return name if name else "Unknown Product Version"

    def get_llm_token_usage_report_data(self) -> Dict[str, Any]:
        """
        Processes the llm_token_usage_log to generate aggregated statistics,
        cost estimates, and a formatted summary string.

        Returns:
            A dictionary containing:
                - "summary_text": A multi-line string summarizing token usage and costs.
                - "total_calls": int
                - "successful_calls": int
                - "failed_calls": int
                - "grand_total_input_tokens": int (sum of 'final_input_tokens_for_cost')
                - "grand_total_completion_tokens": int (sum of 'final_completion_tokens_for_cost')
                - "grand_total_tokens_billed": int (sum of input + completion used for cost)
                - "grand_total_estimated_cost": float
                - "usage_by_model": Dict[str, Dict] (aggregates per model)
                - "usage_by_task": Dict[str, Dict] (aggregates per task type)
                - "detailed_log": List[Dict] (the raw log itself)
        """
        if not self.llm_token_usage_log:
            empty_summary_text = (
                "No LLM calls were made or logged during this report"
                " generation."
            )
            logger.info(empty_summary_text)
            return {
                "summary_text": empty_summary_text,
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "grand_total_input_tokens": 0,
                "grand_total_completion_tokens": 0,
                "grand_total_tokens_billed": 0,
                "grand_total_estimated_cost": 0.0,
                "usage_by_model": {},
                "usage_by_task": {},
                "detailed_log": [],
            }

        report_data = {
            "total_calls": len(self.llm_token_usage_log),
            "successful_calls": 0,
            "failed_calls": 0,
            "grand_total_input_tokens": 0,
            "grand_total_completion_tokens": 0,
            "grand_total_estimated_cost": 0.0,
            "usage_by_model": defaultdict(
                lambda: {
                    "calls": 0,
                    "success": 0,
                    "input_tokens": 0,
                    "completion_tokens": 0,
                    "total_billed_tokens": 0,
                    "cost": 0.0,
                }
            ),
            "usage_by_task": defaultdict(
                lambda: {
                    "calls": 0,
                    "success": 0,
                    "input_tokens": 0,
                    "completion_tokens": 0,
                    "total_billed_tokens": 0,
                    "cost": 0.0,
                }
            ),
        }

        for log_entry in self.llm_token_usage_log:
            is_success = log_entry["status"] == "Success"
            if is_success:
                report_data["successful_calls"] += 1
            else:
                report_data["failed_calls"] += 1

            input_tokens_for_cost = log_entry.get(
                "final_input_tokens_for_cost", 0
            )
            completion_tokens_for_cost = log_entry.get(
                "final_completion_tokens_for_cost", 0
            )
            current_cost = log_entry.get("cost_estimate", 0.0)

            report_data["grand_total_input_tokens"] += input_tokens_for_cost
            report_data[
                "grand_total_completion_tokens"
            ] += completion_tokens_for_cost
            report_data["grand_total_estimated_cost"] += current_cost

            # By model
            model_stats = report_data["usage_by_model"][
                log_entry["model_name"]
            ]
            model_stats["calls"] += 1
            if is_success:
                model_stats["success"] += 1
            model_stats["input_tokens"] += input_tokens_for_cost
            model_stats["completion_tokens"] += completion_tokens_for_cost
            model_stats["total_billed_tokens"] += (
                input_tokens_for_cost + completion_tokens_for_cost
            )
            model_stats["cost"] += current_cost

            # By task
            task_stats = report_data["usage_by_task"][
                log_entry.get("task_type", "unknown_task")
            ]
            task_stats["calls"] += 1
            if is_success:
                task_stats["success"] += 1
            task_stats["input_tokens"] += input_tokens_for_cost
            task_stats["completion_tokens"] += completion_tokens_for_cost
            task_stats["total_billed_tokens"] += (
                input_tokens_for_cost + completion_tokens_for_cost
            )
            task_stats["cost"] += current_cost

        report_data["grand_total_tokens_billed"] = (
            report_data["grand_total_input_tokens"]
            + report_data["grand_total_completion_tokens"]
        )

        # Format cost to a reasonable number of decimal places
        report_data["grand_total_estimated_cost"] = round(
            report_data["grand_total_estimated_cost"], 6
        )
        for model_name in report_data["usage_by_model"]:
            report_data["usage_by_model"][model_name]["cost"] = round(
                report_data["usage_by_model"][model_name]["cost"], 6
            )
        for task_name in report_data["usage_by_task"]:
            report_data["usage_by_task"][task_name]["cost"] = round(
                report_data["usage_by_task"][task_name]["cost"], 6
            )

        # --- Create Formatted Summary Text ---
        summary_lines = [
            "📊 LLM Token Usage & Cost Estimate Report 📊",
            "===========================================",
            (
                f"Total LLM Calls: {report_data['total_calls']} (Successful:"
                f" {report_data['successful_calls']}, Failed:"
                f" {report_data['failed_calls']})"
            ),
            "--- Overall Totals ---",
            (
                "  Total Input Tokens (billed):    "
                f" {report_data['grand_total_input_tokens']:,}"
            ),
            (
                "  Total Completion Tokens"
                f" (billed):{report_data['grand_total_completion_tokens']:,}"
            ),
            (
                "  Grand Total Tokens (billed):    "
                f" {report_data['grand_total_tokens_billed']:,}"
            ),
            (
                "  Grand Total Estimated Cost:     "
                f" ${report_data['grand_total_estimated_cost']:.6f}"
            ),
            "\n--- Usage by Model ---",
        ]
        for model, data in sorted(
            report_data["usage_by_model"].items()
        ):  # Sort for consistent output
            summary_lines.append(
                f"  Model: {model}\n    Calls: {data['calls']} (Success:"
                f" {data['success']})\n    Input Tokens:"
                f" {data['input_tokens']:,}, Completion Tokens:"
                f" {data['completion_tokens']:,}, Total Billed:"
                f" {data['total_billed_tokens']:,}\n    Est. Cost:"
                f" ${data['cost']:.6f}"
            )

        summary_lines.append("\n--- Usage by Task Type ---")
        for task, data in sorted(
            report_data["usage_by_task"].items()
        ):  # Sort for consistent output
            summary_lines.append(
                f"  Task: {task}\n    Calls: {data['calls']} (Success:"
                f" {data['success']})\n    Input Tokens:"
                f" {data['input_tokens']:,}, Completion Tokens:"
                f" {data['completion_tokens']:,}, Total Billed:"
                f" {data['total_billed_tokens']:,}\n    Est. Cost:"
                f" ${data['cost']:.6f}"
            )

        summary_lines.append("\n--- Detailed Log ---")
        for i, entry in enumerate(self.llm_token_usage_log):
            summary_lines.append(
                f"  Call {i+1}:"
                f" [{entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}] Task:"
                f" {entry.get('task_type', 'N/A')}, "
                f"Model: {entry['model_name']}, Status: {entry['status']}, "
                f"In(calc): {entry.get('total_input_tokens_calculated',0):,}, "
                "Out(api):"
                f" {entry.get('final_completion_tokens_for_cost',0):,}, "  # Changed from completion_tokens_api to final_...
                f"Cost: ${entry.get('cost_estimate', 0.0):.6f}"
            )

        report_data["summary_text"] = "\n".join(summary_lines)
        report_data["detailed_log"] = (
            self.llm_token_usage_log
        )  # Keep the original structured log too

        return report_data


# --- Synthetic Data Generation Utility (moved outside class) ---
def generate_synthetic_cve_data(
    start_date: datetime, end_date: datetime, num_records: int = 200
) -> pd.DataFrame:
    """
    Generates a Pandas DataFrame with synthetic CVE data, including detailed metadata
    for each record, simulating what might come from MSRC and NVD.
    The 'build_numbers' field is now populated within the main loop for each record.
    The 'text' field contains a longer paragraph for embedding purposes.
    The 'id_' field is the primary document ID.
    Field names match the structure expected after processing MongoDB records *without*
    unnecessary prefixes.
    """
    logger.info(
        f"Generating {num_records} synthetic CVE records (refactored)..."
    )
    fake = Faker()  # For synthetic data
    data = []

    # Define possible values based on sample and general knowledge
    post_types = ["Critical", "Solution provided", "Information only"]
    severities = ["critical", "high", "medium", "low"]
    cve_categories = [
        "remote_code_execution",
        "privilege_elevation",
        "denial_of_service",
        "disclosure",
        "spoofing",
        "tampering",
        "feature_bypass",
        "nc",
    ]
    attack_vectors = ["Network", "Adjacent", "Local", "Physical"]
    attack_complexities = ["Low", "High"]
    privileges_required = ["None", "Low", "High"]
    user_interactions = ["None", "Required"]
    scopes = ["Unchanged", "Changed"]
    products_list = [
        "windows 11 21H2 x64-based_systems",
        "windows 11 22H2 x64-based_systems",
        "windows 11 23H2 x64-based_systems",
        "windows 11 24H2 x64-based_systems",
        "windows 10 22H2 x64-based_systems",
        "windows 10 22H2 32-bit_systems",
        "windows 10 21H2 x64-based_systems",
        "windows 10 21H2 32-bit_systems",
        "windows 10  x64-based_systems",
        "windows 10  32-bit_systems",
    ]
    candidate_builds_pool = {
        "Windows 10 21H2": [
            "10.0.1.19044",
            "10.0.1.19052",
            "10.0.1.19060",
            "10.0.1.19068",
            "10.0.1.19076",
        ],
        "Windows 10 22H2": [
            "10.0.2.19045",
            "10.0.2.19053",
            "10.0.2.19061",
            "10.0.2.19069",
            "10.0.2.19077",
        ],
        "Windows 11 22H2": [
            "10.0.3.22621",
            "10.0.3.22634",
            "10.0.3.22647",
            "10.0.3.22659",
            "10.0.3.22672",
        ],
        "Windows 11 23H2": [
            "10.0.4.22631",
            "10.0.4.22644",
            "10.0.4.22657",
            "10.0.4.22669",
            "10.0.4.22682",
        ],
        "Windows 11 24H2": [
            "10.0.5.26100",
            "10.0.5.26118",
            "10.0.5.26136",
            "10.0.5.26154",
            "10.0.5.26172",
        ],
    }
    outlier_counts = {
        "Critical_low_end": 0,
        "Important_low": 0,
        "Important_high": 0,
        "Moderate_low": 0,
        "Moderate_high": 0,
        "Low_high_end": 0,
    }
    # --- Calculate number of worst-case CVEs to add (1-2 per 100 records) ---
    num_worst_case = max(
        1, min(int(num_records / 100) * 2, int(num_records * 0.02))
    )
    logger.info(
        f"Will include {num_worst_case} worst-case CVEs in the dataset"
    )

    # --- Flatten CWE IDs for random selection (once at the start) ---
    all_cwe_ids = [
        cwe_id
        for category_data in CWE_ROOT_CAUSE.values()
        for cwe_id in category_data.get("ids", [])
    ]
    if not all_cwe_ids:
        logger.warning(
            "No CWE IDs found in CWE_ROOT_CAUSE. Synthetic CWE data will be"
            " empty."
        )
        all_cwe_ids = ["CWE-0"]  # Add a placeholder to prevent errors if empty

    # --- Prepare build number lists (once at the start) ---
    win10_builds = []
    win11_builds = []
    if (
        'candidate_builds_pool' in locals()
        or 'candidate_builds_pool' in globals()
    ):
        for product_name, builds in candidate_builds_pool.items():
            if "Windows 10" in product_name:
                win10_builds.extend(builds)
            elif "Windows 11" in product_name:
                win11_builds.extend(builds)
    else:
        logger.warning(
            "'candidate_builds_pool' not found. Build number generation will"
            " use placeholders."
        )
        # Provide empty lists so later code doesn't break, placeholder logic will handle it

    adp_nist_fields = [
        f"{prefix}_{prop}"
        for prefix in ['adp', 'nist']
        for prop in [
            'attack_complexity',
            'attack_vector',
            'availability',
            'base_score',
            'base_score_num',
            'base_score_rating',
            'confidentiality',
            'exploitability_score',
            'impact_score',
            'integrity',
            'privileges_required',
            'scope',
            'user_interaction',
            'vector',
        ]
    ]
    TARGET_OUTLIERS_PER_TYPE_DIRECTION = 4
    for i in range(num_records):
        record = {}
        metadata = {}
        is_worst_case = i < num_worst_case
        # --- Add Synthetic Build Numbers (Moved inside the loop) ---
        num_builds = np.random.choice([2, 3])
        builds_for_cve = []

        strategies = []
        if win10_builds:
            strategies.append('win10_only')  # noqa E701
        if win11_builds:
            strategies.append('win11_only')  # noqa E701
        if win10_builds and win11_builds:
            strategies.append('mixed')  # noqa E701

        chosen_strategy = "fallback"
        if strategies:
            chosen_strategy = np.random.choice(strategies)

        if chosen_strategy == 'win10_only':
            builds_for_cve = list(
                np.random.choice(
                    win10_builds,
                    size=min(num_builds, len(win10_builds)),
                    replace=False,
                )
            )
        elif chosen_strategy == 'win11_only':
            builds_for_cve = list(
                np.random.choice(
                    win11_builds,
                    size=min(num_builds, len(win11_builds)),
                    replace=False,
                )
            )
        elif chosen_strategy == 'mixed':
            combined_pool = list(set(win10_builds + win11_builds))
            if combined_pool:
                builds_for_cve = list(
                    np.random.choice(
                        combined_pool,
                        size=min(num_builds, len(combined_pool)),
                        replace=False,
                    )
                )

        current_selected_count = len(builds_for_cve)
        if current_selected_count < num_builds:
            needed_additional = num_builds - current_selected_count
            combined_options = list(
                set(win10_builds + win11_builds) - set(builds_for_cve)
            )
            if combined_options:
                builds_for_cve.extend(
                    list(
                        np.random.choice(
                            combined_options,
                            size=min(needed_additional, len(combined_options)),
                            replace=False,
                        )
                    )
                )

        current_selected_count = len(builds_for_cve)
        if current_selected_count < num_builds:
            needed_at_end = num_builds - current_selected_count
            all_candidate_builds_flat = [
                item
                for sublist in candidate_builds_pool.values()
                for item in sublist
            ]
            if all_candidate_builds_flat:
                unique_from_all_candidates = list(
                    set(all_candidate_builds_flat) - set(builds_for_cve)
                )
                if unique_from_all_candidates:
                    builds_for_cve.extend(
                        list(
                            np.random.choice(
                                unique_from_all_candidates,
                                size=min(
                                    needed_at_end,
                                    len(unique_from_all_candidates),
                                ),
                                replace=False,
                            )
                        )
                    )
                if len(builds_for_cve) < num_builds:
                    needed_with_repeats = num_builds - len(builds_for_cve)
                    builds_for_cve.extend(
                        list(
                            np.random.choice(
                                all_candidate_builds_flat,
                                size=needed_with_repeats,
                                replace=True,
                            )
                        )
                    )

        if not builds_for_cve:  # Final placeholder if all else failed
            builds_for_cve = [
                f"placeholder_build_{j+1}.{fake.msisdn()[-4:]}"
                for j in range(num_builds)
            ]

        metadata['build_numbers'] = [
            [int(p) for p in build_str.split('.')]
            for build_str in set(builds_for_cve)
        ]
        final_fill_needed = num_builds - len(metadata['build_numbers'])
        if final_fill_needed > 0:
            all_available_options = list(
                set(win10_builds + win11_builds)
                - set(metadata['build_numbers'])
            )
            if all_available_options:
                metadata['build_numbers'].extend(
                    list(
                        np.random.choice(
                            all_available_options,
                            size=min(
                                final_fill_needed, len(all_available_options)
                            ),
                            replace=False,
                        )
                    )
                )
        metadata['build_numbers'] = metadata['build_numbers'][:num_builds]
        # --- End of build number logic for this record ---

        num_words = fake.random_int(min=10, max=15)
        fake_sentence_words = fake.sentence(nb_words=num_words)
        fake_title_base = fake_sentence_words.rstrip('.')
        record['id_'] = fake.uuid4()  # Primary ID for the document
        metadata['id'] = record[
            'id_'
        ]  # Also store in metadata for potential flattening later
        metadata['post_id'] = (
            f"CVE-{np.random.randint(2023, end_date.year + 1)}-{np.random.randint(10000, 99999)}"
        )
        metadata['published'] = fake.date_time_between(
            start_date=start_date, end_date=end_date, tzinfo=None
        ).replace(tzinfo=None)
        metadata['revision'] = np.random.choice(
            ["1.000", "1.100", "2.000"], p=[0.85, 0.10, 0.05]
        )
        metadata['severity_type'] = np.random.choice(
            severities, p=[0.05, 0.65, 0.25, 0.05]
        )
        metadata['cve_category'] = np.random.choice(
            cve_categories, p=[0.20, 0.28, 0.15, 0.10, 0.12, 0.05, 0.05, 0.05]
        )
        metadata['impact_type'] = (
            metadata['cve_category'].replace('_', ' ').title()
        )
        metadata['title'] = (
            f"{fake_title_base.capitalize()} ({metadata['post_id']})"
        )
        metadata['description'] = fake.paragraph(nb_sentences=2)
        metadata['summary'] = fake.paragraph(nb_sentences=4)
        metadata['source'] = (
            f"https://msrc.microsoft.com/update-guide/vulnerability/{metadata['post_id']}"
        )
        metadata['collection'] = "msrc_security_update"
        metadata['post_type'] = np.random.choice(
            post_types, p=[0.02, 0.55, 0.43]
        )
        metadata['products'] = fake.random_elements(
            elements=products_list, length=np.random.randint(1, 4), unique=True
        )
        metadata['nvd_published_date'] = metadata['published'] + timedelta(
            days=np.random.randint(1, 4)
        )
        metadata['nvd_description'] = metadata['description']

        selected_cwe_id = np.random.choice(all_cwe_ids)
        metadata['cwe_id'] = selected_cwe_id
        cwe_details = CWE_DETAILS.get(selected_cwe_id)
        if cwe_details:
            metadata['cwe_name'] = cwe_details.get('name', selected_cwe_id)
            metadata['cwe_url'] = cwe_details.get(
                'cwe_url',
                f"https://cwe.mitre.org/data/definitions/{selected_cwe_id.split('-')[-1]}.html",
            )
            metadata['cwe_description'] = cwe_details.get('description', '')
        else:
            metadata['cwe_name'] = selected_cwe_id
            metadata['cwe_url'] = (
                f"https://cwe.mitre.org/data/definitions/{selected_cwe_id.split('-')[-1]}.html"
            )
            metadata['cwe_description'] = ''
        metadata['cwe_source'] = "mitre-cwe"

        current_severity_type = metadata.get('severity_type')
        score = None
        if current_severity_type == 'critical':
            if (
                outlier_counts["Critical_low_end"]
                < TARGET_OUTLIERS_PER_TYPE_DIRECTION
            ):
                score = 8.33  # Force a score at the low end of Critical
                outlier_counts["Critical_low_end"] += 1
            else:
                # Generate scores typically in the higher part of Critical range
                score = np.random.uniform(9.1, 10.0)

        elif current_severity_type == 'high':
            if (
                outlier_counts["Important_low"]
                < TARGET_OUTLIERS_PER_TYPE_DIRECTION
            ):
                score = 7.0  # Force a score at the low end of Important
                outlier_counts["Important_low"] += 1
            elif (
                outlier_counts["Important_high"]
                < TARGET_OUTLIERS_PER_TYPE_DIRECTION
            ):
                score = 9.33  # Force a score at the high end of Important
                outlier_counts["Important_high"] += 1
            else:
                # Generate "normal" Important scores, slightly away from the extremes
                score = np.random.uniform(7.1, 8.8)

        elif current_severity_type == 'medium':
            if (
                outlier_counts["Moderate_low"]
                < TARGET_OUTLIERS_PER_TYPE_DIRECTION
            ):
                score = 3.777  # Force a score at the low end of Moderate
                outlier_counts["Moderate_low"] += 1
            elif (
                outlier_counts["Moderate_high"]
                < TARGET_OUTLIERS_PER_TYPE_DIRECTION
            ):
                score = 7.333  # Force a score at the high end of Moderate
                outlier_counts["Moderate_high"] += 1
            else:
                # Generate "normal" Moderate scores
                score = np.random.uniform(4.1, 6.8)

        else:  # Covers 'Low' or any other default case
            if (
                outlier_counts["Low_high_end"]
                < TARGET_OUTLIERS_PER_TYPE_DIRECTION
            ):
                score = 4.333  # Force a score at the high end of Low
                outlier_counts["Low_high_end"] += 1
            else:
                # Generate "normal" Low scores
                score = np.random.uniform(0.1, 3.8)

        # Ensure score is not None (should be covered by the logic above for known severities)
        if score is None:
            # Fallback for any unexpected severity_type, though ideally this shouldn't be hit
            score = np.random.uniform(0.1, 10.0)
            logger.warning(
                f"Unexpected severity_type '{current_severity_type}',"
                " assigning random score."
            )

        if is_worst_case:
            local_attack_vector = "Network"
            local_attack_complexity = "Low"
            local_privileges_required = "None"
            local_user_interaction = "None"
            local_scope = np.random.choice(scopes)  # Can vary
            local_confidentiality_impact = "High"
            local_integrity_impact = "High"
            local_availability_impact = "High"

            # Force a high CVSS score for worst-case CVEs
            local_base_score_num = round(np.random.uniform(9.1, 10.0), 1)
            local_base_score_rating = "critical"

            # Ensure the severity type matches
            metadata['severity_type'] = "critical"
        else:
            # Original random selection logic
            local_attack_vector = np.random.choice(attack_vectors)
            local_attack_complexity = np.random.choice(attack_complexities)
            local_privileges_required = np.random.choice(privileges_required)
            local_user_interaction = np.random.choice(user_interactions)
            local_scope = np.random.choice(scopes)
            local_confidentiality_impact = np.random.choice(
                ["High", "Low", "None"]
            )
            local_integrity_impact = np.random.choice(["High", "Low", "None"])
            local_availability_impact = np.random.choice(
                ["High", "Low", "None"]
            )

            # Use the 'score' calculated earlier (lines 7657-7704) based on the
            # randomly assigned metadata['severity_type'] (line 7631)
            local_base_score_num = round(score, 1)
            # metadata['severity_type'] for non-worst-case records is the one
            # randomly assigned on line 7631 and used to calculate 'score'.
            local_base_score_rating = metadata['severity_type']

        cvss_vector_parts = [
            f"AV:{local_attack_vector[0]}",
            f"AC:{local_attack_complexity[0]}",
            f"PR:{local_privileges_required[0]}",
            f"UI:{local_user_interaction[0]}",
            f"S:{local_scope[0]}",
            f"C:{local_confidentiality_impact[0]}",
            f"I:{local_integrity_impact[0]}",
            f"A:{local_availability_impact[0]}",
        ]
        # Standard CVSS v3.1 vector string format
        local_vector_str = "CVSS:3.1/" + "/".join(cvss_vector_parts)
        # String representation of the base score, often included with vector
        local_base_score_str = (
            f"{local_base_score_num} {local_base_score_rating}"
        )

        # --- Assign calculated values ONLY to 'cna_' prefixed keys in metadata ---
        # General fields like 'title', 'cwe_id', 'products', 'id' (original CVE ID)
        # should already be present in the 'metadata' dict, unprefixed, from earlier.

        metadata['cna_base_score'] = local_base_score_str
        metadata['cna_base_score_num'] = local_base_score_num
        metadata['cna_base_score_rating'] = local_base_score_rating
        metadata['cna_vector'] = local_vector_str
        metadata['cna_attack_vector'] = local_attack_vector
        metadata['cna_attack_complexity'] = local_attack_complexity
        metadata['cna_privileges_required'] = local_privileges_required
        metadata['cna_user_interaction'] = local_user_interaction
        metadata['cna_scope'] = local_scope
        metadata['cna_confidentiality'] = local_confidentiality_impact
        metadata['cna_integrity'] = local_integrity_impact
        metadata['cna_availability'] = local_availability_impact

        # --- Synthetic Exploitability and Impact Scores (Local Variables) ---
        # CVSS v3.x Exploitability Score can range from 0 to 3.9 (approx)
        # CVSS v3.x Impact Score can range from 0 to 6.0 (approx)
        # For simplicity, we'll use a broader plausible range for synthetic data.
        local_exploitability_score = round(np.random.uniform(0.1, 3.9), 1)
        local_impact_score = round(np.random.uniform(0.1, 6.0), 1)

        # --- Assign to 'cna_' prefixed keys in metadata ---
        metadata['cna_exploitability_score'] = local_exploitability_score
        metadata['cna_impact_score'] = local_impact_score

        # --- Define base names for NIST/CVSS fields that require cna/adp/nist versions ---
        # This list should *only* contain the specific CVSS component names.
        nist_cvss_base_field_names = [
            'base_score',
            'base_score_num',
            'base_score_rating',
            'vector',
            'attack_vector',
            'attack_complexity',
            'privileges_required',
            'user_interaction',
            'scope',
            'confidentiality',
            'integrity',
            'availability',
            'exploitability_score',
            'impact_score',
        ]

        # --- Set 'adp_' and 'nist_' prefixed versions to None in metadata ---
        for prefix_label in ['adp_', 'nist_']:
            for base_field_name in nist_cvss_base_field_names:
                metadata[f'{prefix_label}{base_field_name}'] = None

        # --- Safeguard: Explicitly remove any BARE (unprefixed) NIST/CVSS fields ---
        # This step ensures that no unprefixed CVSS-specific keys remain in metadata.
        # General metadata fields (like 'title', 'id') are NOT in nist_cvss_base_field_names
        # and thus will not be removed by this loop.
        for base_field_to_remove in nist_cvss_base_field_names:
            if base_field_to_remove in metadata:
                del metadata[base_field_to_remove]

        # --- Build the final record with exactly 4 root-level keys ---
        # 'id' (original CVE ID) is expected to be in metadata if applicable.
        # 'id_' is the new unique document ID for this synthetic record.

        record_title = metadata.get('title', "Synthetic Vulnerability")
        record_description = metadata.get(
            'description', "No detailed synthetic description available."
        )
        # Construct the 'text' field for the root of the record
        record_text = f"{record_title}. {record_description} "
        record_text += fake.paragraph(nb_sentences=np.random.randint(3, 7))

        current_record = {
            "id_": fake.uuid4(),  # New unique ID for this synthetic document
            "text": record_text.strip(),
            "kb_ids": (
                []
            ),  # Remains empty for synthetic data as per current logic
            "metadata": (
                metadata
            ),  # The now correctly structured metadata dictionary
        }
        data.append(current_record)

    df = pd.DataFrame(data)
    # Ensure all expected columns are present, filling with None if missing
    # This part would need a comprehensive list of ALL final expected columns
    # For now, focus on what's generated.

    logger.debug(f"Successfully generated {len(df)} synthetic CVE records.")
    return df


def get_paragraphs_from_html(html_text: str) -> list[str]:
    """
    Splits HTML text into a list of paragraphs.
    This is a basic implementation and might need refinement for complex HTML.
    It assumes paragraphs are primarily defined by <p> tags or double line breaks
    if the input was Markdown converted to HTML without <p> tags for each line break.
    """
    if not html_text:
        return []
    # First, try to split by <p> tags if they exist and seem to be primary delimiters
    if "</p>" in html_text.lower():
        # A more robust way would use an HTML parser like BeautifulSoup
        # For a simpler approach here, we can use regex to capture content within <p>...</p>
        # and also text outside <p> tags that might be separated by double line breaks
        # This regex is an attempt to capture <p> content or non-empty lines between <p> blocks
        paragraphs = re.findall(
            r"<p.*?>(.*?)</p>|([^\s<].*?(?=\n\n|<p|$))",
            html_text,
            re.DOTALL | re.IGNORECASE,
        )
        # The regex returns tuples, need to filter and join
        cleaned_paragraphs = []
        for p_content, other_content in paragraphs:
            if p_content.strip():
                cleaned_paragraphs.append(
                    f"<p>{p_content.strip()}</p>"
                )  # Keep the <p> tags
            elif other_content.strip():
                # If it's not in a <p> tag but looks like a paragraph, wrap it
                cleaned_paragraphs.append(f"<p>{other_content.strip()}</p>")

        if cleaned_paragraphs:
            return cleaned_paragraphs

    # Fallback: If no <p> tags are clear delimiters, split by double newlines then wrap
    # This assumes the HTML might be from Markdown where \n\n became <br><br> or similar
    # or just plain text with double newlines.
    raw_paragraphs = re.split(
        r'\n\s*\n', html_text.strip()
    )  # Split by blank lines
    return [f"<p>{p.strip()}</p>" for p in raw_paragraphs if p.strip()]
