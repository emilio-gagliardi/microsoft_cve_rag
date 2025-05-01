import markdownify
import os
import shutil
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px  # Often useful even if using go
from pathlib import Path
from faker import Faker  # For synthetic data
from jinja2 import Environment as JinjaEnvironment  # Type hint
from collections import defaultdict
import logging

try:
    from application.app_utils import REPORTS_DIR, APP_DIR, DATA_DIR
    TEMPLATES_BASE_DIR = DATA_DIR / "templates"
except ImportError:
    logging.warning("Unable to import app_utils. Using relative paths.")
    APP_DIR = Path("microsoft_cve_rag/application")
    REPORTS_DIR = APP_DIR / "reports"
    TEMPLATES_BASE_DIR = DATA_DIR / "templates"
from application.etl.NVDDataExtractor import NVDDataExtractor
from application.services.chat_service import LLMClient

get_nvd_columns = NVDDataExtractor.get_all_possible_columns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReportConfig(BaseModel):
    """
    Configuration defining relative paths and names for a specific report type.
    Provides methods to resolve these into absolute paths based on global base directories.
    """
    report_name: str = Field(
        default="quarterly_deep_dive",
        description="Unique identifier and subdirectory name for this report type under the global reports directory."
    )
    report_title: str = Field(
        default="Quarterly Microsoft CVE Deep Dive",
        description="Display title for the report."
    )

    # --- Relative Directory Names (within the specific report's output) ---
    output_html_subdir: str = Field(
        default="html",
        description="Subdirectory name within the report's base output directory for HTML."
    )
    output_data_subdir: str = Field(
        default="data",
        description="Subdirectory name within the report's base output directory for storing chart JSON data."
    )
    output_css_subdir: str = Field(
        default="css",
        description="Subdirectory name within the report's base output directory for CSS files."
    )
    output_js_subdir: str = Field(
        default="js",
        description="Subdirectory name within the report's base output directory for JavaScript files."
    )
    output_image_subdir: str = Field(
        default="images",
        description="Subdirectory name within the report's base output directory for static images (if any)."
    )
    output_markdown_subdir: str = Field(
        default="markdown",
        description="Subdirectory name within the report's base output directory for Markdown files."
    )
    # --- Relative Template Directory Name (within the global templates base) ---
    template_subdir: str = Field(
        default="quarterly_deep_dive",
        description="Subdirectory name within the global application templates directory containing Jinja templates for this report."
    )
    template_html_filename: str = Field(
        default="report.html",
        description="Filename for the main Jinja HTML template file."
    )
    # --- Output Filenames ---
    output_css_filename: str = Field(
        default="QuarterlyReportStylesheet.css",
        description="Filename for the primary CSS stylesheet."
    )
    output_js_filename: str = Field(
        default="QuarterlyReport.js",
        description="Filename for the primary JavaScript file."
    )

    # --- Methods to Resolve Absolute Paths ---

    def get_report_base_dir(self, global_reports_dir: Path) -> Path:
        """Returns the absolute path to the base directory for this specific report instance."""
        if not global_reports_dir or not isinstance(global_reports_dir, Path):
            raise ValueError("global_reports_dir must be a valid Path object.")
        return global_reports_dir / self.report_name

    def get_html_output_dir(self, global_reports_dir: Path) -> Path:
        """Returns the absolute path to the HTML assets directory for this report instance."""
        return self.get_report_base_dir(global_reports_dir) / self.output_html_subdir

    def get_data_output_dir(self, global_reports_dir: Path) -> Path:
        """Returns the absolute path to the chart data directory."""
        return self.get_html_output_dir(global_reports_dir) / self.output_data_subdir

    def get_css_output_dir(self, global_reports_dir: Path) -> Path:
        """Returns the absolute path to the CSS directory."""
        return self.get_html_output_dir(global_reports_dir) / self.output_css_subdir

    def get_js_output_dir(self, global_reports_dir: Path) -> Path:
        """Returns the absolute path to the JS directory."""
        return self.get_html_output_dir(global_reports_dir) / self.output_js_subdir

    def get_image_output_dir(self, global_reports_dir: Path) -> Path:
        """Returns the absolute path to the image directory."""
        return self.get_html_output_dir(global_reports_dir) / self.output_image_subdir

    def get_markdown_output_dir(self, global_reports_dir: Path) -> Path:
        """Returns the absolute path to the Markdown output directory."""
        return self.get_report_base_dir(global_reports_dir) / self.output_markdown_subdir

    def get_css_filepath(self, global_reports_dir: Path) -> Path:
        """Returns the absolute path to the output CSS file."""
        return self.get_css_output_dir(global_reports_dir) / self.output_css_filename

    def get_js_filepath(self, global_reports_dir: Path) -> Path:
        """Returns the absolute path to the output JS file."""
        return self.get_js_output_dir(global_reports_dir) / self.output_js_filename

    def get_template_dir(self, global_templates_dir: Path) -> Path:
        """Returns the absolute path to the Jinja template directory for this report."""
        if not global_templates_dir or not isinstance(global_templates_dir, Path):
            raise ValueError("global_templates_dir must be a valid Path object.")
        return global_templates_dir / self.template_subdir

    def get_template_filepath(self, global_templates_dir: Path) -> Path:
        """Returns the absolute path to the main Jinja template file."""
        return self.get_template_dir(global_templates_dir) / self.template_html_filename


class GeneratedReportAssets(BaseModel):
    """Holds the paths to all assets generated for a single report run."""
    report_config: ReportConfig
    base_directory: Path  # The root directory for this specific report run
    html_file: Path
    markdown_file: Path
    css_file: Path
    js_file: Path
    chart_data_files: List[Path] = Field(default_factory=list)
    image_files: List[Path] = Field(default_factory=list)  # For future use

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
            *self.image_files
        ]
        return [f for f in all_files if f is not None]  # Filter out potential Nones


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
    stats: Dict[str, Any] = Field(default_factory=dict)
    llm_insights: Dict[str, str] = Field(default_factory=dict)
    appendix_tables: Dict[str, str] = Field(default_factory=dict)  # HTML strings


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
            },
            "toc": {
                "id_html": "table-of-contents",
                "heading": "Table of Contents",
                "prompts": {},
                "charts": [],
                "tables": [],
            },
            "report_conclusion": {
                "id_html": "report-conclusion",
                "heading": "Conclusion",
                "prompts": {
                    "report_conclusion": "report_conclusion.j2",
                },
                "charts": [],
                "tables": [],
            },
            "appendix": {
                "heading": "Appendix",
                "prompts": {},
                "charts": [],
                "tables": [
                    {
                        "id": "appendix_cve_list",
                        "data_prep_key": "appendix_cve_list",  # Key for data prep
                        "table_gen_key": "appendix_cve_list",  # Key for HTML gen
                        "caption": "Table A1. Analyzed CVE List",
                    },
                    {
                        "id": "appendix_outlier_list",
                        "data_prep_key": "appendix_outlier_list",  # Key for data prep
                        "table_gen_key": "appendix_outlier_list",  # Key for HTML gen
                        "caption": "Table A2. Significant Outliers",
                    }
                ]
            }
        },
        "looped_sections": {
            "section_3_vol_sev": {
                "id_html": "section-3-vol-sev",
                "heading": "Volume & Severity Trends",
                "prompts": {
                    "narrative": "section3_narrative.j2",
                    "callout_1": "section3_callout_1.j2",
                    "callout_2": "section3_callout_2.j2",
                    "chart_insight_1": "section3_chart_1_insight.j2",
                    "chart_insight_2": "section3_chart_2_insight.j2",
                    "section_summary": "section3_summary.j2",
                },
                "charts": [
                    {
                        "id": "new_vs_updated",
                        "data_prep_key": "new_vs_updated",
                        "figure_gen_key": "new_vs_updated",
                        "caption": "Figure 1. Total number of New vs. Updated Windows-based CVEs Published Monthly.",
                        "insight_prompt": "section3_chart_1_insight.j2",
                    },
                    {
                        "id": "volume_severity",
                        "data_prep_key": "volume_severity",
                        "figure_gen_key": "volume_severity",
                        "caption": "Figure 2. Monthly CVE Volume by Assessed Severity.",
                        "insight_prompt": "section3_chart_2_insight.j2",
                    },
                ],
                "tables": [],
            },
            "section_4_cvss_dist": {
                "id_html": "section-4-cvss-dist",
                "heading": "CVSS Score Distribution Analysis",
                "prompts": {
                    "narrative": "section4_narrative.j2",
                    "callout_1": "section4_callout_1.j2",
                    "callout_2": "section4_callout_2.j2",
                    "chart_insight_1": "section4_chart_1_insight.j2",
                    "chart_insight_2": "section4_chart_2_insight.j2",
                    "section_summary": "section4_summary.j2",
                },
                "charts": [
                    {
                        "id": "cvss_distribution",
                        "data_prep_key": "cvss_distribution",
                        "figure_gen_key": "cvss_distribution",
                        "caption": "Figure 3. CVSS Base Score Distribution by Vulnerability Category.",
                        "insight_prompt": "section4_chart_1_insight.j2",
                    }
                ],
                "tables": [],
            },
            "section_5_rce_deep_dive": {
                "id_html": "section-5-rce-deep-dive",
                "heading": "Remote Code Execution (RCE) Vulnerabilities",
                "prompts": {
                    "narrative": "section3_narrative.j2",
                    "callout_1": "section5_callout_1.j2",
                    "callout_2": "section5_callout_2.j2",
                    "chart_insight_1": "section5_chart_1_insight.j2",
                    "chart_insight_2": "section5_chart_2_insight.j2",
                    "section_summary": "section5_summary.j2",
                },
                "charts": [
                    {
                        "id": "rce_proportion_monthly",
                        "data_prep_key": "rce_proportion_monthly",
                        "figure_gen_key": "rce_proportion_monthly",
                        "caption": "Figure 4. Proportion of RCE Vulnerabilities Published Monthly.",
                        "insight_prompt": "section5_chart_1_insight.j2",
                    },
                    {
                        "id": "rce_cvss_distribution",
                        "data_prep_key": "rce_cvss_distribution",
                        "figure_gen_key": "rce_cvss_distribution",
                        "caption": "Figure 5. CVSS Base Score Distribution for RCE Vulnerabilities.",
                        "insight_prompt": "section5_chart_2_insight.j2",
                    },
                ],
                "tables": [],
            },
            "section_6_eop_deep_dive": {
                "id_html": "section-6-eop-deep-dive",
                "heading": "Elevated Privilege (EOP) Vulnerabilities",
                "prompts": {
                    "narrative": "section3_narrative.j2",
                    "callout_1": "section6_callout_1.j2",
                    "callout_2": "section6_callout_2.j2",
                    "chart_insight_1": "section6_chart_1_insight.j2",
                    "chart_insight_2": "section6_chart_2_insight.j2",
                    "section_summary": "section6_summary.j2",
                },
                "charts": [
                    {
                        "id": "eop_proportion_monthly",
                        "data_prep_key": "eop_proportion_monthly",
                        "figure_gen_key": "eop_proportion_monthly",
                        "caption": "Figure 6. Proportion of EOP Vulnerabilities Published Monthly.",
                        "insight_prompt": "section6_chart_1_insight.j2",
                    },
                    {
                        "id": "eop_by_attack_vector",
                        "data_prep_key": "eop_by_attack_vector",
                        "figure_gen_key": "eop_by_attack_vector",
                        "caption": "Figure 7. Distribution of Attack Vectors for EoP Vulnerabilities.",
                        "insight_prompt": "section6_chart_2_insight.j2",
                    },
                ],
                "tables": [],
            },
            "section_7_dos_infodisc_deep_dive": {
                "id_html": "section-7-dos-infodisc-deep-dive",
                "heading": "Denial of Service (DoS) and Information Disclosure (InfoDisc) Vulnerabilities",
                "prompts": {
                    "narrative": "section7_narrative.j2",
                    "callout_1": "section7_callout_1.j2",
                    "chart_insight_1": "section7_chart_1_insight.j2",
                    "section_summary": "section7_summary.j2",
                },
                "charts": [
                    {
                        "id": "dos_infodisc_monthly",
                        "data_prep_key": "dos_infodisc_monthly",
                        "figure_gen_key": "dos_infodisc_monthly",
                        "caption": "Figure 8. Monthly Volume of Denial of Service and Information Disclosure Vulnerabilities.",
                        "insight_prompt": "section7_chart_1_insight.j2",
                    },
                ],
                "tables": [],
            },
            "section_8_user_interaction_deep_dive": {
                "id_html": "section-8-user-interaction-deep-dive",
                "heading": "User Interaction Required vs. Direct Exploit",
                "prompts": {
                    "narrative": "section8_narrative.j2",
                    "callout_1": "section8_callout_1.j2",
                    "chart_insight_1": "section8_chart_1_insight.j2",
                    "section_summary": "section8_summary.j2",
                },
                "charts": [
                    {
                        "id": "user_interaction_proportion_monthly",
                        "data_prep_key": "user_interaction_proportion_monthly",
                        "figure_gen_key": "user_interaction_proportion_monthly",
                        "caption": "Figure 9. Monthly Proportion of Vulnerabilities Requiring User Interaction.",
                        "insight_prompt": "section8_chart_1_insight.j2",
                    },
                ],
                "tables": [],
            },
            "section_9_attack_vector_deep_dive": {
                "id_html": "section-9-attack-vector-deep-dive",
                "heading": "A Deeper Look at Attack Vectors",
                "prompts": {
                    "narrative": "section9_narrative.j2",
                    "callout_1": "section9_callout_1.j2",
                    "callout_2": "section9_callout_2.j2",
                    "chart_insight_1": "section9_chart_1_insight.j2",
                    "chart_insight_2": "section9_chart_2_insight.j2",
                    "section_summary": "section9_summary.j2",
                },
                "charts": [
                    {
                        "id": "overall_attack_vector_dist",
                        "data_prep_key": "overall_attack_vector_dist",
                        "figure_gen_key": "overall_attack_vector_dist",
                        "caption": "Figure 10. Overall Distribution of Attack Vectors for All Analyzed CVEs.",
                        "insight_prompt": "section9_chart_1_insight.j2",
                    },
                    {
                        "id": "overall_attack_complexity_dist",
                        "data_prep_key": "overall_attack_complexity_dist",
                        "figure_gen_key": "overall_attack_complexity_dist",
                        "caption": "Figure 11. Overall Distribution of Attack Complexities for All Analyzed CVEs.",
                        "insight_prompt": "section9_chart_2_insight.j2",
                    },
                ],
                "tables": [],
            },
            "section_10_patch_cdf_deep_dive": {
                "id_html": "section-10-patch-cdf-deep-dive",
                "heading": "Deep Dive: Patching Timeliness (Days to NVD Publication)",
                "prompts": {
                    "narrative": "section10_narrative.j2",
                    "callout_1": "section10_callout_1.j2",
                    "callout_2": "section10_callout_2.j2",
                    "chart_insight_1": "section10_chart_1_insight.j2",
                    "chart_insight_2": "section10_chart_2_insight.j2",
                    "section_summary": "section10_summary.j2",
                },
                "charts": [
                    {
                        "id": "patch_delay_risk_scatter",
                        "data_prep_key": "patch_delay_risk_scatter",
                        "figure_gen_key": "patch_delay_risk_scatter",
                        "caption": "Figure 12. Patch Delay vs. CVSS Score, Highlighting High-Risk Exposure.",
                        "insight_prompt": "section10_chart_1_insight.j2",
                    },
                ],
                "tables": [],
            }
        }
    }

    LLM_TASK_MODELS = {
        "narrative": "gpt-4.1-mini",  # cheap | fast | middle smart @ $0.40/1M tokens - 1M context window
        "callout": "gpt-4o-mini",  # cheap | fast | low smart @ $0.15/1M tokens - 128K context window
        "chart_insight": "gpt-4o-mini",  # cheap | fast | smart @ $0.40/1M tokens - 1M context window
        "section_summary": "gpt-o3-mini",  # expensive | slow | very smart @ $1.10/1M tokens - 200K context window
        "executive_summary": "gpt-o3-mini",  # expensive | slow | very smart @ $1.10/1M tokens - 200K context window
        "report_conclusion": "gpt-o3-mini",  # expensive | slow | very smart @ $1.10/1M tokens - 200K context window
        "default": "gpt-4.1-mini"  # cheap | fast | middle smart @ $0.40/1M tokens - 1M context window
    }

    SYSTEM_PROMPTS = {
        "chart_insight": (
            "You are a concise data analyst. Your task is to describe the key findings, trends, or patterns visible in the provided chart data summary. "
            "Focus ONLY on what the data shows (e.g., highs, lows, trends, distributions, comparisons). "
            "Do NOT add external context, interpretations, or recommendations. Be objective and brief (2-3 sentences max)."
        ),
        "callout": (
            "You are a security analyst highlighting a noteworthy observation. Based on the provided context (section heading, chart insights, chart data summaries), identify ONE specific, interesting, or potentially actionable finding relevant to the section's topic. "
            "This might be an anomaly, a significant trend, or a point requiring attention. State the observation clearly and concisely (1-2 sentences)."
        ),
        "narrative": (
            "You are a security report writer. Your task is to generate a descriptive narrative for a report section. "
            "Use the provided section heading, chart insights, callouts, and chart data summaries to explain the key findings and trends for this topic. "
            "Synthesize the information logically. Maintain an objective and analytical tone. Aim for clear, informative paragraphs. Do not introduce recommendations unless explicitly asked."
            "Ensure the use of hierarchical structure (e.g., nested headings, bullet points, numbered lists) to organize the narrative."
            "The target audience is expert level system engineers, use precise and complete technical language. Do not use metaphors to explain basic topics."
            "Use active voice and present tense. Vary the length of sentences, their complexity, and sentence structure to increase dynamism and clarity."
        ),
        "section_summary": (
            "You are a security analyst summarizing report findings. Based on the provided section heading, chart insights, callouts, and narrative, write a brief summary (2-4 sentences) capturing the main conclusions and most important takeaways for this section. "
            "Focus on the key results and their implications, if clear."
            "Ensure the use of hierarchical structure (e.g., nested headings, bullet points, numbered lists) to organize the summary."
            "The target audience is expert level system engineers, use precise and complete technical language. Do not use metaphors to explain basic topics."
            "Use active voice and present tense. Vary the length of sentences, their complexity, and sentence structure to increase dynamism and clarity."
        ),
        "executive_summary": (
            "You are a senior security analyst writing an executive summary for a technical report. "
            "Review the provided overall report statistics and the content summaries from individual sections. "
            "Synthesize the most critical findings, trends, and potential areas of concern across the entire report period. "
            "Focus on high-level takeaways relevant to management and strategic decision-making. Keep it concise and impactful."
            "Ensure the use of hierarchical structure (e.g., nested headings, bullet points, numbered lists) to organize the summary."
            "The target audience is management level, use precise and complete language. Do not use metaphors to explain basic topics."
            "Use active voice and present tense. Vary the length of sentences, their complexity, and sentence structure to increase dynamism and clarity."
        ),
        "report_conclusion": (
             "You are a senior security analyst writing the conclusion for a technical report. "
             "Review the provided overall report statistics and the content from all previous sections (including the executive summary). "
             "Summarize the main themes and findings discussed throughout the report. Briefly reiterate the most significant observations or trends. "
             "You may optionally suggest very high-level areas for future observation based *only* on the data presented. Do not provide specific recommendations."
             "The target audience is management level, use precise and complete language. Do not use metaphors to explain basic topics."
             "Use active voice and present tense. Vary the length of sentences, their complexity, and sentence structure to increase dynamism and clarity."
        ),
        "default": (  # Fallback if type cannot be determined
            "You are a helpful assistant analyzing security vulnerability data. Provide a clear and relevant response based on the user's prompt."
        )
    }

    def __init__(
            self,
            jinja_env: JinjaEnvironment,
            llm_client: Optional[LLMClient] = None,
            global_reports_dir: Optional[Path] = None,
            global_templates_dir: Optional[Path] = None
    ):
        """Initializes the generator with essential dependencies."""
        if not global_reports_dir or not global_reports_dir.is_dir():
            raise ValueError(f"global_reports_dir ('{global_reports_dir}') must be a valid Path object and directory.")
        if not global_templates_dir or not global_templates_dir.is_dir():
            raise ValueError(f"global_templates_dir ('{global_templates_dir}') must be a valid Path object and directory.")
        # Ensure APP_DIR is also initialized before using it here
        if APP_DIR is None or not APP_DIR.is_dir():
            raise ValueError(f"APP_DIR ('{APP_DIR}') is not initialized or not a valid directory.")
        self.start_date = None
        self.end_date = None
        self.jinja_env = jinja_env
        self.llm = llm_client
        self.reports_base_dir = global_reports_dir
        self.templates_base_dir = global_templates_dir
        self.source_static_dir = global_templates_dir / "static" / "dist"
        self.severity_color_map = {
            'Low': '#A0C722',        # Lime
            'Moderate': '#F5B000',   # Yellow/Orange
            'Important': '#EB8B06',  # Report Main Orange
            'Critical': '#D84749',   # Report Dark Accent Red
            'None': '#B9A696',       # Report Light Accent
            'Unknown': '#6B7280'     # Gray
        }
        self.category_color_map = {
            'remote_code_execution': '#D84749',    # Red
            'privilege_elevation': '#EB8B06',      # Orange
            'security_feature_bypass': '#DF0D5F',  # Pink
            'denial_of_service': '#22A5DD',        # Blue
            'information_disclosure': '#67C2C0',   # Teal
            'spoofing': '#A0C722',                 # Lime
            'tampering': '#8B5CF6',                # Purple
            'other': '#B9A696'                     # Light Accent
        }
        # Define orderings
        self.severity_order = ['Low', 'Moderate', 'Important', 'Critical', 'Unknown', 'None']
        self.category_order = [
            'remote_code_execution', 'privilege_elevation', 'security_feature_bypass',
            'denial_of_service', 'information_disclosure', 'spoofing', 'tampering', 'other'
        ]
        self.attack_vector_order = ['Network', 'Adjacent', 'Local', 'Physical', 'Unknown']
        self.attack_complexity_order = ['Low', 'High']
        self.user_interaction_order = ['None', 'Required', 'Not Specified']
        self._create_plotly_themes()
        logger.info(f"Initialized QuarterlyDeepDiveReportGenerator with base_dir: {self.reports_base_dir}")

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
            'grid_dark': '#4b5563',   # Tailwind gray-600
            # Define semantic colors if needed for specific traces
            'info': '#22A5DD',
            'success': '#00A33E',
            'warning': '#EB8B06',  # report-main
            'danger': '#D84749',   # report-dark-accent
            # Add more core colors for sequences if needed
            'color_sequence': ['#EB8B06', '#D84749', '#B9A696', '#22A5DD', '#00A33E', '#A0C722']  # Main, Dark Accent, Light Accent, Brand Blue, Brand Green, Brand Lime
        }
        # --- Light Theme Template ---
        plotly_template_light = go.layout.Template(
            layout=go.Layout(
                font=dict(family="Inter, sans-serif", size=12, color=report_colors['text_light']),
                title=dict(font=dict(size=18, color=report_colors['text_light']), x=0.05),  # Title left-aligned
                paper_bgcolor=report_colors['surface_light'],  # Chart background
                plot_bgcolor=report_colors['surface_light'],   # Plot area background
                xaxis=dict(
                    gridcolor=report_colors['grid_light'],
                    linecolor=report_colors['border_light'],
                    zerolinecolor=report_colors['grid_light'],
                    tickfont=dict(color=report_colors['text_light']),
                    title=dict(font=dict(color=report_colors['text_light']))
                ),
                yaxis=dict(
                    gridcolor=report_colors['grid_light'],
                    linecolor=report_colors['border_light'],
                    zerolinecolor=report_colors['grid_light'],
                    tickfont=dict(color=report_colors['text_light']),
                    title=dict(font=dict(color=report_colors['text_light']))
                ),
                legend=dict(
                    bgcolor='rgba(255,255,255,0.7)',  # Slightly transparent background
                    bordercolor=report_colors['border_light'],
                    font=dict(color=report_colors['text_light'])
                ),
                colorway=report_colors['color_sequence'],  # Default color sequence for traces
                margin=dict(l=50, r=50, t=80, b=50),
            )
        )

        # --- Dark Theme Template ---
        plotly_template_dark = go.layout.Template(
            layout=go.Layout(
                font=dict(family="Inter, sans-serif", size=12, color=report_colors['text_dark']),
                title=dict(font=dict(size=18, color=report_colors['text_dark']), x=0.05),
                paper_bgcolor=report_colors['surface_dark'],
                plot_bgcolor=report_colors['surface_dark'],
                xaxis=dict(
                    gridcolor=report_colors['grid_dark'],
                    linecolor=report_colors['border_dark'],
                    zerolinecolor=report_colors['grid_dark'],
                    tickfont=dict(color=report_colors['text_dark']),
                    title=dict(font=dict(color=report_colors['text_dark']))
                ),
                yaxis=dict(
                    gridcolor=report_colors['grid_dark'],
                    linecolor=report_colors['border_dark'],
                    zerolinecolor=report_colors['grid_dark'],
                    tickfont=dict(color=report_colors['text_dark']),
                    title=dict(font=dict(color=report_colors['text_dark']))
                ),
                legend=dict(
                    bgcolor='rgba(42, 49, 64, 0.7)',  # Slightly transparent dark background
                    bordercolor=report_colors['border_dark'],
                    font=dict(color=report_colors['text_dark'])
                ),
                colorway=report_colors['color_sequence'],
                margin=dict(l=50, r=50, t=80, b=50),
            )
        )

        # --- Register the templates ---
        pio.templates['report_light'] = plotly_template_light
        pio.templates['report_dark'] = plotly_template_dark
        pio.templates.default = 'report_light+plotly_white'  # Set default to light + some plotly defaults

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

    @staticmethod
    def _extract_metadata_columns(
        df: pd.DataFrame,
        metadata_col: str,
        keys: List[str]
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
        score_cols = ['cna_base_score_num', 'adp_base_score_num', 'nist_base_score_num']
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
        complexity_cols = [col for col in df.columns if '_attack_complexity' in col]

        def highest_complexity(row) -> str | None:
            values = [str(row[col]).lower() if pd.notna(row[col]) else None for col in complexity_cols]
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
        precedence = {
            'network': 4,
            'adjacent network': 3,
            'local': 2,
            'physical': 1
        }
        reverse_precedence = {v: k for k, v in precedence.items()}

        def highest_vector(row) -> str | None:
            values = [str(row[col]).lower() if pd.notna(row[col]) else None for col in vector_cols]
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
        priv_cols = [col for col in df.columns if '_privileges_required' in col]
        precedence = {
            'high': 2,
            'low': 1
        }
        reverse_precedence = {v: k for k, v in precedence.items()}

        def highest_priv(row) -> str | None:
            values = [str(row[col]).lower() if pd.notna(row[col]) else None for col in priv_cols]
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
        precedence = {
            'required': 2,
            'none': 1
        }
        reverse_precedence = {v: k.title() for k, v in precedence.items()}

        def highest_ui(row) -> str | None:
            values = [str(row[col]).lower() if pd.notna(row[col]) else None for col in ui_cols]
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
        conf_cols = [col for col in df.columns if '_confidentiality_impact' in col]
        precedence = {
            'high': 3,
            'low': 2,
            'none': 1
        }
        reverse_precedence = {v: k for k, v in precedence.items()}

        def highest_conf(row) -> str | None:
            values = [str(row[col]).lower() if pd.notna(row[col]) else None for col in conf_cols]
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
        integ_cols = [col for col in df.columns if '_integrity_impact' in col]
        precedence = {
            'high': 3,
            'low': 2,
            'none': 1
        }
        reverse_precedence = {v: k for k, v in precedence.items()}

        def highest_integ(row) -> str | None:
            values = [str(row[col]).lower() if pd.notna(row[col]) else None for col in integ_cols]
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
        avail_cols = [col for col in df.columns if '_availability_impact' in col]
        precedence = {
            'high': 3,
            'low': 2,
            'none': 1
        }
        reverse_precedence = {v: k for k, v in precedence.items()}

        def highest_avail(row) -> str | None:
            values = [str(row[col]).lower() if pd.notna(row[col]) else None for col in avail_cols]
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
        expl_cols = [col for col in df.columns if '_exploitability_score' in col]

        def max_exploit(row) -> float | None:
            values = [row[col] for col in expl_cols if pd.notna(row[col])]
            if values:
                return float(max(values))
            return None

        return df.apply(max_exploit, axis=1)

    def _prepare_input_dataframe(self, report_data: pd.DataFrame) -> pd.DataFrame:
        """Validates and prepares the input DataFrame."""
        logger.info("Preparing input DataFrame...")
        if not isinstance(report_data, pd.DataFrame):
            raise TypeError("report_data must be a Pandas DataFrame.")
        if report_data.empty:
            logger.info("Warning: Input DataFrame is empty.")
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
        ]
        # flatten the metadata Object into columns
        all_metadata_keys = nvd_meta_cols + document_meta_cols
        df = QuarterlyDeepDiveReportGenerator._extract_metadata_columns(df, 'metadata', all_metadata_keys)
        df = df.drop(columns=['metadata'])
        document_root_cols = ['id_', 'text', 'kb_ids']
        required_cols = nvd_meta_cols + document_meta_cols + document_root_cols
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")

        # --- Type Conversions & Derived Columns ---
        # Handle Revision for New/Updated Status
        def get_status(revision_str):
            if pd.isna(revision_str):
                return 'Unknown'
            try:
                # Extract the major version number before the decimal
                major_version = float(str(revision_str).split('.')[0])
                return 'New' if major_version == 1.0 else 'Updated'
            except (ValueError, IndexError):
                return 'Unknown'  # Handle parsing errors

        if 'revision' in df.columns:
            df['status'] = df['revision'].apply(get_status)
        else:
            df['status'] = 'Unknown'  # Default if column is missing

        # Handle NVD Published Date as Patch Date Proxy
        if 'nvd_published_date' in df.columns:
            df['patch_date'] = pd.to_datetime(df['nvd_published_date'], errors='coerce')
        else:
            df['patch_date'] = pd.NaT  # No proxy available

        # Handle CVE Category Standardization
        category_map = {
            "privilege_elevation": "privilege_elevation",
            "spoofing": "spoofing",
            "remote_code_execution": "remote_code_execution",
            "tampering": "tampering",  # Might need color/order definition
            "disclosure": "information_disclosure",  # Map to standard name
            "feature_bypass": "security_feature_bypass",  # Map to standard name
            "denial_of_service": "denial_of_service",
            # Handle null, "NC", or any other unexpected values
            pd.NA: 'other',
            None: 'other',
            "NC": 'other',
            "": 'other'
        }
        if 'cve_category' in df.columns:
            # Apply mapping, fill remaining NaNs/unmapped with 'other'
            df['cve_category'] = df['cve_category'].map(category_map).fillna('NC')
        else:
            df['cve_category'] = 'other'  # Default if column is missing

        # Handle User Interaction (Keep strings, clean NAs)
        if 'cna_user_interaction' in df.columns:
            df['user_interaction'] = df['cna_user_interaction'].fillna('None')
            # Ensure values match the order list: 'None', 'Required', 'Not Specified'
            # Correct potential case mismatches if necessary, e.g. .str.title()
            df['user_interaction'] = df['user_interaction'].replace({'NONE': 'None', 'REQUIRED': 'Required'})  # Example correction
        else:
            df['user_interaction'] = 'None'

        # --- Derived Columns ---
        if 'published' in df.columns and pd.api.types.is_datetime64_any_dtype(df['published']):
            df['published_month_dt'] = df['published'].dt.to_period('M')
            df['published_month'] = df['published_month_dt'].astype(str)  # String for grouping/axis
        else:
            # Create placeholder columns if 'published' is missing or invalid
            df['published_month_dt'] = pd.NaT
            df['published_month'] = 'Unknown'

        df['cvss_score'] = QuarterlyDeepDiveReportGenerator._extract_highest_cvss_score(df)
        df['attack_vector'] = QuarterlyDeepDiveReportGenerator._extract_attack_vector(df)
        df['attack_complexity'] = QuarterlyDeepDiveReportGenerator._extract_attack_complexity(df)
        df['privileges_required'] = QuarterlyDeepDiveReportGenerator._extract_privileges_required(df)
        df['user_interaction'] = QuarterlyDeepDiveReportGenerator._extract_user_interaction(df)
        df['confidentiality_impact'] = QuarterlyDeepDiveReportGenerator._extract_confidentiality_impact(df)
        df['integrity_impact'] = QuarterlyDeepDiveReportGenerator._extract_integrity_impact(df)
        df['availability_impact'] = QuarterlyDeepDiveReportGenerator._extract_availability_impact(df)
        df['impact_score'] = QuarterlyDeepDiveReportGenerator._extract_impact_score(df)
        df['exploitability_score'] = QuarterlyDeepDiveReportGenerator._extract_exploitability_score(df)

        logger.info("Input DataFrame prepared successfully.")
        return df

    def _prepare_chart_data(self, df: pd.DataFrame, chart_key: str) -> Any:
        """Generates data formatted for a specific chart type."""
        logger.info(f"Preparing data for chart: {chart_key}")
        if df.empty:
            logger.warning(f"DataFrame is empty for chart '{chart_key}'.")
            return None

        try:
            # --- Chart Specific Logic ---
            if chart_key == "new_vs_updated":
                required_cols = ['published_month', 'status']
                if not all(c in df.columns and df[c].notna().any() for c in required_cols):
                    logger.warning(f"Missing or empty required columns {required_cols} for '{chart_key}'.")
                    return None
                # Ensure 'status' is treated as categorical if ordering matters
                df['status'] = pd.Categorical(df['status'], categories=['New', 'Updated', 'Unknown'], ordered=False)
                agg_data = df.groupby(['published_month', 'status'], observed=False).size().reset_index(name='count')
                return agg_data.sort_values('published_month')

            elif chart_key == "volume_severity":
                required_cols = ['published_month', 'severity_type']
                if not all(c in df.columns and df[c].notna().any() for c in required_cols):
                    logger.warning(f"Missing or empty required columns {required_cols} for '{chart_key}'.")
                    return None
                # Ensure severity is treated as categorical for consistent ordering using self.severity_order
                df['severity_type'] = pd.Categorical(df['severity_type'].fillna('NST'), categories=self.severity_order, ordered=True)
                agg_data = df.groupby(['published_month', 'severity_type'], observed=False).size().reset_index(name='count')
                return agg_data.sort_values(['published_month', 'severity_type'])

            elif chart_key == "cvss_distribution":
                required_cols = ['cve_category', 'cvss_score']
                if not all(c in df.columns and df[c].notna().any() for c in required_cols):
                    logger.warning(f"Missing or empty required columns {required_cols} for '{chart_key}'.")
                    return None
                # Ensure category is categorical using self.category_order
                df['cve_category'] = pd.Categorical(df['cve_category'].fillna('other'), categories=self.category_order, ordered=False)
                chart_data = df[['cve_category', 'cvss_score']].dropna(subset=['cvss_score'])
                if chart_data.empty:
                    logger.warning(f"No valid CVSS scores found for '{chart_key}'.")
                    return None
                return chart_data

            elif chart_key == "rce_proportion_monthly":
                required_cols = ['published_month', 'cve_category']
                if not all(c in df.columns for c in required_cols):  # Allow empty category column if no RCE exist
                    logger.warning(f"Missing required columns {required_cols} for '{chart_key}'.")
                    return None
                total_monthly = df.groupby('published_month', observed=False).size().rename('total_count')
                # Use the standardized category name
                rce_monthly = df[df['cve_category'] == 'remote_code_execution'].groupby('published_month', observed=False).size().rename('rce_count')
                agg_data = pd.merge(total_monthly, rce_monthly, on='published_month', how='left').fillna(0)
                agg_data['proportion'] = (agg_data['rce_count'] / agg_data['total_count'].replace(0, np.nan) * 100).fillna(0)
                return agg_data.reset_index().sort_values('published_month')

            elif chart_key == "rce_cvss_distribution":
                required_cols = ['cve_category', 'cvss_score']
                if not all(c in df.columns for c in required_cols):
                    logger.warning(f"Missing required columns {required_cols} for '{chart_key}'.")
                    return None
                # Use the standardized category name
                chart_data = df[df['cve_category'] == 'remote_code_execution'][['cvss_score']].dropna()
                if chart_data.empty:
                    logger.warning(f"No valid RCE CVSS scores found for '{chart_key}'.")
                    return None
                # Add a column for plotting (violin needs an x-axis category)
                chart_data['category_label'] = 'Remote Code Execution'
                return chart_data

            elif chart_key == "eop_proportion_monthly":
                required_cols = ['published_month', 'cve_category']
                if not all(c in df.columns for c in required_cols):
                    logger.warning(f"Missing required columns {required_cols} for '{chart_key}'.")
                    return None
                total_monthly = df.groupby('published_month', observed=False).size().rename('total_count')
                # Use the standardized category name
                eop_monthly = df[df['cve_category'] == 'privilege_elevation'].groupby('published_month', observed=False).size().rename('eop_count')
                agg_data = pd.merge(total_monthly, eop_monthly, on='published_month', how='left').fillna(0)
                agg_data['proportion'] = (agg_data['eop_count'] / agg_data['total_count'].replace(0, np.nan) * 100).fillna(0)
                return agg_data.reset_index().sort_values('published_month')

            elif chart_key == "eop_by_attack_vector":
                required_cols = ['cve_category', 'attack_vector']
                if not all(c in df.columns for c in required_cols):
                    logger.warning(f"Missing required columns {required_cols} for '{chart_key}'.")
                    return None
                # Use the standardized category name
                df_eop = df[df['cve_category'] == 'privilege_elevation'].copy()
                if df_eop.empty:
                    logger.warning(f"No EoP CVEs found for '{chart_key}'.")
                    return None
                # Use standardized attack_vector column and order
                df_eop['attack_vector'] = pd.Categorical(df_eop['attack_vector'].fillna('None'), categories=self.attack_vector_order, ordered=True)
                agg_data = df_eop.groupby('attack_vector', observed=False).size().reset_index(name='count')
                # Ensure all categories from the order are present, even if count is 0
                agg_data = agg_data.set_index('attack_vector').reindex(self.attack_vector_order, fill_value=0).reset_index()
                return agg_data.sort_values('attack_vector')

            elif chart_key == "dos_infodisc_monthly":  # Renamed key
                required_cols = ['published_month', 'cve_category']
                # Use standardized category names
                categories_to_plot = ['denial_of_service', 'information_disclosure']
                if not all(c in df.columns for c in required_cols):
                    logger.warning(f"Missing required columns {required_cols} for '{chart_key}'.")
                    return None
                df_filtered = df[df['cve_category'].isin(categories_to_plot)].copy()
                if df_filtered.empty:
                    logger.warning(f"No DoS or Info Disclosure CVEs found for '{chart_key}'.")
                    return None
                # Optional: Map category names for display if different from standard internal names
                df_filtered['category_display'] = df_filtered['cve_category'].replace({
                    'denial_of_service': 'DoS', 'information_disclosure': 'Info Disclosure'
                })
                agg_data = df_filtered.groupby(['published_month', 'category_display'], observed=False).size().reset_index(name='count')
                return agg_data.sort_values(['published_month', 'category_display'])

            elif chart_key == "user_interaction_proportion_monthly":  # NEW LOGIC
                required_cols = ['published_month', 'user_interaction']
                if not all(c in df.columns for c in required_cols):
                    logger.warning(f"Missing required columns {required_cols} for '{chart_key}'.")
                    return None
                # Calculate total counts per month
                total_monthly = df.groupby('published_month', observed=False).size().rename('total_count')
                # Calculate counts where interaction is 'Required'
                required_monthly = df[df['user_interaction'] == 'Required'].groupby('published_month', observed=False).size().rename('required_count')
                # Combine and calculate proportion
                agg_data = pd.merge(total_monthly, required_monthly, on='published_month', how='left').fillna(0)
                agg_data['proportion_required'] = (agg_data['required_count'] / agg_data['total_count'].replace(0, np.nan) * 100).fillna(0)
                return agg_data.reset_index().sort_values('published_month')

            elif chart_key == "overall_attack_vector_dist":   # Matched key
                required_cols = ['attack_vector']
                if not all(c in df.columns and df[c].notna().any() for c in required_cols):
                    logger.warning(f"Missing or empty required columns {required_cols} for '{chart_key}'.")
                    return None
                # Use standardized column
                df_filtered = df.dropna(subset=['attack_vector'])
                if df_filtered.empty: return None  # noqa: E701

                # Compute normalized value counts as a Series, reindex to ensure all categories are present
                vc = df_filtered['attack_vector'].value_counts(normalize=True)
                vc = vc.reindex(self.attack_vector_order, fill_value=0) * 100  # fill missing with 0, convert to percent
                agg_data = vc.reset_index()
                agg_data.columns = ['attack_vector', 'percentage']
                agg_data['attack_vector'] = pd.Categorical(agg_data['attack_vector'], categories=self.attack_vector_order, ordered=True)
                return agg_data.sort_values('attack_vector')

            elif chart_key == "overall_attack_complexity_dist":  # Matched key
                required_cols = ['attack_complexity']
                if not all(c in df.columns and df[c].notna().any() for c in required_cols):
                    logger.warning(f"Missing or empty required columns {required_cols} for '{chart_key}'.")
                    return None
                # Use standardized column
                df_filtered = df.dropna(subset=['attack_complexity'])
                if df_filtered.empty: return None  # noqa: E701

                vc = df_filtered['attack_complexity'].value_counts(normalize=True)
                vc = vc.reindex(self.attack_complexity_order, fill_value=0) * 100  # fill missing with 0, convert to percent
                agg_data = vc.reset_index()
                agg_data.columns = ['attack_complexity', 'percentage']
                agg_data['attack_complexity'] = pd.Categorical(agg_data['attack_complexity'], categories=self.attack_complexity_order, ordered=True)
                return agg_data.sort_values('attack_complexity')

            elif chart_key == "patch_delay_risk_scatter":
                # Use standardized columns: 'published', 'patch_date', 'cvss_score', 'cve_category', 'post_id'
                required_cols = ['published', 'patch_date', 'cvss_score', 'cve_category', 'post_id']
                if not all(c in df.columns for c in required_cols):
                    logger.warning(f"Missing some required columns ({required_cols}) for '{chart_key}'. Chart may be incomplete.")
                    # Create missing columns as needed with appropriate null types
                    if 'patch_date' not in df.columns: df['patch_date'] = pd.NaT  # noqa: E701
                    if 'cvss_score' not in df.columns: df['cvss_score'] = np.nan  # noqa: E701
                    if 'cve_category' not in df.columns: df['cve_category'] = 'NC'  # noqa: E701
                    if 'post_id' not in df.columns: df['post_id'] = 'Unknown'  # noqa: E701

                # Calculate days_to_patch (using the proxy date)
                # Ensure both are valid datetimes before subtraction
                valid_dates = df['patch_date'].notna() & df['published'].notna()
                df['days_to_patch'] = np.nan  # Initialize
                df.loc[valid_dates, 'days_to_patch'] = (df.loc[valid_dates, 'patch_date'] - df.loc[valid_dates, 'published']).dt.days
                df.loc[df['days_to_patch'] < 0, 'days_to_patch'] = 0  # Set negative diffs to 0

                # Prepare data for scatter plot
                chart_data = df[['post_id', 'days_to_patch', 'cvss_score', 'cve_category']].copy()
                # Drop rows where essential plotting data is missing
                chart_data.dropna(subset=['days_to_patch', 'cvss_score'], inplace=True)

                if chart_data.empty:
                    logger.warning(f"No valid data points found for '{chart_key}' after calculating patch delay and filtering.")
                    return None
                # Ensure category is categorical using self.category_order
                chart_data['cve_category'] = pd.Categorical(chart_data['cve_category'].fillna('other'), categories=self.category_order, ordered=False)
                return chart_data

            else:
                logger.warning(f"No data preparation logic defined for chart key '{chart_key}'.")
                return None

        except Exception as e:
            logger.exception(f"Error preparing data for chart '{chart_key}': {e}", exc_info=True)
            return None

    def _generate_plotly_figure(self, chart_data: Any, chart_key: str, chart_title: Optional[str] = None) -> Optional[go.Figure]:
        """Creates a Plotly figure object. Uses chart IDs as keys."""
        logger.info(f"Generating Plotly figure for: {chart_key}")
        if chart_data is None or (isinstance(chart_data, pd.DataFrame) and chart_data.empty):
            logger.warning(f"No data provided to generate figure for '{chart_key}'.")
            return None

        if chart_title is None:
            # Use the heading from REPORT_STRUCTURE if available, otherwise generate default
            for section in self.REPORT_STRUCTURE['looped_sections'].values():
                for chart_def in section.get('charts', []):
                    if chart_def['id'] == chart_key:
                        chart_title = chart_def.get('caption', chart_key.replace('_', ' ').title())
                        break
                else:
                    continue
                break
            if chart_title is None:
                chart_title = chart_key.replace('_', ' ').title()

        fig: go.Figure = None
        try:
            # --- Chart Specific Logic ---
            if chart_key == "new_vs_updated":
                fig = px.bar(
                    chart_data,
                    x='published_month',
                    y='count',
                    color='status',
                    title="New vs Updated CVEs Published Monthly",
                    labels={'published_month': 'Month', 'count': 'Number of CVEs', 'status': 'Status'},
                    barmode='group',
                    text_auto=True
                )
                fig.update_traces(textposition='outside')

            elif chart_key == "volume_severity":
                fig = px.bar(
                    chart_data,
                    x='published_month',
                    y='count',
                    color='severity_type',
                    title="Monthly CVE Volume by Assessed Severity",
                    labels={'published_month': 'Month', 'count': 'Number of CVEs', 'severity_type': 'Severity'},
                    category_orders={"severity_type": self.severity_order},
                    color_discrete_map=self.severity_color_map,
                    barmode='group',
                    text_auto=True
                )
                fig.update_traces(textposition='outside')

            elif chart_key == "cvss_distribution":
                fig = px.violin(
                    chart_data,
                    x='cve_category',
                    y='cvss_score',
                    title="CVSS Base Score Distribution by Vulnerability Category",
                    labels={'cve_category': 'Vulnerability Category', 'cvss_score': 'CVSS Base Score'},
                    category_orders={"cve_category": self.category_order},
                    color="cve_category",
                    color_discrete_map=self.category_color_map,
                    box=True,
                    points='outliers'
                )
                fig.update_layout(showlegend=False)

            elif chart_key == "rce_proportion_monthly":
                fig = px.line(
                    chart_data,
                    x='published_month',
                    y='proportion',
                    title="Proportion of RCE Vulnerabilities Published Monthly",
                    labels={'published_month': 'Month', 'proportion': '% of Total CVEs'},
                    markers=True
                )
                fig.update_layout(yaxis_range=[0, max(10, chart_data['proportion'].max() * 1.1)], yaxis_ticksuffix="%")  # Ensure range starts at 0 and has some height

            elif chart_key == "rce_cvss_distribution":
                fig = px.violin(
                    chart_data,
                    x='category_label',  # Use the added label column
                    y='cvss_score',
                    title="CVSS Base Score Distribution for RCE Vulnerabilities",
                    labels={'category_label': '', 'cvss_score': 'CVSS Base Score'},  # Use the added label
                    box=True, points='all',
                    # Use instance attribute color map, provide default if key missing
                    color_discrete_sequence=[self.category_color_map.get('remote_code_execution', '#D84749')]
                )
                fig.update_layout(xaxis_title=None)

            elif chart_key == "eop_proportion_monthly":
                fig = px.line(
                    chart_data,
                    x='published_month',
                    y='proportion',
                    title="Proportion of EoP Vulnerabilities Published Monthly",
                    labels={'published_month': 'Month', 'proportion': '% of Total CVEs'},
                    markers=True
                )
                fig.update_layout(yaxis_range=[0, max(10, chart_data['proportion'].max() * 1.1)], yaxis_ticksuffix="%")

            elif chart_key == "eop_by_attack_vector":
                fig = px.bar(
                    chart_data,
                    x='attack_vector',  # Use standardized column
                    y='count',
                    title="Distribution of Attack Vectors for EoP Vulnerabilities",
                    labels={'attack_vector': 'Attack Vector', 'count': 'Number of EoP CVEs'},
                    category_orders={"attack_vector": self.attack_vector_order},  # Use instance attribute
                    text_auto=True
                )
                # Use instance attribute color map, provide default if key missing
                fig.update_traces(marker_color=self.category_color_map.get('privilege_elevation', '#EB8B06'))

            elif chart_key == "dos_infodisc_monthly":  # Renamed key
                fig = px.bar(
                    chart_data,
                    x='published_month',
                    y='count',
                    color='category_display',  # Use display name column
                    title="Monthly Volume of DoS & Info Disclosure Vulnerabilities",
                    labels={'published_month': 'Month', 'count': 'Number of CVEs', 'category_display': 'Category'},
                    color_discrete_map={  # Use instance attribute color map for specific categories
                        'DoS': self.category_color_map.get('denial_of_service', '#22A5DD'),
                        'Info Disclosure': self.category_color_map.get('information_disclosure', '#67C2C0')
                    },
                    barmode='group', text_auto=True
                )
                fig.update_traces(textposition='outside')

            elif chart_key == "user_interaction_proportion_monthly":  # NEW CHART TYPE
                fig = px.line(
                    chart_data,
                    x='published_month',
                    y='proportion_required',  # Use the calculated proportion column
                    title="Monthly Proportion of Vulnerabilities Requiring User Interaction",
                    labels={'published_month': 'Month', 'proportion_required': '% Requiring User Interaction'},
                    markers=True
                )
                fig.update_layout(yaxis_range=[0, 100], yaxis_ticksuffix="%")  # Proportion is 0-100%

            elif chart_key == "overall_attack_vector_dist":  # Matched key
                fig = px.bar(
                    chart_data,
                    x='attack_vector',  # Use standardized column
                    y='percentage',
                    title="Overall Distribution of Attack Vectors",
                    labels={'attack_vector': 'Attack Vector', 'percentage': '% of Total CVEs'},
                    category_orders={"attack_vector": self.attack_vector_order},  # Use instance attribute
                    text_auto='.1f'  # Format percentage text
                )
                fig.update_layout(yaxis_ticksuffix="%")
                fig.update_traces(textposition='outside')

            elif chart_key == "overall_attack_complexity_dist":  # Matched key
                fig = px.bar(
                    chart_data,
                    x='attack_complexity',  # Use standardized column
                    y='percentage',
                    title="Overall Distribution of Attack Complexity",
                    labels={'attack_complexity': 'Attack Complexity', 'percentage': '% of Total CVEs'},
                    category_orders={"attack_complexity": self.attack_complexity_order},  # Use instance attribute
                    text_auto='.1f'  # Format percentage text
                )
                fig.update_layout(yaxis_ticksuffix="%")
                fig.update_traces(textposition='outside')

            elif chart_key == "patch_delay_risk_scatter":  # Matched key
                fig = px.scatter(
                    chart_data,
                    x='days_to_patch',
                    y='cvss_score',
                    title="Patch Delay (Proxy) vs. CVSS Score",  # Updated title for clarity
                    labels={'days_to_patch': 'Days Until NVD Publication (Proxy)', 'cvss_score': 'CVSS Base Score', 'cve_category': 'Category'},
                    color='cve_category',
                    color_discrete_map=self.category_color_map,  # Use instance attribute
                    category_orders={"cve_category": self.category_order},  # Use instance attribute
                    hover_data=['post_id']  # Use standardized CVE ID column
                )
                # Optional: Add annotations or lines if desired
                # fig.add_vline(x=30, line_dash="dash", annotation_text="30 Days")
                # fig.add_hline(y=7.0, line_dash="dash", annotation_text="CVSS 7.0")

            else:
                logger.warning(f"No figure generation logic defined for chart key '{chart_key}'.")
                return None

            # Apply common layout adjustments AFTER specific chart logic
            if fig:
                active_theme_name = 'report_light'
                fig.update_layout(template=pio.templates[active_theme_name])
                fig.update_layout(
                    title_text=chart_title,  # Ensure final title is set
                    title_x=0.05,
                    legend_title_text='',
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                    margin=dict(l=50, r=40, t=80, b=50)  # Adjusted margins slightly
                )

        except Exception as e:
            logger.exception(f"Error generating Plotly figure for '{chart_key}': {e}", exc_info=True)
            return None

        return fig

    def _export_chart_json(
        self,
        fig: go.Figure,
        chart_id: str,
        data_output_dir: Path
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
            logger.info(f"Exported chart data to: {filepath}")
            return filepath
        except Exception as e:
            logger.warning(f"Error exporting chart JSON for '{chart_id}' to '{filepath}': {e}")
            return None

    def _export_chart_image(
        self,
        fig: go.Figure,
        chart_id: str,  # Use the HTML-friendly chart ID (e.g., chart-new-vs-updated)
        image_output_dir: Path,
        image_format: str = "png",  # Common format, supported by kaleido
        width: int = 800,  # Adjust resolution as needed
        height: int = 600,
        scale: int = 2  # Increase scale for better resolution
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
            logger.warning(f"Cannot export image for '{chart_id}', figure is None.")
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
                scale=scale
            )
            logger.info(f"Exported chart image to: {filepath}")
            return filepath
        except ValueError as ve:
            # Catch specific error if Kaleido is missing
            if "kaleido" in str(ve).lower():
                logger.error("Kaleido engine not found. Please install ('pip install -U kaleido') to export static images.")
                # Optionally raise, or just return None to allow report generation without images
                return None
            else:
                logger.error(f"Error exporting chart image for '{chart_id}' to '{filepath}': {ve}", exc_info=True)
                return None
        except Exception as e:
            logger.error(f"Unexpected error exporting chart image for '{chart_id}' to '{filepath}': {e}", exc_info=True)
            return None

    def _format_data_for_llm(self, chart_data: Any, format_type: str = 'text') -> str:
        """Converts chart data into a string for the LLM prompt."""
        if chart_data is None: return "N/A"  # noqa: E701
        try:
            if isinstance(chart_data, pd.DataFrame):
                if chart_data.empty: return "N/A (DataFrame is empty)"  # noqa: E701
                if format_type == 'markdown':
                    return chart_data.to_markdown(index=False)
                else:  # Default to text string
                    return chart_data.to_string(index=False, max_rows=20)  # Limit rows
            elif isinstance(chart_data, dict):
                # Simple key-value pairs often work well
                return "\n".join([f"- {key.replace('_', ' ').title()}: {value}" for key, value in chart_data.items()])
            else:
                return str(chart_data)
        except Exception as e:
            logging.warning(f"Could not format data for LLM: {e}")
            return "Error formatting data."

    def _get_task_type_from_prompt_name(self, prompt_template_name: str) -> str:
        """Infers the task type from the Jinja template filename."""
        # Simple inference based on keywords in the filename
        if "chart" in prompt_template_name and "insight" in prompt_template_name:
            return "chart_insight"
        elif "callout" in prompt_template_name:
            return "callout"
        elif "narrative" in prompt_template_name:
            return "narrative"
        elif "summary" in prompt_template_name and "executive" not in prompt_template_name:  # Avoid matching executive_summary again
            return "section_summary"
        elif "summary" in prompt_template_name and "executive" in prompt_template_name:  # Avoid matching executive_summary again
            return "executive_summary"
        elif "conclusion" in prompt_template_name:
            return "report_conclusion"
        else:
            logger.warning(f"Could not determine specific task type for prompt '{prompt_template_name}'. Using default.")
            return "default"

    async def _get_llm_response(
        self,
        prompt_template_name: str,  # e.g., "section3_callout_1.j2"
        prompt_context_data: Dict[str, Any],  # Contains base context + config
        llm_model: str = "openrouter/google/gemini-pro",
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
        **kwargs
    ) -> str:
        """
        Generates an LLM insight using a specified Jinja prompt template.
        Can optionally include image paths for multimodal models.
        """
        logger.info(f"Generating LLM insight using template: {prompt_template_name}")

        if not self.llm:
            logger.warning("LLM client not configured/injected. Returning placeholder.")
            return f"Placeholder insight for {prompt_template_name} - LLM client missing."

        try:
            # template_full_path = self.run_specific_template_dir / "llm_prompts" / prompt_template_name
            template_relative_path = Path(self.config.template_subdir) / "llm_prompts" / prompt_template_name
            template_path_str = str(template_relative_path).replace("\\", "/")
            logger.info(f"Loading LLM prompt template from: {template_path_str}")
            template = self.jinja_env.get_template(template_path_str)

            rendered_prompt = template.render(prompt_context_data)
            logger.info(f"--- Rendered Prompt ---\n{rendered_prompt}\n----------------------")

            task_type = self._get_task_type_from_prompt_name(prompt_template_name)
            logger.info(f"Task type: {task_type}")
            system_prompt = self.SYSTEM_PROMPTS.get(task_type, self.SYSTEM_PROMPTS["default"])
            logger.info(f"Selected Task Type: {task_type}, using associated system prompt.")
            kwargs['task_type'] = task_type

            response = await self.llm.aget_completion(
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
                **kwargs  # this catches kwargs passed to _get_llm_response
            )
            return response

        except Exception as e:
            logger.error(f"Error generating LLM insight using template {prompt_template_name}: {e}", exc_info=True)
            return f"Error generating insight for {prompt_template_name}."

    def _generate_appendix_tables(self, df: pd.DataFrame, table_defs: List[Dict[str, str]]) -> Dict[str, str]:
        """Generates HTML tables for the appendix based on definitions."""
        logger.info("Generating appendix tables...")
        tables = {}
        if df.empty:
            logger.warning("DataFrame is empty, cannot generate appendix tables.")
            return tables

        for table_def in table_defs:
            table_id = table_def['id']
            logger.info(f"--> Generating table: {table_id}")
            try:
                if table_id == "appendix_cve_list":
                    cols_to_show = ['post_id', 'published', 'title', 'severity_type', 'cve_category', 'cvss_score']
                    # Select only columns that actually exist in the dataframe
                    cols_to_show = [col for col in cols_to_show if col in df.columns]
                    if not cols_to_show:
                        continue  # Skip if no relevant columns found

                    df_list = df[cols_to_show].sort_values(by='published' if 'published' in cols_to_show else cols_to_show[0], ascending=False)
                    if 'published' in df_list.columns:
                        df_list['published'] = df_list['published'].dt.strftime('%Y-%m-%d')
                    tables[table_id] = df_list.to_html(index=False, border=0, classes='data-table', na_rep='N/A')

                elif table_id == "appendix_outlier_list":
                    cols_to_show_outliers = ['post_id', 'title', 'cvss_score', 'cve_category']
                    cols_to_show_outliers = [col for col in cols_to_show_outliers if col in df.columns]
                    if 'cvss_score' not in cols_to_show_outliers:
                        continue  # Need score to find outliers

                    outliers = df[df['cvss_score'] >= 9.0].sort_values(by='cvss_score', ascending=False)  # Example: CVSS >= 9.0
                    if not outliers.empty:
                        tables[table_id] = outliers[cols_to_show_outliers].to_html(index=False, border=0, classes='data-table', na_rep='N/A')
                    else:
                        tables[table_id] = "<p>No significant outliers found (CVSS >= 9.0).</p>"  # Placeholder if empty
                else:
                    logger.warning(f"Unknown appendix table ID: {table_id}")

            except Exception as e:
                logger.error(f"Error generating appendix table '{table_id}': {e}", exc_info=True)
                tables[table_id] = f"<p>Error generating table '{table_id}'.</p>"

        logger.info(f"Generated {len(tables)} appendix tables.")
        return tables

    def _place_static_assets(
        self,
        css_output_dir: Path,
        js_output_dir: Path,
        config: ReportConfig
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
        source_css = APP_DIR / "static" / "css" / "QuarterlyReportStylesheet.css"
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
                logger.info(f"CSS file successfully copied to: {dest_css}")
            else:
                logger.warning(f"ERROR: Tailwind CSS file not found at expected location: {source_css}")
                logger.warning("Please ensure the Tailwind build process has completed successfully.")

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
            logger.info(f"JS file successfully copied to: {dest_js}")
        else:
            logger.warning(f"ERROR: JS file not found at expected location: {source_js}")
            logger.warning("Please ensure the JS file is present in the expected location.")

        return css_dest_path, js_dest_path

    async def generate_report(
        self,
        report_data: pd.DataFrame,
        config: ReportConfig,
        start_date: datetime,
        end_date: datetime
    ) -> GeneratedReportAssets:
        """
        Orchestrates the report generation process using provided data.
        """
        start_time = datetime.now()
        logger.info(f"Starting report generation: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Report Name: {config.report_name}, Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
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
        markdown_output_dir = run_specific_output_dir / config.output_markdown_subdir
        image_output_dir = run_specific_output_dir / config.output_image_subdir

        final_html_path = html_output_dir / dynamic_html_filename
        final_markdown_path = markdown_output_dir / dynamic_md_filename

        run_specific_template_dir = self.templates_base_dir / config.template_subdir
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
            logger.info(f"Output directories ensured: {html_output_dir}")
        except OSError as e:
            logger.error(f"FATAL: Error creating output directories: {e}")  # Changed to error
            raise

        # --- 2. Prepare Input Data ---
        try:
            main_df = self._prepare_input_dataframe(report_data)
        except (TypeError, ValueError) as e:
            logger.error(f"FATAL: Input data preparation failed: {e}")  # Changed to error
            raise

        if main_df.empty:
            logger.warning("Prepared DataFrame is empty. Report generation will continue but may be incomplete.")
            # Handle empty df gracefully in subsequent steps

        # --- 3. Generate Charts & LLM Insights ---
        generated_charts_context: Dict[str, ChartExport] = {}    # Holds context for charts successfully generated for final report
        generated_chart_files: List[Path] = []  # Holds paths to successfully exported chart JSON
        generated_image_files: List[Path] = []  # Holds paths to successfully exported chart images
        llm_section_content: Dict[str, Dict[str, str]] = defaultdict(dict)  # Stores final LLM text output {section_key: {prompt_type: text}}
        all_chart_data_for_llm: Dict[str, str] = {}  # Stores formatted data strings ONLY for successfully generated charts

        # Base context for ALL LLM prompts passed via Python
        base_llm_context_from_python = {
            "report_title": config.report_title,
            "start_date": start_date,
            "end_date": end_date,
            "total_cves": len(main_df) if not main_df.empty else 0,
            "config": config,  # Needed for finding template paths in _get_llm_response
        }

        # --- Iterate through Looped Sections Defined in Structure ---
        # <<< FIX: Iterate over "looped_sections", not "sections" >>>
        for section_key, section_config in self.REPORT_STRUCTURE.get("looped_sections", {}).items():
            logger.info(f"Processing section: {section_key} - {section_config.get('heading', '')}")

            # <<< FIX: Initialize section-specific containers INSIDE the loop >>>
            section_charts_context: Dict[str, ChartExport] = {}  # Holds ChartExport for charts relevant *to this section*
            section_chart_insights: Dict[str, str] = {}  # Holds LLM insights for charts relevant *to this section*
            section_callouts: Dict[str, str] = {}   # Holds LLM callouts generated *for this section*

            # --- Generate Charts for the Section ---
            for chart_def in section_config.get("charts", []):
                # <<< FIX: Use consistent chart identifier from definition >>>
                chart_id = chart_def["id"]
                logger.info(f"--> Processing chart: {chart_id}")

                # Prepare data using the chart ID
                # Pass copy only if _prepare modifies df, otherwise pass main_df directly for efficiency
                chart_data = self._prepare_chart_data(main_df if main_df.empty else main_df.copy(), chart_id)

                # Generate figure using the chart ID
                figure = self._generate_plotly_figure(chart_data, chart_id, chart_title=chart_def.get("caption"))
                chart_image_path: Optional[Path] = None

                if figure:
                    html_chart_id = f"chart-{chart_id.replace('_', '-')}"
                    json_path = self._export_chart_json(figure, html_chart_id, data_output_dir)
                    chart_image_path = self._export_chart_image(figure, html_chart_id, image_output_dir)
                    if chart_image_path:
                        generated_image_files.append(chart_image_path)  # Track successful image exports

                    if json_path:
                        # Chart generated and exported successfully
                        generated_chart_files.append(json_path)
                        chart_export = ChartExport(
                            chart_id=html_chart_id,
                            caption=chart_def.get("caption", f"Chart: {chart_id}")
                        )
                        generated_charts_context[chart_id] = chart_export  # Add to global context for final report rendering
                        section_charts_context[chart_id] = chart_export  # Add to section context for LLM prompts

                        # <<< FIX: Define formatted_data *here*, after chart_data is generated >>>
                        formatted_data = self._format_data_for_llm(chart_data, format_type='markdown')
                        all_chart_data_for_llm[chart_id] = formatted_data  # Store globally

                        # --- Get LLM Insight for Chart ---
                        insight_prompt_template = chart_def.get("insight_prompt")
                        if insight_prompt_template and self.llm:
                            prompt_context_data = {
                                **base_llm_context_from_python,
                                "section_title": section_config.get('heading', 'Chart Analysis'),
                                "chart_caption": chart_def.get("caption"),
                                "data_context_payload": {
                                    "chart_data_summary": formatted_data,
                                }
                            }
                            insight = await self._get_llm_response(
                                prompt_template_name=insight_prompt_template,
                                prompt_context_data=prompt_context_data,
                                llm_model=self.LLM_TASK_MODELS.get("chart_insight"),
                                image_paths=[chart_image_path],
                            )
                            # <<< FIX: Use chart_id for keys related to specific charts >>>
                            llm_section_content[section_key][f"chart_insight_{chart_id}"] = insight
                            section_chart_insights[chart_id] = insight  # Store for use in narrative/summary prompts
                        elif not self.llm:
                            logger.warning(f"LLM client not available, skipping insight for chart {chart_id}")
                            llm_section_content[section_key][f"chart_insight_{chart_id}"] = "LLM insight skipped (client unavailable)."
                        else:
                            logger.warning(f"No insight_prompt defined for chart {chart_id}")

                    else:  # JSON export failed
                        logger.warning(f"Skipping LLM insight and context for chart '{chart_id}' due to JSON export error.")
                        generated_charts_context[chart_id] = ChartExport(chart_id=html_chart_id, caption="(Chart JSON export failed)")
            # --- End of chart loop ---

            # --- Generate Section-Level LLM Content ---
            section_prompts = section_config.get("prompts", {})
            if not self.llm:
                logger.warning(f"LLM client not available, skipping section prompts for {section_key}")
            else:
                # --- Generate Callouts ---
                for i in range(1, 3):
                    callout_key = f"callout_{i}"
                    callout_prompt_template = section_prompts.get(callout_key)
                    if callout_prompt_template:
                        prompt_context_data = {
                            **base_llm_context_from_python,
                            "section_title": section_config.get('heading', f'Section Callout {i}'),
                            "data_context_payload": {
                                "chart_insights_in_section": section_chart_insights,  # Pass insights from charts *in this section*
                                # <<< FIX: Pass summaries for charts relevant to this section >>>
                                "charts_data_summaries_in_section": {cid: all_chart_data_for_llm.get(cid, "N/A") for cid in section_charts_context},
                            }
                        }
                        callout_text = await self._get_llm_response(
                            prompt_template_name=callout_prompt_template,
                            prompt_context_data=prompt_context_data,
                            llm_model=self.LLM_TASK_MODELS.get("callout"),
                            image_paths=None
                        )
                        llm_section_content[section_key][callout_key] = callout_text
                        section_callouts[callout_key] = callout_text  # Store for use in narrative/summary prompts

                # --- Generate Narrative ---
                narrative_prompt_template = section_prompts.get("narrative")
                if narrative_prompt_template:
                    prompt_context_data = {
                        **base_llm_context_from_python,
                        "section_title": section_config.get('heading', 'Section Narrative'),
                        "data_context_payload": {
                            "chart_insights_in_section": section_chart_insights,  # Pass insights generated above
                            "callouts_in_section": section_callouts,  # Pass callouts generated above
                            "charts_data_summaries_in_section": {cid: all_chart_data_for_llm.get(cid, "N/A") for cid in section_charts_context},
                        }
                    }
                    narrative_text = await self._get_llm_response(
                        prompt_template_name=narrative_prompt_template,
                        prompt_context_data=prompt_context_data,
                        llm_model=self.LLM_TASK_MODELS.get("narrative"),
                        # Adjust tokens based on section (e.g., more for exec summary)
                        llm_max_tokens=1024,
                        image_paths=None
                    )
                    # Store the generated narrative in the main llm_section_content dict
                    llm_section_content[section_key]["narrative"] = narrative_text
                else:
                    # Still add placeholder if no prompt, so key exists if needed later
                    llm_section_content[section_key]["narrative"] = "Narrative generation skipped (no prompt defined)."

                # --- Generate Summary ---
                summary_prompt_template = section_prompts.get("section_summary")
                if summary_prompt_template:
                    prompt_context_data = {
                        **base_llm_context_from_python,
                        "section_title": section_config.get('heading', 'Section Summary'),
                        "data_context_payload": {
                            "chart_insights_in_section": section_chart_insights,  # Pass insights generated above
                            "callouts_in_section": section_callouts,  # Pass callouts generated above
                            # Pass narrative generated above (use .get for safety)
                            "generated_narrative": llm_section_content[section_key].get("narrative", "Narrative not available."),
                            "charts_data_summaries_in_section": {cid: all_chart_data_for_llm.get(cid, "N/A") for cid in section_charts_context},
                        }
                    }
                    summary_text = await self._get_llm_response(
                        prompt_template_name=summary_prompt_template,
                        prompt_context_data=prompt_context_data,
                        llm_model=self.LLM_TASK_MODELS.get("section_summary"),
                        image_paths=None
                    )
                    llm_section_content[section_key]["summary"] = summary_text
                else:
                    llm_section_content[section_key]["summary"] = "Summary generation skipped (no prompt defined)."
        # --- End of looped sections ---

        # --- 4. Generate Appendix Tables ---
        # <<< FIX: Access appendix definition correctly >>>
        appendix_section = self.REPORT_STRUCTURE.get("explicit_sections", {}).get("appendix", {})
        appendix_tables_defs = appendix_section.get("tables", [])
        # Ensure table gen uses standardized names if accessing df columns
        appendix_tables_context = self._generate_appendix_tables(main_df, appendix_tables_defs)

        # --- 5. Calculate Final Stats (Using Standardized Columns) ---
        stats_context = {}
        if not main_df.empty:
            # <<< FIX: Use standardized column names >>>
            stats_context = {
                "total_cves": len(main_df),
                "critical_cves_count": len(main_df[main_df['severity_type'] == 'Critical']),
                "important_cves_count": len(main_df[main_df['severity_type'] == 'Important']),  # Added Important count
                "moderate_cves_count": len(main_df[main_df['severity_type'] == 'Moderate']),  # Added Moderate count
                "low_cves_count": len(main_df[main_df['severity_type'] == 'Low']),
                # Use 'cve_category' column and standardized names
                "rce_cves_count": len(main_df[main_df['cve_category'] == 'remote_code_execution']),
                "eop_cves_count": len(main_df[main_df['cve_category'] == 'privilege_elevation']),
                "dos_cves_count": len(main_df[main_df['cve_category'] == 'denial_of_service']),
                "infodisc_cves_count": len(main_df[main_df['cve_category'] == 'information_disclosure']),
                "spoofing_cves_count": len(main_df[main_df['cve_category'] == 'spoofing']),
                "bypass_cves_count": len(main_df[main_df['cve_category'] == 'security_feature_bypass']),
                # Calculate CVSS stats per category if needed (example for RCE)
                "rce_cvss_stats": main_df[main_df['cve_category'] == 'remote_code_execution']['cvss_score'].agg(['mean', 'median', 'max', 'min']).fillna(0).to_dict() if 'remote_code_execution' in main_df['cve_category'].unique() else {},
                "eop_cvss_stats": main_df[main_df['cve_category'] == 'privilege_elevation']['cvss_score'].agg(['mean', 'median', 'max', 'min']).fillna(0).to_dict() if 'privilege_elevation' in main_df['cve_category'].unique() else {},
                # ... add more stats as required ...
                "period": f"{start_date.strftime('%B %Y')} to {end_date.strftime('%B %Y')}"
            }
        else:  # Handle empty dataframe case for stats
            stats_context = {k: 0 for k in [
                "total_cves", "critical_cves_count", "important_cves_count", "moderate_cves_count",
                "low_cves_count", "rce_cves_count", "eop_cves_count", "dos_cves_count",
                "infodisc_cves_count", "spoofing_cves_count", "bypass_cves_count"
            ]}
            stats_context["period"] = f"{start_date.strftime('%B %Y')} to {end_date.strftime('%B %Y')}"
            stats_context["rce_cvss_stats"] = {}
            stats_context["eop_cvss_stats"] = {}

        # --- 6. Generate Explicit Section LLM Content (e.g., Executive Summary, Conclusion) ---
        if self.llm:
            # <<< FIX: Iterate over "explicit_sections" >>>
            for section_key, section_config in self.REPORT_STRUCTURE.get("explicit_sections", {}).items():
                section_prompts = section_config.get("prompts", {})
                prompt_template = section_prompts.get(section_key)

                if prompt_template:
                    logger.info(f"Generating narrative for explicit section: {section_key}")
                    prompt_context_data = {
                        **base_llm_context_from_python,
                        "section_title": section_config.get('heading', section_key.replace('_', ' ').title()),  # Set section title
                        "data_context_payload": {
                            "overall_report_stats": stats_context,
                            # Pass all previously generated content (looped sections)
                            "all_section_content": llm_section_content,
                        }
                    }
                    narrative_text = await self._get_llm_response(
                        prompt_template_name=prompt_template,
                        prompt_context_data=prompt_context_data,
                        llm_model=self.LLM_TASK_MODELS.get(section_key),
                        # Adjust tokens based on section (e.g., more for exec summary)
                        llm_max_tokens=1500 if section_key == "executive_summary" else 2500,
                        image_paths=None
                    )
                    # Store the generated narrative in the main llm_section_content dict
                    llm_section_content[section_key]["narrative"] = narrative_text
                else:
                    # Create placeholder if no narrative prompt defined
                    llm_section_content[section_key]["narrative"] = f"Narrative not generated for {section_key} (no prompt)."

        # --- 7. Prepare Final Jinja Context ---
        # <<< FIX: Flatten the collected LLM content correctly >>>
        final_llm_insights = {}
        for section_key, content_dict in llm_section_content.items():
            for prompt_type, text in content_dict.items():
                # Create a unique key like 'section_3_vol_sev_narrative' or 'executive_summary_narrative'
                final_llm_insights[f"{section_key}_{prompt_type}"] = text

        report_context = ReportContext(
            report_title=config.report_title,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            start_date=start_date,
            end_date=end_date,
            config=config,
            charts=generated_charts_context,  # Global chart context {chart_id: ChartExport}
            stats=stats_context,
            llm_insights=final_llm_insights,
            appendix_tables=appendix_tables_context,
            report_structure=self.REPORT_STRUCTURE
        )

        # --- 8. Render & Save HTML ---
        html_content = ""
        try:
            # Construct path relative to the *base* template directory Jinja knows
            template_relative_path = Path(config.template_subdir) / config.template_html_filename
            logger.info(f"Template path: {type(template_relative_path)} {template_relative_path}")
            template = self.jinja_env.get_template(template_relative_path.as_posix())
            # Pass the Pydantic model's dict representation to render
            context = report_context.model_dump()
            context["report_structure"] = self.REPORT_STRUCTURE
            # logger.info(f"context keys for Template: {list(context.keys())}\n\n")
            # logger.info(f"keys for llm_insights: {context['llm_insights'].keys()}\n\n")
            # logger.info(f"keys for charts: {context['charts'].keys()}\n\n")
            # logger.info(f"keys for report structure: {context['report_structure'].keys()}\n\n")
            # logger.info(f"keys for explicit sections: {context['report_structure']['explicit_sections'].keys()}\n\n")
            # logger.info(f"keys for looped sections: {context['report_structure']['looped_sections'].keys()}\n\n")
            # logger.info(f"keys for section 3: {context['report_structure']['looped_sections']['section_3_vol_sev'].keys()}\n\n")
            # logger.info(f"keys for section 4: {context['report_structure']['looped_sections']['section_4_cvss_dist'].keys()}\n\n")
            html_content = template.render(context)
            final_html_path.write_text(html_content, encoding="utf-8")
            logger.info(f"Report HTML saved to: {final_html_path}")
        except Exception as e:
            logger.error(f"FATAL: Error rendering or saving HTML report: {e}", exc_info=True)  # Add exc_info
            raise

        # --- 9. Generate & Save Markdown ---

        try:
            if html_content:
                logger.info("Converting generated HTML to Markdown...")
                markdown_output = markdownify.markdownify(html_content, heading_style="ATX")
                final_markdown_path.write_text(markdown_output, encoding="utf-8")
                logger.info(f"Report Markdown saved to: {final_markdown_path}")
            else:
                logger.warning("Skipping Markdown generation because HTML content is empty.")
                final_markdown_path = None
        except Exception as e:
            logger.error(f"Error generating or saving Markdown report: {e}", exc_info=True)  # Add exc_info
            final_markdown_path = None

        # --- 10. Copy Static Assets ---
        css_path, js_path = self._place_static_assets(css_output_dir, js_output_dir, config)

        # --- 11. Collate Asset Paths ---
        generated_assets = GeneratedReportAssets(
            report_config=config,
            base_directory=report_base_dir,
            html_file=final_html_path if html_content else None,  # Pass None if HTML failed
            markdown_file=final_markdown_path,
            css_file=css_path,
            js_file=js_path,  # js_path likely None based on _place_static_assets stub
            chart_data_files=generated_chart_files,
            image_files=generated_image_files
        )

        # --- 12. Finish ---
        end_time = datetime.now()
        logger.info(f"Report generation finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total time: {end_time - start_time}")

        return generated_assets


# --- Synthetic Data Generation Utility (moved outside class) ---
def generate_synthetic_cve_data(
    start_date: datetime,
    end_date: datetime,
    num_records: int = 200
) -> pd.DataFrame:
    """
    Generates a synthetic DataFrame for CVEs, mirroring the flattened
    structure expected after processing MongoDB records *without* unnecessary prefixes.
    """
    logger.info(f"Generating {num_records} synthetic CVE records (refactored)...")
    fake = Faker()
    data = []

    # Define possible values based on sample and general knowledge
    severities = ['Critical', 'Important', 'Moderate', 'Low']
    # Added 'other' as a possibility based on some schemas
    cve_categories = ['remote_code_execution', 'privilege_elevation', 'denial_of_service', 'disclosure', 'spoofing', 'feature_bypass', 'tampering']
    attack_vectors = ['Network', 'Adjacent', 'Local', 'Physical']
    attack_complexities = ['Low', 'High']
    privileges_required = ['None', 'Low', 'High']
    user_interactions = ['Required', 'None']  # Assuming null becomes 'None' or similar in processing
    scopes = ['Unchanged', 'Changed']
    cia_impacts = ['None', 'Low', 'High']
    products_list = [
        "windows_11_24H2", "windows_11_23H2", "windows_11_22H2", "windows_10_22H2",
        "windows_10_21H2", "windows_10_32-bit", "windows_10_64-bit", "windows_11_64-bit"
    ]

    # --- Define expected flattened column names ---
    # CNA fields (originally within metadata)
    # cna_fields = [
    #     'cna_attack_complexity', 'cna_attack_vector', 'cna_availability',
    #     'cna_base_score', 'cna_base_score_num', 'cna_base_score_rating',
    #     'cna_confidentiality', 'cna_exploitability_score', 'cna_impact_score',
    #     'cna_integrity', 'cna_privileges_required', 'cna_scope',
    #     'cna_user_interaction', 'cna_vector'
    # ]
    # ADP/NIST fields (originally within metadata) - generate as nulls
    adp_nist_fields = [
        f"{prefix}_{prop}"
        for prefix in ['adp', 'nist']
        for prop in [
            'attack_complexity', 'attack_vector', 'availability',
            'base_score', 'base_score_num', 'base_score_rating',
            'confidentiality', 'exploitability_score', 'impact_score',
            'integrity', 'privileges_required', 'scope',
            'user_interaction', 'vector'
        ]
    ]

    for i in range(num_records):
        record = {}
        metadata = {}
        # Use a simple index or a fake UUID for the top-level ID
        record['id_'] = fake.uuid4()
        record['text'] = fake.paragraph(nb_sentences=7)
        record['kb_ids'] = []
        metadata['id'] = record['id_']
        metadata['post_id'] = f"CVE-{np.random.randint(2023, end_date.year + 1)}-{np.random.randint(10000, 99999)}"
        metadata['published'] = fake.date_time_between(start_date=start_date, end_date=end_date, tzinfo=None).replace(tzinfo=None)  # Ensure naive datetime
        metadata['revision'] = np.random.choice(
            ["1.000", "1.100", "2.000"],
            p=[0.85, 0.10, 0.05]
        )
        metadata['severity_type'] = np.random.choice(severities, p=[0.05, 0.65, 0.25, 0.05])
        metadata['cve_category'] = np.random.choice(cve_categories, p=[0.20, 0.30, 0.10, 0.15, 0.05, 0.15, 0.05])  # Adjusted probabilities
        metadata['impact_type'] = metadata['cve_category'].replace('_', ' ').title()  # Derive from category
        metadata['title'] = f"{fake.word().capitalize()} Vuln in {fake.word().capitalize()} ({metadata['post_id']})"
        metadata['description'] = fake.paragraph(nb_sentences=2)
        metadata['summary'] = fake.paragraph(nb_sentences=4)
        metadata['source'] = f"https://msrc.microsoft.com/update-guide/vulnerability/{metadata['post_id']}"
        metadata['collection'] = "msrc_security_update"
        metadata['products'] = fake.random_elements(elements=products_list, length=np.random.randint(1, 4), unique=True)
        metadata['nvd_published_date'] = metadata['published'] - timedelta(days=np.random.randint(0, 3))
        metadata['nvd_description'] = metadata['description']  # Often similar

        # --- CNA CVSS Metrics (NO 'cna_' prefix for base_score_num/rating if desired, but keep for vector components) ---
        score = None
        if metadata['severity_type'] == 'Critical': score = np.random.uniform(9.0, 10.0)  # noqa: E701
        elif metadata['severity_type'] == 'Important': score = np.random.uniform(7.0, 8.9)  # noqa: E701 Changed Important lower bound to 7.0
        elif metadata['severity_type'] == 'Moderate': score = np.random.uniform(4.0, 6.9)  # noqa: E701 Changed Moderate upper bound to 6.9
        elif metadata['severity_type'] == 'Low': score = np.random.uniform(0.1, 3.9)  # noqa: E701
        # Keep cna_ prefix for clarity on source, unless flattening merges CNA/NIST/ADP based on priority
        metadata['cna_base_score_num'] = round(score, 1) if score is not None else None

        rating = "n/a"
        if score is not None:
            if score >= 9.0: rating = "critical"  # noqa: E701
            elif score >= 7.0: rating = "high"  # noqa: E701
            elif score >= 4.0: rating = "medium"  # noqa: E701
            else: rating = "low"  # noqa: E701
        metadata['cna_base_score_rating'] = rating
        # Generate combined score string if needed, otherwise might be redundant
        metadata['cna_base_score'] = f"{metadata['cna_base_score_num']} {rating.upper()}" if score is not None else None

        # Generate other vector components (keep cna_ prefix)
        metadata['cna_attack_vector'] = np.random.choice(attack_vectors)
        metadata['cna_attack_complexity'] = np.random.choice(attack_complexities)
        metadata['cna_privileges_required'] = np.random.choice(privileges_required)
        # Determine base probabilities for 'Required' and 'None' (string)
        base_prob_required, base_prob_none_str = (0.7, 0.3) if metadata['cna_attack_vector'] != 'Network' else (0.9, 0.1)
        # Define the probability for choosing Python None (representing unset/missing)
        prob_for_python_none = 0.05
        # Calculate the scaling factor for the original probabilities
        scaling_factor = 1.0 - prob_for_python_none  # This is 0.95
        # Scale the original probabilities
        scaled_prob_required = base_prob_required * scaling_factor
        scaled_prob_none_str = base_prob_none_str * scaling_factor
        # Combine the scaled probabilities with the probability for Python None
        final_probabilities = [scaled_prob_required, scaled_prob_none_str, prob_for_python_none]
        # Define the choices, including Python None
        choices = user_interactions + [None]  # e.g., ['Required', 'None', None]
        # Handle potential null user interaction from sample
        user_interaction_choice = np.random.choice(choices, p=final_probabilities)
        metadata['cna_user_interaction'] = user_interaction_choice if user_interaction_choice is not None else 'None'  # Default null to 'None' or handle downstream

        metadata['cna_scope'] = np.random.choice(scopes)
        metadata['cna_confidentiality'] = np.random.choice(cia_impacts)
        metadata['cna_integrity'] = np.random.choice(cia_impacts)
        metadata['cna_availability'] = np.random.choice(cia_impacts)

        # Construct vector string
        # Handle None for UI - use 'N' if None or 'None'
        ui_char = 'N' if metadata['cna_user_interaction'] in [None, 'None'] else metadata['cna_user_interaction'][0]
        vec = f"AV:{metadata['cna_attack_vector'][0]}/AC:{metadata['cna_attack_complexity'][0]}/PR:{metadata['cna_privileges_required'][0]}/UI:{ui_char}/S:{metadata['cna_scope'][0]}/C:{metadata['cna_confidentiality'][0]}/I:{metadata['cna_integrity'][0]}/A:{metadata['cna_availability'][0]}"
        metadata['cna_vector'] = vec

        metadata['cna_exploitability_score'] = round(np.random.uniform(1.0, 3.9), 1) if score is not None else None
        metadata['cna_impact_score'] = round(np.random.uniform(1.0, 6.0), 1) if score is not None else None

        # Add nulls for ADP/NIST fields (NO 'metadata_' prefix)
        for field in adp_nist_fields:
            metadata[field] = None  # Assign None directly to the flattened field name
        record['metadata'] = metadata
        data.append(record)

    df = pd.DataFrame(data)
    # Convert types after creation
    df['metadata'] = df['metadata'].apply(
        lambda md: {**md, 'published': pd.to_datetime(md['published'])} if md and 'published' in md else md
    )
    df['metadata'] = df['metadata'].apply(
        lambda md: {**md, 'nvd_published_date': pd.to_datetime(md['nvd_published_date'])} if md and 'nvd_published_date' in md else md
    )

    logger.info(f"Generated {len(df)} synthetic records (refactored) matching flattened structure.")
    return df
