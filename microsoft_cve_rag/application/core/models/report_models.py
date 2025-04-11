from pydantic import BaseModel, DirectoryPath, Field, FilePath
from datetime import datetime
from typing import List, Dict, Any, Optional

class ReportConfig(BaseModel):
    start_date: datetime
    end_date: datetime
    output_dir: DirectoryPath = Field(default_factory=lambda: "reports/quarterly_deep_dive/html")
    data_subdir: str = "data"
    template_dir: str = "quarterly_deep_dive" # Relative path within main template folder
    report_filename: str = "index.html"
    use_synthetic_data: bool = False
    # Optional upload settings - can be None if not provided/used
    # sftp_settings: Optional[SFTPSettings] = None
    # azure_settings: Optional[AzureStorageSettings] = None
    upload_sftp: bool = False
    upload_azure: bool = False
    # Add other config like LLM model details if needed

class ChartExport(BaseModel):
    chart_id: str
    caption: Optional[str] = None
    # Add any other metadata needed per chart for the template

class ReportContext(BaseModel):
    # Data passed directly to the Jinja template
    report_title: str
    generation_date: str
    config: ReportConfig # Pass config for potential use in template
    charts: Dict[str, ChartExport] = Field(default_factory=dict)
    stats: Dict[str, Any] = Field(default_factory=dict)
    llm_insights: Dict[str, str] = Field(default_factory=dict)
    appendix_tables: Dict[str, str] = Field(default_factory=dict) # HTML strings
    # Add any other top-level context needed
