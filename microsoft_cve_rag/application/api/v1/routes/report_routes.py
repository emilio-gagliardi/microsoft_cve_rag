"""Handle report generation operations via API."""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from application.services.document_service import DocumentService
from application.services.template_service import TemplateService
from application.etl.extractor import (
    extract_kb_articles_for_report,
    extract_cve_details_for_report
)
from application.etl.transformer import (
    process_cve_data_for_kb_report,
    transform_kb_data_for_kb_report
)


class KBReportRequest(BaseModel):
    """Request model for KB report generation."""
    start_date: datetime
    end_date: datetime


router = APIRouter()


def _sanitize_for_json(obj):
    """Recursively sanitize objects for JSON serialization.
    
    Handles:
    - datetime objects (converts to ISO format)
    - dict objects (recursively sanitizes values)
    - list objects (recursively sanitizes elements)
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: _sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    return obj


@router.post("/reports/kb")
async def generate_kb_report(request: KBReportRequest) -> Dict[str, Any]:
    """Generate a KB report for the specified date range.

    Args:
        request: Contains start_date and end_date for report generation

    Returns:
        Dict containing report status, file path, and processing summary

    Raises:
        HTTPException: If report generation fails
    """
    try:
        # Initialize services
        kb_service = DocumentService(
            db_name="report_docstore",
            collection_name="microsoft_kb_articles"
        )
        docstore_service = DocumentService(
            db_name="report_docstore",
            collection_name="docstore"
        )
        template_service = TemplateService()

        # Extract data
        kb_articles = extract_kb_articles_for_report(
            start_date=request.start_date,
            end_date=request.end_date,
            kb_service=kb_service
        )

        if not kb_articles:
            return {
                "status": "success",
                "data": [],
                "message": "No KB articles found"
            }

        # Get unique CVE IDs from KB articles
        cve_ids = {
            cve_id
            for article in kb_articles
            for cve_id in article.get("cve_ids", [])
        }
        kb_ids = {article.get("kb_id") for article in kb_articles}

        # Extract CVE details
        cve_details = extract_cve_details_for_report(
            list(cve_ids),
            list(kb_ids),
            docstore_service
        )

        # Transform data
        cve_lookup = process_cve_data_for_kb_report(cve_details)
        kb_data = await transform_kb_data_for_kb_report(kb_articles, cve_lookup)
        
        # Convert DataFrame to records and handle JSON serialization
        kb_records = kb_data.replace({float('inf'): None, float('-inf'): None, float('nan'): None}).to_dict(orient="records")
        
        # Recursively sanitize all data including nested structures
        kb_records = _sanitize_for_json(kb_records)
        
        # Render template and generate report
        report_file = template_service.render_kb_report(
            kb_data=kb_records,
            report_date=datetime.now()
        )

        return {
            "status": "success",
            "file_path": report_file,
            "data": kb_records,
            "message": f"Successfully processed {len(kb_articles)} KB articles"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate KB report: {str(e)}"
        )
