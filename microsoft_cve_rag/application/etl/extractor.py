# Purpose: Extract data from various sources
# Inputs: Source configurations
# Outputs: Raw data
# Dependencies: None
from typing import Any, Dict, List
from application.services.document_service import DocumentService
from typing import List, Dict, Any


def extract_from_mongo(
    db_name: str, collection_name: str, query: Dict[str, Any]
) -> List[Dict[str, Any]]:
    document_service = DocumentService(db_name, collection_name)
    return document_service.query_documents(query)
