# Purpose: Handle vector database operations via API
# Inputs: Document data
# Outputs: Search results, operation status
# Dependencies: VectorDBService

from fastapi import APIRouter, Depends
from application.core.schemas import DocumentInput, DocumentOutput
from application.services.vector_db_service import VectorDBService
from typing import List

router = APIRouter()


@router.post("/vector/insert", response_model=bool)
async def insert_document(doc: DocumentInput, service: VectorDBService = Depends()):
    return service.insert(doc.content)


@router.get("/vector/search", response_model=List[DocumentOutput])
async def search_documents(
    query: str, top_k: int = 5, service: VectorDBService = Depends()
):
    return service.search(query, top_k)
