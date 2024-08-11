from fastapi import APIRouter, HTTPException
from core.schemas.document_schemas import (
    DocumentRecordCreate,
    DocumentRecordUpdate,
    DocumentRecordResponse,
    DocumentRecordQuery,
    DocumentRecordQueryResponse,
    BulkDocumentRecordCreate,
    BulkDocumentRecordUpdate,
    BulkDocumentRecordDelete,
)
from application.core.models import Document
from application.services.document_service import DocumentService
from application.services.embedding_service import EmbeddingService
from typing import List

router = APIRouter()
document_db_service = DocumentService()
embedding_service = EmbeddingService()


@router.on_event("shutdown")
def shutdown_event():
    document_db_service.close()


def create_document_record(document, embedding):
    return Document(
        embedding=embedding,
        metadata=document.metadata,
        excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
        excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
        relationships=document.relationships,
        text=document.text,
        start_char_idx=document.start_char_idx,
        end_char_idx=document.end_char_idx,
        text_template=document.text_template,
        metadata_template=document.metadata_template,
        metadata_separator=document.metadata_separator,
        class_name=document.class_name,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


@router.post("/documents/", response_model=DocumentRecordResponse)
def create_document(document: DocumentRecordCreate):
    try:
        embedding = embedding_service.generate_embedding(document.text)
        document_record = create_document_record(document, embedding)
        document_id = document_db_service.create_document(document_record)
        return DocumentRecordResponse(
            id_=document_id,
            message="Document created successfully",
            created_at=document.created_at,
            updated_at=document.updated_at,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}", response_model=DocumentRecordResponse)
def get_document(document_id: str):
    try:
        document = document_db_service.get_document(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return DocumentRecordResponse(
            id_=document["id"],
            message="Document retrieved successfully",
            created_at=document["created_at"],
            updated_at=document["updated_at"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/documents/{document_id}", response_model=DocumentRecordResponse)
def update_document(document_id: str, document: DocumentRecordUpdate):
    try:
        embedding = embedding_service.generate_embedding(document.text)
        document_record = create_document_record(document, embedding)
        updated_document = document_db_service.update_document(
            document_id, document_record
        )
        if updated_document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return DocumentRecordResponse(
            id_=updated_document["id"],
            message="Document updated successfully",
            created_at=updated_document["created_at"],
            updated_at=updated_document["updated_at"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}", response_model=DocumentRecordResponse)
def delete_document(document_id: str):
    try:
        deleted_document = document_db_service.delete_document(document_id)
        if deleted_document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return DocumentRecordResponse(
            id_=deleted_document["id"],
            message="Document deleted successfully",
            created_at=deleted_document["created_at"],
            updated_at=deleted_document["updated_at"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/query/", response_model=DocumentRecordQueryResponse)
def query_documents(query: DocumentRecordQuery):
    try:
        results = document_db_service.query_documents(
            query.query, query.page, query.page_size
        )
        return DocumentRecordQueryResponse(
            results=results["results"],
            total_count=results["total_count"],
            page=query.page,
            page_size=query.page_size,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/bulk/", response_model=List[DocumentRecordResponse])
def create_documents_bulk(documents: BulkDocumentRecordCreate):
    try:
        responses = []
        for document in documents.records:
            embedding = embedding_service.generate_embedding(document.text)
            document_record = create_document_record(document, embedding)
            document_id = document_db_service.create_document(document_record)
            responses.append(
                DocumentRecordResponse(
                    id_=document_id,
                    message="Document created successfully",
                    created_at=document.created_at,
                    updated_at=document.updated_at,
                )
            )
        return responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/documents/bulk/", response_model=List[DocumentRecordResponse])
def update_documents_bulk(documents: BulkDocumentRecordUpdate):
    try:
        responses = []
        for document in documents.records:
            embedding = embedding_service.generate_embedding(document.text)
            document_record = create_document_record(document, embedding)
            updated_document = document_db_service.update_document(
                str(document.id_), document_record
            )
            if updated_document is None:
                raise HTTPException(status_code=404, detail="Document not found")
            responses.append(
                DocumentRecordResponse(
                    id_=updated_document["id"],
                    message="Document updated successfully",
                    created_at=updated_document["created_at"],
                    updated_at=updated_document["updated_at"],
                )
            )
        return responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/bulk/", response_model=List[DocumentRecordResponse])
def delete_documents_bulk(documents: BulkDocumentRecordDelete):
    try:
        responses = []
        for document_id in documents.ids:
            deleted_document = document_db_service.delete_document(document_id)
            if deleted_document is None:
                raise HTTPException(status_code=404, detail="Document not found")
            responses.append(
                DocumentRecordResponse(
                    id_=deleted_document["id"],
                    message="Document deleted successfully",
                    created_at=deleted_document["created_at"],
                    updated_at=deleted_document["updated_at"],
                )
            )
        return responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
