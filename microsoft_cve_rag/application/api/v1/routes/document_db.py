# import os
# import sys

# original_dir = os.getcwd()
# print(sys.path)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
# print(sys.path)
from fastapi import APIRouter, HTTPException
from application.core.schemas.document_schemas import (
    DocumentRecordCreate,
    DocumentRecordUpdate,
    DocumentRecordResponse,
    DocumentRecordQuery,
    DocumentRecordQueryResponse,
    BulkDocumentRecordCreate,
    BulkDocumentRecordUpdate,
    BulkDocumentRecordDelete,
    DocumentMetadata,
)
from application.core.models import Document
from application.services.document_service import DocumentService
from application.services.embedding_service import EmbeddingService
from typing import List, Dict
import requests

router = APIRouter()
document_db_service = DocumentService()
embedding_service = EmbeddingService()


@router.on_event("shutdown")
def shutdown_event():
    """
    Close the document database service on shutdown.
    """
    document_db_service.close()


def create_document_record(document, embedding):
    """
    Create a Document record with the given document data and embedding.

    Args:
        document: The document data.
        embedding: The embedding vector.

    Returns:
        Document: The created Document object.
    """
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
    """
    Create a new document.

    Args:
        document (DocumentRecordCreate): The document data to create.

    Returns:
        DocumentRecordResponse: The response containing the document ID and status message.

    Example:
        Request:
        {
            "text": "Sample document text",
            "metadata": {
                "title": "Sample Document",
                "description": "This is a sample document."
            }
        }

        Response:
        {
            "id_": "document_id",
            "message": "Document created successfully",
            "created_at": "2023-10-01T00:00:00Z",
            "updated_at": "2023-10-01T00:00:00Z"
        }
    """
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
    """
    Retrieve a document by its ID.

    Args:
        document_id (str): The ID of the document to retrieve.

    Returns:
        DocumentRecordResponse: The response containing the document data and status message.

    Example:
        Request:
        GET /documents/document_id

        Response:
        {
            "id_": "document_id",
            "message": "Document retrieved successfully",
            "created_at": "2023-10-01T00:00:00Z",
            "updated_at": "2023-10-01T00:00:00Z"
        }
    """
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
    """
    Update an existing document by its ID.

    Args:
        document_id (str): The ID of the document to update.
        document (DocumentRecordUpdate): The updated document data.

    Returns:
        DocumentRecordResponse: The response containing the updated document data and status message.

    Example:
        Request:
        {
            "text": "Updated document text",
            "metadata": {
                "title": "Updated Document",
                "description": "This is an updated document."
            }
        }

        Response:
        {
            "id_": "document_id",
            "message": "Document updated successfully",
            "created_at": "2023-10-01T00:00:00Z",
            "updated_at": "2023-10-01T00:00:00Z"
        }
    """
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
    """
    Delete a document by its ID.

    Args:
        document_id (str): The ID of the document to delete.

    Returns:
        DocumentRecordResponse: The response containing the deleted document data and status message.

    Example:
        Request:
        DELETE /documents/document_id

        Response:
        {
            "id_": "document_id",
            "message": "Document deleted successfully",
            "created_at": "2023-10-01T00:00:00Z",
            "updated_at": "2023-10-01T00:00:00Z"
        }
    """
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
    """
    Query documents based on filter criteria.

    Args:
        query (DocumentRecordQuery): The query parameters and pagination details.

    Returns:
        DocumentRecordQueryResponse: The response containing the list of matched documents and pagination details.

    Example:
        Request:
        {
            "query": {"metadata.title": "Sample Document"},
            "page": 1,
            "page_size": 10
        }

        Response:
        {
            "results": [
                {
                    "id_": "document_id",
                    "text": "Sample document text",
                    "metadata": {
                        "title": "Sample Document",
                        "description": "This is a sample document."
                    }
                }
            ],
            "total_count": 1,
            "page": 1,
            "page_size": 10
        }
    """
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
    """
    Create multiple documents in bulk.

    Args:
        documents (BulkDocumentRecordCreate): The list of documents to create.

    Returns:
        List[DocumentRecordResponse]: The response containing the IDs and status messages of the created documents.

    Example:
        Request:
        {
            "records": [
                {
                    "text": "Sample document text 1",
                    "metadata": {
                        "title": "Sample Document 1",
                        "description": "This is a sample document 1."
                    }
                },
                {
                    "text": "Sample document text 2",
                    "metadata": {
                        "title": "Sample Document 2",
                        "description": "This is a sample document 2."
                    }
                }
            ]
        }

        Response:
        [
            {
                "id_": "document_id_1",
                "message": "Document created successfully",
                "created_at": "2023-10-01T00:00:00Z",
                "updated_at": "2023-10-01T00:00:00Z"
            },
            {
                "id_": "document_id_2",
                "message": "Document created successfully",
                "created_at": "2023-10-01T00:00:00Z",
                "updated_at": "2023-10-01T00:00:00Z"
            }
        ]
    """
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
    """
    Update multiple documents in bulk.

    Args:
        documents (BulkDocumentRecordUpdate): The list of documents to update.

    Returns:
        List[DocumentRecordResponse]: The response containing the IDs and status messages of the updated documents.

    Example:
        Request:
        {
            "records": [
                {
                    "id_": "document_id_1",
                    "text": "Updated document text 1",
                    "metadata": {
                        "title": "Updated Document 1",
                        "description": "This is an updated document 1."
                    }
                },
                {
                    "id_": "document_id_2",
                    "text": "Updated document text 2",
                    "metadata": {
                        "title": "Updated Document 2",
                        "description": "This is an updated document 2."
                    }
                }
            ]
        }

        Response:
        [
            {
                "id_": "document_id_1",
                "message": "Document updated successfully",
                "created_at": "2023-10-01T00:00:00Z",
                "updated_at": "2023-10-01T00:00:00Z"
            },
            {
                "id_": "document_id_2",
                "message": "Document updated successfully",
                "created_at": "2023-10-01T00:00:00Z",
                "updated_at": "2023-10-01T00:00:00Z"
            }
        ]
    """
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
    """
    Delete multiple documents in bulk.

    Args:
        documents (BulkDocumentRecordDelete): The list of document IDs to delete.

    Returns:
        List[DocumentRecordResponse]: The response containing the IDs and status messages of the deleted documents.

    Example:
        Request:
        {
            "ids": ["document_id_1", "document_id_2"]
        }

        Response:
        [
            {
                "id_": "document_id_1",
                "message": "Document deleted successfully",
                "created_at": "2023-10-01T00:00:00Z",
                "updated_at": "2023-10-01T00:00:00Z"
            },
            {
                "id_": "document_id_2",
                "message": "Document deleted successfully",
                "created_at": "2023-10-01T00:00:00Z",
                "updated_at": "2023-10-01T00:00:00Z"
            }
        ]
    """
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


@router.post("/documents/aggregate/", response_model=List[Dict])
def aggregate_documents(pipeline: List[Dict]):
    """
    Execute an aggregation pipeline on the documents collection.

    Args:
        pipeline (List[Dict]): The aggregation pipeline stages.

    Returns:
        List[Dict]: The result of the aggregation pipeline.
        List[Dict]: The result of the aggregation pipeline.


    Example:
        Request:
    Example:
        Request:
        [
        Request:
        [
        [
            {"$match": {"metadata.severity_type": "High"}},
            {"$group": {"_id": "$metadata.products", "count": {"$sum": 1}}},
            {"$match": {"metadata.severity_type": "High"}},
            {"$group": {"_id": "$metadata.products", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
            {"$group": {"_id": "$metadata.products", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
            {"$sort": {"count": -1}}
        ]

        ]


        Response:
        [
        Response:
        [
            {"_id": "Product A", "count": 10},
        [
            {"_id": "Product A", "count": 10},
            {"_id": "Product A", "count": 10},
            {"_id": "Product B", "count": 5}
            {"_id": "Product B", "count": 5}
        ]
    """
    try:
        results = document_db_service.aggregate_documents(pipeline)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":

    BASE_URL = "http://localhost:8000"

    def test_create_document():
        url = f"{BASE_URL}/documents/"
        data = DocumentRecordCreate(
            text="Sample document text",
            metadata=DocumentMetadata(
                title="Sample Document",
                description="This is a sample document.",
                collection="test_data",
            ),
        )
        data.compute_hash()  # Compute the hash before sending the request
        response = requests.post(url, json=data.model_dump())
        print("Create Document Response:", response.json())
        return response.json()["id_"]

    def test_get_document(document_id):
        url = f"{BASE_URL}/documents/{document_id}"
        response = requests.get(url)
        print("Get Document Response:", response.json())

    def test_update_document(document_id):
        url = f"{BASE_URL}/documents/{document_id}"
        data = {
            "text": "Updated document text",
            "metadata": {
                "title": "Updated Document",
                "description": "This is an updated document.",
            },
        }
        response = requests.put(url, json=data)
        print("Update Document Response:", response.json())

    def test_delete_document(document_id):
        url = f"{BASE_URL}/documents/{document_id}"
        response = requests.delete(url)
        print("Delete Document Response:", response.json())

    def test_query_documents():
        url = f"{BASE_URL}/documents/query/"
        data = {
            "query": {"metadata.title": "Sample Document"},
            "page": 1,
            "page_size": 10,
        }
        response = requests.post(url, json=data)
        print("Query Documents Response:", response.json())

    def test_create_documents_bulk():
        url = f"{BASE_URL}/documents/bulk/"
        data = {
            "records": [
                {
                    "text": "Sample document text 1",
                    "metadata": {
                        "title": "Sample Document 1",
                        "description": "This is a sample document 1.",
                    },
                },
                {
                    "text": "Sample document text 2",
                    "metadata": {
                        "title": "Sample Document 2",
                        "description": "This is a sample document 2.",
                    },
                },
            ]
        }
        response = requests.post(url, json=data)
        print("Create Documents Bulk Response:", response.json())

    def test_update_documents_bulk():
        url = f"{BASE_URL}/documents/bulk/"
        data = {
            "records": [
                {
                    "id_": "document_id_1",
                    "text": "Updated document text 1",
                    "metadata": {
                        "title": "Updated Document 1",
                        "description": "This is an updated document 1.",
                    },
                },
                {
                    "id_": "document_id_2",
                    "text": "Updated document text 2",
                    "metadata": {
                        "title": "Updated Document 2",
                        "description": "This is an updated document 2.",
                    },
                },
            ]
        }
        response = requests.put(url, json=data)
        print("Update Documents Bulk Response:", response.json())

    def test_delete_documents_bulk():
        url = f"{BASE_URL}/documents/bulk/"
        data = {"ids": ["document_id_1", "document_id_2"]}
        response = requests.delete(url, json=data)
        print("Delete Documents Bulk Response:", response.json())

    def test_aggregate_documents():
        url = f"{BASE_URL}/documents/aggregate/"
        data = [
            {"$match": {"metadata.severity_type": "High"}},
            {"$group": {"_id": "$metadata.products", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        response = requests.post(url, json=data)
        print("Aggregate Documents Response:", response.json())

    # Run tests
    document_id = test_create_document()
    test_get_document(document_id)
    test_update_document(document_id)
    test_delete_document(document_id)
    test_query_documents()
    test_create_documents_bulk()
    test_update_documents_bulk()
    test_delete_documents_bulk()
    test_aggregate_documents()