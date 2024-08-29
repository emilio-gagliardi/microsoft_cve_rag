# import os
# import sys

# original_dir = os.getcwd()
# print(sys.path)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
# print(sys.path)

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any

# from hashlib import sha256
from datetime import datetime
from application.app_utils import get_app_config

settings = get_app_config()


class BaseMetadata(BaseModel):
    """
    Base metadata model for documents. Includes common metadata fields.
    """

    revision: Optional[str] = Field(None, description="The version of msrc post types.")
    id: str = Field(..., description="UUID string")
    post_id: Optional[str] = Field(
        None, description="Specific CVE identification ID. eg. CVE-2023-36435"
    )
    published: Optional[datetime] = Field(
        None,
        description="Publication date of the version. Multiple versions have multiple dates.",
    )

    @field_validator("published")
    def parse_published(cls, value):
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return value

    title: Optional[str] = Field(None, description="Title of the document")
    description: Optional[str] = Field(None, description="Description of the document")
    build_numbers: Optional[List[List[int]]] = Field(
        None,
        description="All CVEs are associated to specific OS build numbers.",
    )
    impact_type: Optional[str] = Field(
        None, description="Impact type of the security vulnerability"
    )
    product_build_ids: Optional[List[str]] = Field(
        None,
        description="Identifier that associates products, kb articles, update packages",
    )
    products: Optional[List[str]] = Field(
        None,
        description="The name of the product(s) affected by the CVE",
    )
    severity_type: Optional[str] = Field(None, description="Severity type of the CVE")
    summary: Optional[str] = Field(None, description="Summary of the document")
    collection: Optional[str] = Field(
        None, description="document collection. Currently there are 10."
    )
    source: Optional[str] = Field(None, description="The URL of the ingested document")
    hash: Optional[str] = Field(None, description="Hash of the document")


class DocumentMetadata(BaseMetadata):
    """
    Metadata model for documents. Includes additional fields specific to documents.
    """

    cve_fixes: Optional[str] = Field(
        None, description="CVE fixes mentioned in the document"
    )
    cve_mentions: Optional[str] = Field(
        None, description="CVE mentions in the document"
    )
    tags: Optional[str] = Field(None, description="Tags associated with the document")
    added_to_vector_store: Optional[bool] = Field(
        False, description="Indicates if the document is added to the vector store"
    )
    added_to_summary_index: Optional[bool] = Field(
        False, description="Indicates if the document is added to the summary index"
    )
    added_to_graph_store: Optional[bool] = Field(
        False, description="Indicates if the document is added to the graph store"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "revision": "1.0",
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "post_id": "CVE-2024-post123",
                "published": "2024-07-15T00:00:00+00:00",
                "title": "Sample Microsoft Document Title",
                "description": "In Microsoft documentation, only the CVEs typically contain descriptions that coincide with the version of the document.",
                "build_numbers": [[10, 0, 19041], [10, 0, 19042]],
                "impact_type": "Security",
                "product_build_ids": ["123e4567-e89b-12d3-a456-426614174001"],
                "products": ["Windows 10", "Windows 11"],
                "severity_type": "High",
                "summary": "",
                "collection": "test_data",
                "source": "https://msrc.microsoft.com/update-guide/vulnerability/CVE-2024-38200",
                "hash": "06b7bef8130667e61c93582304930b107a6f6a79bf2389801c8f75c345861143",
                "cve_fixes": "CVE-2023-1234",
                "cve_mentions": "CVE-2023-5678",
                "tags": "security, update",
                "added_to_vector_store": False,
                "added_to_summary_index": False,
                "added_to_graph_store": False,
            }
        }


class DocumentRecordBase(BaseModel):
    """
    Base model for document records. This model includes common fields that are shared across different document record operations.
    """

    id_: Optional[str] = Field(None, description="Unique identifier of the record")
    embedding: Optional[List[float]] = Field(None, description="Embedding vector")
    metadata: Optional[DocumentMetadata] = Field(
        None, description="Metadata associated with the record"
    )
    excluded_embed_metadata_keys: Optional[List[str]] = Field(
        None,
        description="Metadata keys to exclude from embedding. LlamaIndex specific.",
    )
    excluded_llm_metadata_keys: Optional[List[str]] = Field(
        None,
        description="Metadata keys to exclude from LLM. LlamaIndex specific.",
    )
    relationships: Optional[Dict[str, str]] = Field(
        None,
        description="Relationships of the record. LlamaIndex specific.",
    )
    text: Optional[str] = Field(None, description="Text associated with the record")
    start_char_idx: Optional[int] = Field(None, description="Start character index")
    end_char_idx: Optional[int] = Field(None, description="End character index")
    text_template: Optional[str] = Field(None, description="Template for text")
    metadata_template: Optional[str] = Field(None, description="Template for metadata")
    metadata_separator: Optional[str] = Field(
        None, description="Separator for metadata"
    )
    class_name: Optional[str] = Field(
        "Document",
        description="Class name used in RAG processing ie., a LlamaIndex Document in this case.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id_": "123e4567-e89b-12d3-a456-426614174000",
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "metadata": {
                    "revision": "1.0",
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "post_id": "post123",
                    "published": "2023-09-21T00:00:00.000+00:00",
                    "title": "Sample Document",
                    "description": "This is a sample document.",
                    "build_numbers": [[10, 0, 19041], [10, 0, 19042]],
                    "impact_type": "Security",
                    "product_build_ids": ["123e4567-e89b-12d3-a456-426614174001"],
                    "products": ["Windows 10", "Windows 11"],
                    "severity_type": "High",
                    "summary": "Summary of the document.",
                    "collection": "test_data",
                    "source": "Microsoft",
                    "hash": "abc123",
                    "cve_fixes": "CVE-2023-1234",
                    "cve_mentions": "CVE-2023-5678",
                    "tags": "security, update",
                    "added_to_vector_store": False,
                    "added_to_summary_index": False,
                    "added_to_graph_store": False,
                },
                "excluded_embed_metadata_keys": ["hash", "added_to_vector_store"],
                "excluded_llm_metadata_keys": ["hash", "added_to_vector_store"],
                "text": "Sample document text",
                "start_char_idx": 0,
                "end_char_idx": 100,
                "text_template": "Template for text",
                "metadata_template": "Template for metadata",
                "metadata_separator": "|",
                "class_name": "Document",
            }
        }


class DocumentRecordCreate(DocumentRecordBase):
    """
    Model for creating a new document record. Inherits from DocumentRecordBase and includes all necessary fields for creating a new record.
    """

    id_: str = Field(..., description="Unique identifier of the record")
    metadata: DocumentMetadata = Field(
        ..., description="Metadata associated with the record"
    )
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "id_": "123e4567-e89b-12d3-a456-426614174000",
                "text": "Sample document text",
                "metadata": {
                    "revision": "1.0",
                    "post_id": "CVE_2024-post123",
                    "published": "2023-09-21T00:00:00.000+00:00",
                    "title": "Sample Document",
                    "description": "This is a sample document.",
                    "build_numbers": [[10, 0, 19041], [10, 0, 19042]],
                    "impact_type": "Security",
                    "product_build_ids": ["123e4567-e89b-12d3-a456-426614174001"],
                    "products": ["Windows 10", "Windows 11"],
                    "severity_type": "High",
                    "summary": "Summary of the document.",
                    "collection": "test_data",
                    "source": "Microsoft",
                    "hash": "abc123",
                    "cve_fixes": "CVE-2023-1234",
                    "cve_mentions": "CVE-2023-5678",
                    "tags": "security, update",
                    "added_to_vector_store": False,
                    "added_to_summary_index": False,
                    "added_to_graph_store": False,
                },
            }
        }


class DocumentRecordUpdate(DocumentRecordBase):
    """
    Model for updating an existing document record. Inherits from DocumentRecordBase and includes all necessary fields for updating a record.
    """

    id_: str
    metadata: DocumentMetadata = Field(
        ..., description="Metadata associated with the record"
    )


class DocumentRecordDelete(BaseModel):
    """
    Model for deleting a document record. Includes the unique identifier of the record to be deleted.
    """

    id_: str


class DocumentRecordQuery(BaseModel):
    """
    Model for querying document records. Includes query parameters and pagination details.
    """

    query: Dict[str, str] = Field(default_factory=dict, description="Query parameters")
    page: Optional[int] = Field(1, description="Page number for pagination. Default 0")
    page_size: Optional[int] = Field(
        10, description="Number of records per page. Default 10."
    )


class DocumentRecordResponse(BaseModel):
    """
    Response model for document record operations. Includes the unique identifier, message, timestamps of the record, and the document itself.
    """

    id_: Optional[str] = Field(None, description="Unique identifier of the record")
    message: str = Field(..., description="Response message from database")
    document: Optional[DocumentRecordBase] = Field(
        None, description="The document data"
    )


class DocumentRecordQueryResponse(BaseModel):
    """
    Response model for document record queries. Includes the list of records matching the query, total count, and pagination details.
    """

    results: List[DocumentRecordBase] = Field(
        ..., description="List of records matching the query"
    )
    total_count: int = Field(
        ..., description="Total count of records matching the query"
    )
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of records per page")


class BulkDocumentRecordCreate(BaseModel):
    """
    Model for creating multiple document records in bulk. Includes a list of DocumentRecordCreate objects.
    """

    records: List[DocumentRecordCreate]


class BulkDocumentRecordUpdate(BaseModel):
    """
    Model for updating multiple document records in bulk. Includes a list of DocumentRecordUpdate objects.
    """

    records: List[DocumentRecordUpdate]


class BulkDocumentRecordDelete(BaseModel):
    """
    Model for deleting multiple document records in bulk. Includes a list of unique identifiers of the records to be deleted.
    """

    ids: List[str]


class AggregationPipeline(BaseModel):
    pipeline: List[Dict[str, Any]] = Field(
        ..., description="The aggregation pipeline stages"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "pipeline": [
                    {
                        "$match": {
                            "metadata.collection": "msrc_security_update",
                            "metadata.published": {
                                "$gte": "2024-07-01T00:00:00Z",
                                "$lte": "2024-07-15T23:59:59Z",
                            },
                        }
                    },
                    {"$sort": {"metadata.post_id": -1}},
                ]
            }
        }


class ErrorResponse(BaseModel):
    """
    Response model for errors. Includes the error message and error code.
    """

    error: str = Field(..., description="Error message")
    code: int = Field(..., description="Error code")
