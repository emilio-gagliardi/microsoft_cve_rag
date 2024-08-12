# import os
# import sys

# original_dir = os.getcwd()
# print(sys.path)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
# print(sys.path)

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from uuid import UUID, uuid4
from hashlib import sha256
from datetime import datetime, timezone
from application.config import PROJECT_CONFIG


class BaseMetadata(BaseModel):
    """
    Base metadata model for documents. Includes common metadata fields.
    """

    revision: Optional[str] = Field("", description="The version of msrc post types.")
    id: Optional[UUID] = Field(None, description="UUID generated at ingestion")
    post_id: Optional[str] = Field(
        "", description="Specific CVE identification ID. eg. CVE-2023-36435"
    )
    published: Optional[datetime] = Field(
        None,
        description="Publication date of the version. Multiple versions have multiple dates.",
    )
    title: Optional[str] = Field("", description="Title of the document")
    description: Optional[str] = Field("", description="Description of the document")
    build_numbers: Optional[List[List[int]]] = Field(
        default_factory=list, description="All CVEs are associated to specific OS build numbers."
    )
    impact_type: Optional[str] = Field(
        "", description="Impact type of the security vulnerability"
    )
    product_build_ids: Optional[List[UUID]] = Field(
        default_factory=list,
        description="Identifier that associates products, kb articles, update packages",
    )
    products: Optional[List[str]] = Field(
        default_factory=list, description="The name of the product(s) affected by the CVE"
    )
    severity_type: Optional[str] = Field("", description="Severity type of the CVE")
    summary: Optional[str] = Field("", description="Summary of the document")
    collection: Optional[str] = Field(
        "", description="document collection. Currently there are 10."
    )
    source: Optional[str] = Field("", description="The URL of the ingested document")
    hash: Optional[str] = Field("", description="Hash of the document")
    conversationId: Optional[str] = Field(
        "", description="Conversation identifier for patch management emails"
    )
    subject: Optional[str] = Field(
        "", description="Subject of the patch management email"
    )
    receivedDateTime: Optional[str] = Field(
        "", description="The datetime when email was received by google groups."
    )


class DocumentMetadata(BaseMetadata):
    """
    Metadata model for documents. Includes additional fields specific to documents.
    """

    cve_fixes: Optional[str] = Field(
        "", description="CVE fixes mentioned in the document"
    )
    cve_mentions: Optional[str] = Field(
        "", description="CVE mentions in the document"
    )
    tags: Optional[str] = Field("", description="Tags associated with the document")
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
                "conversationId": "conv123",
                "subject": "Sample Subject",
                "receivedDateTime": "2023-10-01T00:00:00Z",
                "cve_fixes": "CVE-2023-1234",
                "cve_mentions": "CVE-2023-5678",
                "tags": "security, update",
                "added_to_vector_store": True,
                "added_to_summary_index": False,
                "added_to_graph_store": True,
            }
        }


class DocumentRecordBase(BaseModel):
    """
    Base model for document records. This model includes common fields that are shared across different document record operations.
    """

    id_: Optional[UUID] = Field(
        default_factory=uuid4, description="Unique identifier of the record"
    )
    embedding: Optional[List[float]] = Field(None, description="Embedding vector")
    metadata: Optional[DocumentMetadata] = Field(
        None, description="Metadata associated with the record"
    )
    excluded_embed_metadata_keys: Optional[List[str]] = Field(
        default_factory=list,
        description="Metadata keys to exclude from embedding. LlamaIndex specific.",
    )
    excluded_llm_metadata_keys: Optional[List[str]] = Field(
        default_factory=list, description="Metadata keys to exclude from LLM. LlamaIndex specific."
    )
    relationships: Optional[Dict[str, str]] = Field(
        default_factory=dict, description="Relationships of the record. LlamaIndex specific."
    )
    text: Optional[str] = Field("", description="Text associated with the record")
    start_char_idx: Optional[int] = Field(None, description="Start character index")
    end_char_idx: Optional[int] = Field(None, description="End character index")
    text_template: Optional[str] = Field("", description="Template for text")
    metadata_template: Optional[str] = Field("", description="Template for metadata")
    metadata_separator: Optional[str] = Field(
        "", description="Separator for metadata"
    )
    class_name: Optional[str] = Field(
        "", description="Class name used in processing ie., a LlamaIndex Document in this case.",
    )
    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the record was created",
    )
    updated_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the record was last updated",
    )

    @field_validator("embedding")
    def check_embedding_length(cls, v):
        if v and len(v) != PROJECT_CONFIG["DEFAULT_EMBEDDING_CONFIG"].embedding_length:
            raise ValueError(
                f"Embedding model {PROJECT_CONFIG['DEFAULT_EMBEDDING_CONFIG'].model_name} must have a length of {PROJECT_CONFIG['DEFAULT_EMBEDDING_CONFIG'].embedding_length}"
            )
        return v

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
                    "conversationId": "conv123",
                    "subject": "Sample Subject",
                    "receivedDateTime": "2023-10-01T00:00:00Z",
                    "cve_fixes": "CVE-2023-1234",
                    "cve_mentions": "CVE-2023-5678",
                    "tags": "security, update",
                    "added_to_vector_store": True,
                    "added_to_summary_index": False,
                    "added_to_graph_store": True,
                },
                "excluded_embed_metadata_keys": ["hash", "conversationId"],
                "excluded_llm_metadata_keys": ["hash", "conversationId"],
                "relationships": {
                    "related_doc": "123e4567-e89b-12d3-a456-426614174002"
                },
                "text": "Sample document text",
                "start_char_idx": 0,
                "end_char_idx": 100,
                "text_template": "Template for text",
                "metadata_template": "Template for metadata",
                "metadata_separator": "|",
                "class_name": "Document",
                "created_at": "2023-10-01T00:00:00Z",
                "updated_at": "2023-10-01T00:00:00Z",
            }
        }


class DocumentRecordCreate(DocumentRecordBase):
    """
    Model for creating a new document record. Inherits from DocumentRecordBase and includes all necessary fields for creating a new record.
    """

    def compute_hash(self):
        """
        Compute the hash for the document based on the collection type.
        """
        if self.metadata.collection == "msrc_security_update":
            hash_input = f"{self.metadata.revision}{self.metadata.source}{self.metadata.category}{self.metadata.published}{self.metadata.description}{self.metadata.title}{self.metadata.collection}"
        elif self.metadata.collection in ["windows_10", "windows_11", "windows_update"]:
            hash_input = f"{self.metadata.source}{self.metadata.summary}{self.metadata.published}{self.metadata.description}{self.metadata.title}{self.metadata.collection}"
        elif self.metadata.collection == "patch_management":
            hash_input = f"{self.metadata.source}{self.metadata.conversationId}{self.metadata.subject}{self.metadata.from_}{self.metadata.receivedDateTime}"
        else:
            hash_input = f"{self.metadata.title}{self.metadata.description}{self.metadata.collection}"

        self.metadata.hash = sha256(hash_input.encode("utf-8")).hexdigest()

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Sample document text",
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
                    "conversationId": "conv123",
                    "subject": "Sample Subject",
                    "receivedDateTime": "2023-10-01T00:00:00Z",
                    "cve_fixes": "CVE-2023-1234",
                    "cve_mentions": "CVE-2023-5678",
                    "tags": "security, update",
                    "added_to_vector_store": True,
                    "added_to_summary_index": False,
                    "added_to_graph_store": True,
                },
            }
        }


class DocumentRecordUpdate(DocumentRecordBase):
    """
    Model for updating an existing document record. Inherits from DocumentRecordBase and includes all necessary fields for updating a record.
    """

    id_: UUID


class DocumentRecordDelete(BaseModel):
    """
    Model for deleting a document record. Includes the unique identifier of the record to be deleted.
    """

    id_: UUID


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
    Response model for document record operations. Includes the unique identifier, message, and timestamps of the record.
    """

    id_: Optional[UUID] = Field(None, description="Unique identifier of the record")
    message: str = Field(..., description="Response message from database")
    created_at: Optional[datetime] = Field(
        None, description="Timestamp when the record was created"
    )
    updated_at: Optional[datetime] = Field(
        None, description="Timestamp when the record was last updated"
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

    ids: List[UUID]


class ErrorResponse(BaseModel):
    """
    Response model for errors. Includes the error message and error code.
    """

    error: str = Field(..., description="Error message")
    code: int = Field(..., description="Error code")
