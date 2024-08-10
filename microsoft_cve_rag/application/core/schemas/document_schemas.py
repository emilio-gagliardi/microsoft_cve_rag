from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from uuid import UUID, uuid4
from datetime import datetime, timezone
from application.config import DEFAULT_EMBEDDING_CONFIG


class Published(BaseModel):
    """
    Model for published information.
    """

    date: Optional[datetime] = None


class BaseMetadata(BaseModel):
    """
    Base metadata model for documents. Includes common metadata fields.
    """

    revision: Optional[str] = None
    id: Optional[UUID] = None
    post_id: Optional[str] = None
    published: Optional[Published] = None
    title: Optional[str] = None
    description: Optional[str] = None
    build_numbers: Optional[List[List[int]]] = None
    impact_type: Optional[str] = None
    product_build_ids: Optional[List[UUID]] = None
    products: Optional[List[str]] = None
    severity_type: Optional[str] = None
    summary: Optional[str] = None


class DocumentMetadata(BaseMetadata):
    """
    Metadata model for documents. Includes additional fields specific to documents.
    """

    cve_fixes: Optional[str] = None
    cve_mentions: Optional[str] = None
    tags: Optional[str] = None
    added_to_vector_store: Optional[bool] = None
    added_to_summary_index: Optional[bool] = None
    added_to_graph_store: Optional[bool] = None


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
        None, description="Metadata keys to exclude from embedding"
    )
    excluded_llm_metadata_keys: Optional[List[str]] = Field(
        None, description="Metadata keys to exclude from LLM"
    )
    relationships: Optional[Dict[str, str]] = Field(
        None, description="Relationships of the record"
    )
    text: Optional[str] = Field(None, description="Text associated with the record")
    start_char_idx: Optional[int] = Field(None, description="Start character index")
    end_char_idx: Optional[int] = Field(None, description="End character index")
    text_template: Optional[str] = Field(None, description="Template for text")
    metadata_template: Optional[str] = Field(None, description="Template for metadata")
    metadata_separator: Optional[str] = Field(
        None, description="Separator for metadata"
    )
    class_name: Optional[str] = Field(None, description="Class name of the record")
    created_at: Optional[datetime] = Field(
        default_factory=datetime.now(datetime.UTC),
        description="Timestamp when the record was created",
    )
    updated_at: Optional[datetime] = Field(
        default_factory=datetime.now(datetime.UTC),
        description="Timestamp when the record was last updated",
    )

    @field_validator("embedding")
    def check_embedding_length(cls, v):
        if v and len(v) != DEFAULT_EMBEDDING_CONFIG.embedding_length:
            raise ValueError(
                f"Embedding model {DEFAULT_EMBEDDING_CONFIG.model_name} must have a length of {DEFAULT_EMBEDDING_CONFIG.embedding_length}"
            )
        return v


class DocumentRecordCreate(DocumentRecordBase):
    """
    Model for creating a new document record. Inherits from DocumentRecordBase and includes all necessary fields for creating a new record.
    """

    pass


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

    query: Dict[str, str] = Field(..., description="Query parameters")
    page: Optional[int] = Field(1, description="Page number for pagination")
    page_size: Optional[int] = Field(10, description="Number of records per page")


class DocumentRecordResponse(BaseModel):
    """
    Response model for document record operations. Includes the unique identifier, message, and timestamps of the record.
    """

    id_: Optional[UUID] = Field(None, description="Unique identifier of the record")
    message: str = Field(..., description="Response message")
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
