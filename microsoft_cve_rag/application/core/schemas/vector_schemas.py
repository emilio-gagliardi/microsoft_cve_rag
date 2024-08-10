from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import Optional, List, Dict
from datetime import datetime, timezone
from uuid import UUID, uuid4
from application.config import DEFAULT_EMBEDDING_CONFIG


class VectorMetadata(BaseModel):
    """
    Metadata model for vectors. Includes CVE fixes, mentions, and tags.
    """

    cve_fixes: Optional[str] = None
    cve_mentions: Optional[str] = None
    tags: Optional[str] = None


class VectorRecordBase(BaseModel):
    """
    Base model for vector records. This model includes common fields that are shared across different vector record operations.
    """

    id_: Optional[UUID] = Field(
        default_factory=uuid4, description="Unique identifier of the record"
    )
    embedding: Optional[List[float]] = Field(None, description="Embedding vector")
    metadata: Optional[VectorMetadata] = Field(
        None, description="Metadata associated with the record"
    )
    relationships: Optional[Dict[str, str]] = Field(
        None, description="Relationships of the record"
    )
    excluded_embed_metadata_keys: Optional[List[str]] = Field(
        None, description="Metadata keys to exclude from embedding"
    )
    excluded_llm_metadata_keys: Optional[List[str]] = Field(
        None, description="Metadata keys to exclude from LLM"
    )
    text: Optional[str] = Field(None, description="Text associated with the record")
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


class VectorRecordCreate(VectorRecordBase):
    """
    Model for creating a new vector record. Inherits from VectorRecordBase and includes all necessary fields for creating a new record.
    """

    pass


class VectorRecordUpdate(VectorRecordBase):
    """
    Model for updating an existing vector record. Inherits from VectorRecordBase and includes all necessary fields for updating a record.
    """

    id_: UUID


class VectorRecordDelete(BaseModel):
    """
    Model for deleting a vector record. Includes the unique identifier of the record to be deleted.
    """

    id_: UUID


class VectorRecordQuery(BaseModel):
    """
    Model for querying vector records. Includes query parameters and pagination details.
    """

    query: Dict[str, str] = Field(..., description="Query parameters")
    page: Optional[int] = Field(1, description="Page number for pagination")
    page_size: Optional[int] = Field(10, description="Number of records per page")


class VectorRecordResponse(BaseModel):
    """
    Response model for vector record operations. Includes the unique identifier, message, and timestamps of the record.
    """

    id_: Optional[UUID] = Field(None, description="Unique identifier of the record")
    message: str = Field(..., description="Response message")
    created_at: Optional[datetime] = Field(
        None, description="Timestamp when the record was created"
    )
    updated_at: Optional[datetime] = Field(
        None, description="Timestamp when the record was last updated"
    )


class VectorRecordQueryResponse(BaseModel):
    """
    Response model for vector record queries. Includes the list of records matching the query, total count, and pagination details.
    """

    results: List[VectorRecordBase] = Field(
        ..., description="List of records matching the query"
    )
    total_count: int = Field(
        ..., description="Total count of records matching the query"
    )
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of records per page")


class BulkVectorRecordCreate(BaseModel):
    """
    Model for creating multiple vector records in bulk. Includes a list of VectorRecordCreate objects.
    """

    records: List[VectorRecordCreate]


class BulkVectorRecordDelete(BaseModel):
    """
    Model for deleting multiple vector records in bulk. Includes a list of unique identifiers of the records to be deleted.
    """

    ids: List[UUID]


class ErrorResponse(BaseModel):
    """
    Response model for errors. Includes the error message and error code.
    """

    error: str = Field(..., description="Error message")
    code: int = Field(..., description="Error code")
