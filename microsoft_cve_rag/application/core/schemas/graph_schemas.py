from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from datetime import datetime, timezone
from uuid import UUID, uuid4


class GraphNodeMetadata(BaseModel):
    """
    Metadata model for graph nodes. Includes CVE fixes, mentions, and tags.
    """

    cve_fixes: Optional[str] = None
    cve_mentions: Optional[str] = None
    tags: Optional[str] = None


class GraphRecordBase(BaseModel):
    """
    Base model for graph records. This model includes common fields that are shared across different graph record operations.
    """

    id_: Optional[UUID] = Field(
        default_factory=uuid4, description="Unique identifier of the record"
    )
    embedding: Optional[List[float]] = Field(None, description="Embedding vector")
    metadata: Optional[GraphNodeMetadata] = Field(
        None, description="Metadata associated with the record"
    )
    relationships: Optional[Dict[str, str]] = Field(
        None, description="Relationships of the record"
    )
    text: Optional[str] = Field(None, description="Text associated with the record")
    class_name: Optional[str] = Field(None, description="Class name of the record")
    created_at: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the record was created",
    )
    updated_at: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the record was last updated",
    )

    @field_validator("created_at", "updated_at", pre=True, always=True)
    def set_timestamps(cls, value, field):
        return value or datetime.now(timezone.utc)


class GraphRecordCreate(GraphRecordBase):
    """
    Model for creating a new graph record. Inherits from GraphRecordBase and includes all necessary fields for creating a new record.
    """

    pass


class GraphRecordUpdate(GraphRecordBase):
    """
    Model for updating an existing graph record. Inherits from GraphRecordBase and includes all necessary fields for updating a record.
    """

    id: str


class GraphRecordDelete(BaseModel):
    """
    Model for deleting a graph record. Includes the unique identifier of the record to be deleted.
    """

    id: str


class GraphRecordQuery(BaseModel):
    """
    Model for querying graph records. Includes query parameters and pagination details.
    """

    query: Dict[str, str] = Field(..., description="Query parameters")
    page: Optional[int] = Field(1, description="Page number for pagination")
    page_size: Optional[int] = Field(10, description="Number of records per page")


class GraphRecordResponse(BaseModel):
    """
    Response model for graph record operations. Includes the unique identifier, message, and timestamps of the record.
    """

    id: Optional[str] = Field(None, description="Unique identifier of the record")
    message: str = Field(..., description="Response message")
    created_at: Optional[datetime] = Field(
        None, description="Timestamp when the record was created"
    )
    updated_at: Optional[datetime] = Field(
        None, description="Timestamp when the record was last updated"
    )


class GraphRecordQueryResponse(BaseModel):
    """
    Response model for graph record queries. Includes the list of records matching the query, total count, and pagination details.
    """

    results: List[GraphRecordBase] = Field(
        ..., description="List of records matching the query"
    )
    total_count: int = Field(
        ..., description="Total count of records matching the query"
    )
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of records per page")


class QueryRequest(BaseModel):
    """
    Request model for executing custom Cypher queries. Includes the Cypher query string and parameters.
    """

    cypher: str
    parameters: Dict


class QueryResponse(BaseModel):
    """
    Response model for custom Cypher queries. Includes the results of the query.
    """

    results: List[Dict]
