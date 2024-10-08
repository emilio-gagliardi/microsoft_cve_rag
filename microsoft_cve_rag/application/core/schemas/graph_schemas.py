from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from datetime import datetime, timezone
from uuid import UUID, uuid4
from application.app_utils import get_app_config
from application.core.schemas.document_schemas import DocumentRecordBase

settings = get_app_config()
graph_db_settings = settings["GRAPHDB_CONFIG"]


class BaseMetadata(BaseModel):
    """
    Base metadata model for documents. Includes common metadata fields.
    """

    revision: Optional[str] = Field(None, description="The version of msrc post types.")
    id: Optional[str] = Field(None, description="UUID string")
    post_id: Optional[str] = Field(
        None, description="Specific CVE identification ID. eg. CVE-2023-36435"
    )
    published: Optional[datetime] = Field(
        None,
        description="Publication date of the version. Multiple versions have multiple dates.",
    )
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

    @field_validator("published")
    def parse_published(cls, value):
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return value


class GraphNodeMetadata(BaseMetadata):
    """
    Metadata model for graph nodes. Includes additional fields specific to graph nodes.
    """

    conversationId: Optional[str] = Field(
        None, description="Conversation identifier for patch management emails"
    )
    subject: Optional[str] = Field(
        None, description="Subject of the patch management email"
    )
    receivedDateTime: Optional[datetime] = Field(
        None, description="The datetime when email was received by google groups."
    )
    cve_fixes: Optional[str] = Field(
        None, description="CVE fixes mentioned in the document"
    )
    cve_mentions: Optional[str] = Field(
        None, description="CVE mentions in the document"
    )
    tags: Optional[str] = Field(None, description="Tags associated with the document")

    class Config:
        json_schema_extra = {
            "example": {
                "conversationId": "123e4567-e89b-12d3-a456-426614174000",
                "subject": "Sample Subject",
                "receivedDateTime": "2024-07-15T00:00:00+00:00",
                "cve_fixes": "CVE-2023-1234",
                "cve_mentions": "CVE-2023-5678",
                "tags": "security, update",
            }
        }


class GraphRecordBase(BaseModel):
    """
    Base model for graph records. This model includes common fields that are shared across different graph record operations.
    """

    id: Optional[str] = Field(None, description="Graph db identifier")
    embedding: Optional[List[float]] = Field(None, description="Embedding vector")
    metadata: Optional[GraphNodeMetadata] = Field(
        None, description="Metadata associated with the record"
    )
    relationships: Optional[Dict[str, str]] = Field(
        None, description="Relationships of the record. LlamaIndex attribute."
    )
    excluded_embed_metadata_keys: Optional[List[str]] = Field(
        None,
        description="Metadata keys to exclude from embedding. LlamaIndex attribute.",
    )
    excluded_llm_metadata_keys: Optional[List[str]] = Field(
        None, description="Metadata keys to exclude from LLM. LlamaIndex attribute."
    )
    text: Optional[str] = Field(None, description="Text associated with the record")
    metadata_template: Optional[str] = Field(
        None, description="Template for metadata. LlamaIndex attribute."
    )
    metadata_separator: Optional[str] = Field(
        None, description="Separator for metadata. LlamaIndex attribute."
    )
    class_name: Optional[str] = Field(None, description="Class name of the record")
    document: Optional[DocumentRecordBase] = Field(
        None, description="The source document"
    )


class GraphRecordCreate(GraphRecordBase):
    """
    Model for creating a new graph record. Inherits from GraphRecordBase and includes all necessary fields for creating a new record.
    """

    metadata: GraphNodeMetadata = Field(
        ..., description="Metadata associated with the record"
    )
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Qdrant has Langchain integrations",
                "metadata": {
                    "conversationId": "123e4567-e89b-12d3-a456-426614174000",
                    "subject": "Langchain Integration",
                    "receivedDateTime": "2024-07-15T00:00:00+00:00",
                    "cve_fixes": "CVE-2023-1234",
                    "cve_mentions": "CVE-2023-5678",
                    "tags": "integration, langchain",
                    "revision": "1.0",
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "post_id": "CVE-2023-36435",
                    "published": "2024-07-15T00:00:00+00:00",
                    "title": "Langchain Integration",
                    "description": "Integration with Langchain for enhanced capabilities",
                    "build_numbers": [[19041, 19042]],
                    "impact_type": "High",
                    "product_build_ids": ["build_1234"],
                    "products": ["Product A"],
                    "severity_type": "Critical",
                    "summary": "This document describes the integration with Langchain.",
                    "collection": "documents",
                    "source": "https://example.com/langchain-integration",
                    "hash": "abc123",
                },
            }
        }


class GraphRecordUpdate(GraphRecordBase):
    """
    Model for updating an existing graph record. Inherits from GraphRecordBase and includes all necessary fields for updating a record.
    """

    text: Optional[str] = Field(None, description="Text content to update")
    metadata: Optional[GraphNodeMetadata] = Field(
        None, description="Metadata to update"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Qdrant has Langchain integrations. Some additional text was added.",
                "metadata": {
                    "revision": "1.5",
                    "description": "AN updated description is typical",
                    "summary": "This summary was generated by an LLM. It captures the important details of the document.",
                },
            }
        }


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
    message: str = Field(..., description="Response message from database")
    status: Optional[str] = Field(
        None, description="Status of the operation from database"
    )
    vector: Optional[List[float]] = Field(
        None, description="The embedding calculated for the document text"
    )
    payload: Optional[dict] = Field(None, description="The metadata of the document")


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
