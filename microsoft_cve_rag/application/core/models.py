from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4
from typing import Optional, List, Dict
from datetime import datetime
from application.config import DEFAULT_EMBEDDING_CONFIG


class Published(BaseModel):
    date: Optional[datetime] = None


class BaseMetadata(BaseModel):
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
    cve_fixes: Optional[str] = None
    cve_mentions: Optional[str] = None
    tags: Optional[str] = None
    added_to_vector_store: Optional[bool] = None
    added_to_summary_index: Optional[bool] = None
    added_to_graph_store: Optional[bool] = None

    class Config:
        orm_mode = True


class Document(BaseModel):
    _id: Optional[Dict[str, str]] = None
    id_: UUID = Field(default_factory=uuid4)
    embedding: Optional[List[float]] = None
    metadata: Optional[DocumentMetadata] = None
    excluded_embed_metadata_keys: Optional[List[str]] = None
    excluded_llm_metadata_keys: Optional[List[str]] = None
    relationships: Optional[Dict[str, str]] = None
    text: Optional[str] = None
    start_char_idx: Optional[int] = None
    end_char_idx: Optional[int] = None
    text_template: Optional[str] = None
    metadata_template: Optional[str] = None
    metadata_separator: Optional[str] = None
    class_name: Optional[str] = None

    class Config:
        orm_mode = True


class GraphNodeMetadata(BaseMetadata):
    cve_fixes: Optional[str] = None
    cve_mentions: Optional[str] = None
    tags: Optional[str] = None

    class Config:
        orm_mode = True


class GraphNode(BaseModel):
    id_: UUID = Field(default_factory=uuid4)
    embedding: Optional[List[float]] = None
    metadata: Optional[GraphNodeMetadata] = None
    relationships: Optional[Dict[str, str]] = None
    text: Optional[str] = None
    class_name: Optional[str] = None
    created_at: Optional[datetime] = Field(default_factory=datetime.now(datetime.UTC))
    updated_at: Optional[datetime] = Field(default_factory=datetime.now(datetime.UTC))

    class Config:
        orm_mode = True

    @validator("embedding")
    def check_embedding_length(cls, v):
        if v and len(v) != DEFAULT_EMBEDDING_CONFIG.embedding_length:
            raise ValueError(
                f"Embedding model {DEFAULT_EMBEDDING_CONFIG.model_name} must have a length of {DEFAULT_EMBEDDING_CONFIG.embedding_length}"
            )
        return v


class VectorMetadata(BaseMetadata):
    cve_fixes: Optional[str] = None
    cve_mentions: Optional[str] = None
    tags: Optional[str] = None

    class Config:
        orm_mode = True


class Vector(BaseModel):
    id_: UUID = Field(default_factory=uuid4)
    embedding: Optional[List[float]] = None
    metadata: Optional[VectorMetadata] = None
    relationships: Optional[Dict[str, str]] = None
    excluded_embed_metadata_keys: Optional[List[str]] = None
    excluded_llm_metadata_keys: Optional[List[str]] = None
    text: Optional[str] = None
    metadata_template: Optional[str] = None
    metadata_separator: Optional[str] = None
    class_name: Optional[str] = None
    created_at: Optional[datetime] = Field(default_factory=datetime.now(datetime.UTC))
    updated_at: Optional[datetime] = Field(default_factory=datetime.now(datetime.UTC))

    class Config:
        orm_mode = True

    @validator("embedding")
    def check_embedding_length(cls, v):
        if v and len(v) != DEFAULT_EMBEDDING_CONFIG.embedding_length:
            raise ValueError(
                f"Embedding model {DEFAULT_EMBEDDING_CONFIG.model_name} must have a length of {DEFAULT_EMBEDDING_CONFIG.embedding_length}"
            )
        return v
