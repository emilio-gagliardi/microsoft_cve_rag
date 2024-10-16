# import os
# import sys

# original_dir = os.getcwd()
# print(sys.path)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# print(sys.path)

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime, timezone
from bson import ObjectId


class BaseMetadata(BaseModel):
    revision: Optional[str] = None
    id: Optional[str] = None
    post_id: Optional[str] = None
    published: Optional[datetime] = None
    title: Optional[str] = None
    description: Optional[str] = None
    build_numbers: Optional[List[List[int]]] = None
    impact_type: Optional[str] = None
    product_build_ids: Optional[List[str]] = None
    products: Optional[List[str]] = None
    severity_type: Optional[str] = None
    summary: Optional[str] = None
    collection: Optional[str] = None
    source: Optional[str] = None
    hash: Optional[str] = None
    conversationId: Optional[str] = None
    subject: Optional[str] = None
    receivedDateTime: Optional[str] = None


class DocumentMetadata(BaseMetadata):
    id: str
    cve_fixes: Optional[str] = None
    cve_mentions: Optional[str] = None
    tags: Optional[str] = None
    added_to_vector_store: Optional[bool] = None
    added_to_summary_index: Optional[bool] = None
    added_to_graph_store: Optional[bool] = None

    class Config:
        from_attributes = True


class Document(BaseModel):
    _id: Optional[ObjectId] = None
    id_: str
    embedding: Optional[List[float]] = None
    metadata: DocumentMetadata
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
    product_mentions: Optional[List[str]] = None
    build_numbers: Optional[List[List[int]]] = None
    kb_mentions: Optional[List[str]] = None

    class Config:
        from_attributes = True


class GraphNodeMetadata(BaseMetadata):
    cve_fixes: Optional[str] = None
    cve_mentions: Optional[str] = None
    tags: Optional[str] = None

    class Config:
        from_attributes = True


class GraphNode(BaseModel):
    id_: str
    embedding: Optional[List[float]] = None
    metadata: Optional[GraphNodeMetadata] = None
    relationships: Optional[Dict[str, str]] = None
    text: Optional[str] = None
    class_name: Optional[str] = None
    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    class Config:
        from_attributes = True


class VectorMetadata(BaseMetadata):
    cve_fixes: Optional[str] = None
    cve_mentions: Optional[str] = None
    tags: Optional[str] = None

    class Config:
        from_attributes = True


class Vector(BaseModel):
    id_: str
    id: str
    embedding: Optional[List[float]] = None
    metadata: Optional[VectorMetadata] = None
    relationships: Optional[Dict[str, str]] = None
    excluded_embed_metadata_keys: Optional[List[str]] = None
    excluded_llm_metadata_keys: Optional[List[str]] = None
    text: Optional[str] = None
    metadata_template: Optional[str] = None
    metadata_separator: Optional[str] = None
    class_name: Optional[str] = None

    class Config:
        from_attributes = True
