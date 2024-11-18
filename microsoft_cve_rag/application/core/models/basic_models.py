# import os
# import sys

# original_dir = os.getcwd()
# print(sys.path)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# print(sys.path)

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from datetime import datetime, timezone
from bson import ObjectId
import pandas as pd

class BaseMetadata(BaseModel):
    id: str  # This is required for all documents
    # Only include fields that are truly common across ALL document types
    model_config = {
        "extra": "allow",  # Allow additional fields at runtime
        "from_attributes": True
    }

class DocumentMetadata(BaseMetadata):

    added_to_vector_store: Optional[bool] = None
    added_to_summary_index: Optional[bool] = None
    added_to_graph_store: Optional[bool] = None
    model_config = {
        "extra": "allow",  # Allow additional fields at runtime
        "arbitrary_types_allowed": True,
        "from_attributes": True
    }
    @field_validator('*')
    def convert_nan_to_none(cls, v):
        if pd.api.types.is_float(v) and pd.isna(v):
            return None
        return v


class Document(BaseModel):
    _id: Optional[ObjectId] = None
    id_: str
    embedding: Optional[List[float]] = None
    metadata: DocumentMetadata
    text: Optional[str] = None
    
    model_config = {
        "extra": "allow",  # Allow extra fields
        "arbitrary_types_allowed": True,  # Allow arbitrary types in metadata
        "from_attributes": True  # Replaces class Config
    }


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
