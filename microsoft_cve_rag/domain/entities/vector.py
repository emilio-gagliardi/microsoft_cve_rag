# vector.py
# Purpose: This script defines what a Vector is in the application.
# Explanation: A Vector is an entity that represents numerical data used in machine learning models.
# In this case, a Vector can have a list of numbers and a creation date.
# Regular Class vs. Pydantic: We use a regular class here because Vector may need additional methods for operations.
# Relationships: This script may be imported by repositories and services that handle vector data.
# Example Usage:
# from microsoft_cve_rag.domain.entities.vector import Vector

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from uuid import UUID, uuid4
from datetime import datetime
from domain.entities.base_metadata import BaseMetadata


class VectorMetadata(BaseMetadata):
    cve_fixes: Optional[str] = None
    cve_mentions: Optional[str] = None
    tags: Optional[str] = None


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
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True
