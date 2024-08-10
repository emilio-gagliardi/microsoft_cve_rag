# document.py
# Purpose: This script defines what a Document is in the application.
# Explanation: A Document is an entity, which means it represents a real-world object with certain properties and behaviors.
# In this case, a Document can have a title, content, and a creation date.
# Regular Class vs. Pydantic: We use a regular class here because Document has behaviors (methods like summary) beyond just holding data.
# Relationships: This script may be imported by repositories and services that handle document data.
# Example Usage:
# from microsoft_cve_rag.domain.entities.document import Document

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from uuid import UUID, uuid4
from domain.entities.base_metadata import BaseMetadata


class DocumentMetadata(BaseMetadata):
    cve_fixes: Optional[str] = None
    cve_mentions: Optional[str] = None
    tags: Optional[str] = None
    added_to_vector_store: Optional[bool] = None
    added_to_summary_index: Optional[bool] = None
    added_to_graph_store: Optional[bool] = None


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
