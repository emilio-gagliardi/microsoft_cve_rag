# graph_node.py
# Purpose: This script defines what a GraphNode is in the application.
# Explanation: A GraphNode is an entity that represents nodes in a graph structure, which can be used in graph databases.
# In this case, a GraphNode can have an ID, a label, and properties.
# Regular Class vs. Pydantic: We use a regular class here because GraphNode may have methods to manipulate its properties.
# Relationships: This script may be imported by repositories and services that handle graph node data.
# Example Usage:
# from microsoft_cve_rag.domain.entities.graph_node import GraphNode


from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from uuid import UUID, uuid4
from datetime import datetime
from domain.entities.base_metadata import BaseMetadata


class GraphNodeMetadata(BaseMetadata):
    cve_fixes: Optional[str] = None
    cve_mentions: Optional[str] = None
    tags: Optional[str] = None


class GraphNode(BaseModel):
    id_: UUID = Field(default_factory=uuid4)
    embedding: Optional[List[float]] = None
    metadata: Optional[GraphNodeMetadata] = None
    relationships: Optional[Dict[str, str]] = None
    text: Optional[str] = None
    class_name: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True
