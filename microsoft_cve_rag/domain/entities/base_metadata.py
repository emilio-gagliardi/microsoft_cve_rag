from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID
from datetime import datetime


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
