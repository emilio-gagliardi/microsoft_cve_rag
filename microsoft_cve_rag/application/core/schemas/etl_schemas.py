from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict, Any


class ETLJobConfig(BaseModel):
    source: str
    destination: str
    transform_type: str


class ETLJobStatus(BaseModel):
    job_id: str
    status: str


class ETLRunResponse(BaseModel):
    message: str


class GenerateEmbeddingsRequest(BaseModel):
    data: dict


class GenerateEmbeddingsResponse(BaseModel):
    embeddings: dict


class FullETLRequest(BaseModel):
    start_date: datetime
    end_date: Optional[datetime] = None

    class Config:
        json_schema_extra = {
            "example": {
                "start_date": "2024-07-01T00:00:00+00:00",
                "end_date": "2024-07-30T00:00:00+00:00",
            }
        }
