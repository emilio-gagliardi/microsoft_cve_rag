from pydantic import BaseModel


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
