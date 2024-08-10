# schemas/chat_schemas.py
from pydantic import BaseModel
from typing import List


class ChatQueryRequest(BaseModel):
    query: str


class ChatQueryResponse(BaseModel):
    response: str


class GenerateCompletionRequest(BaseModel):
    data: dict


class GenerateCompletionResponse(BaseModel):
    completion: str


class EmbeddingRequest(BaseModel):
    text: str


class EmbeddingResponse(BaseModel):
    embedding: List[float]
