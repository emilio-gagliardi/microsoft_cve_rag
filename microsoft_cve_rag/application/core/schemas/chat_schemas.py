# schemas/chat_schemas.py
from typing import List

from pydantic import BaseModel


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
