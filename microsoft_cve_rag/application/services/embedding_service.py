# Purpose: Generate embeddings for text using various models
# Inputs: Text strings
# Outputs: Embeddings (numpy arrays)
# Dependencies: None (external: Ollama, Hugging Face, or OpenAI)

from typing import List

# import numpy as np
from pydantic import BaseModel
from application.config import PROJECT_CONFIG


class EmbeddingRequest(BaseModel):
    texts: List[str]


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]


class EmbeddingService:
    def __init__(self):
        self.model_name = PROJECT_CONFIG["DEFAULT_EMBEDDING_CONFIG"].model_name
        self.embedding_length = PROJECT_CONFIG[
            "DEFAULT_EMBEDDING_CONFIG"
        ].embedding_length
        self.model_card = PROJECT_CONFIG["DEFAULT_EMBEDDING_CONFIG"].model_card

    def generate_embedding(self, text: str) -> List[float]:
        # Example: Generate an embedding of the specified length
        return [0.1] * self.embedding_length
