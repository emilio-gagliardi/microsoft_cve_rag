# embedding_service.py
# Purpose: This script defines the interface (or blueprint) for services that generate embeddings.
# Explanation: An interface in this context defines methods that must be implemented by any class that uses this interface.
# It ensures consistency and standardization across different embedding services.
# Relationships: This script may be implemented by infrastructure services like openai_service.py, groq_service.py, and ollama_service.py.
# Example Usage:
# from microsoft_cve_rag.domain.services.embedding_service import EmbeddingService

from abc import ABC, abstractmethod
from typing import List


class EmbeddingService(ABC):
    @abstractmethod
    def generate(self, text: str) -> List[float]:
        pass
