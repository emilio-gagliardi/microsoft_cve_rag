# chat_service.py
# Purpose: This script defines the interface (or blueprint) for services that handle chat operations.
# Explanation: An interface in this context defines methods that must be implemented by any class that uses this interface.
# It ensures consistency and standardization across different chat services.
# Relationships: This script may be implemented by infrastructure services like openai_service.py, groq_service.py, and ollama_service.py.
# Example Usage:
# from microsoft_cve_rag.domain.services.chat_service import ChatService

from abc import ABC, abstractmethod


class ChatService(ABC):
    @abstractmethod
    def send_message(self, message: str) -> str:
        pass
