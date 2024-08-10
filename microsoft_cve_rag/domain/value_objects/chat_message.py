# chat_message.py
# Purpose: This script defines the ChatMessage value object.
# Explanation: A value object is an object that contains certain values and is used to pass data around.
# In this case, a ChatMessage has the content of the message and a timestamp.
# Regular Class vs. Pydantic: We use a Pydantic model here because it provides strong data validation and serialization capabilities.
# Relationships: This script may be used by services and repositories that handle chat data.
# Example Usage:
# from microsoft_cve_rag.domain.value_objects.chat_message import ChatMessage

from datetime import datetime
from pydantic import BaseModel


class ChatMessage(BaseModel):
    content: str
    timestamp: datetime = None
