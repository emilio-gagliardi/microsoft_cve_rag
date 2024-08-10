# Purpose: Manage vector database operations
# Inputs: Text data or embeddings
# Outputs: Search results, insertion status
# Dependencies: EmbeddingService

from typing import List
from qdrant_client import QdrantClient
from application.services.embedding_service import EmbeddingService
from application.core.models import Vector
from application.app_utils import get_vector_db_credentials


class VectorDBService:
    def __init__(self):
        credentials = get_vector_db_credentials()
        self.client = QdrantClient(host=credentials.host, port=credentials.port)

    def create_vector(self, vector: Vector):
        result = self.client.upsert(
            collection_name="vectors",
            points=[
                {
                    "id": str(vector.id_),
                    "vector": vector.embedding,
                    "payload": vector.model_dump(),
                }
            ],
        )
        return result

    def get_vector(self, vector_id: str):
        result = self.client.retrieve(collection_name="vectors", ids=[vector_id])
        return result

    def update_vector(self, vector_id: str, vector: Vector):
        result = self.client.upsert(
            collection_name="vectors",
            points=[
                {
                    "id": vector_id,
                    "vector": vector.embedding,
                    "payload": vector.model_dump(),
                }
            ],
        )
        return result

    def delete_vector(self, vector_id: str):
        result = self.client.delete(
            collection_name="vectors", points_selector={"ids": [vector_id]}
        )
        return result
