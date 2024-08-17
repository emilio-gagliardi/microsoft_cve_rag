# Purpose: Manage vector database operations
# Inputs: Text data or embeddings
# Outputs: Search results, insertion status
# Dependencies: EmbeddingService

from typing import Optional
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from application.services.embedding_service import EmbeddingService
from application.core.schemas.vector_schemas import (
    VectorRecordCreate,
    VectorRecordUpdate,
)
from application.services.embedding_service import (
    QdrantDefaultProvider,
    FastEmbedProvider,
    OllamaProvider,
)
from application.app_utils import get_app_config, get_vector_db_credentials

# TODO: add vector_persist_file_path to .env
# QdrantClient(path="path/to/db") # Persists changes to disk

settings = get_app_config()
vector_db_settings = settings.get("VECTORDB_CONFIG")


class VectorDBService:
    def __init__(
        self,
        collection: str,
        embedding_config: dict,
        vectordb_config: dict,
        distance_metric: Optional[str] = None,
    ):
        self.credentials = get_vector_db_credentials()
        self.embedding_config = embedding_config
        self.vectordb_config = vectordb_config
        self.sync_client = QdrantClient(
            host=self.credentials.host, port=self.credentials.port
        )
        self.async_client = AsyncQdrantClient(
            host=self.credentials.host, port=self.credentials.port
        )
        self.embedding_service = self._create_embedding_service()
        self._collection = vectordb_config[collection]
        self._distance_metric = Distance.COSINE
        self._embedding_length = self.embedding_service.embedding_length

    @property
    def collection(self) -> str:
        return self._collection

    @property
    def distance_metric(self) -> str:
        return self._distance_metric

    @property
    def embedding_length(self) -> str:
        return self._embedding_length

    def ensure_collection_exists(self):
        if not self.sync_client.collection_exists(self.collection):
            self.sync_client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.embedding_length, distance=Distance.COSINE
                ),
            )
            print(f"Collection '{self.collection}' created successfully.")
        else:
            print(f"Collection '{self.collection}' already exists.")

    async def ensure_collection_exists_async(self):
        print("Collection check...")
        if not await self.async_client.collection_exists(self.collection):
            await self.async_client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.embedding_length, distance=Distance.COSINE
                ),
            )
            print(f"Collection '{self.collection}' created successfully.")
        else:
            print(f"Collection '{self.collection}' already exists.")

    def _create_embedding_service(self) -> EmbeddingService:
        provider_type = self.embedding_config.get(
            "embedding_provider", "qdrant_default"
        )
        if provider_type == "qdrant_default":
            provider = QdrantDefaultProvider(self.sync_client, self.async_client)
        elif provider_type == "fastembed":
            provider = FastEmbedProvider()
        elif provider_type == "ollama":
            provider = OllamaProvider()
        else:
            raise ValueError(f"Unsupported embedding provider: {provider_type}")
        return EmbeddingService(provider)

    async def create_vector(self, vector: VectorRecordCreate) -> str:
        vector_dict = vector.model_dump()
        if "id_" in vector_dict:
            vector_dict["id_"] = str(vector_dict["id_"])
        embedding = await self.embedding_service.generate_embeddings_async(vector.text)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding[0],
            payload={"metadata": vector.metadata.model_dump(), "text": vector.text},
        )
        result = await self.async_client.upsert(
            collection_name=self.collection, points=[point]
        )
        return str(result.upserted_ids[0])

    async def get_vector(self, vector_id: str) -> Optional[dict]:
        result = await self.async_client.retrieve(
            collection_name=self.collection, ids=[vector_id]
        )
        return result[0] if result else None

    async def update_vector(self, vector_id: str, vector: VectorRecordUpdate) -> int:
        vector_dict = vector.model_dump()
        if "id_" in vector_dict:
            vector_dict["id_"] = str(vector_dict["id_"])
        embedding = await self.embedding_service.generate_embeddings_async(vector.text)
        point = PointStruct(
            id=vector_id,
            vector=embedding[0],
            payload={"metadata": vector.metadata.model_dump(), "text": vector.text},
        )
        result = await self.async_client.upsert(
            collection_name=self.collection, points=[point]
        )
        return len(result.upserted_ids)

    async def delete_vector(self, vector_id: str) -> int:
        result = await self.async_client.delete(
            collection_name=self.collection,
            points_selector=rest.PointIdsList(points=[vector_id]),
        )
        return result.deleted_count

    async def aclose(self):
        """Asynchronously close the Qdrant client connections."""
        if self.async_client:
            await self.async_client.close()
        if self.sync_client:
            self.sync_client.close()

    def __del__(self):
        """Destructor to ensure the sync client is closed when the object is garbage collected."""
        if self.sync_client:
            self.sync_client.close()
