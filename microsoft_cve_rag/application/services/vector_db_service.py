# Purpose: Manage vector database operations
# Inputs: Text data or embeddings
# Outputs: Search results, insertion status
# Dependencies: EmbeddingService
from typing import Optional, Union, Any, List
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    PointIdsList,
    WriteOrdering,
)
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
from typing import Optional, List, Union
from fastapi import HTTPException

import logging

logger = logging.getLogger(__name__)

# TODO: add vector_persist_file_path to .env
# QdrantClient(path="path/to/db") # Persists changes to disk

settings = get_app_config()
vector_db_settings = settings.get("VECTORDB_CONFIG")


class VectorDBService:
    """
    Distance Metrics: ['Cosine','Dot','Euclid','Manhattan']
    """

    def __init__(
        self,
        embedding_config: dict,
        vectordb_config: dict,
        collection: str = None,
        distance_metric: Optional[str] = None,
    ):
        distance_mapping = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclid": Distance.EUCLID,
            "manhattan": Distance.MANHATTAN,
        }
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
        self._collection = collection or self.vectordb_config["tier1_collection"]
        self._distance_metric = distance_mapping.get(
            distance_metric or self.vectordb_config["distance_metric"], Distance.COSINE
        )
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
        return EmbeddingService.from_provider_name(
            provider_type, sync_client=self.sync_client, async_client=self.async_client
        )

    async def create_vector(self, vector: VectorRecordCreate) -> str:
        """
        Note. result from qdrant is <class 'qdrant_client.http.models.models.UpdateResult'>

        Args:
            vector (VectorRecordCreate): _description_

        Returns:
            str: _description_
        """

        vector_dict = vector.model_dump()

        if "id_" in vector_dict:
            vector_dict["id_"] = str(vector_dict["id_"])
        # Embedding providers generate different response classes
        # make sure to confirm the response class per embedding provider
        # Fastembed returns a generator but in the implementation a list is returned
        embedding = await self.embedding_service.generate_embeddings_async(vector.text)
        if len(embedding[0]) != self.embedding_length:
            raise ValueError(
                f"Expected vector dimension {self.embedding_length}, got {len(embedding)}"
            )

        point_id = str(uuid.uuid4())
        print(f"point_id is made {point_id}")
        point = PointStruct(
            id=point_id,
            vector=embedding[0],
            payload={"metadata": vector.metadata.model_dump(), "text": vector.text},
        )
        print(f"point is made {point}")
        result = await self.async_client.upsert(
            collection_name=self.collection, points=[point]
        )
        print(f"Upsert result: {result}")
        return {
            "point_id": point_id,
            "status": result.status.value,
        }

    async def get_vector(
        self,
        vector_id: str,
        with_payload: Union[bool, List[str]] = True,
        with_vectors: Union[bool, List[str]] = True,
    ) -> Optional[dict]:
        result = await self.async_client.retrieve(
            collection_name=self.collection,
            ids=[vector_id],
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

        return result[0] if result else None

    async def update_vector(self, vector_id: str, vector: VectorRecordUpdate) -> int:
        point_data = {"id": vector_id}
        payload = {}
        try:
            # Fetch existing vector
            existing_vectors = await self.async_client.retrieve(
                collection_name=self.collection,
                ids=[vector_id],
                with_payload=True,
                with_vectors=True,
            )
            if not existing_vectors:
                raise ValueError(f"No vector found with id {vector_id}")
            existing_vector = existing_vectors[0]
            existing_payload = existing_vector.payload or {}

            # Handle text and embeddings
            if vector.text is not None:
                embedding = await self.embedding_service.generate_embeddings_async(
                    vector.text
                )

                if len(embedding[0]) != self.embedding_length:
                    raise ValueError(
                        f"Expected vector dimension {self.embedding_length}, got {len(embedding)}"
                    )

                point_data["vector"] = embedding[0]
                payload["text"] = vector.text

            # Handle metadata
            if vector.metadata:
                new_metadata = vector.metadata.model_dump(exclude_none=True)
                existing_metadata = existing_payload.get("metadata", {})
                # Merge existing metadata with new metadata
                payload["metadata"] = {**existing_metadata, **new_metadata}
            else:
                # Preserve existing metadata if no new metadata is provided
                payload["metadata"] = existing_payload.get("metadata", {})

            # Preserve other existing payload fields
            for key, value in existing_payload.items():
                if key not in payload:
                    payload[key] = value

            # If payload is not empty, add it to point_data
            if payload:
                point_data["payload"] = payload

            # If no updates are provided, raise an exception
            if len(point_data) == 1:  # Only contains 'id'
                raise ValueError("No updates provided for the vector")

            # Create PointStruct with the merged data
            point = PointStruct(**point_data)

            result = await self.async_client.upsert(
                collection_name=self.collection,
                points=[point],
                wait=True,  # Ensure the operation is completed before returning
            )

            return {"operation_id": {result.operation_id}, "status": {result.status}}

        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # Log the exception for debugging purposes
            print(f"An error occurred: {e}")
            raise HTTPException(status_code=500, detail="An unexpected error occurred")

    async def delete_vector(
        self,
        vector_id: str,
        collection_name: str = None,
        wait: bool = True,
        ordering: Optional[WriteOrdering] = None,
        shard_key_selector: Optional[Union[int, str, List[Union[int, str]]]] = None,
        **kwargs: Any,
    ) -> int:
        try:
            result = await self.async_client.delete_vectors(
                collection_name=collection_name if collection_name else self.collection,
                vectors=[""],
                points=PointIdsList(points=[vector_id]),
                wait=wait,
                ordering=ordering,
                shard_key_selector=shard_key_selector,
                **kwargs,
            )

            return {"operation_id": {result.operation_id}, "status": {result.status}}
        except Exception as e:
            # Log the exception for debugging purposes
            print(f"An error occurred while deleting the vector: {e}")
            raise HTTPException(
                status_code=500,
                detail="An unexpected error occurred while deleting the vector",
            )

    async def delete_point(
        self,
        vector_id: Optional[str] = None,
        metadata_id: Optional[str] = None,
        collection_name: str = None,
        wait: bool = True,
        ordering: Optional[WriteOrdering] = None,
        shard_key_selector: Optional[Union[int, str, List[Union[int, str]]]] = None,
        **kwargs: Any,
    ) -> dict:
        try:
            if vector_id:
                # Use vector_id directly if provided
                points_selector = [vector_id]
            elif metadata_id:
                # Construct filter based on the provided metadata_id
                points_selector = {
                    "filter": {"must": [{"key": "id", "match": {"value": metadata_id}}]}
                }

            # Perform the delete operation
            result = await self.async_client.delete(
                collection_name=collection_name if collection_name else self.collection,
                points_selector=points_selector,
                wait=wait,
                ordering=ordering,
                shard_key_selector=shard_key_selector,
                **kwargs,
            )

            return {"operation_id": result.operation_id, "status": result.status}
        except Exception as e:
            # Log the exception for debugging purposes
            print(f"An error occurred while deleting the point: {e}")
            raise HTTPException(
                status_code=500,
                detail="An unexpected error occurred while deleting the point",
            )

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
