# Purpose: Manage vector database operations
# Inputs: Text data or embeddings
# Outputs: Search results, insertion status
# Dependencies: EmbeddingService
import json
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
# from application.services.embedding_service import EmbeddingService
from application.core.schemas.vector_schemas import (
    VectorRecordCreate,
    VectorRecordUpdate,
)
# from application.services.embedding_service import (
#     QdrantDefaultProvider,
#     FastEmbedProvider,
#     OllamaProvider,
# )
from application.app_utils import get_app_config, get_vector_db_credentials
from typing import Optional, List, Union, Dict, Any
from fastapi import HTTPException

import logging

# TODO: add vector_persist_file_path to .env
# QdrantClient(path="path/to/db") # Persists changes to disk

settings = get_app_config()
vector_db_settings = settings.get("VECTORDB_CONFIG")
logging.getLogger(__name__)

class VectorDBService:
    """
    A service class that manages vector database operations using Qdrant as the backend.
    
    This service handles the creation, retrieval, update, and deletion of vector embeddings
    in a Qdrant database. It serves as a bridge between the application and the vector store,
    integrating with LlamaIndex by providing compatible vector storage and retrieval capabilities.

    Key Features:
    - Manages Qdrant collections for vector storage
    - Handles vector embeddings generation through configurable embedding providers
    - Supports both synchronous and asynchronous operations
    - Provides CRUD operations for vector records
    
    Initialization Process:
    1. Establishes connections to Qdrant (both sync and async clients)
    2. Creates an embedding service based on the specified provider
    3. Initializes or connects to the specified collection
    4. Configures distance metrics for similarity search

    Integration with LlamaIndex:
    - Compatible with LlamaIndex's VectorStore interface
    - Stores document embeddings that can be used by LlamaIndex for retrieval
    - Supports metadata storage alongside vectors for rich document retrieval

    Args:
        embedding_config (dict): Configuration for the embedding service including:
            - embedding_provider: The provider to use (e.g., "qdrant_default", "fast_embed", "ollama")
            - model_name: The specific model to use for embeddings
        vectordb_config (dict): Configuration for the vector database including:
            - tier1_collection: Default collection name
            - distance_metric: Similarity metric to use
        collection (str, optional): The name of the collection to use. Defaults to tier1_collection from config.
        distance_metric (str, optional): The distance metric to use for similarity search.
            Options: ['cosine', 'dot', 'euclid', 'manhattan']. Defaults to 'cosine'.

    Example:
        ```python
        embedding_config = {
            "embedding_provider": "fast_embed",
            "model_name": "BAAI/bge-small-en-v1.5"
        }
        vectordb_config = {
            "tier1_collection": "my_collection",
            "distance_metric": "cosine"
        }
        
        vector_service = VectorDBService(
            embedding_config=embedding_config,
            vectordb_config=vectordb_config
        )
        ```
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
        self._embedding_service = None
        self._collection = collection or self.vectordb_config["tier1_collection"]
        self._distance_metric = distance_mapping.get(
            distance_metric or self.vectordb_config["distance_metric"], Distance.COSINE
        )

    @property
    def collection(self) -> str:
        return self._collection

    @property
    def distance_metric(self) -> str:
        return self._distance_metric

    @property
    def embedding_length(self) -> int:
        """
        Get embedding length, prioritizing initialized service over config
        """
        if self._embedding_service is not None:
            return self._embedding_service.embedding_length
        return self._get_embedding_length()
    
    @property
    def embedding_service(self):
        if self._embedding_service is None:
            self._embedding_service = self._create_embedding_service()
        return self._embedding_service

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
                    size=self.embedding_length,
                    distance=self._distance_metric,
                    # "text": VectorParams(size=300, distance=Distance.COSINE),
                    # "thread_id": VectorParams(size=50, distance=Distance.EUCLID),
                ),
            )
            print(f"Collection '{self.collection}' created successfully.")
        else:
            print(f"Collection '{self.collection}' already exists.")

    def _get_embedding_length(self) -> int:
        """Get embedding length based on selected provider"""
        provider = self.embedding_config.get("embedding_provider", "fastembed")
        provider_length_key = f"{provider}_embedding_length"
        
        # Get provider-specific length or fall back to vector_db_embedding_length
        return self.embedding_config.get(
            provider_length_key,
            self.embedding_config.get("vector_db_embedding_length", 1024)
        )
        
    def _create_embedding_service(self):
        """Create embedding service based on config"""
        provider = self.embedding_config.get("embedding_provider", "fastembed")
        
        if provider == "fastembed":
            from application.services.embedding_service import FastEmbedService
            return FastEmbedService(self.embedding_config)
        elif provider == "ollama":
            from application.services.embedding_service import OllamaEmbedService
            return OllamaEmbedService(self.embedding_config)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

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

    async def bulk_delete_vectors(
        self,
        vector_ids: List[str],
        collection_name: str = None,
        wait: bool = True,
        ordering: Optional[WriteOrdering] = None,
        **kwargs: Any,
    ) -> int:
        """
        Delete multiple vectors by their IDs.
        
        Args:
            vector_ids: List of vector IDs to delete
            collection_name: Optional collection name override
            wait: Whether to wait for operation completion
            ordering: Optional write ordering configuration
            
        Returns:
            int: Number of vectors successfully deleted
        """
        try:
            result = await self.async_client.delete(
                collection_name=collection_name if collection_name else self.collection,
                points_selector=vector_ids,
                wait=wait,
                ordering=ordering,
                **kwargs
            )
            
            return {
                "operation_id": result.operation_id,
                "status": result.status
            }

        except Exception as e:
            logging.error(f"Error in bulk_delete_vectors: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete vectors: {str(e)}"
            )

    async def bulk_delete_points(
        self,
        point_ids: List[str],
        collection_name: str = None,
        wait: bool = True,
        ordering: Optional[WriteOrdering] = None,
        **kwargs: Any,
    ) -> int:
        """
        Delete multiple points by their IDs.
        
        Args:
            point_ids: List of point IDs to delete
            collection_name: Optional collection name override
            wait: Whether to wait for operation completion
            ordering: Optional write ordering configuration
            
        Returns:
            int: Number of points successfully deleted
        """
        try:
            result = await self.async_client.delete(
                collection_name=collection_name if collection_name else self.collection,
                points_selector=point_ids,
                wait=wait,
                ordering=ordering,
                **kwargs
            )
            
            return {
                "operation_id": result.operation_id,
                "status": result.status
            }

        except Exception as e:
            logging.error(f"Error in bulk_delete_points: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete points: {str(e)}"
            )

    async def delete_all_points(self) -> dict:
        """
        Delete all points from the collection while preserving the collection structure.
        
        Returns:
            dict: Operation status including operation_id and status
        """
        try:
            # Get all point IDs from the collection
            scroll_result = await self.async_client.scroll(
                collection_name=self.collection,
                limit=10000,
                with_payload=False,
                with_vectors=False
            )
            
            points = scroll_result[0]  # First element contains the points
            if not points:
                return {
                    "operation_id": "no_points",
                    "status": "success",
                    "message": "No points to delete"
                }
            
            # Extract point IDs
            point_ids = [point.id for point in points]
            
            # Delete all points
            result = await self.async_client.delete(
                collection_name=self.collection,
                points_selector=point_ids,
                wait=True
            )
            
            return {
                "operation_id": result.operation_id,
                "status": result.status,
                "message": f"Successfully deleted {len(point_ids)} points"
            }

        except Exception as e:
            logging.error(f"Error deleting all points: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete all points: {str(e)}"
            )

    async def get_points_with_vector(self, vector: List[float], collection_name: str) -> List[Dict]:
        """Get all points that match a specific vector."""
        results = await self.async_client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=100,
            score_threshold=0.9999,
            with_payload=True,
            with_vectors=True
        )
        
        with open('vector_matches.txt', 'w', encoding='utf-8') as f:
            f.write(f"Points matching vector {vector[:5]}...\n\n")
            for point in results:
                f.write(json.dumps(point.payload, indent=2))
                f.write("\n" + "="*80 + "\n")
        
        return results

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
