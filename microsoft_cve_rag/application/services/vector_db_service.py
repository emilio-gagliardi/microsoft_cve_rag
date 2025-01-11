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
import asyncio
import random
import time

import logging

# TODO: add vector_persist_file_path to .env
# QdrantClient(path="path/to/db") # Persists changes to disk

settings = get_app_config()
vector_db_settings = settings.get("VECTORDB_CONFIG")
logging.getLogger(__name__)


class VectorDBService:
    """
    A service class for managing vector database operations with optimized performance.
    
    Features:
    - Optimized collection configuration for insert/search performance
    - Batch processing with automatic size optimization
    - Dynamic parameter management for different workloads
    - Payload index management for filtered queries
    - Error handling with retry logic
    
    Usage:
        # Initialize service
        service = VectorDBService(
            embedding_config={
                "embedding_provider": "fastembed",
                "fastembed_model_name": "BAAI/bge-small-en-v1.5"
            },
            vectordb_config={
                "tier1_collection": "my_collection"
            }
        )
        
        # Bulk insert with progress tracking
        result = await service.upsert_points(
            points=points_list,
            batch_size=1000,
            wait=False,
            show_progress=True
        )
        
        # Switch to search optimization
        await service.update_collection_params(optimize_for="search")
        
        # Create indexes for filtered fields
        await service.create_payload_index(
            field_name="post_type",
            field_schema={"type": "keyword"}
        )
    
    Configuration:
        embedding_config:
            embedding_provider: Provider for embeddings (e.g., "fastembed")
            fastembed_model_name: Name of the FastEmbed model
            
        vectordb_config:
            tier1_collection: Name of the collection
            distance_metric: Distance metric for vectors (default: "cosine")
            hnsw_config: HNSW index configuration
            optimizer_config: Optimizer settings for resource management
    """
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
        """
        Ensures a collection exists with optimized configuration for vector operations.

        The method configures:
        1. HNSW index parameters for optimal insert/search balance
        2. Optimizer settings for efficient resource usage
        3. Payload schema for better filtering performance

        Configuration can be customized via vectordb_config or falls back to optimized defaults.
        """
        print("Collection check...")
        if not await self.async_client.collection_exists(self.collection):
            # Get HNSW config from vectordb_config or use optimized defaults
            hnsw_config = self.vectordb_config.get("hnsw_config", {
                "m": 16,  # Lower than default for faster inserts
                "ef_construct": 32,  # Lower than default (100) for faster inserts
                "full_scan_threshold": 10000,
                "max_indexing_threads": 4,
                "on_disk": False  # Keep in memory for better performance
            })

            # Get optimizer config from vectordb_config or use optimized defaults
            optimizer_config = self.vectordb_config.get("optimizer_config", {
                "indexing_threshold": 50000,  # Higher threshold to batch index operations
                "memmap_threshold": 10000,
                "vacuum_min_vector_number": 1000,
                "default_segment_number": 2,
                "flush_interval_sec": 30
            })

            # Define payload schema for better filtering
            payload_schema = self.vectordb_config.get("payload_schema", {
                # Lists
                "build_numbers": {
                    "type": "integer",  # Individual numbers in the list
                    "index": True
                },
                "kb_ids": {
                    "type": "keyword",  # Exact string matching
                    "index": True
                },
                "products": {
                    "type": "keyword",  # Exact string matching
                    "index": True
                },

                # String fields with exact matching
                "collection": {
                    "type": "keyword",
                    "index": True
                },
                "cwe_id": {
                    "type": "keyword",
                    "index": True
                },
                "cwe_name": {
                    "type": "keyword",
                    "index": True
                },
                "post_id": {
                    "type": "keyword",
                    "index": True
                },
                "post_type": {
                    "type": "keyword",
                    "index": True
                },
                "entity_type": {
                    "type": "keyword",
                    "index": True
                },
                "source_type": {
                    "type": "keyword",
                    "index": True
                },
                "symptom_label": {
                    "type": "keyword",
                    "index": True
                },
                "cause_label": {
                    "type": "keyword",
                    "index": True
                },
                "fix_label": {
                    "type": "keyword",
                    "index": True
                },
                "tool_label": {
                    "type": "keyword",
                    "index": True
                },
                "tool_url": {
                    "type": "keyword",
                    "index": True
                },
                "tags": {
                    "type": "keyword",
                    "index": True
                },
                "severity_type": {
                    "type": "keyword",
                    "index": True
                },

                # Text fields for full-text search
                "nvd_description": {
                    "type": "text",
                    "index": True
                },
                "title": {
                    "type": "text",
                    "index": True
                },

                # Date field
                "published": {
                    "type": "keyword",  # Store as string for exact matching
                    "index": True
                },

                # Keep generic metadata field for flexibility
                "metadata": {
                    "type": "object"
                }
            })

            try:
                await self.async_client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=self.embedding_length,
                        distance=self._distance_metric,
                        hnsw_config=self._validate_hnsw_config(hnsw_config),
                        optimizers_config=self._validate_optimizer_config(optimizer_config),
                        on_disk=hnsw_config.get("on_disk", False)
                    ),
                    payload_schema=self._validate_payload_schema(payload_schema)
                )
                print(f"Collection '{self.collection}' created successfully with optimized parameters:")
                print(f"- HNSW Config: {hnsw_config}")
                print(f"- Optimizer Config: {optimizer_config}")
                print(f"- Payload Schema: {payload_schema}")
            except Exception as e:
                print(f"Error creating collection: {str(e)}")
                # Attempt to create with minimal config if optimized fails
                await self.async_client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=self.embedding_length,
                        distance=self._distance_metric
                    )
                )
                print(f"Collection '{self.collection}' created with default parameters after optimization attempt failed.")
        else:
            print(f"Collection '{self.collection}' already exists.")

    def _validate_hnsw_config(self, config: dict) -> dict:
        """
        Validates and normalizes HNSW configuration parameters.

        Args:
            config: Dictionary containing HNSW parameters

        Returns:
            Validated and normalized configuration dictionary
        """
        validated = config.copy()

        # Validate m (number of edges per node)
        if "m" in validated:
            validated["m"] = max(8, min(validated["m"], 100))

        # Validate ef_construct (size of dynamic candidate list)
        if "ef_construct" in validated:
            validated["ef_construct"] = max(8, min(validated["ef_construct"], 200))

        # Validate full_scan_threshold
        if "full_scan_threshold" in validated:
            validated["full_scan_threshold"] = max(1000, validated["full_scan_threshold"])

        # Validate max_indexing_threads
        if "max_indexing_threads" in validated:
            validated["max_indexing_threads"] = max(1, min(validated["max_indexing_threads"], 8))

        return validated

    def _validate_optimizer_config(self, config: dict) -> dict:
        """
        Validates and normalizes optimizer configuration parameters.

        Args:
            config: Dictionary containing optimizer parameters

        Returns:
            Validated and normalized configuration dictionary
        """
        validated = config.copy()

        # Validate indexing_threshold
        if "indexing_threshold" in validated:
            validated["indexing_threshold"] = max(10000, validated["indexing_threshold"])

        # Validate memmap_threshold
        if "memmap_threshold" in validated:
            validated["memmap_threshold"] = max(1000, validated["memmap_threshold"])

        # Validate vacuum_min_vector_number
        if "vacuum_min_vector_number" in validated:
            validated["vacuum_min_vector_number"] = max(100, validated["vacuum_min_vector_number"])

        # Validate default_segment_number
        if "default_segment_number" in validated:
            validated["default_segment_number"] = max(1, min(validated["default_segment_number"], 8))

        # Validate flush_interval_sec
        if "flush_interval_sec" in validated:
            validated["flush_interval_sec"] = max(5, validated["flush_interval_sec"])

        return validated

    def _validate_payload_schema(self, schema: dict) -> dict:
        """
        Validates and normalizes payload schema configuration.

        Args:
            schema: Dictionary containing payload schema

        Returns:
            Validated and normalized schema dictionary
        """
        validated = schema.copy()
        valid_types = {"keyword", "integer", "float", "geo", "text", "object"}

        for field, config in validated.items():
            if isinstance(config, dict):
                # Validate field type
                if "type" in config and config["type"] not in valid_types:
                    config["type"] = "keyword"  # Default to keyword for invalid types

                # Ensure index property exists
                if "index" not in config:
                    config["index"] = True  # Default to indexed
            else:
                # Convert simple type strings to full config
                validated[field] = {
                    "type": "keyword" if config not in valid_types else config,
                    "index": True
                }

        return validated

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
            from application.services.embedding_service import EmbeddingService
            from application.services.embedding_service import FastEmbedProvider
            provider_instance = FastEmbedProvider(self.embedding_config.get("fastembed_model_name"))
            return EmbeddingService(provider_instance)
        elif provider == "ollama":
            from application.services.embedding_service import EmbeddingService
            from application.services.embedding_service import OllamaProvider
            provider_instance = OllamaProvider(
                model_name=self.embedding_config.get("ollama_model_name"),
                base_url=self.embedding_config.get("ollama_base_url", "http://localhost:11434")
            )
            return EmbeddingService(provider_instance)
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

    def _validate_batch_size(self, batch_size: int, total_points: int) -> int:
        """
        Validates and optimizes batch size based on total points and system constraints.

        Args:
            batch_size: Requested batch size
            total_points: Total number of points to process

        Returns:
            Optimized batch size
        """
        # Minimum batch size for efficiency
        MIN_BATCH = 10
        # Maximum batch size to prevent memory issues
        MAX_BATCH = 10000

        # Adjust batch size based on total points
        if total_points < batch_size:
            return max(MIN_BATCH, total_points)

        # Keep batch size within reasonable limits
        return max(MIN_BATCH, min(batch_size, MAX_BATCH))

    async def _process_batch_with_retry(
        self,
        batch: List[PointStruct],
        attempt: int = 1,
        max_retries: int = 3,
        wait: bool = False
    ) -> Dict[str, Any]:
        """
        Process a batch of points with retry logic and exponential backoff.

        Args:
            batch: List of points to upsert
            attempt: Current attempt number
            max_retries: Maximum number of retry attempts
            wait: Whether to wait for indexing

        Returns:
            Dict containing operation status and details
        """
        try:
            result = await self.async_client.upsert(
                collection_name=self.collection,
                points=batch,
                wait=wait
            )
            return {
                "status": "success",
                "operation_id": result.operation_id,
                "points": len(batch)
            }
        except Exception as e:
            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = (2 ** attempt) + random.uniform(0, 1)
                print(f"Batch processing failed, retrying in {delay:.2f}s... (Attempt {attempt}/{max_retries})")
                await asyncio.sleep(delay)
                return await self._process_batch_with_retry(
                    batch, attempt + 1, max_retries, wait
                )
            return {
                "status": "failed",
                "error": str(e),
                "points": len(batch)
            }

    async def upsert_points(
        self,
        points: List[PointStruct],
        batch_size: int = 100,
        max_retries: int = 3,
        wait: bool = False,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Upserts points in optimized batches with parallel processing and retry logic.

        Features:
        - Automatic batch size optimization
        - Parallel processing with asyncio.gather
        - Retry logic with exponential backoff
        - Progress tracking
        - Detailed error reporting

        Args:
            points: List of points to upsert
            batch_size: Target batch size (will be optimized)
            max_retries: Maximum number of retry attempts per batch
            wait: Whether to wait for indexing to complete
            show_progress: Whether to show progress information

        Returns:
            Dict containing operation summary
        """
        total_points = len(points)
        if total_points == 0:
            return {"status": "success", "points_processed": 0, "failed_batches": 0}

        # Validate and optimize batch size
        optimized_batch_size = self._validate_batch_size(batch_size, total_points)
        if optimized_batch_size != batch_size:
            print(f"Optimized batch size from {batch_size} to {optimized_batch_size}")

        # Split points into batches
        batches = [
            points[i:i + optimized_batch_size]
            for i in range(0, total_points, optimized_batch_size)
        ]

        # Process batches in parallel with progress tracking
        if show_progress:
            print(f"\nProcessing {total_points} points in {len(batches)} batches...")
            start_time = time.time()

        # Process all batches
        batch_results = await asyncio.gather(
            *[self._process_batch_with_retry(
                batch,
                max_retries=max_retries,
                wait=wait
            ) for batch in batches]
        )

        # Analyze results
        successful_points = 0
        failed_batches = []
        operation_ids = set()

        for i, result in enumerate(batch_results):
            if result["status"] == "success":
                successful_points += result["points"]
                operation_ids.add(result["operation_id"])
            else:
                failed_batches.append({
                    "batch_index": i,
                    "error": result["error"],
                    "points": result["points"],
                    "start_index": i * optimized_batch_size
                })

        # Show progress summary
        if show_progress:
            elapsed = time.time() - start_time
            points_per_second = total_points / elapsed if elapsed > 0 else 0
            print(f"\nProcessing completed in {elapsed:.2f}s")
            print(f"Points/second: {points_per_second:.2f}")
            print(f"Successful points: {successful_points}/{total_points}")
            if failed_batches:
                print(f"Failed batches: {len(failed_batches)}")

        return {
            "status": "completed",
            "total_points": total_points,
            "successful_points": successful_points,
            "points_per_second": points_per_second if show_progress else None,
            "operation_ids": list(operation_ids),
            "failed_batches": len(failed_batches),
            "failed_batch_details": failed_batches if failed_batches else None,
            "batch_size_used": optimized_batch_size
        }

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

    async def update_collection_params(
        self,
        optimize_for: str = "insert",  # or "search"
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Updates collection parameters to optimize for either insert or search performance.
        
        Args:
            optimize_for: Either "insert" or "search"
            custom_params: Optional custom parameters to override defaults
            
        Returns:
            Dict containing operation status and applied parameters
        """
        # Define optimization presets
        optimization_presets = {
            "insert": {
                "hnsw_config": {
                    "m": 16,  # Lower for faster inserts
                    "ef_construct": 32,  # Lower for faster inserts
                    "full_scan_threshold": 10000,
                    "max_indexing_threads": 4
                },
                "optimizer_config": {
                    "indexing_threshold": 50000,  # Higher for batch indexing
                    "memmap_threshold": 10000,
                    "vacuum_min_vector_number": 1000,
                    "default_segment_number": 2,
                    "flush_interval_sec": 30
                }
            },
            "search": {
                "hnsw_config": {
                    "m": 32,  # Higher for better recall
                    "ef_construct": 100,  # Higher for better recall
                    "full_scan_threshold": 1000,
                    "max_indexing_threads": 4
                },
                "optimizer_config": {
                    "indexing_threshold": 20000,  # Lower for more frequent indexing
                    "memmap_threshold": 5000,
                    "vacuum_min_vector_number": 500,
                    "default_segment_number": 4,
                    "flush_interval_sec": 5
                }
            }
        }

        try:
            # Get base configuration
            if optimize_for not in optimization_presets:
                raise ValueError(f"Invalid optimization mode: {optimize_for}. Must be 'insert' or 'search'")
                
            config = optimization_presets[optimize_for].copy()
            
            # Override with custom params if provided
            if custom_params:
                for category in ["hnsw_config", "optimizer_config"]:
                    if category in custom_params:
                        config[category].update(custom_params[category])

            # Validate configurations
            config["hnsw_config"] = self._validate_hnsw_config(config["hnsw_config"])
            config["optimizer_config"] = self._validate_optimizer_config(config["optimizer_config"])

            # Update collection parameters
            success = await self.async_client.update_collection(
                collection_name=self.collection,
                optimizers_config=config["optimizer_config"],
                hnsw_config=config["hnsw_config"]
            )

            return {
                "status": "success" if success else "failed",
                "optimization_mode": optimize_for,
                "applied_config": config
            }

        except Exception as e:
            logging.error(f"Failed to update collection parameters: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update collection parameters: {str(e)}"
            )

    async def create_payload_index(
        self,
        field_name: str,
        field_schema: Dict[str, Any],
        wait: bool = True
    ) -> Dict[str, Any]:
        """
        Creates an index on a payload field for faster filtering.
        
        Args:
            field_name: Name of the field to index
            field_schema: Schema definition for the field
            wait: Whether to wait for the indexing operation to complete
            
        Returns:
            Dict containing operation status
        """
        try:
            # Validate field schema
            valid_schema = self._validate_payload_schema({field_name: field_schema})
            field_config = valid_schema[field_name]

            # Create the index
            result = await self.async_client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema=field_config["type"],
                wait=wait
            )
            
            return {
                "status": "success",
                "operation_id": result.operation_id,
                "field_name": field_name,
                "field_schema": field_config
            }

        except Exception as e:
            logging.error(f"Failed to create payload index: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create payload index: {str(e)}"
            )

    async def list_payload_indexes(self) -> List[Dict[str, Any]]:
        """
        Lists all payload indexes in the collection.
        
        Returns:
            List of payload index configurations
        """
        try:
            collection_info = await self.async_client.get_collection(
                collection_name=self.collection
            )
            
            if hasattr(collection_info, "payload_schema"):
                return [
                    {
                        "field_name": field_name,
                        "schema": schema
                    }
                    for field_name, schema in collection_info.payload_schema.items()
                    if isinstance(schema, dict) and schema.get("index", False)
                ]
            return []

        except Exception as e:
            logging.error(f"Failed to list payload indexes: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to list payload indexes: {str(e)}"
            )
