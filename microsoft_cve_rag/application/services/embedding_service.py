# Purpose: Generate embeddings for text using various models
# Inputs: Text strings
# Outputs: Embeddings (numpy arrays)
# Dependencies: None (external: Ollama, Hugging Face, or OpenAI)

"""
Purpose: Generate embeddings for text using various models
Inputs: Text strings
Outputs: Embeddings (numpy arrays)
Dependencies: None (external: Ollama, Hugging Face, or OpenAI)
"""

from typing import List, Union
from abc import ABC, abstractmethod
import httpx
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.qdrant_fastembed import TextEmbedding
from application.app_utils import get_app_config

settings = get_app_config()
embedding_config = settings["EMBEDDING_CONFIG"]


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        pass

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    async def embed_async(self, texts: List[str]) -> List[List[float]]:
        pass


class QdrantDefaultProvider(EmbeddingProvider):
    """
    Embedding provider using Qdrant's default embedding service.
    """

    def __init__(self, sync_client: QdrantClient, async_client: AsyncQdrantClient):
        """
        Initialize the QdrantDefaultProvider.

        Args:
            sync_client (QdrantClient): Synchronous Qdrant client.
            async_client (AsyncQdrantClient): Asynchronous Qdrant client.
        """
        self.sync_client = sync_client
        self.async_client = async_client
        self._embedding_length = embedding_config.get("vector_db_embedding_length")
        self._model_name = embedding_config.get("vector_db_embedding_model_name")
        print(f"Qdrant loaded with: {self._model_name} ({self._embedding_length})")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_length(self) -> int:
        return self._embedding_length

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings synchronously.

        Args:
            texts (List[str]): List of text strings to embed.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        return self.sync_client.encode(texts)

    async def embed_async(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings asynchronously.

        Args:
            texts (List[str]): List of text strings to embed.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        return await self.async_client.encode(texts)


class FastEmbedProvider(EmbeddingProvider):
    """
    Embedding provider using FastEmbed.
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the FastEmbedProvider.

        Args:
            model_name (str, optional): Name of the FastEmbed model. Defaults to None.
        """
        self._model_name = model_name or embedding_config.get("fastembed_model_name")
        self.model = TextEmbedding(self._model_name)
        self._embedding_length = embedding_config.get("fastembed_embedding_length")
        print(f"fastembed loaded with: {self._model_name} ({self._embedding_length})")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_length(self) -> int:
        return self._embedding_length

    def _convert_to_list(self, embeddings):
        """
        Convert NumPy arrays to nested Python lists.

        Args:
            embeddings (List[np.ndarray]): List of NumPy arrays.

        Returns:
            List[List[float]]: Nested list of floats.
        """
        return [embedding.tolist() for embedding in embeddings]

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings synchronously.

        Args:
            texts (List[str]): List of text strings to embed.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        # print("FastEmbed(sync) embedding...")
        embeddings = list(self.model.embed(texts))
        converted_embeddings = self._convert_to_list(embeddings)

        return converted_embeddings

    async def embed_async(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings asynchronously.

        Args:
            texts (List[str]): List of text strings to embed.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        # print("FastEmbed(async) embedding...")
        # FastEmbed doesn't have a native async API, so we'll use the sync version.
        return await asyncio.to_thread(self.embed, texts)


class OllamaProvider(EmbeddingProvider):
    """
    Embedding provider using Ollama.
    """

    def __init__(
        self, model_name: str = None, base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the OllamaProvider.

        Args:
            model_name (str, optional): Name of the Ollama model. Defaults to None.
            base_url (str, optional): Base URL for the Ollama API. Defaults to "http://localhost:11434".
        """
        self._base_url = base_url
        self._embedding_length = embedding_config.get("ollama_embedding_length")
        self._model_name = model_name or embedding_config.get(
            "ollama_embedding_model_name"
        )
        print(f"ollama loaded with: {self._model_name} ({self._embedding_length})")

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_length(self) -> int:
        return self._embedding_length

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings synchronously.

        Args:
            texts (List[str]): List of text strings to embed.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        embeddings = []
        with httpx.Client() as client:
            for text in texts:
                response = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": text},
                )
                embeddings.append(response.json()["embedding"])
        return embeddings

    async def embed_async(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings asynchronously.

        Args:
            texts (List[str]): List of text strings to embed.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        embeddings = []
        async with httpx.AsyncClient() as client:
            for text in texts:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": text},
                )
                embeddings.append(response.json()["embedding"])
        return embeddings


class EmbeddingService:
    """
    Service for generating embeddings using a specified provider.
    """

    def __init__(self, provider: EmbeddingProvider):
        """
        Initialize the EmbeddingService.

        Args:
            provider (EmbeddingProvider): The embedding provider to use.
        """
        self.provider = provider
        print(f"EmbeddingService received provider: {self.provider}")

    @property
    def model_name(self) -> str:
        return self.provider.model_name

    @property
    def embedding_length(self) -> int:
        return self.provider.embedding_length

    def generate_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings synchronously.

        Args:
            texts (Union[str, List[str]]): Text or list of texts to embed.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        if isinstance(texts, str):
            texts = [texts]

        return self.provider.embed(texts)

    async def generate_embeddings_async(
        self, texts: Union[str, List[str]]
    ) -> List[List[float]]:
        """
        Generate embeddings asynchronously.

        Args:
            texts (Union[str, List[str]]): Text or list of texts to embed.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        if isinstance(texts, str):
            texts = [texts]

        return await self.provider.embed_async(texts)
