# Purpose: Generate embeddings for text using various models
# Inputs: Text strings
# Outputs: Embeddings (numpy arrays)
# Dependencies: None (external: Ollama, Hugging Face, or OpenAI)

from typing import List, Union
from abc import ABC, abstractmethod
import httpx
from qdrant_client import QdrantClient
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from fastembed import TextEmbedding
from application.app_utils import get_app_config

settings = get_app_config()
embedding_config = settings["EMBEDDING_CONFIG"]


class EmbeddingProvider(ABC):
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
    def __init__(self, sync_client: QdrantClient, async_client: AsyncQdrantClient):
        self.sync_client = sync_client
        self.async_client = async_client
        self._embedding_length = embedding_config.get("vector_db_embedding_length")
        self._model_name = embedding_config.get("vector_db_embedding_model_name")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_length(self) -> int:
        return self._embedding_length

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.sync_client.encode(texts)

    async def embed_async(self, texts: List[str]) -> List[List[float]]:
        return await self.async_client.encode(texts)


class FastEmbedProvider(EmbeddingProvider):
    def __init__(self, model_name: str = None):
        self._model_name = model_name or embedding_config.get("fastembed_model_name")
        print(f"{self._model_name}")
        self.model = TextEmbedding(self._model_name)
        self._embedding_length = embedding_config.get("fastembed_embedding_length")
        self._model_name = model_name or embedding_config.get("fastembed_model_name")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_length(self) -> int:
        return self._embedding_length

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed(texts)

    async def embed_async(self, texts: List[str]) -> List[List[float]]:
        # FastEmbed doesn't have a native async API, so we'll use the sync version
        return self.embed(texts)


class OllamaProvider(EmbeddingProvider):
    def __init__(
        self, model_name: str = None, base_url: str = "http://localhost:11434"
    ):
        self._base_url = base_url
        self._embedding_length = embedding_config.get("ollama_embedding_length")
        self._model_name = model_name or embedding_config.get(
            "ollama_embedding_model_name"
        )

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
    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider

    @property
    def model_name(self) -> str:
        return self.provider.model_name

    @property
    def embedding_length(self) -> int:
        return self.provider.embedding_length

    def generate_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        return self.provider.embed(texts)

    async def generate_embeddings_async(
        self, texts: Union[str, List[str]]
    ) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        return await self.provider.embed_async(texts)
