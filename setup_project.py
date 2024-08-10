structure = {
    "microsoft_cve_rag": {
        "domain": {
            "entities": {
                "document.py": """# document.py
# Purpose: This script defines what a Document is in the application.
# Explanation: A Document is an entity, which means it represents a real-world object with certain properties and behaviors.
# In this case, a Document can have a title, content, and a creation date.
# Regular Class vs. Pydantic: We use a regular class here because Document has behaviors (methods like summary) beyond just holding data.
# Relationships: This script may be imported by repositories and services that handle document data.
# Example Usage:
# from microsoft_cve_rag.domain.entities.document import Document

from datetime import datetime

class Document:
    def __init__(self, title: str, content: str, created_at: datetime = None):
        self.title = title
        self.content = content
        self.created_at = created_at or datetime.now()

    def summary(self):
        return self.content[:100]
""",
                "vector.py": """# vector.py
# Purpose: This script defines what a Vector is in the application.
# Explanation: A Vector is an entity that represents numerical data used in machine learning models.
# In this case, a Vector can have a list of numbers and a creation date.
# Regular Class vs. Pydantic: We use a regular class here because Vector may need additional methods for operations.
# Relationships: This script may be imported by repositories and services that handle vector data.
# Example Usage:
# from microsoft_cve_rag.domain.entities.vector import Vector

from datetime import datetime
from typing import List

class Vector:
    def __init__(self, values: List[float], created_at: datetime = None):
        self.values = values
        self.created_at = created_at or datetime.now()

    def dimension(self):
        return len(self.values)
""",
                "graph_node.py": """# graph_node.py
# Purpose: This script defines what a GraphNode is in the application.
# Explanation: A GraphNode is an entity that represents nodes in a graph structure, which can be used in graph databases.
# In this case, a GraphNode can have an ID, a label, and properties.
# Regular Class vs. Pydantic: We use a regular class here because GraphNode may have methods to manipulate its properties.
# Relationships: This script may be imported by repositories and services that handle graph node data.
# Example Usage:
# from microsoft_cve_rag.domain.entities.graph_node import GraphNode

class GraphNode:
    def __init__(self, node_id: str, label: str, properties: dict):
        self.node_id = node_id
        self.label = label
        self.properties = properties

    def add_property(self, key: str, value):
        self.properties[key] = value
""",
                "__init__.py": "# Make the entities package importable\n",
            },
            "services": {
                "embedding_service.py": """# embedding_service.py
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
""",
                "chat_service.py": """# chat_service.py
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
""",
                "__init__.py": "# Make the services package importable\n",
            },
            "value_objects": {
                "embedding.py": """# embedding.py
# Purpose: This script defines the Embedding value object.
# Explanation: A value object is an object that contains certain values and is used to pass data around.
# In this case, an Embedding has a list of numbers representing text in a machine-readable format.
# Regular Class vs. Pydantic: We use a Pydantic model here because it provides strong data validation and serialization capabilities.
# Relationships: This script may be used by services and repositories that handle embedding data.
# Example Usage:
# from microsoft_cve_rag.domain.value_objects.embedding import Embedding

from pydantic import BaseModel
from typing import List

class Embedding(BaseModel):
    values: List[float]
""",
                "chat_message.py": """# chat_message.py
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
""",
                "__init__.py": "# Make the value_objects package importable\n",
            },
            "__init__.py": "# Make the domain package importable\n",
        },
        "application": {
            "use_cases": {
                "document_management": {
                    "create_document.py": """# create_document.py
# Purpose: This script implements the use case for creating a document.
# Explanation: A use case represents a specific action that the application can perform.
# In this case, creating a document involves interacting with the document repository and the Document entity.
# Relationships: This script interacts with the document repository and document entity.
# Example Usage:
# from microsoft_cve_rag.application.use_cases.document_management.create_document import CreateDocument
# from microsoft_cve_rag.domain.entities.document import Document
# from microsoft_cve_rag.application.interfaces.repositories.document_repository import DocumentRepository

class CreateDocument:
    def __init__(self, repository: DocumentRepository):
        self.repository = repository

    def execute(self, title: str, content: str):
        document = Document(title=title, content=content)
        self.repository.save(document)
""",
                    "link_document_to_graph.py": """# link_document_to_graph.py
# Purpose: This script implements the use case for linking a document to a graph.
# Explanation: A use case represents a specific action that the application can perform.
# In this case, linking a document to a graph involves interacting with the document repository, graph repository, and Document entity.
# Relationships: This script interacts with the document repository, graph repository, and document entity.
# Example Usage:
# from microsoft_cve_rag.application.use_cases.document_management.link_document_to_graph import LinkDocumentToGraph
# from microsoft_cve_rag.domain.entities.document import Document
# from microsoft_cve_rag.application.interfaces.repositories.document_repository import DocumentRepository
# from microsoft_cve_rag.application.interfaces.repositories.graph_repository import GraphRepository

class LinkDocumentToGraph:
    def __init__(self, document_repository: DocumentRepository, graph_repository: GraphRepository):
        self.document_repository = document_repository
        self.graph_repository = graph_repository

    def execute(self, document_id: str, graph_node_id: str):
        document = self.document_repository.find_by_id(document_id)
        graph_node = self.graph_repository.find_by_id(graph_node_id)
        self.graph_repository.link(document, graph_node)
""",
                    "__init__.py": "# Make the document_management package importable\n",
                },
                "embedding_generation": {
                    "generate_embedding.py": """# generate_embedding.py
# Purpose: This script implements the use case for generating embeddings.
# Explanation: A use case represents a specific action that the application can perform.
# In this case, generating embeddings involves interacting with the embedding service and the Embedding value object.
# Relationships: This script interacts with the embedding service and embedding value object.
# Example Usage:
# from microsoft_cve_rag.application.use_cases.embedding_generation.generate_embedding import GenerateEmbedding
# from microsoft_cve_rag.domain.services.embedding_service import EmbeddingService
# from microsoft_cve_rag.domain.value_objects.embedding import Embedding

class GenerateEmbedding:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    def execute(self, text: str) -> Embedding:
        values = self.embedding_service.generate(text)
        return Embedding(values=values)
""",
                    "__init__.py": "# Make the embedding_generation package importable\n",
                },
                "chat_management": {
                    "chat_completion.py": """# chat_completion.py
# Purpose: This script implements the use case for completing chat messages.
# Explanation: A use case represents a specific action that the application can perform.
# In this case, completing chat messages involves interacting with the chat service and the ChatMessage value object.
# Relationships: This script interacts with the chat service and chat message value object.
# Example Usage:
# from microsoft_cve_rag.application.use_cases.chat_management.chat_completion import ChatCompletion
# from microsoft_cve_rag.domain.services.chat_service import ChatService
# from microsoft_cve_rag.domain.value_objects.chat_message import ChatMessage

class ChatCompletion:
    def __init__(self, chat_service: ChatService):
        self.chat_service = chat_service

    def execute(self, message: str) -> ChatMessage:
        response = self.chat_service.send_message(message)
        return ChatMessage(content=response)
""",
                    "text_completion.py": """# text_completion.py
# Purpose: This script implements the use case for completing text messages.
# Explanation: A use case represents a specific action that the application can perform.
# In this case, completing text messages involves interacting with the chat service and the ChatMessage value object.
# Relationships: This script interacts with the chat service and chat message value object.
# Example Usage:
# from microsoft_cve_rag.application.use_cases.chat_management.text_completion import TextCompletion
# from microsoft_cve_rag.domain.services.chat_service import ChatService
# from microsoft_cve_rag.domain.value_objects.chat_message import ChatMessage

class TextCompletion:
    def __init__(self, chat_service: ChatService):
        self.chat_service = chat_service

    def execute(self, message: str) -> ChatMessage:
        response = self.chat_service.send_message(message)
        return ChatMessage(content=response)
""",
                    "__init__.py": "# Make the chat_management package importable\n",
                },
                "__init__.py": "# Make the use_cases package importable\n",
            },
            "interfaces": {
                "repositories": {
                    "document_repository.py": """# document_repository.py
# Purpose: This script defines the interface for document repositories.
# Explanation: An interface in this context defines methods that must be implemented by any class that handles document data storage and retrieval.
# Relationships: This script may be implemented by infrastructure repositories like document_repository_impl.py.
# Example Usage:
# from microsoft_cve_rag.application.interfaces.repositories.document_repository import DocumentRepository

from abc import ABC, abstractmethod

class DocumentRepository(ABC):
    @abstractmethod
    def save(self, document):
        pass

    @abstractmethod
    def find_by_id(self, document_id: str):
        pass
""",
                    "vector_repository.py": """# vector_repository.py
# Purpose: This script defines the interface for vector repositories.
# Explanation: An interface in this context defines methods that must be implemented by any class that handles vector data storage and retrieval.
# Relationships: This script may be implemented by infrastructure repositories like vector_repository_impl.py.
# Example Usage:
# from microsoft_cve_rag.application.interfaces.repositories.vector_repository import VectorRepository

from abc import ABC, abstractmethod

class VectorRepository(ABC):
    @abstractmethod
    def save(self, vector):
        pass

    @abstractmethod
    def find_by_id(self, vector_id: str):
        pass
""",
                    "graph_repository.py": """# graph_repository.py
# Purpose: This script defines the interface for graph repositories.
# Explanation: An interface in this context defines methods that must be implemented by any class that handles graph data storage and retrieval.
# Relationships: This script may be implemented by infrastructure repositories like graph_repository_impl.py.
# Example Usage:
# from microsoft_cve_rag.application.interfaces.repositories.graph_repository import GraphRepository

from abc import ABC, abstractmethod

class GraphRepository(ABC):
    @abstractmethod
    def save(self, graph_node):
        pass

    @abstractmethod
    def find_by_id(self, graph_node_id: str):
        pass

    @abstractmethod
    def link(self, document, graph_node):
        pass
""",
                    "__init__.py": "# Make the repositories package importable\n",
                },
                "services": {
                    "embedding_service.py": """# embedding_service.py
# Purpose: This script defines the interface for embedding services.
# Explanation: An interface in this context defines methods that must be implemented by any class that generates embeddings.
# Relationships: This script may be implemented by infrastructure services like openai_service.py, groq_service.py, and ollama_service.py.
# Example Usage:
# from microsoft_cve_rag.application.interfaces.services.embedding_service import EmbeddingService

from abc import ABC, abstractmethod
from typing import List

class EmbeddingService(ABC):
    @abstractmethod
    def generate(self, text: str) -> List[float]:
        pass
""",
                    "chat_service.py": """# chat_service.py
# Purpose: This script defines the interface for chat services.
# Explanation: An interface in this context defines methods that must be implemented by any class that handles chat operations.
# Relationships: This script may be implemented by infrastructure services like openai_service.py, groq_service.py, and ollama_service.py.
# Example Usage:
# from microsoft_cve_rag.application.interfaces.services.chat_service import ChatService

from abc import ABC, abstractmethod

class ChatService(ABC):
    @abstractmethod
    def send_message(self, message: str) -> str:
        pass
""",
                    "__init__.py": "# Make the services package importable\n",
                },
                "__init__.py": "# Make the interfaces package importable\n",
            },
            "__init__.py": "# Make the application package importable\n",
        },
        "infrastructure": {
            "repositories": {
                "document_repository_impl.py": """# document_repository_impl.py
# Purpose: This script implements the DocumentRepository interface to handle document data operations.
# Explanation: This script provides the actual implementation of how documents are stored and retrieved.
# Relationships: This script interacts with the Document entity and the database models.
# Example Usage:
# from microsoft_cve_rag.infrastructure.repositories.document_repository_impl import DocumentRepositoryImpl
# from microsoft_cve_rag.domain.entities.document import Document

class DocumentRepositoryImpl(DocumentRepository):
    def save(self, document: Document):
        # Implementation to save document
        pass

    def find_by_id(self, document_id: str) -> Document:
        # Implementation to find document by ID
        pass
""",
                "vector_repository_impl.py": """# vector_repository_impl.py
# Purpose: This script implements the VectorRepository interface to handle vector data operations.
# Explanation: This script provides the actual implementation of how vectors are stored and retrieved.
# Relationships: This script interacts with the Vector entity and the database models.
# Example Usage:
# from microsoft_cve_rag.infrastructure.repositories.vector_repository_impl import VectorRepositoryImpl
# from microsoft_cve_rag.domain.entities.vector import Vector

class VectorRepositoryImpl(VectorRepository):
    def save(self, vector: Vector):
        # Implementation to save vector
        pass

    def find_by_id(self, vector_id: str) -> Vector:
        # Implementation to find vector by ID
        pass
""",
                "graph_repository_impl.py": """# graph_repository_impl.py
# Purpose: This script implements the GraphRepository interface to handle graph data operations.
# Explanation: This script provides the actual implementation of how graph nodes are stored and retrieved.
# Relationships: This script interacts with the GraphNode entity and the database models.
# Example Usage:
# from microsoft_cve_rag.infrastructure.repositories.graph_repository_impl import GraphRepositoryImpl
# from microsoft_cve_rag.domain.entities.graph_node import GraphNode

class GraphRepositoryImpl(GraphRepository):
    def save(self, graph_node: GraphNode):
        # Implementation to save graph node
        pass

    def find_by_id(self, graph_node_id: str) -> GraphNode:
        # Implementation to find graph node by ID
        pass

    def link(self, document: Document, graph_node: GraphNode):
        # Implementation to link document to graph node
        pass
""",
                "__init__.py": "# Make the repositories package importable\n",
            },
            "services": {
                "openai_service.py": """# openai_service.py
# Purpose: This script implements the service interface for OpenAI.
# Explanation: This script provides the actual implementation of how to interact with the OpenAI API for chat and embedding services.
# Relationships: This script implements the EmbeddingService and ChatService interfaces.
# Example Usage:
# from microsoft_cve_rag.infrastructure.services.openai_service import OpenAIService
# from microsoft_cve_rag.domain.services.embedding_service import EmbeddingService
# from microsoft_cve_rag.domain.services.chat_service import ChatService

class OpenAIService(EmbeddingService, ChatService):
    def generate(self, text: str) -> List[float]:
        # Implementation to generate embeddings using OpenAI
        pass

    def send_message(self, message: str) -> str:
        # Implementation to send chat message using OpenAI
        pass
""",
                "groq_service.py": """# groq_service.py
# Purpose: This script implements the service interface for Groq.
# Explanation: This script provides the actual implementation of how to interact with the Groq API for chat and embedding services.
# Relationships: This script implements the EmbeddingService and ChatService interfaces.
# Example Usage:
# from microsoft_cve_rag.infrastructure.services.groq_service import GroqService
# from microsoft_cve_rag.domain.services.embedding_service import EmbeddingService
# from microsoft_cve_rag.domain.services.chat_service import ChatService

class GroqService(EmbeddingService, ChatService):
    def generate(self, text: str) -> List[float]:
        # Implementation to generate embeddings using Groq
        pass

    def send_message(self, message: str) -> str:
        # Implementation to send chat message using Groq
        pass
""",
                "ollama_service.py": """# ollama_service.py
# Purpose: This script implements the service interface for Ollama.
# Explanation: This script provides the actual implementation of how to interact with the Ollama API for chat and embedding services.
# Relationships: This script implements the EmbeddingService and ChatService interfaces.
# Example Usage:
# from microsoft_cve_rag.infrastructure.services.ollama_service import OllamaService
# from microsoft_cve_rag.domain.services.embedding_service import EmbeddingService
# from microsoft_cve_rag.domain.services.chat_service import ChatService

class OllamaService(EmbeddingService, ChatService):
    def generate(self, text: str) -> List[float]:
        # Implementation to generate embeddings using Ollama
        pass

    def send_message(self, message: str) -> str:
        # Implementation to send chat message using Ollama
        pass
""",
                "__init__.py": "# Make the services package importable\n",
            },
            "database": {
                "models": {
                    "document.py": """# document.py
# Purpose: This script defines the database model for documents.
# Explanation: A database model represents the structure of the data as it is stored in the database.
# Relationships: This script may be used by document_repository_impl.py for database operations.
# Example Usage:
# from microsoft_cve_rag.infrastructure.database.models.document import DocumentModel

from sqlalchemy import Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DocumentModel(Base):
    __tablename__ = 'documents'
    id = Column(String, primary_key=True)
    title = Column(String)
    content = Column(String)
    created_at = Column(DateTime)
""",
                    "vector.py": """# vector.py
# Purpose: This script defines the database model for vectors.
# Explanation: A database model represents the structure of the data as it is stored in the database.
# Relationships: This script may be used by vector_repository_impl.py for database operations.
# Example Usage:
# from microsoft_cve_rag.infrastructure.database.models.vector import VectorModel

from sqlalchemy import Column, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class VectorModel(Base):
    __tablename__ = 'vectors'
    id = Column(String, primary_key=True)
    values = Column(Float)
    created_at = Column(DateTime)
""",
                    "__init__.py": "# Make the models package importable\n",
                },
                "config.py": """# config.py
# Purpose: This script sets up the configuration for the database connection.
# Explanation: Configuration includes the details needed to connect to the database, such as the connection string.
# Relationships: This script may be used by session.py to establish database connections.
# Example Usage:
# from microsoft_cve_rag.infrastructure.database.config import DATABASE_URL

DATABASE_URL = "sqlite:///./test.db"
""",
                "session.py": """# session.py
# Purpose: This script manages database sessions for the application.
# Explanation: A session is used to interact with the database, allowing for operations like querying and committing data.
# Relationships: This script interacts with config.py to establish and manage database connections.
# Example Usage:
# from microsoft_cve_rag.infrastructure.database.session import get_session

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from microsoft_cve_rag.infrastructure.database.config import DATABASE_URL

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
""",
                "__init__.py": "# Make the database package importable\n",
            },
            "neo4j": {
                "config.py": """# config.py
# Purpose: This script sets up the configuration for the Neo4j connection.
# Explanation: Configuration includes the details needed to connect to the Neo4j database, such as the connection string.
# Relationships: This script may be used by session.py to establish Neo4j connections.
# Example Usage:
# from microsoft_cve_rag.infrastructure.neo4j.config import NEO4J_URL

NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
""",
                "session.py": """# session.py
# Purpose: This script manages Neo4j sessions for the application.
# Explanation: A session is used to interact with the Neo4j database, allowing for operations like querying and committing data.
# Relationships: This script interacts with config.py to establish and manage Neo4j connections.
# Example Usage:
# from microsoft_cve_rag.infrastructure.neo4j.session import get_neo4j_session

from neo4j import GraphDatabase
from microsoft_cve_rag.infrastructure.neo4j.config import NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD

driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

def get_neo4j_session():
    with driver.session() as session:
        yield session
""",
                "__init__.py": "# Make the neo4j package importable\n",
            },
            "vector_db": {
                "config.py": """# config.py
# Purpose: This script sets up the configuration for the vector database connection.
# Explanation: Configuration includes the details needed to connect to the vector database, such as the connection string.
# Relationships: This script may be used by session.py to establish vector database connections.
# Example Usage:
# from microsoft_cve_rag.infrastructure.vector_db.config import VECTOR_DB_URL

VECTOR_DB_URL = "http://localhost:8000"
""",
                "session.py": """# session.py
# Purpose: This script manages vector database sessions for the application.
# Explanation: A session is used to interact with the vector database, allowing for operations like querying and committing data.
# Relationships: This script interacts with config.py to establish and manage vector database connections.
# Example Usage:
# from microsoft_cve_rag.infrastructure.vector_db.session import get_vector_db_session

import requests
from microsoft_cve_rag.infrastructure.vector_db.config import VECTOR_DB_URL

def get_vector_db_session():
    session = requests.Session()
    session.base_url = VECTOR_DB_URL
    return session
""",
                "__init__.py": "# Make the vector_db package importable\n",
            },
            "cache": {
                "redis_client.py": """# redis_client.py
# Purpose: This script implements the Redis client for caching operations.
# Explanation: Caching is used to temporarily store frequently accessed data for faster retrieval.
# Relationships: This script may be used by various services and repositories for caching purposes.
# Example Usage:
# from microsoft_cve_rag.infrastructure.cache.redis_client import redis_client

import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)
""",
                "__init__.py": "# Make the cache package importable\n",
            },
            "__init__.py": "# Make the infrastructure package importable\n",
        },
        "presentation": {
            "api": {
                "v1": {
                    "routes": {
                        "document_routes.py": """# document_routes.py
# Purpose: This script defines the API routes for handling document operations.
# Explanation: API routes specify the endpoints that can be accessed to perform operations on documents.
# Relationships: This script interacts with the document use cases and schemas.
# Example Usage:
# from microsoft_cve_rag.presentation.api.v1.routes.document_routes import document_router

from fastapi import APIRouter
from microsoft_cve_rag.application.use_cases.document_management.create_document import CreateDocument
from microsoft_cve_rag.application.use_cases.document_management.link_document_to_graph import LinkDocumentToGraph

document_router = APIRouter()

@document_router.post("/documents/")
async def create_document(title: str, content: str):
    use_case = CreateDocument()
    return use_case.execute(title, content)

@document_router.post("/documents/{document_id}/link")
async def link_document_to_graph(document_id: str, graph_node_id: str):
    use_case = LinkDocumentToGraph()
    return use_case.execute(document_id, graph_node_id)
""",
                        "embedding_routes.py": """# embedding_routes.py
# Purpose: This script defines the API routes for handling embedding operations.
# Explanation: API routes specify the endpoints that can be accessed to perform operations on embeddings.
# Relationships: This script interacts with the embedding use cases and schemas.
# Example Usage:
# from microsoft_cve_rag.presentation.api.v1.routes.embedding_routes import embedding_router

from fastapi import APIRouter
from microsoft_cve_rag.application.use_cases.embedding_generation.generate_embedding import GenerateEmbedding

embedding_router = APIRouter()

@embedding_router.post("/embeddings/")
async def generate_embedding(text: str):
    use_case = GenerateEmbedding()
    return use_case.execute(text)
""",
                        "chat_routes.py": """# chat_routes.py
# Purpose: This script defines the API routes for handling chat operations.
# Explanation: API routes specify the endpoints that can be accessed to perform chat operations.
# Relationships: This script interacts with the chat use cases and schemas.
# Example Usage:
# from microsoft_cve_rag.presentation.api.v1.routes.chat_routes import chat_router

from fastapi import APIRouter
from microsoft_cve_rag.application.use_cases.chat_management.chat_completion import ChatCompletion
from microsoft_cve_rag.application.use_cases.chat_management.text_completion import TextCompletion

chat_router = APIRouter()

@chat_router.post("/chat/completion/")
async def chat_completion(message: str):
    use_case = ChatCompletion()
    return use_case.execute(message)

@chat_router.post("/chat/text/")
async def text_completion(message: str):
    use_case = TextCompletion()
    return use_case.execute(message)
""",
                        "__init__.py": "# Make the routes package importable\n",
                    },
                    "schemas": {
                        "document_schemas.py": """# document_schemas.py
# Purpose: This script defines the Pydantic schemas for document-related API requests and responses.
# Explanation: Pydantic schemas are used to validate and serialize data for API requests and responses.
# Relationships: This script is used by document_routes.py.
# Example Usage:
# from microsoft_cve_rag.presentation.api.v1.schemas.document_schemas import DocumentCreateSchema

from pydantic import BaseModel

class DocumentCreateSchema(BaseModel):
    title: str
    content: str
""",
                        "vector_schemas.py": """# vector_schemas.py
# Purpose: This script defines the Pydantic schemas for vector-related API requests and responses.
# Explanation: Pydantic schemas are used to validate and serialize data for API requests and responses.
# Relationships: This script is used by embedding_routes.py.
# Example Usage:
# from microsoft_cve_rag.presentation.api.v1.schemas.vector_schemas import VectorCreateSchema

from pydantic import BaseModel
from typing import List

class VectorCreateSchema(BaseModel):
    values: List[float]
""",
                        "graph_schemas.py": """# graph_schemas.py
# Purpose: This script defines the Pydantic schemas for graph-related API requests and responses.
# Explanation: Pydantic schemas are used to validate and serialize data for API requests and responses.
# Relationships: This script is used by graph-related routes.
# Example Usage:
# from microsoft_cve_rag.presentation.api.v1.schemas.graph_schemas import GraphNodeCreateSchema

from pydantic import BaseModel

class GraphNodeCreateSchema(BaseModel):
    node_id: str
    label: str
    properties: dict
""",
                        "chat_schemas.py": """# chat_schemas.py
# Purpose: This script defines the Pydantic schemas for chat-related API requests and responses.
# Explanation: Pydantic schemas are used to validate and serialize data for API requests and responses.
# Relationships: This script is used by chat_routes.py.
# Example Usage:
# from microsoft_cve_rag.presentation.api.v1.schemas.chat_schemas import ChatMessageSchema

from pydantic import BaseModel

class ChatMessageSchema(BaseModel):
    content: str
    timestamp: datetime = None
""",
                        "__init__.py": "# Make the schemas package importable\n",
                    },
                    "__init__.py": "# Make the v1 package importable\n",
                },
                "__init__.py": "# Make the api package importable\n",
            },
            "streamlit": {
                "pages": {
                    "home.py": """# home.py
# Purpose: This script implements the home page of the Streamlit application.
# Explanation: This script defines the layout and components of the home page in the Streamlit application.
# Relationships: This script may interact with various components and services for data display.
# Example Usage:
# from microsoft_cve_rag.presentation.streamlit.pages.home import home_page

import streamlit as st

def home_page():
    st.title("Home Page")
    st.write("Welcome to the home page!")
""",
                    "chat.py": """# chat.py
# Purpose: This script implements the chat page of the Streamlit application.
# Explanation: This script defines the layout and components of the chat page in the Streamlit application.
# Relationships: This script may interact with the chat service for handling chat interactions.
# Example Usage:
# from microsoft_cve_rag.presentation.streamlit.pages.chat import chat_page

import streamlit as st
from microsoft_cve_rag.infrastructure.services.chat_service import ChatService

def chat_page(chat_service: ChatService):
    st.title("Chat Page")
    user_input = st.text_input("Enter your message:")
    if st.button("Send"):
        response = chat_service.send_message(user_input)
        st.write("Response:", response)
""",
                    "__init__.py": "# Make the pages package importable\n",
                },
                "components": {
                    "sidebar.py": """# sidebar.py
# Purpose: This script implements the sidebar component for navigation in the Streamlit application.
# Explanation: This script defines the layout and components of the sidebar used for navigation.
# Relationships: This script may interact with various pages for navigation purposes.
# Example Usage:
# from microsoft_cve_rag.presentation.streamlit.components.sidebar import sidebar

import streamlit as st

def sidebar():
    st.sidebar.title("Navigation")
    st.sidebar.radio("Go to", ["Home", "Chat"])
""",
                    "chat_interface.py": """# chat_interface.py
# Purpose: This script implements the chat interface component for chat interactions in the Streamlit application.
# Explanation: This script defines the layout and components of the chat interface used for chat interactions.
# Relationships: This script may interact with the chat service for handling chat interactions.
# Example Usage:
# from microsoft_cve_rag.presentation.streamlit.components.chat_interface import chat_interface
# from microsoft_cve_rag.infrastructure.services.chat_service import ChatService

import streamlit as st

def chat_interface(chat_service: ChatService):
    user_input = st.text_input("Enter your message:")
    if st.button("Send"):
        response = chat_service.send_message(user_input)
        st.write("Response:", response)
""",
                    "__init__.py": "# Make the components package importable\n",
                },
                "app.py": """# app.py
# Purpose: This script serves as the main entry point for the Streamlit application.
# Explanation: This script initializes and runs the Streamlit application, incorporating various pages and components.
# Relationships: This script imports and initializes various modules and components of the application.
# Example Usage:
# from microsoft_cve_rag.presentation.streamlit.app import main

import streamlit as st
from microsoft_cve_rag.presentation.streamlit.pages.home import home_page
from microsoft_cve_rag.presentation.streamlit.pages.chat import chat_page
from microsoft_cve_rag.presentation.streamlit.components.sidebar import sidebar

def main():
    sidebar()
    page = st.sidebar.radio("Select Page", ["Home", "Chat"])
    if page == "Home":
        home_page()
    elif page == "Chat":
        chat_page()

if __name__ == "__main__":
    main()
""",
                "__init__.py": "# Make the streamlit package importable\n",
            },
            "__init__.py": "# Make the presentation package importable\n",
        },
        "config": {
            "settings.py": """# settings.py
# Purpose: This script defines the settings used across the application.
# Explanation: Settings include configuration parameters that are used throughout the application, such as database connection strings, API keys, etc.
# Relationships: This script is imported by various modules to access configuration settings.
# Example Usage:
# from microsoft_cve_rag.config.settings import settings

class Settings:
    DATABASE_URL = "sqlite:///./test.db"
    NEO4J_URL = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"

settings = Settings()
""",
            "logging.py": """# logging.py
# Purpose: This script sets up the logging configuration for the application.
# Explanation: Logging is used to track events that happen when the software runs, which can be helpful for debugging and monitoring.
# Relationships: This script is imported by the main application and other modules to log information.
# Example Usage:
# from microsoft_cve_rag.config.logging import setup_logging

import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO)
""",
            "__init__.py": "# Make the config package importable\n",
        },
        "tests": {
            "unit": {
                "domain": {},
                "application": {},
                "infrastructure": {},
                "__init__.py": "# Make the unit tests package importable\n",
            },
            "integration": {
                "api": {},
                "database": {},
                "__init__.py": "# Make the integration tests package importable\n",
            },
            "e2e": {"__init__.py": "# Make the end-to-end tests package importable\n"},
            "__init__.py": "# Make the tests package importable\n",
        },
        "main.py": """# main.py
# Purpose: This script serves as the main entry point for the application, initializing and running the application.
# Explanation: The main script is responsible for setting up and starting the application.
# Relationships: This script imports and initializes various modules and components of the application.
# Example Usage:
# from microsoft_cve_rag.main import main

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
""",
        "requirements.txt": """# requirements.txt
# Purpose: This file lists all the Python package dependencies required for the application.
# Explanation: The requirements file is used by package managers like pip to install dependencies.
# Relationships: This file is used during the setup and deployment of the application.
# Example Usage:
# pip install -r requirements.txt

fastapi
uvicorn
sqlalchemy
pydantic
neo4j
redis
requests
""",
        "__init__.py": "# Make the microsoft_cve_rag package importable\n",
    }
}

impl_structure = {
    "microsoft_cve_rag": {
        "infrastructure": {
            "repositories": {
                "document_repository_impl.py": """# document_repository_impl.py
# Purpose: This script implements the DocumentRepository interface to handle document data operations.
# Explanation: This script provides the actual implementation of how documents are stored and retrieved, typically interacting with a database.
# Relationships: This script interacts with the Document entity and the database models.
# Example Usage:
# from microsoft_cve_rag.infrastructure.repositories.document_repository_impl import DocumentRepositoryImpl
# from microsoft_cve_rag.domain.entities.document import Document
# from microsoft_cve_rag.application.interfaces.repositories.document_repository import DocumentRepository

class DocumentRepositoryImpl(DocumentRepository):
    def save(self, document: Document):
        # Implementation to save the document to the database
        pass

    def find_by_id(self, document_id: str) -> Document:
        # Implementation to retrieve a document by its ID from the database
        pass
""",
                "vector_repository_impl.py": """# vector_repository_impl.py
# Purpose: This script implements the VectorRepository interface to handle vector data operations.
# Explanation: This script provides the actual implementation of how vectors are stored and retrieved, typically interacting with a database.
# Relationships: This script interacts with the Vector entity and the database models.
# Example Usage:
# from microsoft_cve_rag.infrastructure.repositories.vector_repository_impl import VectorRepositoryImpl
# from microsoft_cve_rag.domain.entities.vector import Vector
# from microsoft_cve_rag.application.interfaces.repositories.vector_repository import VectorRepository

class VectorRepositoryImpl(VectorRepository):
    def save(self, vector: Vector):
        # Implementation to save the vector to the database
        pass

    def find_by_id(self, vector_id: str) -> Vector:
        # Implementation to retrieve a vector by its ID from the database
        pass
""",
                "graph_repository_impl.py": """# graph_repository_impl.py
# Purpose: This script implements the GraphRepository interface to handle graph data operations.
# Explanation: This script provides the actual implementation of how graph nodes are stored and retrieved, typically interacting with a graph database.
# Relationships: This script interacts with the GraphNode entity and the database models.
# Example Usage:
# from microsoft_cve_rag.infrastructure.repositories.graph_repository_impl import GraphRepositoryImpl
# from microsoft_cve_rag.domain.entities.graph_node import GraphNode
# from microsoft_cve_rag.application.interfaces.repositories.graph_repository import GraphRepository

class GraphRepositoryImpl(GraphRepository):
    def save(self, graph_node: GraphNode):
        # Implementation to save the graph node to the database
        pass

    def find_by_id(self, graph_node_id: str) -> GraphNode:
        # Implementation to retrieve a graph node by its ID from the database
        pass

    def link(self, document: Document, graph_node: GraphNode):
        # Implementation to link a document to a graph node in the database
        pass
""",
                "__init__.py": "# Make the repositories package importable\n",
            },
            "services": {
                "openai_service_impl.py": """# openai_service_impl.py
# Purpose: This script implements the service interface for OpenAI.
# Explanation: This script provides the actual implementation of how to interact with the OpenAI API for chat and embedding services.
# Relationships: This script implements the EmbeddingService and ChatService interfaces.
# Example Usage:
# from microsoft_cve_rag.infrastructure.services.openai_service_impl import OpenAIServiceImpl
# from microsoft_cve_rag.domain.services.embedding_service import EmbeddingService
# from microsoft_cve_rag.domain.services.chat_service import ChatService

class OpenAIServiceImpl(EmbeddingService, ChatService):
    def generate(self, text: str) -> List[float]:
        # Implementation to generate embeddings using OpenAI
        pass

    def send_message(self, message: str) -> str:
        # Implementation to send chat message using OpenAI
        pass
""",
                "groq_service_impl.py": """# groq_service_impl.py
# Purpose: This script implements the service interface for Groq.
# Explanation: This script provides the actual implementation of how to interact with the Groq API for chat and embedding services.
# Relationships: This script implements the EmbeddingService and ChatService interfaces.
# Example Usage:
# from microsoft_cve_rag.infrastructure.services.groq_service_impl import GroqServiceImpl
# from microsoft_cve_rag.domain.services.embedding_service import EmbeddingService
# from microsoft_cve_rag.domain.services.chat_service import ChatService

class GroqServiceImpl(EmbeddingService, ChatService):
    def generate(self, text: str) -> List[float]:
        # Implementation to generate embeddings using Groq
        pass

    def send_message(self, message: str) -> str:
        # Implementation to send chat message using Groq
        pass
""",
                "ollama_service_impl.py": """# ollama_service_impl.py
# Purpose: This script implements the service interface for Ollama.
# Explanation: This script provides the actual implementation of how to interact with the Ollama API for chat and embedding services.
# Relationships: This script implements the EmbeddingService and ChatService interfaces.
# Example Usage:
# from microsoft_cve_rag.infrastructure.services.ollama_service_impl import OllamaServiceImpl
# from microsoft_cve_rag.domain.services.embedding_service import EmbeddingService
# from microsoft_cve_rag.domain.services.chat_service import ChatService

class OllamaServiceImpl(EmbeddingService, ChatService):
    def generate(self, text: str) -> List[float]:
        # Implementation to generate embeddings using Ollama
        pass

    def send_message(self, message: str) -> str:
        # Implementation to send chat message using Ollama
        pass
""",
                "__init__.py": "# Make the services package importable\n",
            },
            "__init__.py": "# Make the infrastructure package importable\n",
        }
    }
}


import os


def create_structure_core(base_path, structure):
    for item, sub_items in structure.items():
        path = os.path.join(base_path, item)
        if isinstance(sub_items, dict):
            os.makedirs(path, exist_ok=True)
            create_structure_core(path, sub_items)
        else:
            with open(path, "w") as f:
                f.write(sub_items)


# Generate the project structure
create_structure_core(".", structure)
print("Folder structure created successfully!")


def create_structure_extras(base_path, structure):
    for item, sub_items in structure.items():
        path = os.path.join(base_path, item)
        if isinstance(sub_items, dict):
            os.makedirs(path, exist_ok=True)
            create_structure_extras(path, sub_items)
        else:
            with open(path, "w") as f:
                f.write(sub_items)


# Generate the _impl scripts
create_structure_extras(".", impl_structure)
print("Folder structure with _impl scripts created successfully!")
