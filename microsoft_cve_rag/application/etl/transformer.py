# Purpose: Transform extracted data
# Inputs: Raw data
# Outputs: Transformed data
# Dependencies: None

from typing import Any, Dict, List
from application.core.models.basic_models import (
    Document,
    Vector,
    GraphNode,
    DocumentMetadata,
    VectorMetadata,
    GraphNodeMetadata,
)
from application.services.embedding_service import EmbeddingService
from application.app_utils import get_app_config

settings = get_app_config()
embedding_config = settings["EMBEDDING_CONFIG"]
embedding_service = EmbeddingService.from_provider_name("fastembed")


def transform(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    transformed_data = []
    print("begin transform process")
    # for record in data:
    #     # Transform to Document
    #     document = Document(
    #         embedding=embedding_service.generate_embedding(record.get("text", "")),
    #         metadata=DocumentMetadata(**record.get("metadata", {})),
    #         text=record.get("text", ""),
    #         # Add other fields as needed
    #     )
    #     transformed_data.append({"document": document})

    #     # Transform to Vector
    #     vector = Vector(
    #         embedding=embedding_service.generate_embedding(record.get("text", "")),
    #         metadata=VectorMetadata(**record.get("metadata", {})),
    #         text=record.get("text", ""),
    #         # Add other fields as needed
    #     )
    #     transformed_data.append({"vector": vector})

    #     # Transform to GraphNode
    #     graph_node = GraphNode(
    #         embedding=embedding_service.generate_embedding(record.get("text", "")),
    #         metadata=GraphNodeMetadata(**record.get("metadata", {})),
    #         text=record.get("text", ""),
    #         # Add other fields as needed
    #     )
    #     transformed_data.append({"graph_node": graph_node})

    return transformed_data
