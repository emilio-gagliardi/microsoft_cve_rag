# Purpose: Transform extracted data
# Inputs: Raw data
# Outputs: Transformed data
# Dependencies: None

from typing import Any, Dict, List
from application.core.models import (
    Document,
    Vector,
    GraphNode,
    DocumentMetadata,
    VectorMetadata,
    GraphNodeMetadata,
)
from application.services.embedding_service import EmbeddingService

embedding_service = EmbeddingService()


def transform(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    transformed_data = []
    for record in data:
        # Transform to Document
        document = Document(
            embedding=embedding_service.generate_embedding(record.get("text", "")),
            metadata=DocumentMetadata(**record.get("metadata", {})),
            text=record.get("text", ""),
            # Add other fields as needed
        )
        transformed_data.append({"document": document})

        # Transform to Vector
        vector = Vector(
            embedding=embedding_service.generate_embedding(record.get("text", "")),
            metadata=VectorMetadata(**record.get("metadata", {})),
            text=record.get("text", ""),
            # Add other fields as needed
        )
        transformed_data.append({"vector": vector})

        # Transform to GraphNode
        graph_node = GraphNode(
            embedding=embedding_service.generate_embedding(record.get("text", "")),
            metadata=GraphNodeMetadata(**record.get("metadata", {})),
            text=record.get("text", ""),
            # Add other fields as needed
        )
        transformed_data.append({"graph_node": graph_node})

    return transformed_data
