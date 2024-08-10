# Purpose: Load transformed data into databases
# Inputs: Transformed data
# Outputs: Loading status
# Dependencies: VectorDBService, GraphDBService

from typing import Any, Dict, List
from application.services.vector_db_service import VectorDBService
from application.services.graph_db_service import GraphDBService

vector_db_service = VectorDBService()
graph_db_service = GraphDBService()


def load_to_vector_db(data: List[Dict[str, Any]]) -> bool:
    for record in data:
        if "vector" in record:
            vector_db_service.create_vector(record["vector"])
    return True


def load_to_graph_db(data: List[Dict[str, Any]]) -> bool:
    for record in data:
        if "graph_node" in record:
            graph_db_service.create_node(record["graph_node"])
    return True
