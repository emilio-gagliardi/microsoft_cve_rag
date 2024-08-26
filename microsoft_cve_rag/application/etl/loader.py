# Purpose: Load transformed data into databases
# Inputs: Transformed data
# Outputs: Loading status
# Dependencies: VectorDBService, GraphDBService

from typing import Any, Dict, List
from application.app_utils import get_app_config
from application.services.vector_db_service import VectorDBService


settings = get_app_config()
embedding_config = settings["EMBEDDING_CONFIG"]
vectordb_config = settings["VECTORDB_CONFIG"]
graphdb_config = settings["GRAPHDB_CONFIG"]
# from application.services.graph_db_service import GraphDBService

# VectorDBService loads credentials internally
vector_db_service = VectorDBService(
    embedding_config, vectordb_config, vectordb_config["tier1_collection"]
)


def load_to_vector_db(data: List[Dict[str, Any]]) -> bool:
    for record in data:
        if "vector" in record:
            vector_db_service.create_vector(record["vector"])
    return True


def load_to_graph_db(data: List[Dict[str, Any]]) -> bool:
    for record in data:
        if "graph_node" in record:
            print("store record in graph db placeholder.")
            # graph_db_service.create_node(record["graph_node"])
    return True
