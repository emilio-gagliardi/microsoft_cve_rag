# Purpose: Handle graph database operations via API
# Inputs: Graph queries
# Outputs: Query results
# Dependencies: GraphDBService

from fastapi import APIRouter, HTTPException
from uuid import UUID
from application.services.graph_db_service import GraphDBService
from application.core.schemas.graph_schemas import (
    GraphRecordCreate,
    GraphRecordUpdate,
    GraphRecordResponse,
    QueryRequest,
    QueryResponse,
)
from application.core.models import GraphNode

router = APIRouter()
graph_db_service = GraphDBService()


@router.on_event("shutdown")
def shutdown_event():
    graph_db_service.close()


@router.post("/graph/nodes/", response_model=GraphRecordResponse)
def create_node(node: GraphRecordCreate):
    try:
        graph_node = GraphNode(
            embedding=node.embedding,
            metadata=node.metadata,
            relationships=node.relationships,
            text=node.text,
            class_name=node.class_name,
            created_at=node.created_at,
            updated_at=node.updated_at,
        )
        node_id = graph_db_service.create_node(graph_node)
        return GraphRecordResponse(
            id=str(node_id),
            message="Node created successfully",
            created_at=node.created_at,
            updated_at=node.updated_at,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/nodes/{node_id}", response_model=GraphRecordResponse)
def get_node(node_id: str):
    try:
        node = graph_db_service.get_node(node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Node not found")
        return GraphRecordResponse(
            id=node["id"],
            message="Node retrieved successfully",
            created_at=node["created_at"],
            updated_at=node["updated_at"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/graph/nodes/{node_id}", response_model=GraphRecordResponse)
def update_node(node_id: str, node: GraphRecordUpdate):
    try:
        graph_node = GraphNode(
            id_=UUID(node_id),
            embedding=node.embedding,
            metadata=node.metadata,
            relationships=node.relationships,
            text=node.text,
            class_name=node.class_name,
            created_at=node.created_at,
            updated_at=node.updated_at,
        )
        updated_node = graph_db_service.update_node(node_id, graph_node)
        if updated_node is None:
            raise HTTPException(status_code=404, detail="Node not found")
        return GraphRecordResponse(
            id=updated_node["id"],
            message="Node updated successfully",
            created_at=updated_node["created_at"],
            updated_at=updated_node["updated_at"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/graph/nodes/{node_id}", response_model=GraphRecordResponse)
def delete_node(node_id: str):
    try:
        deleted_node = graph_db_service.delete_node(node_id)
        if deleted_node is None:
            raise HTTPException(status_code=404, detail="Node not found")
        return GraphRecordResponse(
            id=deleted_node["id"],
            message="Node deleted successfully",
            created_at=deleted_node["created_at"],
            updated_at=deleted_node["updated_at"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/query/", response_model=QueryResponse)
def query(query_request: QueryRequest):
    try:
        results = graph_db_service.query(query_request.cypher, query_request.parameters)
        return QueryResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
