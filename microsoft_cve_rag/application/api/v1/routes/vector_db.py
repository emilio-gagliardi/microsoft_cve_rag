from fastapi import APIRouter, HTTPException, Depends
from application.core.schemas.vector_schemas import (
    VectorRecordCreate,
    VectorRecordUpdate,
    VectorRecordResponse,
    VectorRecordQuery,
    VectorRecordQueryResponse,
    BulkVectorRecordCreate,
    BulkVectorRecordDelete,
)
from application.services.vector_db_service import VectorDBService
from application.app_utils import get_app_config
from typing import List, Dict

router = APIRouter()


async def get_vector_db_service():
    settings = get_app_config()
    service = VectorDBService(
        collection=settings["tier1_collection"],
        distance_metric=settings["distance_metric"],
        embedding_config=settings["EMBEDDING_CONFIG"],
        vectordb_config=settings["VECTORDB_CONFIG"],
    )

    try:
        yield service
    finally:
        await service.aclose()


@router.post("/vectors/", response_model=VectorRecordResponse)
def create_vector(
    vector: VectorRecordCreate,
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
):
    """
    Create a new vector.

    Args:
        vector (VectorRecordCreate): The vector data to create.

    Returns:
        VectorRecordResponse: The response containing the vector ID and status message.
    """
    try:
        vector_id = vector_db_service.create_vector(vector)
        return VectorRecordResponse(
            id=vector_id,
            message="Vector created successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vectors/{vector_id}", response_model=VectorRecordResponse)
def get_vector(
    vector_id: str, vector_db_service: VectorDBService = Depends(get_vector_db_service)
):
    """
    Retrieve a vector by its ID.

    Args:
        vector_id (str): The ID of the vector to retrieve.

    Returns:
        VectorRecordResponse: The response containing the vector data and status message.
    """
    try:
        vector = vector_db_service.get_vector(vector_id)
        if vector is None:
            raise HTTPException(status_code=404, detail="Vector not found")
        return VectorRecordResponse(
            id=vector["id"],
            message="Vector retrieved successfully",
            payload=vector["payload"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/vectors/{vector_id}", response_model=VectorRecordResponse)
def update_vector(
    vector_id: str,
    vector: VectorRecordUpdate,
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
):
    """
    Update an existing vector by its ID.

    Args:
        vector_id (str): The ID of the vector to update.
        vector (VectorRecordUpdate): The updated vector data.

    Returns:
        VectorRecordResponse: The response containing the updated vector data and status message.
    """
    try:
        updated_vector = vector_db_service.update_vector(vector_id, vector)
        if updated_vector is None:
            raise HTTPException(status_code=404, detail="Vector not found")
        return VectorRecordResponse(
            id=updated_vector["id"],
            message="Vector updated successfully",
            payload=updated_vector["payload"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/vectors/{vector_id}", response_model=VectorRecordResponse)
def delete_vector(
    vector_id: str, vector_db_service: VectorDBService = Depends(get_vector_db_service)
):
    """
    Delete a vector by its ID.

    Args:
        vector_id (str): The ID of the vector to delete.

    Returns:
        VectorRecordResponse: The response containing the deleted vector data and status message.
    """
    try:
        deleted_vector = vector_db_service.delete_vector(vector_id)
        if deleted_vector is None:
            raise HTTPException(status_code=404, detail="Vector not found")
        return VectorRecordResponse(
            id=deleted_vector["id"],
            message="Vector deleted successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectors/search", response_model=List[VectorRecordQueryResponse])
def search_vectors(
    query: VectorRecordQuery,
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
):
    """
    Search for vectors based on a query.

    Args:
        query (VectorRecordQuery): The search query parameters.

    Returns:
        List[VectorRecordQueryResponse]: A list of vector records matching the query.
    """
    try:
        results = vector_db_service.search_vectors(query.text, query.limit)
        return [
            VectorRecordQueryResponse(id=r.id, score=r.score, payload=r.payload)
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectors/bulk", response_model=List[VectorRecordResponse])
def bulk_create_vectors(
    vectors: BulkVectorRecordCreate,
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
):
    """
    Create multiple vectors in bulk.

    Args:
        vectors (BulkVectorRecordCreate): The list of vectors to create.

    Returns:
        List[VectorRecordResponse]: A list of responses for each created vector.
    """
    try:
        results = vector_db_service.bulk_create_vectors(vectors.vectors)
        return [
            VectorRecordResponse(id=r, message="Vector created successfully")
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectors/bulk-delete", response_model=Dict[str, int])
def bulk_delete_vectors(
    vector_ids: BulkVectorRecordDelete,
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
):
    """
    Delete multiple vectors in bulk.

    Args:
        vector_ids (BulkVectorRecordDelete): The list of vector IDs to delete.

    Returns:
        Dict[str, int]: A dictionary containing the number of successfully deleted vectors.
    """
    try:
        deleted_count = vector_db_service.bulk_delete_vectors(vector_ids.ids)
        return {"deleted_count": deleted_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
