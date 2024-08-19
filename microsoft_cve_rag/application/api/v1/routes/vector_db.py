from fastapi import APIRouter, HTTPException, Depends, Query
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
from typing import Optional, List, Dict, Union
import logging

logger = logging.getLogger(__name__)


router = APIRouter()


async def get_vector_db_service():
    settings = get_app_config()
    vector_db_settings = settings["VECTORDB_CONFIG"]
    embedding_settings = settings["EMBEDDING_CONFIG"]

    service = VectorDBService(
        collection=vector_db_settings["tier1_collection"],
        distance_metric=vector_db_settings["distance_metric"],
        embedding_config=embedding_settings,
        vectordb_config=vector_db_settings,
    )

    try:
        yield service
    finally:
        await service.aclose()


@router.post("/vectors/", response_model=VectorRecordResponse)
async def create_vector(
    vector: VectorRecordCreate,
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
) -> VectorRecordResponse:
    """
    Create a new vector.

    Args:
        vector (VectorRecordCreate): The vector data to create.

    Returns:
        VectorRecordResponse: The response containing the vector ID and status message.
    """
    try:

        result = await vector_db_service.create_vector(vector)
        return VectorRecordResponse(
            id=result["point_id"],
            message="Vector created successfully",
            status=result["status"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vectors/{vector_id}", response_model=VectorRecordResponse)
async def get_vector(
    vector_id: str,
    with_payload: Union[bool, List[str]] = Query(
        "true", description="Payload keys to retrieve"
    ),
    with_vectors: Union[bool, List[str]] = Query(
        "true", description="Whether to retrieve the vector"
    ),
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
) -> VectorRecordResponse:
    """
    Retrieve a vector by its ID.

    Args:
        vector_id (str): The ID of the vector to retrieve.
        with_payload (Union[bool, List[str]]): Payload keys to retrieve. Defaults to True.
        with_vectors (bool): Whether to retrieve the vector. Defaults to True.

    Returns:
        VectorRecordResponse: The response containing the vector data and status message.
    """
    try:
        # Convert string 'true' to boolean True
        def convert_to_bool(value):
            if isinstance(value, str):
                if value.lower() == "true":
                    return True
                elif value.lower() == "false":
                    return False
            return value

        with_payload_actual = convert_to_bool(with_payload[0])
        with_vectors_actual = convert_to_bool(with_vectors[0])

        result = await vector_db_service.get_vector(
            vector_id,
            with_payload=with_payload_actual,
            with_vectors=with_vectors_actual,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Vector not found")
        return VectorRecordResponse(
            id=result.id,
            message="Vector retrieved",
            vector=result.vector,
            payload=result.payload,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/vectors/{vector_id}", response_model=VectorRecordResponse)
async def update_vector(
    vector_id: str,
    vector: VectorRecordUpdate,
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
):
    """
    Update an existing vector by its ID. If text is passed, the service will automatically recompute the vector embeddings and overwrite the existing text associated with the vector. VectorRecordUpdate defines the available metadata keys that can be included in the request.

    Args:
        vector_id (str): The ID of the vector to update.
        vector (VectorRecordUpdate): The updated vector data.

    Returns:
        VectorRecordResponse: The response containing the updated vector_id and message.
    """
    try:
        response = await vector_db_service.update_vector(vector_id, vector)
        print(f"{type(response)}: {response}")
        return VectorRecordResponse(
            id=vector_id,
            message=f"Vector updated successfully. {response['status']}.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/vectors/{vector_id}", response_model=VectorRecordResponse)
async def delete_vector(
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
        response = await vector_db_service.delete_vector(vector_id)

        return VectorRecordResponse(
            id=str(response["operation_id"]),
            message=f"Vector deleted successfully. {response['status']}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/points/{vector_id}", response_model=VectorRecordResponse)
async def delete_point(
    vector_id: Optional[str] = None,
    metadata_id: Optional[str] = None,
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
):
    """
    Delete a point by its vector ID or metadata ID. One is required.

    Args:
        vector_id (Optional[str]): The ID of the vector to delete.
        metadata_id (Optional[str]): The metadata ID to delete.

    Returns:
        VectorRecordResponse: The response containing the deleted point data and status message.
    """
    print(f"vector_id: {vector_id}")
    if not vector_id and not metadata_id:
        raise ValueError("Must pass either vector_id or metadata.id")
    try:
        response = await vector_db_service.delete_point(
            vector_id=vector_id, metadata_id=metadata_id
        )

        return VectorRecordResponse(
            id=str(response["operation_id"]),
            message=f"Point deleted successfully. {response['status']}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectors/search", response_model=List[VectorRecordQueryResponse])
async def search_vectors(
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
        results = await vector_db_service.search_vectors(query.text, query.limit)
        return [
            VectorRecordQueryResponse(id=r.id, score=r.score, payload=r.payload)
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectors/bulk", response_model=List[VectorRecordResponse])
async def bulk_create_vectors(
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
        results = await vector_db_service.bulk_create_vectors(vectors.vectors)
        return [
            VectorRecordResponse(id=r, message="Vector created successfully")
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectors/bulk-delete", response_model=Dict[str, int])
async def bulk_delete_vectors(
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
        deleted_count = await vector_db_service.bulk_delete_vectors(vector_ids.ids)
        return {"deleted_count": deleted_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
