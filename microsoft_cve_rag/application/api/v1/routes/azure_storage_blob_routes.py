# application/routers/blob_storage_routes.py
import logging
import datetime
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

# Imports (adjust as needed)
try:
    from application.app_utils import REPORTS_DIR
    from application.services.azure_storage_blob_service import AzureStorageSettings, AzureBlobStorageService
    from azure.core.exceptions import AzureError
    from pydantic import ValidationError
except ImportError as e:
    logging.error(f"Failed to import necessary modules for Azure Blob router: {e}")
    raise

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/blob-storage-test",
    tags=["Azure Blob Storage Test"],
)

# Dependency(callable)


async def get_azure_settings() -> AzureStorageSettings:
    """
    Dependency callable to load Azure Storage settings.
    Returns:
        AzureStorageSettings: Validated Azure Storage settings.
    Raises:
        HTTPException: If settings validation fails or loading fails.
    """
    try:
        return AzureStorageSettings()
    except ValidationError as e:
        logger.error(f"Azure Storage settings validation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Azure configuration error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading Azure settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load Azure settings")

# --- Test Routes (Modified) ---


@router.get("/status", summary="Check Azure Blob Storage configuration loading")
async def check_blob_status(settings: AzureStorageSettings = Depends(get_azure_settings)):
    # Status check now only validates account settings loading
    logger.info("Route /blob-storage-test/status called.")
    try:
        service = AzureBlobStorageService(settings)
        logger.info("AzureBlobStorageService instantiated successfully (account settings loaded).")
        # Attempt to get service client to test credentials minimally
        try:
            _ = service.blob_service_client  # Access property to trigger init
            init_status = "BlobServiceClient initialized successfully (connection likely OK)."
        except Exception as conn_err:
            init_status = f"Failed to initialize BlobServiceClient: {conn_err}"
            logger.error(init_status)

        return {
            "message": "AzureBlobStorageService instantiated.",
            "account_settings": settings.display_info(),
            "service_client_init_status": init_status
        }
    except Exception as e:
        logger.exception(f"Error instantiating AzureBlobStorageService: {e}")
        raise HTTPException(status_code=500, detail=f"Error instantiating Service: {e}")


class UploadResponse(BaseModel):
    message: str
    container: str
    uploaded_blob_name: str


@router.post("/upload-test", summary="Upload a test file to Azure Blob Storage", response_model=UploadResponse)
async def upload_test_blob(
    container_name: str = Query(..., description="Name of the target container."),  # Added container_name query param
    settings: AzureStorageSettings = Depends(get_azure_settings)
):
    """
    Uploads a test file to a specific blob name within the specified container.
    Creates local dummy file if needed.
    """
    logger.info(f"Route /blob-storage-test/upload-test called for container: '{container_name}'.")

    if not REPORTS_DIR or not REPORTS_DIR.is_dir():  # Check REPORTS_DIR
        logger.error(f"REPORTS_DIR global path is not set or directory not found: {REPORTS_DIR}")
        raise HTTPException(status_code=500, detail="Server configuration error: Base directory not available.")

    # Define/create local test file
    local_test_dir = REPORTS_DIR / "weekly_kb_report" / "html"
    local_file_path = local_test_dir / "kb_report_20240914.html"

    # Define blob name
    blob_name = f"html/{local_file_path.name}"

    # Perform Upload
    service = AzureBlobStorageService(settings)
    try:
        logger.info(f"Attempting to upload '{local_file_path}' to blob '{blob_name}' in container '{container_name}'")
        # Pass the container_name to the service method
        service.upload_blob(
            container_name=container_name,
            blob_name=blob_name,
            data=local_file_path,
            overwrite=True
        )
        logger.info("Blob uploaded successfully.")
        return UploadResponse(
            message="Test file uploaded successfully.",
            container=container_name,
            uploaded_blob_name=blob_name,
        )
    except FileNotFoundError as e:
        # Could be local file OR container not found
        logger.error(f"File not found during upload: {e}", exc_info=True)
        detail = str(e)
        status_code = 404
        if "Azure container" in detail:
            status_code = 404  # Or maybe 400 Bad Request if container name was user input error
        raise HTTPException(status_code=status_code, detail=detail)
    except (ConnectionError, IOError, AzureError) as e:
        logger.error(f"Failed to upload test blob '{blob_name}' to container '{container_name}': {e}", exc_info=True)
        status_code = 503 if isinstance(e, ConnectionError) else 500
        raise HTTPException(status_code=status_code, detail=f"Failed to upload to Azure: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error during blob upload test: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during upload: {e}")


# --- Delete Blob Test ---
class DeleteBlobRequest(BaseModel):
    container_name: str = Field(..., description="Name of the container holding the blob.")
    blob_name: str = Field(..., description="Full name/path of the blob to delete within the container.")


class DeleteBlobResponse(BaseModel):
    message: str
    container: str
    blob_name: str
    deleted: bool


@router.post("/delete-blob", summary="Delete a specific blob", response_model=DeleteBlobResponse)
async def delete_test_blob(
    request_body: DeleteBlobRequest,
    settings: AzureStorageSettings = Depends(get_azure_settings)
):
    container = request_body.container_name
    blob = request_body.blob_name
    logger.info(f"Route /blob-storage-test/delete-blob called for blob: '{blob}' in container '{container}'")
    if not container or not blob:
        raise HTTPException(status_code=400, detail="container_name and blob_name are required.")

    service = AzureBlobStorageService(settings)
    try:
        deleted = service.delete_blob(container_name=container, blob_name=blob)  # Pass container name
        message = f"Blob '{blob}' deleted successfully from '{container}'." if deleted else f"Blob '{blob}' not found in container '{container}'."
        logger.info(message)
        return DeleteBlobResponse(message=message, container=container, blob_name=blob, deleted=deleted)
    except (ConnectionError, IOError, AzureError) as e:
        logger.error(f"Failed to delete blob '{blob}' from container '{container}': {e}", exc_info=True)
        status_code = 503 if isinstance(e, ConnectionError) else 500
        raise HTTPException(status_code=status_code, detail=f"Failed to delete blob: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error during blob deletion for '{blob}' in '{container}': {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during deletion: {e}")


# --- Delete Prefix Test ---
class DeletePrefixRequest(BaseModel):
    container_name: str = Field(..., description="Name of the container to delete from.")
    prefix: str = Field(..., description="Prefix (simulated folder path) to delete blobs from, e.g., 'test_uploads/'.")


class DeletePrefixResponse(BaseModel):
    message: str
    container: str
    prefix: str
    deleted_count: int


@router.post("/delete-prefix", summary="Delete all blobs starting with a prefix", response_model=DeletePrefixResponse)
async def delete_test_prefix(
    request_body: DeletePrefixRequest,
    settings: AzureStorageSettings = Depends(get_azure_settings)
):
    container = request_body.container_name
    prefix = request_body.prefix
    logger.info(f"Route /blob-storage-test/delete-prefix called for prefix: '{prefix}' in container '{container}'")
    if not container or not prefix:
        raise HTTPException(status_code=400, detail="container_name and prefix are required.")

    service = AzureBlobStorageService(settings)
    try:
        deleted_count = service.delete_blobs_with_prefix(container_name=container, prefix=prefix)  # Pass container name
        message = f"Successfully submitted deletion for {deleted_count} blobs with prefix '{prefix}' in container '{container}'." if deleted_count > 0 else f"No blobs found with prefix '{prefix}' in container '{container}'."
        logger.info(message)
        return DeletePrefixResponse(message=message, container=container, prefix=prefix, deleted_count=deleted_count)
    except FileNotFoundError as e:  # Catch container not found specifically
        logger.error(f"Container not found during prefix deletion: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=str(e))
    except (ConnectionError, IOError, AzureError) as e:
        logger.error(f"Failed to delete blobs with prefix '{prefix}' in container '{container}': {e}", exc_info=True)
        status_code = 503 if isinstance(e, ConnectionError) else 500
        raise HTTPException(status_code=status_code, detail=f"Failed to delete blobs by prefix: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error during prefix deletion for '{prefix}' in container '{container}': {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during prefix deletion: {e}")
