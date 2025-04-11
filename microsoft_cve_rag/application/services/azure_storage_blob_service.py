# application/services/azure_storage_blob_service.py
import logging
from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from fastapi import HTTPException
from pathlib import Path
from typing import Union, Optional, List, Type
from types import TracebackType
from azure.storage.blob import (
    BlobServiceClient,
    ContainerClient,
    StorageErrorCode
)
from azure.core.exceptions import ResourceNotFoundError, AzureError

logger = logging.getLogger(__name__)

# ==================================================================
# Main classes =====================================================
# ==================================================================


class AzureStorageSettings(BaseSettings):
    """
    Holds Azure Blob Storage ACCOUNT connection settings, loaded from environment
    variables or a .env file. Container name is specified per operation.
    """
    account_name: str = Field(..., validation_alias='AZURE_STORAGE_ACCOUNT_NAME')
    account_key: SecretStr = Field(..., validation_alias='AZURE_STORAGE_ACCOUNT_KEY')

    model_config = SettingsConfigDict(
        env_file='.env.local',
        extra='ignore',
        case_sensitive=False,
    )

    @property
    def account_url(self) -> str:
        """Construct the blob service endpoint URL."""
        return f"https://{self.account_name}.blob.core.windows.net"

    def display_info(self) -> dict:
        """Return a dictionary with non-sensitive account info for display."""
        return {
            "account_name": self.account_name,
            "account_key_set": bool(self.account_key),
            "account_url": self.account_url,
        }


class AzureBlobStorageService:
    """
    A service class to interact with Azure Blob Storage using a context manager
    for resource handling.

    Ensures connection is active before performing operations like upload or delete.

    Usage:
        settings = AzureStorageSettings(...)
        with AzureBlobStorageService(settings) as service:
            if service.is_connected:
                service.upload_blob(...)
                exists = service.blob_exists(...)
                service.delete_blob(...)
    """
    def __init__(self, settings: AzureStorageSettings):
        """
        Initializes the service with Azure Storage settings.
        Connection is established when entering the 'with' block.
        """
        self._settings = settings
        self._blob_service_client: Optional[BlobServiceClient] = None
        logger.info(f"AzureBlobStorageService initialized for account '{settings.account_name}'. Ready to connect.")

    def __enter__(self) -> 'AzureBlobStorageService':
        """
        Establishes the connection to Azure Blob Storage using the BlobServiceClient.
        Called when entering the 'with' block.
        """
        if not self.is_connected:
            try:
                logger.info(f"Connecting to Azure Blob Storage account '{self._settings.account_name}'...")
                # Use get_secret_value() for secure key handling if using Pydantic SecretStr etc.
                credential = self._settings.account_key.get_secret_value() if hasattr(self._settings.account_key, 'get_secret_value') else self._settings.account_key

                self._blob_service_client = BlobServiceClient(
                    account_url=self._settings.account_url,
                    credential=credential
                )
                # Optionally, add a quick check like listing containers (can add latency)
                # list(self._blob_service_client.list_containers(results_per_page=1))
                logger.info("Connection to Azure Blob Storage established.")
            except AzureError as e:
                logger.error(f"Failed to connect to Azure Blob Storage: {e}", exc_info=True)
                self._blob_service_client = None  # Ensure client is None on failure
                raise ConnectionError(f"Failed to connect to Azure Blob Storage: {e}") from e
            except Exception as e:
                logger.error(f"Unexpected error connecting to Azure Blob Storage: {e}", exc_info=True)
                self._blob_service_client = None  # Ensure client is None on failure
                raise ConnectionError(f"Unexpected error connecting to Azure: {e}") from e
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        """
        Closes the Azure Blob Storage connection.
        Called when exiting the 'with' block.
        """
        if self._blob_service_client:
            try:
                logger.info("Closing Azure Blob Storage connection...")
                self._blob_service_client.close()
                logger.info("Azure Blob Storage connection closed.")
            except Exception as e:
                # Log error during close but don't re-raise unless critical
                logger.error(f"Error closing Azure BlobServiceClient: {e}", exc_info=True)
            finally:
                # Ensure the client attribute is reset regardless of close success/failure
                self._blob_service_client = None
        # If an exception occurred within the 'with' block, it will be re-raised
        # unless this method returns True. We return None (implicitly False) to allow propagation.

    @property
    def is_connected(self) -> bool:
        """Checks if the BlobServiceClient is initialized and likely connected."""
        # Note: This checks if the client object exists. It doesn't guarantee
        # the network connection is still active or credentials are valid.
        # Actual operations will confirm connectivity.
        return self._blob_service_client is not None

    def _ensure_connection(self) -> None:
        """Raises ConnectionError if the client is not connected."""
        if not self.is_connected:
            raise ConnectionError(
                "Azure Blob Storage client is not connected. "
                "Ensure the service is used within a 'with' block."
            )
        # We can be reasonably sure self._blob_service_client is not None here
        # because is_connected would have been False otherwise.

    # --- Internal helper to get container client ---

    def _get_container_client(self, container_name: str) -> ContainerClient:
        """
        Gets a ContainerClient for the specified container name.
        Assumes connection is already ensured.
        """
        if not container_name:
            raise ValueError("Container name cannot be empty.")
        # Connection should be ensured by the calling public method
        assert self._blob_service_client is not None, "Connection not established before getting container client"
        try:
            return self._blob_service_client.get_container_client(container_name)
        except AzureError as e:
            logger.error(f"Failed to get Azure ContainerClient for '{container_name}': {e}", exc_info=True)
            # Map specific errors if needed, otherwise raise a generic connection/IO error
            raise IOError(f"Failed to get Azure ContainerClient for '{container_name}': {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error getting Azure ContainerClient for '{container_name}': {e}", exc_info=True)
            raise IOError(f"Unexpected error getting Azure ContainerClient for '{container_name}': {e}") from e

    # --- Core Blob Operations ---

    def upload_blob(self, container_name: str, blob_name: str, data: Union[bytes, str, Path], overwrite: bool = True) -> None:
        """
        Uploads data to a specific blob within the specified container.
        Ensures connection is active before proceeding.

        Args:
            container_name: The name of the target container.
            blob_name: The full name/path of the blob within the container.
            data: The data to upload (bytes, utf-8 string, or pathlib.Path object).
            overwrite: Whether to overwrite the blob if it already exists. Defaults to True.

        Raises:
            ConnectionError: If the service is not connected.
            FileNotFoundError: If container doesn't exist or local file path is invalid.
            IOError: For Azure storage or other I/O related errors during upload.
            TypeError: If data is of an unsupported type.
            ValueError: If container_name is empty.
        """
        self._ensure_connection()
        logger.info(f"Attempting to upload data to blob: '{blob_name}' in container '{container_name}'")

        processed_data: bytes
        if isinstance(data, Path):
            if not data.is_file():
                logger.error(f"Local file not found for upload: {data}")
                raise FileNotFoundError(f"Local file not found: {data}")
            logger.info(f"Reading data from file: {data}")
            try:
                processed_data = data.read_bytes()
            except OSError as e:
                logger.error(f"Error reading local file {data}: {e}", exc_info=True)
                raise IOError(f"Error reading local file {data}: {e}") from e
        elif isinstance(data, str):
            logger.info("Encoding string data to utf-8")
            processed_data = data.encode('utf-8')
        elif isinstance(data, bytes):
            processed_data = data
        else:
            raise TypeError("Unsupported data type for upload. Use bytes, str, or pathlib.Path.")

        try:
            client = self._get_container_client(container_name)
            logger.info(f"Uploading {len(processed_data)} bytes to '{blob_name}' in '{container_name}'. Overwrite={overwrite}")
            client.upload_blob(name=blob_name, data=processed_data, overwrite=overwrite)
            logger.info(f"Successfully uploaded to blob: '{blob_name}' in container '{container_name}'")
        except ResourceNotFoundError:
            # This error usually means the *container* doesn't exist when uploading
            logger.error(f"Container '{container_name}' not found during upload to blob '{blob_name}'.")
            raise FileNotFoundError(f"Azure container '{container_name}' not found.")
        except AzureError as e:
            logger.error(f"Azure storage error uploading blob '{blob_name}' to container '{container_name}': {e}", exc_info=True)
            if hasattr(e, 'error_code') and e.error_code == StorageErrorCode.authentication_failed:
                # Authentication errors are often connection-related
                raise ConnectionError(f"Azure Authentication Failed: {e}") from e
            # Catch-all for other Azure-specific storage errors
            raise IOError(f"Azure storage error uploading blob '{blob_name}' to container '{container_name}': {e}") from e
        except Exception as e:
            # Catch unexpected errors (network issues, etc.)
            logger.error(f"Unexpected error uploading blob '{blob_name}' to container '{container_name}': {e}", exc_info=True)
            raise IOError(f"Unexpected error uploading blob '{blob_name}' to container '{container_name}': {e}") from e

    def delete_blob(self, container_name: str, blob_name: str) -> bool:
        """
        Deletes a specific blob from the specified container.
        Ensures connection is active before proceeding.

        Args:
            container_name: The name of the target container.
            blob_name: The full name/path of the blob to delete.

        Returns:
            True if the blob was deleted, False if it did not exist (or container didn't exist).

        Raises:
            ConnectionError: If the service is not connected.
            IOError: For Azure storage or other I/O related errors during deletion.
            ValueError: If container_name is empty.
        """
        self._ensure_connection()
        logger.info(f"Attempting to delete blob: '{blob_name}' from container '{container_name}'")
        try:
            client = self._get_container_client(container_name)
            client.delete_blob(blob=blob_name)
            logger.info(f"Successfully deleted blob: '{blob_name}' from container '{container_name}'")
            return True
        except ResourceNotFoundError:
            # This error can mean the container OR the blob doesn't exist.
            # We can check container existence first if needed, but for deletion,
            # blob not found is often sufficient.
            logger.warning(f"Blob '{blob_name}' or container '{container_name}' not found. Cannot delete.")
            return False  # Treat as "did not delete because it wasn't there"
        except AzureError as e:
            logger.error(f"Azure storage error deleting blob '{blob_name}' from container '{container_name}': {e}", exc_info=True)
            raise IOError(f"Azure storage error deleting blob '{blob_name}' from container '{container_name}': {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error deleting blob '{blob_name}' from container '{container_name}': {e}", exc_info=True)
            raise IOError(f"Unexpected error deleting blob '{blob_name}' from container '{container_name}': {e}") from e

    def delete_blobs_with_prefix(self, container_name: str, prefix: str) -> int:
        """
        Deletes all blobs within the specified container that start with the given prefix.
        Ensures connection is active before proceeding.

        Args:
            container_name: The name of the target container.
            prefix: The prefix (e.g., 'directory/path/') to delete blobs from.

        Returns:
            The number of blobs submitted for deletion. Note: Batch deletion is async on Azure's side.

        Raises:
            ConnectionError: If the service is not connected.
            FileNotFoundError: If the container doesn't exist.
            IOError: For Azure storage or other I/O related errors.
            ValueError: If container_name is empty.
        """
        self._ensure_connection()
        if not prefix:
            logger.warning("Attempted to delete blobs with an empty prefix in container '%s'. No action taken.", container_name)
            return 0

        logger.info(f"Attempting to delete all blobs with prefix: '{prefix}' in container '{container_name}'")
        deleted_count = 0
        try:
            client = self._get_container_client(container_name)
            logger.debug(f"Listing blobs with prefix '{prefix}' in container '{container_name}'...")
            # List blobs efficiently by iterating directly
            blobs_to_delete = [blob.name for blob in client.list_blobs(name_starts_with=prefix)]

            if not blobs_to_delete:
                logger.info(f"No blobs found with prefix '{prefix}' in container '{container_name}'. Nothing to delete.")
                return 0

            logger.info(f"Found {len(blobs_to_delete)} blobs to delete with prefix '{prefix}' in container '{container_name}'. Submitting batch deletion...")
            # Use batch deletion for efficiency
            # Note: delete_blobs can raise errors too, e.g., if one blob fails.
            # Consider adding specific handling for partial failures if needed.
            client.delete_blobs(*blobs_to_delete)
            deleted_count = len(blobs_to_delete)
            logger.info(f"Successfully submitted batch deletion for {deleted_count} blobs with prefix '{prefix}' in container '{container_name}'.")
            return deleted_count

        except ResourceNotFoundError:
            logger.error(f"Container '{container_name}' not found during prefix deletion for '{prefix}'.")
            raise FileNotFoundError(f"Azure container '{container_name}' not found.")
        except AzureError as e:
            logger.error(f"Azure storage error during prefix deletion for '{prefix}' in container '{container_name}': {e}", exc_info=True)
            raise IOError(f"Azure storage error during prefix deletion for '{prefix}' in container '{container_name}': {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during prefix deletion for '{prefix}' in container '{container_name}': {e}", exc_info=True)
            raise IOError(f"Unexpected error during prefix deletion for '{prefix}' in container '{container_name}': {e}") from e

    # --- Utility Methods ---

    def list_container(self, container_name: str) -> List[str]:
        """
        Lists the names of all blobs within the specified container.
        Ensures connection is active before proceeding.

        Args:
            container_name: The name of the container to list.

        Returns:
            A list of blob names (strings).

        Raises:
            ConnectionError: If the service is not connected.
            FileNotFoundError: If the container doesn't exist.
            IOError: For Azure storage or other I/O related errors.
            ValueError: If container_name is empty.
        """
        self._ensure_connection()
        logger.info(f"Listing blobs in container '{container_name}'")
        blob_names: List[str] = []
        try:
            client = self._get_container_client(container_name)
            blob_list = client.list_blobs()
            blob_names = [blob.name for blob in blob_list]
            logger.info(f"Found {len(blob_names)} blobs in container '{container_name}'.")
            return blob_names
        except ResourceNotFoundError:
            logger.error(f"Container '{container_name}' not found during listing.")
            raise FileNotFoundError(f"Azure container '{container_name}' not found.")
        except AzureError as e:
            logger.error(f"Azure storage error listing container '{container_name}': {e}", exc_info=True)
            raise IOError(f"Azure storage error listing container '{container_name}': {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error listing container '{container_name}': {e}", exc_info=True)
            raise IOError(f"Unexpected error listing container '{container_name}': {e}") from e

    def blob_exists(self, container_name: str, blob_name: str) -> bool:
        """
        Checks if a specific blob exists within the specified container.
        Ensures connection is active before proceeding.

        Args:
            container_name: The name of the target container.
            blob_name: The full name/path of the blob to check.

        Returns:
            True if the blob exists, False otherwise (including if the container doesn't exist).

        Raises:
            ConnectionError: If the service is not connected.
            IOError: For Azure storage errors other than 'Not Found'.
            ValueError: If container_name or blob_name is empty.
        """
        self._ensure_connection()
        if not blob_name:
            raise ValueError("Blob name cannot be empty.")
        logger.debug(f"Checking existence of blob '{blob_name}' in container '{container_name}'")
        try:
            # Getting container client first handles container existence implicitly
            # if we let ResourceNotFoundError propagate from _get_container_client.
            # However, blob_client.exists() is more direct and handles container
            # not found gracefully by returning False.
            container_client = self._get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)
            exists = blob_client.exists()
            logger.debug(f"Blob '{blob_name}' in container '{container_name}' exists: {exists}")
            return exists
        except ResourceNotFoundError:
            # This can be raised by _get_container_client if the container doesn't exist.
            # In this context, if the container doesn't exist, the blob doesn't either.
            logger.warning(f"Container '{container_name}' not found while checking for blob '{blob_name}'.")
            return False
        except AzureError as e:
            # Catch other potential Azure errors during the exists check
            logger.error(f"Azure storage error checking existence of blob '{blob_name}' in container '{container_name}': {e}", exc_info=True)
            raise IOError(f"Azure storage error checking blob existence: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error checking existence of blob '{blob_name}' in container '{container_name}': {e}", exc_info=True)
            raise IOError(f"Unexpected error checking blob existence: {e}") from e


# --- Example Usage ---
if __name__ == "__main__":
    # Configure logging for example
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Replace with your actual settings or load from environment/config file
    # Ensure AZURE_STORAGE_ACCOUNT_NAME, AZURE_STORAGE_ACCOUNT_URL, AZURE_STORAGE_ACCOUNT_KEY
    # environment variables are set, or provide values directly.
    import os
    try:
        storage_settings = AzureStorageSettings(
            account_name=os.environ["AZURE_STORAGE_ACCOUNT_NAME"],
            account_url=os.environ["AZURE_STORAGE_ACCOUNT_URL"], # e.g., "https://<your_account_name>.blob.core.windows.net"
            account_key=os.environ["AZURE_STORAGE_ACCOUNT_KEY"]
        )
    except KeyError:
        print("Please set AZURE_STORAGE_ACCOUNT_NAME, AZURE_STORAGE_ACCOUNT_URL, and AZURE_STORAGE_ACCOUNT_KEY environment variables.")
        # Dummy settings for demonstration if env vars not set (will fail connection)
        storage_settings = AzureStorageSettings("dummyaccount", "https://dummyaccount.blob.core.windows.net", "dummykey")
        print("Using dummy settings - connection will likely fail.")


    container_name = "mytestcontainera" # Choose a container name
    test_blob_name = "my_test_file.txt"
    test_prefix = "testdir/"
    test_prefix_blob_name = f"{test_prefix}another_file.log"

    try:
        # Use the service as a context manager
        with AzureBlobStorageService(storage_settings) as service:
            print(f"Service connected: {service.is_connected}")

            # Ensure container exists (or create it - add a create_container method if needed)
            # For this example, assume the container exists or handle ResourceNotFoundError

            # 1. Upload a blob
            print(f"\nUploading '{test_blob_name}'...")
            try:
                service.upload_blob(container_name, test_blob_name, "Hello Azure from Context Manager!")
                print("Upload successful.")
            except (ConnectionError, FileNotFoundError, IOError, ValueError) as e:
                 print(f"Upload failed: {e}")
                 # Depending on the error, you might want to exit or try again

            # 2. Check if blob exists
            print(f"\nChecking if '{test_blob_name}' exists...")
            exists = service.blob_exists(container_name, test_blob_name)
            print(f"Blob '{test_blob_name}' exists: {exists}")

            print(f"\nChecking if 'nonexistent_blob.dat' exists...")
            exists_non = service.blob_exists(container_name, "nonexistent_blob.dat")
            print(f"Blob 'nonexistent_blob.dat' exists: {exists_non}")

            # 3. List blobs in container
            print(f"\nListing blobs in '{container_name}'...")
            try:
                blob_list = service.list_container(container_name)
                print(f"Blobs: {blob_list}")
            except (FileNotFoundError, IOError) as e:
                 print(f"Listing failed: {e}")


            # 4. Upload blob with prefix
            print(f"\nUploading '{test_prefix_blob_name}'...")
            try:
                 service.upload_blob(container_name, test_prefix_blob_name, b"Log data\nMore log data")
                 print("Upload with prefix successful.")
            except (ConnectionError, FileNotFoundError, IOError, ValueError) as e:
                 print(f"Upload with prefix failed: {e}")

             # 5. Delete blob with prefix
            print(f"\nDeleting blobs with prefix '{test_prefix}'...")
            try:
                deleted_count = service.delete_blobs_with_prefix(container_name, test_prefix)
                print(f"Deleted {deleted_count} blobs with prefix '{test_prefix}'.")
            except (FileNotFoundError, IOError) as e:
                 print(f"Prefix deletion failed: {e}")

             # 6. Delete the initial blob
            print(f"\nDeleting '{test_blob_name}'...")
            deleted = service.delete_blob(container_name, test_blob_name)
            print(f"Blob '{test_blob_name}' deleted: {deleted}")

            # Try deleting again (should return False)
            print(f"\nAttempting to delete '{test_blob_name}' again...")
            deleted_again = service.delete_blob(container_name, test_blob_name)
            print(f"Blob '{test_blob_name}' deleted again: {deleted_again}")


    except ConnectionError as e:
        print(f"Failed to establish initial connection: {e}")
    except Exception as e:
        # Catch any other unexpected errors during the 'with' block setup or teardown
         print(f"An unexpected error occurred: {e}")
         logger.error("An unexpected error occurred in the main block.", exc_info=True)


    # Outside the 'with' block, the connection should be closed
    # We can verify by creating a new instance and checking is_connected
    # (though this isn't the typical usage pattern)
    print("\nOutside 'with' block:")
    service_outside = AzureBlobStorageService(storage_settings)
    print(f"Service connected (outside): {service_outside.is_connected}")
    try:
        service_outside.list_container(container_name) # This should fail
    except ConnectionError as e:
        print(f"Caught expected error when using outside 'with' block: {e}")


# ==================================================================
# Utility functions ================================================
# ==================================================================


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
