import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
import stat
# Import necessary components from your application structure
# Adjust imports based on your exact file locations and how you expose globals
try:
    # Assuming globals like REPORTS_DIR are exposed directly after init
    # If they are in a config module, import from there, e.g., from application.config import REPORTS_DIR
    from application.app_utils import REPORTS_DIR
    from application.services.sftp_service import SFTPService, SFTPSettings
    # Import specific exceptions if you want to catch them explicitly
    from paramiko.ssh_exception import AuthenticationException, SSHException
    from pydantic import ValidationError
except ImportError as e:
    # Handle error gracefully if imports fail during development/testing
    logging.error(f"Failed to import necessary modules for SFTP router: {e}")
    # You might re-raise or define dummy classes/variables to allow FastAPI to start
    # For this example, we'll assume imports work if the app structure is correct.
    raise

# Setup logger for this router
logger = logging.getLogger(__name__)

# Create the router instance
router = APIRouter(
    prefix="/sftp-test",  # Prefix for all routes in this router
    tags=["SFTP Test"],    # Tag for OpenAPI documentation
)


# --- Helper Function (Optional but Recommended) ---
# Dependency to get configured SFTP settings instance
# This ensures settings are loaded via Pydantic for each request needing them
async def get_sftp_settings() -> SFTPSettings:
    try:
        return SFTPSettings()
    except ValidationError as e:
        logger.error(f"SFTP settings validation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"SFTP configuration error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading SFTP settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load SFTP settings")

# --- Test Routes ---


@router.get("/status", summary="Check SFTP configuration loading")
async def check_sftp_status(settings: SFTPSettings = Depends(get_sftp_settings)):
    """
    Tests if SFTP settings can be loaded and the SFTPService can be instantiated.
    Does NOT attempt to connect.
    """
    logger.info("Route /sftp-test/status called.")
    try:
        # Just instantiating the service tests basic setup
        service = SFTPService(settings)  # noqa: F841 (unused variable is ok here)
        logger.info(f"SFTPService instantiated successfully (settings loaded).\nAsType: {service}")
        # Return limited, non-sensitive info
        return {
            "message": "SFTPService instantiated successfully.",
            "hostname": settings.hostname,
            "port": settings.port,
            "username": settings.username,
            "key_path_exists": settings.key_path.exists(),
        }
    except Exception as e:
        logger.exception(f"Error instantiating SFTPService: {e}")
        raise HTTPException(status_code=500, detail=f"Error instantiating SFTPService: {e}")


# Pydantic model for list/chdir requests (optional but good practice)
class RemotePathQuery(BaseModel):
    remote_path: str = Field(default=".", description="Remote path on the SFTP server (e.g., '.', '/public_html', 'reports/subdir')")


@router.get("/list", summary="List remote directory contents")
async def list_remote_directory(
    path_query: RemotePathQuery = Depends(),  # Use Pydantic model for query params
    settings: SFTPSettings = Depends(get_sftp_settings)  # Use dependency injection
):
    """
    Connects to SFTP and lists contents of the specified remote directory.
    """
    logger.info(f"Route /sftp-test/list called for path: {path_query.remote_path}")
    service = SFTPService(settings)
    try:
        with service:  # Handles connect/disconnect and initial chdir if implemented
            logger.info(f"Attempting to list directory: {path_query.remote_path}")
            contents = service.list_directory(path_query.remote_path)
            logger.info(f"Successfully listed directory '{path_query.remote_path}'. Found {len(contents)} items.")
            return {
                "remote_path": path_query.remote_path,
                "current_directory_after_list": service.get_current_directory(),  # Get CWD after operation
                "contents": contents
            }
    except (ConnectionError, AuthenticationException, SSHException) as e:
        logger.error(f"SFTP Connection/Auth error listing {path_query.remote_path}: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"SFTP Connection/Auth error: {e}")
    except (OSError, IOError) as e:
        logger.error(f"SFTP OS/IO error listing {path_query.remote_path}: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=f"SFTP Error listing directory '{path_query.remote_path}': {e}")  # 404 if path likely not found
    except Exception as e:
        logger.exception(f"Unexpected error listing {path_query.remote_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected SFTP error: {e}")


@router.get("/chdir", summary="Change remote directory and get new CWD")
async def change_remote_directory(
    path_query: RemotePathQuery = Depends(),
    settings: SFTPSettings = Depends(get_sftp_settings)
):
    """
    Connects to SFTP, changes to the specified remote directory,
    and returns the new current working directory.
    """
    logger.info(f"Route /sftp-test/chdir called for path: {path_query.remote_path}")
    if not path_query.remote_path:
        raise HTTPException(status_code=400, detail="Parameter 'remote_path' cannot be empty.")

    service = SFTPService(settings)
    try:
        with service:
            logger.info(f"Attempting to change directory to: {path_query.remote_path}")
            service.change_directory(path_query.remote_path)
            new_cwd = service.get_current_directory()
            logger.info(f"Successfully changed directory. New CWD: {new_cwd}")
            return {
                "requested_path": path_query.remote_path,
                "current_directory_after_change": new_cwd
            }
    except (ConnectionError, AuthenticationException, SSHException) as e:
        logger.error(f"SFTP Connection/Auth error changing dir to {path_query.remote_path}: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"SFTP Connection/Auth error: {e}")
    except (OSError, IOError) as e:
        logger.error(f"SFTP OS/IO error changing dir to {path_query.remote_path}: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=f"SFTP Error changing directory to '{path_query.remote_path}': {e}")  # 404 if path likely not found
    except Exception as e:
        logger.exception(f"Unexpected error changing dir to {path_query.remote_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected SFTP error: {e}")


@router.post("/upload-test", summary="Upload a test report and thumbnail")
async def upload_test_files(settings: SFTPSettings = Depends(get_sftp_settings)):
    """
    Uploads predefined test files (report.html, thumb.png) to a test
    directory on the SFTP server. Creates local dummy files if they don't exist.
    """
    logger.info("Route /sftp-test/upload-test called.")

    if not REPORTS_DIR or not REPORTS_DIR.is_dir():
        logger.error(f"REPORTS_DIR global path is not set or directory not found: {REPORTS_DIR}")
        raise HTTPException(status_code=500, detail="Server configuration error: REPORTS_DIR not available.")

    # --- Define Local Test File Paths ---
    local_test_dir = REPORTS_DIR / "weekly_kb_report"
    local_html_path = local_test_dir / "html" / "kb_report_20240914.html"
    local_thumb_path = local_test_dir / "thumbnails" / "thumbnail1.jpeg"

    # --- Ensure Local Test Files Exist ---
    try:
        local_test_dir.mkdir(parents=True, exist_ok=True)  # Create dir if needed
        if not local_html_path.exists():
            logger.warning(f"Creating dummy local file: {local_html_path}")
            local_html_path.write_text("<html><body><h1>SFTP Test Report</h1></body></html>", encoding='utf-8')
        if not local_thumb_path.exists():
            logger.warning(f"Creating dummy local file: {local_thumb_path}")
            local_thumb_path.write_text("dummy png data", encoding='utf-8')  # Simple text for dummy png
    except OSError as e:
        logger.error(f"Failed to create local test files in {local_test_dir}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to prepare local test files: {e}")

    # --- Define Remote Test Paths ---
    remote_base_dir = "www/portalfuse.io/public_html/wp-content/uploads/kb_weekly"  # Choose a safe test directory name
    remote_html_path = f"{remote_base_dir}/html/report_{Path(local_html_path.name).stem}.html"
    remote_thumb_path = f"{remote_base_dir}/thumbnails/thumb_{Path(local_thumb_path.name).stem}.jpeg"

    # --- Perform Upload ---
    service = SFTPService(settings)
    uploaded_files = []
    errors = []

    try:
        with service:
            # Upload HTML
            try:
                logger.info(f"Attempting to upload {local_html_path} to {remote_html_path}")
                service.upload_file(local_html_path, remote_html_path)
                logger.info("HTML uploaded successfully.")
                uploaded_files.append(remote_html_path)
            except FileNotFoundError:  # Should not happen if creation above worked
                logger.error(f"Local file missing during upload attempt: {local_html_path}")
                errors.append(f"Local file missing: {local_html_path}")
            except (OSError, IOError, ConnectionError, SSHException) as e:
                logger.error(f"Failed to upload {local_html_path}: {e}", exc_info=True)
                errors.append(f"Failed uploading {local_html_path}: {e}")

            # Upload Thumbnail
            try:
                logger.info(f"Attempting to upload {local_thumb_path} to {remote_thumb_path}")
                service.upload_file(local_thumb_path, remote_thumb_path)
                logger.info("Thumbnail uploaded successfully.")
                uploaded_files.append(remote_thumb_path)
            except FileNotFoundError:   # Should not happen if creation above worked
                logger.error(f"Local file missing during upload attempt: {local_thumb_path}")
                errors.append(f"Local file missing: {local_thumb_path}")
            except (OSError, IOError, ConnectionError, SSHException) as e:
                logger.error(f"Failed to upload {local_thumb_path}: {e}", exc_info=True)
                errors.append(f"Failed uploading {local_thumb_path}: {e}")

    except (ConnectionError, AuthenticationException, SSHException) as e:
        logger.error(f"SFTP Connection/Auth error during upload test: {e}", exc_info=True)
        # If connection fails, we can't know which specific upload failed
        raise HTTPException(status_code=503, detail=f"SFTP Connection/Auth error: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error during upload test: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected SFTP error: {e}")

    # --- Return Result ---
    if errors:
        # Return 207 Multi-Status if some uploads succeeded and some failed
        status_code = 207 if uploaded_files else 500
        return {"status_code": status_code, "uploaded": uploaded_files, "errors": errors}
    else:
        return {"message": "Test files uploaded successfully.", "uploaded": uploaded_files}


@router.get("/validate-kb-weekly", summary="Validate 'kb_weekly' directory structure and file counts")
async def validate_kb_weekly_structure(
    # Optional: Allow overriding the base path if needed for testing/other reports
    base_path: str = Query("kb_weekly", description="The base directory to validate (defaults to 'kb_weekly')"),
    settings: SFTPSettings = Depends(get_sftp_settings)
):
    """
    Connects to SFTP and performs a detailed validation of the specified base path,
    expecting 'html', 'css', and 'thumbnails' subdirectories with specific file counts.
    """
    logger.info(f"Route /sftp-test/validate-kb-weekly called for base_path: {base_path}")

    # Define the expected structure and file counts
    expected_structure = {
        "html": {"expected_files": 2, "exists": False, "actual_files": 0, "is_dir": False},
        "css": {"expected_files": 2, "exists": False, "actual_files": 0, "is_dir": False},
        "thumbnails": {"expected_files": 0, "exists": False, "actual_files": 0, "is_dir": False},
    }
    validation_results = {
        "base_path_checked": base_path,
        "base_path_exists": False,
        "base_path_is_dir": False,
        "subdirectories": expected_structure,  # Use a copy or modify in place
        "overall_status": "Error",  # Default to Error
        "details": [],
    }

    service = SFTPService(settings)
    try:
        with service:  # Handles connect/disconnect
            sftp = service._sftp_client  # Get direct access for stat/listdir_attr

            # 1. Check Base Path Existence and Type
            try:
                logger.debug(f"Checking base path: {base_path}")
                base_attrs = sftp.stat(base_path)
                validation_results["base_path_exists"] = True
                # Check if it's a directory using stat module
                if stat.S_ISDIR(base_attrs.st_mode):
                    validation_results["base_path_is_dir"] = True
                    validation_results["details"].append(f"Base path '{base_path}' exists and is a directory.")
                    validation_results["overall_status"] = "OK"  # Tentative OK
                else:
                    validation_results["details"].append(f"Error: Base path '{base_path}' exists but is not a directory.")
                    validation_results["overall_status"] = "Error"
                    # Cannot proceed further if base isn't a directory
                    return validation_results

            except FileNotFoundError:
                logger.warning(f"Base path '{base_path}' not found.")
                validation_results["details"].append(f"Error: Base path '{base_path}' not found.")
                validation_results["overall_status"] = "Error"
                return validation_results  # Cannot proceed further
            except (OSError, IOError) as e:
                logger.error(f"Error checking base path '{base_path}': {e}", exc_info=True)
                validation_results["details"].append(f"Error checking base path '{base_path}': {e}")
                validation_results["overall_status"] = "Error"
                return validation_results  # Cannot proceed further

            # 2. Check Subdirectories and File Counts
            for subdir_name, expected_data in validation_results["subdirectories"].items():
                current_subdir_path = f"{base_path}/{subdir_name}"  # Use forward slash for remote paths
                logger.debug(f"Checking subdirectory: {current_subdir_path}")

                try:
                    # Check if the subdirectory exists and is a directory
                    subdir_attrs = sftp.stat(current_subdir_path)
                    expected_data["exists"] = True
                    if stat.S_ISDIR(subdir_attrs.st_mode):
                        expected_data["is_dir"] = True
                        validation_results["details"].append(f"Subdirectory '{current_subdir_path}' exists and is a directory.")

                        # List contents and count files
                        try:
                            logger.debug(f"Listing attributes for: {current_subdir_path}")
                            # Use listdir_attr to get file types
                            contents_attrs = sftp.listdir_attr(current_subdir_path)
                            file_count = 0
                            for item_attr in contents_attrs:
                                # Check if the item is NOT a directory (i.e., it's a file or link etc.)
                                if not stat.S_ISDIR(item_attr.st_mode):
                                    file_count += 1
                            expected_data["actual_files"] = file_count
                            logger.debug(f"Found {file_count} files in '{current_subdir_path}'.")

                            # Compare counts
                            if file_count == expected_data["expected_files"]:
                                validation_results["details"].append(f"Correct file count ({file_count}) found in '{current_subdir_path}'.")
                            else:
                                validation_results["details"].append(f"Mismatch: Expected {expected_data['expected_files']} files but found {file_count} in '{current_subdir_path}'.")
                                validation_results["overall_status"] = "Mismatch"  # Mark overall as mismatch

                        except (OSError, IOError) as list_e:
                            logger.error(f"Error listing contents of '{current_subdir_path}': {list_e}", exc_info=True)
                            validation_results["details"].append(f"Error listing contents of '{current_subdir_path}': {list_e}")
                            validation_results["overall_status"] = "Error"  # Listing error is more severe

                    else:  # Exists but is not a directory
                        expected_data["is_dir"] = False
                        validation_results["details"].append(f"Error: Expected '{current_subdir_path}' to be a directory, but it's not.")
                        validation_results["overall_status"] = "Error"  # Structure error

                except FileNotFoundError:
                    logger.warning(f"Subdirectory '{current_subdir_path}' not found.")
                    expected_data["exists"] = False
                    validation_results["details"].append(f"Error: Expected subdirectory '{current_subdir_path}' not found.")
                    validation_results["overall_status"] = "Mismatch"  # Missing directory is a mismatch
                except (OSError, IOError) as stat_e:
                    logger.error(f"Error checking subdirectory '{current_subdir_path}': {stat_e}", exc_info=True)
                    validation_results["details"].append(f"Error checking subdirectory '{current_subdir_path}': {stat_e}")
                    validation_results["overall_status"] = "Error"  # Treat stat errors as more severe
            # Final check on status
            if validation_results["overall_status"] == "OK":
                logger.info(f"Validation successful for base path '{base_path}'.")
            else:
                logger.warning(f"Validation found issues for base path '{base_path}'. Status: {validation_results['overall_status']}")

            return validation_results

    except (ConnectionError, AuthenticationException, SSHException) as e:
        logger.error(f"SFTP Connection/Auth error during validation of {base_path}: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"SFTP Connection/Auth error: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error during validation of {base_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected SFTP error during validation: {e}")
