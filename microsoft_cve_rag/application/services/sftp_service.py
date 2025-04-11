import logging
import os
import errno
from pathlib import Path
from typing import List, Optional, Type, Dict
from paramiko import SSHClient, AutoAddPolicy, RSAKey
from paramiko.sftp_client import SFTPClient
from paramiko.ssh_exception import (
    AuthenticationException,
    NoValidConnectionsError,
    SSHException,
)
from pydantic import Field, FilePath, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from fastapi import HTTPException

# --- Configuration ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SFTPSettings(BaseSettings):
    """
    Holds SFTP connection settings, loaded from environment variables
    or a .env file.
    """
    hostname: str = Field(..., validation_alias='SITEGROUND_SSH_HOSTNAME')
    port: int = Field(22, validation_alias='SITEGROUND_SSH_PORT')  # Default SFTP/SSH port
    username: str = Field(..., validation_alias='SITEGROUND_SSH_USERNAME')
    # Use SecretStr to prevent accidental logging of password
    password: Optional[SecretStr] = Field(None, validation_alias='SITEGROUND_SSH_PASSWORD')
    key_path: FilePath = Field(..., validation_alias='SITEGROUND_SSH_KEY_PATH')
    # key_name: Optional[str] = Field(None, validation_alias='SITEGROUND_SSH_KEY_NAME') # Optional, not directly used by paramiko connect

    # Pydantic V2 configuration
    model_config = SettingsConfigDict(
        env_file='.env',  # Load from .env file if present
        extra='ignore',   # Ignore extra fields found in the environment
        # You can add a prefix if all env vars start with something specific,
        # but alias works well here.
        # env_prefix='SITEGROUND_SSH_'
    )

    @property
    def display_password(self) -> str:
        """Safely display whether a password is set."""
        return "****" if self.password else "Not Set"

# --- Service Class ---


class SFTPService:
    """
    A service class to interact with an SFTP server for uploading files.

    Provides context management for handling connections.
    """

    def __init__(self, settings: SFTPSettings):
        self._settings = settings
        self._ssh_client: Optional[SSHClient] = None
        self._sftp_client: Optional[SFTPClient] = None
        logger.info(f"SFTPService initialized for {settings.username}@{settings.hostname}:{settings.port}")

    def _connect(self) -> None:
        """Establishes SSH and SFTP connections."""
        if self._ssh_client and self._ssh_client.get_transport() and self._ssh_client.get_transport().is_active():
            logger.debug("SSH connection already active.")
            if not self._sftp_client or self._sftp_client.sock.closed:
                logger.debug("SFTP client needs reopening.")
                try:
                    self._sftp_client = self._ssh_client.open_sftp()
                    logger.info("SFTP session reopened successfully.")
                except SSHException as e:
                    logger.error(f"Failed to reopen SFTP session: {e}")
                    raise ConnectionError(f"Failed to reopen SFTP session: {e}") from e
            return  # Already connected

        logger.debug(f"Attempting to connect to {self._settings.hostname}:{self._settings.port}...")
        self._ssh_client = SSHClient()
        self._ssh_client.set_missing_host_key_policy(AutoAddPolicy())

        try:
            key_password = self._settings.password.get_secret_value() if self._settings.password else None
            private_key = RSAKey.from_private_key_file(
                str(self._settings.key_path),
                password=key_password
            )
            logger.debug(f"Private key loaded from {self._settings.key_path}")

            self._ssh_client.connect(
                hostname=self._settings.hostname,
                port=self._settings.port,
                username=self._settings.username,
                pkey=private_key,
                # Add password fallback if needed, though key auth is preferred
                # password=key_password, # Often key password and login password differ
                timeout=10  # Add a connection timeout
            )
            logger.info("SSH connection established successfully.")

            self._sftp_client = self._ssh_client.open_sftp()
            logger.info("SFTP session opened successfully.")

        except FileNotFoundError:
            logger.error(f"Private key file not found at {self._settings.key_path}")
            raise ConnectionError(f"Private key file not found: {self._settings.key_path}")
        except AuthenticationException as e:
            logger.error(f"Authentication failed for user {self._settings.username}: {e}. Check username, key, and key password.")
            # Avoid logging password detail directly from settings
            logger.debug(f"Attempted with key: {self._settings.key_path}, password provided: {bool(self._settings.password)}")
            raise ConnectionError(f"Authentication failed: {e}") from e
        except NoValidConnectionsError as e:
            logger.error(f"Failed to connect to host {self._settings.hostname} on port {self._settings.port}: {e}")
            raise ConnectionError(f"Failed to connect to {self._settings.hostname}:{self._settings.port}") from e
        except SSHException as e:
            logger.error(f"SSH error during connection: {e}")
            # Could be key format issue, permissions, etc.
            if "Error reading SSH protocol banner" in str(e):
                logger.error("Could be a non-SSH service running on the port or network issue.")
            elif "Private key file is encrypted" in str(e) and not key_password:
                logger.error("Private key is encrypted, but no password was provided.")
            raise ConnectionError(f"SSH connection error: {e}") from e
        except Exception as e:
            logger.exception(f"An unexpected error occurred during connection: {e}")
            raise ConnectionError(f"Unexpected connection error: {e}") from e

    def _close(self) -> None:
        """Closes SFTP and SSH connections."""
        if self._sftp_client:
            try:
                self._sftp_client.close()
                logger.info("SFTP session closed.")
                self._sftp_client = None
            except Exception as e:
                logger.warning(f"Error closing SFTP session: {e}", exc_info=True)
        if self._ssh_client:
            try:
                self._ssh_client.close()
                logger.info("SSH connection closed.")
                self._ssh_client = None
            except Exception as e:
                logger.warning(f"Error closing SSH connection: {e}", exc_info=True)

    def __enter__(self) -> 'SFTPService':
        """Enter the runtime context related to this object."""
        self._connect()
        try:
            # base_dir_to_set = '/'
            # logger.info(f"Setting initial working directory to: {base_dir_to_set}")
            # self.change_directory(base_dir_to_set)
            return self
        except Exception as e:
            logger.error(f"Failed to enter SFTPService context: {e}")
            raise

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[object]) -> None:
        """Exit the runtime context related to this object."""
        self._close()
        # Optionally re-raise exception if needed, or handle specific ones
        if exc_value:
            logger.error(f"SFTPService context exited with error: {exc_value}", exc_info=(exc_type, exc_value, traceback))
            # return False # To propagate exception (default)
            # return True # To suppress exception

    @property
    def is_connected(self) -> bool:
        """Check if the SFTP client is connected and active."""
        return (self._ssh_client is not None and
                self._ssh_client.get_transport() is not None and
                self._ssh_client.get_transport().is_active() and
                self._sftp_client is not None and
                not self._sftp_client.sock.closed)

    def _ensure_connection(self) -> None:
        """Ensure the SFTP client is ready for use."""
        if not self.is_connected:
            logger.error("SFTP client is not connected. Use within a 'with' block or call connect() explicitly.")
            raise ConnectionError("SFTP client is not connected.")

    def _ensure_remote_dir_exists(self, remote_directory: str) -> bool:
        """
        Recursively creates the directory structure on the remote server.
        Handles both relative and absolute paths.
        Returns True if directory exists or was created, False otherwise.
        """
        self._ensure_connection()
        sftp = self._sftp_client  # Type hint helper

        if not remote_directory or remote_directory == '.':
            logger.debug("No remote directory creation needed for current/empty path.")
            return True

        # Normalize path separators for remote system (usually '/')
        remote_directory = remote_directory.replace('\\', '/')

        # Handle absolute paths vs relative paths
        if remote_directory.startswith('/'):
            current_parts = ['']  # Start from root
        else:
            current_parts = []  # Start from current remote directory (usually home)

        for dir_fragment in remote_directory.split('/'):
            if not dir_fragment:  # Handles leading/trailing/multiple slashes
                continue

            current_parts.append(dir_fragment)
            current_path = '/'.join(current_parts)
            # For absolute paths, the first join might look like '/dir', which is correct.
            # For relative paths, it will be 'dir1', then 'dir1/dir2'.

            try:
                sftp.stat(current_path)
                logger.debug(f"Remote directory fragment exists: {current_path}")
            except IOError as e:
                # errno.ENOENT typically means "No such file or directory"
                if getattr(e, 'errno', None) == errno.ENOENT or isinstance(e, FileNotFoundError):
                    logger.info(f"Remote directory fragment not found, creating: {current_path}")
                    try:
                        sftp.mkdir(current_path)
                        logger.info(f"Successfully created remote directory: {current_path}")
                    except IOError as mkdir_e:
                        logger.error(f"Failed to create remote directory {current_path}: {mkdir_e}")
                        # Check if it's a permission error or something else
                        if getattr(mkdir_e, 'errno', None) == errno.EACCES:
                            logger.error("Permission denied to create directory.")
                            return False
                else:
                    # Handle other I/O errors (e.g., permission denied on stat)
                    logger.error(f"Error checking remote directory {current_path}: {e}")
                    return False
            except Exception as e:
                logger.error(f"Unexpected error checking remote directory {current_path}: {e}", exc_info=True)
                return False
        return True

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        """
        Uploads a single file to the specified remote path.

        Ensures the remote directory structure exists before uploading.

        :param local_path: Path object pointing to the local file.
        :param remote_path: String representing the full remote file path.
        :raises FileNotFoundError: If the local file does not exist.
        :raises ConnectionError: If not connected to the SFTP server.
        :raises OSError: If there's an error during upload or directory creation.
        """
        self._ensure_connection()
        sftp = self._sftp_client

        if not local_path.is_file():
            logger.error(f"Local file not found: {local_path}")
            raise FileNotFoundError(f"Local file not found: {local_path}")

        # Normalize remote path
        remote_path = remote_path.replace('\\', '/')
        remote_dir = os.path.dirname(remote_path)  # Use os.path for remote path manipulation

        logger.info(f"Preparing to upload {local_path} to {remote_path}")

        # Ensure the remote directory exists
        if not self._ensure_remote_dir_exists(remote_dir):
            logger.error(f"Failed to ensure remote directory exists: {remote_dir}. Aborting upload.")
            raise OSError(f"Could not create or access remote directory: {remote_dir}")

        # Perform the upload
        try:
            logger.debug(f"Putting file: {str(local_path)} -> {remote_path}")
            sftp.put(str(local_path), remote_path)
            logger.info(f"Successfully uploaded: {local_path} to {remote_path}")
        except (IOError, OSError, SSHException) as e:
            logger.error(f"Failed to upload {local_path} to {remote_path}: {e}")
            # Check for specific SFTP error codes if available (depends on server/paramiko)
            if getattr(e, 'errno', None) == errno.EACCES or 'permission denied' in str(e).lower():
                logger.error("Permission denied during upload.")
            raise OSError(f"Failed to upload file: {e}") from e
        except Exception as e:
            logger.exception(f"An unexpected error occurred during upload: {e}")
            raise

    def upload_files(self, file_mappings: dict[Path, str]) -> None:
        """
        Uploads multiple files specified in a dictionary.

        Keys are local Path objects, values are remote path strings.
        Uses upload_file for each item, logging successes/failures individually.
        """
        self._ensure_connection()
        logger.info(f"Starting batch upload of {len(file_mappings)} files.")
        success_count = 0
        fail_count = 0
        for local_path, remote_path in file_mappings.items():
            try:
                self.upload_file(local_path, remote_path)
                success_count += 1
            except (FileNotFoundError, ConnectionError, OSError) as e:
                logger.error(f"Failed to upload {local_path} to {remote_path}: {e}")
                fail_count += 1
            except Exception as e:  # Catch any unexpected errors from upload_file
                logger.exception(f"Unexpected error uploading {local_path} to {remote_path}: {e}")
                fail_count += 1

        logger.info(f"Batch upload complete. Success: {success_count}, Failures: {fail_count}")
        if fail_count > 0:
            # Consider raising an exception here if any failure is critical
            logger.warning("Some files failed to upload during batch operation.")
            # raise RuntimeError(f"{fail_count} files failed to upload.")

    # --- Optional helper methods from original class, adapted ---

    def list_directory(self, remote_path: str = ".") -> List[str]:
        """Lists files and directories in the specified remote directory."""
        self._ensure_connection()
        try:
            logger.debug(f"Listing directory: {remote_path}")
            return self._sftp_client.listdir(path=remote_path)
        except (IOError, OSError, SSHException) as e:
            logger.error(f"Failed to list directory {remote_path}: {e}")
            raise OSError(f"Failed to list directory {remote_path}: {e}") from e

    def change_directory(self, remote_path: str) -> None:
        """Changes the current working directory on the remote server."""
        self._ensure_connection()
        try:
            logger.debug(f"Changing remote directory to: {remote_path}")
            self._sftp_client.chdir(remote_path)
            logger.info(f"Remote directory changed to: {self._sftp_client.getcwd()}")
        except (IOError, OSError, SSHException) as e:
            logger.error(f"Failed to change remote directory to {remote_path}: {e}")
            raise OSError(f"Failed to change directory to {remote_path}: {e}") from e

    def get_current_directory(self) -> str:
        """Returns the current working directory on the remote server."""
        self._ensure_connection()
        try:
            cwd = self._sftp_client.getcwd()
            logger.debug(f"Current remote directory: {cwd}")
            return cwd
        except (IOError, OSError, SSHException) as e:
            logger.error(f"Failed to get current remote directory: {e}")
            raise OSError(f"Failed to get current remote directory: {e}") from e

# ==========================================================================
# --- Example Usage ---
# ==========================================================================


if __name__ == "__main__":
    # Create dummy files and .env for demonstration
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("paramiko").setLevel(logging.DEBUG)
    local_base_path = Path("data") / "reports" / "weekly_kb_report"
    report_html_path = local_base_path / "html" / "kb_report_20240914.html"
    report_thumbnail_path = local_base_path / "thumbnails" / "thumbnail1.jpeg"
    remote_base_dir = "www/portalfuse.io/public_html/wp-content/uploads/kb_weekly"

    try:

        # 1. Load settings (automatically reads from .env)
        try:
            sftp_settings = SFTPSettings()
            print("SFTP Settings Loaded:")
            print(f"  Hostname: {sftp_settings.hostname}")
            print(f"  Port: {sftp_settings.port}")
            print(f"  Username: {sftp_settings.username}")
            print(f"  Key Path: {sftp_settings.key_path}")
            # print(f"  Password Set: {sftp_settings.display_password}") # Be careful logging this
        except ValidationError as e:
            print(f"Error loading settings: {e}")
            exit(1)

        # 2. Use the SFTPService within a 'with' block for connection management
        print("\nAttempting to connect and upload (will likely fail with dummy credentials)...")
        try:
            service = SFTPService(sftp_settings)
            with service:  # Connects on entry, closes on exit
                # Ensure service is actually connected (optional check)
                if service.is_connected:

                    print(f"Connected! Current remote directory: {service.get_current_directory()}")
                    print(f"list directories: {service.list_directory()}")
                    service.change_directory('/')
                    print("Successfully changed directory to '/'")
                    cwd_after_cd = service.get_current_directory()
                    print(f"CWD after changing to '/': {cwd_after_cd}")
                    # Define remote paths relative to user's home or specify absolute path
                    remote_html_path = f"{remote_base_dir}/html/kb_report_20240914.html"
                    remote_png_path = f"{remote_base_dir}/thumbnails/thumbnail1.jpeg"

                    # Upload a single file
                    print(f"\nUploading {report_html_path} to {remote_html_path}...")
                    service.upload_file(report_html_path, remote_html_path)

                    # Upload another file to a subdirectory
                    print(f"\nUploading {report_thumbnail_path} to {remote_png_path}...")
                    service.upload_file(report_thumbnail_path, remote_png_path)

                    # Example: List the created report directory
                    print(f"\nListing contents of {remote_base_dir}:")
                    try:
                        contents = service.list_directory(remote_base_dir)
                        print(contents)
                    except OSError as e:
                        print(f"Could not list directory (may not exist or permissions issue): {e}")
                    print("test complete")
                    # Example: Upload multiple files using the batch method
                    # print("\nUploading multiple files using batch method...")
                    # file_map = {
                    #     report_html_path: f"{remote_base_dir}/copy_index.html",
                    #     report_png_path: f"{remote_base_dir}/assets/copy_chart.png"
                    # }
                    # service.upload_files(file_map)

                else:
                    print("Service failed to establish a connection within the 'with' block.")

        except ConnectionError as e:
            print(f"\nConnection failed as expected with dummy details: {e}")
        except (FileNotFoundError, OSError) as e:
            print(f"\nFile or OS error during operation: {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            logger.exception("Traceback for unexpected error:")

    finally:
        # Clean up dummy files
        print("Cleanup complete.")

# Begin utility functions ============================


async def get_sftp_settings() -> SFTPSettings:
    try:
        return SFTPSettings()
    except ValidationError as e:
        logger.error(f"SFTP settings validation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"SFTP configuration error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading SFTP settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load SFTP settings")


def build_file_mappings(
    local_paths: List[str],
    remote_base_path: str
) -> Dict[str, str]:
    """Build mappings between local files and their remote SFTP destinations.

    This function creates a mapping dictionary that defines how local report files
    should be organized when uploaded to the remote SFTP server. It's designed to:

    1. Maintain a consistent directory structure for report assets
    2. Ensure proper organization of different file types (HTML, plots, thumbnails)
    3. Handle path normalization across different operating systems
    4. Prevent path-related issues by converting Windows backslashes to forward slashes

    The function expects local files to be organized in specific subdirectories
    (html, plots, thumbnails) and maintains this structure on the remote server.
    This organization is crucial for:
    - Keeping report assets properly grouped
    - Maintaining correct relative paths in HTML reports
    - Enabling proper asset loading in web browsers

    Args:
        local_paths: List of absolute paths to local files that need to be uploaded
        remote_base_path: Base directory path on the remote SFTP server where
            files will be uploaded

    Returns:
        Dictionary mapping local file paths to their corresponding remote paths.
        All remote paths use forward slashes for cross-platform compatibility.

    Example:
        >>> local_paths = ['/path/to/reports/html/report.html',
                          '/path/to/reports/plots/graph.png']
        >>> remote_base = '/public_html/reports'
        >>> mappings = build_file_mappings(local_paths, remote_base)
        >>> print(mappings)
        {
            '/path/to/reports/html/report.html': '/public_html/reports/html/report.html',
            '/path/to/reports/plots/graph.png': '/public_html/reports/plots/graph.png'
        }
    """
    file_mappings = {}
    for local_path in local_paths:
        # Normalize and extract parts
        path_parts = os.path.normpath(local_path).split(os.sep)
        file_name = path_parts[-1]
        sub_folder = path_parts[-2]  # Assuming the subfolder is right before the file name in the path

        if sub_folder.lower() in ['html', 'plots', 'thumbnails']:  # Expected subfolders
            remote_path = os.path.join(remote_base_path, sub_folder, file_name)
        else:
            # Default to putting it directly under base if unexpected directory
            remote_path = os.path.join(remote_base_path, file_name)

        file_mappings[local_path] = remote_path.replace('\\', '/')

    return file_mappings

# End utility functions ============================
