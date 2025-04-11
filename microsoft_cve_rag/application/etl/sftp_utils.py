import os
from typing import Dict, List, Union


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
