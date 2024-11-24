import requests
from typing import List, Optional, Union
from urllib.parse import quote
import os


def list_hf_files(
    repo_id: str,
    file_types: Optional[Union[str, List[str]]] = None,
    token: Optional[str] = None,
    branch: str = "main",
    repo_type: str = "dataset"
) -> List[str]:
    """
    List all files in a Hugging Face repository without downloading the dataset.
    
    Args:
        repo_id (str): Repository ID (e.g., "mozilla-foundation/common_voice_11_0")
        file_types (str or List[str], optional): File extension(s) to filter by (e.g., "json" or ["json", "csv"])
                                               Don't include the dot in the extension
        token (str, optional): Hugging Face API token for private repos
        branch (str): Repository branch (default: "main")
        repo_type (str): Type of repository ("dataset", "model", or "space")
    
    Returns:
        List[str]: List of file paths in the repository
        
    Example:
        files = list_huggingface_files("mozilla-foundation/common_voice_11_0", file_types=["json", "csv"])
    """
    # Normalize file_types to a list of lowercase extensions without dots
    if file_types:
        if isinstance(file_types, str):
            file_types = [file_types.lower().strip('.')]
        else:
            file_types = [ft.lower().strip('.') for ft in file_types]
    
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    # Construct API URL
    api_url = f"https://huggingface.co/api/{repo_type}s/{quote(repo_id)}/tree/{branch}"
    
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        
        # Extract file paths from the response
        files = []
        for item in response.json():
            if item["type"] == "file":
                if file_types:
                    ext = os.path.splitext(item["path"])[1].lower().strip('.')
                    if ext in file_types:
                        files.append(item["path"])
                else:
                    files.append(item["path"])
            
        return sorted(files)
        
    except requests.exceptions.RequestException as e:
        if response.status_code == 404:
            raise Exception(f"Repository {repo_id} not found")
        elif response.status_code == 401:
            raise Exception("Authentication required. Please provide a valid token.")
        else:
            raise Exception(f"Error accessing repository: {str(e)}")


def list_huggingface_files_recursive(
    repo_id: str,
    token: Optional[str] = None,
    branch: str = "main",
    repo_type: str = "dataset",
    path: str = ""
) -> List[str]:
    """
    Recursively list all files in a Hugging Face repository, including subdirectories.
    
    Args:
        repo_id (str): Repository ID (e.g., "mozilla-foundation/common_voice_11_0")
        token (str, optional): Hugging Face API token for private repos
        branch (str): Repository branch (default: "main")
        repo_type (str): Type of repository ("dataset", "model", or "space")
        path (str): Current path within repository (used for recursion)
    
    Returns:
        List[str]: List of all file paths in the repository and its subdirectories
    """
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    # Construct API URL for the current path
    api_url = f"https://huggingface.co/api/{repo_type}s/{quote(repo_id)}/tree/{branch}/{path}"
    api_url = api_url.rstrip('/')  # Remove trailing slash
    
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        
        files = []
        for item in response.json():
            item_path = f"{path}/{item['path']}" if path else item['path']
            
            if item["type"] == "file":
                files.append(item_path)
            elif item["type"] == "directory":
                # Recursively get files from subdirectory
                subdir_files = list_huggingface_files_recursive(
                    repo_id, token, branch, repo_type, item_path
                )
                files.extend(subdir_files)
                
        return sorted(files)
        
    except requests.exceptions.RequestException as e:
        if response.status_code == 404:
            raise Exception(f"Repository or path {repo_id}/{path} not found")
        elif response.status_code == 401:
            raise Exception("Authentication required. Please provide a valid token.")
        else:
            raise Exception(f"Error accessing repository: {str(e)}")