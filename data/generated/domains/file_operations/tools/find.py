import json
import uuid
import datetime
from typing import Dict, Any, List, Optional, Tuple

def find(data: Dict[str, Any], path: str = ".", name: str = None):
    """
    Find any file or directories under specific path that contain name in its file name.
    This method searches for files of any extension and directories within a specified path 
    that match the given name. If no name is provided, it returns all files and directories 
    in the specified path and its subdirectories.
    """
    import json
    import os
    from typing import Dict, Any, List
    
    try:
        # Get current working directory from active session if path is relative
        current_dir = "/"
        if "sessions" in data:
            # Find the most recently updated session
            latest_session = None
            latest_time = None
            for session_data in data["sessions"].values():
                if latest_time is None or session_data.get("updated_at", "") > latest_time:
                    latest_time = session_data.get("updated_at", "")
                    latest_session = session_data
            
            if latest_session:
                current_dir = latest_session.get("current_working_directory", "/")
        
        # Resolve the search path
        if path == "." or path == "./":
            search_path = current_dir
        elif path.startswith("./"):
            # Relative path from current directory
            search_path = os.path.normpath(os.path.join(current_dir, path[2:]))
        elif not path.startswith("/"):
            # Relative path
            search_path = os.path.normpath(os.path.join(current_dir, path))
        else:
            # Absolute path
            search_path = os.path.normpath(path)
        
        matches = []
        
        # Helper function to check if a path is under the search path
        def is_under_path(item_path: str, base_path: str) -> bool:
            # Normalize paths for comparison
            item_path = os.path.normpath(item_path)
            base_path = os.path.normpath(base_path)
            
            # Check if item_path starts with base_path
            if item_path == base_path:
                return True
            return item_path.startswith(base_path + "/")
        
        # Helper function to get relative path
        def get_relative_path(item_path: str, base_path: str) -> str:
            item_path = os.path.normpath(item_path)
            base_path = os.path.normpath(base_path)
            
            if item_path == base_path:
                return "."
            
            if item_path.startswith(base_path + "/"):
                return item_path[len(base_path) + 1:]
            
            return item_path
        
        # Helper function to check if name matches
        def name_matches(item_name: str, search_name: str) -> bool:
            if search_name is None:
                return True
            return search_name.lower() in item_name.lower()
        
        # Search through directories
        if "directories" in data:
            for dir_path, dir_info in data["directories"].items():
                if is_under_path(dir_path, search_path):
                    dir_name = dir_info.get("name", os.path.basename(dir_path))
                    if name_matches(dir_name, name):
                        relative_path = get_relative_path(dir_path, search_path)
                        if relative_path not in matches:
                            matches.append(relative_path)
        
        # Search through files
        if "files" in data:
            for file_path, file_info in data["files"].items():
                file_dir = file_info.get("directory_path", os.path.dirname(file_path))
                if is_under_path(file_dir, search_path) or is_under_path(file_path, search_path):
                    file_name = file_info.get("name", os.path.basename(file_path))
                    if name_matches(file_name, name):
                        relative_path = get_relative_path(file_path, search_path)
                        if relative_path not in matches:
                            matches.append(relative_path)
        
        # Sort matches for consistent output
        matches.sort()
        
        return json.dumps({
            "success": True,
            "matches": matches
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Search failed: {str(e)}",
            "matches": []
        })