import json
import uuid
import datetime
from typing import Dict, Any, List, Optional, Tuple

def ls(data: Dict[str, Any], a: bool = False):
    """List the contents of the current directory.
    
    Args:
        data: Database state containing file system information
        a: Show hidden files and directories (defaults to False)
    
    Returns:
        JSON string with current directory contents
    """
    import json
    from typing import Dict, Any, List
    
    try:
        # Get current working directory from active session
        # Find the most recently updated session as the active one
        sessions = data.get('sessions', {})
        if not sessions:
            return json.dumps({
                "success": False,
                "error": "No active session found",
                "current_directory_content": []
            })
        
        # Get the most recently updated session
        active_session = max(sessions.values(), key=lambda s: s['updated_at'])
        current_dir = active_session['current_working_directory']
        
        # Get all files and directories in the current directory
        files = data.get('files', {})
        directories = data.get('directories', {})
        
        contents = []
        
        # Add subdirectories
        for dir_path, dir_info in directories.items():
            if dir_info['parent_path'] == current_dir:
                dir_name = dir_info['name']
                # Check if it's a hidden directory (starts with .)
                if not a and dir_name.startswith('.'):
                    continue
                contents.append(dir_name)
        
        # Add files
        for file_path, file_info in files.items():
            if file_info['directory_path'] == current_dir:
                file_name = file_info['name']
                # Check if it's a hidden file (starts with .)
                if not a and file_name.startswith('.'):
                    continue
                contents.append(file_name)
        
        # Sort contents alphabetically
        contents.sort()
        
        return json.dumps({
            "success": True,
            "current_directory_content": contents
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Error listing directory contents: {str(e)}",
            "current_directory_content": []
        })