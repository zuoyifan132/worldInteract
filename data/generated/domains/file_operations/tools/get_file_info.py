import json
import uuid
import datetime
from typing import Dict, Any, List, Optional

def get_file_info(data: Dict[str, Any], file_path: str) -> str:
    """
    Get detailed information about a file.
    
    Args:
        data: In-memory database (nested dictionaries)
        file_path: Path to the file
        
    Returns:
        JSON string containing file information
    """
    try:
        # Validate inputs
        if not file_path:
            raise ValueError("file_path cannot be empty")
        
        # Ensure required tables exist
        if 'files' not in data:
            data['files'] = {}
        
        # Find the file by file_path
        target_file = None
        for file_key, file_record in data['files'].items():
            if file_record.get('file_path') == file_path:
                target_file = file_record
                break
        
        if not target_file:
            raise FileNotFoundError(f"File not found at path: {file_path}")
        
        # Extract file information
        file_info = {
            "name": target_file.get('name', ''),
            "size": target_file.get('size', 0),
            "created": target_file.get('created_at', ''),
            "modified": target_file.get('modified_at', ''),
            "permissions": target_file.get('permissions', ''),
            "file_type": target_file.get('file_type', '')
        }
        
        result = {
            "success": True,
            "file_info": file_info,
            "message": f"File information retrieved successfully for {file_path}",
            "after_execution_state": data
        }
        return json.dumps(result)
        
    except Exception as e:
        error_result = {
            "success": False,
            "file_info": None,
            "message": f"Failed to get file information: {str(e)}",
            "after_execution_state": data
        }
        return json.dumps(error_result)