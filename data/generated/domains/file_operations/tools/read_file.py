import json
import uuid
import datetime
from typing import Dict, Any, List, Optional

def read_file(data: Dict[str, Any], file_path: str, encoding: str = "utf-8") -> str:
    """
    Read content from an existing file.
    
    Args:
        data: In-memory database (nested dictionaries)
        file_path: Path to the file to read
        encoding: File encoding (default: utf-8)
        
    Returns:
        JSON string containing file content and metadata
    """
    try:
        # Validate inputs
        if not file_path:
            raise ValueError("File path cannot be empty")
        
        if not encoding:
            encoding = "utf-8"
        
        # Normalize file path
        file_path = file_path.strip().replace("\\", "/")
        
        # Find the file by path
        target_file = None
        file_key = None
        
        if "files" not in data:
            raise FileNotFoundError(f"File system not initialized")
        
        for key, file_record in data["files"].items():
            if file_record.get("file_path") == file_path:
                target_file = file_record
                file_key = key
                break
        
        if not target_file:
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check if file has read permissions
        file_permissions = target_file.get("permissions", "rw")
        if "r" not in file_permissions:
            raise PermissionError(f"No read permission for file: {file_path}")
        
        # Get file content and metadata
        file_content = target_file.get("content", "")
        file_size = target_file.get("size", len(file_content))
        file_encoding = target_file.get("encoding", "utf-8")
        
        # Validate encoding compatibility
        if encoding != file_encoding and file_encoding != "utf-8":
            import warnings
            warnings.warn(f"Requested encoding '{encoding}' differs from file encoding '{file_encoding}'")
        
        # Log the read operation
        if "file_operations" not in data:
            data["file_operations"] = {}
        
        import time
        operation_id = f"op_{int(time.time() * 1000000) % 1000000:06d}"
        
        operation_record = {
            "operation_id": operation_id,
            "operation_type": "read",
            "file_id": target_file.get("file_id"),
            "directory_id": target_file.get("parent_directory_id"),
            "source_path": file_path,
            "destination_path": None,
            "timestamp": time.time(),
            "success": True,
            "message": f"Successfully read file: {file_path}"
        }
        
        data["file_operations"][operation_id] = operation_record
        
        result = {
            "success": True,
            "content": file_content,
            "file_size": file_size,
            "message": f"Successfully read file: {file_path}",
            "after_execution_state": data
        }
        return json.dumps(result)
        
    except FileNotFoundError as e:
        error_result = {
            "success": False,
            "content": "",
            "file_size": 0,
            "message": str(e),
            "after_execution_state": data
        }
        return json.dumps(error_result)
        
    except PermissionError as e:
        error_result = {
            "success": False,
            "content": "",
            "file_size": 0,
            "message": str(e),
            "after_execution_state": data
        }
        return json.dumps(error_result)
        
    except Exception as e:
        error_result = {
            "success": False,
            "content": "",
            "file_size": 0,
            "message": f"Failed to read file: {str(e)}",
            "after_execution_state": data
        }
        return json.dumps(error_result)