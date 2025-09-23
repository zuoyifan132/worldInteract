import json
import uuid
import datetime
from typing import Dict, Any, List, Optional

def list_directory(data: Dict[str, Any], directory_path: str, include_hidden: bool = False) -> str:
    """
    List files and subdirectories in a directory.
    
    Args:
        data: In-memory database (nested dictionaries)
        directory_path: Path to the directory to list
        include_hidden: Whether to include hidden files (default: False)
        
    Returns:
        JSON string containing operation result
    """
    try:
        # Validate inputs
        if not directory_path:
            raise ValueError("Directory path cannot be empty")
        
        # Normalize the directory path
        directory_path = directory_path.rstrip('/')
        
        # Find the target directory
        target_directory = None
        for dir_id, directory in data.get('directories', {}).items():
            if directory['directory_path'] == directory_path:
                target_directory = directory
                break
        
        if not target_directory:
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Check if directory is hidden and we're not including hidden items
        if target_directory.get('is_hidden', False) and not include_hidden:
            raise PermissionError(f"Cannot list hidden directory: {directory_path}")
        
        files_list = []
        
        # Find all subdirectories in this directory
        for dir_id, directory in data.get('directories', {}).items():
            if directory.get('parent_directory_id') == target_directory['directory_id']:
                # Skip hidden directories if not requested
                if directory.get('is_hidden', False) and not include_hidden:
                    continue
                
                files_list.append({
                    "name": directory['name'],
                    "type": "directory",
                    "size": 0,  # Directories don't have size
                    "modified": directory.get('modified_at', directory.get('created_at', ''))
                })
        
        # Find all files in this directory
        for file_id, file_record in data.get('files', {}).items():
            if file_record.get('parent_directory_id') == target_directory['directory_id']:
                # Skip hidden files if not requested
                if file_record.get('is_hidden', False) and not include_hidden:
                    continue
                
                files_list.append({
                    "name": file_record['name'],
                    "type": file_record.get('file_type', 'file'),
                    "size": file_record.get('size', 0),
                    "modified": file_record.get('modified_at', file_record.get('created_at', ''))
                })
        
        # Sort files by name for consistent output
        files_list.sort(key=lambda x: x['name'])
        
        # Log the operation
        import time
        operation_id = f"op_{int(time.time())}"
        operation_record = {
            'operation_id': operation_id,
            'operation_type': 'list_directory',
            'file_id': None,
            'directory_id': target_directory['directory_id'],
            'source_path': directory_path,
            'destination_path': None,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'success': True,
            'message': f"Listed {len(files_list)} items in directory"
        }
        
        if 'file_operations' not in data:
            data['file_operations'] = {}
        data['file_operations'][operation_id] = operation_record
        
        result = {
            "success": True,
            "files": files_list,
            "message": f"Successfully listed {len(files_list)} items in directory '{directory_path}'",
            "after_execution_state": data
        }
        
        return json.dumps(result)
        
    except FileNotFoundError as e:
        error_result = {
            "success": False,
            "files": [],
            "message": str(e),
            "after_execution_state": data
        }
        return json.dumps(error_result)
        
    except PermissionError as e:
        error_result = {
            "success": False,
            "files": [],
            "message": str(e),
            "after_execution_state": data
        }
        return json.dumps(error_result)
        
    except Exception as e:
        error_result = {
            "success": False,
            "files": [],
            "message": f"Operation failed: {str(e)}",
            "after_execution_state": data
        }
        return json.dumps(error_result)