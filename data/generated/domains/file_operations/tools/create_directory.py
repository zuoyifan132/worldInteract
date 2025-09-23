import json
import uuid
import datetime
from typing import Dict, Any, List, Optional

def create_directory(data: Dict[str, Any], directory_path: str, recursive: bool = True) -> str:
    """
    Create a new directory in the file system database.
    
    Args:
        data: In-memory database (nested dictionaries)
        directory_path: Path where the directory should be created
        recursive: Create parent directories if they don't exist (default: True)
        
    Returns:
        JSON string containing operation result
    """
    try:
        # Validate inputs
        if not directory_path or not isinstance(directory_path, str):
            raise ValueError("Directory path must be a non-empty string")
        
        # Normalize path (remove trailing slashes)
        directory_path = directory_path.rstrip('/')
        if not directory_path:
            directory_path = '/'
            
        # Check if directory already exists
        for dir_id, dir_info in data.get('directories', {}).items():
            if dir_info.get('directory_path') == directory_path:
                raise ValueError(f"Directory already exists: {directory_path}")
        
        # Parse path components
        path_parts = [part for part in directory_path.split('/') if part]
        if not path_parts:
            # Root directory case
            directory_name = 'root'
            parent_directory_id = None
        else:
            directory_name = path_parts[-1]
            parent_path = '/' + '/'.join(path_parts[:-1]) if len(path_parts) > 1 else '/'
            if parent_path == '/':
                parent_path = '/'
            
            # Find parent directory
            parent_directory_id = None
            for dir_id, dir_info in data.get('directories', {}).items():
                if dir_info.get('directory_path') == parent_path:
                    parent_directory_id = dir_id
                    break
            
            # If parent doesn't exist and recursive is True, create parent directories
            if parent_directory_id is None and recursive and parent_path != '/':
                parent_result = create_directory(data, parent_path, recursive)
                parent_data = json.loads(parent_result)
                if not parent_data.get('success'):
                    raise ValueError(f"Failed to create parent directory: {parent_path}")
                parent_directory_id = parent_data.get('directory_id')
            elif parent_directory_id is None and not recursive:
                raise ValueError(f"Parent directory does not exist: {parent_path}")
        
        # Generate unique directory ID
        import time
        import random
        timestamp = str(int(time.time() * 1000))
        random_suffix = str(random.randint(100, 999))
        directory_id = f"dir_{timestamp}_{random_suffix}"
        
        # Ensure directories table exists
        if 'directories' not in data:
            data['directories'] = {}
        
        # Create directory record
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        data['directories'][directory_id] = {
            'directory_id': directory_id,
            'directory_path': directory_path,
            'name': directory_name,
            'parent_directory_id': parent_directory_id,
            'is_hidden': False,
            'created_at': current_time,
            'modified_at': current_time,
            'permissions': '755'
        }
        
        # Log the operation
        if 'file_operations' not in data:
            data['file_operations'] = {}
        
        operation_id = f"op_{timestamp}_{random_suffix}"
        data['file_operations'][operation_id] = {
            'operation_id': operation_id,
            'operation_type': 'create_directory',
            'file_id': None,
            'directory_id': directory_id,
            'source_path': None,
            'destination_path': directory_path,
            'timestamp': current_time,
            'success': True,
            'message': f"Directory created successfully: {directory_path}"
        }
        
        result = {
            "success": True,
            "directory_id": directory_id,
            "message": f"Directory created successfully: {directory_path}",
            "after_execution_state": data
        }
        return json.dumps(result)
        
    except Exception as e:
        # Log failed operation
        try:
            import time
            import random
            timestamp = str(int(time.time() * 1000))
            random_suffix = str(random.randint(100, 999))
            
            if 'file_operations' not in data:
                data['file_operations'] = {}
            
            operation_id = f"op_{timestamp}_{random_suffix}"
            current_time = time.strftime('%Y-%m-%d %H:%M:%S')
            data['file_operations'][operation_id] = {
                'operation_id': operation_id,
                'operation_type': 'create_directory',
                'file_id': None,
                'directory_id': None,
                'source_path': None,
                'destination_path': directory_path,
                'timestamp': current_time,
                'success': False,
                'message': f"Failed to create directory: {str(e)}"
            }
        except:
            pass  # Don't fail if logging fails
        
        error_result = {
            "success": False,
            "directory_id": None,
            "message": f"Failed to create directory: {str(e)}",
            "after_execution_state": data
        }
        return json.dumps(error_result)