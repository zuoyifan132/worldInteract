import json
import uuid
import datetime
from typing import Dict, Any, List, Optional

def delete_file(data: Dict[str, Any], file_path: str) -> str:
    """
    Delete a file from the filesystem.
    
    Args:
        data: In-memory database (nested dictionaries)
        file_path: Path to the file to delete
        
    Returns:
        JSON string containing operation result
    """
    try:
        # Validate inputs
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")
        
        # Normalize file path
        file_path = file_path.strip()
        
        # Find the file to delete
        file_to_delete = None
        file_key = None
        
        for key, file_record in data.get('files', {}).items():
            if file_record.get('file_path') == file_path:
                file_to_delete = file_record
                file_key = key
                break
        
        if not file_to_delete:
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Store file_id for tracking operations and cleaning up related records
        file_id = file_to_delete.get('file_id')
        
        # Delete the file from the files table
        del data['files'][file_key]
        
        # Clean up related records in search_index
        search_keys_to_delete = []
        for idx_key, index_record in data.get('search_index', {}).items():
            if index_record.get('file_id') == file_id:
                search_keys_to_delete.append(idx_key)
        
        for idx_key in search_keys_to_delete:
            del data['search_index'][idx_key]
        
        # Generate unique operation ID
        import time
        import random
        timestamp = str(int(time.time() * 1000))
        random_suffix = str(random.randint(100, 999))
        operation_id = f"op_{timestamp}_{random_suffix}"
        
        # Record the delete operation in file_operations
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Find next available operation key
        existing_op_keys = list(data.get('file_operations', {}).keys())
        if existing_op_keys:
            # Extract numeric parts and find the next number
            max_num = 0
            for key in existing_op_keys:
                if key.startswith('op_'):
                    try:
                        num = int(key.split('_')[1])
                        max_num = max(max_num, num)
                    except (IndexError, ValueError):
                        continue
            next_num = max_num + 1
            op_key = f"op_{next_num:03d}"
        else:
            op_key = "op_001"
        
        operation_record = {
            'operation_id': operation_id,
            'operation_type': 'delete',
            'file_id': file_id,
            'directory_id': file_to_delete.get('parent_directory_id'),
            'source_path': file_path,
            'destination_path': None,
            'timestamp': current_time,
            'success': True,
            'message': f"Successfully deleted file: {file_path}"
        }
        
        # Ensure file_operations table exists
        if 'file_operations' not in data:
            data['file_operations'] = {}
        
        data['file_operations'][op_key] = operation_record
        
        result = {
            "success": True,
            "message": f"File '{file_path}' deleted successfully",
            "after_execution_state": data
        }
        return json.dumps(result)
        
    except FileNotFoundError as e:
        error_result = {
            "success": False,
            "message": str(e),
            "after_execution_state": data
        }
        return json.dumps(error_result)
        
    except Exception as e:
        error_result = {
            "success": False,
            "message": f"Failed to delete file: {str(e)}",
            "after_execution_state": data
        }
        return json.dumps(error_result)