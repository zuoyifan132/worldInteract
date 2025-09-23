import json
import uuid
import datetime
from typing import Dict, Any, List, Optional

def move_file(data: Dict[str, Any], source_path: str, destination_path: str) -> str:
    """
    Move or rename a file.
    
    Args:
        data: In-memory database (nested dictionaries)
        source_path: Current path of the file
        destination_path: New path for the file
        
    Returns:
        JSON string containing operation result
    """
    try:
        # Validate inputs
        if not source_path or not destination_path:
            raise ValueError("Both source_path and destination_path are required")
        
        if source_path == destination_path:
            raise ValueError("Source and destination paths cannot be the same")
        
        # Find the file to move
        file_to_move = None
        file_id = None
        
        for fid, file_record in data.get('files', {}).items():
            if file_record.get('file_path') == source_path:
                file_to_move = file_record
                file_id = fid
                break
        
        if not file_to_move:
            raise FileNotFoundError(f"File not found at path: {source_path}")
        
        # Check if destination already exists
        for file_record in data.get('files', {}).values():
            if file_record.get('file_path') == destination_path:
                raise FileExistsError(f"A file already exists at destination path: {destination_path}")
        
        # Extract destination directory path and filename
        import os
        dest_dir_path = os.path.dirname(destination_path)
        dest_filename = os.path.basename(destination_path)
        
        # Find or validate destination directory
        dest_directory_id = None
        if dest_dir_path:  # If not root directory
            for dir_record in data.get('directories', {}).values():
                if dir_record.get('directory_path') == dest_dir_path:
                    dest_directory_id = dir_record.get('directory_id')
                    break
            
            if dest_directory_id is None:
                raise FileNotFoundError(f"Destination directory not found: {dest_dir_path}")
        
        # Update file record
        import time
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        data['files'][file_id]['file_path'] = destination_path
        data['files'][file_id]['name'] = dest_filename
        data['files'][file_id]['parent_directory_id'] = dest_directory_id
        data['files'][file_id]['modified_at'] = current_time
        
        # Create operation record
        operation_id = f"op_{int(time.time() * 1000)}"
        if 'file_operations' not in data:
            data['file_operations'] = {}
        
        data['file_operations'][operation_id] = {
            'operation_id': operation_id,
            'operation_type': 'move',
            'file_id': file_id,
            'directory_id': dest_directory_id,
            'source_path': source_path,
            'destination_path': destination_path,
            'timestamp': current_time,
            'success': True,
            'message': f"File moved from {source_path} to {destination_path}"
        }
        
        # Update search index if it exists
        if 'search_index' in data:
            for idx_id, index_record in data['search_index'].items():
                if index_record.get('file_id') == file_id:
                    index_record['last_indexed'] = current_time
        
        result = {
            "success": True,
            "message": f"File successfully moved from {source_path} to {destination_path}",
            "after_execution_state": data
        }
        return json.dumps(result)
        
    except Exception as e:
        # Log failed operation
        try:
            import time
            operation_id = f"op_{int(time.time() * 1000)}"
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            
            if 'file_operations' not in data:
                data['file_operations'] = {}
            
            data['file_operations'][operation_id] = {
                'operation_id': operation_id,
                'operation_type': 'move',
                'file_id': None,
                'directory_id': None,
                'source_path': source_path,
                'destination_path': destination_path,
                'timestamp': current_time,
                'success': False,
                'message': f"Move operation failed: {str(e)}"
            }
        except:
            pass  # If logging fails, continue with error response
        
        error_result = {
            "success": False,
            "message": f"Failed to move file: {str(e)}",
            "after_execution_state": data
        }
        return json.dumps(error_result)