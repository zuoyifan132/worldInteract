import json
import uuid
import datetime
from typing import Dict, Any, List, Optional

def create_file(data: Dict[str, Any], file_path: str, content: str, encoding: str = "utf-8") -> str:
    """
    Create a new file with specified content.
    
    Args:
        data: In-memory database (nested dictionaries)
        file_path: Path where the file should be created
        content: Content to write to the file
        encoding: File encoding (default: utf-8)
        
    Returns:
        JSON string containing operation result
    """
    try:
        # Validate inputs
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")
        
        if content is None:
            content = ""
        
        if not encoding:
            encoding = "utf-8"
        
        # Normalize file path
        file_path = file_path.replace('\\', '/')
        if not file_path.startswith('/'):
            file_path = '/' + file_path
        
        # Check if file already exists
        for file_id, file_record in data.get('files', {}).items():
            if file_record.get('file_path') == file_path:
                raise ValueError(f"File already exists at path: {file_path}")
        
        # Extract directory path and file name
        path_parts = file_path.rsplit('/', 1)
        if len(path_parts) == 2:
            directory_path = path_parts[0] if path_parts[0] else '/'
            file_name = path_parts[1]
        else:
            directory_path = '/'
            file_name = file_path.lstrip('/')
        
        if not file_name:
            raise ValueError("Invalid file path: no file name specified")
        
        # Find or create parent directory
        parent_directory_id = None
        for dir_id, dir_record in data.get('directories', {}).items():
            if dir_record.get('directory_path') == directory_path:
                parent_directory_id = dir_id
                break
        
        # If parent directory doesn't exist, create it
        if parent_directory_id is None:
            import time
            import random
            dir_counter = len(data.get('directories', {})) + 1
            parent_directory_id = f"dir_{dir_counter:03d}"
            
            # Ensure directories table exists
            if 'directories' not in data:
                data['directories'] = {}
            
            data['directories'][parent_directory_id] = {
                'directory_id': parent_directory_id,
                'directory_path': directory_path,
                'name': directory_path.split('/')[-1] if directory_path != '/' else 'root',
                'parent_directory_id': None,
                'is_hidden': False,
                'created_at': time.time(),
                'modified_at': time.time(),
                'permissions': 'rwxr-xr-x'
            }
        
        # Generate unique file ID
        import time
        import random
        file_counter = len(data.get('files', {})) + 1
        file_id = f"file_{file_counter:03d}"
        
        # Determine file type based on extension
        file_extension = file_name.split('.')[-1].lower() if '.' in file_name else ''
        file_type_map = {
            'txt': 'text',
            'py': 'python',
            'js': 'javascript',
            'html': 'html',
            'css': 'css',
            'json': 'json',
            'xml': 'xml',
            'md': 'markdown',
            'yml': 'yaml',
            'yaml': 'yaml'
        }
        file_type = file_type_map.get(file_extension, 'unknown')
        
        # Calculate file size (approximate bytes for content)
        content_bytes = content.encode(encoding) if content else b''
        file_size = len(content_bytes)
        
        # Ensure files table exists
        if 'files' not in data:
            data['files'] = {}
        
        # Create file record
        current_time = time.time()
        data['files'][file_id] = {
            'file_id': file_id,
            'file_path': file_path,
            'name': file_name,
            'parent_directory_id': parent_directory_id,
            'content': content,
            'encoding': encoding,
            'size': file_size,
            'file_type': file_type,
            'is_hidden': file_name.startswith('.'),
            'created_at': current_time,
            'modified_at': current_time,
            'permissions': 'rw-r--r--'
        }
        
        # Log the operation
        if 'file_operations' not in data:
            data['file_operations'] = {}
        
        op_counter = len(data['file_operations']) + 1
        operation_id = f"op_{op_counter:03d}"
        
        data['file_operations'][operation_id] = {
            'operation_id': operation_id,
            'operation_type': 'CREATE_FILE',
            'file_id': file_id,
            'directory_id': parent_directory_id,
            'source_path': None,
            'destination_path': file_path,
            'timestamp': current_time,
            'success': True,
            'message': f"File created successfully at {file_path}"
        }
        
        # Update search index if content is provided
        if content and content.strip():
            if 'search_index' not in data:
                data['search_index'] = {}
            
            # Create basic search index entries for content
            content_lines = content.split('\n')
            for line_num, line in enumerate(content_lines, 1):
                if line.strip():  # Only index non-empty lines
                    idx_counter = len(data['search_index']) + 1
                    index_id = f"idx_{idx_counter:03d}"
                    
                    data['search_index'][index_id] = {
                        'index_id': index_id,
                        'file_id': file_id,
                        'content_snippet': line[:100],  # Limit snippet length
                        'line_number': line_num,
                        'match_type': 'content',
                        'last_indexed': current_time
                    }
        
        result = {
            "success": True,
            "file_id": file_id,
            "message": f"File created successfully at {file_path}",
            "after_execution_state": data
        }
        return json.dumps(result)
        
    except Exception as e:
        # Log failed operation
        if 'file_operations' in data:
            import time
            op_counter = len(data['file_operations']) + 1
            operation_id = f"op_{op_counter:03d}"
            
            data['file_operations'][operation_id] = {
                'operation_id': operation_id,
                'operation_type': 'CREATE_FILE',
                'file_id': None,
                'directory_id': None,
                'source_path': None,
                'destination_path': file_path,
                'timestamp': time.time(),
                'success': False,
                'message': f"Failed to create file: {str(e)}"
            }
        
        error_result = {
            "success": False,
            "file_id": None,
            "message": f"Failed to create file: {str(e)}",
            "after_execution_state": data
        }
        return json.dumps(error_result)