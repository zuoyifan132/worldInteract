import json
import uuid
import datetime
from typing import Dict, Any, List, Optional

"""
Generated tool implementations for domain operations.
"""


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


def search_files(data: Dict[str, Any], search_path: str, pattern: str, search_content: bool = False) -> str:
    """
    Search for files by name pattern or content.
    
    Args:
        data: In-memory database (nested dictionaries)
        search_path: Directory to search in
        pattern: Search pattern (file name or regex)
        search_content: Whether to search file contents (default: False)
        
    Returns:
        JSON string containing search results
    """
    try:
        import re
        import json
        from datetime import datetime
        
        # Validate inputs
        if not search_path or not pattern:
            raise ValueError("search_path and pattern are required")
        
        # Normalize search path
        search_path = search_path.rstrip('/')
        
        # Find the target directory
        target_directory_id = None
        for dir_id, directory in data.get('directories', {}).items():
            if directory.get('directory_path') == search_path:
                target_directory_id = dir_id
                break
        
        if target_directory_id is None:
            raise ValueError(f"Directory not found: {search_path}")
        
        matches = []
        
        # Compile regex pattern for file name matching
        try:
            regex_pattern = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # If pattern is not valid regex, treat as literal string
            regex_pattern = re.compile(re.escape(pattern), re.IGNORECASE)
        
        # Search through files
        for file_id, file_record in data.get('files', {}).items():
            file_path = file_record.get('file_path', '')
            file_name = file_record.get('name', '')
            
            # Check if file is in target directory or subdirectory
            if not file_path.startswith(search_path):
                continue
            
            # Search by file name pattern
            if regex_pattern.search(file_name):
                matches.append({
                    "file_path": file_path,
                    "match_type": "filename",
                    "line_number": 0
                })
            
            # Search file content if requested
            if search_content:
                file_content = file_record.get('content', '')
                if file_content:
                    lines = file_content.split('\n')
                    for line_num, line in enumerate(lines, 1):
                        if regex_pattern.search(line):
                            matches.append({
                                "file_path": file_path,
                                "match_type": "content",
                                "line_number": line_num
                            })
        
        # Generate operation ID and log the search operation
        operation_id = f"op_{int(datetime.now().timestamp() * 1000)}"
        
        # Log the search operation
        if 'file_operations' not in data:
            data['file_operations'] = {}
        
        data['file_operations'][operation_id] = {
            "operation_id": operation_id,
            "operation_type": "search",
            "file_id": None,
            "directory_id": target_directory_id,
            "source_path": search_path,
            "destination_path": None,
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "message": f"Search completed: found {len(matches)} matches"
        }
        
        # Update search index if content search was performed
        if search_content and matches:
            if 'search_index' not in data:
                data['search_index'] = {}
            
            for match in matches:
                if match['match_type'] == 'content':
                    index_id = f"idx_{int(datetime.now().timestamp() * 1000)}_{len(data['search_index'])}"
                    
                    # Find file_id for the match
                    file_id = None
                    for fid, file_record in data.get('files', {}).items():
                        if file_record.get('file_path') == match['file_path']:
                            file_id = fid
                            break
                    
                    data['search_index'][index_id] = {
                        "index_id": index_id,
                        "file_id": file_id,
                        "content_snippet": pattern,
                        "line_number": match['line_number'],
                        "match_type": "search_result",
                        "last_indexed": datetime.now().isoformat()
                    }
        
        result = {
            "success": True,
            "matches": matches,
            "message": f"Search completed successfully. Found {len(matches)} matches.",
            "after_execution_state": data
        }
        return json.dumps(result)
        
    except Exception as e:
        # Log failed operation
        try:
            operation_id = f"op_{int(datetime.now().timestamp() * 1000)}"
            if 'file_operations' not in data:
                data['file_operations'] = {}
            
            data['file_operations'][operation_id] = {
                "operation_id": operation_id,
                "operation_type": "search",
                "file_id": None,
                "directory_id": None,
                "source_path": search_path,
                "destination_path": None,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "message": f"Search failed: {str(e)}"
            }
        except:
            pass
        
        error_result = {
            "success": False,
            "matches": [],
            "message": f"Search operation failed: {str(e)}",
            "after_execution_state": data
        }
        return json.dumps(error_result)
