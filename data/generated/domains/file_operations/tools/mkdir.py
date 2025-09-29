import json
import uuid
import datetime
from typing import Dict, Any, List, Optional, Tuple

def mkdir(data: Dict[str, Any], dir_name: str):
    """Create a new directory in the current directory."""
    import json
    from datetime import datetime
    import uuid
    
    try:
        # Get the current session to determine working directory
        sessions = data.get('sessions', {})
        if not sessions:
            return json.dumps({
                "success": False,
                "error": "No active session found"
            })
        
        # Get the first active session (or we could make this configurable)
        current_session = list(sessions.values())[0]
        current_dir = current_session['current_working_directory']
        
        # Validate directory name
        if not dir_name or not isinstance(dir_name, str):
            return json.dumps({
                "success": False,
                "error": "Directory name must be a non-empty string"
            })
        
        # Check for invalid characters in directory name
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in dir_name for char in invalid_chars):
            return json.dumps({
                "success": False,
                "error": f"Directory name contains invalid characters: {invalid_chars}"
            })
        
        # Check if directory name starts with a dot (hidden directory)
        if dir_name.startswith('.'):
            return json.dumps({
                "success": False,
                "error": "Cannot create hidden directories (names starting with '.')"
            })
        
        # Construct the full path for the new directory
        new_dir_path = f"{current_dir.rstrip('/')}/{dir_name}"
        
        # Check if directory already exists
        directories = data.get('directories', {})
        if new_dir_path in directories:
            return json.dumps({
                "success": False,
                "error": f"Directory '{dir_name}' already exists"
            })
        
        # Check if a file with the same name exists
        files = data.get('files', {})
        conflicting_file_path = f"{current_dir.rstrip('/')}/{dir_name}"
        for file_path in files:
            if file_path == conflicting_file_path:
                return json.dumps({
                    "success": False,
                    "error": f"A file named '{dir_name}' already exists"
                })
        
        # Verify parent directory exists
        if current_dir not in directories:
            return json.dumps({
                "success": False,
                "error": f"Parent directory '{current_dir}' does not exist"
            })
        
        # Create the new directory entry
        current_time = datetime.now().isoformat() + 'Z'
        new_directory = {
            "path": new_dir_path,
            "name": dir_name,
            "parent_path": current_dir,
            "size": 4096,  # Standard directory size
            "created_at": current_time,
            "modified_at": current_time,
            "permissions": "drwxr-xr-x"
        }
        
        # Add the directory to the data structure
        data['directories'][new_dir_path] = new_directory
        
        # Update parent directory's modified time
        data['directories'][current_dir]['modified_at'] = current_time
        
        # Log the operation
        operation_id = f"op_{str(uuid.uuid4())[:8]}"
        operation_log = {
            "operation_id": operation_id,
            "session_id": current_session['session_id'],
            "operation_type": "create_directory",
            "command": f"mkdir {dir_name}",
            "source_path": None,
            "destination_path": new_dir_path,
            "success": True,
            "error_message": None,
            "timestamp": current_time
        }
        
        if 'operations_log' not in data:
            data['operations_log'] = {}
        data['operations_log'][operation_id] = operation_log
        
        # Update session's last command
        current_session['last_command'] = f"mkdir {dir_name}"
        current_session['updated_at'] = current_time
        
        return json.dumps({
            "success": True,
            "message": f"Directory '{dir_name}' created successfully",
            "directory_path": new_dir_path,
            "created_at": current_time
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        })