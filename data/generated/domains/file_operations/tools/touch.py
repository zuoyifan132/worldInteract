import json
import uuid
import datetime
from typing import Dict, Any, List, Optional, Tuple

def touch(data: Dict[str, Any], file_name: str):
    """Create a new file of any extension in the current directory."""
    import json
    import os
    from datetime import datetime
    import uuid
    
    try:
        # Get current session to determine working directory
        sessions = data.get('sessions', {})
        if not sessions:
            return json.dumps({
                "success": False,
                "error": "No active session found"
            })
        
        # Get the first active session (or we could require session_id parameter)
        current_session = list(sessions.values())[0]
        current_dir = current_session['current_working_directory']
        
        # Validate file_name - should not contain path separators
        if '/' in file_name or '\\' in file_name:
            return json.dumps({
                "success": False,
                "error": "File name cannot contain path separators. Use local file names only."
            })
        
        if not file_name or file_name.strip() == '':
            return json.dumps({
                "success": False,
                "error": "File name cannot be empty"
            })
        
        # Check if current directory exists
        directories = data.get('directories', {})
        if current_dir not in directories:
            return json.dumps({
                "success": False,
                "error": f"Current directory {current_dir} does not exist"
            })
        
        # Construct full file path
        file_path = os.path.join(current_dir, file_name).replace('\\', '/')
        
        # Check if file already exists
        files = data.get('files', {})
        if file_path in files:
            return json.dumps({
                "success": False,
                "error": f"File {file_name} already exists in {current_dir}"
            })
        
        # Extract file extension
        extension = ''
        if '.' in file_name:
            extension = file_name.split('.')[-1]
        
        # Create timestamp
        now = datetime.now().isoformat() + 'Z'
        
        # Create new file entry
        new_file = {
            "path": file_path,
            "name": file_name,
            "directory_path": current_dir,
            "content": "",
            "size": 0,
            "extension": extension,
            "line_count": 0,
            "word_count": 0,
            "character_count": 0,
            "created_at": now,
            "modified_at": now,
            "accessed_at": now,
            "permissions": "-rw-r--r--"
        }
        
        # Add file to data
        if 'files' not in data:
            data['files'] = {}
        data['files'][file_path] = new_file
        
        # Update directory modified time
        data['directories'][current_dir]['modified_at'] = now
        
        # Log the operation
        operation_id = f"op_{str(uuid.uuid4())[:8]}"
        session_id = current_session['session_id']
        
        operation_entry = {
            "operation_id": operation_id,
            "session_id": session_id,
            "operation_type": "create_file",
            "command": f"touch {file_name}",
            "source_path": None,
            "destination_path": file_path,
            "success": True,
            "error_message": None,
            "timestamp": now
        }
        
        if 'operations_log' not in data:
            data['operations_log'] = {}
        data['operations_log'][operation_id] = operation_entry
        
        # Update session last command
        data['sessions'][session_id]['last_command'] = f"touch {file_name}"
        data['sessions'][session_id]['updated_at'] = now
        
        return json.dumps({
            "success": True,
            "message": f"File '{file_name}' created successfully",
            "file_path": file_path,
            "data": {
                "name": file_name,
                "path": file_path,
                "directory": current_dir,
                "size": 0,
                "extension": extension,
                "created_at": now
            }
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        })