import json
import uuid
import datetime
from typing import Dict, Any, List, Optional, Tuple

def echo(data: Dict[str, Any], content: str, file_name: str = None):
    """
    Write content to a file at current directory or display it in the terminal.
    
    Args:
        data: Database state containing file system information
        content: The content to write or display
        file_name: The name of the file to write to (optional)
    
    Returns:
        JSON string with terminal_output
    """
    import json
    import os
    from datetime import datetime
    
    try:
        # If no file_name provided, just return the content to terminal
        if file_name is None:
            return json.dumps({
                "success": True,
                "terminal_output": content
            })
        
        # Get current working directory from active session
        current_session = None
        for session_id, session_data in data.get("sessions", {}).items():
            current_session = session_data
            break  # Use the first available session
        
        if not current_session:
            return json.dumps({
                "success": False,
                "terminal_output": None,
                "error": "No active session found"
            })
        
        current_dir = current_session["current_working_directory"]
        file_path = os.path.join(current_dir, file_name).replace("\\", "/")
        
        # Check if directory exists
        if current_dir not in data.get("directories", {}):
            return json.dumps({
                "success": False,
                "terminal_output": None,
                "error": f"Directory {current_dir} does not exist"
            })
        
        # Calculate file statistics
        line_count = content.count('\n') + (1 if content and not content.endswith('\n') else 0)
        word_count = len(content.split()) if content else 0
        character_count = len(content)
        size = len(content.encode('utf-8'))
        
        # Get file extension
        extension = ""
        if "." in file_name:
            extension = file_name.split(".")[-1]
        
        current_time = datetime.now().isoformat() + "Z"
        
        # Create or update file in database
        if "files" not in data:
            data["files"] = {}
        
        # Check if file already exists
        file_exists = file_path in data["files"]
        
        data["files"][file_path] = {
            "path": file_path,
            "name": file_name,
            "directory_path": current_dir,
            "content": content,
            "size": size,
            "extension": extension,
            "line_count": line_count,
            "word_count": word_count,
            "character_count": character_count,
            "created_at": data["files"][file_path]["created_at"] if file_exists else current_time,
            "modified_at": current_time,
            "accessed_at": current_time,
            "permissions": "-rw-r--r--"
        }
        
        # Update directory modified time
        if current_dir in data.get("directories", {}):
            data["directories"][current_dir]["modified_at"] = current_time
        
        # Log the operation
        if "operations_log" not in data:
            data["operations_log"] = {}
        
        # Generate unique operation ID
        op_count = len(data["operations_log"]) + 1
        op_id = f"op_{op_count:03d}"
        
        data["operations_log"][op_id] = {
            "operation_id": op_id,
            "session_id": current_session["session_id"],
            "operation_type": "write_file" if file_exists else "create_file",
            "command": f"echo '{content}' > {file_name}",
            "source_path": None,
            "destination_path": file_path,
            "success": True,
            "error_message": None,
            "timestamp": current_time
        }
        
        # Update session last command
        current_session["last_command"] = f"echo '{content}' > {file_name}"
        current_session["updated_at"] = current_time
        
        return json.dumps({
            "success": True,
            "terminal_output": None
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "terminal_output": None,
            "error": str(e)
        })