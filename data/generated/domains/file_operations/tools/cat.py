import json
import uuid
import datetime
from typing import Dict, Any, List, Optional, Tuple

def cat(data: Dict[str, Any], file_name: str):
    """Display the contents of a file from current directory."""
    import json
    from datetime import datetime
    
    try:
        # Get current session to determine working directory
        sessions = data.get('sessions', {})
        if not sessions:
            return json.dumps({
                "success": False,
                "error": "No active session found"
            })
        
        # Get the first active session (in a real system, this would be based on session context)
        current_session = list(sessions.values())[0]
        current_dir = current_session['current_working_directory']
        
        # Construct the full file path
        file_path = f"{current_dir}/{file_name}"
        
        # Check if file exists in the files database
        files = data.get('files', {})
        if file_path not in files:
            return json.dumps({
                "success": False,
                "error": f"cat: {file_name}: No such file or directory"
            })
        
        file_info = files[file_path]
        
        # Update the accessed_at timestamp
        file_info['accessed_at'] = datetime.now().isoformat() + 'Z'
        
        # Log the operation
        operations_log = data.get('operations_log', {})
        op_id = f"op_{len(operations_log) + 1:03d}"
        operations_log[op_id] = {
            "operation_id": op_id,
            "session_id": current_session['session_id'],
            "operation_type": "read_file",
            "command": f"cat {file_name}",
            "source_path": file_path,
            "destination_path": None,
            "success": True,
            "error_message": None,
            "timestamp": datetime.now().isoformat() + 'Z'
        }
        
        # Update last command in session
        current_session['last_command'] = f"cat {file_name}"
        current_session['updated_at'] = datetime.now().isoformat() + 'Z'
        
        return json.dumps({
            "success": True,
            "file_content": file_info['content']
        })
        
    except Exception as e:
        # Log failed operation
        operations_log = data.get('operations_log', {})
        op_id = f"op_{len(operations_log) + 1:03d}"
        operations_log[op_id] = {
            "operation_id": op_id,
            "session_id": current_session.get('session_id', 'unknown') if 'current_session' in locals() else 'unknown',
            "operation_type": "read_file",
            "command": f"cat {file_name}",
            "source_path": None,
            "destination_path": None,
            "success": False,
            "error_message": str(e),
            "timestamp": datetime.now().isoformat() + 'Z'
        }
        
        return json.dumps({
            "success": False,
            "error": f"cat: {str(e)}"
        })