import json
import uuid
import datetime
from typing import Dict, Any, List, Optional, Tuple

def grep(data: Dict[str, Any], file_name: str, pattern: str):
    """Search for lines in a file that contain the specified pattern."""
    import json
    import re
    from typing import Dict, Any, List

    try:
        # Find the session that has the file in its current working directory
        current_session = None
        
        # First, try to find which directory contains the file
        target_file_path = None
        for file_path in data.get('files', {}):
            if data['files'][file_path]['name'] == file_name:
                target_file_path = file_path
                break
        
        if target_file_path:
            # Get the directory path of the file
            file_directory = data['files'][target_file_path]['directory_path']
            
            # Find the session with this directory as current working directory
            for session_id, session_data in data.get('sessions', {}).items():
                if session_data['current_working_directory'] == file_directory:
                    current_session = session_data
                    break
        
        # If no session found with the file's directory, use the most recently updated session
        if not current_session:
            latest_update = None
            for session_id, session_data in data.get('sessions', {}).items():
                session_update_time = session_data.get('updated_at', '')
                if latest_update is None or session_update_time > latest_update:
                    latest_update = session_update_time
                    current_session = session_data

        if not current_session:
            return json.dumps({
                "success": False,
                "error": "No active session found"
            })

        current_dir = current_session['current_working_directory']

        # Construct the full file path
        file_path = f"{current_dir}/{file_name}" if not current_dir.endswith('/') else f"{current_dir}{file_name}"

        # Check if file exists in the files table
        if file_path not in data.get('files', {}):
            return json.dumps({
                "success": False,
                "error": f"File '{file_name}' not found in current directory"
            })

        file_data = data['files'][file_path]
        content = file_data.get('content', '')

        # Split content into lines and search for pattern
        lines = content.split('\n')
        matching_lines = []

        # Use regex for pattern matching to support more flexible searches
        try:
            pattern_regex = re.compile(pattern)
            for line in lines:
                if pattern_regex.search(line):
                    matching_lines.append(line)
        except re.error:
            # If regex compilation fails, fall back to simple string matching
            for line in lines:
                if pattern in line:
                    matching_lines.append(line)

        # Update file access time
        from datetime import datetime
        data['files'][file_path]['accessed_at'] = datetime.now().isoformat() + 'Z'

        # Log the operation
        operation_id = f"op_{len(data.get('operations_log', {})) + 1:03d}"
        data.setdefault('operations_log', {})[operation_id] = {
            "operation_id": operation_id,
            "session_id": current_session['session_id'],
            "operation_type": "grep",
            "command": f"grep {pattern} {file_name}",
            "source_path": file_path,
            "destination_path": None,
            "success": True,
            "error_message": None,
            "timestamp": datetime.now().isoformat() + 'Z'
        }

        # Update session last command
        data['sessions'][current_session['session_id']]['last_command'] = f"grep {pattern} {file_name}"
        data['sessions'][current_session['session_id']]['updated_at'] = datetime.now().isoformat() + 'Z'

        return json.dumps({
            "success": True,
            "matching_lines": matching_lines
        })

    except Exception as e:
        # Log failed operation
        if 'current_session' in locals() and current_session:
            operation_id = f"op_{len(data.get('operations_log', {})) + 1:03d}"
            data.setdefault('operations_log', {})[operation_id] = {
                "operation_id": operation_id,
                "session_id": current_session['session_id'],
                "operation_type": "grep",
                "command": f"grep {pattern} {file_name}",
                "source_path": None,
                "destination_path": None,
                "success": False,
                "error_message": str(e),
                "timestamp": datetime.now().isoformat() + 'Z'
            }

        return json.dumps({
            "success": False,
            "error": f"Error during grep operation: {str(e)}"
        })