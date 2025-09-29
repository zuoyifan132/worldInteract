import json
import uuid
import datetime
from typing import Dict, Any, List, Optional, Tuple

def tail(data: Dict[str, Any], file_name: str, lines: int = 10):
    """Display the last part of a file of any extension."""
    import json
    from typing import Dict, Any

    try:
        # Find which session has access to the requested file
        current_session = None
        file_path = None
        
        # Check all sessions to find one that contains the file
        for session_id, session_info in data.get('sessions', {}).items():
            current_dir = session_info['current_working_directory']
            potential_file_path = f"{current_dir}/{file_name}"
            
            if potential_file_path in data.get('files', {}):
                current_session = session_info
                file_path = potential_file_path
                break
        
        # If no session contains the file, use the most recently updated session
        if not current_session:
            latest_update = None
            for session_id, session_info in data.get('sessions', {}).items():
                session_update_time = session_info.get('updated_at')
                if latest_update is None or session_update_time > latest_update:
                    latest_update = session_update_time
                    current_session = session_info
            
            if not current_session:
                return json.dumps({
                    "success": False,
                    "error": "No active session found"
                })
            
            current_dir = current_session['current_working_directory']
            file_path = f"{current_dir}/{file_name}"

        # Check if file exists
        if file_path not in data.get('files', {}):
            return json.dumps({
                "success": False,
                "error": f"File '{file_name}' not found in current directory"
            })

        file_info = data['files'][file_path]
        content = file_info['content']

        # Split content into lines
        all_lines = content.split('\n')

        # Handle edge cases
        if lines <= 0:
            return json.dumps({
                "success": True,
                "last_lines": ""
            })

        # Get the last 'lines' number of lines
        if lines >= len(all_lines):
            # If requested lines >= total lines, return all lines
            last_lines = all_lines
        else:
            # Get the last 'lines' lines
            last_lines = all_lines[-lines:]

        # Join lines back together
        result = '\n'.join(last_lines)

        # Update file access time
        from datetime import datetime
        data['files'][file_path]['accessed_at'] = datetime.now().isoformat() + 'Z'

        # Log the operation
        operation_id = f"op_{len(data.get('operations_log', {})) + 1:03d}"
        data.setdefault('operations_log', {})[operation_id] = {
            "operation_id": operation_id,
            "session_id": current_session['session_id'],
            "operation_type": "read_file",
            "command": f"tail -n {lines} {file_name}",
            "source_path": file_path,
            "destination_path": None,
            "success": True,
            "error_message": None,
            "timestamp": datetime.now().isoformat() + 'Z'
        }

        return json.dumps({
            "success": True,
            "last_lines": result
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        })