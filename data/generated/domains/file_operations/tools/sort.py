import json
import uuid
import datetime
from typing import Dict, Any, List, Optional, Tuple

def sort(data: Dict[str, Any], file_name: str):
    """Sort the contents of a file line by line."""
    import json
    from typing import Dict, Any

    try:
        # Find the appropriate session - prefer one where the file exists
        current_session = None
        target_file_path = None
        
        # First, try to find a session where the file exists in its working directory
        for session_id, session_info in data.get('sessions', {}).items():
            working_dir = session_info['current_working_directory']
            potential_path = f"{working_dir}/{file_name}" if not file_name.startswith('/') else file_name
            
            if potential_path in data.get('files', {}):
                current_session = session_info
                target_file_path = potential_path
                break
        
        # If no session has the file in its working directory, use the first session
        if not current_session:
            for session_id, session_info in data.get('sessions', {}).items():
                current_session = session_info
                break
            
            if not current_session:
                return json.dumps({
                    "success": False,
                    "error": "No active session found"
                })
            
            # Construct file path based on this session's working directory
            current_dir = current_session['current_working_directory']
            target_file_path = f"{current_dir}/{file_name}" if not file_name.startswith('/') else file_name

        # Check if file exists
        if target_file_path not in data.get('files', {}):
            return json.dumps({
                "success": False,
                "error": f"File '{file_name}' not found in current directory"
            })

        file_info = data['files'][target_file_path]

        # Check if user has read permissions
        permissions = file_info.get('permissions', '-rw-r--r--')
        if 'r' not in permissions:
            return json.dumps({
                "success": False,
                "error": f"Permission denied: cannot read file '{file_name}'"
            })

        # Get file content and sort lines
        content = file_info.get('content', '')
        if not content:
            sorted_content = ''
        else:
            lines = content.split('\n')
            # Sort lines alphabetically (case-sensitive)
            sorted_lines = sorted(lines)
            sorted_content = '\n'.join(sorted_lines)

        # Update file's accessed_at timestamp
        from datetime import datetime
        data['files'][target_file_path]['accessed_at'] = datetime.now().isoformat() + 'Z'

        # Log the operation
        import uuid
        operation_id = f"op_{uuid.uuid4().hex[:8]}"
        data.setdefault('operations_log', {})[operation_id] = {
            "operation_id": operation_id,
            "session_id": current_session['session_id'],
            "operation_type": "sort_file",
            "command": f"sort {file_name}",
            "source_path": target_file_path,
            "destination_path": None,
            "success": True,
            "error_message": None,
            "timestamp": datetime.now().isoformat() + 'Z'
        }

        return json.dumps({
            "success": True,
            "sorted_content": sorted_content
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        })