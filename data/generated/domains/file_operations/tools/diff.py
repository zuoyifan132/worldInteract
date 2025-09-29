import json
import uuid
import datetime
from typing import Dict, Any, List, Optional, Tuple

def diff(data: Dict[str, Any], file_name1: str, file_name2: str):
    """Compare two files line by line at the current directory."""
    import json
    import difflib
    from typing import Dict, Any

    try:
        # Get current working directory from active session
        # Try to find the most recently updated session
        current_session = None
        latest_update = None
        
        for session_id, session_data in data.get('sessions', {}).items():
            if latest_update is None or session_data.get('updated_at', '') > latest_update:
                latest_update = session_data.get('updated_at', '')
                current_session = session_data

        if not current_session:
            return json.dumps({
                "success": False,
                "error": "No active session found"
            })

        current_dir = current_session['current_working_directory']
        files_data = data.get('files', {})

        # Function to find file by name
        def find_file_path(file_name):
            # First try in current directory
            current_path = f"{current_dir}/{file_name}" if not current_dir.endswith('/') else f"{current_dir}{file_name}"
            if current_path in files_data:
                return current_path
            
            # If not found in current directory, search all files by name
            for file_path, file_info in files_data.items():
                if file_info.get('name') == file_name:
                    return file_path
            
            return None

        # Find both files
        file1_path = find_file_path(file_name1)
        file2_path = find_file_path(file_name2)

        if file1_path is None:
            return json.dumps({
                "success": False,
                "error": f"File '{file_name1}' not found in current directory"
            })

        if file2_path is None:
            return json.dumps({
                "success": False,
                "error": f"File '{file_name2}' not found in current directory"
            })

        # Get file contents
        file1_content = files_data[file1_path]['content']
        file2_content = files_data[file2_path]['content']

        # Split content into lines
        file1_lines = file1_content.splitlines(keepends=True)
        file2_lines = file2_content.splitlines(keepends=True)

        # Generate unified diff
        diff_result = list(difflib.unified_diff(
            file1_lines,
            file2_lines,
            fromfile=file_name1,
            tofile=file_name2,
            lineterm=''
        ))

        # Format the diff output
        if not diff_result:
            diff_output = "Files are identical"
        else:
            diff_output = '\n'.join(diff_result)

        # Log the operation
        operation_id = f"op_{len(data.get('operations_log', {})) + 1:03d}"
        if 'operations_log' not in data:
            data['operations_log'] = {}

        data['operations_log'][operation_id] = {
            "operation_id": operation_id,
            "session_id": current_session['session_id'],
            "operation_type": "diff_files",
            "command": f"diff {file_name1} {file_name2}",
            "source_path": file1_path,
            "destination_path": file2_path,
            "success": True,
            "error_message": None,
            "timestamp": "2024-01-20T15:00:00Z"
        }

        return json.dumps({
            "success": True,
            "diff_lines": diff_output
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Error comparing files: {str(e)}"
        })