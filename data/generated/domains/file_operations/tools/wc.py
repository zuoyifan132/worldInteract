import json
import uuid
import datetime
from typing import Dict, Any, List, Optional, Tuple

def wc(data: Dict[str, Any], file_name: str, mode: str = "l"):
    """
    Count the number of lines, words, and characters in a file from current directory.
    
    Args:
        data: Database state containing file system information
        file_name: Name of the file in current directory to perform wc operation on
        mode: Mode of operation ('l' for lines, 'w' for words, 'c' for characters)
    
    Returns:
        JSON string with count and type information
    """
    import json
    from typing import Dict, Any
    
    try:
        # Get current working directory from active session
        current_session = None
        for session_id, session_info in data.get("sessions", {}).items():
            current_session = session_info
            break
        
        if not current_session:
            return json.dumps({
                "success": False,
                "error": "No active session found"
            })
        
        current_dir = current_session["current_working_directory"]
        
        # Construct full file path
        if current_dir.endswith("/"):
            file_path = current_dir + file_name
        else:
            file_path = current_dir + "/" + file_name
        
        # Find the file in the files database
        files = data.get("files", {})
        if file_path not in files:
            return json.dumps({
                "success": False,
                "error": f"File '{file_name}' not found in current directory"
            })
        
        file_info = files[file_path]
        
        # Validate mode parameter
        if mode not in ["l", "w", "c"]:
            return json.dumps({
                "success": False,
                "error": f"Invalid mode '{mode}'. Valid modes are 'l' (lines), 'w' (words), 'c' (characters)"
            })
        
        # Get the appropriate count based on mode
        if mode == "l":
            count = file_info["line_count"]
            count_type = "lines"
        elif mode == "w":
            count = file_info["word_count"]
            count_type = "words"
        else:  # mode == "c"
            count = file_info["character_count"]
            count_type = "characters"
        
        # Update file access time
        from datetime import datetime
        file_info["accessed_at"] = datetime.now().isoformat() + "Z"
        
        # Log the operation
        operation_id = f"op_{len(data.get('operations_log', {})) + 1:03d}"
        session_id = current_session["session_id"]
        
        if "operations_log" not in data:
            data["operations_log"] = {}
        
        data["operations_log"][operation_id] = {
            "operation_id": operation_id,
            "session_id": session_id,
            "operation_type": "read_file",
            "command": f"wc -{mode} {file_name}",
            "source_path": file_path,
            "destination_path": None,
            "success": True,
            "error_message": None,
            "timestamp": datetime.now().isoformat() + "Z"
        }
        
        # Update session's last command
        current_session["last_command"] = f"wc -{mode} {file_name}"
        current_session["updated_at"] = datetime.now().isoformat() + "Z"
        
        return json.dumps({
            "success": True,
            "count": count,
            "type": count_type
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"An error occurred: {str(e)}"
        })