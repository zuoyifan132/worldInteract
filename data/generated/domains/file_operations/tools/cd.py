import json
import uuid
import datetime
from typing import Dict, Any, List, Optional, Tuple

def cd(data: Dict[str, Any], folder: str):
    """
    Change the current working directory to the specified folder.

    Args:
        data: Database state containing directories, sessions, etc.
        folder: The folder name to change to (relative path)

    Returns:
        JSON string with the new current working directory
    """
    import json
    import os
    from datetime import datetime

    try:
        # Get current session (assuming we're working with sess_001 as default)
        sessions = data.get("sessions", {})
        current_session_id = "sess_001"  # Default session

        if current_session_id not in sessions:
            return json.dumps({
                "success": False,
                "error": "No active session found"
            })

        current_session = sessions[current_session_id]
        current_dir = current_session["current_working_directory"]

        # Handle different folder navigation cases
        if folder == "..":
            # Go to parent directory
            if current_dir == "/":
                new_dir = "/"
            else:
                # Find parent directory
                parent_path = "/".join(current_dir.rstrip("/").split("/")[:-1])
                if not parent_path:
                    parent_path = "/"
                new_dir = parent_path
        elif folder == ".":
            # Stay in current directory
            new_dir = current_dir
        elif folder.startswith("/"):
            # Absolute path
            new_dir = folder
        else:
            # Relative path - append to current directory
            if current_dir.endswith("/"):
                new_dir = current_dir + folder
            else:
                new_dir = current_dir + "/" + folder

        # Normalize the path (remove double slashes, etc.)
        new_dir = os.path.normpath(new_dir).replace("\\", "/")
        if new_dir != "/" and new_dir.endswith("/"):
            new_dir = new_dir.rstrip("/")

        # Check if the target directory exists or is valid
        directories = data.get("directories", {})
        
        if folder == "..":
            # For parent directory navigation, check if the target is valid
            # Allow if it's root, or if it's referenced as a parent_path in existing directories
            if new_dir != "/":
                # Check if any existing directory has this path as its parent
                is_valid_parent = any(
                    dir_info.get("parent_path") == new_dir 
                    for dir_info in directories.values()
                )
                if not is_valid_parent:
                    return json.dumps({
                        "success": False,
                        "error": f"Directory '{folder}' not found"
                    })
        else:
            # For non-parent navigation, require directory to exist in database
            if new_dir not in directories and new_dir != "/":
                return json.dumps({
                    "success": False,
                    "error": f"Directory '{folder}' not found"
                })

        # Update the session's current working directory
        sessions[current_session_id]["current_working_directory"] = new_dir
        sessions[current_session_id]["last_command"] = f"cd {folder}"
        sessions[current_session_id]["updated_at"] = datetime.now().isoformat() + "Z"

        # Log the operation
        operations_log = data.get("operations_log", {})
        op_id = f"op_{len(operations_log) + 1:03d}"
        operations_log[op_id] = {
            "operation_id": op_id,
            "session_id": current_session_id,
            "operation_type": "change_directory",
            "command": f"cd {folder}",
            "source_path": current_dir,
            "destination_path": new_dir,
            "success": True,
            "error_message": None,
            "timestamp": datetime.now().isoformat() + "Z"
        }

        return json.dumps({
            "success": True,
            "current_working_directory": new_dir
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to change directory: {str(e)}"
        })