import json
import uuid
import datetime
from typing import Dict, Any, List, Optional, Tuple

def rm(data: Dict[str, Any], file_name: str):
    """Remove a file or directory from the file system."""
    import json
    import os
    from datetime import datetime
    
    try:
        # Get current working directory from active session
        current_session = None
        for session_id, session_info in data.get('sessions', {}).items():
            current_session = session_info
            break
        
        if not current_session:
            return json.dumps({
                "success": False,
                "result": "Error: No active session found"
            })
        
        cwd = current_session['current_working_directory']
        
        # Determine the full path of the file/directory to remove
        if file_name.startswith('/'):
            # Absolute path
            target_path = file_name
        else:
            # Relative path - combine with current working directory
            target_path = os.path.join(cwd, file_name).replace('\\', '/')
            # Normalize path to handle cases like /home/user/../user/file.txt
            target_path = os.path.normpath(target_path).replace('\\', '/')
        
        # Check if it's a file
        if target_path in data.get('files', {}):
            # Remove the file
            del data['files'][target_path]
            
            # Update parent directory's modified time
            parent_dir = os.path.dirname(target_path).replace('\\', '/')
            if parent_dir in data.get('directories', {}):
                data['directories'][parent_dir]['modified_at'] = datetime.now().isoformat() + 'Z'
            
            # Log the operation
            operation_id = f"op_{len(data.get('operations_log', {})) + 1:03d}"
            data.setdefault('operations_log', {})[operation_id] = {
                "operation_id": operation_id,
                "session_id": current_session['session_id'],
                "operation_type": "remove_file",
                "command": f"rm {file_name}",
                "source_path": target_path,
                "destination_path": None,
                "success": True,
                "error_message": None,
                "timestamp": datetime.now().isoformat() + 'Z'
            }
            
            # Update session's last command
            current_session['last_command'] = f"rm {file_name}"
            current_session['updated_at'] = datetime.now().isoformat() + 'Z'
            
            return json.dumps({
                "success": True,
                "result": f"File '{file_name}' removed successfully"
            })
        
        # Check if it's a directory
        elif target_path in data.get('directories', {}):
            # Check if directory is empty (no files or subdirectories)
            has_files = any(file_info['directory_path'] == target_path 
                          for file_info in data.get('files', {}).values())
            has_subdirs = any(dir_info['parent_path'] == target_path 
                            for dir_info in data.get('directories', {}).values())
            
            if has_files or has_subdirs:
                # Log failed operation
                operation_id = f"op_{len(data.get('operations_log', {})) + 1:03d}"
                data.setdefault('operations_log', {})[operation_id] = {
                    "operation_id": operation_id,
                    "session_id": current_session['session_id'],
                    "operation_type": "remove_directory",
                    "command": f"rm {file_name}",
                    "source_path": target_path,
                    "destination_path": None,
                    "success": False,
                    "error_message": "Directory not empty",
                    "timestamp": datetime.now().isoformat() + 'Z'
                }
                
                return json.dumps({
                    "success": False,
                    "result": f"Error: Directory '{file_name}' is not empty. Use rm -r to remove non-empty directories"
                })
            
            # Remove empty directory
            del data['directories'][target_path]
            
            # Update parent directory's modified time
            parent_dir = os.path.dirname(target_path).replace('\\', '/')
            if parent_dir in data.get('directories', {}):
                data['directories'][parent_dir]['modified_at'] = datetime.now().isoformat() + 'Z'
            
            # Log the operation
            operation_id = f"op_{len(data.get('operations_log', {})) + 1:03d}"
            data.setdefault('operations_log', {})[operation_id] = {
                "operation_id": operation_id,
                "session_id": current_session['session_id'],
                "operation_type": "remove_directory",
                "command": f"rm {file_name}",
                "source_path": target_path,
                "destination_path": None,
                "success": True,
                "error_message": None,
                "timestamp": datetime.now().isoformat() + 'Z'
            }
            
            # Update session's last command
            current_session['last_command'] = f"rm {file_name}"
            current_session['updated_at'] = datetime.now().isoformat() + 'Z'
            
            return json.dumps({
                "success": True,
                "result": f"Directory '{file_name}' removed successfully"
            })
        
        else:
            # File or directory doesn't exist
            # Log failed operation
            operation_id = f"op_{len(data.get('operations_log', {})) + 1:03d}"
            data.setdefault('operations_log', {})[operation_id] = {
                "operation_id": operation_id,
                "session_id": current_session['session_id'],
                "operation_type": "remove_file",
                "command": f"rm {file_name}",
                "source_path": target_path,
                "destination_path": None,
                "success": False,
                "error_message": "File or directory not found",
                "timestamp": datetime.now().isoformat() + 'Z'
            }
            
            return json.dumps({
                "success": False,
                "result": f"Error: '{file_name}' not found"
            })
            
    except Exception as e:
        # Log failed operation
        if 'current_session' in locals() and current_session:
            operation_id = f"op_{len(data.get('operations_log', {})) + 1:03d}"
            data.setdefault('operations_log', {})[operation_id] = {
                "operation_id": operation_id,
                "session_id": current_session['session_id'],
                "operation_type": "remove_file",
                "command": f"rm {file_name}",
                "source_path": None,
                "destination_path": None,
                "success": False,
                "error_message": str(e),
                "timestamp": datetime.now().isoformat() + 'Z'
            }
        
        return json.dumps({
            "success": False,
            "result": f"Error: {str(e)}"
        })