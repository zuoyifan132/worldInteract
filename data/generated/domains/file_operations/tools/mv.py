import json
import uuid
import datetime
from typing import Dict, Any, List, Optional, Tuple

def mv(data: Dict[str, Any], source: str, destination: str):
    """Move a file or directory from one location to another within the current directory"""
    import json
    import uuid
    from datetime import datetime
    
    try:
        # Get current session to determine working directory
        sessions = data.get('sessions', {})
        current_session = None
        for session in sessions.values():
            current_session = session
            break
        
        if not current_session:
            return json.dumps({
                "success": False,
                "error": "No active session found"
            })
        
        current_dir = current_session['current_working_directory']
        
        # Validate that source and destination are local names (no paths)
        if '/' in source or '/' in destination:
            return json.dumps({
                "success": False,
                "error": "Source and destination must be local to current directory (no paths allowed)"
            })
        
        # Construct full paths
        source_path = f"{current_dir}/{source}" if current_dir != "/" else f"/{source}"
        destination_path = f"{current_dir}/{destination}" if current_dir != "/" else f"/{destination}"
        
        # Check if source exists (file or directory)
        source_is_file = source_path in data.get('files', {})
        source_is_dir = source_path in data.get('directories', {})
        
        if not source_is_file and not source_is_dir:
            return json.dumps({
                "success": False,
                "error": f"Source '{source}' not found in current directory"
            })
        
        # Check if destination already exists
        dest_is_file = destination_path in data.get('files', {})
        dest_is_dir = destination_path in data.get('directories', {})
        
        if dest_is_file or dest_is_dir:
            return json.dumps({
                "success": False,
                "error": f"Destination '{destination}' already exists"
            })
        
        current_time = datetime.now().isoformat() + 'Z'
        
        # Perform the move operation
        if source_is_file:
            # Moving a file
            file_data = data['files'][source_path].copy()
            file_data['path'] = destination_path
            file_data['name'] = destination
            file_data['modified_at'] = current_time
            
            # Add to new location and remove from old
            data['files'][destination_path] = file_data
            del data['files'][source_path]
            
            operation_type = "move_file"
            result_message = f"File '{source}' moved to '{destination}'"
            
        else:
            # Moving a directory
            dir_data = data['directories'][source_path].copy()
            dir_data['path'] = destination_path
            dir_data['name'] = destination
            dir_data['modified_at'] = current_time
            
            # Add to new location and remove from old
            data['directories'][destination_path] = dir_data
            del data['directories'][source_path]
            
            # Update any child directories and files
            old_prefix = source_path
            new_prefix = destination_path
            
            # Update child directories
            dirs_to_update = {}
            for dir_path, dir_info in data.get('directories', {}).items():
                if dir_path.startswith(old_prefix + "/"):
                    new_path = dir_path.replace(old_prefix, new_prefix, 1)
                    new_info = dir_info.copy()
                    new_info['path'] = new_path
                    new_info['parent_path'] = new_path.rsplit('/', 1)[0] if '/' in new_path else '/'
                    new_info['modified_at'] = current_time
                    dirs_to_update[new_path] = new_info
            
            # Apply directory updates
            for old_path in list(data.get('directories', {}).keys()):
                if old_path.startswith(old_prefix + "/"):
                    del data['directories'][old_path]
            for new_path, new_info in dirs_to_update.items():
                data['directories'][new_path] = new_info
            
            # Update child files
            files_to_update = {}
            for file_path, file_info in data.get('files', {}).items():
                if file_path.startswith(old_prefix + "/"):
                    new_path = file_path.replace(old_prefix, new_prefix, 1)
                    new_info = file_info.copy()
                    new_info['path'] = new_path
                    new_info['directory_path'] = new_path.rsplit('/', 1)[0] if '/' in new_path else '/'
                    new_info['modified_at'] = current_time
                    files_to_update[new_path] = new_info
            
            # Apply file updates
            for old_path in list(data.get('files', {}).keys()):
                if old_path.startswith(old_prefix + "/"):
                    del data['files'][old_path]
            for new_path, new_info in files_to_update.items():
                data['files'][new_path] = new_info
            
            operation_type = "move_directory"
            result_message = f"Directory '{source}' moved to '{destination}'"
        
        # Update parent directory modification time
        parent_dir_data = data.get('directories', {}).get(current_dir)
        if parent_dir_data:
            parent_dir_data['modified_at'] = current_time
        
        # Log the operation
        operation_id = f"op_{uuid.uuid4().hex[:8]}"
        if 'operations_log' not in data:
            data['operations_log'] = {}
        
        data['operations_log'][operation_id] = {
            "operation_id": operation_id,
            "session_id": current_session['session_id'],
            "operation_type": operation_type,
            "command": f"mv {source} {destination}",
            "source_path": source_path,
            "destination_path": destination_path,
            "success": True,
            "error_message": None,
            "timestamp": current_time
        }
        
        # Update session
        current_session['last_command'] = f"mv {source} {destination}"
        current_session['updated_at'] = current_time
        
        return json.dumps({
            "success": True,
            "result": result_message
        })
        
    except Exception as e:
        # Log failed operation
        try:
            operation_id = f"op_{uuid.uuid4().hex[:8]}"
            if 'operations_log' not in data:
                data['operations_log'] = {}
            
            data['operations_log'][operation_id] = {
                "operation_id": operation_id,
                "session_id": current_session['session_id'] if current_session else "unknown",
                "operation_type": "move",
                "command": f"mv {source} {destination}",
                "source_path": None,
                "destination_path": None,
                "success": False,
                "error_message": str(e),
                "timestamp": datetime.now().isoformat() + 'Z'
            }
        except:
            pass
        
        return json.dumps({
            "success": False,
            "error": f"Move operation failed: {str(e)}"
        })