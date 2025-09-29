import json
import uuid
import datetime
from typing import Dict, Any, List, Optional, Tuple

def cp(data: Dict[str, Any], source: str, destination: str) -> str:
    """Copy a file or directory from one location to another."""
    import json
    import uuid
    from datetime import datetime
    
    try:
        # Get current session to determine working directory
        current_session = None
        for session_id, session_data in data.get('sessions', {}).items():
            current_session = session_data
            break
        
        if not current_session:
            return json.dumps({
                "result": "Error: No active session found"
            })
        
        current_dir = current_session['current_working_directory']
        
        # Construct full paths
        if source.startswith('/'):
            source_path = source
        else:
            source_path = f"{current_dir}/{source}" if current_dir != '/' else f"/{source}"
        
        if destination.startswith('/'):
            dest_path = destination
        else:
            dest_path = f"{current_dir}/{destination}" if current_dir != '/' else f"/{destination}"
        
        # Check if source exists (file or directory)
        source_is_file = source_path in data.get('files', {})
        source_is_dir = source_path in data.get('directories', {})
        
        if not source_is_file and not source_is_dir:
            # Log failed operation
            op_id = f"op_{uuid.uuid4().hex[:8]}"
            data.setdefault('operations_log', {})[op_id] = {
                "operation_id": op_id,
                "session_id": current_session['session_id'],
                "operation_type": "copy",
                "command": f"cp {source} {destination}",
                "source_path": source_path,
                "destination_path": dest_path,
                "success": False,
                "error_message": f"Source '{source}' not found",
                "timestamp": datetime.now().isoformat() + "Z"
            }
            return json.dumps({
                "result": f"cp: cannot stat '{source}': No such file or directory"
            })
        
        # Check if destination is a directory
        dest_is_dir = dest_path in data.get('directories', {})
        
        if dest_is_dir:
            # Copy into the directory
            if source_is_file:
                source_name = data['files'][source_path]['name']
                final_dest_path = f"{dest_path}/{source_name}"
            else:
                source_name = data['directories'][source_path]['name']
                final_dest_path = f"{dest_path}/{source_name}"
        else:
            final_dest_path = dest_path
        
        # Check if destination already exists
        if final_dest_path in data.get('files', {}) or final_dest_path in data.get('directories', {}):
            # Log failed operation
            op_id = f"op_{uuid.uuid4().hex[:8]}"
            data.setdefault('operations_log', {})[op_id] = {
                "operation_id": op_id,
                "session_id": current_session['session_id'],
                "operation_type": "copy",
                "command": f"cp {source} {destination}",
                "source_path": source_path,
                "destination_path": final_dest_path,
                "success": False,
                "error_message": f"Destination '{final_dest_path}' already exists",
                "timestamp": datetime.now().isoformat() + "Z"
            }
            return json.dumps({
                "result": f"cp: '{destination}' already exists"
            })
        
        # Perform the copy operation
        timestamp = datetime.now().isoformat() + "Z"
        
        if source_is_file:
            # Copy file
            source_file = data['files'][source_path]
            final_dest_dir = '/'.join(final_dest_path.split('/')[:-1]) or '/'
            final_dest_name = final_dest_path.split('/')[-1]
            
            data.setdefault('files', {})[final_dest_path] = {
                "path": final_dest_path,
                "name": final_dest_name,
                "directory_path": final_dest_dir,
                "content": source_file['content'],
                "size": source_file['size'],
                "extension": source_file['extension'],
                "line_count": source_file['line_count'],
                "word_count": source_file['word_count'],
                "character_count": source_file['character_count'],
                "created_at": timestamp,
                "modified_at": timestamp,
                "accessed_at": timestamp,
                "permissions": source_file['permissions']
            }
            
            result_msg = f"File '{source}' copied to '{final_dest_path}'"
        
        else:
            # Copy directory
            source_dir = data['directories'][source_path]
            final_dest_parent = '/'.join(final_dest_path.split('/')[:-1]) or '/'
            final_dest_name = final_dest_path.split('/')[-1]
            
            data.setdefault('directories', {})[final_dest_path] = {
                "path": final_dest_path,
                "name": final_dest_name,
                "parent_path": final_dest_parent,
                "size": source_dir['size'],
                "created_at": timestamp,
                "modified_at": timestamp,
                "permissions": source_dir['permissions']
            }
            
            result_msg = f"Directory '{source}' copied to '{final_dest_path}'"
        
        # Log successful operation
        op_id = f"op_{uuid.uuid4().hex[:8]}"
        data.setdefault('operations_log', {})[op_id] = {
            "operation_id": op_id,
            "session_id": current_session['session_id'],
            "operation_type": "copy",
            "command": f"cp {source} {destination}",
            "source_path": source_path,
            "destination_path": final_dest_path,
            "success": True,
            "error_message": None,
            "timestamp": timestamp
        }
        
        # Update session
        current_session['last_command'] = f"cp {source} {destination}"
        current_session['updated_at'] = timestamp
        
        return json.dumps({
            "result": result_msg
        })
        
    except Exception as e:
        # Log failed operation
        op_id = f"op_{uuid.uuid4().hex[:8]}"
        data.setdefault('operations_log', {})[op_id] = {
            "operation_id": op_id,
            "session_id": current_session['session_id'] if current_session else "unknown",
            "operation_type": "copy",
            "command": f"cp {source} {destination}",
            "source_path": source_path if 'source_path' in locals() else None,
            "destination_path": dest_path if 'dest_path' in locals() else None,
            "success": False,
            "error_message": str(e),
            "timestamp": datetime.now().isoformat() + "Z"
        }
        
        return json.dumps({
            "result": f"cp: error occurred - {str(e)}"
        })