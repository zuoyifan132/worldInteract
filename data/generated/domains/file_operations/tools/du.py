import json
import uuid
import datetime
from typing import Dict, Any, List, Optional, Tuple

def du(data: Dict[str, Any], human_readable: bool = False):
    """
    Estimate the disk usage of a directory and its contents.
    
    Args:
        data: Database state containing file system information
        human_readable: If True, returns size in human-readable format (KB, MB, etc.)
    
    Returns:
        JSON string with disk usage information
    """
    import json
    from typing import Dict, Any
    
    try:
        # Get current working directory from active session
        sessions = data.get('sessions', {})
        current_dir = None
        
        # Find the most recently updated session to get current directory
        latest_session = None
        latest_time = None
        
        for session_id, session_info in sessions.items():
            if latest_time is None or session_info.get('updated_at', '') > latest_time:
                latest_time = session_info.get('updated_at', '')
                latest_session = session_info
        
        if latest_session:
            current_dir = latest_session.get('current_working_directory', '/')
        else:
            current_dir = '/'
        
        # Calculate total disk usage for the current directory
        total_size = 0
        directories = data.get('directories', {})
        files = data.get('files', {})
        
        # Add size of current directory itself if it exists
        if current_dir in directories:
            total_size += directories[current_dir].get('size', 0)
        
        # Add sizes of all subdirectories under current directory
        for dir_path, dir_info in directories.items():
            if dir_path.startswith(current_dir) and dir_path != current_dir:
                # Check if it's a direct or indirect subdirectory
                relative_path = dir_path[len(current_dir):].lstrip('/')
                if relative_path:  # Not empty, so it's a subdirectory
                    total_size += dir_info.get('size', 0)
        
        # Add sizes of all files in current directory and subdirectories
        for file_path, file_info in files.items():
            file_dir = file_info.get('directory_path', '')
            if file_dir.startswith(current_dir):
                total_size += file_info.get('size', 0)
        
        # Format the result based on human_readable flag
        if human_readable:
            disk_usage = _format_human_readable(total_size)
        else:
            disk_usage = str(total_size)
        
        return json.dumps({
            "success": True,
            "disk_usage": disk_usage
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to calculate disk usage: {str(e)}",
            "disk_usage": "0"
        })

def _format_human_readable(size_bytes: int) -> str:
    """Convert bytes to human readable format"""
    if size_bytes == 0:
        return "0B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    if unit_index == 0:
        return f"{int(size)}B"
    else:
        return f"{size:.1f}{units[unit_index]}"