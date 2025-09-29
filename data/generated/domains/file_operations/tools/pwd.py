import json
import uuid
import datetime
from typing import Dict, Any, List, Optional, Tuple

def pwd(data: Dict[str, Any]):
    """Return the current working directory path."""
    import json
    from typing import Dict, Any
    
    try:
        # Get sessions data
        sessions = data.get('sessions', {})
        
        # Find the most recently updated session to get current working directory
        current_session = None
        latest_update = None
        
        for session_id, session_info in sessions.items():
            if current_session is None or session_info['updated_at'] > latest_update:
                current_session = session_info
                latest_update = session_info['updated_at']
        
        if current_session is None:
            # If no sessions exist, default to root directory
            current_working_directory = "/"
        else:
            current_working_directory = current_session['current_working_directory']
        
        return json.dumps({
            "success": True,
            "current_working_directory": current_working_directory
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to get current working directory: {str(e)}"
        })