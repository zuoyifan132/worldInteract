import json
import uuid
import datetime
from typing import Dict, Any, List, Optional

def search_files(data: Dict[str, Any], search_path: str, pattern: str, search_content: bool = False) -> str:
    """
    Search for files by name pattern or content.
    
    Args:
        data: In-memory database (nested dictionaries)
        search_path: Directory to search in
        pattern: Search pattern (file name or regex)
        search_content: Whether to search file contents (default: False)
        
    Returns:
        JSON string containing search results
    """
    try:
        import re
        import json
        from datetime import datetime
        
        # Validate inputs
        if not search_path or not pattern:
            raise ValueError("search_path and pattern are required")
        
        # Normalize search path
        search_path = search_path.rstrip('/')
        
        # Find the target directory
        target_directory_id = None
        for dir_id, directory in data.get('directories', {}).items():
            if directory.get('directory_path') == search_path:
                target_directory_id = dir_id
                break
        
        if target_directory_id is None:
            raise ValueError(f"Directory not found: {search_path}")
        
        matches = []
        
        # Compile regex pattern for file name matching
        try:
            regex_pattern = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # If pattern is not valid regex, treat as literal string
            regex_pattern = re.compile(re.escape(pattern), re.IGNORECASE)
        
        # Search through files
        for file_id, file_record in data.get('files', {}).items():
            file_path = file_record.get('file_path', '')
            file_name = file_record.get('name', '')
            
            # Check if file is in target directory or subdirectory
            if not file_path.startswith(search_path):
                continue
            
            # Search by file name pattern
            if regex_pattern.search(file_name):
                matches.append({
                    "file_path": file_path,
                    "match_type": "filename",
                    "line_number": 0
                })
            
            # Search file content if requested
            if search_content:
                file_content = file_record.get('content', '')
                if file_content:
                    lines = file_content.split('\n')
                    for line_num, line in enumerate(lines, 1):
                        if regex_pattern.search(line):
                            matches.append({
                                "file_path": file_path,
                                "match_type": "content",
                                "line_number": line_num
                            })
        
        # Generate operation ID and log the search operation
        operation_id = f"op_{int(datetime.now().timestamp() * 1000)}"
        
        # Log the search operation
        if 'file_operations' not in data:
            data['file_operations'] = {}
        
        data['file_operations'][operation_id] = {
            "operation_id": operation_id,
            "operation_type": "search",
            "file_id": None,
            "directory_id": target_directory_id,
            "source_path": search_path,
            "destination_path": None,
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "message": f"Search completed: found {len(matches)} matches"
        }
        
        # Update search index if content search was performed
        if search_content and matches:
            if 'search_index' not in data:
                data['search_index'] = {}
            
            for match in matches:
                if match['match_type'] == 'content':
                    index_id = f"idx_{int(datetime.now().timestamp() * 1000)}_{len(data['search_index'])}"
                    
                    # Find file_id for the match
                    file_id = None
                    for fid, file_record in data.get('files', {}).items():
                        if file_record.get('file_path') == match['file_path']:
                            file_id = fid
                            break
                    
                    data['search_index'][index_id] = {
                        "index_id": index_id,
                        "file_id": file_id,
                        "content_snippet": pattern,
                        "line_number": match['line_number'],
                        "match_type": "search_result",
                        "last_indexed": datetime.now().isoformat()
                    }
        
        result = {
            "success": True,
            "matches": matches,
            "message": f"Search completed successfully. Found {len(matches)} matches.",
            "after_execution_state": data
        }
        return json.dumps(result)
        
    except Exception as e:
        # Log failed operation
        try:
            operation_id = f"op_{int(datetime.now().timestamp() * 1000)}"
            if 'file_operations' not in data:
                data['file_operations'] = {}
            
            data['file_operations'][operation_id] = {
                "operation_id": operation_id,
                "operation_type": "search",
                "file_id": None,
                "directory_id": None,
                "source_path": search_path,
                "destination_path": None,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "message": f"Search failed: {str(e)}"
            }
        except:
            pass
        
        error_result = {
            "success": False,
            "matches": [],
            "message": f"Search operation failed: {str(e)}",
            "after_execution_state": data
        }
        return json.dumps(error_result)