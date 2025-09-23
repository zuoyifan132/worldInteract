"""
Utility functions for parsing and extracting content from text responses.
"""

import json
import re
import logging
from typing import Any, Optional


logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON content from text that may contain other content.
    
    Args:
        text: Input text that may contain JSON
        
    Returns:
        Extracted JSON string
        
    Raises:
        ValueError: If no valid JSON is found
    """
    if not text:
        raise ValueError("Empty text provided")
    
    # Try to find JSON in code blocks first
    json_patterns = [
        r'```json\s*\n(.*?)\n```',  # JSON code blocks
        r'```\s*\n(\{.*?\})\s*\n```',  # Generic code blocks with JSON
        r'```\s*\n(\[.*?\])\s*\n```',  # Generic code blocks with JSON arrays
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                # Validate that it's valid JSON
                json.loads(match.strip())
                return match.strip()
            except json.JSONDecodeError:
                continue
    
    # Try to find JSON objects or arrays in the text
    json_object_patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple nested JSON objects
        r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # Simple nested JSON arrays
    ]
    
    for pattern in json_object_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                # Validate that it's valid JSON
                json.loads(match.strip())
                return match.strip()
            except json.JSONDecodeError:
                continue
    
    # Try to extract the largest JSON-like structure
    # Look for balanced braces
    brace_stack = []
    start_idx = None
    
    for i, char in enumerate(text):
        if char == '{':
            if not brace_stack:
                start_idx = i
            brace_stack.append(char)
        elif char == '}' and brace_stack:
            brace_stack.pop()
            if not brace_stack and start_idx is not None:
                # Found a complete JSON object
                candidate = text[start_idx:i+1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue
    
    # Look for balanced brackets for arrays
    bracket_stack = []
    start_idx = None
    
    for i, char in enumerate(text):
        if char == '[':
            if not bracket_stack:
                start_idx = i
            bracket_stack.append(char)
        elif char == ']' and bracket_stack:
            bracket_stack.pop()
            if not bracket_stack and start_idx is not None:
                # Found a complete JSON array
                candidate = text[start_idx:i+1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue
    
    # If nothing else works, try the whole text as JSON
    try:
        json.loads(text.strip())
        return text.strip()
    except json.JSONDecodeError:
        pass
    
    # Log the problematic text for debugging
    logger.error(f"Could not extract valid JSON from text: {text[:200]}...")
    raise ValueError("No valid JSON found in the provided text")


def extract_python_code_from_text(text: str) -> str:
    """
    Extract Python code from text that may contain other content.
    
    Args:
        text: Input text that may contain Python code
        
    Returns:
        Extracted Python code string
        
    Raises:
        ValueError: If no valid Python code is found
    """
    if not text:
        raise ValueError("Empty text provided")
    
    # Try to find Python code in code blocks first
    python_patterns = [
        r'```python\s*\n(.*?)\n```',  # Python code blocks
        r'```py\s*\n(.*?)\n```',     # Python code blocks (short form)
        r'```\s*\n(def .*?)\n```',   # Generic code blocks starting with def
    ]
    
    for pattern in python_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            if match.strip():
                return match.strip()
    
    # Look for function definitions in the text
    function_pattern = r'def\s+\w+\s*\([^)]*\)\s*(?:->[^:]+)?\s*:.*?(?=\ndef|\Z)'
    matches = re.findall(function_pattern, text, re.DOTALL)
    
    if matches:
        # Return the first function found
        return matches[0].strip()
    
    # If nothing else works, try the whole text as Python code
    if 'def ' in text:
        return text.strip()
    
    logger.error(f"Could not extract valid Python code from text: {text[:200]}...")
    raise ValueError("No valid Python code found in the provided text")


def clean_response_text(text: str) -> str:
    """
    Clean response text by removing common artifacts from LLM responses.
    
    Args:
        text: Raw response text
        
    Returns:
        Cleaned text
    """
    if not text:
        return text
    
    # Remove common prefixes/suffixes
    prefixes_to_remove = [
        "Here's the",
        "Here is the",
        "The following is",
        "Below is the",
        "```json",
        "```python",
        "```"
    ]
    
    suffixes_to_remove = [
        "```",
        "Let me know if you need any modifications!",
        "Is there anything else you'd like me to help with?",
        "Feel free to ask if you have any questions!"
    ]
    
    cleaned = text.strip()
    
    # Remove prefixes
    for prefix in prefixes_to_remove:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    
    # Remove suffixes
    for suffix in suffixes_to_remove:
        if cleaned.lower().endswith(suffix.lower()):
            cleaned = cleaned[:-len(suffix)].strip()
    
    return cleaned


def validate_json_structure(json_str: str, required_keys: list = None) -> bool:
    """
    Validate that a JSON string has the required structure.
    
    Args:
        json_str: JSON string to validate
        required_keys: List of required top-level keys
        
    Returns:
        True if valid, False otherwise
    """
    try:
        data = json.loads(json_str)
        
        if required_keys:
            if isinstance(data, dict):
                return all(key in data for key in required_keys)
            else:
                return False
        
        return True
        
    except json.JSONDecodeError:
        return False