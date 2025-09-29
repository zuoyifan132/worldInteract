#!/usr/bin/env python3
"""
Test script to verify API collection normalization functionality.
"""

import json
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from worldInteract.utils.parser_utils import normalize_api_collection


def test_json_schema_format():
    """Test conversion from JSON Schema format to simplified format."""
    
    # Example tool in JSON Schema format (like api_collection_example.json)
    json_schema_tool = {
        "name": "cat",
        "description": "Display file contents",
        "parameters": {
            "type": "dict",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "The name of the file to display"
                }
            },
            "required": ["file_name"]
        },
        "returns": {
            "type": "dict", 
            "properties": {
                "file_content": {
                    "type": "string",
                    "description": "The content of the file"
                }
            }
        }
    }
    
    api_collection = {
        "domain": "test_domain",
        "tools": [json_schema_tool]
    }
    
    # Normalize the collection
    normalized = normalize_api_collection(api_collection)
    
    # Check the results
    normalized_tool = normalized["tools"][0]
    
    print("Original parameters:", json.dumps(json_schema_tool["parameters"], indent=2))
    print("Normalized parameters:", json.dumps(normalized_tool["parameters"], indent=2))
    print()
    print("Original returns:", json.dumps(json_schema_tool["returns"], indent=2))
    print("Normalized returns:", json.dumps(normalized_tool["returns"], indent=2))
    
    # Verify conversion
    expected_parameters = {
        "file_name": {
            "type": "string",
            "description": "The name of the file to display"
        }
    }
    
    expected_returns = {
        "file_content": {
            "type": "string",
            "description": "The content of the file"
        }
    }
    
    assert normalized_tool["parameters"] == expected_parameters, "Parameters normalization failed"
    assert normalized_tool["returns"] == expected_returns, "Returns normalization failed"
    
    print("✅ JSON Schema format test passed!")


def test_simplified_format():
    """Test that simplified format remains unchanged."""
    
    # Example tool already in simplified format
    simplified_tool = {
        "name": "test_tool",
        "description": "Test tool",
        "parameters": {
            "param1": {
                "type": "string",
                "description": "Parameter 1"
            }
        },
        "returns": {
            "result": {
                "type": "string", 
                "description": "Result field"
            }
        }
    }
    
    api_collection = {
        "domain": "test_domain",
        "tools": [simplified_tool]
    }
    
    # Normalize the collection
    normalized = normalize_api_collection(api_collection)
    
    # Should remain unchanged
    normalized_tool = normalized["tools"][0]
    
    assert normalized_tool["parameters"] == simplified_tool["parameters"], "Simplified parameters should remain unchanged"
    assert normalized_tool["returns"] == simplified_tool["returns"], "Simplified returns should remain unchanged"
    
    print("✅ Simplified format test passed!")


def test_real_api_collection():
    """Test with the actual api_collection_example.json file."""
    
    api_file = project_root / "data" / "apis_collections" / "api_collection_example.json"
    
    if api_file.exists():
        with open(api_file, 'r', encoding='utf-8') as f:
            original_collection = json.load(f)
        
        print(f"Testing with real API collection: {original_collection.get('domain', 'unknown')}")
        print(f"Number of tools: {len(original_collection.get('tools', []))}")
        
        # Normalize the collection
        normalized = normalize_api_collection(original_collection)
        
        # Check the first tool
        if normalized["tools"]:
            first_tool = normalized["tools"][0]
            print(f"First tool: {first_tool['name']}")
            print("Normalized parameters:", json.dumps(first_tool.get("parameters", {}), indent=2))
            print("Normalized returns:", json.dumps(first_tool.get("returns", {}), indent=2))
            
            # Verify it's in simplified format (no "type" and "properties" at top level)
            parameters = first_tool.get("parameters", {})
            returns = first_tool.get("returns", {})
            
            # Parameters should not have "type" and "properties" at top level
            assert "type" not in parameters or "properties" not in parameters, "Parameters should be in simplified format"
            assert "type" not in returns or "properties" not in returns, "Returns should be in simplified format"
            
        print("✅ Real API collection test passed!")
    else:
        print("⚠️  API collection file not found, skipping real file test")


if __name__ == "__main__":
    print("Testing API collection normalization...")
    print("=" * 50)
    
    test_json_schema_format()
    print()
    
    test_simplified_format()
    print()
    
    test_real_api_collection()
    print()
    
    print("🎉 All tests passed! API normalization is working correctly.")
