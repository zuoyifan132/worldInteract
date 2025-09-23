# Tool Generator

The Tool Generator creates executable Python implementations from API descriptions and database schemas using Large Language Models.

## Overview

The Tool Generator automatically converts API tool specifications into working Python functions that operate on in-memory JSON databases. Each generated tool follows τ-bench patterns for data manipulation while maintaining compatibility with the specific database schema.

## Generation Approach

### 🎯 **Schema-Aware Generation**
- Uses database schema to understand data structure
- Generates code that properly accesses nested dictionaries
- Maintains referential integrity in operations

### 🔧 **Function-Based Implementation**
- Each tool becomes a standalone Python function
- Consistent parameter interface with `data` dictionary
- JSON string returns for all operations

### 🛡️ **Robust Error Handling**
- Comprehensive try-catch blocks
- Graceful failure with informative messages
- Input validation and type checking

## Tool Function Template

Generated tools follow this template:

```python
def tool_name(data: Dict[str, Any], param1: type1, param2: type2, ...) -> str:
    """
    Tool description here.
    
    Args:
        data: In-memory database (nested dictionaries)
        param1: Parameter 1 description
        param2: Parameter 2 description
        
    Returns:
        JSON string containing operation result
    """
    try:
        # 1. Validate inputs
        # 2. Access/modify data dictionary
        # 3. Perform the required operation
        # 4. Return result as JSON string
        
        result = {
            "success": True,
            "data": {...},
            "message": "Operation completed successfully"
        }
        return json.dumps(result)
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "message": f"Operation failed: {str(e)}"
        }
        return json.dumps(error_result)
```

## Usage Example

```python
from worldInteract.core.tool_generator import ToolGenerator

# Initialize generator
tool_gen = ToolGenerator()

# Generate tools from API collection, schema and initial state
tools = tool_gen.generate_tools(api_collection, schema, initial_state)

# Save generated tools
tool_gen.save_tools(tools, "file_operations")

# Access individual tool code
create_file_code = tools["create_file"]
```

## Generation Process

### 1. **API Analysis**
- Parse tool descriptions and parameters
- Extract expected return types and formats
- Identify operation types (read/write/mixed)

### 2. **Schema Integration**
- Understand database structure from schema
- Map tool operations to data access patterns
- Ensure proper relationship handling

### 3. **Code Generation**
- Generate Python functions using LLM
- Include proper type hints and documentation
- Add comprehensive error handling

### 4. **Validation**
- Syntax validation using AST parsing
- Function definition verification
- Required pattern checking

## Configuration

Tool generation is configured through `config/model_config.yaml`:

```yaml
tool_generation:
  model: "claude_3d7"
  temperature: 0.1
  max_tokens: 6000
  retry_attempts: 3
```

Environment settings in `config/environment_config.yaml`:

```yaml
tool_generation:
  enforce_type_hints: true
  include_error_handling: true
  max_function_length: 100  # lines
  validation_required: true
```

## Key Features

### 🤖 **LLM-Powered Implementation**
- Uses advanced language models for code generation
- Incorporates programming best practices
- Handles complex database operations intelligently

### 🔄 **Retry Mechanism**
- Automatic retry on generation failures
- Up to 3 attempts with refined prompts
- Progressive prompt improvement

### ✅ **Code Validation**
- Syntax checking with AST parsing
- Function definition verification
- Import and pattern validation

### 📁 **Multiple Output Formats**
- Individual tool files for modularity
- Combined tools file for convenience
- Proper import statements and documentation

## Generated Tool Examples

### File Operations

```python
def create_file(data: Dict[str, Any], file_path: str, content: str, encoding: str = "utf-8") -> str:
    """
    Create a new file with specified content.
    
    Args:
        data: In-memory database
        file_path: Path where the file should be created
        content: Content to write to the file
        encoding: File encoding (default: utf-8)
        
    Returns:
        JSON string containing operation result
    """
    try:
        import uuid
        import datetime
        
        # Validate inputs
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")
        
        # Generate unique file ID
        file_id = f"file_{uuid.uuid4().hex[:8]}"
        
        # Create file record
        file_record = {
            "file_id": file_id,
            "name": file_path.split("/")[-1],
            "path": file_path,
            "content": content,
            "size": len(content.encode(encoding)),
            "encoding": encoding,
            "created_at": datetime.datetime.now().isoformat(),
            "modified_at": datetime.datetime.now().isoformat()
        }
        
        # Add to database
        if "files" not in data:
            data["files"] = {}
        
        data["files"][file_id] = file_record
        
        result = {
            "success": True,
            "file_id": file_id,
            "message": f"File created successfully at {file_path}",
            "after_execution_state": data
        }
        return json.dumps(result)
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "message": f"Failed to create file: {str(e)}",
            "after_execution_state": data
        }
        return json.dumps(error_result)
```

## File Organization

Generated tools are saved in the following structure:

```
data/generated/domains/{domain}/
├── tools.py                    # Combined tools file
└── tools/                      # Individual tool files
    ├── create_file.py
    ├── read_file.py
    ├── delete_file.py
    └── ...
```

## Code Quality Standards

### 📝 **Documentation**
- Comprehensive docstrings for all functions
- Clear parameter and return descriptions
- Usage examples where appropriate

### 🔒 **Type Safety**
- Type hints for all parameters and returns
- Input validation and type checking
- Proper handling of optional parameters

### 🛡️ **Error Handling**
- Try-catch blocks for all operations
- Informative error messages
- Graceful degradation on failures

### 🎯 **Consistency**
- Standardized return format (JSON strings)
- Consistent naming conventions
- Uniform error handling patterns

## Integration Points

- **Environment Manager**: Called during environment creation
- **Schema Generator**: Receives schema for code generation
- **Validator**: Provides tools for validation testing
- **Model Manager**: Uses configured LLMs for generation

## Best Practices

1. **Schema Understanding**: Ensure generated code properly uses schema
2. **Error Handling**: Include comprehensive exception handling
3. **Type Safety**: Use proper type hints and validation
4. **Documentation**: Provide clear function documentation
5. **Testing**: Validate generated code through execution testing

## Troubleshooting

### Common Issues

1. **Syntax Errors**: Check AST parsing and regenerate if needed
2. **Import Issues**: Ensure all required imports are included
3. **Schema Misalignment**: Verify tool operations match schema structure
4. **Type Mismatches**: Validate parameter types against API specification

### Debug Tips

1. **Enable Logging**: Set log level to DEBUG for detailed information
2. **Check Generated Code**: Review individual tool files for issues
3. **Validate Schema**: Ensure schema is correct before tool generation
4. **Test Manually**: Execute generated functions with sample data

