# Schema Generator

The Schema Generator creates JSON-based database schemas from API collections using Large Language Models.

## Overview

The Schema Generator analyzes API tool collections and automatically designs lightweight, in-memory database schemas that support all tool operations. It follows τ-bench principles for JSON-based data storage while ensuring compatibility with diverse tool requirements.

## Design Principles

### 🎯 **Tool-Centric Design**
- Analyzes tool parameters and operations
- Ensures schema supports all read/write operations
- Maintains compatibility with tool requirements

### 📊 **Lightweight Architecture**
- JSON objects as tables/collections
- In-memory operation optimized
- No traditional database dependencies

### 🔗 **Relationship Management**
- Proper entity relationships
- Referential integrity through ID fields
- Support for one-to-one, one-to-many relationships

## Schema Structure

Generated schemas follow this structure:

```json
{
  "table_name": {
    "description": "Table description",
    "primary_key": "field_name",
    "fields": {
      "field_name": {
        "type": "string|integer|float|boolean|array|object",
        "description": "Field description",
        "required": true|false,
        "default": "default_value (optional)"
      }
    },
    "relationships": {
      "field_name": {
        "type": "one_to_one|one_to_many|many_to_many",
        "table": "related_table_name",
        "field": "related_field_name"
      }
    }
  }
}
```

## Usage Example

```python
from worldInteract.core.schema_generator import SchemaGenerator

# Initialize generator
schema_gen = SchemaGenerator()

# Load API collection
with open("data/apis_collections/file_operations.json") as f:
    api_collection = json.load(f)

# Generate schema
schema = schema_gen.generate_schema(api_collection)

# Save schema
schema_gen.save_schema(schema, "file_operations")
```

## Generation Process

### 1. **API Analysis**
- Parse tool descriptions and parameters
- Identify data entities and operations
- Determine read/write patterns

### 2. **Schema Design**
- Create tables for identified entities
- Define field types and constraints
- Establish relationships between tables

### 3. **LLM Generation**
- Use specialized prompts for schema generation
- Leverage domain knowledge from tool descriptions
- Ensure comprehensive coverage of tool needs

### 4. **Validation**
- Verify schema structure compliance
- Check for required fields and relationships
- Validate against tool requirements

## Configuration

Schema generation is configured through `config/model_config.yaml`:

```yaml
schema_generation:
  model: "openai_gpt"
  temperature: 0.3
  max_tokens: 4000
  retry_attempts: 3
```

Domain-specific settings in `config/environment_config.yaml`:

```yaml
domains:
  file_operations:
    schema_tables: ["files", "directories", "permissions", "metadata"]
    max_file_size_mb: 100
    supported_extensions: [".txt", ".json", ".csv", ".md", ".py"]
```

## Key Features

### 🤖 **LLM-Powered Generation**
- Uses advanced language models for schema design
- Incorporates domain knowledge and best practices
- Handles complex tool requirements intelligently

### 🔄 **Retry Mechanism**
- Automatic retry on generation failures
- Up to 3 attempts with exponential backoff
- Fallback to simpler schemas if needed

### ✅ **Comprehensive Validation**
- Schema structure validation
- Field type checking
- Relationship consistency verification

### 💾 **Persistence**
- Save schemas to JSON files
- Load existing schemas for reuse
- Organized file structure

## Generated Schema Examples

### File Operations Domain
```json
{
  "files": {
    "description": "File metadata and content storage",
    "primary_key": "file_id",
    "fields": {
      "file_id": {"type": "string", "required": true},
      "name": {"type": "string", "required": true},
      "path": {"type": "string", "required": true},
      "content": {"type": "string", "required": false},
      "size": {"type": "integer", "required": true},
      "created_at": {"type": "string", "required": true},
      "modified_at": {"type": "string", "required": true}
    },
    "relationships": {
      "directory_id": {
        "type": "many_to_one",
        "table": "directories",
        "field": "directory_id"
      }
    }
  },
  "directories": {
    "description": "Directory structure and metadata",
    "primary_key": "directory_id",
    "fields": {
      "directory_id": {"type": "string", "required": true},
      "name": {"type": "string", "required": true},
      "path": {"type": "string", "required": true},
      "parent_id": {"type": "string", "required": false}
    }
  }
}
```

## Error Handling

The Schema Generator handles various error scenarios:

- **LLM Generation Failures**: Retry with modified prompts
- **JSON Parsing Errors**: Regenerate with stricter formatting
- **Schema Validation Failures**: Fix common issues automatically
- **Domain Configuration Missing**: Use default settings

## Integration Points

- **Environment Manager**: Called during environment creation
- **Tool Generator**: Provides schema for tool implementation
- **Model Manager**: Uses configured LLMs for generation
- **Config Manager**: Accesses domain and model configurations

## Best Practices

1. **Domain Configuration**: Define domain-specific table suggestions
2. **Schema Validation**: Always validate generated schemas
3. **Relationship Design**: Ensure proper entity relationships
4. **Field Types**: Use appropriate data types for tool compatibility
5. **Documentation**: Include clear descriptions for all components

