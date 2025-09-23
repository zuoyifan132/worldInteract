# Environment Manager

The Environment Manager orchestrates the complete environment construction pipeline, from API collection input to fully validated environment output.

## Overview

The Environment Manager is the main orchestrator that coordinates all other components to create a complete, testable environment for agentic tasks. It follows the pipeline:

1. **API Collection Loading**: Load domain-specific tool collections
2. **Schema Generation**: Create database schemas using LLM
3. **State Generation**: Generate realistic initial database states
4. **Tool Generation**: Create executable tool implementations
5. **Validation**: Test and validate all generated components
6. **Storage**: Save all components for future use

## Architecture

```
API Collection → Schema → Initial State → Tools → Validation → Environment
```

## Key Components

### EnvironmentManager

The main class that orchestrates the entire pipeline.

**Key Methods:**
- `create_environment()`: Complete environment creation from API collection
- `load_api_collection()`: Load API collection from file
- `generate_initial_state()`: Generate realistic database states
- `load_environment()`: Load existing environments

## Usage Example

```python
from worldInteract.core.environment import EnvironmentManager

# Initialize environment manager
env_manager = EnvironmentManager()

# Create complete environment from API collection
# Output directory is automatically determined from the "domain" field in the API collection
environment = env_manager.create_environment(
    api_collection_path="data/apis_collections/file_operations.json",
    validate_tools=True
)

# Access generated components
schema = environment["schema"]
initial_state = environment["initial_state"]
tools = environment["tools"]
validation_results = environment["validation_results"]
```

## Configuration

Environment generation can be configured through `config/model_config.yaml`:

```yaml
state_generation:
  model: "qwen3_32b"
  temperature: 0.4
  max_tokens: 3000
```

## Features

### 🔄 **Complete Pipeline**
- Automated end-to-end environment construction
- Integrated validation and regeneration
- Comprehensive error handling

### 🛠️ **Tool Regeneration**
- Automatic retry for failed tool validations
- Up to 3 regeneration attempts per tool
- Intelligent failure analysis

### 📁 **Environment Persistence**
- Save complete environments to disk
- Load existing environments for reuse
- Structured file organization

### 🔍 **Quality Assurance**
- Integrated validation pipeline
- Schema compliance checking
- Tool execution testing

## Generated Environment Structure

The output directory is **automatically determined** from the `domain` field in the API collection:

```
data/generated/domains/{domain}/    # Domain from API collection
├── schema.json                     # Database schema
├── initial_state.json              # Initial database state
├── tools.py                        # Combined tool implementations
├── tools/                          # Individual tool files
│   ├── create_file.py
│   ├── read_file.py
│   └── ...
├── validation_report.json          # Validation results
└── environment_metadata.json       # Environment metadata
```

### Automatic Directory Creation

- **Domain Extraction**: Reads the `"domain"` field from the API collection JSON
- **Path Generation**: Creates `data/generated/domains/{domain}/` automatically
- **No Manual Paths**: No need to specify output directories manually
- **Consistent Structure**: All domains follow the same directory structure

## Error Handling

The Environment Manager includes comprehensive error handling:

- **Schema Generation Failures**: Retry with different prompts
- **Tool Generation Errors**: Regenerate failed tools up to 3 times
- **Validation Failures**: Automatic retry and regeneration
- **File I/O Errors**: Graceful handling with informative messages

## Logging

All operations are logged with appropriate levels:
- `INFO`: Pipeline progress and major milestones
- `WARNING`: Non-fatal issues and fallback operations
- `ERROR`: Failures that require attention
- `DEBUG`: Detailed operation information

## Integration

The Environment Manager integrates with:
- **Schema Generator**: For database schema creation
- **Tool Generator**: For tool implementation generation
- **Validator**: For quality assurance and testing
- **Model Manager**: For LLM interactions
- **Config Manager**: For configuration management

