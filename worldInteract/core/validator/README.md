# Tool Validator

The Tool Validator tests generated tools and validates their execution results using LLM-powered verification.

## Overview

The Tool Validator ensures that generated tools work correctly by creating test cases, executing the tools, and validating the results. It uses a multi-stage validation approach with automatic retry mechanisms to guarantee tool quality.

## Validation Pipeline

```
Test Generation → Tool Execution → Result Validation → Report Generation
```

### 1. **Test Case Generation**
- LLM generates realistic test parameters
- Considers database schema and current state
- Creates both positive and edge case scenarios

### 2. **Tool Execution**
- Executes tools in isolated environments
- Captures both results and state changes
- Handles execution errors gracefully

### 3. **Result Validation**
- LLM evaluates execution results
- Checks database state consistency
- Validates against expected behavior

### 4. **Report Generation**
- Creates comprehensive validation reports
- Tracks pass/fail status for each tool
- Provides debugging information

## Usage Example

```python
from worldInteract.core.validator.code_agent import CodeAgent

# Initialize code agent (handles both generation and validation)
code_agent = CodeAgent()

# Generate and validate tools in one step
tools, requirements, validation_results = code_agent.generate_and_validate_tools(
    api_collection=api_collection,
    schema=schema,
    initial_state=initial_state
)

# Check results
for tool_name, passed in validation_results.items():
    print(f"{tool_name}: {'PASSED' if passed else 'FAILED'}")

## Validation Process

### 🧪 **Test Case Generation**

The validator creates realistic test cases using LLM:

```json
{
  "parameters": {
    "file_path": "/documents/test.txt",
    "content": "Hello, World!",
    "encoding": "utf-8"
  },
  "expected_behavior": {
    "type": "write",
    "description": "Create a new file with the specified content",
    "should_succeed": true,
    "expected_changes": "New file record added to files table"
  },
  "test_description": "Basic file creation functionality"
}
```

### 🏃 **Tool Execution**

Tools are executed in isolated environments:

```python
# Create copy of initial state
test_state = copy.deepcopy(initial_state)

# Execute tool with test parameters
result = tool_function(test_state, **test_parameters)

# Capture both result and final state
return result, test_state
```

### ✅ **Result Validation**

LLM validates execution results by analyzing:
- **Tool Specification Compliance**: Does result match tool description?
- **Parameter Handling**: Are inputs processed correctly?
- **Database Changes**: Are state modifications appropriate?
- **Error Handling**: Do failures occur when expected?

## Configuration

Validation is configured through `config/model_config.yaml`:

```yaml
validation:
  model: "qwen3_32b"
  temperature: 0.5
  max_tokens: 2000
  retry_attempts: 3

test_generation:
  model: "gemini_2d5_pro"
  temperature: 0.6
  max_tokens: 2000
  retry_attempts: 3
```

Environment settings in `config/environment_config.yaml`:

```yaml
validation:
  run_execution_tests: true
  check_state_consistency: true
  validate_return_types: true
  max_test_cases: 5
```

## Key Features

### 🎯 **Comprehensive Testing**
- Realistic test case generation
- Multiple test scenarios per tool
- Edge case consideration

### 🔄 **Automatic Retry**
- Up to 3 retry attempts for failed validations
- Tenacity-based retry mechanism
- Progressive improvement strategies

### 🧠 **LLM-Powered Validation**
- Intelligent result analysis
- Context-aware error detection
- Natural language reasoning about correctness

### 📊 **Detailed Reporting**
- Pass/fail status for each tool
- Execution error details
- State change analysis
- Debugging information

## Validation Criteria

### ✅ **Success Criteria**
- Tool executes without errors
- Returns valid JSON format
- Database state changes are appropriate
- Result matches expected behavior
- Maintains data consistency

### ❌ **Failure Criteria**
- Execution throws unhandled exceptions
- Returns invalid or malformed JSON
- Database state becomes inconsistent
- Result doesn't match tool specification
- Violates schema constraints

## Generated Reports

Validation reports provide comprehensive analysis:

```json
{
  "domain": "file_operations",
  "validation_results": {
    "create_file": true,
    "read_file": true,
    "delete_file": false,
    "list_directory": true
  },
  "summary": {
    "total_tools": 4,
    "passed": 3,
    "failed": 1
  },
  "details": {
    "delete_file": {
      "error": "Database state inconsistency",
      "test_case": {...},
      "execution_result": {...}
    }
  }
}
```

## Error Analysis

### 📋 **Common Issues**

1. **Syntax Errors**: Invalid Python code generation
2. **Type Mismatches**: Parameter type inconsistencies
3. **Schema Violations**: Improper database access patterns
4. **Logic Errors**: Incorrect business logic implementation

### 🔧 **Debug Information**

The validator provides detailed debugging information:
- **Test Parameters**: What inputs were used
- **Execution Trace**: Step-by-step execution details
- **State Differences**: Before/after database comparisons
- **Error Messages**: Specific failure descriptions

## Integration Points

- **Environment Manager**: Called during environment creation
- **Tool Generator**: Validates generated tool implementations
- **Model Manager**: Uses configured LLMs for validation
- **Config Manager**: Accesses validation configuration

## Advanced Features

### 🔍 **State Difference Analysis**

The validator tracks database state changes:

```python
{
  "files": {
    "added": {
      "file_123": {"name": "test.txt", "content": "..."}
    },
    "modified": {
      "file_456": {
        "before": {"size": 100},
        "after": {"size": 200}
      }
    },
    "removed": {}
  }
}
```

### 🎲 **Fallback Test Cases**

When LLM test generation fails, the validator creates simple fallback tests:

```python
def _create_fallback_test_case(self, tool_name: str, tool_desc: Dict[str, Any]):
    # Generate simple test parameters based on parameter types
    # Ensure basic functionality can be tested
```

### 🔒 **Execution Isolation**

Each tool execution is isolated:
- Deep copy of database state
- Separate execution environment
- No cross-contamination between tests

## Best Practices

1. **Test Coverage**: Ensure all tool operations are tested
2. **Realistic Data**: Use realistic test parameters and data
3. **Error Scenarios**: Test both success and failure cases
4. **State Validation**: Always check database state consistency
5. **Comprehensive Reporting**: Generate detailed validation reports

## Troubleshooting

### 🐛 **Common Validation Failures**

1. **Import Errors**: Missing required imports in generated code
2. **Schema Misalignment**: Tools don't match database schema
3. **Type Issues**: Parameter types don't match expectations
4. **Logic Errors**: Incorrect business logic implementation

### 🔧 **Debug Strategies**

1. **Check Test Cases**: Verify generated test parameters are valid
2. **Manual Execution**: Run tools manually with known inputs
3. **State Inspection**: Compare before/after database states
4. **Log Analysis**: Review detailed execution logs
5. **Schema Verification**: Ensure schema matches tool requirements

## Performance Considerations

- **Parallel Validation**: Future enhancement for concurrent testing
- **Caching**: Reuse successful test cases where appropriate
- **Timeout Handling**: Prevent infinite execution loops
- **Memory Management**: Clean up test states after validation

