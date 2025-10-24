"""
Tool code generator that creates executable Python implementations from API descriptions and database schemas.
"""

import json
import logging
import ast
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_fixed

from worldInteract.utils.camel_generator import generate
from worldInteract.utils.config_manager import config_manager
from worldInteract.utils.parser_utils import extract_python_code_from_text, extract_json_from_text, extract_requirements_from_text


logger = logging.getLogger(__name__)


class ToolGenerator:
    """Generates executable tool implementations from API descriptions and database schemas."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize tool generator.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_manager = config_manager
        self.model_config = self.config_manager.get_model_config("tool_generation")
    
    def generate_tools(
        self, 
        api_collection: Dict[str, Any], 
        schema: Dict[str, Any],
        initial_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, str], List[str]]:
        """
        Generate tool implementations for all tools in the API collection.
        
        Args:
            api_collection: API collection containing tool descriptions
            schema: Database schema for the domain
            initial_state: Initial database state to ensure compatibility
            
        Returns:
            Tuple of (Dictionary mapping tool names to their Python implementations, List of requirements)
        """
        domain = api_collection.get("domain", "unknown")
        tools = api_collection.get("tools", [])
        
        logger.info(f"Generating {len(tools)} tools for domain: {domain}")
        
        generated_tools = {}
        all_requirements = set()
        
        for tool in tools:
            tool_name = tool["name"]
            logger.info(f"Generating tool: {tool_name}")
            
            try:
                tool_code, requirements = self._generate_tool_with_llm(tool, schema, domain, initial_state)
                generated_tools[tool_name] = tool_code
                all_requirements.update(requirements)
                logger.info(f"Successfully generated tool: {tool_name}")
            except Exception as e:
                logger.error(f"Failed to generate tool {tool_name}: {e}")
                raise
        
        logger.info(f"Successfully generated all {len(generated_tools)} tools")
        return generated_tools, list(all_requirements)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2)
    )
    def _generate_tool_with_llm(
        self, 
        tool: Dict[str, Any], 
        schema: Dict[str, Any],
        domain: str,
        initial_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[str]]:
        """
        Generate tool implementation using LLM with retry mechanism.
        
        Args:
            tool: Tool description and parameters
            schema: Database schema
            domain: Domain name
            initial_state: Initial database state for compatibility
            
        Returns:
            Tuple of (Generated Python code for the tool, List of requirements)
        """
        system_prompt = self._create_tool_system_prompt()
        user_prompt = self._create_tool_user_prompt(tool, schema, domain, initial_state)
        
        try:
            thinking_content, answer_text, function_calls = generate(
                model_key=self.model_config["model"],
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.model_config.get("temperature", 0.1),
                max_tokens=self.model_config.get("max_tokens", 6000)
            )
            
            # Extract Python code from the response
            tool_code = extract_python_code_from_text(answer_text)
            
            # Extract requirements from the response
            requirements = extract_requirements_from_text(answer_text)
            
            # Validate the generated code
            self._validate_tool_code(tool_code, tool["name"])

            logger.info(f"Tool code generated: {tool_code}")
            logger.info(f"Requirements extracted: {requirements}")
            
            return tool_code, requirements
            
        except Exception as e:
            logger.error(f"Failed to generate tool {tool['name']}: {e}")
            raise

    def _create_tool_system_prompt(self) -> str:
        """Create system prompt for tool generation."""
        return """You are an expert Python developer specializing in creating tool implementations for database operations.

Your task is to generate a Python function that implements the given tool specification and operates on an in-memory JSON-based database.

Code Requirements:
1. **Function Signature**: Must match the tool's parameter specification exactly
2. **Database Operations**: Operate directly on the provided `data` dictionary (in-memory database)
3. **Error Handling**: Include proper error handling with try-catch blocks
4. **Type Safety**: Use type hints for all parameters and return values
5. **JSON Returns**: Always return JSON-serializable objects (typically strings)
6. **τ-bench Style**: Follow τ-bench patterns for data manipulation
7. **State Tracking**: MUST include `after_execution_state` in the result to show final database state

Function Template:
```python
def tool_name(data: Dict[str, Any], param1: type1, param2: type2, ...) -> str:
    \"\"\"
    Tool description here.
    
    Args:
        data: In-memory database (nested dictionaries)
        param1: Parameter 1 description
        param2: Parameter 2 description
        
    Returns:
        JSON string containing operation result
    \"\"\"
    try:
        # 1. Validate inputs
        # 2. Access/modify data dictionary
        # 3. Perform the required operation
        # 4. Return result as JSON string
        
        result = {
            "success": True,
            "data": {...},
            "message": "Operation completed successfully",
            "after_execution_state": data  # Include final database state
        }
        return json.dumps(result)
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "message": f"Operation failed: {str(e)}",
            "after_execution_state": data  # Include current database state even on error
        }
        return json.dumps(error_result)
```

Important Notes:
- The `data` parameter is always first and contains the entire database state
- Modify `data` directly for write operations (create, update, delete)
- For read operations, access `data` without modification
- Generate unique IDs when creating new records
- Maintain relationships between entities according to the schema
- Always return valid JSON strings
- Include comprehensive error handling
- **CRITICAL**: ALWAYS include `after_execution_state: data` in the result JSON to capture final database state

Requirements Specification:
If your implementation requires external Python packages (beyond standard library), provide them in a JSON format:

```json
["package1", "package2==1.0.0", "package3>=2.0"]
```

Place this JSON block after your Python code implementation."""
    
    def _create_tool_user_prompt(
        self, 
        tool: Dict[str, Any], 
        schema: Dict[str, Any],
        domain: str,
        initial_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create user prompt for tool generation."""
        tool_name = tool["name"]
        tool_description = tool["description"]
        parameters = tool.get("parameters", {})
        returns = tool.get("returns", {})
        
        # Format parameters
        param_info = []
        for param_name, param_def in parameters.items():
            param_type = param_def.get("type", "Any")
            param_desc = param_def.get("description", "")
            default = param_def.get("default")
            default_info = f" (default: {default})" if default is not None else ""
            param_info.append(f"  - {param_name} ({param_type}): {param_desc}{default_info}")
        
        # Format schema information
        schema_info = []
        for table_name, table_def in schema.items():
            table_desc = table_def.get("description", "")
            fields = list(table_def.get("fields", {}).keys())
            schema_info.append(f"  - {table_name}: {table_desc}")
            schema_info.append(f"    Fields: {fields}")
        
        # Format initial state information if provided
        state_info = ""
        if initial_state:
            state_info = "\n**Current Database Structure**:\n"
            for table_name, table_data in initial_state.items():
                if isinstance(table_data, dict):
                    sample_keys = list(table_data.keys())[:3]  # Show first 3 records as examples
                    state_info += f"  - {table_name}: Dict structure with keys like {sample_keys}\n"
                    if sample_keys:
                        # Show one example record structure
                        example_record = table_data[sample_keys[0]]
                        example_fields = list(example_record.keys()) if isinstance(example_record, dict) else "N/A"
                        state_info += f"    Example record fields: {example_fields}\n"
                elif isinstance(table_data, list):
                    state_info += f"  - {table_name}: List structure with {len(table_data)} items\n"
                else:
                    state_info += f"  - {table_name}: {type(table_data).__name__} structure\n"
            
            state_info += "\n**IMPORTANT**: Your code must work with the EXACT data structure shown above.\n"
            state_info += "- If tables are dictionaries, use dict operations (data[table][key] = value)\n"
            state_info += "- If tables are lists, use list operations (data[table].append(value))\n"
        
        return f"""Generate a Python function that implements this tool:

**Tool Name**: {tool_name}
**Description**: {tool_description}
**Domain**: {domain}

**Parameters**:
{chr(10).join(param_info) if param_info else "  No parameters"}

**Expected Returns**: 
{json.dumps(returns, indent=2) if returns else "JSON object with success status and relevant data"}

**Database Schema**:
{chr(10).join(schema_info)}{state_info}

**Requirements**:
1. Implement the function following the exact specification
2. Use the provided database schema to understand data relationships
3. Access and modify the `data` dictionary directly (it represents the in-memory database)
4. Include proper error handling and validation
5. Return results as JSON strings
6. Generate realistic IDs for new records (use timestamp or random components)
7. Maintain data consistency and relationships
8. **CRITICAL**: ALWAYS include `"after_execution_state": data` in the returned JSON to capture the final database state

Generate ONLY the Python function implementation. Do not include imports or additional code."""
    
    def _extract_python_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Look for code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        # If no code blocks, try to find the function
        lines = response.strip().split('\n')
        code_lines = []
        in_function = False
        
        for line in lines:
            if line.strip().startswith('def '):
                in_function = True
            if in_function:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        # Return the entire response if we can't find code blocks
        return response.strip()
    
    def _validate_tool_code(self, code: str, tool_name: str) -> None:
        """
        Validate the generated tool code.
        
        Args:
            code: Generated Python code
            tool_name: Name of the tool
            
        Raises:
            ValueError: If code is invalid
        """
        try:
            # Parse the code to check syntax
            ast.parse(code)
            
            # Check if the function is defined
            if f"def {tool_name}" not in code:
                raise ValueError(f"Function '{tool_name}' not found in generated code")
            
            # Check for required imports
            required_patterns = ["json.dumps", "Dict[str, Any]"]
            missing_patterns = []
            
            for pattern in required_patterns:
                if pattern not in code:
                    missing_patterns.append(pattern)
            
            if missing_patterns:
                logger.warning(f"Code might be missing required patterns: {missing_patterns}")
            
            logger.info(f"Tool code validation passed for: {tool_name}")
            
        except SyntaxError as e:
            logger.error(f"Syntax error in generated code for {tool_name}: {e}")
            raise ValueError(f"Generated code has syntax errors: {e}")
    
    def save_tools(
        self, 
        tools: Dict[str, str], 
        domain: str, 
        output_dir: Optional[str] = None
    ) -> str:
        """
        Save generated tools to file.
        
        Args:
            tools: Dictionary of tool implementations
            domain: Domain name
            output_dir: Output directory (defaults to data/generated_env/domains/)
            
        Returns:
            Path to saved tools file
        """
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            output_dir = project_root / "data" / "generated_env" / "domains" / domain
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual tool files
        tools_dir = output_path / "tools"
        tools_dir.mkdir(exist_ok=True)
        
        for tool_name, tool_code in tools.items():
            tool_file = tools_dir / f"{tool_name}.py"
            
            # Add necessary imports
            full_code = self._add_imports(tool_code)
            
            with open(tool_file, 'w', encoding='utf-8') as f:
                f.write(full_code)
        
        # Save combined tools file
        combined_file = output_path / "tools.py"
        self._save_combined_tools(tools, combined_file)
        
        logger.info(f"Tools saved to: {output_path}")
        return str(output_path)
    
    def _add_imports(self, code: str) -> str:
        """Add necessary imports to tool code."""
        imports = [
            "import json",
            "import uuid",
            "import datetime",
            "from typing import Dict, Any, List, Optional, Tuple",
            ""
        ]
        
        return '\n'.join(imports) + '\n' + code
    
    def _save_combined_tools(self, tools: Dict[str, str], output_file: Path) -> None:
        """Save all tools in a single file."""
        imports = [
            "import json",
            "import uuid", 
            "import datetime",
            "from typing import Dict, Any, List, Optional, Tuple",
            "",
            '"""',
            "Generated tool implementations for domain operations.",
            '"""',
            ""
        ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(imports))
            
            for tool_name, tool_code in tools.items():
                f.write(f"\n\n{tool_code}\n")
        
        logger.info(f"Combined tools saved to: {output_file}")
    
    def load_tools(self, domain: str, tools_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Load existing tools from file.
        
        Args:
            domain: Domain name
            tools_dir: Directory containing tool files
            
        Returns:
            Dictionary of tool implementations
        """
        if tools_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            tools_dir = project_root / "data" / "generated_env" / "domains" / domain / "tools"
        
        tools_path = Path(tools_dir)
        
        if not tools_path.exists():
            raise FileNotFoundError(f"Tools directory not found: {tools_path}")
        
        tools = {}
        
        for tool_file in tools_path.glob("*.py"):
            tool_name = tool_file.stem
            with open(tool_file, 'r', encoding='utf-8') as f:
                tools[tool_name] = f.read()
        
        logger.info(f"Loaded {len(tools)} tools from: {tools_path}")
        return tools

