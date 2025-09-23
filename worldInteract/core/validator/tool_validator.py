"""
Tool validator that tests generated tools and validates their execution results.
"""

import json
import logging
import copy
import traceback
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_fixed

from worldInteract.utils.model_manager import generate
from worldInteract.utils.config_manager import config_manager
from worldInteract.utils.parser_utils import extract_json_from_text


logger = logging.getLogger(__name__)


class ToolValidator:
    """Validates generated tools through execution testing and result verification."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize tool validator.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_manager = config_manager
        self.validation_config = self.config_manager.get_model_config("validation")
        self.test_config = self.config_manager.get_model_config("test_generation")
    
    def validate_tools(
        self,
        tools: Dict[str, str],
        schema: Dict[str, Any],
        initial_state: Dict[str, Any],
        api_collection: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Validate all tools through execution testing.
        
        Args:
            tools: Dictionary of tool implementations
            schema: Database schema
            initial_state: Initial database state
            api_collection: Original API collection with tool descriptions
            
        Returns:
            Dictionary mapping tool names to validation results (True/False)
        """
        domain = api_collection.get("domain", "unknown")
        logger.info(f"Validating {len(tools)} tools for domain: {domain}")
        
        validation_results = {}
        
        for tool_name, tool_code in tools.items():
            logger.info(f"Validating tool: {tool_name}")
            
            try:
                # Find the tool description
                tool_desc = self._find_tool_description(tool_name, api_collection)
                if not tool_desc:
                    logger.error(f"Tool description not found for: {tool_name}")
                    validation_results[tool_name] = False
                    continue
                
                # Validate the tool
                is_valid = self._validate_single_tool(
                    tool_name, tool_code, tool_desc, schema, initial_state
                )
                validation_results[tool_name] = is_valid
                
                logger.info(f"Tool {tool_name} validation: {'PASSED' if is_valid else 'FAILED'}")
                
            except Exception as e:
                logger.error(f"Error validating tool {tool_name}: {e}")
                validation_results[tool_name] = False
        
        passed = sum(1 for result in validation_results.values() if result)
        total = len(validation_results)
        logger.info(f"Validation completed: {passed}/{total} tools passed")
        
        return validation_results
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2)
    )
    def _validate_single_tool(
        self,
        tool_name: str,
        tool_code: str,
        tool_desc: Dict[str, Any],
        schema: Dict[str, Any],
        initial_state: Dict[str, Any]
    ) -> bool:
        """
        Validate a single tool through execution testing.
        
        Args:
            tool_name: Name of the tool
            tool_code: Python implementation of the tool
            tool_desc: Tool description from API collection
            schema: Database schema
            initial_state: Initial database state
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Step 1: Generate test case
            # TODO: using function calling stead of prompting LLM to generate call from content
            test_case = self._generate_test_case(tool_name, tool_desc, schema, initial_state)
            
            # Step 2: Execute the tool
            result, final_state = self._execute_tool(tool_code, test_case, initial_state, tool_name)
            
            # Step 3: Validate the result
            is_valid = self._validate_result(
                tool_name, tool_desc, test_case, result, initial_state, final_state
            )
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Validation failed for {tool_name}: {e}")
            return False
    
    def _generate_test_case(
        self,
        tool_name: str,
        tool_desc: Dict[str, Any],
        schema: Dict[str, Any],
        initial_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a test case for the tool using LLM.
        
        Args:
            tool_name: Name of the tool
            tool_desc: Tool description
            schema: Database schema
            initial_state: Initial database state
            
        Returns:
            Generated test case with parameters and expected behavior
        """
        system_prompt = self._create_test_generation_system_prompt()
        user_prompt = self._create_test_generation_user_prompt(
            tool_name, tool_desc, schema, initial_state
        )
        
        think_content, answer_text, function_calls = generate(
            model_key=self.test_config["model"],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.test_config.get("temperature", 0.6),
            max_tokens=self.test_config.get("max_tokens", 2000)
        )

        # extract ```json``` from answer_text
        json_test_case = extract_json_from_text(answer_text)
        
        try:
            test_case = json.loads(json_test_case.strip())
            logger.info(f"Test case generated: {json.dumps(test_case, indent=2)}")
            return test_case
        except json.JSONDecodeError:
            # Fallback: create a simple test case
            logger.info(f"Failed to parse test case JSON: {json_test_case}")
            return self._create_fallback_test_case(tool_name, tool_desc)
    
    def _create_test_generation_system_prompt(self) -> str:
        """Create system prompt for test case generation."""
        return """You are a test case generator for tool validation.

Your task is to generate realistic test parameters for a given tool that can be used to validate its implementation.

Requirements:
1. **Realistic Parameters**: Generate parameters that reflect real-world usage
2. **Schema Compliance**: Ensure parameters work with the provided database schema
3. **Expected Behavior**: Describe what the tool should do with these parameters
4. **Edge Cases**: Consider boundary conditions when appropriate

Output Format:
Return ONLY a valid JSON object with this structure:
```json
{
  "parameters": {
    "param_name": "param_value",
    ...
  },
  "expected_behavior": {
    "type": "read|write|mixed",
    "description": "What the tool should accomplish",
    "should_succeed": true|false,
    "expected_changes": "Description of expected database changes (for write operations)"
  },
  "test_description": "Brief description of what this test validates"
}
```

Make the test case realistic and appropriate for the tool's functionality."""
    
    def _create_test_generation_user_prompt(
        self,
        tool_name: str,
        tool_desc: Dict[str, Any],
        schema: Dict[str, Any],
        initial_state: Dict[str, Any]
    ) -> str:
        """Create user prompt for test case generation."""
        parameters = tool_desc.get("parameters", {})
        
        # Format parameter information
        param_info = []
        for param_name, param_def in parameters.items():
            param_type = param_def.get("type", "Any")
            param_desc = param_def.get("description", "")
            param_info.append(f"  - {param_name} ({param_type}): {param_desc}")
        
        # Sample some data from initial state
        state_sample = {}
        for table_name, table_data in initial_state.items():
            if isinstance(table_data, dict) and table_data:
                # Get first few records as samples
                sample_keys = list(table_data.keys())[:3]
                state_sample[table_name] = {k: table_data[k] for k in sample_keys}
        
        return f"""Generate a test case for this tool:

**Tool**: {tool_name}
**Description**: {tool_desc['description']}

**Parameters**:
{chr(10).join(param_info) if param_info else "  No parameters"}

**Database Schema**: {json.dumps(schema, indent=2)}

**Sample Current Data**: {json.dumps(state_sample, indent=2)}

Generate realistic test parameters that:
1. Work with the current database state
2. Test the tool's core functionality
3. Are appropriate for the tool's purpose
4. Include valid IDs/references where needed

Provide a complete test case following the specified JSON format."""
    
    def _create_fallback_test_case(self, tool_name: str, tool_desc: Dict[str, Any]) -> Dict[str, Any]:
        """Create a simple fallback test case."""
        parameters = tool_desc.get("parameters", {})
        
        # Generate simple test parameters
        test_params = {}
        for param_name, param_def in parameters.items():
            param_type = param_def.get("type", "string")
            if param_type == "string":
                test_params[param_name] = f"test_{param_name}"
            elif param_type == "integer":
                test_params[param_name] = 1
            elif param_type == "boolean":
                test_params[param_name] = True
            else:
                test_params[param_name] = f"test_value_{param_name}"
        
        return {
            "parameters": test_params,
            "expected_behavior": {
                "type": "mixed",
                "description": f"Test basic functionality of {tool_name}",
                "should_succeed": True,
                "expected_changes": "Tool should execute without errors"
            },
            "test_description": f"Basic functionality test for {tool_name}"
        }
    
    def _execute_tool(
        self,
        tool_code: str,
        test_case: Dict[str, Any],
        initial_state: Dict[str, Any],
        tool_name: str = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Execute the tool with the test case.
        
        Args:
            tool_code: Python implementation of the tool
            test_case: Test case with parameters
            initial_state: Initial database state
            tool_name: Name of the tool (optional, used for function detection)
            
        Returns:
            Tuple of (result, final_state)
        """
        # Create a copy of the initial state for testing
        test_state = copy.deepcopy(initial_state)
        
        try:
            # Prepare the execution environment
            exec_globals = {
                'json': json,
                'uuid': __import__('uuid'),
                'datetime': __import__('datetime'),
                'Dict': Dict,
                'Any': Any,
                'List': List,
                'Optional': Optional
            }
            
            # Execute the tool code to define the function
            exec(tool_code, exec_globals)
            
            # Find the function name (should match the tool name)
            func_name = None
            excluded_names = ['json', 'uuid', 'datetime', 'Dict', 'Any', 'List', 'Optional']
            
            # First, try to find a function that matches the expected tool name (if provided)
            if tool_name and tool_name in exec_globals:
                obj = exec_globals[tool_name]
                if (callable(obj) and hasattr(obj, '__code__')):
                    func_name = tool_name
                    logger.info(f"Found exact tool function match: {func_name}")
            
            # If no exact match, find any user-defined function
            if not func_name:
                for name, obj in exec_globals.items():
                    if (callable(obj) and 
                        not name.startswith('_') and 
                        name not in excluded_names and
                        hasattr(obj, '__code__')):  # Ensure it's an actual function, not a class or type
                        func_name = name
                        logger.info(f"Found function: {func_name}")
                        break
            
            if not func_name:
                # Debug information
                logger.error(f"Failed to find function for tool: {tool_name or 'unknown'}")
                logger.error("Available callables in exec_globals:")
                for name, obj in exec_globals.items():
                    if callable(obj):
                        logger.error(f"  - {name}: {type(obj)} (has __code__: {hasattr(obj, '__code__')})")
                raise ValueError(f"No callable function found in tool code for tool: {tool_name or 'unknown'}")
            
            tool_function = exec_globals[func_name]
            
            # Prepare parameters
            test_params = test_case.get("parameters", {})
            
            # Execute the tool
            result = tool_function(test_state, **test_params)
            
            logger.info(f"Tool execution result: {result}")
            
            return result, test_state
            
        except Exception as e:
            logger.error(f"Tool execution error for {tool_name or 'unknown'}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"EXECUTION_ERROR: {str(e)}", test_state
    
    def _validate_result(
        self,
        tool_name: str,
        tool_desc: Dict[str, Any],
        test_case: Dict[str, Any],
        result: str,
        initial_state: Dict[str, Any],
        final_state: Dict[str, Any]
    ) -> bool:
        """
        Validate the tool execution result using LLM.
        
        Args:
            tool_name: Name of the tool
            tool_desc: Tool description
            test_case: Test case used
            result: Tool execution result
            initial_state: Database state before execution
            final_state: Database state after execution
            
        Returns:
            True if result is valid, False otherwise
        """
        # Check for execution errors first
        if result.startswith("EXECUTION_ERROR"):
            logger.error(f"Tool {tool_name} had execution error: {result}")
            return False
        
        # Try to parse result as JSON
        try:
            json.loads(result)
        except json.JSONDecodeError:
            logger.error(f"Tool {tool_name} returned invalid JSON: {result}")
            return False
        
        # Use LLM to validate the result
        system_prompt = self._create_validation_system_prompt()
        user_prompt = self._create_validation_user_prompt(
            tool_name, tool_desc, test_case, result, initial_state, final_state
        )
        
        try:
            think_content, answer_text, function_calls = generate(
                model_key=self.validation_config["model"],
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.validation_config.get("temperature", 0.5),
                max_tokens=self.validation_config.get("max_tokens", 2000)
            )
            
            # Parse the validation response
            validation_result = answer_text.strip().lower()
            return "valid" in validation_result and "invalid" not in validation_result
            
        except Exception as e:
            logger.error(f"Validation error for {tool_name}: {e}")
            return False
    
    def _create_validation_system_prompt(self) -> str:
        """Create system prompt for result validation."""
        return """You are a tool validation expert responsible for verifying that tool execution results are correct.

Your task is to analyze the tool execution and determine if the result is valid based on:
1. **Tool Specification**: Does the result match what the tool is supposed to do?
2. **Parameter Inputs**: Are the results appropriate for the given inputs?
3. **Database Changes**: For write operations, were the database changes correct?
4. **Result Format**: Is the result properly formatted and complete?
5. **Error Handling**: If the operation should fail, did it fail appropriately?

Analysis Framework:
- Check if the tool behavior matches its description
- Verify that database modifications are logically correct
- Ensure the result JSON contains expected fields
- Validate that relationships and constraints are maintained

Output Format:
Respond with one word: "VALID" or "INVALID"
Then provide a brief explanation of your reasoning."""
    
    def _create_validation_user_prompt(
        self,
        tool_name: str,
        tool_desc: Dict[str, Any],
        test_case: Dict[str, Any],
        result: str,
        initial_state: Dict[str, Any],
        final_state: Dict[str, Any]
    ) -> str:
        """Create user prompt for result validation."""
        expected_behavior = test_case.get("expected_behavior", {})
        
        # Calculate state differences
        state_diff = self._calculate_state_diff(initial_state, final_state)
        
        return f"""Validate this tool execution:

**Tool**: {tool_name}
**Description**: {tool_desc['description']}

**Test Case**:
- Parameters: {json.dumps(test_case.get('parameters', {}), indent=2)}
- Expected Behavior: {json.dumps(expected_behavior, indent=2)}

**Execution Result**: 
{result}

**Database State Changes**:
{json.dumps(state_diff, indent=2) if state_diff else "No changes detected"}

**Analysis Required**:
1. Does the result match the tool's intended functionality?
2. Are the database changes (if any) correct and appropriate?
3. Is the result format valid and complete?
4. Did the tool handle the test parameters correctly?

Provide your validation decision and reasoning."""
    
    def _calculate_state_diff(self, initial_state: Dict[str, Any], final_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate differences between initial and final database states."""
        differences = {}
        
        for table_name in set(initial_state.keys()) | set(final_state.keys()):
            initial_table = initial_state.get(table_name, {})
            final_table = final_state.get(table_name, {})
            
            if initial_table != final_table:
                table_diff = {
                    "added": {},
                    "modified": {},
                    "removed": {}
                }
                
                # Find added records
                for key in final_table:
                    if key not in initial_table:
                        table_diff["added"][key] = final_table[key]
                
                # Find removed records
                for key in initial_table:
                    if key not in final_table:
                        table_diff["removed"][key] = initial_table[key]
                
                # Find modified records
                for key in initial_table:
                    if key in final_table and initial_table[key] != final_table[key]:
                        table_diff["modified"][key] = {
                            "before": initial_table[key],
                            "after": final_table[key]
                        }
                
                if any(table_diff.values()):
                    differences[table_name] = table_diff
        
        return differences
    
    def _find_tool_description(self, tool_name: str, api_collection: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find tool description in API collection."""
        tools = api_collection.get("tools", [])
        for tool in tools:
            if tool.get("name") == tool_name:
                return tool
        return None
    
    def save_validation_report(
        self,
        validation_results: Dict[str, bool],
        domain: str,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Save validation report to file.
        
        Args:
            validation_results: Validation results for each tool
            domain: Domain name
            output_dir: Output directory
            
        Returns:
            Path to saved report
        """
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            output_dir = project_root / "data" / "generated" / "domains" / domain
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / "validation_report.json"
        
        report = {
            "domain": domain,
            "validation_results": validation_results,
            "summary": {
                "total_tools": len(validation_results),
                "passed": sum(1 for result in validation_results.values() if result),
                "failed": sum(1 for result in validation_results.values() if not result)
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Validation report saved to: {report_file}")
        return str(report_file)

