"""
Tool code generator that creates executable Python implementations from API descriptions and database schemas.
"""

import json
import logging
import ast
import re
from textwrap import dedent
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_fixed

from worldInteract.utils.camel_generator import generate
from worldInteract.utils.config_manager import config_manager
from worldInteract.utils.parser_utils import extract_python_code_from_text, extract_json_from_text, extract_requirements_from_text
from worldInteract.core.sandbox import CodeExecutor
from worldInteract.agents import ReactAgent


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
        return dedent(
            """You are an expert Python developer specializing in creating tool implementations for database operations.

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

            Place this JSON block after your Python code implementation.""")
    
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
        
        return dedent(
            f"""Generate a Python function that implements this tool:

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

            Generate ONLY the Python function implementation. Do not include imports or additional code.""")
    
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
        """
        Save all tools in a single file with consolidated imports.
        
        Extracts imports from individual tool codes, deduplicates them,
        and places them once at the top of the file.
        """
        # Collect all unique imports from tool codes
        all_imports = set()
        tool_functions = {}
        
        for tool_name, tool_code in tools.items():
            # Split code into lines
            lines = tool_code.split('\n')
            imports_lines = []
            function_lines = []
            
            in_imports = True
            for line in lines:
                stripped = line.strip()
                # Check if this is an import line
                if in_imports and (stripped.startswith('import ') or 
                                  stripped.startswith('from ') or
                                  stripped.startswith('#') or
                                  stripped == ''):
                    if stripped.startswith('import ') or stripped.startswith('from '):
                        all_imports.add(stripped)
                    # Skip comments and empty lines in import section
                else:
                    in_imports = False
                    function_lines.append(line)
            
            # Store the function code without imports
            tool_functions[tool_name] = '\n'.join(function_lines).strip()
        
        # Sort imports for consistency (alphabetically)
        sorted_imports = sorted(all_imports)
        
        # Build the file header with only the imports found in tool codes
        header = sorted_imports + [
            "",
            '"""',
            "Generated tool implementations for domain operations.",
            '"""',
            ""
        ]
        
        # Write the combined file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(header))
            
            for tool_name, function_code in tool_functions.items():
                f.write(f"\n\n{function_code}\n")
        
        logger.info(f"Combined tools saved to: {output_file} (consolidated {len(all_imports)} unique imports)")
    
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

    def refine_tools(
        self,
        tools: Dict[str, str],
        api_collection: Dict[str, Any],
        schema: Dict[str, Any],
        initial_state: Dict[str, Any],
        test_cases: Dict[str, List[Dict[str, Any]]],
        requirements: Optional[List[str]] = None
    ) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        """
        Refine all tools for consistency across the tool suite.
        
        This method takes individually generated tools and refines them to ensure:
        - Consistent parameter naming (first parameter always 'data')
        - Consistent data access patterns
        - Consistent timestamp formatting
        - Consistent error handling
        - Consistent ID generation
        - Consistent return value formats
        
        Args:
            tools: Dictionary mapping tool names to their code implementations
            api_collection: API collection containing tool descriptions
            schema: Database schema
            initial_state: Initial database state
            test_cases: Dictionary mapping tool names to their test cases
            requirements: Optional list of existing requirements
            
        Returns:
            Tuple of (refined_tools, refinement_changes) where:
                - refined_tools: Dictionary of refined tool implementations
                - refinement_changes: Dictionary mapping tool names to list of changes made
        """
        domain = api_collection.get("domain", "unknown")
        logger.info(f"Starting tool refinement with ReAct agent for domain: {domain} ({len(tools)} tools)")
        
        # Initialize code executor for validation
        code_executor = CodeExecutor()
        
        # Get refinement config
        refinement_config = self.config_manager.get_model_config("tool_refinement")
        env_config = self.config_manager.get_environment_config("code_agent")
        max_rounds = env_config.get("max_code_agent_rounds", 10)
        
        # Initialize ReAct agent for refinement
        agent = ReactAgent(config_key="tool_refinement")
        
        # Set up system prompt for refinement with ReAct pattern
        system_prompt = self._create_refinement_react_system_prompt()
        agent.set_system_prompt(system_prompt)
        
        # Create initial user prompt
        initial_user_prompt = self._create_refinement_user_prompt(
            tools, api_collection, schema, initial_state, test_cases, requirements
        )
        agent.add_user_message(initial_user_prompt)
        
        current_refined_tools = None
        refinement_changes = {}
        rounds = 0
        
        # ReAct loop: Iterative refinement based on test feedback
        while rounds < max_rounds:
            rounds += 1
            round_description = "Initial refinement" if rounds == 1 else f"Round {rounds-1} iteration"
            logger.info(f"Refinement {round_description} ({rounds}/{max_rounds})")
            
            try:
                # 1. Get agent response (refined tools)
                thinking, answer_text, function_calls = agent.step()
                
                logger.info(f"ReAct agent response length: {len(answer_text)} chars")
                if thinking:
                    logger.debug(f"Agent thinking: {thinking[:200]}...")
                
                # 2. Extract refined tools from agent response
                current_refined_tools, refinement_changes = self._extract_refined_tools(
                    answer_text, list(tools.keys())
                )
                
                if not current_refined_tools:
                    logger.warning(f"Failed to extract refined tools in round {rounds}")
                    agent.add_observation(
                        f"Error: Could not extract refined tools from your response. "
                        f"Please provide all tools in the specified format with # TOOL: markers."
                    )
                    continue
                
                # 3. Validate all refined tools against test cases
                logger.info("Validating refined tools against test cases...")
                validation_results = {}
                failed_tools_details = []
                
                for tool_name, tool_code in current_refined_tools.items():
                    if tool_name not in test_cases:
                        logger.warning(f"No test cases found for tool: {tool_name}")
                        validation_results[tool_name] = (True, "No test cases to validate")
                        continue
                    
                    tool_test_cases = test_cases[tool_name]
                    if not tool_test_cases:
                        logger.warning(f"Empty test cases for tool: {tool_name}")
                        validation_results[tool_name] = (True, "Empty test cases")
                        continue
                    
                    # Execute test cases
                    success, message, test_results = code_executor.execute_code(
                        tool_code, requirements or [], tool_test_cases, initial_state, tool_name
                    )
                    
                    validation_results[tool_name] = (success, message)
                    
                    if not success:
                        failed_tools_details.append({
                            "tool_name": tool_name,
                            "message": message,
                            "test_results": test_results
                        })
                        logger.warning(f"Tool {tool_name} failed validation: {message}")
                
                # 4. Check if all tools passed
                all_passed = all(success for success, _ in validation_results.values())
                
                if all_passed:
                    logger.info("All refined tools passed validation!")
                    agent.reset()
                    return current_refined_tools, refinement_changes
                
                # 5. Format test failures as observation
                observation = self._format_refinement_test_results(
                    validation_results, failed_tools_details, rounds
                )
                
                # 6. Add observation for next iteration
                observation_title = f"Round {rounds} Test Results"
                agent.add_observation(f"{observation_title}:\n{observation}")
                
                logger.info(
                    f"{round_description} - "
                    f"Passed: {sum(1 for s, _ in validation_results.values() if s)}/{len(validation_results)}"
                )
                    
            except Exception as e:
                logger.error(f"Error in refinement round {rounds}: {e}")
                import traceback
                traceback.print_exc()
                # Add error as observation and continue
                agent.add_observation(
                    f"Error occurred during refinement: {str(e)}. "
                    f"Please analyze the issue and provide corrected refined tools."
                )
        
        # Max rounds reached without success
        logger.warning(f"Tool refinement failed after {max_rounds} rounds")
        agent.reset()
        
        if current_refined_tools:
            logger.warning("Returning last attempt of refined tools (may not pass all tests)")
            return current_refined_tools, refinement_changes
        else:
            logger.error("No refined tools generated, using original tools")
            return tools, {}
    
    def _create_refinement_react_system_prompt(self) -> str:
        """Create ReAct-style system prompt for iterative tool refinement."""
        return dedent(
            """You are an expert Python code refactoring ReAct agent. Your task is to iteratively refine a suite of tool implementations based on test feedback until all tests pass.

            ## Your ReAct workflow:
            You will receive a suite of tools that need consistency improvements. Then:
            - Step 1. **Analyze**: Review the current tools and identify inconsistencies
            - Step 2. **Refine**: Generate improved versions of ALL tools with consistent patterns
            - Step 3. **Receive Feedback**: Get test execution results showing which tools passed/failed
            - Step 4. **Iterate**: If tests fail, analyze failures and provide corrected tools
            - Step 5. **Repeat** until all tests pass or max rounds reached

            ## Response Choices:

            **Always provide ALL refined tools in this exact format:**
            ```python
            # ALL imports must be at the top
            import json
            import uuid
            import datetime
            from typing import Dict, Any, List, Optional
            # ... any other imports needed

            # TOOL: tool_name_1
            def tool_name_1(data: Dict[str, Any], ...) -> str:
                \"\"\"Tool description.\"\"\"
                # Implementation (do NOT repeat imports here)
            
            # TOOL: tool_name_2
            def tool_name_2(data: Dict[str, Any], ...) -> str:
                \"\"\"Tool description.\"\"\"
                # Implementation (do NOT repeat imports here)
            
            # ... all other tools
            ```

            **Then provide a summary of changes:**
            ```json
            {
                "tool_name_1": [
                    "Changed first parameter from 'current_state' to 'data'",
                    "Standardized timestamp format to isoformat()",
                    "Unified record lookup pattern"
                ],
                "tool_name_2": [
                    "Fixed error handling to match test expectations"
                ]
            }
            ```

            ## Consistency Requirements You Must Enforce:

            ### Parameter Naming
            **First parameter MUST always be named `data`** (not `current_state`, `database`, etc.)

            ### Data Access Patterns
            **Standardize how records are found in tables** - Choose ONE consistent pattern

            ### Error Handling
            **Consistent error response format**

            ### Success Response Format
            **Consistent success response format**

            ## Critical Rules:
            - **MUST** place ALL import statements at the very beginning of the code block
            - Do NOT repeat import statements within individual tool functions
            - Do NOT change the business logic unless it causes test failures
            - DO fix any code that causes test failures
            - DO ensure all tools follow the same patterns
            - DO maintain proper type hints
            - DO keep comprehensive error handling
            - DO follow PEP 8 code style guidelines
            
            ## When You Receive Test Feedback:
            - Carefully analyze which tests failed and why
            - Identify the root cause (logic error, inconsistent format, etc.)
            - Fix ONLY the failing tools while maintaining consistency
            - If a pattern causes failures, adjust ALL tools to a working pattern
            - Always provide ALL tools in your response, not just the fixed ones"""
        )
    
    def _create_refinement_system_prompt(self) -> str:
        """Create system prompt for tool refinement."""
        return dedent(
            """You are an expert Python code refactoring specialist. Your task is to refine a suite of tool implementations to ensure consistency and quality across all tools.

            ### Parameter Naming
            **First parameter MUST always be named `data`** (not `current_state`, `database`, etc.)
            ```python
            def tool_name(data: Dict[str, Any], param1: type1, ...) -> str:
            ```

            ### Data Access Patterns
            **Standardize how records are found in tables**
            - If initial state shows dict structure with record IDs as keys: use dict access
            - Choose ONE consistent pattern for finding records (prefer iterating when IDs are dynamic)
            ```python
            # Good: Consistent iteration pattern
            for record_id, record in data['table'].items():
                if record['field'] == search_value:
                    target_record = record
                    break
            ```

            ### Error Handling
            **Consistent error response format**
            ```python
            return json.dumps({
                "success": False,
                "error": "Clear error message"
            })
            ```

            ### Success Response Format
            **Consistent success response format**
            ```python
            return json.dumps({
                "success": True,
                "data": result_data,
                "message": "Optional success message"
            })
            ```

            ## Output Format
            Provide your refined tools in this exact format:

            **IMPORTANT**: Place ALL import statements at the very beginning, before any tool definitions.

            ```python
            # ALL imports must be at the top
            import json
            import uuid
            import datetime
            from typing import Dict, Any, List, Optional
            # ... any other imports needed

            # TOOL: tool_name_1
            def tool_name_1(data: Dict[str, Any], ...) -> str:
                \"\"\"Tool description.\"\"\"
                # Implementation (do NOT repeat imports here)
            
            # TOOL: tool_name_2
            def tool_name_2(data: Dict[str, Any], ...) -> str:
                \"\"\"Tool description.\"\"\"
                # Implementation (do NOT repeat imports here)
            
            # ... all other tools
            ```

            Then provide a summary of changes in JSON format:
            ```json
            {
                "tool_name_1": [
                    "Changed first parameter from 'current_state' to 'data'",
                    "Standardized timestamp format to isoformat()",
                    "Unified record lookup pattern"
                ],
                "tool_name_2": [
                    "Changed parameter handling from kwargs to direct parameters",
                    "Standardized error response format"
                ]
            }
            ```

            ## Critical Rules
            - **MUST** place ALL import statements at the very beginning of the code block
            - Do NOT repeat import statements within individual tool functions
            - Do NOT change the business logic of any tool
            - Do NOT modify function signatures beyond parameter renaming for consistency
            - Do NOT break compatibility with existing test cases
            - DO ensure all tools follow the same patterns
            - DO maintain proper type hints
            - DO keep comprehensive error handling
            - DO follow PEP 8 code style guidelines"""
        )
    
    def _create_refinement_user_prompt(
        self,
        tools: Dict[str, str],
        api_collection: Dict[str, Any],
        schema: Dict[str, Any],
        initial_state: Dict[str, Any],
        test_cases: Dict[str, List[Dict[str, Any]]],
        requirements: Optional[List[str]] = None
    ) -> str:
        """Create user prompt for tool refinement."""
        domain = api_collection.get("domain", "unknown")
        
        # Format all current tool implementations
        tools_text = []
        for tool_name, tool_code in tools.items():
            tools_text.append(f"# TOOL: {tool_name}")
            tools_text.append(tool_code)
            tools_text.append("")
        
        # Format test cases summary (just counts, not full details)
        test_summary = []
        for tool_name, cases in test_cases.items():
            test_summary.append(f"  - {tool_name}\n    - {cases}")
        
        # Format schema summary
        schema_summary = []
        for table_name, table_def in schema.items():
            fields = list(table_def.get("fields", {}).keys())
            schema_summary.append(f"  - {table_name}: {fields}")
        
        # Analyze initial state structure
        state_structure = []
        for table_name, table_data in initial_state.items():
            if isinstance(table_data, dict) and table_data:
                sample_key = list(table_data.keys())[0]
                sample_record = table_data[sample_key]
                if isinstance(sample_record, dict):
                    fields = list(sample_record.keys())
                    state_structure.append(f"  - {table_name}: Dict[str, Dict] with record keys like '{sample_key}'")
                    state_structure.append(f"    Record fields: {fields}")
            elif isinstance(table_data, list):
                state_structure.append(f"  - {table_name}: List structure")
        
        return dedent(
            f"""Please refine this suite of tool implementations for consistency.

            **Domain**: {domain}
            **Number of Tools**: {len(tools)}

            ## Current Tool Implementations:
            ```python
            {chr(10).join(tools_text)}
            ```

            ## Database Schema:
            {chr(10).join(schema_summary)}

            ## Database Structure (from initial_state):
            {chr(10).join(state_structure)}

            ## Test Cases Summary:
            {chr(10).join(test_summary)}

            ## Current Requirements:
            {json.dumps(requirements or [], indent=2)}

            ## Your Task:
            1. Review all {len(tools)} tool implementations
            2. Identify inconsistencies in:
               - Parameter naming (especially the first parameter)
               - Data access patterns (how records are found/updated)
               - Timestamp formats
               - Error handling
               - Return value formats
               - ID generation
               - Parameter handling style
            3. Refine ALL tools to use consistent patterns
            4. Ensure all refined tools will still pass their test cases
            5. Provide a summary of changes made to each tool

            **CRITICAL**: The refined tools MUST maintain the exact same functionality and pass all existing test cases. Only change the implementation patterns for consistency, not the business logic.

            Please provide the refined tools and change summary in the specified format."""
        )
    
    def _extract_refined_tools(
        self, 
        response: str, 
        tool_names: List[str]
    ) -> Tuple[Optional[Dict[str, str]], Dict[str, List[str]]]:
        """
        Extract refined tools and changes from LLM response.
        
        Args:
            response: LLM response containing refined tools
            tool_names: List of expected tool names
            
        Returns:
            Tuple of (refined_tools, refinement_changes)
        """
        refined_tools = {}
        refinement_changes = {}
        
        # Extract Python code block
        python_match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
        if not python_match:
            logger.error("No Python code block found in refinement response")
            return None, {}
        
        full_code = python_match.group(1).strip()
        
        # Find the first "# TOOL:" marker to separate imports from tool definitions
        code_lines = full_code.split('\n')
        first_tool_line = -1
        
        for i, line in enumerate(code_lines):
            if line.strip().startswith('# TOOL:'):
                first_tool_line = i
                break
        
        # Extract imports (everything before the first tool marker)
        if first_tool_line > 0:
            imports_section = code_lines[:first_tool_line]
            imports_code = '\n'.join(imports_section).strip()
            logger.info(f"Extracted imports section ({len(imports_section)} lines)")
            remaining_code = '\n'.join(code_lines[first_tool_line:])
        else:
            # No tool markers found, try to extract imports differently
            imports_section = []
            i = 0
            while i < len(code_lines):
                line = code_lines[i].strip()
                # Check if line is import or from...import (but not # TOOL:)
                if (line.startswith('import ') or 
                    line.startswith('from ') or 
                    (line.startswith('#') and not line.startswith('# TOOL:')) or 
                    line == ''):
                    imports_section.append(code_lines[i])
                    i += 1
                else:
                    break
            imports_code = '\n'.join(imports_section).strip()
            logger.info(f"Extracted imports section ({len(imports_section)} lines, no tool markers found)")
            remaining_code = '\n'.join(code_lines[i:])
            
        tool_pattern = r"# TOOL: (\w+)\n(.*?)(?=# TOOL: |\Z)"
        tool_matches = re.findall(tool_pattern, remaining_code, re.DOTALL)
        
        if not tool_matches:
            logger.warning("No tool markers found, trying alternative extraction")
            # Fallback: try to extract without imports prefix
            tool_matches = re.findall(tool_pattern, full_code, re.DOTALL)
        
        for tool_name, tool_code in tool_matches:
            # Each tool gets the imports prepended
            if imports_code:
                refined_tools[tool_name] = imports_code + '\n\n' + tool_code.strip()
            else:
                refined_tools[tool_name] = tool_code.strip()
            logger.info(f"Extracted refined tool: {tool_name}")
        
        # Extract changes summary (JSON)
        json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
        if json_match:
            try:
                refinement_changes = json.loads(json_match.group(1).strip())
                logger.info(f"Extracted changes for {len(refinement_changes)} tools")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse refinement changes JSON: {e}")
        
        # Verify we got all expected tools
        missing_tools = set(tool_names) - set(refined_tools.keys())
        if missing_tools:
            logger.warning(f"Missing refined tools: {missing_tools}")
            return None, {}
        
        extra_tools = set(refined_tools.keys()) - set(tool_names)
        if extra_tools:
            logger.warning(f"Unexpected tools in refinement: {extra_tools}")
        
        return refined_tools, refinement_changes
    
    def _format_refinement_test_results(
        self,
        validation_results: Dict[str, Tuple[bool, str]],
        failed_tools_details: List[Dict[str, Any]],
        round_num: int
    ) -> str:
        """
        Format test results for refined tools as observation for ReAct agent.
        
        Args:
            validation_results: Dictionary mapping tool names to (success, message) tuples
            failed_tools_details: List of detailed failure information for failed tools
            round_num: Current round number
            
        Returns:
            Formatted observation string
        """
        result_text = f"Refinement Round {round_num} Validation Results\n"
        result_text += "=" * 80 + "\n\n"
        
        # Summary
        total_tools = len(validation_results)
        passed_tools = sum(1 for success, _ in validation_results.values() if success)
        failed_tools = total_tools - passed_tools
        
        result_text += f"Summary:\n"
        result_text += f"  Total tools: {total_tools}\n"
        result_text += f"  Passed: {passed_tools}\n"
        result_text += f"  Failed: {failed_tools}\n\n"
        
        if failed_tools == 0:
            result_text += "🎉 ALL TESTS PASSED! All tools are now consistent and functional.\n"
            return result_text
        
        # Show passed tools briefly
        result_text += "Passed Tools:\n"
        for tool_name, (success, message) in validation_results.items():
            if success:
                result_text += f"  ✓ {tool_name}\n"
        result_text += "\n"
        
        # Show failed tools with details
        result_text += "Failed Tools (need to fix):\n"
        result_text += "-" * 80 + "\n"
        
        for failure_detail in failed_tools_details:
            tool_name = failure_detail["tool_name"]
            message = failure_detail["message"]
            test_results = failure_detail["test_results"]
            
            result_text += f"\n❌ Tool: {tool_name}\n"
            result_text += f"   Error: {message}\n\n"
            
            # Show individual test results
            result_text += f"   Test Results:\n"
            for i, test_result in enumerate(test_results, 1):
                test_success = test_result.get("success", False)
                test_case = test_result.get("test_case", {})
                
                if test_success:
                    result_text += f"     Test {i}: ✓ PASSED\n"
                else:
                    result_text += f"     Test {i}: ✗ FAILED\n"
                    result_text += f"       Parameters: {json.dumps(test_case.get('parameters', {}))}\n"
                    result_text += f"       Error: {test_result.get('error', 'Unknown error')}\n"
                    
                    if 'traceback' in test_result:
                        # Show first few lines of traceback
                        traceback_lines = test_result['traceback'].split('\n')[:5]
                        result_text += f"       Traceback (first 5 lines):\n"
                        for line in traceback_lines:
                            result_text += f"         {line}\n"
            result_text += "\n"
        
        result_text += "-" * 80 + "\n"
        result_text += "\n**Action Required:**\n"
        result_text += "Please analyze the failed tools above and provide corrected implementations.\n"
        result_text += "Make sure to:\n"
        result_text += "1. Fix the specific issues causing test failures\n"
        result_text += "2. Maintain consistency with tools that passed\n"
        result_text += "3. Provide ALL tools (not just the fixed ones) in your response\n"
        result_text += "4. Keep the same output format with # TOOL: markers\n"
        
        return result_text