"""
CodeAgent for tool code generation and validation using ReAct pattern.
"""

import json
import re
from loguru import logger
from textwrap import dedent
from typing import Dict, Any, List, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from worldInteract.agents import ReactAgent
from worldInteract.utils.camel_generator import generate
from worldInteract.utils.config_manager import config_manager
from worldInteract.utils.parser_utils import extract_python_code_from_text, extract_json_from_text
from worldInteract.core.sandbox import CodeExecutor


class CodeAgent:
    """ReAct-based agent for tool code generation and validation."""
    
    def __init__(self):
        """Initialize CodeAgent."""
        self.config = config_manager.get_environment_config("code_agent")
        self.model_config = config_manager.get_model_config("code_agent")
        
        self.max_rounds = self.config.get("max_code_agent_rounds", 10)
        self.test_case_num = self.config.get("test_case_num", 3)
        
        self.code_executor = CodeExecutor()
        
        # Note: ReactAgent will be created per validation session
        # to ensure clean state for each tool
        self.agent = ReactAgent(config_key="code_agent")

        # Set up system prompt for validation
        validation_system_prompt = self._create_validation_system_prompt()
        self.agent.set_system_prompt(validation_system_prompt)
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry attempt {retry_state.attempt_number}/3 for generate_code_and_tests"
        )
    )
    def generate_code_and_tests(
        self,
        tool_desc: Dict[str, Any],
        schema: Dict[str, Any],
        initial_state: Dict[str, Any],
        domain: str
    ) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        """
        Pre-generate tool code and test cases with automatic retry (max 3 attempts).
        
        Args:
            tool_desc: Tool description from API collection
            schema: Database schema
            initial_state: Initial database state
            domain: Domain name
            
        Returns:
            Tuple of (code, requirements, test_cases)
        """
        tool_name = tool_desc["name"]
        logger.info(f"Pre-generating code and tests for tool: {tool_name}")
        
        # Generate initial code and test cases
        system_prompt = self._create_generation_system_prompt()
        user_prompt = self._create_initial_user_prompt(tool_desc, schema, initial_state, domain)

        thinking_content, answer_text, function_calls = generate(
            config_key=self.model_config["model"],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.model_config.get("temperature", 0.3),
            max_tokens=self.model_config.get("max_tokens", 12288)
        )
        
        # Extract code, requirements, and test cases
        code, requirements, test_cases = self._extract_code_requirements_tests(answer_text)
  
        if not code:
            logger.error(f"Failed to generate initial code for tool: {tool_name}")
            raise ValueError(f"Failed to generate initial code for tool: {tool_name}")
        
        if not test_cases:
            logger.error(f"Failed to generate test cases for tool: {tool_name}")
            raise ValueError(f"Failed to generate test cases for tool: {tool_name}")
        
        logger.info(f"Successfully pre-generated code and {len(test_cases)} test cases for {tool_name}")
        return code, requirements or [], test_cases

    def generate_and_validate_tool(
        self,
        code: str,
        requirements: List[str],
        test_cases: List[Dict[str, Any]],
        schema: Dict[str, Any],
        initial_state: Dict[str, Any],
        domain: str,
        tool_desc: Dict[str, Any]
    ) -> Tuple[bool, str, List[str], str]:
        """
        Validate pre-generated tool code using ReAct pattern with ReactAgent.
        
        Args:
            code: Pre-generated tool code
            requirements: Pre-generated requirements
            test_cases: Pre-generated test cases
            schema: Database schema
            initial_state: Initial database state
            domain: Domain name
            tool_desc: Tool description from API collection
            
        Returns:
            Tuple of (is_success, final_code, requirements, message)
        """
        tool_name = tool_desc["name"]
        logger.info(f"Starting code validation for tool: {tool_name}")
        
        # Add initial validation request
        initial_user_prompt = self._create_validation_user_prompt(
            code, requirements, test_cases, schema, initial_state, domain, tool_desc
        )
        self.agent.add_user_message(initial_user_prompt)
        
        current_code = code
        current_requirements = requirements
        rounds = 0
        
        # ReAct loop: Test-first iterative improvement
        while rounds < self.max_rounds:
            rounds += 1
            round_description = "Initial test" if rounds == 1 else f"Round {rounds-1} iteration"
            logger.info(f"Code validation {round_description} ({rounds}/{self.max_rounds})")
            
            try:
                # 1. Execute code with current implementation
                success, message, test_results = self.code_executor.execute_code(
                    current_code, current_requirements, test_cases, initial_state, tool_name
                )
                
                # 2. Format observation for agent
                observation = self._format_execution_results(success, message, test_results)
                
                # 3. Add test results as observation (automatically added to history by ReactAgent)
                observation_title = "Initial Test Execution Results" if rounds == 1 else f"Round {rounds-1} Test Execution Results"
                self.agent.add_observation(f"{observation_title}:\n{observation}")
                
                logger.info(f"{round_description} - Success: {success}, Message: {message}")
                
                # 4. Get ReAct model response (automatically added to history by ReactAgent)
                try:
                    thinking, answer_text, function_calls = self.agent.step()
                    
                    logger.info(f"ReAct model response length: {len(answer_text)} chars")
                    if thinking:
                        logger.debug(f"Agent thinking: {thinking[:100]}...")
                        
                except Exception as e:
                    logger.error(f"Error in code generation round {rounds}: {e}")
                    self.agent.reset()
                    return False, current_code, current_requirements, f"Code generation failed: {str(e)}"
                
                # Check if agent declares success
                if "ALL TEST CASES PASSED" in answer_text.upper():
                    logger.info("All test cases passed")
                    self.agent.reset()
                    return True, current_code, current_requirements, "All test cases passed"
                
                # 5. Extract code and requirements from agent response
                updated_code, updated_requirements, _ = self._extract_code_requirements_tests(answer_text)
                
                # 6. Update current code and requirements for next iteration
                if updated_code:
                    current_code = updated_code
                    logger.info("Code updated by agent")
                else:
                    logger.warning("Agent did not provide updated code, using previous version")
                    
                if updated_requirements is not None:
                    current_requirements = updated_requirements
                    logger.info("Requirements updated by agent")
                    
            except Exception as e:
                logger.error(f"Error in code generation round {rounds}: {e}")
                # Add error as observation and continue
                self.agent.add_observation(f"Error occurred: {str(e)}. Please try again.")
        
        # Max rounds reached without success
        logger.warning(f"Code generation failed after {self.max_rounds} rounds")
        self.agent.reset()
        return False, current_code or "", current_requirements, f"Code generation failed after {self.max_rounds} rounds"
    
    def _create_generation_system_prompt(self) -> str:
        """Create system prompt for initial code generation."""
        return dedent(
            f"""You are a tool code generation agent. Your task is to generate complete, working tool implementations from specifications.

            ## Your workflow:
            - Step 1. **Generate Code**: Create Python tool implementation and specify any needed pip packages for that code
            - Step 2. **Generate Test Cases**: Create exactly {self.test_case_num} comprehensive test cases

            ## Example:

            ### Example of code generation and requirements generation:
            **For initial code generation:**
            ```python
            def tool_name(data: Dict[str, Any], ...):
                \"\"\"Tool implementation\"\"\"
                try:
                    # Your implementation here - directly modify data as needed
                    # Use imports as needed: json, uuid, datetime, etc.
                    
                    return json.dumps({{
                        "success": True,
                        "data": result_data
                        # Note: No need to return after_execution_state, framework handles it
                    }})
                except Exception as e:
                    return json.dumps({{
                        "success": False,
                        "error": str(e)
                    }})
            ```
            **Requirements (if needed):**
            ```json
            ["package1", "package2>=1.0"]
            ```
            
            ### Example of test cases generation:
            **Test Cases (generate exactly {self.test_case_num} test cases):**
            ```json
            [
                {{
                    "test_id": 1,
                    "parameters": {{"param": "value"}},
                    "expected_behavior": {{
                        "type": "read|write|mixed",
                        "description": "What should happen",
                        "should_succeed": true
                    }},
                    "test_description": "Test description"
                }},
                {{
                    "test_id": 2,
                    "parameters": {{"param": "edge_case_value"}},
                    "expected_behavior": {{
                        "type": "read|write|mixed", 
                        "description": "Edge case behavior",
                        "should_succeed": true
                    }},
                    "test_description": "Edge case test"
                }}
                // Continue until you have exactly desired number of test cases
            ]
            ```

            ## Guidelines:
            ### Code Generation:
            - Include proper error handling in code
            - Your code implementation should directly manipulate the input `data` parameter for any database changes instead of using a simulation
            - Focus on business logic and return meaningful results 
            - Be thorough but concise in your implementations
            - Make sure the requirements are compatible with the code you generate

            ### Test Cases Generation:
            - Generate exactly desired number of realistic, comprehensive test cases in one JSON array
            - Test both success and failure scenarios when appropriate
            - Ensure test cases cover: normal operations, edge cases, and error conditions
            - Ensure your test cases are compatible with the current database state
            """
        )

    def _create_validation_system_prompt(self) -> str:
        """Create system prompt for code validation ReAct agent."""
        return dedent(
            """You are a tool code validation ReAct(Reasoning and Act) agent. Your task is to analyze test execution results and iteratively fix code issues until all tests pass.

            ## Your ReAct workflow:
            You will receive test execution results showing which tests passed/failed. Based on these results:
            - Step 1. **Analyze**: Examine the test execution results to understand what went wrong
            - Step 2. **Think**: Reason about the root causes and potential solutions  
            - Step 3. **Act**: Provide improved code implementation

            ## Response Choices:
            Then provide one of the following choices:

            **Option A - Fix Code (when tests are failing):**
            ```python
            import json
            import uuid
            import datetime
            from typing import Dict, Any, List, Optional

            def tool_name(data: Dict[str, Any], ...):
                \"\"\"Tool implementation\"\"\"
                try:
                    # Your improved implementation based on test results
                    # Use imports as needed: json, uuid, datetime, etc.
                    
                    return json.dumps({{
                        "success": True,
                        "data": result_data
                    }})
                except Exception as e:
                    return json.dumps({{
                        "success": False,
                        "error": str(e)
                    }})
            ```

            **Fixed Requirements (if needed):**
            ```json
            ["package1", "package2>=1.0"]
            ```

            **Option B - Declare Success (only when all tests pass):**
            ALL TEST CASES PASSED

            ## Guidelines:
            - You will first see initial test results, then provide fixes based on what failed
            - Carefully analyze error messages, tracebacks, and failed vs passed test cases
            - Focus on fixing specific issues while maintaining logic for tests that already pass
            - Maintain the original tool specification and function signature
            - Include proper error handling and validation
            - Your code implementation should directly manipulate the input `data` parameter for database changes
            - Return meaningful JSON results as specified in the tool specification
            - Only declare "ALL TEST CASES PASSED" when you're confident all tests will actually pass
            - Be methodical: fix one issue at a time based on the test feedback"""
        )

    def _create_initial_user_prompt(
        self,
        tool_desc: Dict[str, Any],
        schema: Dict[str, Any],
        initial_state: Dict[str, Any],
        domain: str
    ) -> str:
        """Create initial user prompt with tool specification."""
        tool_name = tool_desc["name"]
        tool_description = tool_desc["description"]
        parameters = tool_desc.get("parameters", {})
        returns = tool_desc.get("returns", {})
        
        # Format parameters
        param_info = []
        for param_name, param_def in parameters.items():
            if param_name != "required":
                param_type = param_def.get("type", "Any")
                param_desc = param_def.get("description", "")
                default = param_def.get("default")
                default_info = f" (default: {default})" if default is not None else ""
                param_info.append(f"  - {param_name} ({param_type}): {param_desc}{default_info}")
            else:
                param_info.append(f"  - {param_name}: {param_def}")
        
        # Format schema information
        schema_info = []
        for table_name, table_def in schema.items():
            table_desc = table_def.get("description", "")
            fields = list(table_def.get("fields", {}).keys())
            schema_info.append(f"  - {table_name}: {table_desc}")
            schema_info.append(f"    Fields: {fields}")
        
        # Format initial state information
        state_info = json.dumps(initial_state, indent=2)
        
        return dedent(
            f"""Please generate a complete tool implementation for this specification:

            **Tool Name**: {tool_name}
            **Description**: {tool_description}
            **Domain**: {domain}

            **Parameters**:
            {chr(10).join(param_info) if param_info else "  No parameters"}

            **Expected Returns**: 
            {json.dumps(returns, indent=2) if returns else "JSON object with success status and relevant data"}

            **Database Schema**:
            {chr(10).join(schema_info)}
            
            **Database current state**
            {state_info}

            **Requirements**:
            1. Generate the Python function implementation and specify any required pip packages (beyond standard library)
            2. Create exactly {self.test_case_num} comprehensive test cases in a single JSON array that cover:
            - Normal operation scenarios
            - Edge cases and boundary conditions  
            - Error scenarios (if applicable)
            3. Ensure all test cases will pass with your implementation

            I will execute your code and provide feedback. Based on the results, you should iterate until all tests pass.""")
    
    def _create_validation_user_prompt(
        self,
        code: str,
        requirements: List[str],
        test_cases: List[Dict[str, Any]],
        schema: Dict[str, Any],
        initial_state: Dict[str, Any],
        domain: str,
        tool_desc: Dict[str, Any]
    ) -> str:
        """Create user prompt for validation with pre-generated code and tests."""
        tool_name = tool_desc["name"]
        tool_description = tool_desc["description"]
        
        return dedent(
            f"""Please validate this tool implementation by analyzing test execution results and fixing any issues:

            **Tool Name**: {tool_name}
            **Description**: {tool_description}
            **Domain**: {domain}

            **Current Implementation**:
            ```python
            {code}
            ```

            **Current Requirements**:
            ```json
            {json.dumps(requirements, indent=2)}
            ```

            **Test Cases to Pass**:
            ```json
            {json.dumps(test_cases, indent=2)}
            ```

            **Database Schema**: 
            ```json
            {json.dumps(schema, indent=2)}
            ```

            **Initial Database State**: 
            ```json
            {json.dumps(initial_state, indent=2)}
            ```

            Your goal is to ensure this tool passes all test cases. I will execute the code with the test cases and provide you with detailed test execution results. You should analyze these results and provide improved code implementations based on what failed.

            The process will be:
            1. I'll run the current implementation against all test cases
            2. You'll receive detailed results showing which tests passed/failed and why
            3. You analyze the results and provide improved code
            4. We repeat until all tests pass

            Let me start by running the initial test execution..."""
        )

    def _parse_model_response(self, response: str) -> Tuple[str, str]:
        """Parse model response to extract thought and action."""
        # Extract thought
        thought_match = re.search(r"<thought>(.*?)</thought>", response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        
        # Extract action (everything after thought)
        if thought_match:
            action = response[thought_match.end():].strip()
        else:
            action = response.strip()
        
        return thought, action
    
    def _extract_code_requirements_tests(self, action: str) -> Tuple[Optional[str], Optional[List[str]], Optional[List[Dict[str, Any]]]]:
        """Extract code, requirements, and test cases from agent action."""
        code = None
        requirements = None
        test_cases = None
        
        # Extract Python code
        python_match = re.search(r"```python\n(.*?)\n```", action, re.DOTALL)
        if python_match:
            code = python_match.group(1).strip()
        
        # Extract JSON blocks (for requirements and test cases)
        json_matches = re.findall(r"```json\n(.*?)\n```", action, re.DOTALL)
        for json_content in json_matches:
            try:
                parsed = json.loads(json_content.strip())
                if isinstance(parsed, list):
                    if all(isinstance(item, str) for item in parsed):
                        # This looks like requirements
                        requirements = parsed
                    elif all(isinstance(item, dict) and "test_id" in item for item in parsed):
                        # This looks like test cases
                        test_cases = parsed
                    elif len(parsed) > 0 and isinstance(parsed[0], dict) and "parameters" in parsed[0]:
                        # This also looks like test cases (alternative format)
                        test_cases = parsed
            except json.JSONDecodeError:
                continue
        
        return code, requirements, test_cases
    
    def _format_execution_results(
        self,
        success: bool,
        message: str,
        test_results: List[Dict[str, Any]]
    ) -> str:
        """Format execution results for the agent."""
        result_text = f""
        result_text += f"Message: {message}\n\n"
        
        result_text += "Individual Test Results:\n"
        for i, test_result in enumerate(test_results, 1):
            test_success = test_result.get("success", False)
            test_case = test_result.get("test_case", {})
            
            result_text += f"\nTest {i}:\n"
            result_text += f"  Parameters: {json.dumps(test_case.get('parameters', {}))}\n"
            
            if test_success:
                result_text += f"  Result: {json.dumps(test_result.get('result', {}), indent=2)}\n"
            else:
                result_text += f"  Error: {test_result.get('error', 'Unknown error')}\n"
                if 'traceback' in test_result:
                    result_text += f"  Traceback:\n{test_result['traceback']}\n"
        
        return result_text
    
    def generate_and_validate_tools(
        self,
        api_collection: Dict[str, Any],
        schema: Dict[str, Any],
        initial_state: Dict[str, Any]
    ) -> Tuple[Dict[str, str], List[str], Dict[str, bool], Dict[str, List[Dict[str, Any]]]]:
        """
        Generate and validate all tools in the API collection.
        
        Args:
            api_collection: API collection containing tool descriptions
            schema: Database schema
            initial_state: Initial database state
            
        Returns:
            Tuple of (successful_tools, all_requirements, validation_results, test_cases)
        """
        domain = api_collection.get("domain", "unknown")
        tools = api_collection.get("tools", [])
        
        logger.info(f"Generating and validating {len(tools)} tools for domain: {domain}")
        
        successful_tools = {}
        all_requirements = set()
        validation_results = {}
        all_test_cases = {}
        
        for tool_desc in tools:
            tool_name = tool_desc["name"]
            logger.info(f"Processing tool: {tool_name}")
            
            try:
                # Phase 1: Generate code and test cases
                code, requirements, test_cases = self.generate_code_and_tests(
                    tool_desc, schema, initial_state, domain
                )
                
                # Store test cases regardless of validation success
                all_test_cases[tool_name] = test_cases
                
                # Phase 2: Validate with ReAct
                is_success, final_code, final_requirements, message = self.generate_and_validate_tool(
                    code, requirements, test_cases, schema, initial_state, domain, tool_desc
                )
                
                validation_results[tool_name] = is_success
                
                if is_success:
                    successful_tools[tool_name] = final_code
                    all_requirements.update(final_requirements)
                    logger.info(f"Tool {tool_name} validation: SUCCESS")
                else:
                    logger.warning(f"Tool {tool_name} validation: FAILED - {message}")
                    
            except Exception as e:
                logger.error(f"Error processing tool {tool_name}: {e}")
                validation_results[tool_name] = False
                # Still try to store empty test cases for failed tools
                all_test_cases[tool_name] = []
        
        passed = sum(1 for result in validation_results.values() if result)
        total = len(validation_results)
        logger.info(f"Code generation completed: {passed}/{total} tools successful")
        
        return successful_tools, list(all_requirements), validation_results, all_test_cases
