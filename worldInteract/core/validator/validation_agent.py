"""
ValidationAgent for tool validation using ReAct pattern.
"""

import json
import re
import logging
from textwrap import dedent
from typing import Dict, Any, List, Optional, Generator, Tuple

from worldInteract.utils.model_manager import generate
from worldInteract.utils.config_manager import config_manager
from worldInteract.core.sandbox import CodeExecutor


logger = logging.getLogger(__name__)


class ValidationAgent:
    """ReAct-based agent for tool validation through iterative testing and code improvement."""
    
    def __init__(self):
        """Initialize ValidationAgent."""
        self.config = config_manager.get_environment_config("validation_agent")
        self.model_config = config_manager.get_model_config("validation_agent")
        
        self.max_rounds = self.config.get("max_validation_agent_rounds", 10)
        self.test_case_num = self.config.get("test_case_num", 3)
        
        self.code_executor = CodeExecutor()
        self.history = []
        
    def validate_tool(
        self,
        tool_name: str,
        tool_code: str,
        requirements: List[str],
        tool_desc: Dict[str, Any],
        schema: Dict[str, Any],
        initial_state: Dict[str, Any]
    ) -> Tuple[bool, str, str]:
        """
        Validate tool using ReAct pattern with test case execution.
        
        Args:
            tool_name: Name of the tool
            tool_code: Python implementation of the tool
            requirements: List of pip requirements
            tool_desc: Tool description from API collection
            schema: Database schema
            initial_state: Initial database state
            
        Returns:
            Tuple of (is_valid, final_code, validation_message)
        """
        logger.info(f"Starting validation for tool: {tool_name}")
        
        # Reset agent state
        self.history = []
        
        # Generate test cases
        test_cases = self._generate_test_cases(tool_name, tool_desc, schema, initial_state)
        
        # Initialize conversation
        system_prompt = self._create_system_prompt()
        initial_user_prompt = self._create_initial_user_prompt(
            tool_name, tool_code, requirements, test_cases, initial_state
        )
        
        # Add initial messages
        self.history.extend([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user_prompt}
        ])
        
        current_code = tool_code
        current_requirements = requirements
        rounds = 0
        
        # ReAct loop
        while rounds < self.max_rounds:
            rounds += 1
            logger.info(f"Validation round {rounds}/{self.max_rounds}")
            
            try:
                # Get model response
                # For conversation with history, we need to format the messages properly
                if len(self.history) > 2:  # More than just system + initial user
                    # Create a single prompt from the conversation history
                    conversation_text = ""
                    for msg in self.history[1:]:  # Skip system prompt
                        role = msg["role"]
                        content = msg["content"]
                        if role == "user":
                            conversation_text += f"User: {content}\n\n"
                        elif role == "assistant":
                            conversation_text += f"Assistant: {content}\n\n"
                    
                    user_prompt = conversation_text + "Please analyze the latest execution results and provide your response."
                else:
                    user_prompt = self.history[-1]["content"]  # Use the latest user message
                
                thinking_content, answer_text, function_calls = generate(
                    model_key=self.model_config["model"],
                    system_prompt=self.history[0]["content"],  # System prompt
                    user_prompt=user_prompt,
                    temperature=self.model_config.get("temperature", 0.3),
                    max_tokens=self.model_config.get("max_tokens", 8192)
                )
                
                # Parse model response
                thought, action = self._parse_model_response(answer_text)
                
                # Add assistant message to history
                self.history.append({
                    "role": "assistant", 
                    "content": f"<thought>{thought}</thought>\n{action}"
                })
                
                # Check if agent declares success
                if "ALL TEST CASES PASSED" in action:
                    logger.info("Agent declares all test cases passed")
                    return True, current_code, "All test cases passed successfully"
                
                # Extract updated code if provided
                updated_code, updated_requirements = self._extract_code_and_requirements(action)
                if updated_code:
                    current_code = updated_code
                    logger.info("Code updated by agent")
                if updated_requirements is not None:
                    current_requirements = updated_requirements
                    logger.info("Requirements updated by agent")
                
                # Execute test cases with current code
                success, message, test_results = self.code_executor.execute_code(
                    current_code, current_requirements, test_cases, initial_state, tool_name
                )
                
                # Format execution results for agent
                observation = self._format_execution_results(success, message, test_results)
                
                # Add observation to history
                self.history.append({
                    "role": "user",
                    "content": f"Execution Results:\n{observation}"
                })
                
                logger.info(f"Round {rounds} - Success: {success}, Message: {message}")
                
                # If all tests passed, we're done
                if success:
                    logger.info("All test cases passed!")
                    return True, current_code, "All test cases passed successfully"
                    
            except Exception as e:
                logger.error(f"Error in validation round {rounds}: {e}")
                # Add error to history and continue
                self.history.append({
                    "role": "user",
                    "content": f"Error occurred: {str(e)}"
                })
        
        # Max rounds reached without success
        logger.warning(f"Validation failed after {self.max_rounds} rounds")
        return False, current_code, f"Validation failed after {self.max_rounds} rounds"
    
    def _generate_test_cases(
        self,
        tool_name: str,
        tool_desc: Dict[str, Any],
        schema: Dict[str, Any],
        initial_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate test cases for the tool."""
        # Use the same test generation logic as the original validator
        from worldInteract.core.validator.tool_validator import ToolValidator
        
        temp_validator = ToolValidator()
        test_cases = []
        
        for i in range(self.test_case_num):
            try:
                test_case = temp_validator._generate_test_case(tool_name, tool_desc, schema, initial_state)
                test_case["test_id"] = i + 1
                test_cases.append(test_case)
            except Exception as e:
                logger.warning(f"Failed to generate test case {i+1}: {e}")
                # Create a fallback test case
                fallback_test_case = temp_validator._create_fallback_test_case(tool_name, tool_desc)
                fallback_test_case["test_id"] = i + 1
                test_cases.append(fallback_test_case)
        
        logger.info(f"Generated {len(test_cases)} test cases")
        return test_cases
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the validation agent."""
        return dedent(
            """You are a tool validation agent responsible for ensuring that tool implementations work correctly.

            Your task is to analyze test execution results and iteratively improve the tool code until all test cases pass.

            Instructions:
            1. **Analyze**: Examine test execution results to understand what went wrong
            2. **Think**: Reason about the issues and potential solutions  
            3. **Act**: Either fix the code or declare success

            Response Format:
            Always structure your response as:

            <thought>
            Your reasoning about the test results and what needs to be fixed
            </thought>

            Then provide one of:

            **Option A - Fix Code:**
            ```python
            def tool_name(data: Dict[str, Any], ...):
                # Your improved implementation
                pass
            ```

            **Requirements (if needed):**
            ```json
            ["package1", "package2>=1.0"]
            ```

            **Option B - Declare Success:**
            ALL TEST CASES PASSED

            Guidelines:
            - Carefully analyze error messages and failed test cases
            - Focus on fixing the specific issues identified in test results
            - Maintain the original tool specification and signature
            - Include proper error handling and validation
            - Always include `after_execution_state: data` in return JSON
            - Only declare success when confident all tests will pass
            - Be concise but thorough in your fixes"""
        )

    def _create_initial_user_prompt(
        self,
        tool_name: str,
        tool_code: str,
        requirements: List[str],
        test_cases: List[Dict[str, Any]],
        initial_state: Dict[str, Any]
    ) -> str:
        """Create initial user prompt with tool info and test cases."""
        test_cases_str = json.dumps(test_cases, indent=2)
        requirements_str = json.dumps(requirements, indent=2) if requirements else "[]"
        
        return dedent(
            f"""Please validate this tool implementation:

            **Tool Name**: {tool_name}

            **Current Implementation**:
            ```python
            {tool_code}
            ```

            **Current Requirements**:
            ```json
            {requirements_str}
            ```

            **Test Cases to Pass**:
            ```json
            {test_cases_str}
            ```

            **Initial Database State**: 
            ```json
            {json.dumps(initial_state, indent=2)}
            ```

            Your goal is to ensure this tool passes all test cases. I will execute the code and provide you with the results. Based on the results, you should either fix the code or declare that all test cases pass.

            Please analyze the current implementation and let me know if you want to make any changes before we start testing."""
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
    
    def _extract_code_and_requirements(self, action: str) -> Tuple[Optional[str], Optional[List[str]]]:
        """Extract updated code and requirements from agent action."""
        code = None
        requirements = None
        
        # Extract Python code
        python_match = re.search(r"```python\n(.*?)\n```", action, re.DOTALL)
        if python_match:
            code = python_match.group(1).strip()
        
        # Extract requirements
        json_matches = re.findall(r"```json\n(.*?)\n```", action, re.DOTALL)
        for json_content in json_matches:
            try:
                parsed = json.loads(json_content.strip())
                if isinstance(parsed, list):
                    requirements = parsed
                    break
            except json.JSONDecodeError:
                continue
        
        return code, requirements
    
    def _format_execution_results(
        self,
        success: bool,
        message: str,
        test_results: List[Dict[str, Any]]
    ) -> str:
        """Format execution results for the agent."""
        result_text = f"Overall Result: {'SUCCESS' if success else 'FAILED'}\n"
        result_text += f"Message: {message}\n\n"
        
        result_text += "Individual Test Results:\n"
        for i, test_result in enumerate(test_results, 1):
            test_success = test_result.get("success", False)
            test_case = test_result.get("test_case", {})
            
            result_text += f"\nTest {i}: {'PASSED' if test_success else 'FAILED'}\n"
            result_text += f"  Parameters: {json.dumps(test_case.get('parameters', {}))}\n"
            
            if test_success:
                result_text += f"  Result: {json.dumps(test_result.get('result', {}), indent=2)}\n"
            else:
                result_text += f"  Error: {test_result.get('error', 'Unknown error')}\n"
                if 'traceback' in test_result:
                    result_text += f"  Traceback:\n{test_result['traceback']}\n"
        
        return result_text
