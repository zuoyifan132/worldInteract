"""
Task Agent module for ReAct-based interaction with tools.
Implements the Thought -> Action -> Observation loop for task execution.
"""

import json
import copy
from loguru import logger
from typing import Dict, Any, List, Tuple, Optional
from textwrap import dedent

from worldInteract.agents import ReactAgent
from worldInteract.utils.config_manager import config_manager
from worldInteract.core.sandbox import CodeExecutor


class TaskAgent:
    """ReAct agent for executing user tasks with tools."""
    
    def __init__(self, domain_tools: Dict[str, Any], env_path: str):
        """
        Initialize TaskAgent.
        
        Args:
            domain_tools: Domain tools information (from domain json file)
            env_path: Path to environment directory with tools.py, schema, initial_state
        """
        self.config = config_manager.get_environment_config("trajectory_generation")
        self.model_config = config_manager.get_model_config("trajectory_generation")
        
        self.domain_tools = domain_tools
        self.env_path = env_path
        self.max_rounds = self.config.get("max_react_rounds", 15)
        
        # Create ReactAgent for task execution
        self.agent = ReactAgent(config_key="trajectory_generation")
        
        # Load environment components
        self._load_environment()
        
    def _load_environment(self):
        """Load environment schema, initial state, and tool implementations."""
        import os
        
        # Load schema
        schema_path = os.path.join(self.env_path, "schema.json")
        with open(schema_path, 'r', encoding='utf-8') as f:
            self.schema = json.load(f)
        
        # Load initial state
        initial_state_path = os.path.join(self.env_path, "initial_state.json")
        with open(initial_state_path, 'r', encoding='utf-8') as f:
            self.initial_state = json.load(f)
        
        # Load tool implementations
        tools_path = os.path.join(self.env_path, "tools.py")
        with open(tools_path, 'r', encoding='utf-8') as f:
            self.tools_code = f.read()
        
        # Create code executor
        self.code_executor = CodeExecutor()
        
        logger.info(f"Loaded environment from {self.env_path}")
    
    def execute_task(
        self,
        user_query: str,
        available_tools: List[str],
        current_state: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute a user task using ReAct pattern with ReactAgent.
        
        Args:
            user_query: User's question or task
            available_tools: List of tool names available for this task
            current_state: Current database state
            
        Returns:
            Tuple of (interaction_history, final_state):
            - interaction_history: List of all interactions (user, assistant, tool)
            - final_state: Final state after all tool executions
        """
        logger.info(f"Executing task: {user_query[:100]}...")
        
        # Create system prompt with available tools and initial state
        system_prompt = self._create_system_prompt(available_tools, current_state)
        self.agent.set_system_prompt(system_prompt)
        
        # Add initial user query
        self.agent.add_user_message(user_query)
        
        # Format tools for API
        formatted_tools = self._format_tools_for_api(available_tools)
        
        # ReAct loop
        rounds = 0
        working_state = copy.deepcopy(current_state)
        
        while rounds < self.max_rounds:
            rounds += 1
            logger.info(f"ReAct round {rounds}/{self.max_rounds}")
            
            try:
                # Get model response (Thought + Action)
                thinking, answer_text, function_calls = self.agent.step(tools=formatted_tools)
                
                logger.info(f"ReAct model response length: {len(answer_text)} chars")
                if thinking:
                    logger.debug(f"Agent thinking: {thinking[:100]}...")
                
                # Check if task is complete (no function calls)
                if not function_calls:
                    logger.info("Task complete - no function calls")
                    break
                
                # Execute each function call (Observation)
                for func_call in function_calls:
                    tool_name = func_call.get("name")
                    tool_params = func_call.get("arguments", {})
                    tool_id = func_call.get("id")
                    
                    # Parse arguments if they are JSON string
                    if isinstance(tool_params, str):
                        try:
                            tool_params = json.loads(tool_params)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse tool parameters: {tool_params}")
                            tool_params = {}
                    
                    logger.info(f"Executing tool: {tool_name} with params: {tool_params}")
                    
                    # Execute tool and get result
                    success, result, new_state = self._execute_tool(
                        tool_name, tool_params, working_state
                    )
                    
                    # Update working state
                    if success:
                        working_state = new_state
                    
                    # Add tool result to agent history
                    self.agent.add_observation(json.dumps(result, ensure_ascii=False))
                    
                    logger.info(f"Tool execution {'succeeded' if success else 'failed'}")
                
            except Exception as e:
                logger.error(f"Error in ReAct round {rounds}: {e}")
                # Add error as observation and continue
                self.agent.add_observation(
                    f"Error occurred: {str(e)}. Please try again or complete the task."
                )
                # Break on error to avoid infinite loops
                break
        
        if rounds >= self.max_rounds:
            logger.warning(f"Task execution reached max rounds ({self.max_rounds})")
        
        logger.info(f"Task execution completed in {rounds} rounds")
        
        # Get final history from agent
        history = self.agent.get_history()
        
        # Reset agent for next task
        self.agent.reset()
        
        return history, working_state
    
    def _execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """
        Execute a tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            current_state: Current database state
            
        Returns:
            Tuple of (success, result, new_state)
        """
        try:
            # Create test case for execution
            test_case = {
                "parameters": parameters,
                "expected_behavior": { # TODO: expected behavior to be defined
                    "type": "mixed",
                    "description": "User-requested operation",
                    "should_succeed": True
                }
            }
            
            # Execute using code executor
            success, message, test_results = self.code_executor.execute_code(
                code=self.tools_code,
                requirements=[],  # TODO: if provided requirements, pass in
                test_cases=[test_case],
                initial_state=current_state,
                tool_name=tool_name
            )
            
            if success and test_results:
                test_result = test_results[0]
                result = test_result.get("result", {})
                
                # Extract after_execution_state
                new_state = result.get("after_execution_state", current_state)
                
                # Remove after_execution_state from result to avoid redundancy
                clean_result = {k: v for k, v in result.items() if k != "after_execution_state"}
                
                return True, clean_result, new_state
            else:
                # Execution failed
                error_msg = test_results[0].get("error", message) if test_results else message
                return False, {"success": False, "error": error_msg}, current_state
                
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return False, {"success": False, "error": str(e)}, current_state
    
    def _create_system_prompt(self, available_tools: List[str], initial_state: Dict[str, Any]) -> str:
        """
        Create system prompt for ReAct agent including initial state information.
        
        Args:
            available_tools: List of available tool names
            initial_state: Initial database/environment state
            
        Returns:
            System prompt string
        """
        domain = self.domain_tools.get("domain", "unknown")
        domain_desc = self.domain_tools.get("description", "")
        
        # Get tool descriptions
        tools_desc = []
        for tool in self.domain_tools.get("tools", []):
            if tool["name"] in available_tools:
                tools_desc.append(f"- **{tool['name']}**: {tool['description']}")
        
        # Format initial state for display
        state_str = json.dumps(initial_state, ensure_ascii=False, indent=2)
        
        return dedent(
            f"""You are an intelligent task assistant capable of using provided tools to complete user tasks.

            ## Domain Information
            **Domain**: {domain}
            **Description**: {domain_desc}

            ## Current Environment State
            The current state of the environment is:
            ```json
            {state_str}
            ```

            ## Available Tools
            {chr(10).join(tools_desc)}

            ## Workflow (ReAct Pattern)
            1. **Thought**: Analyze user requirements and think about which tools to use
            2. **Action**: Call appropriate tools with correct parameters
            3. **Observation**: Review tool execution results
            4. Repeat the above process until task is complete

            ## Task Completion Criteria
            When you believe the user's task is complete:
            - Provide a summary response to the user
            - **Do not call any tools** (do not provide function calls)
            - This will trigger the task completion signal

            ## Important Notes
            - Carefully read tool descriptions and parameter requirements
            - Pay attention to the current environment state when making decisions
            - Decide next steps based on tool execution results
            - If errors occur, try alternative approaches or inform the user
            - Keep conversations natural and friendly
            - You don't need to clarify anything at start
            - Must stop calling tools after task completion"""
        )
    
    def _format_tools_for_api(self, available_tools: List[str]) -> List[Dict[str, Any]]:
        """
        Format tools to generic format.
        
        Returns tools in generic format (Anthropic-style) which will be automatically
        converted to platform-specific format by ReactAgent.format_tools().
        
        Generic format: {
            "name": str,
            "description": str,
            "input_schema": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }
        """
        formatted_tools = []
        
        for tool in self.domain_tools.get("tools", []):
            if tool["name"] not in available_tools:
                continue
            
            # Convert parameters to input_schema format
            properties = {}
            required = []
            
            for param_name, param_info in tool.get("parameters", {}).items():
                properties[param_name] = {
                    "type": param_info.get("type", "string"),
                    "description": param_info.get("description", "")
                }
                
                # Add default if present
                if "default" in param_info:
                    properties[param_name]["default"] = param_info["default"]
                
                # Check if required (no default means required)
                if "default" not in param_info:
                    required.append(param_name)
            
            formatted_tool = {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
            
            # Preserve returns information if present
            if "returns" in tool:
                formatted_tool["returns"] = tool["returns"]
            
            formatted_tools.append(formatted_tool)
        
        return formatted_tools

