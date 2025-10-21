"""
ReactAgent - ReAct Pattern Agent with CAMEL Integration

This agent implements the ReAct (Reasoning and Acting) pattern with automatic
history management using CAMEL's memory system. It provides a simple interface
for iterative refinement workflows.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from anthropic.types import ContentBlock

from camel.messages import BaseMessage
from camel.memories import ChatHistoryMemory, MemoryRecord
from camel.memories.context_creators import ScoreBasedContextCreator
from camel.types import OpenAIBackendRole, RoleType, ModelType
from camel.utils import OpenAITokenCounter

from worldInteract.utils.camel_model_manager import camel_model_manager
from worldInteract.utils.config_manager import config_manager


logger = logging.getLogger(__name__)


class ReactAgent:
    """
    ReAct agent with automatic history management.
    
    This agent uses CAMEL's ChatHistoryMemory to manage conversation history
    and provides a clean interface for ReAct-style interactions. The key features:
    
    - Automatic history management (assistant responses auto-added after step())
    - Manual observation insertion (via add_observation())
    - Returns separated thinking/content/function_calls
    - Uses CAMEL models for flexible model selection
    
    Typical usage in a validation loop:
        >>> agent = ReactAgent(config_key="code_agent")
        >>> agent.set_system_prompt("You are a code validator...")
        >>> agent.add_user_message("Here's the code to validate...")
        >>> 
        >>> # ReAct loop
        >>> for i in range(max_rounds):
        >>>     # Execute code and get results
        >>>     success, message, results = execute_code(...)
        >>>     
        >>>     # Add observation
        >>>     agent.add_observation(f"Test results: {results}")
        >>>     
        >>>     # Get agent's response
        >>>     thinking, content, functions = agent.step()
        >>>     
        >>>     if "ALL TESTS PASSED" in content:
        >>>         break
    
    Args:
        config_key: Configuration key from model_config.yaml (e.g., "code_agent")
        model_config_override: Optional parameters to override (e.g., {"temperature": 0.5})
    """
    
    def __init__(
        self,
        config_key: str,
        model_config_override: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ReactAgent.
        
        Args:
            config_key: Configuration key from model_config.yaml
            model_config_override: Optional parameter overrides
        """
        self.config_key = config_key
        self.model_config_override = model_config_override or {}
        
        # Create model using CAMEL model manager
        self.model = camel_model_manager.create_model(
            config_key=self.config_key,
            override_params=self.model_config_override
        )
        
        # Get model configuration to determine token limit
        model_config = config_manager.get_model_config(config_key)
        max_tokens = model_config.get("max_tokens", 8192)
        token_limit = max_tokens * 3  # Set context limit to ~3x max_tokens for safety
        
        # Create token counter (using GPT-4 as default for compatibility)
        token_counter = OpenAITokenCounter(ModelType.GPT_4O_MINI)
        
        # Create context creator with token management
        context_creator = ScoreBasedContextCreator(
            token_counter=token_counter,
            token_limit=token_limit
        )
        
        # Initialize memory for conversation history with context creator
        self.memory = ChatHistoryMemory(context_creator=context_creator)
        
        # System message storage
        self.system_message: Optional[BaseMessage] = None
        
        logger.info(f"ReactAgent initialized with config_key='{config_key}', token_limit={token_limit}")
    
    def set_system_prompt(self, prompt: str):
        """
        Set or update the system prompt.
        
        This will clear existing memory and add the new system message.
        
        Args:
            prompt: System prompt text
            
        Example:
            >>> agent.set_system_prompt(
            ...     "You are a helpful code validation agent. "
            ...     "Analyze test results and fix code issues."
            ... )
        """
        # Create system message
        self.system_message = BaseMessage.make_assistant_message(
            role_name="System",
            content=prompt
        )
        
        # Clear memory and add system message
        self.memory.clear()
        self.memory.write_record(
            MemoryRecord(
                message=self.system_message,
                role_at_backend=OpenAIBackendRole.SYSTEM,
                timestamp=0.0,
                agent_id="react_agent"
            )
        )
        
        logger.debug("System prompt set and memory initialized")
    
    def add_user_message(self, content: str):
        """
        Add a user message to history.
        
        Use this to add initial task descriptions or any user input.
        
        Args:
            content: User message content
            
        Example:
            >>> agent.add_user_message("Please validate this tool implementation...")
        """
        message = BaseMessage.make_user_message(
            role_name="User",
            content=content
        )
        
        self.memory.write_record(
            MemoryRecord(
                message=message,
                role_at_backend=OpenAIBackendRole.USER,
                timestamp=0.0,
                agent_id="react_agent"
            )
        )
        
        logger.debug(f"Added user message (length: {len(content)})")
    
    def add_observation(self, observation: str):
        """
        Add an observation (e.g., test results, execution feedback) to history.
        
        This is the key method for ReAct loops - after each action, you add
        the observation which becomes the next input for the agent.
        
        Args:
            observation: Observation text (e.g., test execution results)
            
        Example:
            >>> observation = f"Test Results:\\n{format_results(results)}"
            >>> agent.add_observation(observation)
        """
        self.add_user_message(observation)
        logger.debug("Added observation to history")
    
    def step(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        """
        Execute one step of ReAct reasoning following CAMEL's standard pattern.
        
        This method:
        1. Retrieves conversation history from memory
        2. Calls the model
        3. Extracts response components (thinking, content, function_calls)
        4. Stores response in memory using CAMEL's standard BaseMessage format
        5. Returns the extracted components
        
        Args:
            temperature: Optional temperature override for this step
            max_tokens: Optional max_tokens override for this step
            
        Returns:
            Tuple of (thinking, content, function_calls):
                - thinking: Thinking/reasoning text (empty string if not available)
                - content: Main response content
                - function_calls: List of function call dicts (empty list if none)
                
        Example:
            >>> thinking, content, functions = agent.step()
            >>> print(f"Agent thinking: {thinking}")
            >>> print(f"Agent response: {content}")
            >>> if functions:
            ...     print(f"Agent wants to call: {functions}")
        """
        # Get conversation history from memory
        messages, _ = self.memory.get_context()
        
        logger.debug(f"Calling model with {len(messages)} messages in history")
        
        # Prepare model parameters
        model_params = {}
        if temperature is not None:
            model_params["temperature"] = temperature
        if max_tokens is not None:
            model_params["max_tokens"] = max_tokens
        
        # Call model
        try:
            # TODO: pass tool
            response = self.model.run(messages=messages)
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            raise
        
        # Parse response and extract components
        thinking, content, function_calls = self._parse_response(response)
        
        # Create assistant message following CAMEL's standard pattern
        # Store thinking and function_calls in meta_dict (CAMEL's way of storing metadata)
        meta_dict = {}
        if thinking:
            meta_dict["thinking"] = thinking
        if function_calls:
            meta_dict["function_calls"] = function_calls
        
        assistant_message = BaseMessage.make_assistant_message(
            role_name="Assistant",
            content=content,  # Store clean content without thinking
            meta_dict=meta_dict if meta_dict else None
        )
        
        # Add to memory using CAMEL's update_memory pattern
        self.memory.write_record(
            MemoryRecord(
                message=assistant_message,
                role_at_backend=OpenAIBackendRole.ASSISTANT,
                timestamp=0.0,
                agent_id="react_agent"
            )
        )
        
        logger.debug(
            f"Step completed - thinking: {len(thinking)} chars, "
            f"content: {len(content)} chars, "
            f"functions: {len(function_calls)}"
        )
        
        return thinking, content, function_calls
    
    def _parse_response(
        self,
        response: Any
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        """
        Parse model response into components.
        
        Different models return responses in different formats. This method
        handles the parsing for various response types.
        
        Args:
            response: Model response object
            
        Returns:
            Tuple of (thinking, content, function_calls)
        """
        thinking = ""
        content = ""
        function_calls = []
        
        # Handle ChatCompletion response (standard format)
        if hasattr(response, "choices") and len(response.choices) > 0:
            message = response.choices[0].message
            
            # Extract content
            if hasattr(message, "content") and message.content:
                content = message.content
            
            # Extract function/tool calls
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    function_calls.append({
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    })
        
        # Handle direct content string
        elif isinstance(response, str):
            content = response
        
        # Handle Anthropic-style response with content blocks
        elif hasattr(response, "content") and isinstance(response.content, list):
            for block in response.content:
                if isinstance(block, ContentBlock):
                    if block.type == "text":
                        content += block.text
                    elif block.type == "thinking":
                        thinking += block.text
                    elif block.type == "tool_use":
                        function_calls.append({
                            "id": block.id,
                            "name": block.name,
                            "arguments": block.input
                        })
                elif hasattr(block, "text"):
                    content += block.text
        
        # Fallback: try to get text attribute
        elif hasattr(response, "text"):
            content = response.text
        
        else:
            logger.warning(f"Unknown response format: {type(response)}")
            content = str(response)
        
        return thinking, content, function_calls
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the conversation history.
        
        Returns:
            List of message dictionaries with role and content
            
        Example:
            >>> history = agent.get_history()
            >>> for msg in history:
            ...     print(f"{msg['role']}: {msg['content'][:50]}...")
        """
        messages, _ = self.memory.get_context()
        return messages
    
    def get_last_response_details(self) -> Dict[str, Any]:
        """
        Get detailed information from the last assistant response.
        
        This extracts thinking and function_calls from the meta_dict
        of the last assistant message, following CAMEL's metadata storage pattern.
        
        Returns:
            Dictionary with keys:
                - thinking: str - Thinking/reasoning content
                - content: str - Main response content
                - function_calls: List[Dict] - Function calls if any
                - has_metadata: bool - Whether metadata was found
                
        Example:
            >>> thinking, content, functions = agent.step()
            >>> details = agent.get_last_response_details()
            >>> print(f"Thinking: {details['thinking']}")
            >>> print(f"Content: {details['content']}")
        """
        messages, _ = self.memory.get_context()
        
        # Find the last assistant message
        for message in reversed(messages):
            if message.get("role") == "assistant":
                content = message.get("content", "")
                
                # Try to get metadata if available
                # Note: meta_dict might not be in the OpenAI format
                # We need to check the actual BaseMessage object
                # For now, return basic info
                return {
                    "thinking": "",
                    "content": content,
                    "function_calls": [],
                    "has_metadata": False
                }
        
        return {
            "thinking": "",
            "content": "",
            "function_calls": [],
            "has_metadata": False
        }
    
    def clear_history(self):
        """
        Clear conversation history.
        
        Note: This also clears the system message. You'll need to call
        set_system_prompt() again after clearing.
        """
        self.memory.clear()
        self.system_message = None
        logger.debug("History cleared")
    
    def reset(self):
        """
        Reset the agent to initial state.
        
        This clears history and reinitializes with system prompt if it was set.
        """
        system_prompt = None
        if self.system_message:
            system_prompt = self.system_message.content
        
        self.clear_history()
        
        if system_prompt:
            self.set_system_prompt(system_prompt)
        
        logger.debug("Agent reset")
    
    def get_message_count(self) -> int:
        """
        Get the number of messages in history.
        
        Returns:
            Number of messages (including system message)
        """
        messages, _ = self.memory.get_context()
        return len(messages)

