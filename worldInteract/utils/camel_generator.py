"""
CAMEL Model Generator Wrapper

This module provides a wrapper function that uses ReactAgent to mimic the old
model_manager.generate() interface for backward compatibility.
"""

from typing import Tuple, List, Dict, Any, Optional
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from worldInteract.agents import ReactAgent


@retry(
    wait=wait_fixed(10),
    stop=stop_after_attempt(3)
)
def generate(
    config_key: str,
    system_prompt: str = None,
    user_prompt: str = None,
    messages: list = None,
    **kwargs
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Generate text using CAMEL ReactAgent with backward-compatible interface.
    
    This function creates a temporary ReactAgent instance to handle single generation requests.
    
    Args:
        config_key: Configuration key from model_config.yaml (e.g. "scenario_collection")
        system_prompt: System prompt (method 1)
        user_prompt: User prompt (method 1)
        messages: Pre-organized message array (method 2)
        **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
    Returns:
        Tuple of (thinking_content, answer_text, function_calls)
        - thinking_content: Thinking/reasoning content
        - answer_text: Generated text response
        - function_calls: List of function/tool calls made by the model
        
    Example:
        >>> thinking, answer, calls = generate(
        ...     "scenario_collection",
        ...     system_prompt="You are a helpful assistant",
        ...     user_prompt="What is the weather?",
        ...     temperature=0.7
        ... )
    """
    # Validate parameters
    if messages is not None:
        if system_prompt is not None or user_prompt is not None:
            raise ValueError("Cannot pass both messages and system_prompt/user_prompt")
    else:
        if system_prompt is None or user_prompt is None:
            raise ValueError("Must pass either messages or (system_prompt and user_prompt)")
    
    try:
        # Create ReactAgent instance
        agent = ReactAgent(
            config_key=config_key,
            model_config_override=kwargs
        )
        
        # Handle two calling methods
        if messages is not None:
            # Method 2: Use pre-organized message array
            # Extract system and user messages from messages
            system_msg = None
            user_msgs = []
            
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                
                if role == "system":
                    system_msg = content
                elif role == "user":
                    user_msgs.append(content)
                elif role == "assistant":
                    # Add assistant messages if present (for multi-turn conversations)
                    agent.add_user_message(content)
            
            # Set system prompt if present
            if system_msg:
                agent.set_system_prompt(system_msg)
            
            # Add all user messages
            for user_msg in user_msgs:
                agent.add_user_message(user_msg)
        else:
            # Method 1: Use system_prompt and user_prompt
            if system_prompt:
                agent.set_system_prompt(system_prompt)
            
            agent.add_user_message(user_prompt)
        
        # Call step() to get response
        thinking, content, function_calls = agent.step()
        
        logger.debug(f"Generated response using CAMEL: {len(content)} chars")
        return thinking, content, function_calls
        
    except Exception as e:
        logger.error(f"Failed to generate using CAMEL model manager: {e}")
        raise


@retry(
    wait=wait_fixed(10),
    stop=stop_after_attempt(3)
)
def react_generate(
    config_key: str,
    messages: list,
    **kwargs
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    ReAct-specific generation function using CAMEL.
    
    Args:
        config_key: Configuration key from model_config.yaml
        messages: ReAct conversation history message array
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (thinking_content, answer_text, function_calls)
    """
    return generate(
        config_key=config_key,
        messages=messages,
        **kwargs
    )


@retry(
    wait=wait_fixed(10),
    stop=stop_after_attempt(3)
)
def stream_generate(
    config_key: str,
    system_prompt: str,
    user_prompt: str,
    **kwargs
):
    """
    Stream generation (placeholder implementation).
    
    Args:
        config_key: Configuration key from model_config.yaml
        system_prompt: System prompt
        user_prompt: User prompt
        **kwargs: Additional parameters
        
    Yields:
        Generated text fragments
        
    Note:
        This is a placeholder implementation. CAMEL streaming support needs to be
        added based on specific model backend.
    """
    # Currently use non-streaming approach, return complete result at once
    thinking, answer, calls = generate(
        config_key=config_key,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        **kwargs
    )
    yield answer

