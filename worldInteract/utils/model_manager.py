"""
Simplified model manager

Supports dynamic import of model modules using a simple factory pattern
"""

import sys
import importlib
from typing import Callable, Any
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed


# Default model configuration
DEFAULT_MODEL = "qwen3_32b"


def get_model_generator(model_name: str, func_name: str = "generate") -> Callable[[str, str, Any], str]:
    """
    Get model generator function
    
    Args:
        model_name: Model name, corresponding to module name in model_generator directory
        func_name: Function name to retrieve (generate, react_generate, stream_generate, etc.)
        
    Returns:
        Model generation function
    """
    try:
        # Dynamically import model module
        model_module = importlib.import_module(f"worldInteract.utils.model_generator.{model_name}")
        logger.info(f"Using {func_name} function from {model_name} model for generation")
        
        # Check specified function name first
        if hasattr(model_module, func_name):
            return getattr(model_module, func_name)
        
        # Fallback to default check order (backward compatibility)
        if hasattr(model_module, "generate"):
            generate_func = model_module.generate
        elif hasattr(model_module, "stream_generate"):
            generate_func = model_module.stream_generate
        else:
            raise AttributeError(f"Method '{func_name}', 'generate' or 'stream_generate' does not exist in module `{model_name}`!")
            
        return generate_func
        
    except ImportError as e:
        logger.error(f"Failed to import model module: {model_name}, error: {e}")
        # Fallback to default model
        if model_name != DEFAULT_MODEL:
            logger.warning(f"Falling back to default model: {DEFAULT_MODEL}")
            return get_model_generator(DEFAULT_MODEL, func_name)
        else:
            raise ImportError(f"Unable to import model module: {model_name}")


@retry(
    wait=wait_fixed(10),
    stop=stop_after_attempt(3)
)
def generate(model_key: str, system_prompt: str = None, user_prompt: str = None, messages: list = None, **kwargs) -> str:
    """
    Generate text using specified model (supports two calling methods)
    
    Args:
        model_key: Model name
        system_prompt: System prompt (method 1)
        user_prompt: User input (method 1)
        messages: Pre-organized message array (method 2)
        **kwargs: Other parameters
        
    Returns:
        Generated text
        
    Note:
        - Method 1: Pass system_prompt and user_prompt
        - Method 2: Pass messages array
        - Both methods cannot be used simultaneously
    """
    # Validate parameters
    if messages is not None:
        if system_prompt is not None or user_prompt is not None:
            raise ValueError("Cannot pass both messages and system_prompt/user_prompt")
    else:
        if system_prompt is None or user_prompt is None:
            raise ValueError("Must pass either messages or (system_prompt and user_prompt)")
    
    generate_func = get_model_generator(model_key)
    
    # Pass correct parameters
    if messages is not None:
        return generate_func(messages=messages, **kwargs)
    else:
        return generate_func(system_prompt, user_prompt, **kwargs)

@retry(
    wait=wait_fixed(10),
    stop=stop_after_attempt(3)
)
def react_generate(model_key: str, messages: list, **kwargs) -> str:
    """
    ReAct-specific generation function
    
    Args:
        model_key: Model name
        messages: ReAct conversation history message array
        **kwargs: Other parameters
        
    Returns:
        Generated text
    """
    # Prioritize model's react_generate function, fallback to generate if not available
    try:
        react_func = get_model_generator(model_key, "react_generate")
        return react_func(messages=messages, **kwargs)
    except AttributeError:
        # If model doesn't have react_generate function, use standard generate
        logger.info(f"Model {model_key} doesn't have react_generate function, using standard generate")
        generate_func = get_model_generator(model_key, "generate")
        return generate_func(messages=messages, **kwargs)

@retry(
    wait=wait_fixed(10),
    stop=stop_after_attempt(3)
)
def stream_generate(model_key: str, system_prompt: str, user_prompt: str, **kwargs):
    """
    Stream generation using specified model (compatible with original interface)
    
    Args:
        model_key: Model name (compatible with original model_key parameter)
        system_prompt: System prompt
        user_prompt: User input
        **kwargs: Other parameters
        
    Yields:
        Generated text fragments
    """
    generate_func = get_model_generator(model_key)
    
    # If it's a streaming function, return directly
    if hasattr(generate_func, '__name__') and 'stream' in generate_func.__name__:
        yield from generate_func(system_prompt, user_prompt, **kwargs)
    else:
        # If it's not a streaming function, return all at once
        result = generate_func(system_prompt, user_prompt, **kwargs)
        yield result
