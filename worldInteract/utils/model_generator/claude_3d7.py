#%%
import json
import os
import dotenv
import anthropic
from typing import List
from loguru import logger
from anthropic.types import ContentBlock


dotenv.load_dotenv("../../../.env")


def react_generate(messages: list, **kwargs) -> tuple[ContentBlock, ContentBlock, List[ContentBlock]]:
    """
    Claude model calling function specifically for ReAct
    
    Args:
        messages: ReAct conversation history (including system prompts, test results, model responses, etc.)
        **kwargs: Other parameters, including api_key, max_tokens, temperature, etc.
    
    Returns:
        tuple: (thinking_content, answer_text, function_calls)
    """
    # Get API key
    api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("API key not provided. Please set it via api_key parameter or ANTHROPIC_API_KEY environment variable.")

    # Create Anthropic client
    client = anthropic.Anthropic(api_key=api_key)

    # Prepare messages based on calling method
    if messages is not None:
        # Method 2: Use pre-organized message array
        api_messages = messages
        system_prompt = None
        # Extract system prompt from messages (if any)
        if messages and messages[0].get("role") == "system":
            system_prompt = messages[0]["content"]
            api_messages = messages[1:]  # Remove system message

    # Prepare request parameters
    request_params = {
        "model": kwargs.get("model", "claude-sonnet-4-20250514"),
        "max_tokens": kwargs.get("max_tokens", 16384),
        "temperature": kwargs.get("temperature", 1.0),
        "messages": api_messages,
    }

    # Only add system prompt if present
    if system_prompt:
        request_params["system"] = system_prompt

    # Add tools parameter (if any)
    tools = kwargs.get("tools", [])
    if tools:
        request_params["tools"] = tools

    try:
        # Call API
        response = client.messages.create(**request_params)

        # Parse response data
        thinking_block = {}
        answer_block = {}
        function_blocks = []
        
        # Handle thinking (if model supports it)
        # In Claude standard API, thinking is usually in response.thinking
        if hasattr(response, 'thinking') and response.thinking:
            thinking_content = response.thinking
        
        # Handle text content and tool calls
        for content_block in response.content:
            if content_block.type == "text":
                answer_block = content_block
            elif content_block.type == "tool_use":
                function_blocks.append(content_block)
            elif content_block.type == "thinking":
                thinking_block = content_block
        
        return thinking_block, answer_block, function_blocks
        
    except anthropic.APIError as e:
        logger.error(f"Anthropic API call failed: {e}")
        raise Exception(f"Model inference failed: {e}")
    except Exception as e:
        logger.error(f"Model inference error: {e}")
        raise Exception(f"Model inference error: {e}")


def generate(system: str = None, user: str = None, messages: list = None, **kwargs) -> tuple:
    """
    Call Claude model using standard Anthropic API (supports two calling methods)
    
    Args:
        system: System prompt (method 1)
        user: User message (method 1)
        messages: Pre-organized message array (method 2)
        **kwargs: Other parameters, including api_key, max_tokens, temperature, tools, etc.
    
    Returns:
        tuple: (thinking_content, answer_text, function_calls)
    """
    # Get API key
    api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("API key not provided. Please set it via api_key parameter or ANTHROPIC_API_KEY environment variable.")
    
    # Create Anthropic client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Prepare messages based on calling method
    if messages is not None:
        # Method 2: Use pre-organized message array
        api_messages = messages
        system_prompt = None
        # Extract system prompt from messages (if any)
        if messages and messages[0].get("role") == "system":
            system_prompt = messages[0]["content"]
            api_messages = messages[1:]  # Remove system message
    else:
        # Method 1: Use system and user parameters
        api_messages = [{"role": "user", "content": user}]
        system_prompt = system
    
    # Prepare request parameters
    request_params = {
        "model": kwargs.get("model", "claude-sonnet-4-20250514"),
        "max_tokens": kwargs.get("max_tokens", 16384),
        "temperature": kwargs.get("temperature", 1.0),
        "messages": api_messages,
    }

    # Add system prompt only if present
    if system_prompt:
        request_params["system"] = system_prompt
    
    # Add tools parameter (if any)
    tools = kwargs.get("tools", [])
    if tools:
        request_params["tools"] = tools
    
    try:
        # Call API
        response = client.messages.create(**request_params)
        
        # Parse response data
        thinking_content = ""
        answer_text = ""
        function_calls = []
        
        # Handle thinking (if model supports it)
        # In Claude standard API, thinking is usually in response.thinking
        if hasattr(response, 'thinking') and response.thinking:
            thinking_content = response.thinking
        
        # Handle text content and tool calls
        for content_block in response.content:
            if content_block.type == "text":
                answer_text += content_block.text
            elif content_block.type == "tool_use":
                function_calls.append({
                    "name": content_block.name,
                    "parameters": content_block.input
                })
        
        return thinking_content, answer_text, function_calls
        
    except anthropic.APIError as e:
        logger.error(f"Anthropic API call failed: {e}")
        raise Exception(f"Model inference failed: {e}")
    except Exception as e:
        logger.error(f"Model inference error: {e}")
        raise Exception(f"Model inference error: {e}")


def stream_generate(system: str = None, user: str = None, messages: list = None, **kwargs) -> str:
    """
    Stream call Claude model using standard Anthropic API (supports two calling methods)
    
    Args:
        system: System prompt (method 1)
        user: User message (method 1)
        messages: Pre-organized message array (method 2)
        **kwargs: Other parameters, including api_key, max_tokens, temperature, tools, etc.
    
    Returns:
        str: Complete text content generated by streaming
    """
    # Get API key
    api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("API key not provided. Please set it via api_key parameter or ANTHROPIC_API_KEY environment variable.")
    
    # Create Anthropic client
    client = anthropic.Anthropic(api_key=api_key)
    
    if messages is not None:
        # Method 2: Use pre-organized message array
        api_messages = messages
        system_prompt = None
        # Extract system prompt from messages (if any)
        if messages and messages[0].get("role") == "system":
            system_prompt = messages[0]["content"]
            api_messages = messages[1:]  # Remove system message
    else:
        # Method 1: Use system and user parameters
        api_messages = [{"role": "user", "content": user}]
        system_prompt = system
    
    # Prepare request parameters
    request_params = {
        "model": kwargs.get("model", "claude-sonnet-4-20250514"),
        "max_tokens": kwargs.get("max_tokens", 8192),
        "temperature": kwargs.get("temperature", 0),
        "messages": api_messages,
        "stream": True,
    }
    
    # Add system prompt only if present
    if system_prompt:
        request_params["system"] = system_prompt
    
    # Add tools parameter (if any)
    tools = kwargs.get("tools", [])
    if tools:
        request_params["tools"] = tools
    
    try:
        # Stream API call
        answer_content = ""
        answer_flag = False
        
        with client.messages.stream(**request_params) as stream:
            for text in stream.text_stream:
                if not answer_flag:
                    print("<answer>\n", end="", flush=True)
                    answer_flag = True
                answer_content += text
                print(text, end="", flush=True)
        
        if answer_content:
            print("\n</answer>", flush=True)
        
        return answer_content
        
    except anthropic.APIError as e:
        logger.error(f"Anthropic streaming API call failed: {e}")
        raise Exception(f"Streaming request failed: {e}")
    except Exception as e:
        logger.error(f"Streaming processing error: {e}")
        raise Exception(f"Streaming processing error: {e}")


if __name__ == "__main__":
    # Function call example
    
    # Define tools
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather information for a specified city",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g.: Beijing, Shanghai, New York, etc."
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                        "default": "celsius"
                    }
                },
                "required": ["location"]
            }
        },
        {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to calculate, e.g.: 2+3*4"
                    }
                },
                "required": ["expression"]
            }
        }
    ]

    # Example 1: Weather query
    print("=== Example 1: Weather Query ===")
    thinking, answer, func_calls = generate(
        system="You are a helpful assistant",
        user="What is Erdős theorem",
        # tools=tools,
        max_tokens=1024,
        temperature=0.7,
    )
    
    print(f"Thinking: {thinking}")
    print(f"Answer: {answer}")
    print(f"Function Calls: {func_calls}")
    
    # Example 2: Mathematical calculation
    print("\n=== Example 2: Mathematical Calculation ===")
    thinking, answer, func_calls = generate(
        system="You are a helpful assistant that can query weather and perform calculations. When users need calculations, please use the calculate tool.",
        user="Please help me calculate the result of 15 * 24 + 100",
        tools=tools,
        max_tokens=1024,
        temperature=0.3,
    )
    
    print(f"Thinking: {thinking}")
    print(f"Answer: {answer}")
    print(f"Function Calls: {func_calls}")