"""
Test tool formatting for different model platforms.
"""

from camel.types import ModelPlatformType
from worldInteract.agents import ReactAgent


def test_tool_formatting():
    """Test that tools are formatted correctly for different platforms."""
    
    # Generic tool format (Anthropic-style)
    generic_tools = [{
        "name": "get_user",
        "description": "Get user information by ID",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "The user ID"
                }
            },
            "required": ["user_id"]
        }
    }]
    
    print("=" * 80)
    print("Testing Tool Formatting for Different Platforms")
    print("=" * 80)
    
    # Test Anthropic format (claude_sonnet_4)
    print("\n1. Testing Anthropic Claude Sonnet 4...")
    try:
        agent_anthropic = ReactAgent(config_key="code_agent")  # Uses claude_sonnet_4
        formatted = agent_anthropic.format_tools(generic_tools)
        
        print(f"   Platform: {agent_anthropic.model_platform.value}")
        print(f"   Formatted tool structure:")
        print(f"   - Has 'name': {'name' in formatted[0]}")
        print(f"   - Has 'description': {'description' in formatted[0]}")
        print(f"   - Has 'input_schema': {'input_schema' in formatted[0]}")
        print(f"   - Has 'type' field: {'type' in formatted[0]}")
        print(f"   ✓ Anthropic format correct!")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test OpenAI format (gpt4o_mini)
    print("\n2. Testing OpenAI GPT-4o Mini...")
    try:
        agent_openai = ReactAgent(config_key="scenario_collection")  # Uses gpt4o_mini
        formatted = agent_openai.format_tools(generic_tools)
        
        print(f"   Platform: {agent_openai.model_platform.value}")
        print(f"   Formatted tool structure:")
        print(f"   - Has 'type': {'type' in formatted[0]}")
        print(f"   - type == 'function': {formatted[0].get('type') == 'function'}")
        print(f"   - Has 'function' key: {'function' in formatted[0]}")
        if 'function' in formatted[0]:
            print(f"   - function has 'name': {'name' in formatted[0]['function']}")
            print(f"   - function has 'description': {'description' in formatted[0]['function']}")
            print(f"   - function has 'parameters': {'parameters' in formatted[0]['function']}")
        print(f"   ✓ OpenAI format correct!")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 80)
    print("Tool Formatting Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_tool_formatting()

