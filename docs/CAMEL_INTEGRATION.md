# CAMEL Integration Guide

This guide explains how to use the CAMEL-based model system in WorldInteract framework.

## 📋 Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Model Configuration](#model-configuration)
- [Using ReactAgent](#using-reactagent)
- [Model Selection](#model-selection)
- [API Key Management](#api-key-management)
- [Examples](#examples)

## Overview

WorldInteract now uses [CAMEL](https://github.com/camel-ai/camel) (Communicative Agents for "Mind" Exploration of Large Language Model Society) as the underlying model management system. This provides:

- **Unified API** for multiple model providers (Anthropic, OpenAI, Gemini, etc.)
- **Flexible model switching** via configuration
- **Automatic history management** with ReactAgent
- **Secure API key handling** via environment variables

## Setup

### 1. Install Dependencies

```bash
# Install CAMEL
pip install camel-ai[all]

# Or install specific components
pip install camel-ai anthropic openai google-generativeai
```

### 2. Configure API Keys

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your actual API keys:

```bash
# Required: Add your API keys
ANTHROPIC_API_KEY=sk-ant-xxxxx
OPENAI_API_KEY=sk-xxxxx
GEMINI_API_KEY=xxxxx
```

**Important:** Never commit the `.env` file to version control!

### 3. Configure Models

Edit `config/model_config.yaml` to select models for different components:

```yaml
code_agent:
  model: "claude_sonnet_4"  # or "claude_3d7", "gpt4o", etc.
  temperature: 0.3
  max_tokens: 12288

trajectory_generation:
  model: "claude_3d7"
  temperature: 0.7
  max_tokens: 8192
```

## Model Configuration

### Available Models

The framework supports multiple models through simple names:

#### Anthropic Claude
- `claude_sonnet_4` - Claude Sonnet 4 (latest, recommended)
- `claude_opus_4` - Claude Opus 4 (most capable)
- `claude_3d7` - Claude 3.7 Sonnet
- `claude_3_5_sonnet` - Claude 3.5 Sonnet
- `claude_3_5_haiku` - Claude 3.5 Haiku (fast)

#### OpenAI GPT
- `gpt4o` - GPT-4o (recommended)
- `gpt4o_mini` - GPT-4o mini (cost-effective)
- `o1` - O1 (reasoning model)
- `o1_mini` - O1 mini
- `gpt4_turbo` - GPT-4 Turbo

#### Google Gemini
- `gemini_2_5_pro` - Gemini 2.5 Pro
- `gemini_2_5_flash` - Gemini 2.5 Flash
- `gemini_1_5_pro` - Gemini 1.5 Pro

### Model Mapping

Model names are mapped to CAMEL's `ModelType` in `worldInteract/utils/model_mapping.py`. You can extend this mapping to add more models.

## Using ReactAgent

### Basic Usage

```python
from worldInteract.agents import ReactAgent

# Create agent
agent = ReactAgent(config_key="code_agent")

# Set system prompt
agent.set_system_prompt("You are a helpful coding assistant.")

# Add user message
agent.add_user_message("Please help me validate this code...")

# Get response (automatically added to history)
thinking, content, functions = agent.step()

print(f"Thinking: {thinking}")
print(f"Response: {content}")
```

### ReAct Validation Loop

```python
from worldInteract.agents import ReactAgent

# Create agent
agent = ReactAgent(config_key="code_agent")

# Set system prompt
agent.set_system_prompt("""
You are a code validation agent. Analyze test results and fix issues.
If all tests pass, respond with: ALL TEST CASES PASSED
""")

# Initial setup
agent.add_user_message("Here's the code to validate: ...")

# ReAct loop
for round_num in range(max_rounds):
    # Execute code/tests
    success, message, results = execute_code(...)
    
    # Add observation (automatically added to history)
    observation = format_results(success, message, results)
    agent.add_observation(observation)
    
    # Get agent's response (automatically added to history)
    thinking, content, functions = agent.step()
    
    # Check success condition
    if "ALL TEST CASES PASSED" in content:
        break
    
    # Extract and update code...
```

### Parameter Overrides

```python
# Override temperature and max_tokens for this agent
agent = ReactAgent(
    config_key="code_agent",
    model_config_override={
        "temperature": 0.1,
        "max_tokens": 4096
    }
)

# Or override per step
thinking, content, functions = agent.step(
    temperature=0.5,
    max_tokens=2048
)
```

### History Management

```python
# Get message count
count = agent.get_message_count()

# Get full history
history = agent.get_history()

# Reset agent (keeps system prompt)
agent.reset()

# Clear everything
agent.clear_history()
```

## Model Selection

### Via Configuration File

The recommended way is to configure models in `config/model_config.yaml`:

```yaml
# Use different models for different tasks
code_agent:
  model: "claude_sonnet_4"  # Best for code
  
trajectory_generation:
  model: "gpt4o"  # Good for creative tasks
  
edge_validation:
  model: "gpt4o_mini"  # Cost-effective for simple tasks
```

### Via CamelModelManager

For advanced use cases:

```python
from worldInteract.utils import camel_model_manager

# Create model from config
model = camel_model_manager.create_model("code_agent")

# With overrides
model = camel_model_manager.create_model(
    "code_agent",
    override_params={"temperature": 0.5}
)

# With caching
model = camel_model_manager.get_or_create_model("code_agent")
```

## API Key Management

### Environment Variables

API keys are loaded from the `.env` file. Required environment variables depend on which models you use:

| Provider | Environment Variable | Required For |
|----------|---------------------|--------------|
| Anthropic | `ANTHROPIC_API_KEY` | Claude models |
| OpenAI | `OPENAI_API_KEY` | GPT models |
| Google | `GEMINI_API_KEY` | Gemini models |
| Mistral | `MISTRAL_API_KEY` | Mistral models |

### Security Best Practices

1. **Never commit `.env` file** - It's already in `.gitignore`
2. **Rotate keys regularly** - Especially if exposed
3. **Use separate keys** - Different keys for dev/prod
4. **Monitor usage** - Set billing alerts on provider dashboards
5. **Use cheaper models for testing** - e.g., `gpt4o_mini`, `claude_3_5_haiku`

## Examples

### Complete Code Validation Example

See `examples/react_agent_example.py` for complete examples including:

- Basic usage
- ReAct validation loop
- Parameter overrides
- History management
- Model selection

Run the examples:

```bash
python examples/react_agent_example.py
```

### Integrating with Existing Code

Update your existing code to use ReactAgent:

**Before:**
```python
# Old code_agent.py
class CodeAgent:
    def __init__(self):
        self.history = []  # Manual history management
        
    def validate(self, code):
        # Manually manage history
        self.history.append({"role": "user", "content": code})
        
        # Call model
        response = react_generate(messages=self.history, ...)
        
        # Manually add response
        self.history.append({"role": "assistant", "content": response})
```

**After:**
```python
# New code_agent.py with ReactAgent
from worldInteract.agents import ReactAgent

class CodeAgent:
    def __init__(self):
        self.agent = ReactAgent(config_key="code_agent")
        
    def validate(self, code):
        # System prompt set once
        self.agent.set_system_prompt("You are a code validator...")
        
        # Add user message
        self.agent.add_user_message(f"Validate: {code}")
        
        # Get response (history managed automatically)
        thinking, content, functions = self.agent.step()
        
        return thinking, content, functions
```

## Troubleshooting

### "API key not found" Error

Make sure you have:
1. Created `.env` file from `.env.example`
2. Added the correct API key for your chosen provider
3. The key name matches the provider (e.g., `ANTHROPIC_API_KEY` for Claude)

### "Unknown model name" Error

Check that:
1. The model name in `model_config.yaml` exists in `MODEL_MAPPING`
2. You spelled the model name correctly
3. See available models: `from worldInteract.utils import list_available_models; print(list_available_models())`

### Import Errors

Install CAMEL and dependencies:
```bash
pip install camel-ai[all]
```

Or install specific providers:
```bash
pip install camel-ai anthropic openai
```

## Advanced Topics

### Adding New Models

To add a new model to the system:

1. Add to `worldInteract/utils/model_mapping.py`:
```python
MODEL_MAPPING = {
    # ... existing models ...
    "my_new_model": (ModelPlatformType.ANTHROPIC, ModelType.CLAUDE_NEW),
}
```

2. Use in config:
```yaml
code_agent:
  model: "my_new_model"
```

### Custom Platform Configuration

If you need platform-specific settings:

```python
from camel.configs import AnthropicConfig

# Create custom config
config = AnthropicConfig(
    temperature=0.3,
    max_tokens=4096,
    top_p=0.9,
    stop_sequences=["END"],
)

# Use with model manager
model = camel_model_manager.create_model(
    "code_agent",
    override_params=config.as_dict()
)
```

## References

- [CAMEL GitHub](https://github.com/camel-ai/camel)
- [CAMEL Documentation](https://docs.camel-ai.org/)
- [Anthropic API Docs](https://docs.anthropic.com/)
- [OpenAI API Docs](https://platform.openai.com/docs/)

