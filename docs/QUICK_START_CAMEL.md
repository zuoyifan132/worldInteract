# Quick Start: CAMEL Integration

Get started with CAMEL-based model system in 5 minutes!

## 🚀 Quick Setup

### 1. Install CAMEL

```bash
pip install camel-ai[all]
```

Or for specific providers only:
```bash
pip install camel-ai anthropic openai
```

### 2. Set Up API Keys

```bash
# Copy example file
cp .env.example .env

# Edit .env and add your API keys
# ANTHROPIC_API_KEY=sk-ant-xxxxx
# OPENAI_API_KEY=sk-xxxxx
```

### 3. Verify Installation

```python
from worldInteract.agents import ReactAgent
from worldInteract.utils import list_available_models

# List available models
print("Available models:", list_available_models())

# Create a simple agent
agent = ReactAgent(config_key="code_agent")
print("✅ CAMEL integration working!")
```

## 📝 Basic Usage

### Simple Q&A

```python
from worldInteract.agents import ReactAgent

# Create agent
agent = ReactAgent(config_key="code_agent")

# Set system prompt
agent.set_system_prompt("You are a helpful assistant.")

# Ask question
agent.add_user_message("What is 2+2?")

# Get response
thinking, content, functions = agent.step()
print(content)  # "4"
```

### ReAct Loop (Code Validation)

```python
from worldInteract.agents import ReactAgent

# Create agent for code validation
agent = ReactAgent(config_key="code_agent")

# Set up for validation
agent.set_system_prompt("""
You are a code validator. Analyze test results and fix issues.
If all tests pass, respond with: ALL TEST CASES PASSED
""")

# Add code to validate
agent.add_user_message("Here's code to validate: def add(a,b): return a-b")

# ReAct loop
for round in range(5):
    # Add test results
    agent.add_observation(f"Test failed: add(2,3) returned -1, expected 5")
    
    # Get agent response
    thinking, content, functions = agent.step()
    
    if "ALL TEST CASES PASSED" in content:
        print("✅ Validation complete!")
        break
```

## 🎯 Model Selection

### Option 1: Via YAML Config (Recommended)

Edit `config/model_config.yaml`:

```yaml
code_agent:
  model: "claude_sonnet_4"  # Change this!
  temperature: 0.3
  max_tokens: 12288
```

### Option 2: Via Code

```python
agent = ReactAgent(
    config_key="code_agent",
    model_config_override={
        "temperature": 0.1,
        "max_tokens": 4096
    }
)
```

## 🔥 Common Use Cases

### 1. Update Existing CodeAgent

**Old code:**
```python
from worldInteract.utils.model_manager import react_generate

class CodeAgent:
    def __init__(self):
        self.history = []
        
    def validate(self, code):
        self.history.append({"role": "user", "content": code})
        response = react_generate(self.model, messages=self.history)
        self.history.append({"role": "assistant", "content": response})
```

**New code:**
```python
from worldInteract.agents import ReactAgent

class CodeAgent:
    def __init__(self):
        self.agent = ReactAgent(config_key="code_agent")
        self.agent.set_system_prompt("You are a code validator...")
        
    def validate(self, code):
        self.agent.add_user_message(f"Validate: {code}")
        thinking, content, functions = self.agent.step()
        return thinking, content, functions
```

### 2. Multi-Model Setup

Use different models for different tasks:

```yaml
# config/model_config.yaml
code_agent:
  model: "claude_sonnet_4"  # Best for code
  
trajectory_generation:
  model: "gpt4o"  # Good for creative tasks
  
edge_validation:
  model: "gpt4o_mini"  # Cheap for simple validation
```

### 3. Testing Different Models

```python
# Test with Claude
agent_claude = ReactAgent(config_key="code_agent")  # Uses claude_sonnet_4

# Test with GPT (by changing config or override)
agent_gpt = ReactAgent(
    config_key="code_agent",
    model_config_override={"model": "gpt4o"}
)
```

## ⚙️ Configuration

### Available Models

**Anthropic Claude:**
- `claude_sonnet_4` ⭐ (recommended for code)
- `claude_opus_4` (most capable)
- `claude_3d7`
- `claude_3_5_sonnet`
- `claude_3_5_haiku` (fast & cheap)

**OpenAI GPT:**
- `gpt4o` ⭐ (recommended all-purpose)
- `gpt4o_mini` (cheap)
- `o1` (reasoning)
- `gpt4_turbo`

**Google Gemini:**
- `gemini_2_5_pro`
- `gemini_2_5_flash`

### Cost-Effective Choices

For development/testing:
- `gpt4o_mini` - Very cheap, good quality
- `claude_3_5_haiku` - Fast and affordable
- `gemini_2_5_flash` - Free tier available

For production:
- `claude_sonnet_4` - Best for code
- `gpt4o` - Excellent all-rounder
- `gemini_2_5_pro` - Good value

## 🛠️ Troubleshooting

### "API key not found"

```bash
# Check .env file exists
ls -la .env

# Check key is set
grep ANTHROPIC_API_KEY .env

# Make sure .env is in project root
cd /path/to/worldInteract
```

### "Unknown model name"

```python
# List available models
from worldInteract.utils import list_available_models
print(list_available_models())

# Check your config
cat config/model_config.yaml | grep "model:"
```

### Import errors

```bash
# Install CAMEL
pip install camel-ai[all]

# Or minimal install
pip install camel-ai anthropic openai python-dotenv
```

## 📚 Next Steps

- Read full guide: [CAMEL_INTEGRATION.md](./CAMEL_INTEGRATION.md)
- Run examples: `python examples/react_agent_example.py`
- Update your code: Start with replacing manual history management
- Experiment: Try different models to find the best fit

## 💡 Tips

1. **Start simple** - Use one model first, then experiment
2. **Use config file** - Easier to manage than code changes
3. **Monitor costs** - Set billing alerts on provider dashboards
4. **Test with cheap models** - Use `gpt4o_mini` for development
5. **Check history** - Use `agent.get_message_count()` to debug

## 🎉 You're Ready!

You now have:
- ✅ CAMEL installed
- ✅ API keys configured
- ✅ ReactAgent ready to use
- ✅ Model switching capability

Start coding! 🚀

