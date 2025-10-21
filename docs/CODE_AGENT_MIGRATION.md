# CodeAgent Migration to ReactAgent

## Overview

This document describes the migration of `CodeAgent.generate_and_validate_tool()` from manual history management to using the `ReactAgent` class.

## Changes Summary

### Before (Manual History Management)

```python
class CodeAgent:
    def __init__(self):
        self.history = []  # Manual history tracking
        
    def generate_and_validate_tool(...):
        # Reset history
        self.history = []
        
        # Manually add system message
        self.history.extend([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_prompt}
        ])
        
        while rounds < self.max_rounds:
            # Execute tests...
            
            # Manually add observation
            self.history.append({
                "role": "user",
                "content": f"{observation_title}:\n{observation}"
            })
            
            # Call model with manual history
            thinking_block, answer_block, function_blocks = react_generate(
                model_key=self.model_config["model"],
                messages=self.history,
                ...
            )
            
            # Manually add assistant response
            self.history.append({
                "role": "assistant", 
                "content": [thinking_block, answer_block, *function_blocks]
            })
            
            # Extract text manually
            answer_text = self.extract_text_from_answer_block(answer_block)
```

**Problems:**
- ❌ Manual history management is error-prone
- ❌ Need to manually format and append messages
- ❌ Complex response parsing logic
- ❌ Tightly coupled to specific model response format

---

### After (Using ReactAgent)

```python
class CodeAgent:
    def __init__(self):
        # No history needed - ReactAgent manages it
        pass
        
    def generate_and_validate_tool(...):
        # Create fresh ReactAgent for this session
        agent = ReactAgent(config_key="code_agent")
        
        # Set system prompt
        agent.set_system_prompt(system_prompt)
        
        # Add initial message
        agent.add_user_message(initial_prompt)
        
        while rounds < self.max_rounds:
            # Execute tests...
            
            # Add observation (automatically added to history)
            agent.add_observation(f"{observation_title}:\n{observation}")
            
            # Get response (automatically added to history)
            thinking, answer_text, function_calls = agent.step(
                temperature=self.model_config.get("temperature", 0.3),
                max_tokens=self.model_config.get("max_tokens", 12288)
            )
            
            # Extract code directly from text
            # (no need for manual response parsing)
```

**Benefits:**
- ✅ Automatic history management
- ✅ Clean, simple API
- ✅ Response components separated (thinking, content, functions)
- ✅ Model-agnostic (works with any CAMEL backend)
- ✅ Follows CAMEL's standard patterns

---

## Key Improvements

### 1. Automatic History Management

**Before:**
```python
self.history = []
self.history.append({"role": "user", "content": message})
self.history.append({"role": "assistant", "content": response})
```

**After:**
```python
agent.add_user_message(message)  # Auto-added to history
thinking, content, functions = agent.step()  # Auto-added to history
```

### 2. Cleaner Response Handling

**Before:**
```python
thinking_block, answer_block, function_blocks = react_generate(...)
answer_text = self.extract_text_from_answer_block(answer_block)
```

**After:**
```python
thinking, answer_text, function_calls = agent.step()
# Already extracted and ready to use
```

### 3. CAMEL Standards Compliance

**Before:**
- Custom response format with ContentBlock
- Manual history structure
- Direct model API calls

**After:**
- Uses BaseMessage with meta_dict for metadata
- CAMEL's ChatHistoryMemory
- CAMEL's ModelFactory and backends

### 4. Better Separation of Concerns

**Before:**
- CodeAgent manages both validation logic AND history
- Tightly coupled to Anthropic's response format

**After:**
- CodeAgent focuses on validation logic
- ReactAgent handles all history and model interaction
- Works with any CAMEL-supported model

---

## Migration Guide

### For Other Agent Classes

If you have similar agents with manual history management, follow these steps:

#### Step 1: Update Imports

```python
# Remove
from worldInteract.utils.model_manager import react_generate
from anthropic.types import ContentBlock
from tenacity import RetryError

# Add
from worldInteract.agents import ReactAgent
```

#### Step 2: Remove Manual History

```python
class YourAgent:
    def __init__(self):
        # Remove
        # self.history = []
        
        # ReactAgent will be created per session
        pass
```

#### Step 3: Replace History Management

```python
def your_method(...):
    # Create ReactAgent
    agent = ReactAgent(config_key="your_config_key")
    
    # Set system prompt
    agent.set_system_prompt("Your system prompt...")
    
    # Add initial message
    agent.add_user_message("Initial task...")
    
    # Replace manual history.append with agent methods
    agent.add_observation("Observation text...")
    
    # Replace react_generate with agent.step
    thinking, content, functions = agent.step()
```

#### Step 4: Update Response Parsing

```python
# Remove custom parsing methods
# def extract_text_from_answer_block(self, answer_block): ...

# Use direct string access
# thinking, content, functions = agent.step()
# content already contains the text
```

---

## Testing Considerations

### 1. History Inspection

**Before:**
```python
print(f"History length: {len(self.history)}")
for msg in self.history:
    print(f"{msg['role']}: {msg['content'][:50]}...")
```

**After:**
```python
print(f"History length: {agent.get_message_count()}")
for msg in agent.get_history():
    print(f"{msg['role']}: {msg['content'][:50]}...")
```

### 2. State Reset

**Before:**
```python
self.history = []  # Manual reset
```

**After:**
```python
agent.reset()  # Resets while keeping system prompt
# Or create new agent for complete fresh start
agent = ReactAgent(config_key="code_agent")
```

### 3. Debugging

ReactAgent provides better debugging capabilities:

```python
# Get message count
count = agent.get_message_count()

# Get full history
history = agent.get_history()

# Get detailed response info (with metadata)
details = agent.get_last_response_details()
```

---

## Performance Considerations

### Memory Usage

- **Before:** Each CodeAgent instance maintains its own history
- **After:** ReactAgent instance created per validation session, garbage collected after

### Model Calls

- No change in number of model calls
- Same ReAct loop logic
- Model caching handled by CAMEL's ModelFactory

---

## Backward Compatibility

### Breaking Changes

- ❌ `self.history` no longer exists
- ❌ `extract_text_from_answer_block()` removed
- ❌ `react_generate()` no longer used

### Migration Path

1. Update imports
2. Replace history management with ReactAgent
3. Update any code that directly accessed `self.history`
4. Test validation flows

---

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Code Lines** | ~100 lines | ~70 lines |
| **History Management** | Manual | Automatic |
| **Model Switching** | Hard-coded | Config-based |
| **Response Parsing** | Custom logic | Built-in |
| **CAMEL Compliance** | Partial | Full |
| **Maintainability** | Medium | High |
| **Testability** | Medium | High |

---

## Next Steps

1. ✅ CodeAgent migrated to ReactAgent
2. ⏭️ Consider migrating TaskAgent if it has similar patterns
3. ⏭️ Add unit tests for ReactAgent integration
4. ⏭️ Monitor performance in production

---

## References

- [ReactAgent Documentation](./CAMEL_INTEGRATION.md#using-reactagent)
- [CAMEL ChatAgent Design]( reference/camel/camel/agents/chat_agent.py)
- [ReactAgent Implementation](../worldInteract/agents/react_agent.py)

---

**Migration Date:** 2025-01-XX  
**Status:** ✅ Complete  
**Lines Changed:** ~50 lines  
**Breaking Changes:** Yes (internal only)  
**Testing:** Required

