#!/usr/bin/env python3
"""
Example showing how to use the new generate function with two calling modes.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def demonstrate_new_generate_usage():
    """Demonstrate the new generate function usage patterns."""
    
    print("=" * 70)
    print("New Generate Function - Two Calling Modes")
    print("=" * 70)
    
    print("\n🔴 OLD APPROACH - Only support system + user:")
    print("-" * 50)
    
    old_code = '''
# 原来的调用方式
from worldInteract.utils.model_manager import generate

# 只能这样调用
result = generate(
    model_key="claude_3d7",
    system_prompt="You are a helpful assistant",
    user_prompt="Hello, how are you?"
)
'''
    
    print("OLD CODE:")
    print(old_code)
    
    print("\n🟢 NEW APPROACH - Support both modes:")
    print("-" * 50)
    
    new_code = '''
from worldInteract.utils.model_manager import generate

# 方式1: 传统方式（向后兼容）
result1 = generate(
    model_key="claude_3d7",
    system_prompt="You are a helpful assistant",
    user_prompt="Hello, how are you?"
)

# 方式2: 消息数组方式（新功能）
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "Hello! I'm doing well, thank you for asking."},
    {"role": "user", "content": "Can you help me with coding?"}
]

result2 = generate(
    model_key="claude_3d7",
    messages=messages
)

# ❌ 错误用法 - 不能同时传入两种参数
# result3 = generate(
#     model_key="claude_3d7", 
#     system_prompt="system",
#     user_prompt="user",
#     messages=messages  # 这会报错
# )
'''
    
    print("NEW CODE:")
    print(new_code)
    
    print("\n📋 CODE AGENT 使用场景:")
    print("-" * 50)
    
    code_agent_example = '''
# CodeAgent 中的使用示例
class CodeAgent:
    def __init__(self):
        self.history = []
    
    def validate_tool(self, ...):
        # 初始对话
        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user_prompt}
        ]
        
        for round in range(max_rounds):
            # 🟢 NEW: 使用 messages 方式进行多轮对话
            thinking, answer, function_calls = generate(
                model_key=self.model_config["model"],
                messages=self.history  # 直接传入对话历史
            )
            
            # 添加 assistant 响应到历史
            self.history.append({
                "role": "assistant", 
                "content": f"<thought>{thinking}</thought>\\n{answer}"
            })
            
            # 执行代码并添加观察结果
            observation = execute_code(...)
            self.history.append({
                "role": "user",
                "content": f"Execution Results:\\n{observation}"
            })
            
            # 下一轮自动包含完整对话历史
'''
    
    print("CODE AGENT EXAMPLE:")
    print(code_agent_example)
    
    print("\n🔧 MODEL GENERATOR 更新:")
    print("-" * 50)
    
    model_generator_changes = '''
# 所有 model generator 现在支持：

def generate(system: str = None, user: str = None, messages: list = None, **kwargs):
    """
    Args:
        system: 系统提示（方式1）
        user: 用户消息（方式1）  
        messages: 预组织的消息数组（方式2）
    """
    
    if messages is not None:
        # 使用 messages 数组
        api_messages = messages
        # 可能需要从消息中提取 system prompt
    else:
        # 使用传统的 system + user 方式
        api_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    
    # 调用相应的 API...
'''
    
    print("MODEL GENERATOR CHANGES:")
    print(model_generator_changes)
    
    print("\n✅ 新方式的优势:")
    print("-" * 50)
    print("1. 🔄 支持多轮对话历史")
    print("2. 🧠 更好的上下文保持")
    print("3. 🎯 ReAct Agent 可以直接传入历史")
    print("4. 📈 向后兼容原有调用方式")
    print("5. 🛡️ 参数验证防止错误使用")
    print("6. 🚀 简化 Agent 实现逻辑")
    
    print("\n📝 使用规则:")
    print("-" * 50)
    print("• 方式1: system_prompt + user_prompt（传统）")
    print("• 方式2: messages 数组（新功能）")
    print("• ❌ 不能同时使用两种方式")
    print("• ✅ model_manager 会自动验证参数")
    print("• ✅ 所有 model generator 都支持")


if __name__ == "__main__":
    demonstrate_new_generate_usage()
