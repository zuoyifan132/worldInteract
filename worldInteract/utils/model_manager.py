"""
简化的模型管理器

支持动态导入模型模块，采用简单的工厂模式
"""

import sys
import importlib
from typing import Callable, Any
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed


# 默认模型配置
DEFAULT_MODEL = "qwen3_32b"


def get_model_generator(model_name: str, func_name: str = "generate") -> Callable[[str, str, Any], str]:
    """
    获取模型生成器函数
    
    Args:
        model_name: 模型名称，对应 model_generator 目录下的模块名
        func_name: 要获取的函数名（generate, react_generate, stream_generate等）
        
    Returns:
        模型生成函数
    """
    try:
        # 动态导入模型模块
        model_module = importlib.import_module(f"worldInteract.utils.model_generator.{model_name}")
        logger.info(f"使用 {model_name} 模型的 {func_name} 函数进行生成")
        
        # 优先检查指定的函数名
        if hasattr(model_module, func_name):
            return getattr(model_module, func_name)
        
        # 回退到默认检查顺序（向后兼容）
        if hasattr(model_module, "generate"):
            generate_func = model_module.generate
        elif hasattr(model_module, "stream_generate"):
            generate_func = model_module.stream_generate
        else:
            raise AttributeError(f"模块 `{model_name}` 中不存在 '{func_name}', 'generate' 或 'stream_generate' 方法!")
            
        return generate_func
        
    except ImportError as e:
        logger.error(f"导入模型模块失败: {model_name}, 错误: {e}")
        # 回退到默认模型
        if model_name != DEFAULT_MODEL:
            logger.warning(f"回退到默认模型: {DEFAULT_MODEL}")
            return get_model_generator(DEFAULT_MODEL, func_name)
        else:
            raise ImportError(f"无法导入模型模块: {model_name}")


@retry(
    wait=wait_fixed(10),
    stop=stop_after_attempt(3)
)
def generate(model_key: str, system_prompt: str = None, user_prompt: str = None, messages: list = None, **kwargs) -> str:
    """
    使用指定模型生成文本（支持两种调用方式）
    
    Args:
        model_key: 模型名称
        system_prompt: 系统提示词（方式1）
        user_prompt: 用户输入（方式1）
        messages: 预组织的消息数组（方式2）
        **kwargs: 其他参数
        
    Returns:
        生成的文本
        
    Note:
        - 方式1: 传入 system_prompt 和 user_prompt
        - 方式2: 传入 messages 数组
        - 两种方式不能同时使用
    """
    # 验证参数
    if messages is not None:
        if system_prompt is not None or user_prompt is not None:
            raise ValueError("不能同时传入 messages 和 system_prompt/user_prompt")
    else:
        if system_prompt is None or user_prompt is None:
            raise ValueError("必须传入 messages 或者 (system_prompt 和 user_prompt)")
    
    generate_func = get_model_generator(model_key)
    
    # 传递正确的参数
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
    ReAct专用生成函数
    
    Args:
        model_key: 模型名称
        messages: ReAct对话历史消息数组
        **kwargs: 其他参数
        
    Returns:
        生成的文本
    """
    # 优先使用模型的react_generate函数，如果没有则回退到generate
    try:
        react_func = get_model_generator(model_key, "react_generate")
        return react_func(messages=messages, **kwargs)
    except AttributeError:
        # 如果模型没有react_generate函数，使用标准generate
        logger.info(f"模型 {model_key} 没有react_generate函数，使用标准generate")
        generate_func = get_model_generator(model_key, "generate")
        return generate_func(messages=messages, **kwargs)


def stream_generate(model_key: str, system_prompt: str, user_prompt: str, **kwargs):
    """
    使用指定模型进行流式生成（兼容原有接口）
    
    Args:
        model_key: 模型名称（兼容原有的model_key参数）
        system_prompt: 系统提示词
        user_prompt: 用户输入
        **kwargs: 其他参数
        
    Yields:
        生成的文本片段
    """
    generate_func = get_model_generator(model_key)
    
    # 如果是流式函数，直接返回
    if hasattr(generate_func, '__name__') and 'stream' in generate_func.__name__:
        yield from generate_func(system_prompt, user_prompt, **kwargs)
    else:
        # 如果不是流式函数，一次性返回
        result = generate_func(system_prompt, user_prompt, **kwargs)
        yield result
