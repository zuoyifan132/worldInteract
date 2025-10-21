"""
Model name mapping: configuration file model names -> CAMEL ModelType

This module provides mapping between simple model names used in configuration files
and CAMEL's ModelPlatformType and ModelType enums. It also provides helper functions
to retrieve model information and API key environment variable names.
"""

from camel.types import ModelPlatformType, ModelType
from typing import Tuple


# Model name mapping table: config_name -> (platform, model_type)
MODEL_MAPPING = {
    # ============================================================================
    # Anthropic Claude Series
    # ============================================================================
    "claude_3d7": (ModelPlatformType.ANTHROPIC, ModelType.CLAUDE_3_7_SONNET),
    "claude_sonnet_4": (ModelPlatformType.ANTHROPIC, ModelType.CLAUDE_SONNET_4),
    "claude_opus_4": (ModelPlatformType.ANTHROPIC, ModelType.CLAUDE_OPUS_4),
    "claude_opus_4_1": (ModelPlatformType.ANTHROPIC, ModelType.CLAUDE_OPUS_4_1),
    "claude_3_5_sonnet": (ModelPlatformType.ANTHROPIC, ModelType.CLAUDE_3_5_SONNET),
    "claude_3_5_haiku": (ModelPlatformType.ANTHROPIC, ModelType.CLAUDE_3_5_HAIKU),
    "claude_3_opus": (ModelPlatformType.ANTHROPIC, ModelType.CLAUDE_3_OPUS),
    "claude_3_sonnet": (ModelPlatformType.ANTHROPIC, ModelType.CLAUDE_3_SONNET),
    "claude_3_haiku": (ModelPlatformType.ANTHROPIC, ModelType.CLAUDE_3_HAIKU),
    
    # ============================================================================
    # OpenAI GPT Series
    # ============================================================================
    "openai_gpt": (ModelPlatformType.OPENAI, ModelType.GPT_4O),
    "gpt4o": (ModelPlatformType.OPENAI, ModelType.GPT_4O),
    "gpt4o_mini": (ModelPlatformType.OPENAI, ModelType.GPT_4O_MINI),
    "gpt4_turbo": (ModelPlatformType.OPENAI, ModelType.GPT_4_TURBO),
    "gpt4": (ModelPlatformType.OPENAI, ModelType.GPT_4),
    "gpt3_5_turbo": (ModelPlatformType.OPENAI, ModelType.GPT_3_5_TURBO),
    
    # OpenAI O1 Series
    "o1": (ModelPlatformType.OPENAI, ModelType.O1),
    "o1_mini": (ModelPlatformType.OPENAI, ModelType.O1_MINI),
    "o1_preview": (ModelPlatformType.OPENAI, ModelType.O1_PREVIEW),
    
    # ============================================================================
    # Google Gemini Series
    # ============================================================================
    "gemini_2_5_pro": (ModelPlatformType.GEMINI, ModelType.GEMINI_2_5_PRO),
    "gemini_2_5_flash": (ModelPlatformType.GEMINI, ModelType.GEMINI_2_5_FLASH),
    "gemini_2_0_flash": (ModelPlatformType.GEMINI, ModelType.GEMINI_2_0_FLASH_EXP),
    "gemini_1_5_pro": (ModelPlatformType.GEMINI, ModelType.GEMINI_1_5_PRO),
    "gemini_1_5_flash": (ModelPlatformType.GEMINI, ModelType.GEMINI_1_5_FLASH),
    
    # ============================================================================
    # Other Providers
    # ============================================================================
    # Can add more providers as needed: Mistral, Cohere, etc.
}


# API Key environment variable mapping: platform -> env_var_name
API_KEY_ENV_MAP = {
    ModelPlatformType.ANTHROPIC: "ANTHROPIC_API_KEY",
    ModelPlatformType.OPENAI: "OPENAI_API_KEY",
    ModelPlatformType.GEMINI: "GEMINI_API_KEY",
    ModelPlatformType.MISTRAL: "MISTRAL_API_KEY",
    ModelPlatformType.GROQ: "GROQ_API_KEY",
    ModelPlatformType.OLLAMA: None,  # Ollama doesn't require API key
    ModelPlatformType.VLLM: None,  # vLLM doesn't require API key
}


def get_model_info(model_name: str) -> Tuple[ModelPlatformType, ModelType]:
    """
    Get CAMEL model information from configuration model name.
    
    Args:
        model_name: Model name from config file, e.g., "claude_3d7", "gpt4o"
        
    Returns:
        Tuple of (ModelPlatformType, ModelType)
        
    Raises:
        ValueError: If model_name is not in the mapping table
        
    Example:
        >>> platform, model_type = get_model_info("claude_3d7")
        >>> print(platform)  # ModelPlatformType.ANTHROPIC
        >>> print(model_type)  # ModelType.CLAUDE_3_7_SONNET
    """
    if model_name not in MODEL_MAPPING:
        available_models = ", ".join(sorted(MODEL_MAPPING.keys()))
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Available models: {available_models}"
        )
    
    return MODEL_MAPPING[model_name]


def get_api_key_env_name(platform: ModelPlatformType) -> str:
    """
    Get the environment variable name for API key of the specified platform.
    
    Args:
        platform: Model platform type
        
    Returns:
        Environment variable name, e.g., "ANTHROPIC_API_KEY"
        
    Example:
        >>> env_name = get_api_key_env_name(ModelPlatformType.ANTHROPIC)
        >>> print(env_name)  # "ANTHROPIC_API_KEY"
    """
    env_name = API_KEY_ENV_MAP.get(platform)
    
    if env_name is None:
        # For platforms without API key requirement, return None
        # For unknown platforms, construct a default name
        if platform in [ModelPlatformType.OLLAMA, ModelPlatformType.VLLM]:
            return None
        return f"{platform.value.upper()}_API_KEY"
    
    return env_name


def list_available_models() -> list:
    """
    List all available model names that can be used in configuration.
    
    Returns:
        Sorted list of available model names
    """
    return sorted(MODEL_MAPPING.keys())


def get_platform_models(platform: ModelPlatformType) -> list:
    """
    Get all model names for a specific platform.
    
    Args:
        platform: Model platform type
        
    Returns:
        List of model names for the specified platform
        
    Example:
        >>> anthropic_models = get_platform_models(ModelPlatformType.ANTHROPIC)
        >>> print(anthropic_models)  # ['claude_3d7', 'claude_sonnet_4', ...]
    """
    return [
        name for name, (plat, _) in MODEL_MAPPING.items()
        if plat == platform
    ]

