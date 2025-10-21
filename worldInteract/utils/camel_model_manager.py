"""
CAMEL-based Model Manager

This module provides a unified interface for creating and managing models using CAMEL's
ModelFactory. It reads configuration from model_config.yaml and API keys from .env file,
making it easy to switch between different models and providers.
"""

import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from camel.models import ModelFactory
from camel.configs import (
    AnthropicConfig,
    ChatGPTConfig,
    GeminiConfig,
    MistralConfig,
)
from camel.types import ModelPlatformType

from worldInteract.utils.config_manager import config_manager
from worldInteract.utils.model_mapping import get_model_info, get_api_key_env_name


logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


class CamelModelManager:
    """
    CAMEL-based model manager that creates models from configuration.
    
    This manager:
    - Reads model configuration from model_config.yaml
    - Maps simple model names to CAMEL ModelType
    - Loads API keys from .env file
    - Creates and caches model instances
    - Supports parameter overrides
    
    Example:
        >>> manager = CamelModelManager()
        >>> model = manager.create_model("code_agent")
        >>> # Model is ready to use with configured parameters
    """
    
    def __init__(self):
        """Initialize the model manager."""
        self._model_cache = {}  # Cache for created model instances
        logger.info("CamelModelManager initialized")
    
    def create_model(
        self,
        config_key: str,
        override_params: Optional[Dict[str, Any]] = None
    ):
        """
        Create a CAMEL model instance based on configuration key.
        
        Args:
            config_key: Configuration key from model_config.yaml (e.g., "code_agent")
            override_params: Optional parameters to override config values
                           (e.g., {"temperature": 0.5, "max_tokens": 4096})
        
        Returns:
            CAMEL BaseModelBackend instance ready for use
            
        Raises:
            ValueError: If model name is unknown or API key is missing
            
        Example:
            >>> model = manager.create_model("code_agent")
            >>> # Or with overrides:
            >>> model = manager.create_model(
            ...     "code_agent",
            ...     override_params={"temperature": 0.8}
            ... )
        """
        # Read model configuration from YAML
        model_config = config_manager.get_model_config(config_key)
        
        # Get model name from config
        model_name = model_config.get("model", "claude_3d7")
        logger.info(f"Creating model for config_key='{config_key}', model_name='{model_name}'")
        
        # Map to CAMEL model type
        platform, model_type = get_model_info(model_name)
        
        # Build configuration dictionary
        config_dict = self._build_config_dict(
            platform,
            model_config,
            override_params
        )
        
        # Get API key from environment
        api_key_env = get_api_key_env_name(platform)
        api_key = None
        
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(
                    f"API key not found for {platform.value}. "
                    f"Please set {api_key_env} in your .env file"
                )
            logger.debug(f"API key loaded from environment variable: {api_key_env}")
        
        # Create model using CAMEL ModelFactory
        try:
            model = ModelFactory.create(
                model_platform=platform,
                model_type=model_type,
                model_config_dict=config_dict,
                api_key=api_key
            )
            logger.info(f"Successfully created model: {platform.value}/{model_type.value}")
            return model
        
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise
    
    def _build_config_dict(
        self,
        platform: ModelPlatformType,
        model_config: Dict[str, Any],
        override_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build model configuration dictionary based on platform.
        
        Args:
            platform: Model platform type
            model_config: Configuration from YAML file
            override_params: Optional parameters to override
            
        Returns:
            Configuration dictionary compatible with CAMEL configs
        """
        # Extract base parameters from config
        base_params = {
            "temperature": model_config.get("temperature", 1.0),
            "max_tokens": model_config.get("max_tokens", 8192),
        }
        
        # Apply override parameters
        if override_params:
            base_params.update(override_params)
            logger.debug(f"Applied parameter overrides: {override_params}")
        
        # Create platform-specific configuration object
        config = self._create_platform_config(platform, base_params)
        
        return config.as_dict()
    
    def _create_platform_config(
        self,
        platform: ModelPlatformType,
        params: Dict[str, Any]
    ):
        """
        Create platform-specific configuration object.
        
        Args:
            platform: Model platform type
            params: Configuration parameters
            
        Returns:
            Platform-specific config object (AnthropicConfig, ChatGPTConfig, etc.)
        """
        if platform == ModelPlatformType.ANTHROPIC:
            return AnthropicConfig(**params)
        
        elif platform == ModelPlatformType.OPENAI:
            return ChatGPTConfig(**params)
        
        elif platform == ModelPlatformType.GEMINI:
            return GeminiConfig(**params)
        
        elif platform == ModelPlatformType.MISTRAL:
            return MistralConfig(**params)
        
        else:
            # Fallback to AnthropicConfig for unknown platforms
            logger.warning(
                f"Unknown platform {platform}, using AnthropicConfig as fallback"
            )
            return AnthropicConfig(**params)
    
    def get_or_create_model(
        self,
        config_key: str,
        cache: bool = True,
        **override_params
    ):
        """
        Get existing model from cache or create a new one.
        
        Args:
            config_key: Configuration key from model_config.yaml
            cache: Whether to cache the model instance
            **override_params: Keyword arguments to override config parameters
            
        Returns:
            CAMEL BaseModelBackend instance
            
        Example:
            >>> # First call creates the model
            >>> model1 = manager.get_or_create_model("code_agent")
            >>> # Second call returns cached instance
            >>> model2 = manager.get_or_create_model("code_agent")
            >>> assert model1 is model2  # Same instance
        """
        # Create cache key based on config_key and override params
        cache_key = f"{config_key}_{str(sorted(override_params.items()))}"
        
        if cache and cache_key in self._model_cache:
            logger.debug(f"Returning cached model for cache_key='{cache_key}'")
            return self._model_cache[cache_key]
        
        # Create new model
        model = self.create_model(
            config_key,
            override_params=override_params or None
        )
        
        # Cache the model if requested
        if cache:
            self._model_cache[cache_key] = model
            logger.debug(f"Cached model with cache_key='{cache_key}'")
        
        return model
    
    def clear_cache(self):
        """Clear the model cache."""
        self._model_cache.clear()
        logger.info("Model cache cleared")
    
    def get_cache_size(self) -> int:
        """
        Get the number of cached models.
        
        Returns:
            Number of models in cache
        """
        return len(self._model_cache)


# Global singleton instance
camel_model_manager = CamelModelManager()

