"""
Configuration management for WorldInteract.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """Manages configuration files for different tasks and models."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            # Default to config directory relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"
        
        self.config_dir = Path(config_dir)
        self._model_config = None
        self._env_config = None
    
    def load_model_config(self) -> Dict[str, Any]:
        """Load model configuration."""
        if self._model_config is None:
            config_file = self.config_dir / "model_config.yaml"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    self._model_config = yaml.safe_load(f)
            else:
                # Default configuration
                self._model_config = {
                    "default": {
                        "model": "qwen3_32b",
                        "temperature": 0.7,
                        "max_tokens": 4000,
                        "retry_attempts": 3
                    }
                }
        return self._model_config
    
    def load_environment_config(self) -> Dict[str, Any]:
        """Load environment configuration."""
        if self._env_config is None:
            config_file = self.config_dir / "environment_config.yaml"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    self._env_config = yaml.safe_load(f)
            else:
                # Default configuration
                self._env_config = {
                    "general": {
                        "max_retries": 3,
                        "timeout_seconds": 120,
                        "enable_logging": True,
                        "log_level": "INFO"
                    }
                }
        return self._env_config
    
    def get_model_config(self, task_name: str) -> Dict[str, Any]:
        """
        Get model configuration for a specific task.
        
        Args:
            task_name: Name of the task (e.g., 'schema_generation', 'tool_generation')
            
        Returns:
            Model configuration dictionary
        """
        model_config = self.load_model_config()
        
        if task_name in model_config:
            return model_config[task_name]
        else:
            # Fallback to default configuration
            return model_config.get("default", {
                "model": "qwen3_32b",
                "temperature": 0.7,
                "max_tokens": 4000,
                "retry_attempts": 3
            })
    
    def get_domain_config(self, domain_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific domain.
        
        Args:
            domain_name: Name of the domain
            
        Returns:
            Domain configuration dictionary
        """
        env_config = self.load_environment_config()
        domains_config = env_config.get("domains", {})
        
        return domains_config.get(domain_name, {})
    
    def get_environment_config(self, task_name: str) -> Dict[str, Any]:
        """
        Get environment configuration for a specific task.
        
        Args:
            task_name: Name of the task (e.g., 'scenario_collection', 'tool_generation')
            
        Returns:
            Environment configuration dictionary
        """
        env_config = self.load_environment_config()
        
        if task_name in env_config:
            return env_config[task_name]
        else:
            # Fallback to general configuration
            return env_config.get("general", {
                "max_retries": 3,
                "timeout_seconds": 120,
                "enable_logging": True,
                "log_level": "INFO"
            })
    
    def get_general_config(self) -> Dict[str, Any]:
        """Get general configuration settings."""
        env_config = self.load_environment_config()
        return env_config.get("general", {})


# Global config manager instance
config_manager = ConfigManager()

