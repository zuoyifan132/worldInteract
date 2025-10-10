"""
WorldInteract: A scalable framework for automatic environment construction 
and agentic intelligence training.
"""

__version__ = "0.1.0"
__author__ = "WorldInteract Team"

from .core.build_environment import EnvironmentManager, SchemaGenerator, ToolGenerator, CodeAgent

__all__ = [
    "EnvironmentManager",
    "SchemaGenerator", 
    "ToolGenerator",
    "CodeAgent"
]

