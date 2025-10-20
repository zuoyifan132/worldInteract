"""
Trajectories Module

This module provides tools for generating agent task trajectories
from task graphs and random walks.
"""

# Import new trajectory generation modules
from worldInteract.trajectories.prepare_task import TaskPreparer
from worldInteract.trajectories.task_agent import TaskAgent
from worldInteract.trajectories.trajectory_generator import TrajectoryGenerator

# Try to import legacy modules (may fail if dependencies missing)
try:
    from worldInteract.trajectories.task_generator import (
        TaskGenerator,
        AgentTask
    )
    __all__ = [
        'TaskGenerator',
        'AgentTask',
        'TaskPreparer',
        'TaskAgent',
        'TrajectoryGenerator'
    ]
except ImportError:
    __all__ = [
        'TaskPreparer',
        'TaskAgent',
        'TrajectoryGenerator'
    ]

