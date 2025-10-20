"""
Trajectory Generator - Main orchestrator for generating complete interaction trajectories.
Integrates task preparation, ReAct agent execution, and state tracking.
"""

import json
import copy
import os
from loguru import logger
from typing import Dict, Any, List
from pathlib import Path

from worldInteract.trajectories.prepare_task import TaskPreparer
from worldInteract.trajectories.task_agent import TaskAgent


class TrajectoryGenerator:
    """Main generator for creating complete interaction trajectories from random walks."""
    
    def __init__(
        self,
        domain_tools_path: str,
        env_domain_path: str
    ):
        """
        Initialize TrajectoryGenerator.
        
        Args:
            domain_tools_path: Path to domain tools JSON file (e.g., data/domain_graphs/.../domains/file_operations.json)
            env_domain_path: Path to environment domain directory (e.g., data/generated_env/domains/file_operations/)
        """
        self.domain_tools_path = domain_tools_path
        self.env_domain_path = env_domain_path
        
        # Load domain tools information
        with open(domain_tools_path, 'r', encoding='utf-8') as f:
            self.domain_tools = json.load(f)
        
        # Initialize components
        self.task_preparer = TaskPreparer()
        self.task_agent = TaskAgent(self.domain_tools, env_domain_path)
        
        logger.info(f"Initialized TrajectoryGenerator for domain: {self.domain_tools['domain']}")
    
    def generate_trajectory(
        self,
        random_walk: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate complete trajectory from a random walk.
        
        Args:
            random_walk: Random walk data with sequence, nodes, edges, dag_structure
            
        Returns:
            Complete trajectory with:
            - domain information
            - all tool descriptions
            - complete interaction history (user queries, model responses, tool calls, tool results)
            - state snapshots after each tool execution
        """
        walk_id = random_walk["id"]
        logger.info(f"Generating trajectory for random walk: {walk_id}")
        
        # Step 1: Generate user queries for each node
        logger.info("Step 1: Generating user queries...")
        user_queries = self.task_preparer.generate_user_queries(
            random_walk=random_walk,
            domain_info=self.domain_tools,
            initial_state=self.task_agent.initial_state
        )
        
        # Step 2: Execute tasks sequentially using ReAct agent
        logger.info("Step 2: Executing tasks with ReAct agent...")
        complete_history = []
        state_snapshots = []
        current_state = copy.deepcopy(self.task_agent.initial_state)
        
        # Record initial state
        state_snapshots.append({
            "snapshot_id": 0,
            "description": "Initial state before any operations",
            "state": copy.deepcopy(current_state)
        })
        
        for idx, query_info in enumerate(user_queries):
            node_id = query_info["node_id"]
            user_query = query_info["user_query"]
            
            logger.info(f"Executing query {idx+1}/{len(user_queries)} for node: {node_id}")
            logger.info(f"User query: {user_query}")
            
            # Execute task with ReAct agent
            task_history, new_state = self.task_agent.execute_task(
                user_query=user_query,
                available_tools=[node_id],  # Only make this tool available for this step
                current_state=current_state
            )
            
            # Add task history to complete history
            complete_history.extend(task_history)
            
            # Update current state
            current_state = new_state
            
            # Record state snapshot
            state_snapshots.append({
                "snapshot_id": idx + 1,
                "after_node": node_id,
                "after_query": user_query,
                "state": copy.deepcopy(current_state)
            })
            
            logger.info(f"Completed execution for node: {node_id}")
        
        # Step 3: Build complete trajectory
        logger.info("Step 3: Building complete trajectory...")
        trajectory = self._build_trajectory(
            random_walk, user_queries, complete_history, state_snapshots
        )
        
        logger.info(f"Successfully generated trajectory for walk: {walk_id}")
        return trajectory
    
    def _build_trajectory(
        self,
        random_walk: Dict[str, Any],
        user_queries: List[Dict[str, Any]],
        complete_history: List[Dict[str, Any]],
        state_snapshots: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build the final trajectory structure."""
        
        trajectory = {
            # Metadata
            "trajectory_id": random_walk["id"],
            "walk_type": random_walk["walk_type"],
            
            # Domain information
            "domain": self.domain_tools["domain"],
            "domain_description": self.domain_tools["description"],
            
            # Tools information
            "tools": self.domain_tools["tools"],
            
            # Random walk information
            "sequence": random_walk["sequence"],
            "dag_structure": random_walk.get("dag_structure", {}),
            "walk_metadata": random_walk.get("metadata", {}),
            
            # User queries
            "user_queries": user_queries,
            
            # Complete interaction history
            "interaction_history": self._format_history(complete_history),
            
            # State snapshots
            "state_snapshots": state_snapshots,
            
            # Statistics
            "statistics": {
                "num_nodes": len(random_walk["sequence"]),
                "num_user_queries": len(user_queries),
                "num_interactions": len(complete_history),
                "num_state_changes": len(state_snapshots) - 1,  # Exclude initial state
            }
        }
        
        return trajectory
    
    def _format_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format interaction history for better readability."""
        formatted_history = []
        
        for item in history:
            role = item["role"]
            content = item["content"]
            
            # Check if this is a tool result message and change role to "tool_response"
            is_tool_result = False
            if role == "user" and isinstance(content, list):
                # Check if content contains tool_result
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        is_tool_result = True
                        break
            
            formatted_item = {
                "role": "tool_response" if is_tool_result else role
            }
            
            # Format content based on type
            if isinstance(content, str):
                formatted_item["content"] = content
            elif isinstance(content, list):
                # Handle content blocks (thinking, text, tool_use, tool_result)
                formatted_content = []
                for block in content:
                    if hasattr(block, 'type'):
                        # ContentBlock from anthropic
                        formatted_block = self._format_content_block(block)
                        formatted_content.append(formatted_block)
                    elif isinstance(block, dict):
                        # Already a dict
                        formatted_content.append(block)
                    else:
                        formatted_content.append(str(block))
                
                formatted_item["content"] = formatted_content
            else:
                formatted_item["content"] = str(content)
            
            formatted_history.append(formatted_item)
        
        return formatted_history
    
    def _format_content_block(self, block) -> Dict[str, Any]:
        """Format ContentBlock from anthropic API."""
        formatted = {
            "type": block.type
        }
        
        if block.type == "text":
            formatted["text"] = block.text
        elif block.type == "thinking":
            formatted["thinking"] = block.thinking if hasattr(block, 'thinking') else ""
        elif block.type == "tool_use":
            formatted["tool_use"] = {
                "id": block.id,
                "name": block.name,
                "input": block.input
            }
        
        return formatted
    
    def generate_trajectories_batch(
        self,
        random_walks_dir: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Generate trajectories for all random walks in a directory.
        
        Args:
            random_walks_dir: Directory containing random walk JSON files
            output_dir: Directory to save generated trajectories
            
        Returns:
            Summary of generation results
        """
        logger.info(f"Batch generating trajectories from: {random_walks_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all random walk files
        random_walk_files = list(Path(random_walks_dir).glob("*.json"))
        logger.info(f"Found {len(random_walk_files)} random walk files")
        
        # Process each random walk
        results = {
            "total": len(random_walk_files),
            "successful": 0,
            "failed": 0,
            "failed_walks": []
        }
        
        for walk_file in random_walk_files:
            walk_id = walk_file.stem
            logger.info(f"\nProcessing random walk: {walk_id}")
            
            try:
                # Load random walk
                with open(str(walk_file), 'r', encoding='utf-8') as f:
                    random_walk = json.load(f)
                
                # Generate trajectory
                trajectory = self.generate_trajectory(random_walk)
                
                # Save trajectory
                output_path = os.path.join(output_dir, f"{walk_id}.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(trajectory, f, indent=2, ensure_ascii=False)
                
                results["successful"] += 1
                logger.info(f"✓ Successfully generated trajectory: {walk_id}")
                
            except Exception as e:
                results["failed"] += 1
                results["failed_walks"].append({
                    "walk_id": walk_id,
                    "error": str(e)
                })
                logger.error(f"✗ Failed to generate trajectory for {walk_id}: {e}")
        
        # Save summary
        summary_path = os.path.join(output_dir, "_generation_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nBatch generation complete:")
        logger.info(f"  Total: {results['total']}")
        logger.info(f"  Successful: {results['successful']}")
        logger.info(f"  Failed: {results['failed']}")
        
        return results

