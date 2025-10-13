"""
Task Generator for creating agent tasks from random walks.
Uses LLM to filter valid walks and generate task descriptions.
"""

import json
import textwrap
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger
from tqdm import tqdm

from worldInteract.utils.config_manager import config_manager
from worldInteract.utils.model_manager import generate
from worldInteract.utils.parser_utils import extract_json_from_text
from worldInteract.core.build_task_graph.random_walker import RandomWalk, WalkType


@dataclass
class AgentTask:
    """Agent task data structure"""
    id: str
    task_description: str
    function_sequence: List[str]
    parameter_flow: List[Dict[str, Any]]
    walk_id: str
    walk_type: str
    complexity_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'id': self.id,
            'task_description': self.task_description,
            'function_sequence': self.function_sequence,
            'parameter_flow': self.parameter_flow,
            'walk_id': self.walk_id,
            'walk_type': self.walk_type,
            'complexity_score': self.complexity_score,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentTask':
        """Create agent task from dictionary"""
        return cls(
            id=data['id'],
            task_description=data['task_description'],
            function_sequence=data['function_sequence'],
            parameter_flow=data['parameter_flow'],
            walk_id=data['walk_id'],
            walk_type=data['walk_type'],
            complexity_score=data.get('complexity_score', 0.0),
            metadata=data.get('metadata', {})
        )


class TaskGenerator:
    """
    Task Generator
    
    Generates agent tasks from random walks using LLM:
    1. Filters walks for validity (parameter matching, semantic coherence)
    2. Generates task descriptions and parameter flows
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize task generator
        
        Args:
            config_dir: Configuration directory (optional)
        """
        self.config_manager = config_manager
        self.task_config = self.config_manager.get_environment_config("task_generation")
        self.model_config = self.config_manager.get_model_config("task_generation")
        
        # Task generation configuration
        self.enable_llm_filter = self.task_config.get('enable_llm_filter', True)
        self.enable_llm_task_generation = self.task_config.get('enable_llm_task_generation', True)
        self.min_task_complexity = self.task_config.get('min_task_complexity', 3)
        self.max_task_complexity = self.task_config.get('max_task_complexity', 10)
        self.llm_filter_batch_size = self.task_config.get('llm_filter_batch_size', 5)
        
        logger.info("Task generator initialization completed")
        logger.info(f"LLM filter enabled: {self.enable_llm_filter}")
        logger.info(f"LLM task generation enabled: {self.enable_llm_task_generation}")
        logger.info(f"Task complexity range: {self.min_task_complexity}-{self.max_task_complexity}")
    
    def generate_tasks(
        self,
        walks: List[RandomWalk],
        tool_definitions: Dict[str, Dict[str, Any]],
        output_dir: Optional[str] = None
    ) -> List[AgentTask]:
        """
        Generate agent tasks from random walks
        
        Args:
            walks: List of random walks
            tool_definitions: Dictionary mapping tool names to their definitions
            output_dir: Directory to save tasks (optional)
            
        Returns:
            List of generated agent tasks
        """
        logger.info(f"Generating tasks from {len(walks)} random walks")
        
        # Step 1: Filter valid walks using LLM
        if self.enable_llm_filter:
            logger.info("Step 1: Filtering walks with LLM...")
            valid_walks = self._filter_walks_with_llm(walks, tool_definitions)
            logger.info(f"Filtered to {len(valid_walks)} valid walks")
        else:
            valid_walks = walks
            logger.info("Step 1: Skipping LLM filtering")
        
        if not valid_walks:
            logger.warning("No valid walks found")
            return []
        
        # Step 2: Generate tasks from valid walks
        if self.enable_llm_task_generation:
            logger.info("Step 2: Generating tasks with LLM...")
            tasks = self._generate_tasks_with_llm(valid_walks, tool_definitions)
            logger.info(f"Generated {len(tasks)} tasks")
        else:
            logger.warning("LLM task generation is disabled")
            tasks = []
        
        # Step 3: Save tasks
        if output_dir and tasks:
            logger.info("Step 3: Saving tasks...")
            self._save_tasks(tasks, output_dir)
        
        return tasks
    
    def _filter_walks_with_llm(
        self,
        walks: List[RandomWalk],
        tool_definitions: Dict[str, Dict[str, Any]]
    ) -> List[RandomWalk]:
        """
        Filter walks using LLM to check validity
        
        Criteria:
        - Parameter type matching between consecutive functions
        - Semantic coherence of function sequence
        - Required parameters can be obtained
        
        Args:
            walks: List of random walks
            tool_definitions: Tool definitions
            
        Returns:
            List of valid walks
        """
        valid_walks = []
        
        # Process in batches
        for i in tqdm(range(0, len(walks), self.llm_filter_batch_size), desc="Filtering walks"):
            batch = walks[i:i + self.llm_filter_batch_size]
            
            for walk in batch:
                try:
                    is_valid, reason = self._validate_walk_with_llm(walk, tool_definitions)
                    
                    if is_valid:
                        valid_walks.append(walk)
                        logger.debug(f"Walk {walk.id[:8]} is valid: {reason}")
                    else:
                        logger.debug(f"Walk {walk.id[:8]} is invalid: {reason}")
                
                except Exception as e:
                    logger.error(f"Failed to validate walk {walk.id}: {e}")
        
        return valid_walks
    
    def _validate_walk_with_llm(
        self,
        walk: RandomWalk,
        tool_definitions: Dict[str, Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Validate a single walk using LLM
        
        Args:
            walk: Random walk
            tool_definitions: Tool definitions
            
        Returns:
            (is_valid, reason) tuple
        """
        # Prepare function sequence information
        function_info = []
        for func_name in walk.sequence:
            if func_name in tool_definitions:
                func_def = tool_definitions[func_name]
                function_info.append({
                    'name': func_name,
                    'description': func_def.get('description', ''),
                    'parameters': func_def.get('parameters', {}),
                    'returns': func_def.get('returns', {})
                })
        
        if not function_info:
            return False, "No valid function definitions found"
        
        system_prompt = textwrap.dedent("""
            You are an API call sequence validation expert. Please analyze the given function call sequence and determine whether it can form a valid agent task.
            
            Validation criteria:
            1. **Parameter Matching**: Can the output parameter types and semantics of the previous function serve as input to the next function?
            2. **Semantic Coherence**: Does the function call sequence have logical meaning and can it complete a coherent task?
            3. **Parameter Availability**: Can all required parameters be obtained from previous function outputs or external inputs?
            4. **Task Completeness**: Can the sequence achieve a meaningful goal?
            
            Please respond in JSON format:
            ```json
            {
                "is_valid": true/false,
                "reason": "Detailed explanation of the validation result",
                "parameter_issues": ["List parameter matching issues (if any)"],
                "semantic_issues": ["List semantic coherence issues (if any)"]
            }
            ```
        """).strip()
        
        # Format walk structure for LLM
        if walk.walk_type == WalkType.DAG and walk.dag_structure:
            walk_desc = f"DAG Structure (supports parallel execution):\nStart: {walk.dag_structure['start']}\n"
            for i, layer in enumerate(walk.dag_structure['layers']):
                nodes = layer['nodes']
                is_parallel = layer.get('parallel', False)
                if is_parallel:
                    walk_desc += f"Layer {i+1} (Parallel): {', '.join(nodes)}\n"
                else:
                    walk_desc += f"Layer {i+1}: {nodes[0]}\n"
        else:
            walk_desc = f"Chain Structure: {' → '.join(walk.sequence)}"
        
        user_prompt = textwrap.dedent(f"""
            Please validate whether the following function call sequence is valid:
            
            **Call Structure**:
            {walk_desc}
            
            **Function Definitions**:
            {json.dumps(function_info, indent=2, ensure_ascii=False)}
            
            Please analyze whether this sequence can form a valid agent task.
        """).strip()
        
        try:
            thinking_content, answer_text, function_calls = generate(
                self.model_config["model"],
                system_prompt,
                user_prompt,
                temperature=self.model_config.get("temperature", 0.1),
                max_tokens=800
            )
            
            # Parse response
            extracted_json = extract_json_from_text(answer_text.strip())
            response_data = json.loads(extracted_json)
            
            is_valid = response_data.get('is_valid', False)
            reason = response_data.get('reason', 'No reason provided')
            
            return is_valid, reason
        
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            # Default to invalid if validation fails
            return False, f"Validation error: {str(e)}"
    
    def _generate_tasks_with_llm(
        self,
        walks: List[RandomWalk],
        tool_definitions: Dict[str, Dict[str, Any]]
    ) -> List[AgentTask]:
        """
        Generate tasks from valid walks using LLM
        
        Args:
            walks: List of valid walks
            tool_definitions: Tool definitions
            
        Returns:
            List of generated tasks
        """
        tasks = []
        
        for walk in tqdm(walks, desc="Generating tasks"):
            try:
                task = self._generate_task_from_walk(walk, tool_definitions)
                if task:
                    tasks.append(task)
            
            except Exception as e:
                logger.error(f"Failed to generate task from walk {walk.id}: {e}")
        
        return tasks
    
    def _generate_task_from_walk(
        self,
        walk: RandomWalk,
        tool_definitions: Dict[str, Dict[str, Any]]
    ) -> Optional[AgentTask]:
        """
        Generate a single task from a walk
        
        Args:
            walk: Random walk
            tool_definitions: Tool definitions
            
        Returns:
            Agent task or None
        """
        # Prepare function sequence information
        function_info = []
        for func_name in walk.sequence:
            if func_name in tool_definitions:
                func_def = tool_definitions[func_name]
                function_info.append({
                    'name': func_name,
                    'description': func_def.get('description', ''),
                    'parameters': func_def.get('parameters', {}),
                    'returns': func_def.get('returns', {})
                })
        
        if not function_info:
            return None
        
        system_prompt = textwrap.dedent("""
            You are an agent task design expert. Please design a complete agent task based on the given function call sequence.
            
            Task requirements:
            1. **Task Description**: Describe in natural language the goal the user wants to achieve
            2. **Function Sequence**: List the functions to be called in execution order
            3. **Parameter Flow**: Explain in detail where each function's input parameters come from (external input or previous function outputs)
            4. **Expected Output**: Describe the expected result after task completion
            
            Please respond in JSON format:
            ```json
            {
                "task_description": "Natural language description of the task",
                "function_sequence": ["func1", "func2", "func3"],
                "parameter_flow": [
                    {
                        "function": "func1",
                        "input_parameters": {
                            "param1": {"source": "user_input", "description": "Parameter provided by user"}
                        },
                        "output_parameters": ["output1", "output2"]
                    },
                    {
                        "function": "func2",
                        "input_parameters": {
                            "param1": {"source": "func1.output1", "description": "Output from func1"}
                        },
                        "output_parameters": ["result"]
                    }
                ],
                "expected_output": "Description of the expected result after task completion",
                "complexity_score": 0.7
            }
            ```
        """).strip()
        
        # Format walk structure
        if walk.walk_type == WalkType.DAG and walk.dag_structure:
            walk_desc = f"DAG Structure (supports parallel execution):\nStart: {walk.dag_structure['start']}\n"
            for i, layer in enumerate(walk.dag_structure['layers']):
                nodes = layer['nodes']
                is_parallel = layer.get('parallel', False)
                if is_parallel:
                    walk_desc += f"Layer {i+1} (Parallel): {', '.join(nodes)}\n"
                else:
                    walk_desc += f"Layer {i+1}: {nodes[0]}\n"
            walk_desc += "\nFunctions in parallel layers can execute simultaneously, and their outputs can be used by subsequent functions."
        else:
            walk_desc = f"Chain Structure: {' → '.join(walk.sequence)}"
        
        user_prompt = textwrap.dedent(f"""
            Please design an agent task based on the following function call sequence:
            
            **Call Structure**:
            {walk_desc}
            
            **Function Definitions**:
            {json.dumps(function_info, indent=2, ensure_ascii=False)}
            
            Please design a reasonable task that explains how to use these functions to accomplish a meaningful goal.
        """).strip()
        
        try:
            thinking_content, answer_text, function_calls = generate(
                self.model_config["model"],
                system_prompt,
                user_prompt,
                temperature=self.model_config.get("temperature", 0.3),
                max_tokens=1500
            )
            
            # Parse response
            extracted_json = extract_json_from_text(answer_text.strip())
            response_data = json.loads(extracted_json)
            
            task = AgentTask(
                id=f"task_{walk.id}",
                task_description=response_data.get('task_description', ''),
                function_sequence=response_data.get('function_sequence', walk.sequence),
                parameter_flow=response_data.get('parameter_flow', []),
                walk_id=walk.id,
                walk_type=walk.walk_type.value,
                complexity_score=response_data.get('complexity_score', 0.5),
                metadata={
                    'expected_output': response_data.get('expected_output', ''),
                    'walk_metadata': walk.metadata
                }
            )
            
            return task
        
        except Exception as e:
            logger.error(f"Failed to generate task with LLM: {e}")
            return None
    
    def _save_tasks(self, tasks: List[AgentTask], output_dir: str) -> None:
        """
        Save tasks to directory
        
        Args:
            tasks: List of agent tasks
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual tasks
        for task in tasks:
            task_file = output_path / f"{task.id}.json"
            with open(task_file, 'w', encoding='utf-8') as f:
                json.dump(task.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Save summary
        summary = {
            'total_tasks': len(tasks),
            'task_ids': [task.id for task in tasks],
            'statistics': {
                'avg_complexity': sum(t.complexity_score for t in tasks) / len(tasks) if tasks else 0,
                'avg_sequence_length': sum(len(t.function_sequence) for t in tasks) / len(tasks) if tasks else 0,
                'walk_types': {
                    'chain': sum(1 for t in tasks if t.walk_type == 'chain'),
                    'dag': sum(1 for t in tasks if t.walk_type == 'dag')
                }
            }
        }
        
        summary_file = output_path / "tasks_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(tasks)} tasks to {output_dir}")
        logger.info(f"Summary: {summary['statistics']}")
    
    def load_task(self, task_file: str) -> Optional[AgentTask]:
        """
        Load task from file
        
        Args:
            task_file: Path to task file
            
        Returns:
            Agent task or None
        """
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return AgentTask.from_dict(data)
        
        except Exception as e:
            logger.error(f"Failed to load task from {task_file}: {e}")
            return None
    
    def load_all_tasks(self, output_dir: str) -> List[AgentTask]:
        """
        Load all tasks from directory
        
        Args:
            output_dir: Directory containing task files
            
        Returns:
            List of agent tasks
        """
        tasks = []
        output_path = Path(output_dir)
        
        if not output_path.exists():
            logger.warning(f"Output directory does not exist: {output_dir}")
            return tasks
        
        for task_file in output_path.glob("task_*.json"):
            task = self.load_task(str(task_file))
            if task:
                tasks.append(task)
        
        logger.info(f"Loaded {len(tasks)} tasks from {output_dir}")
        return tasks

