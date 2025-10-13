"""
Random Walker for generating function call sequences from subgraphs.
Supports both DAG walks (with parallel branches) and Chain walks (linear paths).
"""

import json
import random
import uuid
import networkx as nx
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
from tqdm import tqdm

from worldInteract.utils.config_manager import config_manager


# Set random seed for reproducibility
random.seed(42)


class WalkType(Enum):
    """Random walk type enumeration"""
    CHAIN = "chain"  # Linear path: A → B → C → D
    DAG = "dag"      # DAG with branches: A → [B, C] → D (parallel execution)


@dataclass
class RandomWalk:
    """Random walk data structure"""
    id: str
    walk_type: WalkType
    sequence: List[str]  # Linear sequence for chain walks
    dag_structure: Optional[Dict[str, Any]] = None  # DAG structure for dag walks
    length: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'id': self.id,
            'walk_type': self.walk_type.value,
            'sequence': self.sequence,
            'dag_structure': self.dag_structure,
            'length': self.length,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RandomWalk':
        """Create random walk from dictionary"""
        return cls(
            id=data['id'],
            walk_type=WalkType(data['walk_type']),
            sequence=data['sequence'],
            dag_structure=data.get('dag_structure'),
            length=data.get('length', len(data['sequence'])),
            metadata=data.get('metadata', {})
        )


class RandomWalker:
    """
    Random Walker
    
    Generates random walks on directed subgraphs, supporting both:
    - Chain walks: Linear sequences (serial execution)
    - DAG walks: Directed acyclic paths with branches (parallel execution)
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize random walker
        
        Args:
            config_dir: Configuration directory (optional)
        """
        self.config_manager = config_manager
        self.config = self.config_manager.get_environment_config("task_generation")
        
        # Walk configuration
        self.min_walk_length = self.config.get('min_walk_length', 5)
        self.max_walk_length = self.config.get('max_walk_length', 10)
        self.num_walks_per_subgraph = self.config.get('num_walks_per_subgraph', 2)
        
        # Walk types to generate
        walk_types_config = self.config.get('walk_types', ['dag', 'chain'])
        self.walk_types = [WalkType(wt) for wt in walk_types_config]
        
        # DAG-specific configuration
        self.dag_branch_probability = self.config.get('dag_branch_probability', 0.3)
        self.max_parallel_branches = self.config.get('max_parallel_branches', 3)
        
        logger.info("Random walker initialization completed")
        logger.info(f"Walk length range: {self.min_walk_length}-{self.max_walk_length}")
        logger.info(f"Walk types: {[wt.value for wt in self.walk_types]}")
        logger.info(f"Walks per subgraph: {self.num_walks_per_subgraph}")
    
    def generate_walks(
        self,
        subgraph: nx.DiGraph,
        num_walks: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> List[RandomWalk]:
        """
        Generate random walks on a subgraph
        
        Args:
            subgraph: Directed subgraph to walk on
            num_walks: Number of walks to generate (uses config if None)
            output_dir: Directory to save walks (optional)
            
        Returns:
            List of random walks
        """
        if num_walks is None:
            num_walks = self.num_walks_per_subgraph
        
        if subgraph.number_of_nodes() < self.min_walk_length:
            logger.warning(f"Subgraph too small ({subgraph.number_of_nodes()} nodes) for walks")
            return []
        
        walks = []
        attempts = 0
        max_attempts = num_walks * 20
        
        while len(walks) < num_walks and attempts < max_attempts:
            # Randomly select walk type
            walk_type = random.choice(self.walk_types)
            
            # Generate walk
            walk = self._generate_walk(subgraph, walk_type)
            
            if walk and self._is_valid_walk(walk) and not self._is_duplicate(walk, walks):
                walks.append(walk)
                
                # Save walk if output directory is provided
                if output_dir:
                    self._save_walk(walk, output_dir)
            
            attempts += 1
        
        logger.debug(f"Generated {len(walks)} walks in {attempts} attempts")
        return walks
    
    def generate_walks_batch(
        self,
        subgraphs: List[nx.DiGraph],
        output_dir: Optional[str] = None
    ) -> Dict[str, List[RandomWalk]]:
        """
        Generate walks for multiple subgraphs
        
        Args:
            subgraphs: List of subgraphs
            output_dir: Directory to save walks (optional)
            
        Returns:
            Dictionary mapping subgraph indices to their walks
        """
        all_walks = {}
        
        for idx, subgraph in enumerate(tqdm(subgraphs, desc="Generating walks")):
            walks = self.generate_walks(subgraph, output_dir=output_dir)
            if walks:
                all_walks[f"subgraph_{idx}"] = walks
        
        logger.info(f"Generated walks for {len(all_walks)}/{len(subgraphs)} subgraphs")
        total_walks = sum(len(walks) for walks in all_walks.values())
        logger.info(f"Total walks generated: {total_walks}")
        
        return all_walks
    
    def _generate_walk(
        self,
        subgraph: nx.DiGraph,
        walk_type: WalkType
    ) -> Optional[RandomWalk]:
        """
        Generate a single random walk
        
        Args:
            subgraph: Directed subgraph
            walk_type: Type of walk to generate
            
        Returns:
            Random walk or None if generation failed
        """
        try:
            if walk_type == WalkType.CHAIN:
                return self._generate_chain_walk(subgraph)
            elif walk_type == WalkType.DAG:
                return self._generate_dag_walk(subgraph)
            else:
                logger.warning(f"Unknown walk type: {walk_type}")
                return None
        
        except Exception as e:
            logger.debug(f"Walk generation failed ({walk_type}): {e}")
            return None
    
    def _generate_chain_walk(self, subgraph: nx.DiGraph) -> Optional[RandomWalk]:
        """
        Generate a chain walk (linear path)
        
        Strategy: Follow directed edges to create a linear sequence
        
        Args:
            subgraph: Directed subgraph
            
        Returns:
            Chain walk or None
        """
        nodes = list(subgraph.nodes())
        
        # Try to find a good starting node (prefer nodes with no predecessors or few predecessors)
        node_scores = []
        for node in nodes:
            in_degree = subgraph.in_degree(node)
            out_degree = subgraph.out_degree(node)
            # Prefer nodes with few incoming edges and many outgoing edges
            score = out_degree - in_degree
            node_scores.append((node, score))
        
        # Sort by score and randomly select from top candidates
        node_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [node for node, _ in node_scores[:max(3, len(nodes) // 3)]]
        start_node = random.choice(top_candidates)
        
        # Build chain by following edges
        sequence = [start_node]
        current = start_node
        visited = {start_node}
        
        target_length = random.randint(self.min_walk_length, self.max_walk_length)
        
        while len(sequence) < target_length:
            successors = [n for n in subgraph.successors(current) if n not in visited]
            
            if not successors:
                # Try to continue from any unvisited node with edges
                unvisited_nodes = [n for n in nodes if n not in visited and subgraph.out_degree(n) > 0]
                if unvisited_nodes:
                    current = random.choice(unvisited_nodes)
                    sequence.append(current)
                    visited.add(current)
                else:
                    break
            else:
                # Prefer successors with more outgoing edges (longer chains)
                successor_scores = [(s, subgraph.out_degree(s)) for s in successors]
                successor_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Randomly select from top successors
                top_successors = [s for s, _ in successor_scores[:max(2, len(successors) // 2)]]
                next_node = random.choice(top_successors)
                
                sequence.append(next_node)
                visited.add(next_node)
                current = next_node
        
        if len(sequence) >= self.min_walk_length:
            return RandomWalk(
                id=str(uuid.uuid4()),
                walk_type=WalkType.CHAIN,
                sequence=sequence,
                length=len(sequence),
                metadata={
                    'start_node': start_node,
                    'end_node': sequence[-1]
                }
            )
        
        return None
    
    def _generate_dag_walk(self, subgraph: nx.DiGraph) -> Optional[RandomWalk]:
        """
        Generate a DAG walk (directed acyclic graph with parallel branches)
        
        Strategy: Start from a node and allow branching (parallel paths) that
        eventually converge to common nodes.
        
        Args:
            subgraph: Directed subgraph
            
        Returns:
            DAG walk or None
        """
        nodes = list(subgraph.nodes())
        
        # Find a good starting node
        node_scores = []
        for node in nodes:
            in_degree = subgraph.in_degree(node)
            out_degree = subgraph.out_degree(node)
            score = out_degree - in_degree
            node_scores.append((node, score))
        
        node_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [node for node, _ in node_scores[:max(3, len(nodes) // 3)]]
        start_node = random.choice(top_candidates)
        
        # Build DAG structure
        dag_structure = {
            'start': start_node,
            'layers': [],  # List of layers, each layer contains nodes that can execute in parallel
            'edges': []    # List of (from, to) edges
        }
        
        visited = {start_node}
        current_layer = [start_node]
        all_nodes = [start_node]
        
        target_length = random.randint(self.min_walk_length, self.max_walk_length)
        
        while len(all_nodes) < target_length and current_layer:
            next_layer = []
            layer_edges = []
            
            for node in current_layer:
                successors = [n for n in subgraph.successors(node) if n not in visited]
                
                if not successors:
                    continue
                
                # Decide whether to branch (create parallel paths)
                should_branch = random.random() < self.dag_branch_probability and len(successors) > 1
                
                if should_branch:
                    # Select multiple successors for parallel execution
                    num_branches = min(
                        len(successors),
                        self.max_parallel_branches,
                        target_length - len(all_nodes)
                    )
                    selected_successors = random.sample(successors, num_branches)
                else:
                    # Select single successor for serial execution
                    selected_successors = [random.choice(successors)]
                
                for successor in selected_successors:
                    if successor not in visited and len(all_nodes) < target_length:
                        next_layer.append(successor)
                        visited.add(successor)
                        all_nodes.append(successor)
                        layer_edges.append((node, successor))
            
            if next_layer:
                dag_structure['layers'].append({
                    'nodes': next_layer,
                    'parallel': len(next_layer) > 1
                })
                dag_structure['edges'].extend(layer_edges)
                current_layer = next_layer
            else:
                break
        
        # Create a flat sequence (topological order) for compatibility
        sequence = [start_node]
        for layer in dag_structure['layers']:
            sequence.extend(layer['nodes'])
        
        if len(sequence) >= self.min_walk_length:
            return RandomWalk(
                id=str(uuid.uuid4()),
                walk_type=WalkType.DAG,
                sequence=sequence,
                dag_structure=dag_structure,
                length=len(sequence),
                metadata={
                    'start_node': start_node,
                    'num_layers': len(dag_structure['layers']),
                    'num_branches': sum(1 for layer in dag_structure['layers'] if layer.get('parallel', False)),
                    'max_parallelism': max([len(layer['nodes']) for layer in dag_structure['layers']] or [1])
                }
            )
        
        return None
    
    def _is_valid_walk(self, walk: RandomWalk) -> bool:
        """
        Validate a random walk
        
        Args:
            walk: Random walk to validate
            
        Returns:
            True if valid
        """
        # Check length
        if walk.length < self.min_walk_length or walk.length > self.max_walk_length:
            return False
        
        # Check for empty sequence
        if not walk.sequence:
            return False
        
        # For DAG walks, validate structure
        if walk.walk_type == WalkType.DAG and walk.dag_structure:
            # Check that DAG structure is valid
            if 'start' not in walk.dag_structure or 'layers' not in walk.dag_structure:
                return False
        
        return True
    
    def _is_duplicate(self, new_walk: RandomWalk, existing_walks: List[RandomWalk]) -> bool:
        """
        Check if walk is duplicate or too similar to existing walks
        
        Args:
            new_walk: New walk
            existing_walks: List of existing walks
            
        Returns:
            True if duplicate
        """
        new_sequence_set = set(new_walk.sequence)
        
        for existing in existing_walks:
            existing_sequence_set = set(existing.sequence)
            
            # Check if sequences are identical
            if new_walk.sequence == existing.sequence:
                return True
            
            # Check if node sets are too similar (Jaccard similarity > 0.8)
            intersection = len(new_sequence_set & existing_sequence_set)
            union = len(new_sequence_set | existing_sequence_set)
            
            if union > 0:
                jaccard_similarity = intersection / union
                if jaccard_similarity > 0.8:
                    return True
        
        return False
    
    def _save_walk(self, walk: RandomWalk, output_dir: str) -> None:
        """
        Save walk to file
        
        Args:
            walk: Random walk
            output_dir: Output directory
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            walk_file = output_path / f"{walk.id}.json"
            
            with open(walk_file, 'w', encoding='utf-8') as f:
                json.dump(walk.to_dict(), f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Saved walk {walk.id} to {walk_file}")
        
        except Exception as e:
            logger.error(f"Failed to save walk: {e}")
    
    def load_walk(self, walk_file: str) -> Optional[RandomWalk]:
        """
        Load walk from file
        
        Args:
            walk_file: Path to walk file
            
        Returns:
            Random walk or None if failed
        """
        try:
            with open(walk_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return RandomWalk.from_dict(data)
        
        except Exception as e:
            logger.error(f"Failed to load walk from {walk_file}: {e}")
            return None
    
    def load_all_walks(self, output_dir: str) -> List[RandomWalk]:
        """
        Load all walks from directory
        
        Args:
            output_dir: Directory containing walk files
            
        Returns:
            List of random walks
        """
        walks = []
        output_path = Path(output_dir)
        
        if not output_path.exists():
            logger.warning(f"Output directory does not exist: {output_dir}")
            return walks
        
        for walk_file in output_path.glob("*.json"):
            walk = self.load_walk(str(walk_file))
            if walk:
                walks.append(walk)
        
        logger.info(f"Loaded {len(walks)} walks from {output_dir}")
        return walks
    
    def visualize_dag_walk(self, walk: RandomWalk) -> str:
        """
        Create a text visualization of a DAG walk
        
        Args:
            walk: Random walk with DAG structure
            
        Returns:
            Text visualization
        """
        if walk.walk_type != WalkType.DAG or not walk.dag_structure:
            return f"Chain walk: {' → '.join(walk.sequence)}"
        
        lines = []
        lines.append(f"DAG Walk (ID: {walk.id[:8]}...)")
        lines.append(f"Start: {walk.dag_structure['start']}")
        lines.append("")
        
        for i, layer in enumerate(walk.dag_structure['layers']):
            nodes = layer['nodes']
            is_parallel = layer.get('parallel', False)
            
            if is_parallel:
                lines.append(f"Layer {i+1} (Parallel):")
                for node in nodes:
                    lines.append(f"  ├─ {node}")
            else:
                lines.append(f"Layer {i+1}:")
                lines.append(f"  └─ {nodes[0]}")
            lines.append("")
        
        return "\n".join(lines)
