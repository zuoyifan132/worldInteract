"""
Random Walker for generating function call sequences from subgraphs.
Supports both DAG walks (with parallel branches) and Chain walks (linear paths).
"""

import json
import random
import uuid
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
from tqdm import tqdm

from worldInteract.utils.config_manager import config_manager
from worldInteract.utils.model_manager import generate
from worldInteract.utils.io_utils import extract_json


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
        self.dag_merge_probability = self.config.get('dag_merge_probability', 0.25)
        self.max_parallel_branches = self.config.get('max_parallel_branches', 3)
        
        # Edge validation configuration
        self.enable_edge_validation = self.config.get('enable_edge_validation', True)
        edge_validation_model_config = self.config_manager.get_model_config("edge_validation")
        self.edge_validation_model = edge_validation_model_config.get('model', 'claude_3d7')
        self.min_matching_score = self.config.get('min_matching_score', 0.5)
        
        logger.info("Random walker initialization completed")
        logger.info(f"Walk length range: {self.min_walk_length}-{self.max_walk_length}")
        logger.info(f"Walk types: {[wt.value for wt in self.walk_types]}")
        logger.info(f"Walks per subgraph: {self.num_walks_per_subgraph}")
        logger.info(f"Edge validation enabled: {self.enable_edge_validation}")
    
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
        
        # Create a working copy of the subgraph for validation
        working_graph = subgraph.copy()
        
        walks = []
        attempts = 0
        max_attempts = num_walks * 20
        
        while len(walks) < num_walks and attempts < max_attempts:
            # Randomly select walk type
            walk_type = random.choice(self.walk_types)
            
            # Generate walk
            walk = self._generate_walk(working_graph, walk_type)
            
            # Validate edges and clean both walk and graph immediately after generation
            if walk:
                cleaned_walk, cleaned_graph = self._validate_walk(walk, working_graph)
            else:
                cleaned_graph = working_graph
                cleaned_walk = None
            
            # Update working graph for next iteration
            if cleaned_graph is not None:
                working_graph = cleaned_graph
            else:
                # Graph too small after cleaning, stop
                logger.warning("Graph too small after validation, stopping walk generation")
                break
            
            # Validate and add the cleaned walk
            if cleaned_walk and self._is_valid_walk(cleaned_walk) and not self._is_duplicate(cleaned_walk, walks):
                walks.append(cleaned_walk)
                
                # Save walk if output directory is provided
                if output_dir:
                    self._save_walk(cleaned_walk, output_dir)
            
            attempts += 1
        
        logger.debug(f"Generated {len(walks)} walks in {attempts} attempts")
        logger.debug(f"Final graph size: {working_graph.number_of_nodes()} nodes, {working_graph.number_of_edges()} edges")
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
        
        Strategy: Use "Active Frontier" method to create natural DAG structures with:
        - Branch points: one node connects to multiple successors
        - Merge points: multiple nodes converge to the same target
        - Variable path lengths: different branches can have different depths
        
        Args:
            subgraph: Directed subgraph
            
        Returns:
            DAG walk or None
        """
        nodes = list(subgraph.nodes())
        
        # Find a good starting node (prefer nodes with high out-degree and low in-degree)
        node_scores = []
        for node in nodes:
            in_degree = subgraph.in_degree(node)
            out_degree = subgraph.out_degree(node)
            score = out_degree - in_degree
            node_scores.append((node, score))
        
        node_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [node for node, _ in node_scores[:max(3, len(nodes) // 3)]]
        start_node = random.choice(top_candidates)
        
        # Initialize active frontier and tracking
        frontier = {start_node}  # Nodes that can be expanded
        visited = {start_node}   # All nodes in the DAG
        edges = []               # DAG edges
        node_depths = {start_node: 0}  # Track depth for visualization
        
        target_length = random.randint(self.min_walk_length, self.max_walk_length)
        
        # Probabilities for different expansion strategies
        merge_probability = self.dag_merge_probability  # Probability of merging to existing node
        branch_probability = self.dag_branch_probability  # Probability of creating branches
        
        # Active frontier expansion
        max_iterations = target_length * 3  # Prevent infinite loops
        iteration = 0
        
        while len(visited) < target_length and frontier and iteration < max_iterations:
            iteration += 1
            
            # Select a node from frontier to expand (prefer older nodes for better structure)
            frontier_list = list(frontier)
            # Weight selection by how long node has been in frontier (via depth)
            weights = [1.0 / (node_depths.get(node, 0) + 1) for node in frontier_list]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            current = random.choices(frontier_list, weights=weights, k=1)[0]
            
            current_depth = node_depths.get(current, 0)
            
            # Get available successors (unvisited neighbors in graph)
            available_successors = [n for n in subgraph.successors(current) if n not in visited]
            
            # Get potential merge targets (visited nodes that current can connect to)
            # Only consider nodes that are reachable and at greater depth
            potential_merge_targets = []
            for node in visited:
                if node != current and subgraph.has_edge(current, node):
                    node_depth = node_depths.get(node, 0)
                    # Can merge if target is at similar or greater depth (avoid cycles)
                    if node_depth >= current_depth:
                        potential_merge_targets.append(node)
            
            # Decide expansion strategy
            expansion_type = None
            
            # Try merge first (if available and probabilistic check passes)
            if potential_merge_targets and random.random() < merge_probability:
                expansion_type = 'merge'
            
            # Try branch (if multiple successors available)
            elif len(available_successors) > 1 and random.random() < branch_probability:
                expansion_type = 'branch'
            
            # Default to single extension
            elif available_successors:
                expansion_type = 'extend'
            
            else:
                # No available moves, remove from frontier
                frontier.remove(current)
                continue
            
            # Execute expansion
            if expansion_type == 'merge':
                # Merge: connect to existing visited node
                target = random.choice(potential_merge_targets)
                edges.append((current, target))
                frontier.remove(current)  # Current node is done
                logger.debug(f"Merge: {current} -> {target} (existing)")
            
            elif expansion_type == 'branch':
                # Branch: connect to multiple new nodes
                num_branches = min(
                    random.randint(2, 3),
                    len(available_successors),
                    self.max_parallel_branches,
                    target_length - len(visited)
                )
                selected_targets = random.sample(available_successors, num_branches)
                
                for target in selected_targets:
                    edges.append((current, target))
                    visited.add(target)
                    frontier.add(target)
                    node_depths[target] = current_depth + 1
                
                frontier.remove(current)  # Current node is done after branching
                logger.debug(f"Branch: {current} -> {selected_targets}")
            
            elif expansion_type == 'extend':
                # Extend: connect to single new node
                target = random.choice(available_successors)
                edges.append((current, target))
                visited.add(target)
                frontier.add(target)
                node_depths[target] = current_depth + 1
                frontier.remove(current)  # Current node is done
                logger.debug(f"Extend: {current} -> {target}")
        
        # Check if we have enough nodes
        if len(visited) < self.min_walk_length:
            logger.debug(f"DAG walk too short: {len(visited)} nodes")
            return None
        
        # Build DAG structure for compatibility with existing code
        # Organize nodes into layers by depth
        max_depth = max(node_depths.values()) if node_depths else 0
        layers = []
        
        for depth in range(1, max_depth + 1):
            layer_nodes = [node for node, d in node_depths.items() if d == depth]
            if layer_nodes:
                layers.append({
                    'nodes': layer_nodes,
                    'parallel': len(layer_nodes) > 1
                })
        
        dag_structure = {
            'start': start_node,
            'layers': layers,
            'edges': edges
        }
        
        # Create a flat sequence (topological order) for compatibility
        try:
            # Build a graph from edges to do topological sort
            dag_graph = nx.DiGraph()
            dag_graph.add_node(start_node)
            dag_graph.add_edges_from(edges)
            sequence = list(nx.topological_sort(dag_graph))
        except:
            # Fallback: use depth-based ordering
            sequence = sorted(visited, key=lambda n: node_depths.get(n, 0))
        
        # Calculate metadata
        num_branches = sum(1 for layer in layers if layer.get('parallel', False))
        max_parallelism = max([len(layer['nodes']) for layer in layers] or [1])
        num_merges = len([e for e in edges if any(e[1] == other_e[1] and e != other_e for other_e in edges)])
        
        return RandomWalk(
            id=str(uuid.uuid4()),
            walk_type=WalkType.DAG,
            sequence=sequence,
            dag_structure=dag_structure,
            length=len(sequence),
            metadata={
                'start_node': start_node,
                'num_layers': len(layers),
                'num_branches': num_branches,
                'max_parallelism': max_parallelism,
                'num_merges': num_merges,
                'num_edges': len(edges)
            }
        )
    
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
    
    def _visualize_walk(self, walk: RandomWalk, output_path: Path) -> None:
        """
        Generate walk visualization and save as PNG
        
        Args:
            walk: Random walk to visualize
            output_path: Path to save the PNG file
        """
        try:
            if walk.walk_type == WalkType.CHAIN:
                # Visualize chain walk as a linear sequence
                self._visualize_chain_walk(walk, output_path)
            elif walk.walk_type == WalkType.DAG:
                # Visualize DAG walk with parallel branches
                self._visualize_dag_walk_graph(walk, output_path)
            
            logger.debug(f"Saved walk visualization to {output_path}")
        
        except Exception as e:
            logger.warning(f"Failed to visualize walk: {e}")
    
    def _visualize_chain_walk(self, walk: RandomWalk, output_path: Path) -> None:
        """
        Visualize a chain walk as a linear directed graph
        
        Args:
            walk: Chain walk
            output_path: Path to save visualization
        """
        plt.figure(figsize=(max(12, len(walk.sequence) * 1.5), 6))
        
        # Create a simple directed graph
        G = nx.DiGraph()
        for i in range(len(walk.sequence) - 1):
            G.add_edge(walk.sequence[i], walk.sequence[i + 1])
        
        # Create horizontal layout
        pos = {}
        for i, node in enumerate(walk.sequence):
            pos[node] = (i, 0)
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000, alpha=0.8)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                               arrowsize=20, arrowstyle='->', width=2)
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
        
        plt.title(f"Chain Walk\nID: {walk.id}\nLength: {walk.length}", 
                  fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _visualize_dag_walk_graph(self, walk: RandomWalk, output_path: Path) -> None:
        """
        Visualize a DAG walk with parallel branches
        
        Args:
            walk: DAG walk
            output_path: Path to save visualization
        """
        if not walk.dag_structure:
            # Fallback to chain visualization
            self._visualize_chain_walk(walk, output_path)
            return
        
        plt.figure(figsize=(14, max(10, len(walk.dag_structure['layers']) * 2)))
        
        # Create graph from DAG structure
        G = nx.DiGraph()
        
        # Add all nodes
        for node in walk.sequence:
            G.add_node(node)
        
        # Add edges from DAG structure
        for edge in walk.dag_structure.get('edges', []):
            G.add_edge(edge[0], edge[1])
        
        # Create layered layout
        pos = {}
        start_node = walk.dag_structure['start']
        pos[start_node] = (0, 0)
        
        # Position nodes layer by layer
        y_offset = -2
        for layer_idx, layer in enumerate(walk.dag_structure['layers']):
            nodes = layer['nodes']
            is_parallel = layer.get('parallel', False)
            
            # Center the nodes horizontally
            x_start = -(len(nodes) - 1) / 2 * 3
            
            for i, node in enumerate(nodes):
                pos[node] = (x_start + i * 3, y_offset)
            
            y_offset -= 2
        
        # Determine node colors based on layers
        node_colors = []
        for node in G.nodes():
            if node == start_node:
                node_colors.append('lightgreen')
            else:
                # Find which layer this node belongs to
                layer_found = False
                for layer in walk.dag_structure['layers']:
                    if node in layer['nodes']:
                        if layer.get('parallel', False):
                            node_colors.append('lightcoral')  # Parallel nodes in red
                        else:
                            node_colors.append('lightblue')  # Serial nodes in blue
                        layer_found = True
                        break
                if not layer_found:
                    node_colors.append('lightgray')
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2500, alpha=0.8)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                               arrowsize=20, arrowstyle='->', width=2)
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
        
        # Add legend
        green_patch = mpatches.Patch(color='lightgreen', label='Start Node')
        blue_patch = mpatches.Patch(color='lightblue', label='Serial Node')
        red_patch = mpatches.Patch(color='lightcoral', label='Parallel Node')
        plt.legend(handles=[green_patch, blue_patch, red_patch], loc='upper right')
        
        # Add title with metadata
        max_parallelism = walk.metadata.get('max_parallelism', 1)
        num_layers = walk.metadata.get('num_layers', 0)
        plt.title(f"DAG Walk\nID: {walk.id}\nLength: {walk.length} | Layers: {num_layers} | Max Parallelism: {max_parallelism}", 
                  fontsize=14, fontweight='bold')
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_walk(self, walk: RandomWalk, output_dir: str) -> None:
        """
        Save walk to file (both JSON and PNG visualization)
        
        Args:
            walk: Random walk
            output_dir: Output directory
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save JSON
            walk_file = output_path / f"{walk.id}.json"
            with open(walk_file, 'w', encoding='utf-8') as f:
                json.dump(walk.to_dict(), f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Saved walk {walk.id} to {walk_file}")
            
            # Save PNG visualization
            png_file = output_path / f"{walk.id}.png"
            self._visualize_walk(walk, png_file)
        
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
    
    def _validate_matching_pair_with_llm(
        self, 
        source_node: str,
        target_node: str,
        matching_pair: List[Any],
        subgraph: nx.DiGraph
    ) -> bool:
        """
        Use LLM to validate if a matching_pair is semantically valid
        
        Args:
            source_node: Source node ID
            target_node: Target node ID  
            matching_pair: [source_param, target_param, similarity_score]
            subgraph: The subgraph containing node information
            
        Returns:
            True if matching is valid, False otherwise
        """
        try:
            # Extract matching pair information
            source_param = matching_pair[0]
            target_param = matching_pair[1]
            similarity_score = matching_pair[2]
            
            # Get node descriptions
            source_node_data = subgraph.nodes[source_node]
            target_node_data = subgraph.nodes[target_node]
            
            # Get parameter descriptions
            source_param_desc = ""
            target_param_desc = ""
            
            # Get source parameter description (from returns)
            if 'returns' in source_node_data and source_node_data['returns']:
                returns_data = source_node_data['returns']
                if 'properties' in returns_data and source_param in returns_data['properties']:
                    source_param_desc = returns_data['properties'][source_param].get('description', '')
            
            # Get target parameter description (from parameters)
            if 'parameters' in target_node_data and target_node_data['parameters']:
                params_data = target_node_data['parameters']
                if target_param in params_data:
                    target_param_desc = params_data[target_param].get('description', '')
            
            # Construct prompt for LLM
            system_prompt = """You are an expert at analyzing function parameter compatibility in API workflows.
Your task is to determine if an output parameter from one function can be used as an input parameter for another function.

You should consider:
1. Semantic meaning: Do the parameter descriptions indicate they represent the same or compatible data?
2. Data type compatibility: Can the output type be used as the input type?
3. Logical flow: Does it make sense to pass this data from one function to another?

Respond with ONLY a JSON object in this exact format:
{
    "reasoning": "brief explanation",
    "confidence": 0.0-1.0,
    "is_valid": true/false,
}"""
            
            user_prompt = f"""Analyze this parameter matching:

Source Function: {source_node}
Description: {source_node_data.get('description', 'N/A')}
Output Parameter: {source_param}
Output Description: {source_param_desc if source_param_desc else 'N/A'}

Target Function: {target_node}
Description: {target_node_data.get('description', 'N/A')}
Input Parameter: {target_param}
Input Description: {target_param_desc if target_param_desc else 'N/A'}

Original Similarity Score: {similarity_score}

Is this a valid parameter match? Can the output from source function be used as input to target function?"""
            
            # Call LLM
            thinking_content, answer_text, function_calls = generate(
                model_key=self.edge_validation_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            # Parse response
            # Extract JSON from response (handle markdown code blocks)
            response_text = extract_json(answer_text)
            
            result = json.loads(response_text)
            
            is_valid = result.get('is_valid', False)
            confidence = result.get('confidence', 0.0)
            reasoning = result.get('reasoning', '')
            
            logger.debug(f"Edge validation: {source_node}[{source_param}] -> {target_node}[{target_param}]")
            logger.debug(f"  Valid: {is_valid}, Confidence: {confidence}, Reason: {reasoning}")
            
            # Consider valid if LLM says valid and confidence meets threshold
            return is_valid and confidence >= self.min_matching_score
            
        except Exception as e:
            logger.warning(f"Edge validation failed for {source_node} -> {target_node}: {e}")
            # On error, conservatively keep the edge
            return True
    
    def _validate_walk(
        self,
        walk: RandomWalk,
        subgraph: nx.DiGraph
    ) -> Tuple[Optional[RandomWalk], Optional[nx.DiGraph]]:
        """
        Validate walk edges and clean both walk and graph if needed
        
        This method:
        1. Validates all edges in the walk using LLM
        2. Removes invalid edges from the graph
        3. Cleans up disconnected components, keeping only the largest
        4. Checks if remaining graph meets minimum size requirements
        5. Returns cleaned walk (None if walk contains invalid edges) and cleaned graph
        
        Args:
            walk: The random walk to validate edges
            subgraph: The subgraph containing edge information
            
        Returns:
            Tuple of (cleaned_walk, cleaned_graph):
            - cleaned_walk: Original walk if all edges valid, None if walk contains invalid edges
            - cleaned_graph: Graph with invalid edges removed, or None if graph becomes too small
        """
        if not self.enable_edge_validation:
            return walk, subgraph
        
        # Step 1: Collect all edges in the walk
        edges_in_walk = []
        
        # For DAG walks, use the edges from dag_structure (more accurate)
        if walk.walk_type == WalkType.DAG and walk.dag_structure:
            edges_in_walk = walk.dag_structure.get('edges', [])
        else:
            # For chain walks, derive edges from sequence (consecutive pairs)
            for i in range(len(walk.sequence) - 1):
                source = walk.sequence[i]
                target = walk.sequence[i + 1]
                if subgraph.has_edge(source, target):
                    edges_in_walk.append((source, target))
        
        # Step 2: Validate each edge and collect invalid ones
        invalid_edges = []
        for source, target in edges_in_walk:
            edge_data = subgraph.edges[source, target]
            matching_pairs = edge_data.get('matching_pairs', [])
            
            if not matching_pairs:
                # No matching pairs means edge should be invalid
                invalid_edges.append((source, target))
                continue
            
            # Validate all matching pairs for this edge
            all_pairs_invalid = True
            for matching_pair in matching_pairs:
                if self._validate_matching_pair_with_llm(source, target, matching_pair, subgraph):
                    all_pairs_invalid = False
                    break  # At least one valid pair means edge is valid
            
            if all_pairs_invalid:
                invalid_edges.append((source, target))
        
        # Step 3: If no invalid edges, return original walk and graph
        if not invalid_edges:
            logger.debug(f"All edges in walk {walk.id} are valid")
            return walk, subgraph
        
        # Step 4: Walk contains invalid edges, discard it
        logger.info(f"Found {len(invalid_edges)} invalid edges in walk {walk.id}, discarding walk and cleaning graph...")
        
        # Step 5: Remove invalid edges from graph
        cleaned_graph = subgraph.copy()
        
        for source, target in invalid_edges:
            if cleaned_graph.has_edge(source, target):
                cleaned_graph.remove_edge(source, target)
                logger.debug(f"Removed invalid edge: {source} -> {target}")
        
        # Step 6: Find weakly connected components and keep only the largest
        components = list(nx.weakly_connected_components(cleaned_graph))
        
        if len(components) > 1:
            largest_component = max(components, key=len)
            nodes_to_remove = []
            for component in components:
                if component != largest_component:
                    nodes_to_remove.extend(component)
            
            cleaned_graph.remove_nodes_from(nodes_to_remove)
            logger.debug(f"Removed {len(nodes_to_remove)} nodes from smaller components")
            logger.debug(f"Kept largest component with {len(largest_component)} nodes")
        
        # Step 7: Check if cleaned graph meets minimum requirements
        if cleaned_graph.number_of_nodes() < self.min_walk_length:
            logger.warning(
                f"Graph too small after cleaning ({cleaned_graph.number_of_nodes()} nodes), "
                f"minimum required: {self.min_walk_length}"
            )
            return None, None
        
        # Walk is invalid but graph can be reused for next walk generation
        return None, cleaned_graph
