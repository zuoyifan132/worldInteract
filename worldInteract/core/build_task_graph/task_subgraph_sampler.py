"""
Task Subgraph Sampler for sampling subgraphs from task graphs.
"""

import json
import random
import uuid
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
from tqdm import tqdm

from worldInteract.utils.config_manager import config_manager


# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Branching factors for different sampling strategies
BFS_BRANCHING_FACTOR = 3
DFS_BRANCHING_FACTOR = 2
TREE_BRANCHING_FACTOR = 2


class SamplingStrategy(Enum):
    """Sampling strategy enumeration"""
    RANDOM = "random"
    BFS = "bfs"
    DFS = "dfs"
    COMMUNITY = "community"
    STAR = "star"
    CHAIN = "chain"
    TREE = "tree"


@dataclass
class TaskSubgraphData:
    """Task subgraph data class"""
    id: str
    nodes: List[str]  # List of node IDs
    edges: List[Dict[str, Any]]  # List of edge data
    graph: nx.DiGraph  # NetworkX graph object
    strategy: SamplingStrategy
    complexity_score: float = 0.0
    topology_features: Dict[str, Any] = field(default_factory=dict)
    node_details: List[Dict[str, Any]] = field(default_factory=list)  # Complete node information
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'id': self.id,
            'nodes': self.node_details if self.node_details else self.nodes,  # Save full node info if available
            'edges': self.edges,
            'strategy': self.strategy.value,
            'complexity_score': self.complexity_score,
            'topology_features': self.topology_features
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskSubgraphData':
        """Create task subgraph data from dictionary"""
        # Check if nodes contain full information or just IDs
        nodes_data = data.get('nodes', [])
        if nodes_data and isinstance(nodes_data[0], dict):
            # Full node information
            node_ids = [node['id'] for node in nodes_data]
            node_details = nodes_data
        else:
            # Just node IDs (backward compatibility)
            node_ids = nodes_data
            node_details = []
        
        # Rebuild graph
        graph = nx.DiGraph()
        
        # Add nodes with attributes if available
        if node_details:
            for node in node_details:
                node_id = node['id']
                node_attrs = {k: v for k, v in node.items() if k != 'id'}
                graph.add_node(node_id, **node_attrs)
        else:
            graph.add_nodes_from(node_ids)
        
        # Add edges
        for edge in data.get('edges', []):
            graph.add_edge(
                edge['source'],
                edge['target'],
                **{k: v for k, v in edge.items() if k not in ['source', 'target']}
            )
        
        return cls(
            id=data['id'],
            nodes=node_ids,
            edges=data.get('edges', []),
            graph=graph,
            strategy=SamplingStrategy(data.get('strategy', 'random')),
            complexity_score=data.get('complexity_score', 0.0),
            topology_features=data.get('topology_features', {}),
            node_details=node_details
        )


class TaskSubgraphSampler:
    """
    Task Subgraph Sampler
    
    Samples subgraphs with different topological structures from task graphs,
    supporting multiple sampling strategies.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize sampler
        
        Args:
            config_dir: Configuration directory (optional)
        """
        self.config_manager = config_manager
        self.config = self.config_manager.get_environment_config("task_generation")
        
        # Sampling configuration
        self.min_nodes = self.config.get('min_subgraph_nodes', 5)
        self.max_nodes = self.config.get('max_subgraph_nodes', 20)
        self.num_samples = self.config.get('num_subgraphs_per_graph', 10)
        self.diversity_threshold = self.config.get('subgraph_diversity_threshold', 0.3)
        self.save_original_subgraph = self.config.get('save_original_subgraph', True)
        
        # Sampling strategy weights
        strategy_weights_config = self.config.get('sampling_strategies', {})
        self.strategy_weights = {
            SamplingStrategy.RANDOM: strategy_weights_config.get('random', 0.15),
            SamplingStrategy.BFS: strategy_weights_config.get('bfs', 0.2),
            SamplingStrategy.DFS: strategy_weights_config.get('dfs', 0.2),
            SamplingStrategy.COMMUNITY: strategy_weights_config.get('community', 0.15),
            SamplingStrategy.STAR: strategy_weights_config.get('star', 0.1),
            SamplingStrategy.CHAIN: strategy_weights_config.get('chain', 0.1),
            SamplingStrategy.TREE: strategy_weights_config.get('tree', 0.1)
        }
        
        logger.info("Task subgraph sampler initialization completed")
        logger.info(f"Min nodes: {self.min_nodes}, Max nodes: {self.max_nodes}")
        logger.info(f"Target samples per graph: {self.num_samples}")
        logger.info(f"Save original subgraph: {self.save_original_subgraph}")
    
    def sample_subgraphs(
        self,
        task_graph: nx.DiGraph,
        num_samples: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> List[TaskSubgraphData]:
        """
        Sample subgraphs from task graph
        
        Args:
            task_graph: Task graph to sample from
            num_samples: Number of samples (uses config if None)
            output_dir: Directory to save subgraphs (optional)
            
        Returns:
            List of sampled subgraphs
        """
        if num_samples is None:
            num_samples = self.num_samples
        
        logger.info(f"Task graph: {task_graph.number_of_nodes()} nodes, {task_graph.number_of_edges()} edges")
        
        # If save_original_subgraph is True, save the original graph as-is
        if self.save_original_subgraph:
            logger.info("save_original_subgraph=True: Saving original task graph without sampling")
            
            # Create a single subgraph that contains the entire graph
            all_nodes = list(task_graph.nodes())
            
            # Extract complete node information with all attributes
            node_details = []
            for node_id in all_nodes:
                # Get node attributes from the graph
                node_attrs = task_graph.nodes[node_id]
                # Create node data with id and all attributes
                node_data = {'id': node_id, **node_attrs}
                node_details.append(node_data)
            
            # Extract all edges with their attributes
            edges = []
            for u, v, data in task_graph.edges(data=True):
                edge_data = {
                    'source': u,
                    'target': v,
                    **data
                }
                edges.append(edge_data)
            
            # Calculate complexity and topology features
            complexity_score = self._calculate_complexity(task_graph)
            topology_features = self._calculate_topology_features(task_graph)
            
            # Create TaskSubgraphData for the original graph
            original_graph_data = TaskSubgraphData(
                id=str(uuid.uuid4()),
                nodes=all_nodes,
                edges=edges,
                graph=task_graph.copy(),
                strategy=SamplingStrategy.RANDOM,  # Use RANDOM as placeholder
                complexity_score=complexity_score,
                topology_features=topology_features,
                node_details=node_details
            )
            
            # Save if output directory is provided
            if output_dir:
                self._save_subgraph(original_graph_data, output_dir)
            
            logger.info(f"Saved original task graph with {len(all_nodes)} nodes and {len(edges)} edges")
            return [original_graph_data]
        
        # Normal subgraph sampling
        logger.info(f"Starting to sample {num_samples} subgraphs")
        
        subgraphs = []
        attempts = 0
        max_attempts = num_samples * 20  # Allow more attempts
        
        with tqdm(total=num_samples, desc="Sampling subgraphs") as pbar:
            while len(subgraphs) < num_samples and attempts < max_attempts:
                # Randomly select strategy based on weights
                strategy = random.choices(
                    population=list(self.strategy_weights.keys()),
                    weights=list(self.strategy_weights.values()),
                    k=1
                )[0]
                
                # Sample subgraph
                subgraph = self._sample_with_strategy(task_graph, strategy)
                
                if subgraph and self._is_diverse(subgraph, subgraphs):
                    subgraphs.append(subgraph)
                    
                    # Save subgraph if output directory is provided
                    if output_dir:
                        self._save_subgraph(subgraph, output_dir)
                    
                    pbar.update(1)
                
                attempts += 1
        
        logger.info(f"Successfully sampled {len(subgraphs)} subgraphs in {attempts} attempts")
        return subgraphs
    
    def _sample_with_strategy(
        self,
        task_graph: nx.DiGraph,
        strategy: SamplingStrategy
    ) -> Optional[TaskSubgraphData]:
        """
        Sample subgraph using specified strategy
        
        Args:
            task_graph: Task graph
            strategy: Sampling strategy
            
        Returns:
            Sampled subgraph, returns None if failed
        """
        try:
            if strategy == SamplingStrategy.RANDOM:
                return self._random_sampling(task_graph)
            elif strategy == SamplingStrategy.BFS:
                return self._bfs_sampling(task_graph)
            elif strategy == SamplingStrategy.DFS:
                return self._dfs_sampling(task_graph)
            elif strategy == SamplingStrategy.COMMUNITY:
                return self._community_sampling(task_graph)
            elif strategy == SamplingStrategy.STAR:
                return self._star_sampling(task_graph)
            elif strategy == SamplingStrategy.CHAIN:
                return self._chain_sampling(task_graph)
            elif strategy == SamplingStrategy.TREE:
                return self._tree_sampling(task_graph)
            else:
                logger.warning(f"Unknown sampling strategy: {strategy}")
                return None
        
        except Exception as e:
            logger.debug(f"Sampling failed (strategy: {strategy}): {e}")
            return None
    
    def _random_sampling(self, task_graph: nx.DiGraph) -> Optional[TaskSubgraphData]:
        """Random sampling"""
        node_ids = list(task_graph.nodes())
        if len(node_ids) < self.min_nodes:
            return None
        
        # Randomly select number of nodes
        num_nodes = random.randint(self.min_nodes, min(self.max_nodes, len(node_ids)))
        selected_nodes = random.sample(node_ids, num_nodes)
        
        return self._create_subgraph(task_graph, selected_nodes, SamplingStrategy.RANDOM)
    
    def _bfs_sampling(self, task_graph: nx.DiGraph) -> Optional[TaskSubgraphData]:
        """Breadth-first search sampling"""
        node_ids = list(task_graph.nodes())
        if len(node_ids) < self.min_nodes:
            return None
        
        # Randomly select starting node
        start_node = random.choice(node_ids)
        
        # BFS traversal (follow edge direction)
        visited = set()
        queue = [start_node]
        selected_nodes = []
        
        while queue and len(selected_nodes) < self.max_nodes:
            current = queue.pop(0)
            if current not in visited:
                visited.add(current)
                selected_nodes.append(current)
                
                # Add successor nodes (following edge direction)
                successors = list(task_graph.successors(current))
                random.shuffle(successors)
                queue.extend(successors[:BFS_BRANCHING_FACTOR])
        
        if len(selected_nodes) >= self.min_nodes:
            return self._create_subgraph(task_graph, selected_nodes, SamplingStrategy.BFS)
        return None
    
    def _dfs_sampling(self, task_graph: nx.DiGraph) -> Optional[TaskSubgraphData]:
        """Depth-first search sampling"""
        node_ids = list(task_graph.nodes())
        if len(node_ids) < self.min_nodes:
            return None
        
        # Randomly select starting node
        start_node = random.choice(node_ids)
        
        # DFS traversal (follow edge direction)
        visited = set()
        stack = [start_node]
        selected_nodes = []
        
        while stack and len(selected_nodes) < self.max_nodes:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                selected_nodes.append(current)
                
                # Add successor nodes (following edge direction)
                successors = list(task_graph.successors(current))
                random.shuffle(successors)
                stack.extend(successors[:DFS_BRANCHING_FACTOR])
        
        if len(selected_nodes) >= self.min_nodes:
            return self._create_subgraph(task_graph, selected_nodes, SamplingStrategy.DFS)
        return None
    
    def _community_sampling(self, task_graph: nx.DiGraph) -> Optional[TaskSubgraphData]:
        """Community detection sampling"""
        if len(task_graph.nodes()) < self.min_nodes:
            return None
        
        try:
            # Convert to undirected for community detection
            undirected_graph = task_graph.to_undirected()
            
            # Use Louvain algorithm to detect communities
            communities = nx.community.louvain_communities(undirected_graph)
            
            if not communities:
                return None
            
            # Filter communities that meet minimum size requirement
            valid_communities = [c for c in communities if len(c) >= self.min_nodes]
            
            if valid_communities:
                # Randomly select from valid communities
                community = random.choice(valid_communities)
            else:
                # If no valid community, try to merge small communities
                communities_list = sorted(communities, key=lambda c: len(c), reverse=True)
                merged_nodes = set()
                for comm in communities_list:
                    merged_nodes.update(comm)
                    if len(merged_nodes) >= self.min_nodes:
                        break
                
                if len(merged_nodes) < self.min_nodes:
                    return None
                
                community = merged_nodes
            
            # If community is too large, randomly sample
            if len(community) > self.max_nodes:
                selected_nodes = random.sample(list(community), self.max_nodes)
            else:
                selected_nodes = list(community)
            
            return self._create_subgraph(task_graph, selected_nodes, SamplingStrategy.COMMUNITY)
        
        except Exception as e:
            logger.debug(f"Community detection failed: {e}")
        
        return None
    
    def _star_sampling(self, task_graph: nx.DiGraph) -> Optional[TaskSubgraphData]:
        """Star structure sampling (hub node with spokes)"""
        node_ids = list(task_graph.nodes())
        if len(node_ids) < self.min_nodes:
            return None
        
        # Sort nodes by degree (descending) to try high-degree nodes first
        degrees = dict(task_graph.degree())
        sorted_nodes = sorted(node_ids, key=lambda n: degrees.get(n, 0), reverse=True)
        
        # Try to find a suitable center node with enough neighbors
        for center_node in sorted_nodes[:min(10, len(sorted_nodes))]:  # Try top 10 nodes
            # Get neighbors (both predecessors and successors)
            neighbors = set(task_graph.predecessors(center_node)) | set(task_graph.successors(center_node))
            
            if len(neighbors) >= self.min_nodes - 1:
                # Select neighbor nodes
                num_neighbors = min(len(neighbors), self.max_nodes - 1)
                selected_neighbors = random.sample(list(neighbors), num_neighbors)
                
                selected_nodes = [center_node] + selected_neighbors
                return self._create_subgraph(task_graph, selected_nodes, SamplingStrategy.STAR)
        
        return None
    
    def _chain_sampling(self, task_graph: nx.DiGraph) -> Optional[TaskSubgraphData]:
        """Chain structure sampling (linear path)"""
        node_ids = list(task_graph.nodes())
        if len(node_ids) < self.min_nodes:
            return None
        
        # Try multiple starting nodes to find a valid chain
        random.shuffle(node_ids)
        
        for start_node in node_ids[:min(10, len(node_ids))]:  # Try up to 10 starting nodes
            # Build chain structure by following edges (forward direction)
            chain = [start_node]
            current = start_node
            
            while len(chain) < self.max_nodes:
                successors = list(task_graph.successors(current))
                unvisited = [n for n in successors if n not in chain]
                
                if not unvisited:
                    break
                
                # Randomly select next node
                next_node = random.choice(unvisited)
                chain.append(next_node)
                current = next_node
            
            # If forward chain is not enough, try backward extension
            if len(chain) < self.min_nodes:
                current = start_node
                while len(chain) < self.max_nodes:
                    predecessors = list(task_graph.predecessors(current))
                    unvisited = [n for n in predecessors if n not in chain]
                    
                    if not unvisited:
                        break
                    
                    # Randomly select previous node
                    prev_node = random.choice(unvisited)
                    chain.insert(0, prev_node)
                    current = prev_node
            
            if len(chain) >= self.min_nodes:
                return self._create_subgraph(task_graph, chain, SamplingStrategy.CHAIN)
        
        return None
    
    def _tree_sampling(self, task_graph: nx.DiGraph) -> Optional[TaskSubgraphData]:
        """Tree structure sampling"""
        node_ids = list(task_graph.nodes())
        if len(node_ids) < self.min_nodes:
            return None
        
        # Calculate out-degrees and sort nodes (prefer nodes with more successors as root)
        out_degrees = dict(task_graph.out_degree())
        sorted_nodes = sorted(node_ids, key=lambda n: out_degrees.get(n, 0), reverse=True)
        
        # Try multiple potential root nodes
        for root in sorted_nodes[:min(10, len(sorted_nodes))]:
            # Build tree structure
            tree_nodes = [root]
            queue = [root]
            visited = {root}
            
            while queue and len(tree_nodes) < self.max_nodes:
                current = queue.pop(0)
                successors = list(task_graph.successors(current))
                unvisited = [n for n in successors if n not in visited]
                
                # Limit number of children for each node
                children = random.sample(unvisited, min(len(unvisited), TREE_BRANCHING_FACTOR))
                
                for child in children:
                    if len(tree_nodes) < self.max_nodes:
                        tree_nodes.append(child)
                        queue.append(child)
                        visited.add(child)
            
            if len(tree_nodes) >= self.min_nodes:
                return self._create_subgraph(task_graph, tree_nodes, SamplingStrategy.TREE)
        
        return None
    
    def _create_subgraph(
        self,
        task_graph: nx.DiGraph,
        node_ids: List[str],
        strategy: SamplingStrategy
    ) -> Optional[TaskSubgraphData]:
        """
        Create subgraph data
        
        Args:
            task_graph: Original task graph
            node_ids: List of node IDs to include
            strategy: Sampling strategy used
            
        Returns:
            Subgraph data or None if invalid
        """
        # Create subgraph
        subgraph = task_graph.subgraph(node_ids).copy()
        
        # Check if subgraph is weakly connected (important for task generation)
        if not nx.is_weakly_connected(subgraph):
            logger.debug(f"Subgraph sampled by {strategy} is not weakly connected")
            return None
        
        # Extract complete node information with all attributes
        node_details = []
        for node_id in node_ids:
            # Get node attributes from the original graph
            node_attrs = task_graph.nodes[node_id]
            # Create node data with id and all attributes
            node_data = {'id': node_id, **node_attrs}
            node_details.append(node_data)
        
        # Extract edges with their attributes
        edges = []
        for u, v, data in subgraph.edges(data=True):
            edge_data = {
                'source': u,
                'target': v,
                **data
            }
            edges.append(edge_data)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity(subgraph)
        
        # Calculate topology features
        topology_features = self._calculate_topology_features(subgraph)
        
        return TaskSubgraphData(
            id=str(uuid.uuid4()),
            nodes=node_ids,
            edges=edges,
            graph=subgraph,
            strategy=strategy,
            complexity_score=complexity_score,
            topology_features=topology_features,
            node_details=node_details
        )
    
    def _calculate_complexity(self, graph: nx.DiGraph) -> float:
        """
        Calculate subgraph complexity score
        
        Args:
            graph: Subgraph
            
        Returns:
            Complexity score (0-1)
        """
        if len(graph.nodes()) == 0:
            return 0.0
        
        # Calculate various complexity metrics
        density = nx.density(graph)
        avg_degree = sum(dict(graph.degree()).values()) / len(graph.nodes())
        
        # Check for cycles (more complex if has cycles)
        is_dag = nx.is_directed_acyclic_graph(graph)
        dag_penalty = 0.0 if is_dag else 0.2
        
        # Normalize and weight
        complexity = (
            density * 0.4 +
            min(avg_degree / 10, 1.0) * 0.4 +
            dag_penalty * 0.2
        )
        return complexity
    
    def _calculate_topology_features(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Calculate topology features
        
        Args:
            graph: Subgraph
            
        Returns:
            Topology features dictionary
        """
        if len(graph.nodes()) == 0:
            return {}
        
        features = {
            'num_nodes': len(graph.nodes()),
            'num_edges': len(graph.edges()),
            'density': nx.density(graph),
            'avg_degree': sum(dict(graph.degree()).values()) / len(graph.nodes()),
            'is_dag': nx.is_directed_acyclic_graph(graph),
            'weakly_connected_components': nx.number_weakly_connected_components(graph),
            'strongly_connected_components': nx.number_strongly_connected_components(graph)
        }
        
        # Add diameter for weakly connected graphs
        if nx.is_weakly_connected(graph):
            try:
                features['diameter'] = nx.diameter(graph.to_undirected())
            except:
                features['diameter'] = 0
        else:
            features['diameter'] = 0
        
        return features
    
    def _is_diverse(
        self,
        new_subgraph: TaskSubgraphData,
        existing_subgraphs: List[TaskSubgraphData]
    ) -> bool:
        """
        Check if the subgraph is sufficiently different from existing subgraphs
        
        Args:
            new_subgraph: New subgraph
            existing_subgraphs: List of existing subgraphs
            
        Returns:
            Whether it is sufficiently different
        """
        if not existing_subgraphs:
            return True
        
        # Check node overlap
        new_nodes = set(new_subgraph.nodes)
        
        for existing in existing_subgraphs:
            existing_nodes = set(existing.nodes)
            
            # Calculate Jaccard similarity
            intersection = len(new_nodes & existing_nodes)
            union = len(new_nodes | existing_nodes)
            
            if union > 0:
                jaccard_similarity = intersection / union
                
                if jaccard_similarity > (1.0 - self.diversity_threshold):
                    logger.debug(f"Subgraph too similar (Jaccard: {jaccard_similarity:.3f})")
                    return False
        
        return True
    
    def _visualize_subgraph(self, subgraph: TaskSubgraphData, output_path: Path) -> None:
        """
        Generate subgraph visualization
        
        Args:
            subgraph: Subgraph data
            output_path: Path to save visualization
        """
        graph = subgraph.graph
        
        if graph.number_of_nodes() == 0:
            logger.warning("Cannot visualize empty subgraph")
            return
        
        try:
            plt.figure(figsize=(12, 10))
            
            # Use hierarchical layout for DAG if possible
            try:
                if nx.is_directed_acyclic_graph(graph):
                    # Try to use graphviz_layout for better DAG visualization
                    try:
                        pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
                    except:
                        # Fall back to spring layout
                        pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
                else:
                    pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
            except:
                pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
            
            # Color nodes by domain
            domains = list(set(nx.get_node_attributes(graph, 'domain').values()))
            if domains:
                domain_colors = plt.cm.Set3(np.linspace(0, 1, len(domains)))
                domain_color_map = {domain: domain_colors[i] for i, domain in enumerate(domains)}
                
                node_colors = [
                    domain_color_map.get(graph.nodes[node].get('domain', ''), 'gray')
                    for node in graph.nodes()
                ]
            else:
                node_colors = ['lightblue'] * len(graph.nodes())
            
            # Draw nodes
            nx.draw_networkx_nodes(
                graph, pos,
                node_color=node_colors,
                node_size=500,
                alpha=0.8
            )
            
            # Draw edges with varying thickness based on weight
            edges = graph.edges()
            weights = [graph[u][v].get('weight', 1) for u, v in edges]
            max_weight = max(weights) if weights else 1
            edge_widths = [1 + 3 * (w / max_weight) for w in weights]
            
            nx.draw_networkx_edges(
                graph, pos,
                edgelist=edges,
                width=edge_widths,
                edge_color='gray',
                alpha=0.5,
                arrows=True,
                arrowsize=15,
                arrowstyle='->'
            )
            
            # Draw labels
            nx.draw_networkx_labels(
                graph, pos,
                font_size=9,
                font_weight='bold'
            )
            
            # Title with subgraph info
            title = (
                f"Task Subgraph ({subgraph.strategy.value})\n"
                f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges | "
                f"DAG: {nx.is_directed_acyclic_graph(graph)} | "
                f"Complexity: {subgraph.complexity_score:.3f}"
            )
            
            plt.title(title, fontsize=12, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.debug(f"Saved visualization to {output_path}")
            
        except Exception as e:
            logger.warning(f"Failed to generate subgraph visualization: {e}")
    
    def _save_subgraph(self, subgraph: TaskSubgraphData, output_dir: str) -> None:
        """
        Save subgraph to file
        
        Args:
            subgraph: Subgraph data
            output_dir: Output directory
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save JSON format
            subgraph_file = output_path / f"{subgraph.id}.json"
            
            data = subgraph.to_dict()
            
            with open(subgraph_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Saved subgraph {subgraph.id} to {subgraph_file}")
            
            # Generate and save visualization
            viz_file = output_path / f"{subgraph.id}.png"
            self._visualize_subgraph(subgraph, viz_file)
        
        except Exception as e:
            logger.error(f"Failed to save subgraph: {e}")
    
    def load_subgraph(self, subgraph_file: str) -> Optional[TaskSubgraphData]:
        """
        Load subgraph from file
        
        Args:
            subgraph_file: Path to subgraph file
            
        Returns:
            Subgraph data, returns None if failed
        """
        try:
            with open(subgraph_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return TaskSubgraphData.from_dict(data)
        
        except Exception as e:
            logger.error(f"Failed to load subgraph from {subgraph_file}: {e}")
            return None
    
    def load_all_subgraphs(self, output_dir: str) -> List[TaskSubgraphData]:
        """
        Load all subgraphs from directory
        
        Args:
            output_dir: Directory containing subgraph files
            
        Returns:
            List of subgraph data
        """
        subgraphs = []
        output_path = Path(output_dir)
        
        if not output_path.exists():
            logger.warning(f"Output directory does not exist: {output_dir}")
            return subgraphs
        
        for subgraph_file in output_path.glob("*.json"):
            subgraph = self.load_subgraph(str(subgraph_file))
            if subgraph:
                subgraphs.append(subgraph)
        
        logger.info(f"Loaded {len(subgraphs)} subgraphs from {output_dir}")
        return subgraphs

