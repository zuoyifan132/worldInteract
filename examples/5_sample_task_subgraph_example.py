"""
Task Subgraph Sampling Example

This example demonstrates how to sample subgraphs from a task graph.
"""

import sys
import json
import argparse
from pathlib import Path
import networkx as nx
from loguru import logger

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from worldInteract.core.build_task_graph.task_subgraph_sampler import TaskSubgraphSampler


def load_task_graph(task_graph_file: str, project_root: Path) -> nx.DiGraph:
    """
    Load task graph from JSON file
    
    Args:
        task_graph_file: Path to task graph JSON file
        project_root: Project root directory path
        
    Returns:
        NetworkX DiGraph
    """
    # Convert relative path to absolute path based on project root
    task_graph_path = Path(task_graph_file)
    if not task_graph_path.is_absolute():
        task_graph_path = project_root / task_graph_path
    
    logger.info(f"Loading task graph from: {task_graph_path}")
    
    if not task_graph_path.exists():
        raise FileNotFoundError(f"Task graph file not found: {task_graph_path}")
    
    with open(task_graph_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create directed graph
    graph = nx.DiGraph()
    
    # Add nodes with all their attributes
    for node in data.get('nodes', []):
        node_id = node['id']
        # Add node with all attributes except 'id'
        node_attrs = {k: v for k, v in node.items() if k != 'id'}
        graph.add_node(node_id, **node_attrs)
    
    # Add edges with their attributes
    for edge in data.get('edges', []):
        source = edge['source']
        target = edge['target']
        # Add edge with all attributes except 'source' and 'target'
        edge_attrs = {k: v for k, v in edge.items() if k not in ['source', 'target']}
        graph.add_edge(source, target, **edge_attrs)
    
    logger.info(f"Loaded task graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    return graph


def main():
    parser = argparse.ArgumentParser(
        description='Sample subgraphs from task graph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python sample_task_subgraph_example.py --task-graph data/task_graphs/file_operations_task_graph/task_graph.json --output data/task_subgraphs/file_operations_task_subgraphs
  python sample_task_subgraph_example.py --task-graph data/task_graphs/file_operations_task_graph/task_graph.json --output data/task_subgraphs/file_operations_task_subgraphs --num-samples 5
        """
    )
    parser.add_argument(
        '--task-graph',
        type=str,
        required=True,
        help='Path to task graph JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for subgraphs'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of subgraphs to sample (uses config if not specified)'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    
    # Load task graph
    task_graph = load_task_graph(args.task_graph, project_root)
    
    # Handle output directory path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create sampler
    sampler = TaskSubgraphSampler()
    
    # Sample subgraphs
    logger.info(f"Sampling subgraphs...")
    logger.info(f"Output directory: {output_path}")
    
    subgraphs = sampler.sample_subgraphs(
        task_graph=task_graph,
        num_samples=args.num_samples,
        output_dir=str(output_path)
    )
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Subgraph Sampling Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total subgraphs sampled: {len(subgraphs)}")
    
    if subgraphs:
        logger.info(f"\nSubgraph Details:")
        for i, subgraph in enumerate(subgraphs, 1):
            logger.info(f"\n  Subgraph {i}:")
            logger.info(f"    ID: {subgraph.id}")
            logger.info(f"    Strategy: {subgraph.strategy.value}")
            logger.info(f"    Nodes: {len(subgraph.nodes)}")
            logger.info(f"    Edges: {len(subgraph.edges)}")
            logger.info(f"    Complexity: {subgraph.complexity_score:.3f}")
            logger.info(f"    Topology:")
            for key, value in subgraph.topology_features.items():
                logger.info(f"      {key}: {value}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Subgraphs saved to: {output_path}")
    logger.info(f"{'='*60}\n")
    
    # Also demonstrate loading subgraphs
    logger.info("Demonstrating loading subgraphs...")
    loaded_subgraphs = sampler.load_all_subgraphs(str(output_path))
    logger.info(f"Successfully loaded {len(loaded_subgraphs)} subgraphs")


if __name__ == "__main__":
    main()

