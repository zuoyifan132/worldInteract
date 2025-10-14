"""
Task Subgraph Sampling Example

This example demonstrates how to sample subgraphs from a task graph.
"""

import json
import argparse
from pathlib import Path
import networkx as nx
from loguru import logger

from worldInteract.core.build_task_graph.task_subgraph_sampler import TaskSubgraphSampler


def load_task_graph(task_graph_file: str) -> nx.DiGraph:
    """
    Load task graph from JSON file
    
    Args:
        task_graph_file: Path to task graph JSON file
        
    Returns:
        NetworkX DiGraph
    """
    logger.info(f"Loading task graph from: {task_graph_file}")
    
    with open(task_graph_file, 'r', encoding='utf-8') as f:
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
    parser = argparse.ArgumentParser(description='Sample subgraphs from task graph')
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
    
    # Load task graph
    task_graph = load_task_graph(args.task_graph)
    
    # Create sampler
    sampler = TaskSubgraphSampler()
    
    # Sample subgraphs
    logger.info(f"Sampling subgraphs...")
    logger.info(f"Output directory: {args.output}")
    
    subgraphs = sampler.sample_subgraphs(
        task_graph=task_graph,
        num_samples=args.num_samples,
        output_dir=args.output
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
    logger.info(f"Subgraphs saved to: {args.output}")
    logger.info(f"{'='*60}\n")
    
    # Also demonstrate loading subgraphs
    logger.info("Demonstrating loading subgraphs...")
    loaded_subgraphs = sampler.load_all_subgraphs(args.output)
    logger.info(f"Successfully loaded {len(loaded_subgraphs)} subgraphs")


if __name__ == "__main__":
    main()

