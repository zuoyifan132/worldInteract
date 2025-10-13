#!/usr/bin/env python3
"""
Example: Subtask Graph Sampling
This example demonstrates how to sample subgraphs from a task graph.
"""

import os
import sys
import json
import dotenv
import argparse
import networkx as nx
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from worldInteract.core.build_task_graph import SubtaskGraphSampler

# Load environment variables
dotenv.load_dotenv("../.env")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Sample subgraphs from a task graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Sample subgraphs from task graph
    python sample_subtask_graph_example.py \\
        --task-graph data/task_graphs/file_operations_task_graph/task_graph.json \\
        --output data/subtask_graphs/file_operations_subtask_graphs \\
        --num-samples 10
        """
    )
    
    parser.add_argument(
        "--task-graph", "-t",
        type=str,
        required=True,
        help="Path to task graph JSON file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for subtask graphs (default: data/subtask_graphs/<auto_name>)"
    )
    
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=None,
        help="Number of subgraphs to sample (default: from config)"
    )
    
    return parser.parse_args()


def load_task_graph(task_graph_file: Path) -> nx.DiGraph:
    """Load task graph from JSON file"""
    with open(task_graph_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # Rebuild graph
    graph = nx.DiGraph()
    
    # Add nodes
    for node_data in graph_data['nodes']:
        node_id = node_data.pop('id')
        graph.add_node(node_id, **node_data)
    
    # Add edges
    for edge_data in graph_data['edges']:
        source = edge_data['source']
        target = edge_data['target']
        weight = edge_data.get('weight', 1)
        matching_pairs = edge_data.get('matching_pairs', [])
        graph.add_edge(source, target, weight=weight, matching_pairs=matching_pairs)
    
    return graph


def main():
    """Run the subtask graph sampling example"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info("=" * 80)
    logger.info("Subtask Graph Sampling Example")
    logger.info("=" * 80)
    
    # Validate task graph file
    task_graph_file = Path(args.task_graph)
    if not task_graph_file.is_absolute():
        task_graph_file = project_root / task_graph_file
    
    if not task_graph_file.exists():
        logger.error(f"Task graph file does not exist: {task_graph_file}")
        return
    
    logger.info(f"Input task graph: {task_graph_file}")
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
    else:
        # Auto-generate output directory name based on task graph
        graph_name = task_graph_file.parent.name
        output_dir = project_root / "data" / "subtask_graphs" / f"{graph_name}_subtask_graphs"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Load task graph
        logger.info("\n" + "=" * 80)
        logger.info("Loading Task Graph")
        logger.info("=" * 80)
        task_graph = load_task_graph(task_graph_file)
        logger.info(f"✅ Loaded graph: {task_graph.number_of_nodes()} nodes, {task_graph.number_of_edges()} edges")
        
        # Initialize subtask graph sampler
        logger.info("\n" + "=" * 80)
        logger.info("Initializing SubtaskGraphSampler")
        logger.info("=" * 80)
        sampler = SubtaskGraphSampler()
        
        # Sample subgraphs
        logger.info("\n" + "=" * 80)
        logger.info("Sampling Subgraphs")
        logger.info("=" * 80)
        logger.info("This process will:")
        logger.info("1. Sample subgraphs using different strategies")
        logger.info("2. Ensure diversity among sampled subgraphs")
        logger.info("3. Calculate topology features for each subgraph")
        logger.info("4. Save subgraphs to output directory")
        
        subgraphs = sampler.sample_subgraphs(
            task_graph=task_graph,
            num_samples=args.num_samples,
            output_dir=str(output_dir)
        )
        
        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("Subtask Graph Sampling Completed")
        logger.info("=" * 80)
        logger.info(f"✅ Sampled {len(subgraphs)} subgraphs")
        
        # Show statistics
        if subgraphs:
            strategies = {}
            for subgraph in subgraphs:
                strategy = subgraph.strategy.value
                strategies[strategy] = strategies.get(strategy, 0) + 1
            
            logger.info("\nSubgraphs by strategy:")
            for strategy, count in strategies.items():
                logger.info(f"  - {strategy}: {count}")
            
            logger.info("\nSubgraph statistics:")
            avg_nodes = sum(sg.topology_features.get('num_nodes', 0) for sg in subgraphs) / len(subgraphs)
            avg_edges = sum(sg.topology_features.get('num_edges', 0) for sg in subgraphs) / len(subgraphs)
            avg_complexity = sum(sg.complexity_score for sg in subgraphs) / len(subgraphs)
            
            logger.info(f"  - Average nodes: {avg_nodes:.1f}")
            logger.info(f"  - Average edges: {avg_edges:.1f}")
            logger.info(f"  - Average complexity: {avg_complexity:.3f}")
            
            # Show sample subgraphs
            logger.info("\nSample subgraphs:")
            for i, subgraph in enumerate(subgraphs[:3]):
                logger.info(f"\n  Subgraph {i+1} ({subgraph.strategy.value}):")
                logger.info(f"    ID: {subgraph.id}")
                logger.info(f"    Nodes: {subgraph.topology_features.get('num_nodes', 0)}")
                logger.info(f"    Edges: {subgraph.topology_features.get('num_edges', 0)}")
                logger.info(f"    Is DAG: {subgraph.topology_features.get('is_dag', False)}")
                logger.info(f"    Complexity: {subgraph.complexity_score:.3f}")
        
        # List generated files
        logger.info("\n" + "=" * 80)
        logger.info("Generated Files")
        logger.info("=" * 80)
        
        subgraph_files = list(output_dir.glob("*.json"))
        logger.info(f"📁 {len(subgraph_files)} subgraph files saved")
        
        logger.info("\n" + "=" * 80)
        logger.info("Example Completed Successfully!")
        logger.info("=" * 80)
        logger.info("Next steps:")
        logger.info("1. Review the sampled subgraphs")
        logger.info("2. Use random_walk_example.py to generate random walks")
        
    except Exception as e:
        logger.error(f"Subtask graph sampling failed: {e}")
        logger.exception(e)
        raise


if __name__ == "__main__":
    main()

