#!/usr/bin/env python3
"""
Example: Random Walk Generation
This example demonstrates how to generate random walks from task subgraphs.
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
from worldInteract.core.build_task_graph import RandomWalker

# Load environment variables
dotenv.load_dotenv("../.env")


def load_subgraph_from_json(subgraph_file: str) -> nx.DiGraph:
    """
    Load a task subgraph from JSON file and convert to NetworkX DiGraph
    
    Args:
        subgraph_file: Path to subgraph JSON file
        
    Returns:
        NetworkX DiGraph representing the task subgraph
    """
    with open(subgraph_file, 'r', encoding='utf-8') as f:
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
    
    return graph


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate random walks from task subgraphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Generate random walks from task subgraphs
    python random_walk_example.py \\
        --task-subgraphs data/task_subgraphs/file_operations_task_subgraphs \\
        --output data/random_walks/file_operations_random_walks \\
        --num-walks 2
        """
    )
    
    parser.add_argument(
        "--task-subgraphs", "-s",
        type=str,
        required=True,
        help="Path to directory containing task subgraph JSON files"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for random walks (default: data/random_walks/<auto_name>)"
    )
    
    parser.add_argument(
        "--num-walks", "-n",
        type=int,
        default=None,
        help="Number of walks per subgraph (default: from config)"
    )
    
    return parser.parse_args()


def main():
    """Run the random walk generation example"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info("=" * 80)
    logger.info("Random Walk Generation Example")
    logger.info("=" * 80)
    
    # Validate task subgraphs directory
    task_subgraphs_dir = Path(args.task_subgraphs)
    if not task_subgraphs_dir.is_absolute():
        task_subgraphs_dir = project_root / task_subgraphs_dir
    
    if not task_subgraphs_dir.exists():
        logger.error(f"Task subgraphs directory does not exist: {task_subgraphs_dir}")
        return
    
    logger.info(f"Input task subgraphs: {task_subgraphs_dir}")
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
    else:
        # Auto-generate output directory name
        graph_name = task_subgraphs_dir.name.replace("_task_subgraphs", "")
        output_dir = project_root / "data" / "random_walks" / f"{graph_name}_random_walks"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Load task subgraphs from JSON files
        logger.info("\n" + "=" * 80)
        logger.info("Loading Task Subgraphs")
        logger.info("=" * 80)
        
        subgraph_files = list(task_subgraphs_dir.glob("*.json"))
        
        if not subgraph_files:
            logger.error(f"No task subgraph JSON files found in {task_subgraphs_dir}")
            return
        
        logger.info(f"Found {len(subgraph_files)} subgraph files")
        
        subgraphs = []
        for subgraph_file in subgraph_files:
            try:
                graph = load_subgraph_from_json(str(subgraph_file))
                subgraphs.append({
                    'file': subgraph_file.name,
                    'graph': graph,
                    'nodes': graph.number_of_nodes(),
                    'edges': graph.number_of_edges()
                })
                logger.debug(f"Loaded {subgraph_file.name}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            except Exception as e:
                logger.warning(f"Failed to load {subgraph_file.name}: {e}")
        
        if not subgraphs:
            logger.error("No valid task subgraphs loaded")
            return
        
        logger.info(f"✅ Successfully loaded {len(subgraphs)} task subgraphs")
        
        # Initialize random walker
        logger.info("\n" + "=" * 80)
        logger.info("Initializing RandomWalker")
        logger.info("=" * 80)
        walker = RandomWalker()
        
        # Generate random walks
        logger.info("\n" + "=" * 80)
        logger.info("Generating Random Walks")
        logger.info("=" * 80)
        logger.info("This process will:")
        logger.info("1. Generate chain walks (linear paths)")
        logger.info("2. Generate DAG walks (parallel branches)")
        logger.info("3. Ensure walk diversity")
        logger.info("4. Save walks to output directory")
        
        all_walks = []
        for i, subgraph_data in enumerate(subgraphs):
            logger.info(f"\nProcessing subgraph {i+1}/{len(subgraphs)} ({subgraph_data['file']})...")
            logger.info(f"  Nodes: {subgraph_data['nodes']}, Edges: {subgraph_data['edges']}")
            
            walks = walker.generate_walks(
                subgraph=subgraph_data['graph'],
                num_walks=args.num_walks,
                output_dir=str(output_dir)
            )
            
            all_walks.extend(walks)
            logger.info(f"  Generated {len(walks)} walks")
        
        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("Random Walk Generation Completed")
        logger.info("=" * 80)
        logger.info(f"✅ Generated {len(all_walks)} total walks")
        
        # Show statistics
        if all_walks:
            walk_types = {}
            for walk in all_walks:
                walk_type = walk.walk_type.value
                walk_types[walk_type] = walk_types.get(walk_type, 0) + 1
            
            logger.info("\nWalks by type:")
            for walk_type, count in walk_types.items():
                logger.info(f"  - {walk_type}: {count}")
            
            logger.info("\nWalk statistics:")
            avg_length = sum(walk.length for walk in all_walks) / len(all_walks)
            min_length = min(walk.length for walk in all_walks)
            max_length = max(walk.length for walk in all_walks)
            
            logger.info(f"  - Average length: {avg_length:.1f}")
            logger.info(f"  - Min length: {min_length}")
            logger.info(f"  - Max length: {max_length}")
            
            # Show sample walks
            logger.info("\nSample walks:")
            for i, walk in enumerate(all_walks[:3]):
                logger.info(f"\n  Walk {i+1} ({walk.walk_type.value}):")
                logger.info(f"    ID: {walk.id}")
                logger.info(f"    Length: {walk.length}")
                logger.info(f"    Sequence: {' → '.join(walk.sequence[:5])}")
                if len(walk.sequence) > 5:
                    logger.info(f"              ... ({len(walk.sequence)-5} more)")
                
                # Show DAG structure if applicable
                if walk.walk_type.value == 'dag' and walk.dag_structure:
                    logger.info(f"    Layers: {len(walk.dag_structure['layers'])}")
                    logger.info(f"    Max parallelism: {walk.metadata.get('max_parallelism', 1)}")
                    
                    # Show visualization
                    if i == 0:  # Only show for first DAG walk
                        logger.info("\n" + walker.visualize_dag_walk(walk))
        
        # List generated files
        logger.info("\n" + "=" * 80)
        logger.info("Generated Files")
        logger.info("=" * 80)
        
        walk_json_files = list(output_dir.glob("*.json"))
        walk_png_files = list(output_dir.glob("*.png"))
        logger.info(f"📁 {len(walk_json_files)} walk JSON files saved")
        logger.info(f"🖼️  {len(walk_png_files)} walk PNG visualizations saved")
        
        logger.info("\n" + "=" * 80)
        logger.info("Example Completed Successfully!")
        logger.info("=" * 80)
        logger.info("Next steps:")
        logger.info("1. Review the generated random walks")
        logger.info("2. Use build_task_example.py to generate agent tasks")
        
    except Exception as e:
        logger.error(f"Random walk generation failed: {e}")
        logger.exception(e)
        raise


if __name__ == "__main__":
    main()

