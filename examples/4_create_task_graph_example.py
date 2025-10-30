#!/usr/bin/env python3
"""
Example: Task Graph Building
This example demonstrates how to build a task graph from generated environments.
"""

import os
import sys
import json
import dotenv
import argparse
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from worldInteract.core.build_task_graph import TaskGraphBuilder

# Load environment variables
dotenv.load_dotenv("../.env")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Build task graph from generated environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Build task graph from file_operations domain
    python create_task_graph_example.py \\
        --env-dirs data/generated_env/domains/file_operations \\
        --domain-graph data/domain_graphs/my_domain_graphs \\
        --output data/task_graphs/file_operations_task_graph
    
    # Build from multiple domains
    python create_task_graph_example.py \\
        --env-dirs data/generated_env/domains/file_operations data/generated_env/domains/database_operations \\
        --domain-graph data/domain_graphs/my_domain_graphs \\
        --output data/task_graphs/multi_domain_task_graph
        """
    )
    
    parser.add_argument(
        "--env-dirs", "-e",
        nargs='+',
        required=True,
        help="Paths to generated environment directories (one or more)"
    )
    
    parser.add_argument(
        "--domain-graph", "-d",
        type=str,
        required=True,
        help="Path to domain graph directory"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for task graph (default: data/task_graphs/<auto_name>)"
    )
    
    parser.add_argument(
        "--graph-name", "-n",
        type=str,
        default=None,
        help="Name for the task graph (default: auto-generated)"
    )
    
    return parser.parse_args()


def main():
    """Run the task graph building example"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info("=" * 80)
    logger.info("Task Graph Building Example")
    logger.info("=" * 80)
    
    # Validate input paths
    env_dirs = []
    for env_dir_str in args.env_dirs:
        env_dir = Path(env_dir_str)
        if not env_dir.is_absolute():
            env_dir = project_root / env_dir
        
        if not env_dir.exists():
            logger.error(f"Environment directory does not exist: {env_dir}")
            return
        
        env_dirs.append(str(env_dir))
        logger.info(f"Input environment: {env_dir}")
    
    # Validate domain graph directory
    domain_graph_dir = Path(args.domain_graph)
    if not domain_graph_dir.is_absolute():
        domain_graph_dir = project_root / domain_graph_dir
    
    if not domain_graph_dir.exists():
        logger.error(f"Domain graph directory does not exist: {domain_graph_dir}")
        return
    
    logger.info(f"Domain graph directory: {domain_graph_dir}")
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
    else:
        # Auto-generate output directory name
        domain_names = [Path(d).name for d in env_dirs]
        output_name = "_".join(domain_names[:2])  # Use first 2 domain names
        if len(domain_names) > 2:
            output_name += f"_and_{len(domain_names)-2}_more"
        output_dir = project_root / "data" / "task_graphs" / f"{output_name}_task_graph"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Initialize task graph builder
        logger.info("\n" + "=" * 80)
        logger.info("Initializing TaskGraphBuilder")
        logger.info("=" * 80)
        builder = TaskGraphBuilder()
        
        # Build task graph
        logger.info("\n" + "=" * 80)
        logger.info("Building Task Graph")
        logger.info("=" * 80)
        logger.info("This process will:")
        logger.info("1. Load tools from generated environments")
        logger.info("2. Generate embeddings for all parameters")
        logger.info("3. Build dependency graph based on parameter similarity")
        logger.info("4. Save task graph and visualizations")
        
        result = builder.build_task_graph(
            generated_env_dirs=env_dirs,
            domain_graph_dir=str(domain_graph_dir),
            output_dir=str(output_dir),
            graph_name=args.graph_name
        )
        
        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("Task Graph Building Completed")
        logger.info("=" * 80)
        
        if result and 'statistics' in result:
            stats = result['statistics']
            logger.info(f"✅ Nodes: {stats.get('node_count', 0)}")
            logger.info(f"✅ Edges: {stats.get('edge_count', 0)}")
            logger.info(f"✅ Density: {stats.get('density', 0):.4f}")
            logger.info(f"✅ Is DAG: {stats.get('is_dag', False)}")
            logger.info(f"✅ Weakly connected components: {stats.get('weakly_connected_components', 0)}")
            logger.info(f"✅ Strongly connected components: {stats.get('strongly_connected_components', 0)}")
        
        # List generated files
        logger.info("\n" + "=" * 80)
        logger.info("Generated Files")
        logger.info("=" * 80)
        
        task_graph_file = output_dir / "task_graph.json"
        if task_graph_file.exists():
            file_size = task_graph_file.stat().st_size
            logger.info(f"📄 task_graph.json ({file_size:,} bytes)")
            
            # Show some details
            with open(task_graph_file, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            logger.info(f"   - Domains: {', '.join(graph_data.get('metadata', {}).get('domains', []))}")
            logger.info(f"   - Similarity threshold: {graph_data.get('metadata', {}).get('similarity_threshold', 0)}")
        
        embeddings_file = output_dir / "embeddings.json"
        if embeddings_file.exists():
            file_size = embeddings_file.stat().st_size
            logger.info(f"📄 embeddings.json ({file_size:,} bytes)")
        
        viz_file = output_dir / "task_graph_visualization.png"
        if viz_file.exists():
            file_size = viz_file.stat().st_size
            logger.info(f"🖼️  task_graph_visualization.png ({file_size:,} bytes)")
        
        logger.info("\n" + "=" * 80)
        logger.info("Example Completed Successfully!")
        logger.info("=" * 80)
        logger.info("Next steps:")
        logger.info("1. Review the task graph visualization")
        logger.info("2. Examine the edge relationships between functions")
        logger.info("3. Use sample_task_subgraph_example.py to sample subgraphs")
        
    except Exception as e:
        logger.error(f"Task graph building failed: {e}")
        logger.exception(e)
        raise


if __name__ == "__main__":
    main()

