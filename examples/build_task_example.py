#!/usr/bin/env python3
"""
Example: Complete Task Generation Pipeline
This example demonstrates the complete pipeline from task graph to agent tasks.
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
from worldInteract.core.build_task_graph import (
    TaskGraphBuilder,
    SubtaskGraphSampler,
    RandomWalker
)
from worldInteract.trajectories import TaskGenerator

# Load environment variables
dotenv.load_dotenv("../.env")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Complete task generation pipeline: task graph → subgraphs → walks → tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Run complete pipeline for file_operations domain
    python build_task_example.py \\
        --env-dirs data/generated_env/domains/file_operations \\
        --domain-graph data/domain_graphs/my_domain_graphs \\
        --output data/agent_tasks/file_operations_tasks
    
    # Run for multiple domains
    python build_task_example.py \\
        --env-dirs data/generated_env/domains/file_operations data/generated_env/domains/database_operations \\
        --domain-graph data/domain_graphs/my_domain_graphs \\
        --output data/agent_tasks/multi_domain_tasks \\
        --num-subgraphs 10 \\
        --num-walks 2
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
        help="Output directory for tasks (default: data/agent_tasks/<auto_name>)"
    )
    
    parser.add_argument(
        "--num-subgraphs", "-ns",
        type=int,
        default=None,
        help="Number of subgraphs to sample (default: from config)"
    )
    
    parser.add_argument(
        "--num-walks", "-nw",
        type=int,
        default=None,
        help="Number of walks per subgraph (default: from config)"
    )
    
    parser.add_argument(
        "--skip-filter", 
        action="store_true",
        help="Skip LLM filtering of walks"
    )
    
    return parser.parse_args()


def load_task_graph_from_file(task_graph_file: Path) -> nx.DiGraph:
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


def extract_tool_definitions(task_graph_file: Path) -> dict:
    """Extract tool definitions from task graph"""
    with open(task_graph_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    tool_definitions = {}
    for node_data in graph_data['nodes']:
        tool_name = node_data['id']
        tool_definitions[tool_name] = {
            'description': node_data.get('description', ''),
            'parameters': node_data.get('parameters', {}),
            'returns': node_data.get('returns', {}),
            'domain': node_data.get('domain', '')
        }
    
    return tool_definitions


def main():
    """Run the complete task generation pipeline"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info("=" * 80)
    logger.info("Complete Task Generation Pipeline")
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
    
    # Set up output directories
    if args.output:
        base_output_dir = Path(args.output)
        if not base_output_dir.is_absolute():
            base_output_dir = project_root / base_output_dir
    else:
        # Auto-generate output directory name
        domain_names = [Path(d).name for d in env_dirs]
        output_name = "_".join(domain_names[:2])
        if len(domain_names) > 2:
            output_name += f"_and_{len(domain_names)-2}_more"
        base_output_dir = project_root / "data" / "agent_tasks" / f"{output_name}_tasks"
    
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    task_graph_dir = base_output_dir / "task_graph"
    subtask_graphs_dir = base_output_dir / "subtask_graphs"
    random_walks_dir = base_output_dir / "random_walks"
    tasks_dir = base_output_dir / "tasks"
    
    logger.info(f"Base output directory: {base_output_dir}")
    
    try:
        # ========== STEP 1: Build Task Graph ==========
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Building Task Graph")
        logger.info("=" * 80)
        
        task_graph_builder = TaskGraphBuilder()
        task_graph_result = task_graph_builder.build_task_graph(
            generated_env_dirs=env_dirs,
            domain_graph_dir=str(domain_graph_dir),
            output_dir=str(task_graph_dir),
            graph_name="pipeline_task_graph"
        )
        
        logger.info(f"✅ Task graph built: {task_graph_result.get('statistics', {}).get('node_count', 0)} nodes, "
                   f"{task_graph_result.get('statistics', {}).get('edge_count', 0)} edges")
        
        # Load the task graph
        task_graph_file = task_graph_dir / "task_graph.json"
        task_graph = load_task_graph_from_file(task_graph_file)
        tool_definitions = extract_tool_definitions(task_graph_file)
        
        # ========== STEP 2: Sample Subtask Graphs ==========
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Sampling Subtask Graphs")
        logger.info("=" * 80)
        
        subtask_graph_sampler = SubtaskGraphSampler()
        subgraphs = subtask_graph_sampler.sample_subgraphs(
            task_graph=task_graph,
            num_samples=args.num_subgraphs,
            output_dir=str(subtask_graphs_dir)
        )
        
        logger.info(f"✅ Sampled {len(subgraphs)} subtask graphs")
        
        # ========== STEP 3: Generate Random Walks ==========
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Generating Random Walks")
        logger.info("=" * 80)
        
        random_walker = RandomWalker()
        all_walks = []
        
        for i, subgraph_data in enumerate(subgraphs):
            walks = random_walker.generate_walks(
                subgraph=subgraph_data.graph,
                num_walks=args.num_walks,
                output_dir=str(random_walks_dir)
            )
            all_walks.extend(walks)
        
        logger.info(f"✅ Generated {len(all_walks)} random walks")
        
        # ========== STEP 4: Generate Agent Tasks ==========
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Generating Agent Tasks")
        logger.info("=" * 80)
        
        task_generator = TaskGenerator()
        
        # Override filter setting if requested
        if args.skip_filter:
            logger.warning("Skipping LLM filtering as requested")
            task_generator.enable_llm_filter = False
        
        tasks = task_generator.generate_tasks(
            walks=all_walks,
            tool_definitions=tool_definitions,
            output_dir=str(tasks_dir)
        )
        
        logger.info(f"✅ Generated {len(tasks)} agent tasks")
        
        # ========== Display Final Results ==========
        logger.info("\n" + "=" * 80)
        logger.info("Pipeline Completed Successfully!")
        logger.info("=" * 80)
        
        logger.info("\nPipeline Summary:")
        logger.info(f"  1. Task Graph:     {task_graph_result.get('statistics', {}).get('node_count', 0)} nodes, "
                   f"{task_graph_result.get('statistics', {}).get('edge_count', 0)} edges")
        logger.info(f"  2. Subtask Graphs: {len(subgraphs)} sampled")
        logger.info(f"  3. Random Walks:   {len(all_walks)} generated")
        logger.info(f"  4. Agent Tasks:    {len(tasks)} created")
        
        if tasks:
            # Task statistics
            walk_types = {'chain': 0, 'dag': 0}
            for task in tasks:
                walk_types[task.walk_type] = walk_types.get(task.walk_type, 0) + 1
            
            avg_complexity = sum(t.complexity_score for t in tasks) / len(tasks)
            avg_sequence_length = sum(len(t.function_sequence) for t in tasks) / len(tasks)
            
            logger.info("\nTask Statistics:")
            logger.info(f"  - Chain tasks: {walk_types.get('chain', 0)}")
            logger.info(f"  - DAG tasks:   {walk_types.get('dag', 0)}")
            logger.info(f"  - Average complexity: {avg_complexity:.3f}")
            logger.info(f"  - Average sequence length: {avg_sequence_length:.1f}")
            
            # Show sample tasks
            logger.info("\nSample Tasks:")
            for i, task in enumerate(tasks[:3]):
                logger.info(f"\n  Task {i+1}:")
                logger.info(f"    ID: {task.id}")
                logger.info(f"    Type: {task.walk_type}")
                logger.info(f"    Description: {task.task_description[:100]}...")
                logger.info(f"    Functions: {' → '.join(task.function_sequence[:5])}")
                if len(task.function_sequence) > 5:
                    logger.info(f"               ... ({len(task.function_sequence)-5} more)")
        
        logger.info("\n" + "=" * 80)
        logger.info("Output Structure:")
        logger.info("=" * 80)
        logger.info(f"📁 {base_output_dir}")
        logger.info(f"  ├── 📁 task_graph/")
        logger.info(f"  │    ├── task_graph.json")
        logger.info(f"  │    ├── embeddings.json")
        logger.info(f"  │    └── task_graph_visualization.png")
        logger.info(f"  ├── 📁 subtask_graphs/")
        logger.info(f"  │    └── {len(subgraphs)} subgraph JSON files")
        logger.info(f"  ├── 📁 random_walks/")
        logger.info(f"  │    └── {len(all_walks)} walk JSON files")
        logger.info(f"  └── 📁 tasks/")
        logger.info(f"       ├── tasks_summary.json")
        logger.info(f"       └── {len(tasks)} task JSON files")
        
        logger.info("\n" + "=" * 80)
        logger.info("Next Steps:")
        logger.info("=" * 80)
        logger.info("1. Review generated tasks in tasks/ directory")
        logger.info("2. Examine task descriptions and function sequences")
        logger.info("3. Use these tasks for agent training or evaluation")
        logger.info("4. Visualize the task graph to understand dependencies")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.exception(e)
        raise


if __name__ == "__main__":
    main()

