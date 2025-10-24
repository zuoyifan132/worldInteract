#!/usr/bin/env python3
"""
Example: Trajectory Generation from Random Walks
This example demonstrates how to generate complete interaction trajectories using ReAct agent.
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
from worldInteract.trajectories import TrajectoryGenerator

# Load environment variables
dotenv.load_dotenv("../.env")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate complete interaction trajectories from random walks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Generate trajectories for all random walks in a directory
    python generate_trajectories_example.py \\
        --domain-tools data/domain_graphs/my_domain_graphs/domains/file_operations.json \\
        --environment data/generated_env/domains/file_operations \\
        --random-walks data/random_walks/file_operations_random_walks \\
        --output data/generated_trajectories/file_operations_trajectories
    
    # Generate single trajectory
    python generate_trajectories_example.py \\
        --domain-tools data/domain_graphs/my_domain_graphs/domains/file_operations.json \\
        --environment data/generated_env/domains/file_operations \\
        --random-walk data/random_walks/file_operations_random_walks/0c4a385e-28ce-4170-b1da-1a3b6920fcad.json \\
        --output data/generated_trajectories/file_operations_trajectories
        """
    )
    
    parser.add_argument(
        "--domain-tools", "-d",
        type=str,
        required=True,
        help="Path to domain tools JSON file (e.g., data/domain_graphs/.../domains/file_operations.json)"
    )
    
    parser.add_argument(
        "--environment", "-e",
        type=str,
        required=True,
        help="Path to environment domain directory (e.g., data/generated_env/domains/file_operations)"
    )
    
    parser.add_argument(
        "--random-walks", "-w",
        type=str,
        default=None,
        help="Path to directory containing random walk JSON files (for batch generation)"
    )
    
    parser.add_argument(
        "--random-walk", "-s",
        type=str,
        default=None,
        help="Path to single random walk JSON file (for single generation)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for generated trajectories"
    )
    
    return parser.parse_args()


def main():
    """Run the trajectory generation example"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info("=" * 80)
    logger.info("Trajectory Generation Example")
    logger.info("=" * 80)
    
    # Validate domain tools path
    domain_tools_path = Path(args.domain_tools)
    if not domain_tools_path.is_absolute():
        domain_tools_path = project_root / domain_tools_path
    
    if not domain_tools_path.exists():
        logger.error(f"Domain tools file does not exist: {domain_tools_path}")
        return
    
    # Validate environment path
    env_domain_path = Path(args.environment)
    if not env_domain_path.is_absolute():
        env_domain_path = project_root / env_domain_path
    
    if not env_domain_path.exists():
        logger.error(f"Environment directory does not exist: {env_domain_path}")
        return
    
    # Set output directory
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Domain tools: {domain_tools_path}")
    logger.info(f"Environment: {env_domain_path}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Initialize trajectory generator
        logger.info("\n" + "=" * 80)
        logger.info("Initializing TrajectoryGenerator")
        logger.info("=" * 80)
        
        generator = TrajectoryGenerator(
            domain_tools_path=str(domain_tools_path),
            env_domain_path=str(env_domain_path)
        )
        
        logger.info(f"✅ Domain: {generator.domain_tools['domain']}")
        logger.info(f"✅ Tools count: {len(generator.domain_tools['tools'])}")
        logger.info(f"✅ Environment loaded successfully")
        
        # Determine mode: single or batch
        if args.random_walk:
            # Single trajectory generation
            logger.info("\n" + "=" * 80)
            logger.info("Single Trajectory Generation Mode")
            logger.info("=" * 80)
            
            random_walk_path = Path(args.random_walk)
            if not random_walk_path.is_absolute():
                random_walk_path = project_root / random_walk_path
            
            if not random_walk_path.exists():
                logger.error(f"Random walk file does not exist: {random_walk_path}")
                return
            
            logger.info(f"Input random walk: {random_walk_path}")
            
            # Load random walk
            with open(random_walk_path, 'r', encoding='utf-8') as f:
                random_walk = json.load(f)
            
            logger.info(f"Walk ID: {random_walk['id']}")
            logger.info(f"Walk type: {random_walk['walk_type']}")
            logger.info(f"Sequence: {' → '.join(random_walk['sequence'])}")
            
            # Generate trajectory
            logger.info("\n" + "=" * 80)
            logger.info("Generating Trajectory")
            logger.info("=" * 80)
            logger.info("This process will:")
            logger.info("1. Generate user queries for each step using LLM")
            logger.info("2. Execute ReAct agent interactions with tools")
            logger.info("3. Track state changes after each tool execution")
            logger.info("4. Save complete interaction history")
            
            trajectory = generator.generate_trajectory(random_walk)
            
            # Save trajectory
            output_path = output_dir / f"{random_walk['id']}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(trajectory, f, indent=2, ensure_ascii=False)
            
            # Display results
            logger.info("\n" + "=" * 80)
            logger.info("Trajectory Generation Completed")
            logger.info("=" * 80)
            logger.info(f"✅ Trajectory saved to: {output_path}")
            
            # Show statistics
            stats = trajectory['statistics']
            logger.info("\n📊 Trajectory Statistics:")
            logger.info(f"  - Domain: {trajectory['domain']}")
            logger.info(f"  - Sequence length: {stats['num_nodes']}")
            logger.info(f"  - User queries: {stats['num_user_queries']}")
            logger.info(f"  - Total interactions: {stats['num_interactions']}")
            logger.info(f"  - State changes: {stats['num_state_changes']}")
            
        elif args.random_walks:
            # Batch trajectory generation
            logger.info("\n" + "=" * 80)
            logger.info("Batch Trajectory Generation Mode")
            logger.info("=" * 80)
            
            random_walks_dir = Path(args.random_walks)
            if not random_walks_dir.is_absolute():
                random_walks_dir = project_root / random_walks_dir
            
            if not random_walks_dir.exists():
                logger.error(f"Random walks directory does not exist: {random_walks_dir}")
                return
            
            logger.info(f"Input random walks: {random_walks_dir}")
            
            # Count random walk files
            walk_files = list(random_walks_dir.glob("*.json"))
            logger.info(f"Found {len(walk_files)} random walk files")
            
            # Generate trajectories
            logger.info("\n" + "=" * 80)
            logger.info("Generating Trajectories")
            logger.info("=" * 80)
            logger.info("This process will:")
            logger.info("1. Generate user queries for each step using LLM")
            logger.info("2. Execute ReAct agent interactions with tools")
            logger.info("3. Track state changes after each tool execution")
            logger.info("4. Save complete interaction history")
            logger.info("")
            
            results = generator.generate_trajectories_batch(
                random_walks_dir=str(random_walks_dir),
                output_dir=str(output_dir)
            )
            
            # Display results
            logger.info("\n" + "=" * 80)
            logger.info("Batch Trajectory Generation Completed")
            logger.info("=" * 80)
            logger.info(f"✅ Generated {results['successful']}/{results['total']} trajectories")
            
            if results['failed'] > 0:
                logger.warning(f"⚠️  {results['failed']} trajectory generation(s) failed")
                logger.info("\nFailed walks:")
                for failed in results['failed_walks']:
                    logger.info(f"  - {failed['walk_id']}: {failed['error']}")
            
            # List generated files
            logger.info("\n" + "=" * 80)
            logger.info("Generated Files")
            logger.info("=" * 80)
            
            trajectory_files = list(output_dir.glob("*.json"))
            trajectory_files = [f for f in trajectory_files if f.name != "_generation_summary.json"]
            logger.info(f"📁 {len(trajectory_files)} trajectory JSON files saved")
            
            if trajectory_files:
                logger.info("\nSample trajectories:")
                for i, traj_file in enumerate(trajectory_files[:3]):
                    with open(traj_file, 'r', encoding='utf-8') as f:
                        traj = json.load(f)
                    
                    logger.info(f"\n  Trajectory {i+1}:")
                    logger.info(f"    ID: {traj['trajectory_id']}")
                    logger.info(f"    Domain: {traj['domain']}")
                    logger.info(f"    Sequence: {' → '.join(traj['sequence'][:5])}")
                    if len(traj['sequence']) > 5:
                        logger.info(f"              ... ({len(traj['sequence'])-5} more)")
        else:
            logger.error("Please specify either --random-walk or --random-walks")
            return
        
        logger.info("\n" + "=" * 80)
        logger.info("Example Completed Successfully!")
        logger.info("=" * 80)
        logger.info("Next steps:")
        logger.info("1. Review the generated trajectories")
        logger.info("2. Analyze interaction histories and state changes")
        logger.info("3. Use trajectories for agent training or evaluation")
        
    except Exception as e:
        logger.error(f"Trajectory generation failed: {e}")
        logger.exception(e)
        raise


if __name__ == "__main__":
    main()

