#!/usr/bin/env python3
"""
Example: Complete Scenario Collection and Dependency Graph Modeling Pipeline

This example demonstrates the complete workflow from raw API data
to dependency graph modeling.
"""

import os
import sys
import dotenv
import argparse
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from worldInteract.core.scenario_collection import APICleaner
from worldInteract.core.dependency_graph import DependencyGraphBuilder


dotenv.load_dotenv("../.env")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run complete scenario pipeline: from raw API data to dependency graph modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python scenario_pipeline_example.py
  python scenario_pipeline_example.py --input-dir /path/to/raw/apis --output-dir /path/to/output
  python scenario_pipeline_example.py -i data/raw_apis -o output/my_pipeline_run
        """
    )
    
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        default=None,
        help="Input directory path for raw API data (default: data/raw_apis)"
    )
    
    parser.add_argument(
        "--output-dir", "-o", 
        type=str,
        default=None,
        help="Output directory path (default: data/processed_apis/example_run and data/dependency_graphs/example_run)"
    )
    
    return parser.parse_args()


def main():
    """Run the scenario pipeline example"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info("Starting the scenario pipeline example")
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    
    # Use command line arguments or default paths
    if args.input_dir:
        raw_apis_path = Path(args.input_dir)
        if not raw_apis_path.is_absolute():
            raw_apis_path = project_root / raw_apis_path
    else:
        raw_apis_path = project_root / "data" / "raw_apis"
    
    # Output directories
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
        if not base_output_dir.is_absolute():
            base_output_dir = project_root / base_output_dir
        
        processed_apis_output_dir = base_output_dir / "processed_apis"
        dependency_graphs_dir = base_output_dir / "dependency_graphs"
    else:
        processed_apis_output_dir = project_root / "data" / "processed_apis" / "example_run"
        dependency_graphs_dir = project_root / "data" / "dependency_graphs" / "example_run"
    
    # Create output directories
    processed_apis_output_dir.mkdir(parents=True, exist_ok=True)
    dependency_graphs_dir.mkdir(parents=True, exist_ok=True)
    
    processed_apis_path = processed_apis_output_dir / "cleaned_apis.json"
    
    # Log the paths being used
    logger.info(f"Input directory: {raw_apis_path}")
    logger.info(f"Processed APIs output path: {processed_apis_path}")
    logger.info(f"Dependency graphs output directory: {dependency_graphs_dir}")
    
    # Check input directory
    if not raw_apis_path.exists():
        logger.error(f"Raw APIs directory does not exist: {raw_apis_path}")
        logger.info("Please ensure you have raw API data in the data/raw_apis directory")
        return
    
    # Check for input files
    api_files = list(raw_apis_path.glob("*.json"))
    if not api_files:
        logger.error(f"No JSON files found in directory: {raw_apis_path}")
        logger.info("Please add some raw API JSON files to process")
        return
    
    logger.info(f"Found {len(api_files)} API files to process:")
    for api_file in api_files:
        logger.info(f"  - {api_file.name}")
    
    # try:
    # Step 1: Scenario Collection
    logger.info("=== Starting Scenario Collection phase ===")
    cleaner = APICleaner()
    cleaning_result = cleaner.clean_apis(str(raw_apis_path), str(processed_apis_path))
    
    logger.info("Scenario Collection completed, statistics:")
    stats = cleaning_result.get("metadata", {}).get("processing_stats", {})
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Step 2: Tool Dependency Graph Modeling  
    logger.info("=== Starting Tool Dependency Graph Modeling phase ===")
    builder = DependencyGraphBuilder()
    graph_result = builder.build_dependency_graph(
        str(processed_apis_path), 
        str(dependency_graphs_dir)
    )
    
    logger.info("Dependency Graph Modeling completed, statistics:")
    for phase, phase_stats in graph_result.items():
        if isinstance(phase_stats, dict):
            logger.info(f"  {phase}:")
            for key, value in phase_stats.items():
                logger.info(f"    {key}: {value}")
    
    logger.info("\n=== Pipeline completed ===")
    logger.info(f"Cleaned APIs: {processed_apis_path}")
    logger.info(f"Dependency graph data: {dependency_graphs_dir}")
    logger.info(f"Generated domains: {dependency_graphs_dir / 'domains'}")
    
    # Show generated domains
    domains_dir = dependency_graphs_dir / "domains"
    if domains_dir.exists():
        domain_files = list(domains_dir.glob("*.json"))
        logger.info(f"Generated {len(domain_files)} domains:")
        for domain_file in domain_files:
            logger.info(f"  - {domain_file.name}")
    
    logger.info("\nExample run completed!")
        
    # except Exception as e:
    #     logger.error(f"Example run failed: {e}")
    #     raise


if __name__ == "__main__":
    main()
