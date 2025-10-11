#!/usr/bin/env python3
"""
Scenario Collection and Dependency Graph Pipeline Script

This script implements the pipeline from raw APIs to domain graphs:
1. Scenario Collection (API cleaning and standardization)
2. Domain Dependency Graph Modeling (similarity analysis and community detection)

Note: Tool generation is handled separately by generate_domain.py script.
"""

import os
import sys
import argparse
from pathlib import Path
from loguru import logger

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from worldInteract.core.scenario_collection import APICleaner
from worldInteract.core.build_domain_graph import DomainGraphBuilder


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level
    )


def run_scenario_collection(raw_apis_path: str, output_path: str) -> str:
    """
    Run scenario collection phase.
    
    Args:
        raw_apis_path: Path to raw APIs JSON file
        output_path: Path to save cleaned APIs
        
    Returns:
        Path to cleaned APIs file
    """
    logger.info("=== Starting Scenario Collection Phase ===")
    
    cleaner = APICleaner()
    result = cleaner.clean_apis(raw_apis_path, output_path)
    
    logger.info(f"Scenario Collection completed. Statistics:")
    stats = result.get("metadata", {}).get("processing_stats", {})
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    return output_path


def run_dependency_graph_modeling(cleaned_apis_path: str, output_dir: str) -> str:
    """
    Run dependency graph modeling phase.
    
    Args:
        cleaned_apis_path: Path to cleaned APIs JSON file
        output_dir: Directory to save graph outputs
        
    Returns:
        Path to domains directory
    """
    logger.info("=== Starting Tool Dependency Graph Modeling Phase ===")
    
    builder = DomainGraphBuilder()
    result = builder.build_dependency_graph(cleaned_apis_path, output_dir)
    
    logger.info(f"Dependency Graph Modeling completed. Statistics:")
    for phase, stats in result.items():
        if isinstance(stats, dict):
            logger.info(f"  {phase}:")
            for key, value in stats.items():
                logger.info(f"    {key}: {value}")
    
    return os.path.join(output_dir, "domains")




def run_complete_pipeline(
    raw_apis_path: str,
    output_base_dir: str,
    skip_scenario_collection: bool = False,
    skip_dependency_graph: bool = False
):
    """
    Run the complete pipeline from raw APIs to domain graphs.
    
    This pipeline performs:
    1. Scenario Collection (API cleaning and standardization)
    2. Dependency Graph Modeling (domain analysis and graph building)
    
    Note: Tool generation is handled separately by generate_domain.py script.
    
    Args:
        raw_apis_path: Path to raw APIs JSON file or directory
        output_base_dir: Base directory for all outputs
        skip_scenario_collection: Skip scenario collection phase
        skip_dependency_graph: Skip dependency graph modeling phase
        
    Returns:
        Dictionary with paths to all generated outputs
    """
    logger.info("=== Starting Scenario Pipeline ===")
    logger.info("This pipeline will:")
    logger.info("  1. Clean and standardize raw APIs")
    logger.info("  2. Build domain dependency graphs")
    logger.info("  Note: Use generate_domain.py to generate tools from domain graphs")
    
    # Create output directories
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    processed_apis_dir = output_base / "processed_apis"
    dependency_graphs_dir = output_base / "dependency_graphs"
    
    processed_apis_dir.mkdir(exist_ok=True)
    dependency_graphs_dir.mkdir(exist_ok=True)
    
    results = {
        "raw_apis_path": raw_apis_path,
        "output_base_dir": str(output_base),
    }
    
    # Phase 1: Scenario Collection
    if not skip_scenario_collection:
        cleaned_apis_path = processed_apis_dir / "cleaned_apis.json"
        results["cleaned_apis_path"] = run_scenario_collection(raw_apis_path, str(cleaned_apis_path))
    else:
        # Assume cleaned APIs already exist
        cleaned_apis_path = processed_apis_dir / "cleaned_apis.json"
        if not cleaned_apis_path.exists():
            raise FileNotFoundError(f"Cleaned APIs not found: {cleaned_apis_path}")
        results["cleaned_apis_path"] = str(cleaned_apis_path)
    
    # Phase 2: Dependency Graph Modeling
    if not skip_dependency_graph:
        results["domains_dir"] = run_dependency_graph_modeling(
            results["cleaned_apis_path"], 
            str(dependency_graphs_dir)
        )
    else:
        # Assume dependency graphs already exist
        domains_dir = dependency_graphs_dir / "domains"
        if not domains_dir.exists():
            raise FileNotFoundError(f"Domains directory not found: {domains_dir}")
        results["domains_dir"] = str(domains_dir)
    
    logger.info("=== Pipeline Finished ===")
    logger.info(f"Results summary:")
    for key, value in results.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\n=== Next Steps ===")
    logger.info(f"To generate tools for domains, run:")
    domains_dir = Path(results["domains_dir"])
    if domains_dir.exists():
        domain_files = list(domains_dir.glob("*.json"))
        for domain_file in domain_files[:3]:  # Show first 3 examples
            logger.info(f"  python scripts/generate_domain.py {domain_file}")
        if len(domain_files) > 3:
            logger.info(f"  ... and {len(domain_files) - 3} more domains")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Scenario Collection and Dependency Graph Pipeline")
    parser.add_argument(
        "raw_apis_path",
        help="Path to raw APIs JSON file or directory containing JSON files"
    )
    parser.add_argument(
        "-o", "--output",
        default="./output/scenario_pipeline",
        help="Output base directory (default: ./output/scenario_pipeline)"
    )
    parser.add_argument(
        "--skip-scenario-collection",
        action="store_true",
        help="Skip scenario collection phase"
    )
    parser.add_argument(
        "--skip-dependency-graph",
        action="store_true", 
        help="Skip dependency graph modeling phase"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate input path
    if not os.path.exists(args.raw_apis_path):
        logger.error(f"Raw APIs path not found: {args.raw_apis_path}")
        sys.exit(1)
    
    try:
        # Run pipeline
        results = run_complete_pipeline(
            raw_apis_path=args.raw_apis_path,
            output_base_dir=args.output,
            skip_scenario_collection=args.skip_scenario_collection,
            skip_dependency_graph=args.skip_dependency_graph
        )
        
        logger.info("Pipeline completed successfully!")
        
        # Save results summary
        import json
        results_file = Path(args.output) / "pipeline_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results summary saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
