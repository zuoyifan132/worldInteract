#!/usr/bin/env python3
"""
Scenario Collection and Dependency Graph Pipeline Script

This script implements the complete pipeline from raw APIs to domain-specific tool environments:
1. Scenario Collection (API cleaning and standardization)
2. Tool Dependency Graph Modeling (similarity analysis and community detection)
3. Integration with existing tool generation pipeline
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
from worldInteract.core.build_environment import ToolGenerator, SchemaGenerator, EnvironmentManager


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


def run_tool_generation(domains_dir: str, output_dir: str):
    """
    Run tool generation for each domain.
    
    Args:
        domains_dir: Directory containing domain JSON files
        output_dir: Directory to save generated tools
        
    Returns:
        List of generated domain paths
    """
    logger.info("=== Starting Tool Generation Phase ===")
    
    domains_path = Path(domains_dir)
    if not domains_path.exists():
        raise FileNotFoundError(f"Domains directory not found: {domains_dir}")
    
    # Get all domain JSON files
    domain_files = list(domains_path.glob("*.json"))
    if not domain_files:
        raise FileNotFoundError(f"No domain files found in {domains_dir}")
    
    logger.info(f"Found {len(domain_files)} domains to process")
    
    # Initialize generators
    schema_generator = SchemaGenerator()
    tool_generator = ToolGenerator()
    env_manager = EnvironmentManager()
    
    generated_domains = []
    
    for domain_file in domain_files:
        try:
            logger.info(f"Processing domain: {domain_file.stem}")
            
            # Load domain data
            import json
            with open(domain_file, 'r', encoding='utf-8') as f:
                domain_data = json.load(f)
            
            domain_name = domain_data["domain"]
            tools = domain_data["tools"]
            
            # Create output directory for this domain
            domain_output_dir = Path(output_dir) / domain_name
            domain_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to API collection format expected by existing generators
            api_collection = {
                "domain": domain_name,
                "description": domain_data["description"],
                "tools": tools
            }
            
            # Generate schema
            logger.info(f"Generating schema for domain: {domain_name}")
            schema = schema_generator.generate_schema(api_collection)
            
            # Save schema
            schema_path = domain_output_dir / "schema.json"
            with open(schema_path, 'w', encoding='utf-8') as f:
                json.dump(schema, f, indent=2, ensure_ascii=False)
            
            # Generate initial state
            logger.info(f"Generating initial state for domain: {domain_name}")
            initial_state = env_manager.generate_initial_state(api_collection, schema)
            
            # Save initial state
            state_path = domain_output_dir / "initial_state.json"
            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(initial_state, f, indent=2, ensure_ascii=False)
            
            # Generate tools
            logger.info(f"Generating tools for domain: {domain_name}")
            generated_tools = tool_generator.generate_tools(api_collection, schema, initial_state)
            
            # Save tools
            tools_dir = domain_output_dir / "tools"
            tools_dir.mkdir(exist_ok=True)
            
            # Save individual tool files
            for tool_name, tool_code in generated_tools.items():
                tool_file = tools_dir / f"{tool_name}.py"
                with open(tool_file, 'w', encoding='utf-8') as f:
                    f.write(tool_code)
            
            # Save combined tools file
            combined_tools_path = domain_output_dir / "tools.py"
            tool_generator.save_combined_tools(generated_tools, combined_tools_path)
            
            # Copy API collection for reference
            api_collection_path = domain_output_dir / "api_collection.json"
            with open(api_collection_path, 'w', encoding='utf-8') as f:
                json.dump(api_collection, f, indent=2, ensure_ascii=False)
            
            generated_domains.append(str(domain_output_dir))
            logger.info(f"Successfully generated domain: {domain_name}")
            
        except Exception as e:
            logger.error(f"Failed to process domain {domain_file.stem}: {e}")
            continue
    
    logger.info(f"Tool generation completed. Generated {len(generated_domains)} domains.")
    return generated_domains


def run_complete_pipeline(
    raw_apis_path: str,
    output_base_dir: str,
    skip_scenario_collection: bool = False,
    skip_dependency_graph: bool = False,
    skip_tool_generation: bool = False
):
    """
    Run the complete pipeline from raw APIs to generated tools.
    
    Args:
        raw_apis_path: Path to raw APIs JSON file
        output_base_dir: Base directory for all outputs
        skip_scenario_collection: Skip scenario collection phase
        skip_dependency_graph: Skip dependency graph modeling phase
        skip_tool_generation: Skip tool generation phase
        
    Returns:
        Dictionary with paths to all generated outputs
    """
    logger.info("=== Starting Complete Scenario Pipeline ===")
    
    # Create output directories
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    processed_apis_dir = output_base / "processed_apis"
    dependency_graphs_dir = output_base / "dependency_graphs"
    generated_domains_dir = output_base / "generated_domains"
    
    processed_apis_dir.mkdir(exist_ok=True)
    dependency_graphs_dir.mkdir(exist_ok=True)
    generated_domains_dir.mkdir(exist_ok=True)
    
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
        results["dependency_graphs_dir"] = run_dependency_graph_modeling(
            results["cleaned_apis_path"], 
            str(dependency_graphs_dir)
        )
    else:
        # Assume dependency graphs already exist
        domains_dir = dependency_graphs_dir / "domains"
        if not domains_dir.exists():
            raise FileNotFoundError(f"Domains directory not found: {domains_dir}")
        results["dependency_graphs_dir"] = str(domains_dir)
    
    # Phase 3: Tool Generation
    if not skip_tool_generation:
        results["generated_domains"] = run_tool_generation(
            results["dependency_graphs_dir"],
            str(generated_domains_dir)
        )
    else:
        logger.info("Skipping tool generation phase")
        results["generated_domains"] = []
    
    logger.info("=== Complete Pipeline Finished ===")
    logger.info(f"Results summary:")
    for key, value in results.items():
        logger.info(f"  {key}: {value}")
    
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
        "--skip-tool-generation",
        action="store_true",
        help="Skip tool generation phase"
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
            skip_dependency_graph=args.skip_dependency_graph,
            skip_tool_generation=args.skip_tool_generation
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
