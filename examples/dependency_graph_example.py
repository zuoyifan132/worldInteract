#!/usr/bin/env python3
"""
Example: Tool Dependency Graph Modeling Pipeline
This example demonstrates how to use the DependencyGraphBuilder to create
tool dependency graphs from cleaned API scenarios.
"""

import os
import sys
import json
import dotenv
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from worldInteract.core.dependency_graph import DependencyGraphBuilder

# Load environment variables
dotenv.load_dotenv("../.env")


def main():
    """Run the dependency graph modeling example"""
    
    logger.info("Starting Tool Dependency Graph Modeling Example")
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    
    # Look for cleaned APIs in multiple possible locations
    possible_input_paths = [
        project_root / "data" / "processed_apis" / "example_run" / "cleaned_apis.json",
    ]
    
    # Find the most recent cleaned APIs file
    cleaned_apis_path = None
    for path in possible_input_paths:
        if path.exists():
            cleaned_apis_path = path
            break
    
    if not cleaned_apis_path:
        logger.error("No cleaned APIs file found!")
        logger.info("Available search paths:")
        for path in possible_input_paths:
            logger.info(f"  - {path}")
        logger.info("\nPlease run scenario_collection_example.py first to generate cleaned APIs")
        return
    
    # Create output directory for dependency graphs
    dependency_graphs_dir = project_root / "data" / "dependency_graphs" / "dependency_graph_example"
    dependency_graphs_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Using cleaned APIs from: {cleaned_apis_path}")
    logger.info(f"Dependency graphs will be saved to: {dependency_graphs_dir}")
    
    # Validate input file
    try:
        with open(cleaned_apis_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Show input data overview
        if isinstance(data, dict):
            logger.info("Input data structure:")
            for key, value in data.items():
                if isinstance(value, list):
                    logger.info(f"  {key}: {len(value)} items")
                elif isinstance(value, dict):
                    logger.info(f"  {key}: {len(value)} keys")
                else:
                    logger.info(f"  {key}: {type(value).__name__}")
        elif isinstance(data, list):
            logger.info(f"Input data: {len(data)} API entries")
            
    except Exception as e:
        logger.error(f"Failed to read cleaned APIs file: {e}")
        return
    
    try:
        # Initialize the dependency graph builder
        logger.info("=== Initializing DependencyGraphBuilder ===")
        builder = DependencyGraphBuilder()
        
        # Build tool dependency graphs
        logger.info("=== Starting Tool Dependency Graph Modeling ===")
        logger.info("This process will:")
        logger.info("1. Analyze API scenarios for tool dependencies")
        logger.info("2. Group related tools into domains")
        logger.info("3. Generate tool implementations and schemas")
        logger.info("4. Create validation reports")
        
        graph_result = builder.build_dependency_graph(
            str(cleaned_apis_path), 
            str(dependency_graphs_dir)
        )
        
        # Display detailed results
        logger.info("=== Tool Dependency Graph Modeling Completed ===")
        
        if isinstance(graph_result, dict):
            logger.info("Processing Results:")
            for phase, phase_stats in graph_result.items():
                if isinstance(phase_stats, dict):
                    logger.info(f"  {phase}:")
                    for key, value in phase_stats.items():
                        if isinstance(value, (int, float, str)):
                            logger.info(f"    {key}: {value}")
                        elif isinstance(value, list):
                            logger.info(f"    {key}: {len(value)} items")
                        elif isinstance(value, dict):
                            logger.info(f"    {key}: {len(value)} entries")
                else:
                    logger.info(f"  {phase}: {phase_stats}")
        
        # Explore generated outputs
        logger.info("\n=== Generated Outputs ===")
        
        # Check for domain directories
        domains_dir = dependency_graphs_dir / "domains"
        if domains_dir.exists():
            domain_folders = [d for d in domains_dir.iterdir() if d.is_dir()]
            logger.info(f"Generated {len(domain_folders)} domain(s):")
            
            for domain_folder in domain_folders:
                logger.info(f"\n  📁 Domain: {domain_folder.name}")
                
                # Check for key files in each domain
                key_files = ["schema.json", "tools.py", "initial_state.json", "environment_metadata.json"]
                for key_file in key_files:
                    file_path = domain_folder / key_file
                    if file_path.exists():
                        file_size = file_path.stat().st_size
                        logger.info(f"    ✅ {key_file} ({file_size:,} bytes)")
                    else:
                        logger.info(f"    ❌ {key_file} (missing)")
                
                # Check tools directory
                tools_dir = domain_folder / "tools"
                if tools_dir.exists() and tools_dir.is_dir():
                    tool_files = list(tools_dir.glob("*.py"))
                    logger.info(f"    🔧 Generated {len(tool_files)} tool(s):")
                    for tool_file in tool_files:
                        logger.info(f"       - {tool_file.name}")
                
                # Check validation report
                validation_report = domain_folder / "validation_report.json"
                if validation_report.exists():
                    try:
                        with open(validation_report, 'r', encoding='utf-8') as f:
                            report_data = json.load(f)
                        logger.info(f"    📊 Validation Report:")
                        if "summary" in report_data:
                            summary = report_data["summary"]
                            for key, value in summary.items():
                                logger.info(f"       {key}: {value}")
                    except Exception as e:
                        logger.warning(f"    ⚠️  Could not read validation report: {e}")
        else:
            logger.warning("No domains directory found in output")
        
        # Check for other output files
        other_files = [f for f in dependency_graphs_dir.iterdir() if f.is_file()]
        if other_files:
            logger.info(f"\nOther generated files:")
            for file_path in other_files:
                file_size = file_path.stat().st_size
                logger.info(f"  - {file_path.name} ({file_size:,} bytes)")
        
        logger.info("\n=== Tool Dependency Graph Example Completed Successfully! ===")
        logger.info("Next steps:")
        logger.info("1. Review the generated domain structures")
        logger.info("2. Examine the tool implementations in each domain")
        logger.info("3. Check validation reports for any issues")
        logger.info("4. Use the generated tools in your applications")
        
    except Exception as e:
        logger.error(f"Tool Dependency Graph Modeling failed: {e}")
        logger.error("Please check the error details above and ensure:")
        logger.error("1. Cleaned APIs data is properly formatted")
        logger.error("2. All required dependencies are installed")
        logger.error("3. Environment variables are correctly set")
        logger.error("4. Sufficient disk space for output generation")
        raise


if __name__ == "__main__":
    main()
