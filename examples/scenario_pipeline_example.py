#!/usr/bin/env python3
"""
示例：演示Scenario Collection和Dependency Graph建模的完整流程
"""

import os
import sys
import dotenv
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from worldInteract.core.scenario_collection import APICleaner
from worldInteract.core.dependency_graph import DependencyGraphBuilder


dotenv.load_dotenv("../.env")


def main():
    """Run the scenario pipeline example"""
    
    logger.info("Starting the scenario pipeline example")
    
    # set paths
    project_root = Path(__file__).parent.parent
    raw_apis_path = project_root / "data" / "raw_apis"
    
    # output directory
    processed_apis_output_dir = project_root / "data" / "processed_apis" / "example_run"
    processed_apis_output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_apis_path = processed_apis_output_dir / "cleaned_apis.json"
    dependency_graphs_dir = project_root / "data" / "dependency_graphs" / "example_run"
    dependency_graphs_dir.mkdir(parents=True, exist_ok=True)
    
    # check input directory
    if not raw_apis_path.exists():
        logger.error(f"Raw APIs directory does not exist: {raw_apis_path}")
        return
    
    # try:
    # step1: Scenario Collection
    logger.info("=== Starting Scenario Collection phase ===")
    cleaner = APICleaner()
    cleaning_result = cleaner.clean_apis(str(raw_apis_path), str(processed_apis_path))
    
    logger.info("Scenario Collection completed, statistics:")
    stats = cleaning_result.get("metadata", {}).get("processing_stats", {})
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # step2: Tool Dependency Graph Modeling  
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
    
    # show generated domains
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
