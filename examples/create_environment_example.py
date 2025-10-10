#!/usr/bin/env python3
"""
Environment creation example for WorldInteract framework.

This example demonstrates how to create a complete environment
from an API collection.
"""

import sys
import json
import dotenv
import logging
import argparse
from pathlib import Path


# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from worldInteract.core.build_environment import EnvironmentManager


# Load environment variables
dotenv.load_dotenv("../.env")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Create a complete environment from an API collection using WorldInteract framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python create_environment_example.py --api-collection data/dependency_graphs/my_dependency_graphs/domains/<any-domain-json-file>.json
        """
    )
    
    parser.add_argument(
        "--api-collection", "-a",
        type=str,
        default=None,
        help="Path to API collection file (default: data/apis_collections/api_collection_example.json)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for generated environment (default: auto-generated based on domain)"
    )
    
    parser.add_argument(
        "--use-code-agent",
        action="store_true",
        default=True,
        help="Use code agent for validation (always enabled, required for proper functionality)"
    )
    
    return parser.parse_args()


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main example function."""
    # Parse command line arguments
    args = parse_arguments()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting WorldInteract environment creation example")
    
    # Initialize environment manager
    env_manager = EnvironmentManager()
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    
    # Use command line arguments or default path for API collection
    if args.api_collection:
        api_collection_path = Path(args.api_collection)
        if not api_collection_path.is_absolute():
            logger.info(f"API collection path is not absolute, using project root: {project_root}")
            api_collection_path = project_root / api_collection_path
    else:
        raise ValueError("API collection path is required")
    
    # Log the paths being used
    logger.info(f"API collection file: {api_collection_path}")
    logger.info(f"Use code agent: {args.use_code_agent}")
    if args.output_dir:
        logger.info(f"Output directory: {args.output_dir}")
    
    if not api_collection_path.exists():
        logger.error(f"API collection file not found: {api_collection_path}")
        return
    
    try:
        # Create complete environment
        logger.info("Creating environment from API collection...")
        environment = env_manager.create_environment(
            api_collection_path=str(api_collection_path),
            use_code_agent=args.use_code_agent,
            output_dir=args.output_dir
        )
        
        # Display results
        domain = environment["domain"]
        tools_count = len(environment["tools"])
        validation_results = environment["validation_results"]
        
        logger.info(f"Environment created for domain: {domain}")
        logger.info(f"Generated {tools_count} tools")
        
        # Show validation results
        if validation_results:
            passed = sum(1 for result in validation_results.values() if result)
            total = len(validation_results)
            logger.info(f"Validation results: {passed}/{total} tools passed")
            
            # Show failed tools
            failed_tools = [name for name, passed in validation_results.items() if not passed]
            if failed_tools:
                logger.warning(f"Failed tools: {failed_tools}")
        
        # Display schema overview
        schema = environment["schema"]
        logger.info("Generated schema tables:")
        for table_name, table_def in schema.items():
            field_count = len(table_def.get("fields", {}))
            logger.info(f"  - {table_name}: {field_count} fields")
        
        # Display initial state overview
        initial_state = environment["initial_state"]
        logger.info("Initial state overview:")
        for table_name, table_data in initial_state.items():
            record_count = len(table_data) if isinstance(table_data, dict) else 0
            logger.info(f"  - {table_name}: {record_count} records")
        
        # Show generated tool names
        tool_names = list(environment["tools"].keys())
        logger.info(f"Generated tools: {tool_names}")
        
        # Display test cases overview
        test_cases = environment.get("test_cases", {})
        if test_cases:
            logger.info("Test cases overview:")
            for tool_name, cases in test_cases.items():
                case_count = len(cases) if isinstance(cases, list) else 0
                logger.info(f"  - {tool_name}: {case_count} test cases")
        
        # Show output directory
        output_dir = environment["output_dir"]
        logger.info(f"Environment saved to: {output_dir}")
        
        logger.info("Environment creation example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during environment creation: {e}")
        raise


if __name__ == "__main__":
    main()

