"""
Environment creation example for WorldInteract framework.

This example demonstrates how to create a complete environment
from an API collection.
"""

import sys
import json
import logging
from pathlib import Path

sys.path.append("..")

from worldInteract.core.environment import EnvironmentManager


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main example function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting WorldInteract environment creation example")
    
    # Initialize environment manager
    env_manager = EnvironmentManager()
    
    # Path to API collection
    project_root = Path(__file__).parent.parent
    api_collection_path = project_root / "data" / "apis_collections" / "api_collection_example.json"
    # api_collection_path = project_root / "data" / "apis_collections" / "ticket_api_example.json"
    
    if not api_collection_path.exists():
        logger.error(f"API collection file not found: {api_collection_path}")
        return
    
    try:
        # Create complete environment
        logger.info("Creating environment from API collection...")
        environment = env_manager.create_environment(
            api_collection_path=str(api_collection_path),
            validate_tools=True
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
        
        # Show output directory
        output_dir = environment["output_dir"]
        logger.info(f"Environment saved to: {output_dir}")
        
        logger.info("Environment creation example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during environment creation: {e}")
        raise


if __name__ == "__main__":
    main()

