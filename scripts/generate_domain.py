#!/usr/bin/env python3
"""
Domain generation script for WorldInteract.

This script provides a command-line interface for generating environments
from API collections.
"""

import argparse
import json
import logging
import sys
import dotenv
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# load .env file
dotenv.load_dotenv("../.env")

from worldInteract.core.environment import EnvironmentManager


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def generate_domain(
    api_collection_path: str,
    validate: bool = True,
    verbose: bool = False
):
    """
    Generate a domain environment from API collection.
    
    Args:
        api_collection_path: Path to API collection JSON file
        validate: Whether to validate generated tools
        verbose: Enable verbose logging
        
    Note:
        Output directory is automatically determined from the domain field 
        in the API collection: data/generated/domains/{domain}/
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    # Validate input file
    api_path = Path(api_collection_path)
    if not api_path.exists():
        logger.error(f"API collection file not found: {api_collection_path}")
        return False
    
    # Load API collection to get domain name
    try:
        with open(api_path, 'r', encoding='utf-8') as f:
            api_collection = json.load(f)
        domain = api_collection.get("domain", "unknown")
        logger.info(f"Generating environment for domain: {domain}")
    except Exception as e:
        logger.error(f"Failed to load API collection: {e}")
        return False
    
    # Initialize environment manager
    env_manager = EnvironmentManager()
    
    try:
        # Create environment (output_dir is automatically determined from domain)
        logger.info("Starting environment generation...")
        environment = env_manager.create_environment(
            api_collection_path=api_collection_path,
            validate_tools=validate
        )
        
        # Display results
        logger.info(f"Environment generation completed for domain: {domain}")
        logger.info(f"Generated {len(environment['tools'])} tools")
        
        if validate and environment.get("validation_results"):
            validation_results = environment["validation_results"]
            passed = sum(1 for result in validation_results.values() if result)
            total = len(validation_results)
            success_rate = (passed / total) * 100 if total > 0 else 0
            logger.info(f"Validation: {passed}/{total} tools passed ({success_rate:.1f}%)")
            
            failed_tools = [name for name, passed in validation_results.items() if not passed]
            if failed_tools:
                logger.warning(f"Failed tools: {failed_tools}")
        
        output_path = environment["output_dir"]
        logger.info(f"Environment saved to: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate environment: {e}")
        if verbose:
            import traceback
            logger.error(traceback.format_exc())
        return False


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Generate domain environment from API collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate environment with validation
  python scripts/generate_domain.py data/apis_collections/file_operations.json
  
  # Generate without validation
  python scripts/generate_domain.py data/apis_collections/web_browsing.json --no-validate
  
  # Enable verbose logging
  python scripts/generate_domain.py api_collection.json --verbose
        """
    )
    
    parser.add_argument(
        "api_collection",
        help="Path to API collection JSON file"
    )
    
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip tool validation (faster but less reliable)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Generate domain
    success = generate_domain(
        api_collection_path=args.api_collection,
        validate=not args.no_validate,
        verbose=args.verbose
    )
    
    if success:
        print("Domain generation completed successfully!")
        sys.exit(0)
    else:
        print("Domain generation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

