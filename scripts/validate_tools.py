#!/usr/bin/env python3
"""
Tool validation script for WorldInteract.

This script provides a command-line interface for validating existing
generated tools.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from worldInteract.core.validator import ToolValidator
from worldInteract.core.environment import EnvironmentManager


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def validate_domain_tools(
    domain: str,
    environment_dir: str = None,
    verbose: bool = False
):
    """
    Validate tools for a specific domain.
    
    Args:
        domain: Domain name
        environment_dir: Environment directory (optional)
        verbose: Enable verbose logging
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Validating tools for domain: {domain}")
    
    # Load environment
    env_manager = EnvironmentManager()
    
    try:
        environment = env_manager.load_environment(domain, environment_dir)
        logger.info(f"Loaded environment for domain: {domain}")
    except FileNotFoundError:
        logger.error(f"Environment not found for domain: {domain}")
        if environment_dir:
            logger.error(f"Checked directory: {environment_dir}")
        return False
    except Exception as e:
        logger.error(f"Failed to load environment: {e}")
        return False
    
    # Extract components
    tools = environment["tools"]
    schema = environment["schema"]
    initial_state = environment["initial_state"]
    
    # Create mock API collection for validation
    api_collection = {
        "domain": domain,
        "tools": []
    }
    
    # Try to infer tool descriptions from tool names
    for tool_name in tools.keys():
        tool_desc = {
            "name": tool_name,
            "description": f"Tool: {tool_name}",
            "parameters": {},
            "returns": {"type": "object"}
        }
        api_collection["tools"].append(tool_desc)
    
    # Validate tools
    validator = ToolValidator()
    
    try:
        logger.info(f"Validating {len(tools)} tools...")
        validation_results = validator.validate_tools(
            tools, schema, initial_state, api_collection
        )
        
        # Display results
        passed = sum(1 for result in validation_results.values() if result)
        total = len(validation_results)
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        logger.info(f"Validation completed: {passed}/{total} tools passed ({success_rate:.1f}%)")
        
        # Show detailed results
        print("\nValidation Results:")
        print("=" * 50)
        
        for tool_name, passed in validation_results.items():
            status = "PASS" if passed else "FAIL"
            print(f"{tool_name:30} {status}")
        
        print("=" * 50)
        print(f"Summary: {passed}/{total} tools passed ({success_rate:.1f}%)")
        
        # Save validation report
        validator.save_validation_report(validation_results, domain)
        logger.info("Validation report saved")
        
        return success_rate == 100.0
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        if verbose:
            import traceback
            logger.error(traceback.format_exc())
        return False


def validate_specific_tools(
    domain: str,
    tool_names: list,
    environment_dir: str = None,
    verbose: bool = False
):
    """
    Validate specific tools in a domain.
    
    Args:
        domain: Domain name
        tool_names: List of tool names to validate
        environment_dir: Environment directory (optional)
        verbose: Enable verbose logging
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Validating specific tools in domain: {domain}")
    logger.info(f"Tools to validate: {tool_names}")
    
    # Load environment
    env_manager = EnvironmentManager()
    
    try:
        environment = env_manager.load_environment(domain, environment_dir)
    except Exception as e:
        logger.error(f"Failed to load environment: {e}")
        return False
    
    # Filter tools
    all_tools = environment["tools"]
    filtered_tools = {name: code for name, code in all_tools.items() if name in tool_names}
    
    missing_tools = set(tool_names) - set(filtered_tools.keys())
    if missing_tools:
        logger.error(f"Tools not found: {missing_tools}")
        return False
    
    # Create API collection for filtered tools
    api_collection = {
        "domain": domain,
        "tools": [
            {
                "name": tool_name,
                "description": f"Tool: {tool_name}",
                "parameters": {},
                "returns": {"type": "object"}
            }
            for tool_name in filtered_tools.keys()
        ]
    }
    
    # Validate filtered tools
    validator = ToolValidator()
    
    try:
        validation_results = validator.validate_tools(
            filtered_tools,
            environment["schema"],
            environment["initial_state"],
            api_collection
        )
        
        # Display results
        print("\nValidation Results:")
        print("=" * 50)
        
        for tool_name, passed in validation_results.items():
            status = "PASS" if passed else "FAIL"
            print(f"{tool_name:30} {status}")
        
        passed = sum(1 for result in validation_results.values() if result)
        total = len(validation_results)
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        print("=" * 50)
        print(f"Summary: {passed}/{total} tools passed ({success_rate:.1f}%)")
        
        return success_rate == 100.0
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


def list_domains():
    """List available domains."""
    project_root = Path(__file__).parent.parent
    domains_dir = project_root / "data" / "generated" / "domains"
    
    if not domains_dir.exists():
        print("No domains found. Generate some domains first.")
        return
    
    domains = [d.name for d in domains_dir.iterdir() if d.is_dir()]
    
    if not domains:
        print("No domains found.")
        return
    
    print("Available domains:")
    for domain in sorted(domains):
        print(f"  - {domain}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Validate generated tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all tools in a domain
  python scripts/validate_tools.py file_operations
  
  # Validate specific tools
  python scripts/validate_tools.py file_operations --tools create_file read_file
  
  # List available domains
  python scripts/validate_tools.py --list-domains
  
  # Use custom environment directory
  python scripts/validate_tools.py mydomain --env-dir /custom/path
        """
    )
    
    parser.add_argument(
        "domain",
        nargs="?",
        help="Domain name to validate"
    )
    
    parser.add_argument(
        "--tools",
        nargs="+",
        help="Specific tool names to validate (default: all tools)"
    )
    
    parser.add_argument(
        "--env-dir",
        help="Custom environment directory"
    )
    
    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="List available domains"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Handle list domains
    if args.list_domains:
        list_domains()
        return
    
    # Require domain for other operations
    if not args.domain:
        parser.error("Domain name is required (or use --list-domains)")
    
    # Validate tools
    if args.tools:
        success = validate_specific_tools(
            domain=args.domain,
            tool_names=args.tools,
            environment_dir=args.env_dir,
            verbose=args.verbose
        )
    else:
        success = validate_domain_tools(
            domain=args.domain,
            environment_dir=args.env_dir,
            verbose=args.verbose
        )
    
    if success:
        print("\nValidation completed successfully!")
        sys.exit(0)
    else:
        print("\nValidation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

