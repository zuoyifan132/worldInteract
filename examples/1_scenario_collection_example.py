#!/usr/bin/env python3
"""
Example: Scenario Collection Pipeline
This example demonstrates how to use the APICleaner to process raw API data
and create cleaned scenarios for further processing.
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

# Load environment variables
dotenv.load_dotenv("../.env")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run scenario collection example to process raw API data and create cleaned scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python scenario_collection_example.py --input-dir data/raw_apis --output-file data/processed_apis/my_cleaned_apis.json
        """
    )
    
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        default=None,
        help="Input directory path for raw API data (default: data/raw_apis)"
    )
    
    parser.add_argument(
        "--output-file", "-o", 
        type=str,
        default=None,
        help="Output file path for cleaned API data (default: data/processed_apis/scenario_collection_example/cleaned_apis.json)"
    )
    
    return parser.parse_args()


def main():
    """Run the scenario collection example"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info("Starting Scenario Collection Example")
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    
    # Use command line arguments or default paths
    if args.input_dir:
        raw_apis_path = Path(args.input_dir)
        if not raw_apis_path.is_absolute():
            raw_apis_path = project_root / raw_apis_path
    else:
        raw_apis_path = project_root / "data" / "raw_apis"
    
    if args.output_file:
        processed_apis_path = Path(args.output_file)
        if not processed_apis_path.is_absolute():
            processed_apis_path = project_root / processed_apis_path
        # Create output directory if it doesn't exist
        processed_apis_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Create output directory for processed APIs (default path)
        processed_apis_output_dir = project_root / "data" / "processed_apis" / "scenario_collection_example"
        processed_apis_output_dir.mkdir(parents=True, exist_ok=True)
        processed_apis_path = processed_apis_output_dir / "cleaned_apis.json"
    
    # Log the paths being used
    logger.info(f"Input directory: {raw_apis_path}")
    logger.info(f"Output file: {processed_apis_path}")
    
    # Validate input directory exists
    if not raw_apis_path.exists():
        logger.error(f"Raw APIs directory does not exist: {raw_apis_path}")
        logger.info("Please ensure you have raw API data in the data/raw_apis directory")
        return
    
    # Check for input files
    api_files = list(raw_apis_path.glob("*.json"))
    if not api_files:
        logger.error(f"No JSON files found in: {raw_apis_path}")
        logger.info("Please add some raw API JSON files to process")
        return
    
    logger.info(f"Found {len(api_files)} API files to process:")
    for api_file in api_files:
        logger.info(f"  - {api_file.name}")
    
    try:
        # Initialize the API cleaner
        logger.info("=== Initializing APICleaner ===")
        cleaner = APICleaner()
        
        # Perform scenario collection (API cleaning)
        logger.info("=== Starting Scenario Collection ===")
        logger.info(f"Processing APIs from: {raw_apis_path}")
        logger.info(f"Output will be saved to: {processed_apis_path}")
        
        cleaning_result = cleaner.clean_apis(str(raw_apis_path), str(processed_apis_path))
        
        # Display processing statistics
        logger.info("=== Scenario Collection Completed ===")
        
        if "metadata" in cleaning_result and "processing_stats" in cleaning_result["metadata"]:
            stats = cleaning_result["metadata"]["processing_stats"]
            logger.info("Processing Statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.info("No detailed statistics available")
        
        # Verify output file was created
        if processed_apis_path.exists():
            file_size = processed_apis_path.stat().st_size
            logger.info(f"✅ Cleaned APIs saved successfully")
            logger.info(f"   File: {processed_apis_path}")
            logger.info(f"   Size: {file_size:,} bytes")
            
            # Show a brief preview of the cleaned data structure
            import json
            with open(processed_apis_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    logger.info(f"   Data structure keys: {list(data.keys())}")
                    if "scenarios" in data:
                        logger.info(f"   Number of scenarios: {len(data['scenarios'])}")
                elif isinstance(data, list):
                    logger.info(f"   Number of API entries: {len(data)}")
        else:
            logger.error("❌ Output file was not created")
        
        logger.info("\n=== Scenario Collection Example Completed Successfully! ===")
        logger.info("Next steps:")
        logger.info("1. Review the cleaned APIs in the output file")
        logger.info("2. Use domain_graph_example.py to build domain graphs")
        logger.info("3. Or run the full pipeline with scenario_pipeline_example.py")
        
    except Exception as e:
        logger.error(f"Scenario Collection failed: {e}")
        logger.error("Please check the error details above and ensure:")
        logger.error("1. Raw API data is properly formatted")
        logger.error("2. All required dependencies are installed")
        logger.error("3. Environment variables are correctly set")
        raise


if __name__ == "__main__":
    main()
