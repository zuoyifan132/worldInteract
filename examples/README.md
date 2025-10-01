# WorldInteract Examples

This directory contains various usage examples for the WorldInteract framework. All examples support command line arguments and you can view detailed usage instructions with `--help` or `-h`.

## Example List

### 1. Environment Creation Example (create_environment_example.py)

Demonstrates how to create a complete environment from an API collection.

**Basic Usage:**
```bash
# Use default settings
python create_environment_example.py

# Specify API collection file
python create_environment_example.py --api-collection data/apis_collections/ticket_api_example.json

# Specify output directory
python create_environment_example.py --output-dir output/my_environment
```

**Parameters:**
- `--api-collection, -a`: API collection file path (default: data/apis_collections/api_collection_example.json)
- `--output-dir, -o`: Output directory path (default: auto-generated based on domain)
- `--use-code-agent`: Enable code agent validation (default: enabled)
- `--no-code-agent`: Disable code agent validation

### 2. Scenario Collection Example (scenario_collection_example.py)

Demonstrates how to process raw API data and create cleaned scenarios.

**Basic Usage:**
```bash
# Use default settings
python scenario_collection_example.py

# Specify input directory and output file
python scenario_collection_example.py --input-dir data/raw_apis --output-file output/my_cleaned_apis.json

# Use short parameters
python scenario_collection_example.py -i data/raw_apis -o output/cleaned_apis.json
```

**Parameters:**
- `--input-dir, -i`: Input directory path for raw API data (default: data/raw_apis)
- `--output-file, -o`: Output file path for cleaned API data (default: data/processed_apis/scenario_collection_example/cleaned_apis.json)

### 3. Dependency Graph Example (dependency_graph_example.py)

Demonstrates how to create tool dependency graphs from cleaned API scenarios.

**Basic Usage:**
```bash
# Use default settings
python dependency_graph_example.py

# Specify input file and output directory
python dependency_graph_example.py --input-file data/processed_apis/example_run/cleaned_apis.json --output-dir output/my_dependency_graphs

# Use short parameters
python dependency_graph_example.py -i data/processed_apis/cleaned_apis.json -o output/dependency_graphs
```

**Parameters:**
- `--input-file, -i`: Input file path for cleaned API data (default: auto-search for cleaned_apis.json in processed_apis directories)
- `--output-dir, -o`: Output directory path for dependency graphs (default: data/dependency_graphs/dependency_graph_example)

### 4. Scenario Pipeline Example (scenario_pipeline_example.py)

Demonstrates the complete scenario pipeline: from raw API data to dependency graph modeling.

**Basic Usage:**
```bash
# Use default settings
python scenario_pipeline_example.py

# Specify input directory and output directory
python scenario_pipeline_example.py --input-dir data/raw_apis --output-dir output/my_pipeline_run

# Use short parameters
python scenario_pipeline_example.py -i data/raw_apis -o output/pipeline_output
```

**Parameters:**
- `--input-dir, -i`: Input directory path for raw API data (default: data/raw_apis)
- `--output-dir, -o`: Output directory path (default: data/processed_apis/example_run and data/dependency_graphs/example_run)

## Recommended Execution Order

If you're using the WorldInteract framework for the first time, we recommend running the examples in the following order:

1. **Scenario Collection Example** - First process raw API data
   ```bash
   python scenario_collection_example.py
   ```

2. **Dependency Graph Example** - Create dependency graphs based on cleaned data
   ```bash
   python dependency_graph_example.py
   ```

3. **Environment Creation Example** - Create complete environment from API collection
   ```bash
   python create_environment_example.py
   ```

4. **Scenario Pipeline Example** - Run the complete end-to-end process
   ```bash
   python scenario_pipeline_example.py
   ```

## General Tips

- All examples support the `--help` parameter to view detailed usage instructions
- Relative paths are resolved relative to the project root directory
- Output directories are automatically created if they don't exist
- Ensure required input files exist before running examples
- Check log output to understand processing progress and results

## Troubleshooting

If you encounter issues, please check:

1. **File Paths**: Ensure input files exist and paths are correct
2. **Dependencies**: Ensure all required dependencies are installed (`pip install -r requirements.txt`)
3. **Environment Variables**: Ensure `.env` file is configured correctly
4. **Permissions**: Ensure you have write permissions to output directories
5. **Logs**: Check detailed log output to understand specific error messages

For more help, please refer to the `README.md` file in the project root directory or check source code comments.
