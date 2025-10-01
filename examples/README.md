# WorldInteract Examples

This directory contains various usage examples for the WorldInteract framework. All examples support command line arguments and you can view detailed usage instructions with `--help` or `-h`.

## Example List

### 1. Scenario Collection Example (scenario_collection_example.py)

Demonstrates how to process raw API data and create cleaned scenarios.

**Basic Usage:**
```bash
# Specify input directory and output file
python scenario_collection_example.py --input-dir data/raw_apis --output-file data/processed_apis/my_cleaned_apis.json
```

**Parameters:**
- `--input-dir, -i`: Input directory path for raw API data (default: data/raw_apis)
- `--output-file, -o`: Output file path for cleaned API data (default: data/processed_apis/scenario_collection_example/cleaned_apis.json)

### 2. Dependency Graph Example (dependency_graph_example.py)

Demonstrates how to create tool dependency graphs from cleaned API scenarios.

**Basic Usage:**
```bash
# Specify input file and output directory
python dependency_graph_example.py --input-file data/processed_apis/my_cleaned_apis.json --output-dir data/dependency_graphs/my_dependency_graphs
```

**Parameters:**
- `--input-file, -i`: Input file path for cleaned API data (default: auto-search for cleaned_apis.json in processed_apis directories)
- `--output-dir, -o`: Output directory path for dependency graphs (default: data/dependency_graphs/dependency_graph_example)

### 3. Environment Creation Example (create_environment_example.py)

Demonstrates how to create a complete environment from an API collection.

**Basic Usage:**
```bash
# Specify API collection file
python create_environment_example.py --api-collection data/dependency_graphs/my_dependency_graphs/domains/<any-domain-json-file>.json
```

**Parameters:**
- `--api-collection, -a`: API collection file path (default: data/apis_collections/api_collection_example.json)
- `--output-dir, -o`: Output directory path (default: auto-generated based on domain, for this example the output will be in data/generated/domains/file_operations)
- `--use-code-agent`: Code agent validation (always enabled, required for proper functionality)

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
