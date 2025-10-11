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

**Output File Structure:**
After running this example, the following output files will be generated:
- `cleaned_apis.json`: Cleaned API data file containing standardized API scenario information
  - Includes processing statistics and metadata
  - Formatted API call examples and parameter descriptions
  - Deduplicated and standardized API collections

### 2. Domain Graph Example (domain_graph_example.py)

Demonstrates how to create tool domain graphs from cleaned API scenarios.

**Basic Usage:**
```bash
# Specify input file and output directory
python domain_graph_example.py --input-file data/processed_apis/my_cleaned_apis.json --output-dir data/domain_graphs/my_domain_graphs
```

**Parameters:**
- `--input-file, -i`: Input file path for cleaned API data (default: auto-search for cleaned_apis.json in processed_apis directories)
- `--output-dir, -o`: Output directory path for domain graphs (default: data/domain_graphs/domain_graph_example)

**Output File Structure:**
After running this example, the following file structure will be generated in the output directory:
- `domain_graph.json`: Tool domain graph data
- `communities.json`: Tool community clustering information
- `domains.json`: Domain classification summary information
- `embeddings.json`: Tool vector embedding data
- `graph_visualization.png`: Domain graph visualization image
- `domains/`: Domain classification directory containing JSON files for each domain
  - `{domain_name}.json`: Specific API tool collections for each domain
  - Examples: `file_management.json`, `database_management.json`, `user_management.json`, etc.

### 3. Environment Creation Example (create_environment_example.py)

Demonstrates how to create a complete environment from an API collection.

**Basic Usage:**
```bash
# Specify API collection file
python create_environment_example.py --api-collection data/domain_graphs/my_domain_graphs/domains/<any-domain-json-file>.json
```

**Parameters:**
- `--api-collection, -a`: API collection file path (default: data/apis_collections/api_collection_example.json)
- `--output-dir, -o`: Output directory path (default: auto-generated based on domain, for this example the output will be in data/generated_env/domains/file_operations)
- `--use-code-agent`: Code agent validation (always enabled, required for proper functionality)

**Output File Structure:**
After running this example, a complete environment file structure will be generated in the output directory:
- `schema.json`: Database schema definition file
- `initial_state.json`: Environment initial state data
- `environment_metadata.json`: Environment metadata information
- `test_cases.json`: Test case collections
- `validation_report.json`: Code validation report
- `tools.py`: Main tools collection file
- `tools/`: Tool implementation directory
  - `{tool_name}.py`: Specific implementation file for each tool
  - Examples: `create_table.py`, `delete_record.py`, `ls.py`, `mkdir.py`, etc.
  - Each tool file contains complete function implementations and documentation

## Recommended Execution Order

If you're using the WorldInteract framework for the first time, we recommend running the examples in the following order:

1. **Scenario Collection Example** - First process raw API data
2. **Domain Graph Example** - Create domain graphs based on 
3. **Environment Creation Example** - Create complete environment from 