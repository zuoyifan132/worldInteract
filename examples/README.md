# WorldInteract Examples

This directory contains various usage examples for the WorldInteract framework. All examples support command line arguments and you can view detailed usage instructions with `--help` or `-h`.

## Module-Example Mapping Table

| Example Script | Core Module | Sub-Module | Description | Output |
|----------------|-------------|------------|-------------|--------|
| `scenario_collection_example.py` | **Input Processing** | API Cleaning & Standardization | Process raw API data and create cleaned, standardized scenarios | Cleaned API JSON files with metadata |
| `domain_graph_example.py` | **Input Processing** | Domain Graph Building | Create tool relationship graphs and domain clustering based on embeddings | Domain graph, communities, embeddings, visualizations |
| `create_environment_example.py` | **Environment Construction** | Schema Generator → State Generator → Code Generator → CodeAgent | Generate complete environments with schema, initial state, initial code and validated tools | Schema, initial state, tools, validation reports |
| `create_task_graph_example.py` | **Task Graph Construction** | Task Graph (Dependency Graph) | Build task dependency graphs from generated environments based on parameter similarity | Task graph JSON, embeddings, visualization |
| `sample_task_subgraph_example.py` | **Task Graph Construction** | Task Subgraph Sampling | Sample diverse subgraphs using multiple strategies (Random, BFS, DFS, Community, Star, Chain, Tree) | Sampled subgraph JSON files |
| `random_walk_example.py` | **Task Graph Construction** | Random Walk Generation (Chain & DAG) | Generate Chain and DAG execution sequences from task subgraphs | Chain walk JSON files, DAG walk JSON files |
| `build_task_example.py` | **Complete Pipeline** | Task Graph → Subgraph Sampling → Random Walk → Task Generation | End-to-end pipeline from environment to agent tasks | Task graphs, subgraphs, walks, agent tasks |

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

### 4. Task Graph Creation Example (create_task_graph_example.py)

Demonstrates how to build task dependency graphs from generated environments.

**Basic Usage:**
```bash
# Build task graph from environment
python create_task_graph_example.py --env-dirs ../data/generated_env/domains/file_operations --domain-graph ../data/domain_graphs/my_domain_graphs --output ../data/task_graphs/file_operations_task_graph
```

**Parameters:**
- `--env-dirs`: One or more generated environment directories (required)
- `--domain-graph`: Domain graph directory path (required)
- `--output`: Output directory for task graph (required)

**Output File Structure:**
After running this example, the following files will be generated:
- `task_graph.json`: Task dependency graph with parameter relationships
- `embeddings.json`: Parameter embeddings for similarity computation
- `task_graph_visualization.png`: Graph visualization image

### 5. Task Subgraph Sampling Example (sample_task_subgraph_example.py)

Demonstrates how to sample diverse subgraphs from a task graph using multiple strategies.

**Basic Usage:**
```bash
# Sample subgraphs from task graph
python sample_task_subgraph_example.py --task-graph ../data/task_graphs/file_operations_task_graph/task_graph.json --output ../data/task_subgraphs/file_operations_task_subgraphs --num-samples 10
```

**Parameters:**
- `--task-graph`: Task graph JSON file path (required)
- `--output`: Output directory for sampled subgraphs (required)
- `--num-samples`: Number of subgraphs to sample (default: 10)

**Output File Structure:**
After running this example, multiple subgraph JSON files will be generated:
- `subgraph_0.json`: Sampled subgraph with nodes and edges
- `subgraph_1.json`: Another sampled subgraph
- `...`: Additional subgraphs based on num-samples

**Sampling Strategies:**
- Random: Random node selection
- BFS: Breadth-first traversal
- DFS: Depth-first traversal
- Community: Community detection
- Star: Hub-and-spoke structure
- Chain: Linear path
- Tree: Tree structure

### 6. Random Walk Generation Example (random_walk_example.py)

Demonstrates how to generate random walks (Chain and DAG) from task subgraphs.

**Basic Usage:**
```bash
# Generate random walks from task subgraphs
python random_walk_example.py --task-subgraphs ../data/task_subgraphs/file_operations_task_subgraphs --output ../data/random_walks/file_operations_random_walks --num-walks 2
```

**Parameters:**
- `--task-subgraphs`: Directory containing task subgraph JSON files (required)
- `--output`: Output directory for random walks (required)
- `--num-walks`: Number of walks to generate per subgraph (default: 2)

**Output File Structure:**
After running this example, random walk JSON files will be generated:
- `walk_chain_0.json`: Chain walk (linear sequence)
- `walk_dag_0.json`: DAG walk (directed acyclic graph)
- `...`: Additional walks based on num-walks

**Walk Types:**
- **Chain Walk**: Linear sequence representing serial execution (A → B → C → D → E)
- **DAG Walk**: Directed acyclic graph supporting parallel execution

### 7. Complete Task Generation Pipeline (build_task_example.py)

Demonstrates the complete pipeline from environment to agent tasks.

**Basic Usage:**
```bash
# Run complete pipeline
python build_task_example.py --env-dirs ../data/generated_env/domains/file_operations --domain-graph ../data/domain_graphs/my_domain_graphs --output ../data/agent_tasks/file_operations_tasks
```

**Parameters:**
- `--env-dirs`: One or more generated environment directories (required)
- `--domain-graph`: Domain graph directory path (required)
- `--output`: Output directory for generated tasks (required)
- `--num-subgraphs`: Number of subgraphs to sample (default: 10)
- `--num-walks`: Number of walks per subgraph (default: 2)
- `--skip-filter`: Skip LLM filtering for faster processing (optional)

**Output File Structure:**
After running this example, a complete task generation structure will be created:
- `task_graph/`: Task dependency graph and embeddings
  - `task_graph.json`: Dependency graph
  - `embeddings.json`: Parameter embeddings
  - `task_graph_visualization.png`: Graph visualization
- `task_subgraphs/`: Sampled subgraph JSON files
  - `subgraph_*.json`: Various sampled subgraphs
- `random_walks/`: Generated walk sequences
  - `walk_*.json`: Chain and DAG walks
- `tasks/`: Final agent tasks
  - `tasks_summary.json`: Summary statistics
  - `task_*.json`: Individual agent tasks with descriptions and steps

**Pipeline Steps:**
1. Build task graph with parameter dependencies
2. Sample diverse subgraphs using multiple strategies
3. Generate random walks (Chain and DAG)
4. Create agent tasks with LLM assistance

## Recommended Execution Order

If you're using the WorldInteract framework for the first time, we recommend running the examples in the following order:

1. **Scenario Collection Example** - First process raw API data
2. **Domain Graph Example** - Create domain graphs based on cleaned APIs
3. **Environment Creation Example** - Create complete environment from domain APIs
4. **Task Graph Creation Example** - Build task dependency graphs from environments
5. **Task Subgraph Sampling Example** - Sample diverse subgraphs for task generation
6. **Random Walk Generation Example** - Generate function call sequences
7. **Complete Task Generation Pipeline** - Generate agent tasks (or run this directly to execute steps 4-7)

## Configuration

All parameters can be configured in `config/environment_config.yaml`. Key sections include:
- `scenario_collection`: API cleaning and normalization settings
- `domain_graph`: Domain clustering and graph building settings
- `environment_generation`: Schema and tool generation settings
- `task_generation`: Task graph building and walk generation settings

## Additional Resources

For detailed documentation, see:
- [Main README](../README.md)
- [Task Generation Guide](../docs/TASK_GENERATION.md)
- [Scenario Pipeline Guide](../docs/SCENARIO_PIPELINE.md)
- [Configuration Reference](../config/environment_config.yaml)
