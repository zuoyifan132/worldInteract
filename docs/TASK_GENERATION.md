# Task Trajectory Generation

This document describes how to use worldInteract's task trajectory generation functionality to build task graphs from generated environments and generate agent tasks.

## Overview

The task trajectory generation pipeline consists of four main steps:

```
Generated Environments
    ↓
1. Task Graph Building
    ↓
2. Task Subgraph Sampling
    ↓
3. Random Walk Generation
    ↓
4. Agent Task Generation
    ↓
Agent Tasks
```

## Prerequisites

Before using task trajectory generation, you need to have:

1. Completed API cleaning and classification (Scenario Collection)
2. Completed Domain Graph modeling
3. Completed function environment code generation (Environment Generation)

## Configuration

Task generation configuration is located in the `task_generation` section of `config/environment_config.yaml`:

```yaml
task_generation:
  # Task Graph Building
  parameter_similarity_threshold: 0.7  # Parameter similarity threshold
  
  # Subgraph Sampling
  min_subgraph_nodes: 5                # Minimum nodes in subgraph
  max_subgraph_nodes: 20               # Maximum nodes in subgraph
  num_subgraphs_per_graph: 10          # Number of subgraphs to sample per task graph
  subgraph_diversity_threshold: 0.3    # Subgraph diversity threshold
  sampling_strategies:                 # Subgraph sampling strategies
    random: 0.15
    bfs: 0.2
    dfs: 0.2
    community: 0.15
    star: 0.1
    chain: 0.1
    tree: 0.1
  
  # Random Walk
  min_walk_length: 5                   # Minimum walk length
  max_walk_length: 10                  # Maximum walk length
  num_walks_per_subgraph: 2            # Number of walks per subgraph
  walk_types: ["dag", "chain"]         # Walk types
  dag_branch_probability: 0.3          # DAG branch probability
  max_parallel_branches: 3             # Maximum parallel branches
  
  # Task Generation
  enable_llm_filter: true              # Enable LLM filtering
  enable_llm_task_generation: true     # Enable LLM task generation
  min_task_complexity: 3               # Minimum task complexity
  max_task_complexity: 10              # Maximum task complexity
  llm_filter_batch_size: 5             # LLM filter batch size
```

## Usage

### Method 1: Complete Pipeline (Recommended)

Use `build_task_example.py` to complete all steps at once:

```bash
python examples/build_task_example.py \
    --env-dirs data/generated_env/domains/file_operations \
    --domain-graph data/domain_graphs/my_domain_graphs \
    --output data/agent_tasks/file_operations_tasks
```

Multiple domains:

```bash
python examples/build_task_example.py \
    --env-dirs data/generated_env/domains/file_operations \
               data/generated_env/domains/database_operations \
    --domain-graph data/domain_graphs/my_domain_graphs \
    --output data/agent_tasks/multi_domain_tasks \
    --num-subgraphs 10 \
    --num-walks 2
```

### Method 2: Step-by-Step Execution

#### Step 1: Build Task Graph

```bash
python examples/create_task_graph_example.py \
    --env-dirs data/generated_env/domains/file_operations \
    --domain-graph data/domain_graphs/my_domain_graphs \
    --output data/task_graphs/file_operations_task_graph
```

**Output**:
- `task_graph.json`: Task dependency graph
- `embeddings.json`: Parameter embeddings
- `task_graph_visualization.png`: Graph visualization

#### Step 2: Sample Subgraphs

```bash
python examples/sample_task_subgraph_example.py \
    --task-graph data/task_graphs/file_operations_task_graph/task_graph.json \
    --output data/task_subgraphs/file_operations_task_subgraphs \
    --num-samples 10
```

**Output**:
- Multiple subgraph JSON files (each with a UUID)

#### Step 3: Generate Random Walks

```bash
python examples/random_walk_example.py \
    --task-subgraphs data/task_subgraphs/file_operations_task_subgraphs \
    --output data/random_walks/file_operations_random_walks \
    --num-walks 2
```

**Output**:
- Multiple random walk JSON files
- Contains both chain and DAG walk types

#### Step 4: Generate Agent Tasks

This step needs to be done in Python code as it requires loading previous results:

```python
from worldInteract.trajectories import TaskGenerator
from worldInteract.core.build_task_graph import RandomWalker

# Load random walks
walker = RandomWalker()
walks = walker.load_all_walks("data/random_walks/file_operations_random_walks")

# Load tool definitions (from task_graph.json)
import json
with open("data/task_graphs/file_operations_task_graph/task_graph.json") as f:
    graph_data = json.load(f)

tool_definitions = {}
for node in graph_data['nodes']:
    tool_definitions[node['id']] = {
        'description': node['description'],
        'parameters': node['parameters'],
        'returns': node['returns']
    }

# Generate tasks
task_generator = TaskGenerator()
tasks = task_generator.generate_tasks(
    walks=walks,
    tool_definitions=tool_definitions,
    output_dir="data/agent_tasks/file_operations_tasks"
)
```

## Core Concepts

### 1. Task Graph

- **Nodes**: Functions/tools
- **Edges**: Parameter dependencies (from output function → input function)
- **Weights**: Number of matching parameter pairs

Edge creation rule: If any output parameter of function A has semantic similarity exceeding the threshold (default 0.7) with any input parameter of function B, create a directed edge from A to B.

### 2. Task Subgraph

Small subgraphs sampled from the complete task graph, supporting multiple sampling strategies:

- **Random**: Random node selection
- **BFS**: Breadth-first search
- **DFS**: Depth-first search
- **Community**: Community detection
- **Star**: Star structure (hub node + spoke nodes)
- **Chain**: Chain structure (linear path)
- **Tree**: Tree structure

### 3. Random Walk

Generate function call sequences on subgraphs, supporting two types:

#### Chain Walk
Linear sequence, serial execution:
```
A → B → C → D → E
```

#### DAG Walk
Supports parallel branches, can fork and merge:
```
    A
   / \
  B   C  (parallel execution)
   \ /
    D
    |
    E
```

### 4. Agent Task

The final generated task contains:

```json
{
  "id": "task_xxx",
  "task_description": "Natural language task description",
  "function_sequence": ["func1", "func2", "func3"],
  "parameter_flow": [
    {
      "function": "func1",
      "input_parameters": {
        "param1": {"source": "user_input", "description": "..."}
      },
      "output_parameters": ["output1"]
    },
    {
      "function": "func2",
      "input_parameters": {
        "param1": {"source": "func1.output1", "description": "..."}
      },
      "output_parameters": ["result"]
    }
  ],
  "walk_type": "dag",
  "complexity_score": 0.7,
  "metadata": {
    "expected_output": "...",
    "walk_metadata": {...}
  }
}
```

## Output Directory Structure

After using the complete pipeline, the output directory structure looks like:

```
data/agent_tasks/file_operations_tasks/
├── task_graph/
│   ├── task_graph.json              # Task dependency graph
│   ├── embeddings.json              # Parameter embeddings
│   └── task_graph_visualization.png # Graph visualization
├── task_subgraphs/
│   ├── <uuid1>.json                 # Subgraph 1
│   ├── <uuid2>.json                 # Subgraph 2
│   └── ...
├── random_walks/
│   ├── <uuid1>.json                 # Walk 1
│   ├── <uuid2>.json                 # Walk 2
│   └── ...
└── tasks/
    ├── tasks_summary.json           # Task summary
    ├── task_<uuid1>.json            # Task 1
    ├── task_<uuid2>.json            # Task 2
    └── ...
```

## Advanced Usage

### Custom Sampling Strategy Weights

Adjust the weights in `sampling_strategies` in the config file:

```yaml
sampling_strategies:
  random: 0.1     # Reduce random sampling
  bfs: 0.3        # Increase BFS
  dfs: 0.3        # Increase DFS
  community: 0.2
  star: 0.05
  chain: 0.05
  tree: 0.0       # Disable tree sampling
```

### Generate Only Chain Walks

```yaml
walk_types: ["chain"]  # Use only chain walks
```

### Increase DAG Parallelism

```yaml
dag_branch_probability: 0.5    # Increase branch probability
max_parallel_branches: 5       # Increase max parallel branches
```

### Skip LLM Filtering

To speed up or save API calls:

```bash
python examples/build_task_example.py \
    --env-dirs data/generated_env/domains/file_operations \
    --domain-graph data/domain_graphs/my_domain_graphs \
    --output data/agent_tasks/file_operations_tasks \
    --skip-filter
```

## Debugging and Monitoring

### View Logs

All steps output detailed log information, including:
- Number of nodes and edges
- Distribution of sampling strategies
- Walk types and lengths
- Task generation statistics

### Visualize Task Graph

The generated `task_graph_visualization.png` helps understand function dependencies:
- Node colors represent different domains
- Edge thickness represents number of matching parameters

### Check Intermediate Results

You can check intermediate step outputs at any time:

```python
# Check subgraphs
from worldInteract.core.build_task_graph import TaskSubgraphSampler
sampler = TaskSubgraphSampler()
subgraphs = sampler.load_all_subgraphs("data/task_subgraphs/xxx")
for sg in subgraphs:
    print(f"Subgraph {sg.id}: {sg.nodes}")

# Check walks
from worldInteract.core.build_task_graph import RandomWalker
walker = RandomWalker()
walks = walker.load_all_walks("data/random_walks/xxx")
for walk in walks:
    print(f"Walk {walk.id} ({walk.walk_type.value}): {walk.sequence}")
    if walk.walk_type.value == 'dag':
        print(walker.visualize_dag_walk(walk))
```

## Common Issues

### Q: Task graph has few or no edges?

**A**: Possible causes:
1. Parameter similarity threshold is too high, try lowering `parameter_similarity_threshold`
2. Functions truly have no obvious dependencies
3. Parameter descriptions are not detailed enough, leading to low embedding similarity

### Q: Generated task quality is poor?

**A**: Try:
1. Adjust `min_task_complexity` and `max_task_complexity`
2. Increase `num_walks_per_subgraph` for more samples
3. Check LLM temperature parameter (in `model_config.yaml`)
4. Ensure tool definitions have clear descriptions

### Q: Subgraph sampling fails or insufficient samples?

**A**: Possible causes:
1. Task graph is too small, try lowering `min_subgraph_nodes`
2. Graph is not well connected, check task graph connectivity
3. Diversity threshold is too high, try lowering `subgraph_diversity_threshold`

### Q: Random walks are always short?

**A**: Check:
1. Whether subgraphs are large enough
2. Whether graph has sufficient out-degree (successor nodes)
3. Try increasing attempts (adjust `max_attempts` in code)

## Performance Optimization

### Parallel Processing

The current pipeline is serial. For large-scale data processing, consider:

1. Parallel subgraph sampling
2. Parallel random walk generation
3. Batch LLM API calls

### Cache Embeddings

Parameter embeddings are saved in `embeddings.json`. If running repeatedly, these embeddings can be reused.

### Reduce LLM Calls

- Increase `llm_filter_batch_size` to reduce API calls
- Use `--skip-filter` to skip filtering step (may affect quality)

## Next Steps

Generated tasks can be used for:

1. **Agent Training**: As training data for training agent models
2. **Agent Evaluation**: As test sets for evaluating agent performance
3. **Tool Chain Testing**: Verify effectiveness of tool combinations
4. **Task Planning Research**: Research task planning algorithms

## Related Documentation

- [Scenario Collection](../README.md#scenario-collection)
- [Domain Graph Building](../README.md#domain-graph-building)
- [Environment Generation](../README.md#environment-generation)
- [Configuration Guide](../config/environment_config.yaml)
