# Task Generation Examples

This directory contains examples for the complete task trajectory generation pipeline.

## Overview

The task generation pipeline transforms generated environments into agent tasks through four main steps:

1. **Task Graph Building** - Build dependency graphs from function parameters
2. **Subtask Graph Sampling** - Sample diverse subgraphs using multiple strategies
3. **Random Walk Generation** - Generate function call sequences (Chain and DAG)
4. **Agent Task Generation** - Create complete tasks with LLM assistance

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

Generate agent tasks in one command:

```bash
python build_task_example.py \
    --env-dirs ../data/generated_env/domains/file_operations \
    --domain-graph ../data/domain_graphs/my_domain_graphs \
    --output ../data/agent_tasks/file_operations_tasks
```

This will:
- Build task graph with parameter dependencies
- Sample 10 diverse subgraphs
- Generate 2 random walks per subgraph
- Create agent tasks with LLM filtering and generation

### Option 2: Run Step-by-Step

Execute each step individually:

#### Step 1: Build Task Graph

```bash
python create_task_graph_example.py \
    --env-dirs ../data/generated_env/domains/file_operations \
    --domain-graph ../data/domain_graphs/my_domain_graphs \
    --output ../data/task_graphs/file_operations_task_graph
```

**Output**: Task dependency graph, embeddings, visualization

#### Step 2: Sample Subtask Graphs

```bash
python sample_subtask_graph_example.py \
    --task-graph ../data/task_graphs/file_operations_task_graph/task_graph.json \
    --output ../data/subtask_graphs/file_operations_subtask_graphs \
    --num-samples 10
```

**Output**: Multiple subgraph JSON files

#### Step 3: Generate Random Walks

```bash
python random_walk_example.py \
    --subtask-graphs ../data/subtask_graphs/file_operations_subtask_graphs \
    --output ../data/random_walks/file_operations_random_walks \
    --num-walks 2
```

**Output**: Chain and DAG walk sequences

#### Step 4: Generate Tasks

Use the complete pipeline script or integrate into your own code (see documentation).

## Examples

### Example 1: Single Domain

```bash
python build_task_example.py \
    --env-dirs ../data/generated_env/domains/file_operations \
    --domain-graph ../data/domain_graphs/my_domain_graphs \
    --output ../data/agent_tasks/file_operations_tasks
```

### Example 2: Multiple Domains

```bash
python build_task_example.py \
    --env-dirs ../data/generated_env/domains/file_operations \
               ../data/generated_env/domains/database_operations \
    --domain-graph ../data/domain_graphs/my_domain_graphs \
    --output ../data/agent_tasks/multi_domain_tasks
```

### Example 3: Custom Configuration

```bash
python build_task_example.py \
    --env-dirs ../data/generated_env/domains/file_operations \
    --domain-graph ../data/domain_graphs/my_domain_graphs \
    --output ../data/agent_tasks/custom_tasks \
    --num-subgraphs 20 \
    --num-walks 3 \
    --skip-filter  # Skip LLM filtering for faster processing
```

## Output Structure

After running the complete pipeline:

```
data/agent_tasks/file_operations_tasks/
├── task_graph/
│   ├── task_graph.json              # Dependency graph
│   ├── embeddings.json              # Parameter embeddings
│   └── task_graph_visualization.png # Graph visualization
├── subtask_graphs/
│   └── *.json                       # Sampled subgraphs
├── random_walks/
│   └── *.json                       # Generated walks
└── tasks/
    ├── tasks_summary.json           # Summary statistics
    └── task_*.json                  # Agent tasks
```

## Configuration

All parameters can be configured in `config/environment_config.yaml` under the `task_generation` section:

```yaml
task_generation:
  parameter_similarity_threshold: 0.7  # For edge creation
  min_subgraph_nodes: 5                # Subgraph size range
  max_subgraph_nodes: 20
  num_subgraphs_per_graph: 10          # Sampling count
  min_walk_length: 5                   # Walk length range
  max_walk_length: 10
  num_walks_per_subgraph: 2            # Walks per subgraph
  walk_types: ["dag", "chain"]         # Walk types
  enable_llm_filter: true              # LLM filtering
  enable_llm_task_generation: true     # LLM task generation
```

## Sampling Strategies

The subgraph sampler supports multiple strategies:

- **Random**: Random node selection
- **BFS**: Breadth-first traversal
- **DFS**: Depth-first traversal
- **Community**: Community detection
- **Star**: Hub-and-spoke structure
- **Chain**: Linear path
- **Tree**: Tree structure

Adjust strategy weights in the configuration file.

## Walk Types

### Chain Walk
Linear sequence representing serial execution:
```
A → B → C → D → E
```

### DAG Walk
Directed acyclic graph supporting parallel execution:
```
    A
   / \
  B   C  (parallel)
   \ /
    D
    |
    E
```

## Troubleshooting

### Issue: No edges in task graph

**Solution**: Lower `parameter_similarity_threshold` in config (try 0.6 or 0.5)

### Issue: Insufficient subgraphs sampled

**Solution**: 
- Lower `min_subgraph_nodes`
- Lower `subgraph_diversity_threshold`
- Increase max attempts (modify code)

### Issue: Short random walks

**Solution**:
- Ensure subgraphs are large enough
- Check graph connectivity
- Verify sufficient outgoing edges

### Issue: Poor task quality

**Solution**:
- Adjust `min_task_complexity` and `max_task_complexity`
- Increase `num_walks_per_subgraph`
- Tune LLM temperature in model config

## Performance Tips

1. **Parallel Processing**: For large-scale generation, consider parallelizing subgraph sampling and walk generation
2. **Caching**: Reuse embeddings from `embeddings.json` for repeated runs
3. **Batch LLM Calls**: Increase `llm_filter_batch_size` to reduce API calls
4. **Skip Filtering**: Use `--skip-filter` for faster testing (trades quality for speed)

## API Usage

You can also use the modules programmatically:

```python
from worldInteract.core.build_task_graph import (
    TaskGraphBuilder,
    SubtaskGraphSampler,
    RandomWalker
)
from worldInteract.trajectories import TaskGenerator

# Build task graph
builder = TaskGraphBuilder()
result = builder.build_task_graph(
    generated_env_dirs=["path/to/env"],
    domain_graph_dir="path/to/domain_graph",
    output_dir="output/task_graph"
)

# Sample subgraphs
sampler = SubtaskGraphSampler()
subgraphs = sampler.sample_subgraphs(task_graph, num_samples=10)

# Generate walks
walker = RandomWalker()
walks = []
for subgraph in subgraphs:
    walks.extend(walker.generate_walks(subgraph.graph))

# Generate tasks
task_generator = TaskGenerator()
tasks = task_generator.generate_tasks(walks, tool_definitions)
```

## Documentation

For detailed documentation, see:
- [Task Generation Guide](../docs/TASK_GENERATION.md)
- [Configuration Reference](../config/environment_config.yaml)
- [Main README](../README.md)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the detailed documentation
3. Examine log output for specific error messages
4. Verify configuration parameters

## Next Steps

After generating tasks:
1. Review task quality and diversity
2. Use tasks for agent training or evaluation
3. Analyze task complexity distribution
4. Iterate on configuration parameters for optimal results

