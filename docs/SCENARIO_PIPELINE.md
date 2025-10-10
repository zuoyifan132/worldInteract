# Scenario Pipeline: API集合与工具依赖图建模

本文档介绍WorldInteract框架中新增的Scenario Collection和Tool Dependency Graph Modeling功能。

## 概述

基于论文《Towards General Agentic Intelligence via Environment Scaling》的方法，我们实现了从原始API数据到结构化工具环境的完整流水线：

1. **Scenario Collection**: API清洗和标准化
2. **Tool Dependency Graph Modeling**: 工具依赖关系分析和领域聚类
3. **Function Schema Programmatic Materialization**: 与现有工具生成器集成

## 架构设计

```
Raw APIs (脏数据)
    ↓
📍 Scenario Collection
    ↓ 
Cleaned APIs (标准格式)
    ↓
📍 Tool Dependency Graph Modeling  
    ↓
Domain-grouped APIs + Dependency Graph
    ↓
📍 Function Schema Programmatic Materialization (现有)
    ↓
Executable Python Code + Database Schema
```

## 核心模块

### 1. Scenario Collection (`worldInteract.core.scenario_collection`)

#### 功能
- 清洗和标准化原始API描述
- 统一不同格式的API规范（OpenAI、Claude、MCP等）
- 检测和移除重复工具
- 使用LLM增强工具描述

#### 使用方法
```python
from worldInteract.core.scenario_collection import APICleaner

cleaner = APICleaner()
result = cleaner.clean_apis(
    raw_apis_path="data/raw_apis/dirty_apis_sample.json",
    output_path="data/processed_apis/cleaned_apis.json"
)
```

#### 标准化格式
所有API都会被转换为统一格式：
```json
{
  "name": "create_file",
  "description": "Create a new file with specified content",
  "parameters": {
    "file_path": {
      "type": "string",
      "description": "Path where the file should be created"
    },
    "content": {
      "type": "string", 
      "description": "Content to write to the file"
    }
  },
  "returns": {
    "type": "object",
    "properties": {
      "success": {"type": "boolean"},
      "file_id": {"type": "string"},
      "message": {"type": "string"}
    }
  }
}
```

### 2. Tool Dependency Graph Modeling (`worldInteract.core.dependency_graph`)

#### 功能
- 基于工具参数描述的向量化分析
- 计算工具间相似度并构建依赖图
- 使用Louvain算法进行社区检测
- LLM验证工具依赖关系（可选）
- 生成领域分组和描述

#### 使用方法
```python
from worldInteract.core.build_domain_graph import DomainGraphBuilder

builder = DomainGraphBuilder()
result = builder.build_dependency_graph(
    cleaned_apis_path="data/processed_apis/cleaned_apis.json",
    output_dir="data/dependency_graphs"
)
```

#### 算法流程
1. **向量化**: 使用OpenAI embeddings对工具参数描述进行向量化
2. **相似度计算**: 计算任意两个工具参数间的余弦相似度
3. **图构建**: 相似度超过阈值时在工具间创建边
4. **社区检测**: 使用Louvain算法识别工具社区
5. **LLM验证**: 使用大模型验证和优化工具分组
6. **领域生成**: 为每个社区生成领域名称和描述

### 3. Embedding Utils (`worldInteract.utils.embedding`)

#### 功能
- OpenAI embeddings API封装
- 批量文本向量化
- 余弦相似度计算
- 工具参数嵌入生成

## 配置说明

### Model Config (`config/model_config.yaml`)
```yaml
# Scenario Collection配置
scenario_collection:
  model: "claude_3d7"
  temperature: 0.3
  max_tokens: 8124
  retry_attempts: 3

# Dependency Graph配置
dependency_graph:
  model: "claude_3d7"
  temperature: 0.1
  max_tokens: 4096
  retry_attempts: 3

# Embedding配置
embedding:
  model: "text-embedding-3-large"
  api_base: "https://api.openai.com/v1"
  dimensions: 3072
  batch_size: 100
```

### Environment Config (`config/environment_config.yaml`)
```yaml
# Scenario Collection设置
scenario_collection:
  min_tool_name_length: 3
  max_tool_name_length: 50
  required_fields: ["name", "description", "parameters"]
  duplicate_threshold: 0.8
  description_min_length: 10
  
# Dependency Graph设置  
dependency_graph:
  similarity_threshold: 0.75  # 相似度阈值
  min_community_size: 2       # 最小社区大小
  max_community_size: 20      # 最大社区大小
  louvain_resolution: 1.0     # Louvain算法分辨率
  enable_llm_validation: true # 是否启用LLM验证
```

## 使用示例

### 完整流水线
```bash
# 运行完整流水线（使用目录）
python scripts/scenario_pipeline.py \
    data/raw_apis \
    -o output/my_run \
    -v

# 运行完整流水线（使用单个文件）
python scripts/scenario_pipeline.py \
    data/raw_apis/dirty_apis_sample.json \
    -o output/my_run \
    -v

# 只运行前两个阶段
python scripts/scenario_pipeline.py \
    data/raw_apis \
    -o output/my_run \
    --skip-tool-generation
```

### 编程接口
```python
# 运行示例
python examples/scenario_pipeline_example.py

# 或者在代码中使用
from scripts.scenario_pipeline import run_complete_pipeline

# 使用目录（推荐）
results = run_complete_pipeline(
    raw_apis_path="data/raw_apis",
    output_base_dir="output/my_experiment"
)

# 使用单个文件
results = run_complete_pipeline(
    raw_apis_path="data/raw_apis/dirty_apis_sample.json",
    output_base_dir="output/my_experiment"
)
```

## 输出文件结构

```
output/
└── scenario_pipeline/
    ├── processed_apis/
    │   └── cleaned_apis.json          # 清洗后的API
    ├── dependency_graphs/
    │   ├── dependency_graph.json      # 完整依赖图
    │   ├── communities.json           # 社区检测结果
    │   ├── domains.json              # 域分组汇总
    │   ├── embeddings.json           # 工具嵌入向量
    │   ├── graph_visualization.png    # 图可视化
    │   └── domains/                  # 各域详细信息
    │       ├── file_operations.json
    │       ├── user_management.json
    │       └── ...
    └── generated_domains/            # 生成的工具环境
        ├── file_operations/
        │   ├── schema.json
        │   ├── initial_state.json
        │   ├── tools.py
        │   └── tools/
        └── ...
```

## 性能优化

### 并行处理
- 批量API处理
- 向量化批处理
- 多进程社区检测

### 缓存机制
- 嵌入向量缓存
- LLM响应缓存
- 相似度计算缓存

### 内存管理
- 流式处理大型API集合
- 分块向量化
- 增量图构建

## 故障排除

### 常见问题

1. **OpenAI API密钥错误**
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

2. **依赖包缺失**
   ```bash
   pip install -r requirements.txt
   ```

3. **内存不足**
   - 调整批处理大小
   - 减少并行进程数
   - 使用流式处理

4. **相似度阈值调优**
   - 阈值过高：工具过度分散
   - 阈值过低：无关工具聚合
   - 建议范围：0.6-0.8

### 调试模式
```bash
python scripts/scenario_pipeline.py \
    data/raw_apis/dirty_apis_sample.json \
    -o output/debug \
    -v --skip-tool-generation
```

## 扩展开发

### 添加新的清洗规则
在`APICleaner`中扩展`_fix_api_format`方法

### 自定义相似度算法
在`DomainGraphBuilder`中覆盖`_build_similarity_graph`方法

### 新的社区检测算法
替换`community_louvain`为其他算法如Leiden

### 自定义领域命名
扩展`_generate_domain_name`和`_generate_domain_description`方法

## 相关论文

- Fang et al. "Towards General Agentic Intelligence via Environment Scaling" (2025)
- Community Detection in Networks (Newman, 2006)
- Louvain Method for Community Detection (Blondel et al., 2008)
