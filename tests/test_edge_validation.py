#!/usr/bin/env python3
"""
测试边验证功能
演示如何在生成random walk时使用LLM验证边的匹配对
"""

import os
import sys
import json
import dotenv
import networkx as nx
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from worldInteract.core.build_task_graph import RandomWalker

# Load environment variables
dotenv.load_dotenv()


def load_subgraph_from_json(subgraph_file: str) -> nx.DiGraph:
    """加载子图从JSON文件"""
    with open(subgraph_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create directed graph
    graph = nx.DiGraph()
    
    # Add nodes with all their attributes
    for node in data.get('nodes', []):
        node_id = node['id']
        node_attrs = {k: v for k, v in node.items() if k != 'id'}
        graph.add_node(node_id, **node_attrs)
    
    # Add edges with their attributes
    for edge in data.get('edges', []):
        source = edge['source']
        target = edge['target']
        edge_attrs = {k: v for k, v in edge.items() if k not in ['source', 'target']}
        graph.add_edge(source, target, **edge_attrs)
    
    return graph


def main():
    """测试边验证功能"""
    
    logger.info("=" * 80)
    logger.info("测试边验证功能")
    logger.info("=" * 80)
    
    # 加载一个测试子图
    subgraph_file = project_root / "data/task_subgraphs/file_operations_task_subgraphs/1b308a57-3ff9-4758-86ac-54366d65764e.json"
    
    if not subgraph_file.exists():
        logger.error(f"子图文件不存在: {subgraph_file}")
        return
    
    logger.info(f"加载子图: {subgraph_file}")
    graph = load_subgraph_from_json(str(subgraph_file))
    logger.info(f"子图大小: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
    
    # 创建输出目录
    output_dir = project_root / "data/random_walks/test_edge_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化RandomWalker (边验证默认启用)
    logger.info("\n" + "=" * 80)
    logger.info("初始化RandomWalker (边验证已启用)")
    logger.info("=" * 80)
    walker = RandomWalker()
    
    # 生成walks (会自动验证边)
    logger.info("\n" + "=" * 80)
    logger.info("生成Random Walks (包含边验证)")
    logger.info("=" * 80)
    logger.info("这个过程会:")
    logger.info("1. 生成random walk")
    logger.info("2. 使用LLM验证每条边的matching_pairs")
    logger.info("3. 删除无效的边")
    logger.info("4. 清理图，保留最大连通分量")
    logger.info("5. 验证图是否满足最小节点数要求")
    
    walks = walker.generate_walks(
        subgraph=graph,
        num_walks=2,
        output_dir=str(output_dir)
    )
    
    # 显示结果
    logger.info("\n" + "=" * 80)
    logger.info("边验证测试完成")
    logger.info("=" * 80)
    logger.info(f"✅ 成功生成 {len(walks)} 个有效的walks")
    
    if walks:
        logger.info("\n生成的walks:")
        for i, walk in enumerate(walks, 1):
            logger.info(f"\nWalk {i}:")
            logger.info(f"  ID: {walk.id}")
            logger.info(f"  类型: {walk.walk_type.value}")
            logger.info(f"  长度: {walk.length}")
            logger.info(f"  序列: {' → '.join(walk.sequence[:5])}")
            if len(walk.sequence) > 5:
                logger.info(f"         ... (还有 {len(walk.sequence)-5} 个节点)")
    
    logger.info(f"\n输出目录: {output_dir}")
    logger.info("查看生成的JSON和PNG文件以了解详情")
    
    logger.info("\n" + "=" * 80)
    logger.info("测试说明:")
    logger.info("=" * 80)
    logger.info("- 边验证配置在 config/environment_config.yaml 中")
    logger.info("- enable_edge_validation: 是否启用边验证")
    logger.info("- edge_validation_model: 使用的LLM模型")
    logger.info("- min_matching_score: 最小置信度阈值")


if __name__ == "__main__":
    main()

