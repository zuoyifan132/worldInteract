#!/usr/bin/env python3
"""
示例：演示Scenario Collection和Dependency Graph建模的完整流程
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from worldInteract.core.scenario_collection import APICleaner
from worldInteract.core.dependency_graph import DependencyGraphBuilder


def main():
    """运行示例流程"""
    
    logger.info("开始运行Scenario Pipeline示例")
    
    # 设置路径
    project_root = Path(__file__).parent.parent
    raw_apis_path = project_root / "data" / "raw_apis"
    
    # 输出目录
    output_dir = project_root / "output" / "example_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_apis_path = output_dir / "cleaned_apis.json"
    dependency_graphs_dir = output_dir / "dependency_graphs"
    
    # 检查输入目录
    if not raw_apis_path.exists():
        logger.error(f"Raw APIs目录不存在: {raw_apis_path}")
        return
    
    try:
        # 阶段1: Scenario Collection
        logger.info("=== 开始Scenario Collection阶段 ===")
        cleaner = APICleaner()
        cleaning_result = cleaner.clean_apis(str(raw_apis_path), str(processed_apis_path))
        
        logger.info("Scenario Collection完成，统计信息:")
        stats = cleaning_result.get("metadata", {}).get("processing_stats", {})
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # 阶段2: Tool Dependency Graph Modeling  
        logger.info("=== 开始Tool Dependency Graph Modeling阶段 ===")
        builder = DependencyGraphBuilder()
        graph_result = builder.build_dependency_graph(
            str(processed_apis_path), 
            str(dependency_graphs_dir)
        )
        
        logger.info("Dependency Graph Modeling完成，统计信息:")
        for phase, phase_stats in graph_result.items():
            if isinstance(phase_stats, dict):
                logger.info(f"  {phase}:")
                for key, value in phase_stats.items():
                    logger.info(f"    {key}: {value}")
        
        # 展示结果
        logger.info("\n=== 流程完成 ===")
        logger.info(f"清洗后的APIs: {processed_apis_path}")
        logger.info(f"依赖图数据: {dependency_graphs_dir}")
        logger.info(f"生成的域: {dependency_graphs_dir / 'domains'}")
        
        # 列出生成的域文件
        domains_dir = dependency_graphs_dir / "domains"
        if domains_dir.exists():
            domain_files = list(domains_dir.glob("*.json"))
            logger.info(f"生成了 {len(domain_files)} 个域:")
            for domain_file in domain_files:
                logger.info(f"  - {domain_file.name}")
        
        logger.info("\n示例运行成功完成！")
        
    except Exception as e:
        logger.error(f"示例运行失败: {e}")
        raise


if __name__ == "__main__":
    main()
