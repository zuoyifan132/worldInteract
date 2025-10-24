#!/usr/bin/env python3
"""
测试 CAMEL 集成与 scenario_collection

这个脚本测试使用 ReactAgent 的新 camel_generator wrapper 与 APICleaner 的集成
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from worldInteract.core.scenario_collection import APICleaner

# 配置 logger
logger.remove()
logger.add(sys.stderr, level="INFO")


def create_test_api_data():
    """创建最小化的测试 API 数据"""
    return {
        "apis": [
            {
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {
                    "city": {
                        "type": "string",
                        "description": "The city name"
                    }
                }
            },
            {
                "name": "calculate",
                "description": "Calculate",
                "parameters": {
                    "expression": {
                        "type": "string"
                    }
                }
            }
        ]
    }


def test_camel_scenario_collection():
    """测试 CAMEL 与 APICleaner 的集成"""
    logger.info("=== 测试 CAMEL 与 Scenario Collection 的集成 ===")
    
    # 创建临时目录用于测试
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # 创建测试输入文件
        input_file = tmpdir_path / "test_apis.json"
        output_file = tmpdir_path / "cleaned_apis.json"
        
        test_data = create_test_api_data()
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
        
        logger.info(f"创建测试输入: {input_file}")
        logger.info(f"测试数据: {len(test_data['apis'])} 个 APIs")
        
        try:
            # 初始化 APICleaner (现在使用 CAMEL)
            logger.info("初始化 APICleaner，使用 CAMEL model manager...")
            cleaner = APICleaner()
            
            # 使用 CAMEL 测试 API 清理
            logger.info("使用 CAMEL 测试 API 清理...")
            result = cleaner.clean_apis(str(input_file), str(output_file))
            
            # 验证结果
            logger.info("=== 测试结果 ===")
            if "metadata" in result and "processing_stats" in result["metadata"]:
                stats = result["metadata"]["processing_stats"]
                logger.info("处理统计:")
                for key, value in stats.items():
                    logger.info(f"  {key}: {value}")
            
            # 检查输出文件
            if output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    output_data = json.load(f)
                
                logger.info(f"✅ 输出文件创建成功")
                logger.info(f"   总 APIs: {output_data['metadata']['total_apis']}")
                
                # 显示清理后的 API 样本
                if output_data['apis']:
                    sample_api = output_data['apis'][0]
                    logger.info(f"\n清理后的 API 样本:")
                    logger.info(f"  名称: {sample_api['name']}")
                    logger.info(f"  描述: {sample_api['description']}")
                    logger.info(f"  参数: {list(sample_api['parameters'].keys())}")
                
                logger.info("\n✅ CAMEL 集成测试通过!")
                return True
            else:
                logger.error("❌ 输出文件未创建")
                return False
                
        except Exception as e:
            logger.error(f"❌ 测试失败，错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


def main():
    """运行测试"""
    logger.info("启动 CAMEL Scenario Collection 测试")
    logger.info(f"项目根目录: {project_root}")
    
    success = test_camel_scenario_collection()
    
    if success:
        logger.info("\n🎉 所有测试通过!")
        sys.exit(0)
    else:
        logger.error("\n❌ 测试失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()

