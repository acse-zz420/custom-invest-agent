#!/usr/bin/env python3
"""
全面的日志配置解决方案
隐藏各种已知的警告和错误信息
"""
import logging


def setup_comprehensive_logging():
    """设置全面的日志配置"""

    # 设置 Graphiti 核心库的日志级别
    graphiti_modules = [
        'graphiti_core.utils.maintenance.edge_operations',
        'graphiti_core.driver.neo4j_driver',
        'graphiti_core',
        'graphiti_core.llm_client',
        'graphiti_core.embedder',
        'graphiti_core.cross_encoder',
    ]
    for module in graphiti_modules:
        logger = logging.getLogger(module)
        logger.setLevel(logging.WARNING)

    # 将特定模块提升到 ERROR 级别（隐藏 WARNING）
    error_modules = [
        'graphiti_core.utils.maintenance.edge_operations',
        'graphiti_core.driver.neo4j_driver',
    ]
    for module in error_modules:
        logger = logging.getLogger(module)
        logger.setLevel(logging.ERROR)

    # 减少 httpx 的 HTTP 请求日志
    logging.getLogger('httpx').setLevel(logging.WARNING)

    print("全面日志配置已设置")


if __name__ == "__main__":
    setup_comprehensive_logging()