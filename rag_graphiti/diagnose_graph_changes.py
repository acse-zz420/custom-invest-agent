#!/usr/bin/env python3
"""
图数据库变化诊断脚本
用于分析添加数据前后图数据库的变化
"""
import asyncio
import logging
from typing import Optional, Dict, Any

from neo4j_connector import Neo4jConnector
from ollama_graphiti_manager import OllamaGraphitiManager
from knowledge_graph_builder import KnowledgeGraphBuilder
from knowledge_graph_searcher import KnowledgeGraphSearcher

# 设置日志
logging.basicConfig(level=print)
logger = logging.getLogger(__name__)


class GraphChangeDiagnostic:
    """图数据库变化诊断器"""

    def __init__(self):
        self.neo4j_connector: Optional[Neo4jConnector] = None
        self.graphiti_manager: Optional[OllamaGraphitiManager] = None
        self.graph_builder: Optional[KnowledgeGraphBuilder] = None
        self.searcher: Optional[KnowledgeGraphSearcher] = None
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    async def initialize(self) -> None:
        """初始化各组件"""
        self.logger.info("初始化诊断器...")
        try:
            # 初始化 Neo4j 连接
            self.neo4j_connector = Neo4jConnector()
            self.neo4j_connector.validate_connection()

            # 初始化 Ollama / Graphiti
            self.graphiti_manager = OllamaGraphitiManager(self.neo4j_connector)
            await self.graphiti_manager.initialize_graphiti()
            await self.graphiti_manager.setup_database()

            # 构建器与搜索器
            self.graph_builder = KnowledgeGraphBuilder(self.graphiti_manager)
            self.searcher = KnowledgeGraphSearcher(self.graphiti_manager)

            self.logger.info("诊断器初始化完成")
        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            raise

    # ------------------------------------------------------------------ #
    # 统计信息
    # ------------------------------------------------------------------ #
    def get_detailed_graph_stats(self) -> Optional[Dict[str, Any]]:
        """获取详细的图统计信息"""
        try:
            with self.neo4j_connector.driver.session() as session:
                # 基本统计
                node_count = session.run(
                    "MATCH (n) RETURN count(n) AS node_count"
                ).single()["node_count"]
                rel_count = session.run(
                    "MATCH ()-[r]->() RETURN count(r) AS rel_count"
                ).single()["rel_count"]

                # 节点标签分布
                label_stats = [
                    (rec["labels"], rec["count"])
                    for rec in session.run(
                        "MATCH (n) RETURN labels(n) AS labels, count(n) AS count"
                    )
                ]

                # 关系类型分布
                type_stats = [
                    (rec["type"], rec["count"])
                    for rec in session.run(
                        "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count"
                    )
                ]

                # 节点属性示例
                node_props = [
                    (rec["name"], rec["props"])
                    for rec in session.run(
                        "MATCH (n) "
                        "RETURN n.name AS name, properties(n) AS props "
                        "LIMIT 10"
                    )
                ]

                # 关系属性示例
                rel_props = [
                    (rec["type"], rec["props"])
                    for rec in session.run(
                        "MATCH ()-[r]->() "
                        "RETURN type(r) AS type, properties(r) AS props "
                        "LIMIT 10"
                    )
                ]

            return {
                "node_count": node_count,
                "rel_count": rel_count,
                "label_stats": label_stats,
                "type_stats": type_stats,
                "node_props": node_props,
                "rel_props": rel_props,
            }

        except Exception as e:
            self.logger.error(f"获取图统计时出错: {e}")
            return None

    # ------------------------------------------------------------------ #
    # 打印
    # ------------------------------------------------------------------ #
    @staticmethod
    def print_graph_stats(stats: Optional[Dict[str, Any]], title: str) -> None:
        """打印图统计信息"""
        print(f"\n{'=' * 60}")
        print(title)
        print(f"{'=' * 60}")

        if not stats:
            print("无法获取统计信息")
            return

        print(f"节点数量: {stats['node_count']}")
        print(f"关系数量: {stats['rel_count']}")

        print("\n节点标签统计:")
        for labels, count in stats["label_stats"]:
            print(f"  {labels}: {count}")

        print("\n关系类型统计:")
        for rel_type, count in stats["type_stats"]:
            print(f"  {rel_type}: {count}")

        print("\n节点属性示例 (前10个):")
        for name, props in stats["node_props"]:
            print(f"  {name}: {props}")

        print("\n关系属性示例 (前10个):")
        for rel_type, props in stats["rel_props"]:
            print(f"  {rel_type}: {props}")


# ------------------------------------------------------------------ #
# 测试 / CLI
# ------------------------------------------------------------------ #
if __name__ == "__main__":

    async def main():
        diag = GraphChangeDiagnostic()
        await diag.initialize()

        # 变化前
        before_stats = diag.get_detailed_graph_stats()
        diag.print_graph_stats(before_stats, "变化前统计")

        # TODO: 在此处插入对图的变更（写入 / 修改）逻辑
        # 例如 await diag.graph_builder.add_single_episode(...)

        # 变化后
        after_stats = diag.get_detailed_graph_stats()
        diag.print_graph_stats(after_stats, "变化后统计")

    asyncio.run(main())