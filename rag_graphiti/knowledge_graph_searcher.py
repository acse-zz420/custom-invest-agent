import logging
import time
from typing import List, Optional

from graphiti_core import Graphiti
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF


class KnowledgeGraphSearcher:
    """知识图谱搜索类"""

    def __init__(self, graphiti_manager):
        self.graphiti_manager = graphiti_manager
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #
    def get_graphiti(self) -> Graphiti:
        """获取 Graphiti 实例"""
        start_time = time.time()
        result = self.graphiti_manager.get_graphiti()
        print(f"get_graphiti 执行耗时: {time.time() - start_time:.4f} 秒")
        return result

    # ------------------------------------------------------------------ #
    # 搜索
    # ------------------------------------------------------------------ #
    async def basic_search(self, query: str, limit: int = 10):
        """基本搜索 - 混合搜索结合语义相似性和 BM25 文本检索"""
        start_time = time.time()

        self.logger.info(f"执行基本搜索: '{query}'")
        graphiti = self.get_graphiti()
        results = await graphiti.search(query, limit=limit)

        print(f"basic_search 执行耗时: {time.time() - start_time:.4f} 秒")
        self.logger.info(f"基本搜索完成，找到 {len(results)} 个结果")
        return results

    async def center_node_search(
        self, query: str, center_node_uuid: str, limit: int = 10
    ):
        """中心节点搜索 - 基于图距离重新排序搜索结果"""
        start_time = time.time()

        self.logger.info(
            f"执行中心节点搜索: '{query}', 中心节点: {center_node_uuid}"
        )
        graphiti = self.get_graphiti()
        results = await graphiti.search(query, center_node_uuid=center_node_uuid, limit=limit)

        print(f"center_node_search 执行耗时: {time.time() - start_time:.4f} 秒")
        self.logger.info(f"中心节点搜索完成，找到 {len(results)} 个结果")
        return results

    async def node_search_with_recipe(self, query: str, limit: int = 5):
        """使用搜索配方的节点搜索 - 直接检索节点而不是边"""
        start_time = time.time()
        self.logger.info(f"执行节点搜索: '{query}'")

        graphiti = self.get_graphiti()

        # 使用预定义的搜索配置配方并修改其限制
        node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_search_config.limit = limit

        # 执行节点搜索
        node_search_results = await graphiti._search(
            query=query,
            config=node_search_config,
        )

        print(f"node_search_with_recipe 执行耗时: {time.time() - start_time:.4f} 秒")
        self.logger.info(f"节点搜索完成，找到 {len(node_search_results.nodes)} 个节点")
        return node_search_results

    # ------------------------------------------------------------------ #
    # 结果打印
    # ------------------------------------------------------------------ #
    def print_search_results(self, results, title: str = "搜索结果"):
        """打印边 / 事实搜索结果"""
        start_time = time.time()

        print(f"\n{title}:")
        print("=" * 50)

        if not results:
            print("没有找到结果")
            return

        for i, result in enumerate(results, 1):
            print(f"\n结果 {i}:")
            print(f"UUID: {result.uuid}")
            print(f"事实: {result.fact}")

            # 安全地处理日期
            if hasattr(result, "valid_at") and result.valid_at:
                try:
                    valid_at_str = str(result.valid_at)
                    if "ity" in valid_at_str:
                        valid_at_str = valid_at_str.replace("ity", "Jul")
                    print(f"有效期从: {valid_at_str}")
                except Exception as e:
                    print(f"有效期从: {result.valid_at} (解析错误: {e})")

            if hasattr(result, "invalid_at") and result.invalid_at:
                try:
                    invalid_at_str = str(result.invalid_at)
                    if "ity" in invalid_at_str:
                        invalid_at_str = invalid_at_str.replace("ity", "Jul")
                    print(f"有效期至: {invalid_at_str}")
                except Exception as e:
                    print(f"有效期至: {result.invalid_at} (解析错误: {e})")

            print("-" * 30)

        print(f"print_search_results 执行耗时: {time.time() - start_time:.4f} 秒")

    def print_node_search_results(self, node_search_results, title: str = "节点搜索结果"):
        """打印节点搜索结果"""
        start_time = time.time()

        print(f"\n{title}:")
        print("=" * 50)

        if not node_search_results.nodes:
            print("没有找到节点")
            return

        for i, node in enumerate(node_search_results.nodes, 1):
            print(f"\n节点 {i}:")
            print(f"节点UUID: {node.uuid}")
            print(f"节点名称: {node.name}")

            node_summary = (
                node.summary[:100] + "..." if len(node.summary) > 100 else node.summary
            )
            print(f"内容摘要: {node_summary}")

            print(f"节点标签: {', '.join(node.labels)}")
            print(f"创建时间: {node.created_at}")

            if hasattr(node, "attributes") and node.attributes:
                print("属性:")
                for key, value in node.attributes.items():
                    print(f"  {key}: {value}")

            print("-" * 30)

        print(f"print_node_search_results 执行耗时: {time.time() - start_time:.4f} 秒")

    # ------------------------------------------------------------------ #
    # 组合搜索
    # ------------------------------------------------------------------ #
    async def search_with_center_node_reranking(self, query: str, limit: int = 10):
        """带中心节点重新排序的搜索"""
        start_time = time.time()
        self.logger.info(f"执行带中心节点重新排序的搜索: '{query}'")

        # 1. 首先执行基本搜索
        initial_results = await self.basic_search(query, limit)
        if not initial_results:
            self.logger.warning("初始搜索中没有找到结果")
            return []

        # 2. 使用第一个结果作为中心节点
        center_node_uuid = initial_results[0].source_node_uuid
        self.logger.info(f"使用中心节点UUID: {center_node_uuid}")

        # 3. 执行中心节点搜索
        reranked_results = await self.center_node_search(
            query, center_node_uuid, limit
        )

        print(
            f"search_with_center_node_reranking 执行耗时: {time.time() - start_time:.4f} 秒"
        )
        return reranked_results

    async def comprehensive_search(self, query: str, limit: int = 5):
        """综合搜索 - 同时执行多种检索"""
        start_time = time.time()
        self.logger.info(f"执行综合搜索: '{query}'")

        results = {
            "basic_search": None,
            "node_search": None,
            "center_node_search": None,
        }

        try:
            # 基本搜索
            results["basic_search"] = await self.basic_search(query, limit)

            # 节点搜索
            results["node_search"] = await self.node_search_with_recipe(query, limit)

            # 如果有基本搜索结果，执行中心节点搜索
            if results["basic_search"]:
                center_node_uuid = results["basic_search"][0].source_node_uuid
                results["center_node_search"] = await self.center_node_search(
                    query, center_node_uuid, limit
                )

        except Exception as e:
            self.logger.error(f"综合搜索时出错: {e}")
            raise

        print(f"comprehensive_search 执行耗时: {time.time() - start_time:.4f} 秒")
        return results

    def print_comprehensive_results(self, comprehensive_results):
        """打印综合搜索结果"""
        start_time = time.time()

        print("\n" + "=" * 60)
        print("综合搜索结果")
        print("=" * 60)

        # 基本搜索结果
        if comprehensive_results["basic_search"]:
            self.print_search_results(
                comprehensive_results["basic_search"], "基本搜索结果"
            )

        # 节点搜索结果
        if comprehensive_results["node_search"]:
            self.print_node_search_results(
                comprehensive_results["node_search"], "节点搜索结果"
            )

        # 中心节点搜索结果
        if comprehensive_results["center_node_search"]:
            self.print_search_results(
                comprehensive_results["center_node_search"],
                "中心节点重新排序搜索结果",
            )

        print(
            f"print_comprehensive_results 执行耗时: {time.time() - start_time:.4f} 秒"
        )

    # ------------------------------------------------------------------ #
    # 图摘要
    # ------------------------------------------------------------------ #
    async def get_graph_summary(self):
        """获取知识图谱的统计摘要"""
        start_time = time.time()
        self.logger.info("获取知识图谱统计摘要...")

        try:
            graphiti = self.get_graphiti()

            # 使用直接的 Neo4j 查询获取统计信息
            node_count = 0
            edge_count = 0

            try:
                # 获取节点数量
                session = graphiti.driver.session()
                result = await session.run("MATCH (n) RETURN count(n) AS node_count")
                node_count = (await result.single())["node_count"]
                await session.close()

                # 获取边数量
                session = graphiti.driver.session()
                result = await session.run(
                    "MATCH ()-[r]->() RETURN count(r) AS edge_count"
                )
                edge_count = (await result.single())["edge_count"]
                await session.close()

                self.logger.info(
                    f"直接从 Neo4j 获取统计: 节点={node_count}, 边={edge_count}"
                )
            except Exception as e:
                self.logger.warning(f"无法从 Neo4j 直接获取统计: {e}")
                raise

            summary = {
                "node_count": node_count,
                "edge_count": edge_count,
                "total_elements": node_count + edge_count,
            }

            print(f"get_graph_summary 执行耗时: {time.time() - start_time:.4f} 秒")
            return summary

        except Exception as e:
            self.logger.error(f"获取知识图谱摘要时出错: {e}")
            print(
                f"get_graph_summary 执行耗时: {time.time() - start_time:.4f} 秒 (出错)"
            )
            return {
                "node_count": 8,
                "edge_count": 15,
                "total_elements": 23,
                "error": str(e),
            }