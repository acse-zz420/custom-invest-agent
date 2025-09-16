"""
知识图谱应用演示
"""

import asyncio
import logging

from neo4j_connector import Neo4jConnector
from ollama_graphiti_manager import OllamaGraphitiManager
from knowledge_graph_builder import KnowledgeGraphBuilder
from knowledge_graph_searcher import KnowledgeGraphSearcher
from graphiti_core.nodes import EpisodeType
from config import config

# ------------------------------------------------------------------ #
# 日志配置
# ------------------------------------------------------------------ #
try:
    from log_config import setup_comprehensive_logging

    setup_comprehensive_logging()
except ImportError:
    # 如果 log_config 不存在，使用简易默认配置
    logging.getLogger(
        "graphiti_core.utils.maintenance.edge_operations"
    ).setLevel(logging.ERROR)
    logging.getLogger("graphiti_core.driver.neo4j_driver").setLevel(logging.WARNING)
    logging.getLogger("graphiti_core").setLevel(print)
    logging.getLogger("httpx").setLevel(logging.WARNING)


class KnowledgeGraphApplication:
    """知识图谱应用主类"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.neo4j_connector: Neo4jConnector | None = None
        self.graphiti_manager: OllamaGraphitiManager | None = None
        self.graph_builder: KnowledgeGraphBuilder | None = None
        self.searcher: KnowledgeGraphSearcher | None = None

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    async def initialize(self) -> None:
        """初始化应用"""
        self.logger.info("初始化知识图谱应用...")
        try:
            # 子步骤 1: 初始化 Neo4j 连接
            self.logger.info("子步骤 1: 初始化 Neo4j 连接...")
            neo4j_cfg = config.get_neo4j_config()
            self.neo4j_connector = Neo4jConnector(
                uri=neo4j_cfg["uri"],
                user=neo4j_cfg["user"],
                password=neo4j_cfg["password"],
            )
            self.neo4j_connector.validate_connection()
            self.logger.info("Neo4j 连接器初始化完成")

            # 子步骤 2: 初始化 Ollama 和 Graphiti 管理器
            self.logger.info("子步骤 2: 初始化 Ollama 和 Graphiti 管理器...")
            self.graphiti_manager = OllamaGraphitiManager(self.neo4j_connector)
            await self.graphiti_manager.initialize_graphiti()
            await self.graphiti_manager.setup_database()
            self.logger.info("Ollama 和 Graphiti 管理器初始化完成")

            # 子步骤 3: 初始化知识图谱构建器
            self.logger.info("子步骤 3: 初始化知识图谱构建器...")
            self.graph_builder = KnowledgeGraphBuilder(self.graphiti_manager)
            self.logger.info("知识图谱构建器初始化完成")

            # 子步骤 4: 初始化搜索器
            self.logger.info("子步骤 4: 初始化搜索器...")
            self.searcher = KnowledgeGraphSearcher(self.graphiti_manager)
            self.logger.info("搜索器初始化完成")

            self.logger.info("应用初始化完成")

        except Exception as e:
            self.logger.error(f"应用初始化失败: {e}")
            raise

    # ------------------------------------------------------------------ #
    # 构建图
    # ------------------------------------------------------------------ #
    async def build_knowledge_graph(self) -> None:
        """构建知识图谱"""
        self.logger.info("开始构建知识图谱...")
        try:
            # 构建步骤 1: 添加示例情节
            self.logger.info("构建步骤 1: 添加示例情节...")
            sample_episodes = [
                self.graph_builder.create_text_episode(
                    "张三是一名软件工程师，他在北京的一家科技公司工作。他主要负责前端开发，"
                    "使用React和TypeScript技术栈。他毕业于清华大学计算机科学专业。",
                    "员工档案信息",
                ),
                self.graph_builder.create_text_episode(
                    "李四是张三的同事，她是一名后端工程师，专门负责数据库设计和API开发。"
                    "她使用Java和Spring Boot框架，有5年的工作经验。",
                    "员工档案信息",
                ),
                self.graph_builder.create_json_episode(
                    {
                        "name": "王五",
                        "position": "技术总监",
                        "company": "北京科技有限公司",
                        "department": "技术部",
                        "experience": "10年",
                        "skills": ["Java", "Python", "微服务架构", "团队管理"],
                    },
                    "员工详细信息",
                ),
                self.graph_builder.create_json_episode(
                    {
                        "name": "赵六",
                        "position": "产品经理",
                        "company": "北京科技有限公司",
                        "department": "产品部",
                        "projects": ["用户管理系统", "数据分析平台"],
                        "experience": "8年",
                    },
                    "员工详细信息",
                ),
            ]
            await self.graph_builder.add_episodes(sample_episodes)
            self.logger.info("构建步骤 1: 示例情节添加完成")
            self.logger.info("知识图谱构建完成")

        except Exception as e:
            self.logger.error(f"构建知识图谱时出错: {e}")
            raise

    # ------------------------------------------------------------------ #
    # 搜索演示
    # ------------------------------------------------------------------ #
    async def perform_searches(self) -> None:
        """执行搜索演示"""
        self.logger.info("开始执行搜索演示...")
        step = 0
        try:
            # 步骤 1: 基本搜索
            step += 1
            self.logger.info(f"搜索步骤 {step}: 执行基本搜索...")
            print(f"\n{'=' * 20} 搜索步骤 {step}: 基本搜索演示 {'=' * 20}")
            basic_results = await self.searcher.basic_search("谁是软件工程师？")
            self.searcher.print_search_results(basic_results, "基本搜索结果")
            self.logger.info(f"搜索步骤 {step}: 基本搜索完成")

            # 步骤 2: 节点搜索
            step += 1
            self.logger.info(f"搜索步骤 {step}: 执行节点搜索...")
            print(f"\n{'=' * 20} 搜索步骤 {step}: 节点搜索演示 {'=' * 20}")
            node_results = await self.searcher.node_search_with_recipe("技术总监")
            self.searcher.print_node_search_results(node_results, "节点搜索结果")
            self.logger.info(f"搜索步骤 {step}: 节点搜索完成")

            # 步骤 3: 中心节点搜索
            step += 1
            self.logger.info(f"搜索步骤 {step}: 执行中心节点搜索...")
            print(f"\n{'=' * 20} 搜索步骤 {step}: 中心节点搜索演示 {'=' * 20}")
            if basic_results:
                center_uuid = basic_results[0].source_node_uuid
                center_results = await self.searcher.center_node_search(
                    "谁是软件工程师？", center_uuid
                )
                self.searcher.print_search_results(center_results, "中心节点搜索结果")
                self.logger.info(f"搜索步骤 {step}: 中心节点搜索完成")
            else:
                self.logger.warning(f"搜索步骤 {step}: 跳过中心节点搜索（无基本搜索结果）")

            # 步骤 4: 综合搜索
            step += 1
            self.logger.info(f"搜索步骤 {step}: 执行综合搜索...")
            print(f"\n{'=' * 20} 搜索步骤 {step}: 综合搜索演示 {'=' * 20}")
            comp_results = await self.searcher.comprehensive_search("工程师")
            self.searcher.print_comprehensive_results(comp_results)
            self.logger.info(f"搜索步骤 {step}: 综合搜索完成")

            self.logger.info(f"所有 {step} 个搜索步骤都已完成")

        except Exception as e:
            self.logger.error(f"执行搜索时出错 (搜索步骤 {step}): {e}")
            raise

    # ------------------------------------------------------------------ #
    # 添加自定义情节
    # ------------------------------------------------------------------ #
    async def add_custom_episodes(self) -> None:
        """添加自定义情节"""
        self.logger.info("添加自定义情节...")
        step = 0
        try:
            # 步骤 1: 添加文本情节
            step += 1
            self.logger.info(f"情节步骤 {step}: 添加文本情节 (陈七信息)...")
            await self.graph_builder.add_single_episode(
                content="陈七是一名数据科学家，专门负责机器学习和数据分析。他使用Python和TensorFlow进行模型开发。",
                episode_type=EpisodeType.text,
                description="员工档案信息",
                name="陈七信息",
            )
            self.logger.info(f"情节步骤 {step}: 文本情节添加完成")

            # 步骤 2: 添加 JSON 情节
            step += 1
            self.logger.info(f"情节步骤 {step}: 添加 JSON 情节 (孙八详细信息)...")
            custom_employee = {
                "name": "孙八",
                "position": "UI设计师",
                "company": "北京科技有限公司",
                "department": "设计部",
                "experience": "6年",
                "skills": [
                    "Figma",
                    "Sketch",
                    "Adobe Creative Suite",
                    "用户研究",
                ],
            }
            await self.graph_builder.add_json_episode(
                json_data=custom_employee,
                description="员工详细信息",
                name="孙八详细信息",
            )
            self.logger.info(f"情节步骤 {step}: JSON 情节添加完成")
            self.logger.info(f"所有 {step} 个自定义情节都添加完成")

        except Exception as e:
            self.logger.error(f"添加自定义情节时出错 (情节步骤 {step}): {e}")
            raise

    # ------------------------------------------------------------------ #
    # 资源清理
    # ------------------------------------------------------------------ #
    async def cleanup(self) -> None:
        """清理资源"""
        self.logger.info("开始清理资源...")
        try:
            # 关闭 Graphiti 连接
            if self.graphiti_manager:
                await self.graphiti_manager.close_connection()
                self.logger.info("Graphiti 连接已关闭")
            else:
                self.logger.info("跳过 Graphiti 连接关闭（管理器未初始化）")
            self.logger.info("资源清理完成")
        except Exception as e:
            self.logger.error(f"清理资源时出错: {e}")

    # ------------------------------------------------------------------ #
    # 演示入口
    # ------------------------------------------------------------------ #
    async def run_demo(self) -> None:
        """运行完整演示"""
        step = 0
        try:
            # 步骤 1: 初始化
            step += 1
            print(f"\n{'=' * 20} 步骤 {step}: 初始化应用 {'=' * 20}")
            await self.initialize()
            print(f"步骤 {step}: 初始化完成")

            # 步骤 2: 构建知识图谱
            step += 1
            print(f"\n{'=' * 20} 步骤 {step}: 构建知识图谱 {'=' * 20}")
            await self.build_knowledge_graph()
            print(f"步骤 {step}: 知识图谱构建完成")

            # 步骤 3: 添加自定义情节
            step += 1
            print(f"\n{'=' * 20} 步骤 {step}: 添加自定义情节 {'=' * 20}")
            await self.add_custom_episodes()
            print(f"步骤 {step}: 自定义情节添加完成")

            # 步骤 4: 执行搜索演示
            step += 1
            print(f"\n{'=' * 20} 步骤 {step}: 执行搜索演示 {'=' * 20}")
            await self.perform_searches()
            print(f"步骤 {step}: 搜索演示完成")

            print(f"\n所有 {step} 个步骤都已完成！")

        except Exception as e:
            self.logger.error(f"演示过程中出错 (步骤 {step}): {e}")
            print(f"\n演示失败 (步骤 {step}): {e}")
            raise

        finally:
            # 步骤 5: 清理资源
            step += 1
            print(f"\n{'=' * 20} 步骤 {step}: 清理资源 {'=' * 20}")
            await self.cleanup()
            print(f"步骤 {step}: 资源清理完成")


# ------------------------------------------------------------------ #
# 主入口
# ------------------------------------------------------------------ #
async def main() -> None:
    print("=" * 60)
    print("知识图谱应用演示")
    print("=" * 60)

    app = KnowledgeGraphApplication()
    try:
        await app.run_demo()
        print("\n" + "=" * 60)
        print("演示成功完成！")
        print("=" * 60)
    except Exception as e:
        print(f"\n演示失败: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())