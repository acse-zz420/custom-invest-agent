import json
from typing import Tuple, List
from pathlib import Path
import logging
from llama_index.core import (
    PropertyGraphIndex,
    Settings,
    Document,
)
from timer_tool import timer
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llm import VolcengineLLM
from prompt import EXTRACTOR_PROMPT, FINANCE_ENTITIES, FINANCE_RELATIONS, FINANCE_VALIDATION_SCHEMA
from config import get_embedding_model
from config import API_KEY, AURA_URI, AURA_DB_PASSWORD, AURA_DB_USER_NAME, AURA_DATABASE, MD_TEST_DIR
from hybrid_chunking import custom_chunk_pipeline

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@timer
def setup_llm_and_embed_model():
    """配置并返回 LLM 和嵌入模型"""
    logging.info("正在配置 LLM 和嵌入模型...")
    llm = VolcengineLLM(api_key=API_KEY)
    _, _, embed_model = get_embedding_model()

    # 使用 LlamaIndex 的全局 Settings 进行配置
    Settings.llm = llm
    Settings.embed_model = embed_model

    logging.info("LLM 和嵌入模型配置完成。")
    return llm, embed_model


def json_parser(response_str: str) -> List[Tuple[str, str, str]]:
    """
    解析器，用于解析 LLM 返回的 JSON 格式的三元组列表。
    """
    triples = []
    try:
        # LLM 可能返回被 markdown 代码块包围的 JSON
        if "```json" in response_str:
            response_str = response_str.split("```json")[1].split("```")

        data = json.loads(response_str)

        # 确保 data 是一个列表
        if not isinstance(data, list):
            return []

        for item in data:
            # 确保 item 是一个字典并包含所有需要的键
            if isinstance(item, dict) and 'subject' in item and 'predicate' in item and 'object' in item:
                # 去除可能存在的多余空格
                subject = str(item['subject']).strip()
                predicate = str(item['predicate']).strip()
                object_val = str(item['object']).strip()
                triples.append((subject, predicate, object_val))
    except json.JSONDecodeError:
        print(f"解析失败: LLM 返回的不是有效的 JSON。响应: {response_str}")
        return []
    except Exception as e:
        print(f"处理响应时发生未知错误: {e}。响应: {response_str}")
        return []

    return triples

@timer
def get_neo4j_graph_store():
    """初始化并验证 Neo4j 连接，返回 graph_store 实例"""
    logging.info(f"正在初始化 Neo4jPropertyGraphStore，数据库: {AURA_DATABASE}...")
    try:
        graph_store = Neo4jPropertyGraphStore(
            username=AURA_DB_USER_NAME,  # 使用实际的用户名变量
            password=AURA_DB_PASSWORD,
            url=AURA_URI,
            database=AURA_DATABASE,
        )

        logging.info("验证 Neo4j 连接和写入权限...")
        with graph_store._driver.session(database=AURA_DATABASE) as session:
            session.run("CREATE (n:TestNode {id: 'test_connection'})")
            result = session.run("MATCH (n:TestNode {id: 'test_connection'}) RETURN count(n) AS count")
            count = result.single()["count"]
            if count == 0:
                raise ConnectionError("无法写入 Neo4j 数据库，请检查权限或数据库配置。")
            session.run("MATCH (n:TestNode {id: 'test_connection'}) DELETE n")
        logging.info("Neo4j 连接和写入权限验证成功。")
        return graph_store
    except Exception as e:
        logging.error(f"Neo4j 连接或验证失败: {e}")
        raise

@timer
def load_and_chunk_documents(directory: str) -> list[Document]:
    """从指定目录加载 Markdown 文件并进行自定义分块"""
    logging.info(f"正在从 '{directory}' 加载并使用自定义逻辑分块...")
    all_nodes = []
    md_files = list(Path(directory).rglob("*.md"))

    if not md_files:
        logging.error(f"错误：在目录 '{directory}' 中未找到任何 .md 文件。")
        return []

    logging.info(f"发现 {len(md_files)} 个 Markdown 文件，开始逐一处理...")
    for file_path in md_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            chunks_with_metadata = custom_chunk_pipeline(content)
            logging.info(f"处理文件 {file_path.name}: 分块数 = {len(chunks_with_metadata)}")

            for i, (is_table, chunk_text) in enumerate(chunks_with_metadata):
                if is_table or not chunk_text.strip():
                    logging.info(f"跳过块 {i}（is_table={is_table}, 空文本={not chunk_text.strip()}）")
                    continue

                node = Document(
                    id_=f"{file_path.stem}_{i}",
                    text=chunk_text,
                    metadata={
                        "file_name": file_path.name,
                        "is_table": False,
                        "chunk_source": "custom_pipeline"
                    }
                )
                logging.debug(f"创建 Node: ID={node.id_}, Text='{node.text[:50]}...'")
                all_nodes.append(node)
        except Exception as e:
            logging.error(f"处理文件 {file_path} 时出错: {e}", exc_info=True)
            continue

    if not all_nodes:
        logging.warning("自定义分块流程未能生成任何有效的文本块。")

    logging.info(f"自定义分块完成，共生成 {len(all_nodes)} 个文本块 (nodes)。")
    return all_nodes


def clear_database(graph_store: Neo4jPropertyGraphStore):
    """清理 Neo4j 数据库中的所有节点和关系"""
    logging.info("正在清理 Neo4j 数据库...")
    try:
        with graph_store._driver.session(database=NEO4J_DATABASE) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logging.info("数据库清理完毕。")
    except Exception as e:
        logging.error(f"清理 Neo4j 数据库时出错: {e}", exc_info=True)


@timer
def verify_extraction_results(graph_store: Neo4jPropertyGraphStore):
    """查询并打印数据库中提取到的实体和关系"""
    logging.info("\n--- 调试信息：检查提取到的实体和关系 ---")
    try:
        # 查询实体
        entity_query = "MATCH (n:entity) RETURN n.id AS id, n.label AS type, n.description AS description LIMIT 25"
        with graph_store._driver.session(database=AURA_DATABASE) as session:
            entity_results = session.run(entity_query).data()

        logging.info(f"提取到 {len(entity_results)} 个实体：")
        for i, record in enumerate(entity_results[:5]):
            logging.info(
                f"  {i + 1}: Entity ID='{record['id']}', Type='{record.get('type', 'N/A')}', Description='{record.get('description', 'N/A')}'")

        # 查询关系
        relation_query = "MATCH (a)-[r]->(b) RETURN a.id AS head, type(r) AS relation, b.id AS tail LIMIT 25"
        with graph_store._driver.session(database=AURA_DATABASE) as session:
            relation_results = session.run(relation_query).data()

        logging.info(f"提取到 {len(relation_results)} 个关系：")
        for i, record in enumerate(relation_results[:5]):
            logging.info(f"  {i + 1}: ('{record['head']}')-[{record['relation']}]->('{record['tail']}')")

        if len(entity_results) == 0 or len(relation_results) == 0:
            logging.warning("警告：未能提取到任何实体或关系。请检查文档内容、分块大小或更换 LLM/Prompt。")

    except Exception as e:
        logging.error(f"查询实体或关系时发生错误: {e}", exc_info=True)

# Leiden算法
@timer
def run_leiden_community_detection(graph_store: Neo4jPropertyGraphStore):
    """在 Neo4j 中使用 GDS 运行 Leiden 算法"""
    logging.info("\n--- 开始运行 Leiden 社区发现算法 ---")

    # GDS 内存图的名称
    graph_name = "knowledge_graph_projection"

    # 写入社区ID的节点属性名
    write_property_name = "leidenCommunityId"

    try:
        with graph_store._driver.session(database=AURA_DATABASE) as session:
            logging.info("检查 GDS 插件并清理旧的图投影...")
            try:
                check_gds_query = "RETURN gds.graph.exists($graph_name) AS exists"
                result = session.run(check_gds_query, graph_name=graph_name)
                if result.single()['exists']:
                    session.run("CALL gds.graph.drop($graph_name)", graph_name=graph_name)
                    logging.info(f"已删除旧的图投影 '{graph_name}'.")
            except Exception:
                logging.error("GDS 插件似乎未安装或配置不正确。无法继续执行社区发现。")
                logging.error("请确保已在 Neo4j 中安装 Graph Data Science 插件。")
                return

            # 创建图的内存投影
            # 将所有的 'entity' 节点和之间的所有关系投影到内存中
            logging.info(f"正在创建 GDS 图投影 '{graph_name}'...")
            project_query = """
            CALL gds.graph.project(
            $graph_name,
            '*',
            { ALL: { type: '*', orientation: 'UNDIRECTED' } }
            )
            YIELD graphName, nodeCount, relationshipCount
            """
            result = session.run(project_query, graph_name=graph_name).data()
            logging.info(f"图投影创建成功: {result[0]}")

            # 运行 Leiden 算法并把结果写回数据库
            logging.info("正在运行 Leiden 算法...")
            leiden_query = """
            CALL gds.leiden.write(
              $graph_name,
              {
                minCommunitySize: 3,
                writeProperty: $write_property,
                gamma: 2.0
              }
            )
            YIELD communityCount, nodePropertiesWritten
            """
            result = session.run(leiden_query, graph_name=graph_name, write_property=write_property_name).data()
            logging.info(f"Leiden 算法执行完成: {result[0]}")

            # 清理 GDS 内存中的图投影
            logging.info(f"正在清理并删除 GDS 图投影 '{graph_name}'...")
            session.run("CALL gds.graph.drop($graph_name)", graph_name=graph_name)
            logging.info("GDS 资源清理完毕。")

    except Exception as e:
        logging.error(f"运行 Leiden 算法时出错: {e}", exc_info=True)


@timer
def build_property_graph():
    """主函数，执行完整的知识图谱构建流程
    Args:
        use_leiden (bool): 是否启用 Leiden 社区发现算法，默认为 False
    """
    graph_store = None
    try:
        llm, _ = setup_llm_and_embed_model()
        graph_store = get_neo4j_graph_store()

        clear_database(graph_store)

        nodes = load_and_chunk_documents(MD_TEST_DIR)
        if not nodes:
            logging.info("没有可处理的文档节点，脚本执行结束。")
            return

        kg_extractor = SchemaLLMPathExtractor(
            llm=llm,
            possible_entities=FINANCE_ENTITIES,
            # possible_relations=FINANCE_RELATIONS,
            # kg_validation_schema=FINANCE_VALIDATION_SCHEMA,
            strict=False,  # if false, will allow triplets outside of the schema
            extract_prompt=EXTRACTOR_PROMPT,
            num_workers=4,
            max_triplets_per_chunk=3,
        )

        logging.info("开始构建 PropertyGraphIndex...")
        index = PropertyGraphIndex(
            nodes=nodes,
            property_graph_store=graph_store,
            kg_extractors=[kg_extractor],
            show_progress=True,
        )

        # 5. 验证结果
        with graph_store._driver.session(database=AURA_DATABASE) as session:
            result = session.run("MATCH (n) RETURN count(n) AS node_count")
            node_count = result.single()["node_count"]
            logging.info(f"图谱构建完成，Neo4j 中总节点数: {node_count}")
            if node_count == 0:
                logging.error("未能将任何节点存入 Neo4j，请检查权限或配置问题。")

        # verify_extraction_results(graph_store)
        # if use_leiden:
        #     run_leiden_community_detection(graph_store)
        # else:
        #     logging.info("Leiden 社区发现算法未启用，跳过执行。")
    except Exception as e:
        logging.error(f"构建属性图谱过程中发生严重错误: {e}", exc_info=True)
    finally:
        # 确保 Neo4j 驱动程序被关闭
        if graph_store and graph_store._driver:
            graph_store._driver.close()
            logging.info("Neo4j driver closed.")


@timer
def build_graph_community(use_leiden: bool = True):
    graph_store = get_neo4j_graph_store()
    verify_extraction_results(graph_store)
    if use_leiden:
        run_leiden_community_detection(graph_store)
    else:
        logging.info("Leiden 社区发现算法未启用，跳过执行。")


if __name__ == "__main__":
    print("\n--- 开始运行知识图谱构建脚本 ---")
    build_property_graph()
    print("\n--- 脚本执行完毕 ---")
    print("开始划分社区")
    build_graph_community()
    print("\n--- 脚本执行完毕 ---")



