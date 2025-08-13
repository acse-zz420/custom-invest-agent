import re, ast
from typing import List

from llama_index.core import PropertyGraphIndex, Settings
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.retrievers import (
    CustomPGRetriever,
    VectorContextRetriever,
    TextToCypherRetriever,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, TextNode

from llm import VolcengineLLM
from rag_milvus.config import get_embedding_model
from config import *
from tool import timer
from prompt import CYPHER_PROMPT


class HybridGraphRetriever(CustomPGRetriever):
    """
    一个混合图检索器，融合了向量检索和文本到Cypher查询。
    """

    def init(
            self,
            index: PropertyGraphIndex,
            similarity_top_k: int = 4,
            path_depth: int = 2,
            community_expansion: bool = False,
            **kwargs,
    ):
        """
        初始化两个子检索器：向量检索器和Cypher检索器。

        Args:
            index (PropertyGraphIndex): 已加载的图谱索引。
            similarity_top_k (int): 向量检索返回的最相似节点数。
            path_depth (int): 从向量检索到的节点出发，在图中探索的深度。
            community_expansion: 使用Leiden算法
        """
        self._community_expansion = community_expansion
        # 初始化向量上下文检索器
        self._vector_retriever = VectorContextRetriever(
            self.graph_store,
            vector_store=index.vector_store,
            include_text=True,
            similarity_top_k=similarity_top_k,
            path_depth=path_depth,
        )

        # 初始化文本到Cypher检索器
        self._cypher_retriever = TextToCypherRetriever(
            self.graph_store,
            llm=Settings.llm,
            text_to_cypher_template= CYPHER_PROMPT
        )

    @timer
    def _expand_with_community(self, initial_nodes: List[NodeWithScore]) -> dict:
        """
            基于向量检索到的文本块，直接在图中查询这些文本块提及的实体，
            再根据这些实体的社区ID，扩展到整个社区所有实体关联的文本块。
            """
        if not initial_nodes:
            return {}

        # 1. 从初始检索到的文本块节点中，获取它们的ID
        initial_chunk_ids = [node.node.node_id for node in initial_nodes]

        # 2. 直接在图中查询这些Chunk ID提及的所有实体ID
        mentioned_entity_ids = set()
        with self.graph_store._driver.session(database=NEO4J_DATABASE) as session:
            # 这个查询是新的、更健壮的核心
            cypher_get_entities = """
                UNWIND $chunk_ids AS chunk_id
                MATCH (c:Chunk {id: chunk_id})-[:MENTIONS]->(e:entity)
                RETURN DISTINCT e.id AS entityId
                """
            results = session.run(cypher_get_entities, chunk_ids=initial_chunk_ids)
            for record in results:
                mentioned_entity_ids.add(record["entityId"])

        if not mentioned_entity_ids:
            print("    - 初始文本块中未找到提及的实体，无法进行社区扩展。")
            return {}

        print(f"    - 从入口节点中提取到 {len(mentioned_entity_ids)} 个实体ID。")

        # 2. 查询这些实体的社区ID
        community_ids_to_expand = set()
        with self.graph_store._driver.session(database=NEO4J_DATABASE) as session:
            cypher_get_ids = """
            UNWIND $entity_ids AS entity_id
            MATCH (n:entity {id: entity_id})
            WHERE n.leidenCommunityId IS NOT NULL
            RETURN DISTINCT n.leidenCommunityId AS communityId
            """
            results = session.run(cypher_get_ids, entity_ids=list(mentioned_entity_ids))
            for record in results:
                community_ids_to_expand.add(record["communityId"])

        if not community_ids_to_expand:
            print("    - 提及的实体均未分配社区ID，跳过扩展。")
            return {}

        print(f"    - 发现相关社区ID: {list(community_ids_to_expand)}。正在扩展上下文...")

        # 3. 获取这些社区的所有实体关联的文本块 (Chunk)
        community_results = {}  # 创建一个字典来存储结果
        with self.graph_store._driver.session(database=NEO4J_DATABASE) as session:
            for comm_id in community_ids_to_expand:
                cypher_get_community_chunks = """
                MATCH (n:entity {leidenCommunityId: $comm_id})
                MATCH (source_node:Chunk)-[:MENTIONS]->(n)
                RETURN DISTINCT source_node
                """
                results = session.run(cypher_get_community_chunks, comm_id=comm_id)

                community_nodes = []
                for record in results:
                    node_data = record["source_node"]
                    node_properties = dict(node_data)
                    node_id = node_properties.get("id", node_data.element_id)
                    node_text = node_properties.get("text", "")

                    # 手动创建一个 LlamaIndex 的 TextNode 对象
                    node_obj = TextNode(
                        id_=node_id,
                        text=node_text,
                        metadata=node_properties,
                    )
                    community_nodes.append(NodeWithScore(node=node_obj, score=10.0))

                if community_nodes:
                    community_results[comm_id] = community_nodes

        return community_results

    @timer
    def custom_retrieve(self, query_str: str) -> List[NodeWithScore]:
        """
        执行自定义的检索逻辑：
        1. 分别用向量和Cypher进行检索。
        2. 如果开启，进行社区上下文扩展。
        3. 合并并去重结果。
        """
        print("--- 开始执行混合检索 ---")

        # 向量检索
        print("  - 步骤1: 执行向量检索...")
        vector_nodes = self._vector_retriever.retrieve(query_str)
        for node in vector_nodes:
            node.node.metadata["retrieval_source"] = "Vector Search"
        print(f"  - 向量检索找到 {len(vector_nodes)} 个节点。")

        # Cypher检索
        print("  - 步骤2: 执行文本到Cypher检索...")
        cypher_nodes = []  # 默认cypher_nodes为空列表
        try:
            retrieved_cypher_nodes = self._cypher_retriever.retrieve(query_str)

            # 检查返回的是否是代表空查询的节点
            if retrieved_cypher_nodes and 'Generated Cypher query:\n\n\nCypher Response:\n[]' in retrieved_cypher_nodes[
                0].text:
                print("  - Cypher检索返回空查询，跳过。")
            else:
                cypher_nodes = retrieved_cypher_nodes
                for node in cypher_nodes:
                    node.node.metadata["retrieval_source"] = "Text-to-Cypher"
        except Exception as e:
            # 捕获其他可能的异常
            print(f"  - Cypher检索时发生未知错误: {e}")

        print(f"  - Cypher检索找到 {len(cypher_nodes)} 个节点。")


        # --- 社区扩展逻辑 ---
        community_nodes_map = {}
        if self._community_expansion and vector_nodes:
            print("  - 步骤3: (已开启) 执行社区上下文扩展...")
            # community_nodes_map 是一个字典 {comm_id: [node1, node2], ...}
            community_nodes_map = self._expand_with_community(vector_nodes)

            # 为节点打上标签，并把社区ID存入元数据
            for comm_id, nodes in community_nodes_map.items():
                for node_with_score in nodes:
                    node_with_score.node.metadata["retrieval_source"] = "Community Expansion"
                    # --- 在这里，我们将确切的社区ID存入元数据 ---
                    node_with_score.node.metadata["community_id"] = comm_id

            # 打印统计信息
            total_community_nodes = sum(len(nodes) for nodes in community_nodes_map.values())
            print(f"  - 社区扩展找到 {len(community_nodes_map)} 个社区，共 {total_community_nodes} 个额外节点。")

        # ------------------------

        # 合并与去重
        print("  - 步骤4: 合并与去重...")
        combined_nodes = {}
        # 社区扩展的结果优先级最高，其次是向量和Cypher
        all_community_nodes = [node for nodes in community_nodes_map.values() for node in nodes]
        all_retrieved_nodes = all_community_nodes + vector_nodes + cypher_nodes
        for node in all_retrieved_nodes:
            if node.node.node_id not in combined_nodes:
                combined_nodes[node.node.node_id] = node

        final_nodes = list(combined_nodes.values())
        print(f"--- 混合检索完成，合并去重后共 {len(final_nodes)} 个节点。 ---\n")

        return final_nodes

    @timer
    def print_categorized_source_nodes(self, response):
        """
        分类别地打印源节点，展示其来源、文本内容和实体关系。
        """
        if not response.source_nodes:
            print("[上下文] 未找到源节点。")
            return

        print("\n" + "=" * 20 + " [上下文详细分析] " + "=" * 20)

        # 按来源对节点进行分组
        categorized_nodes = {
            "Vector Search": [],
            "Text-to-Cypher": [],
            "Community Expansion": [],
            "Unknown": [],
        }
        community_groups = {}
        for node in response.source_nodes:
            source = node.node.metadata.get("retrieval_source", "Unknown")
            if source == "Community Expansion":
                # 假设所有社区扩展出的节点都有leidenCommunityId
                community_id = node.node.metadata.get("leidenCommunityId", "unknown_community")
                if community_id not in community_groups:
                    community_groups[community_id] = []
                community_groups[community_id].append(node)
            else:
                categorized_nodes[source].append(node)

        # 将社区分组重新放回主分类，以便统一遍历
        if community_groups:
            categorized_nodes["Community Expansion"] = list(community_groups.values())

        for source, nodes_or_groups in categorized_nodes.items():
            if not nodes_or_groups:
                continue

            print(f"\n--- 来源: {source} ({len(nodes_or_groups)}个节点/社区) ---")

            # --- 新增分支：专门处理社区扩展的逻辑 ---
            if source == "Community Expansion":
                for i, group in enumerate(nodes_or_groups):

                    community_id = group[0].node.metadata.get("community_id", "N/A")

                    if community_id == "N/A":
                        print("    -> 无法确定社区ID，跳过查询。")
                        continue

                    print(f"  社区 {i + 1} (ID: {community_id}, 包含 {len(group)} 个节点):")

                    try:
                        with self.graph_store._driver.session(database=NEO4J_DATABASE) as session:
                            # 1. 查询并打印整个社区的关系网络
                            print("    -> [社区关系网络]:")
                            cypher_relations = """
                                       MATCH (n:entity)-[r]-(m:entity)
                                       WHERE n.leidenCommunityId = $comm_id AND m.leidenCommunityId = $comm_id
                                       RETURN DISTINCT n.name AS source, type(r) AS relation, m.name AS target
                                       LIMIT 50
                                       """
                            rel_results = session.run(cypher_relations, comm_id=community_id)
                            found_rels = 0
                            for record in rel_results:
                                print(f"        - ({record['source']})-[{record['relation']}]->({record['target']})")
                                found_rels += 1
                            if found_rels == 0:
                                print("        - 未找到社区内的关系。")

                            # 2. 查询并打印社区所有节点对应的所有元Chunk文本
                            print("    -> [社区元 Chunk 文本]:")
                            chunk_ids = {node.node.node_id for node in group}
                            cypher_chunks = """
                                UNWIND $id_list AS chunk_id
                                MATCH (c:Chunk {id: chunk_id})
                                RETURN c.id, c.text AS original_text
                                """
                            chunk_results = session.run(cypher_chunks, id_list=list(chunk_ids))
                            found_chunks = 0
                            for record in chunk_results:
                                original_text = record["original_text"].strip().replace('\n', ' ')
                                print(f"        - [ID: {record['c.id']}] {original_text}")
                                found_chunks += 1
                            if found_chunks == 0:
                                print("        - 未找到社区关联的Chunk文本。")

                    except Exception as e:
                        print(f"        - 查询社区信息时出错: {e}")
            else:
                for i, node in enumerate(nodes_or_groups):
                    print(f"  源 {i + 1} (ID: {node.node.node_id}, 分数: {node.score:.4f}):")

                    # 分支1: 处理 Text-to-Cypher 的虚拟节点
                    if source == "Text-to-Cypher":
                        cypher_content = node.text.strip()
                        print(f"    -> [整理的实体关系内容]: {cypher_content}")

                        # 步骤1: 从文本中稳健地提取出响应列表字符串
                        response_str = ""
                        try:
                            response_part = cypher_content.split("Cypher Response:", 1)[1].strip()
                            match = re.search(r'\[.*\]', response_part, re.DOTALL)
                            if match:
                                response_str = match.group(0)
                            else:
                                print("        - 未能在响应中找到有效的列表格式 `[...]`。")
                                continue
                        except IndexError:
                            print("        - 未能从节点文本中解析出'Cypher Response'部分。")
                            continue

                        # 步骤2: 使用 ast.literal_eval 安全解析字符串，并提取ID
                        unique_chunk_ids = set()
                        try:
                            response_data = ast.literal_eval(response_str)

                            if not isinstance(response_data, list):
                                print("        - 解析出的数据不是一个列表。")
                                continue

                            for item in response_data:
                                if isinstance(item, dict):
                                    # 遍历字典中的所有值，查找包含'triplet_source_id'的键
                                    for key, value in item.items():
                                        if 'triplet_source_id' in key and value:
                                            unique_chunk_ids.add(value)
                        except (ValueError, SyntaxError) as e:
                            # ast.literal_eval 在格式错误时会抛出这些异常
                            print(f"        - Cypher响应内容不是有效的Python字面量格式: {e}")
                            continue

                        if not unique_chunk_ids:
                            print("        - Cypher响应中未找到任何'triplet_source_id'。")
                            continue

                        # 步骤3: 批量查询
                        print("    -> [元 Chunk 文本 (来自Cypher响应)]: ")
                        try:
                            with self.graph_store._driver.session(database=NEO4J_DATABASE) as session:
                                cypher_query = """
                                UNWIND $id_list AS chunk_id
                                MATCH (c:Chunk {id: chunk_id})
                                RETURN c.id, c.text AS original_text
                                """
                                results = session.run(cypher_query, id_list=list(unique_chunk_ids))

                                found_texts = 0
                                for record in results:
                                    original_text = record["original_text"].strip().replace('\n', ' ')
                                    print(f"        - [ID: {record['c.id']}] {original_text}")
                                    found_texts += 1

                                if found_texts == 0:
                                    print("        - 数据库中未找到任何对应ID的Chunk。")
                        except Exception as e:
                            print(f"        - 批量查询元文本时出错: {e}")

                    # 分支2: 处理返回真实Chunk节点的其他来源 (Community Expansion)
                    else:
                        for i, node in enumerate(nodes_or_groups):
                            print(f"  源 {i + 1} (ID: {node.node.node_id}, 分数: {node.score:.4f}):")

                            # 分支1: 处理 Text-to-Cypher 的虚拟节点
                            if source == "Text-to-Cypher":
                                # ... (您原来的Text-to-Cypher处理逻辑，完全不变) ...
                                # ... 为了简洁，这里省略，请将您原来的代码粘贴回来 ...
                                cypher_content = node.text.strip()
                                print(f"    -> [整理的实体关系内容]: {cypher_content}")
                            # 分支2: 处理返回真实Chunk节点的来源 (Vector Search)
                            else:
                                retriever_content = node.text.strip().replace('\n', ' ')
                                print(f"    -> [整理的实体关系内容]: {retriever_content}")
                                print("    -> [元 Chunk 文本]:")

                                cypher_query = """
                                                MATCH (c:Chunk {id: $chunk_id})
                                                RETURN c.text AS original_text
                                                """
                                try:
                                    with self.graph_store._driver.session(
                                            database=NEO4J_DATABASE) as session:
                                        record = session.run(cypher_query, chunk_id=node.node.node_id).single()

                                        if record and record["original_text"]:
                                            original_text = record["original_text"].strip().replace('\n', ' ')
                                            print(f"        - {original_text}")
                                        else:
                                            print("        - 未能在数据库中找到对应的元文本。")
                                except Exception as e:

                                    print(f"        - 查询元文本时出错: {e}")
@timer
def load_existing_graph_index():
    """从现有的 Neo4j 数据库中加载 PropertyGraphIndex。"""
    print("正在配置 LLM 和嵌入模型...")
    Settings.llm = VolcengineLLM(api_key=API_KEY)
    _, _, Settings.embed_model = get_embedding_model()

    print("正在连接到 Neo4j 并加载现有图谱索引...")
    graph_store = Neo4jPropertyGraphStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URI,
        database=NEO4J_DATABASE
    )
    index = PropertyGraphIndex.from_existing(property_graph_store=graph_store)
    print("图谱索引加载成功！")
    return index




@timer
def run_queries(index: PropertyGraphIndex):
    """使用加载好的索引进行查询。"""
    if not index:
        print("索引对象无效，无法查询。")
        return

    # --- 查询模式: 自定义混合检索 vs. 基础向量检索 ---
    print("\n" + "=" * 60)
    print("--- 自定义混合检索 vs. 基础向量检索 ---")
    query = "券商担任了哪些角色"
    print(f"问题: {query}")

    # --- 1. 自定义混合检索查询引擎 ---
    print("\n 正在构建自定义混合检索查询引擎...")
    hybrid_retriever = HybridGraphRetriever(
        graph_store=index.property_graph_store,
        index=index,
        similarity_top_k=5,
        community_expansion = True,
    )
    custom_query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        llm=Settings.llm,
    )


    # 执行自定义查询
    print("\n" + "-" * 25 + " 执行自定义混合查询 " + "-" * 25)
    custom_response = custom_query_engine.query(query)
    print("\n[结果] 自定义混合查询的最终答案:")
    print(custom_response.response)

    hybrid_retriever.print_categorized_source_nodes(custom_response)



if __name__ == "__main__":
    graph_index = load_existing_graph_index()
    if graph_index:
        run_queries(graph_index)