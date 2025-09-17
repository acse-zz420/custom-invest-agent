import re, ast
from typing import List

from llama_index.core import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.retrievers import (
    CustomPGRetriever,
    VectorContextRetriever,
    TextToCypherRetriever,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, TextNode

from llm import VolcengineLLM
from config import *
from timer_tool import timer
from prompt import CYPHER_PROMPT
from opentelemetry.trace import Tracer
from llama_index.core.llms.llm import LLM
from rag_milvus.tracing import *
from opentelemetry.trace import Status, StatusCode

llm = VolcengineLLM(api_key=API_KEY)
_, _, embed_model = get_embedding_model()

class HybridGraphRetriever(CustomPGRetriever):
    """
    一个混合图检索器，融合了向量检索和文本到Cypher查询。
    """

    def init(
            self,
            index: PropertyGraphIndex,
            tracer: Tracer,
            llm: LLM,
            similarity_top_k: int = 4,
            path_depth: int = 1,
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
        self._tracer = tracer
        self._llm = llm
        self._community_expansion = community_expansion
        # 初始化向量上下文检索器
        self._vector_retriever = VectorContextRetriever(
            self.graph_store,
            vector_store=index.vector_store,
            embed_model=embed_model,
            include_text=True,
            similarity_top_k=similarity_top_k,
            path_depth=path_depth,
        )

        # 初始化文本到Cypher检索器
        self._cypher_retriever = TextToCypherRetriever(
            self.graph_store,
            llm=llm,
            text_to_cypher_template=CYPHER_PROMPT
        )

    @timer
    def _expand_with_community(self, initial_nodes: List[NodeWithScore]) -> dict:
        """
        基于向量检索到的文本块，找到其中提及的实体，再根据这些实体的社区ID，
        扩展到整个社区所有实体关联的文本块。
        返回一个包含入口实体、触发的社区ID和社区节点的字典。
        """
        with self._tracer.start_as_current_span("_expand_with_community") as span:
            if not initial_nodes:
                return {}

            initial_chunk_ids = [node.node.node_id for node in initial_nodes]

            # 1. 从入口Chunk中提取提及的实体ID
            mentioned_entity_ids = set()
            with self.graph_store._driver.session(database=AURA_DATABASE) as session:
                cypher_get_entities = """
                UNWIND $chunk_ids AS chunk_id
                MATCH (c:Chunk {id: chunk_id})-[:MENTIONS]->(e:__Entity__)
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
            with self.graph_store._driver.session(database=AURA_DATABASE) as session:
                cypher_get_ids = """
                UNWIND $entity_ids AS entity_id
                MATCH (n:__Entity__ {id: entity_id})
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
            community_nodes_map = {}
            with self.graph_store._driver.session(database=AURA_DATABASE) as session:
                for comm_id in community_ids_to_expand:
                    cypher_get_community_chunks = """
                    MATCH (source_node:Chunk)-[:MENTIONS]->(n:__Entity__)
                    WHERE n.leidenCommunityId = $comm_id AND NONE(id IN $initial_chunk_ids WHERE id = source_node.id)
                    RETURN DISTINCT source_node
                    """
                    results = session.run(cypher_get_community_chunks, comm_id=comm_id, initial_chunk_ids=initial_chunk_ids)
                    community_nodes = []
                    for record in results:
                        node_data = dict(record["source_node"])
                        node_obj = TextNode(
                            id_=node_data.get("id", node_data.get("element_id")),
                            text=node_data.get("text", ""),
                            metadata=node_data,
                        )
                        community_nodes.append(NodeWithScore(node=node_obj, score=10.0))
                    if community_nodes:
                        community_nodes_map[comm_id] = community_nodes

            # 4. 返回一个包含所有信息的字典
            span.set_attribute("input.initial_nodes_count", len(initial_nodes))
            span.set_attribute("output.mentioned_entity_ids_count", len(mentioned_entity_ids))
            span.set_attribute("output.community_ids_count", len(community_ids_to_expand))

            total_community_nodes = sum(len(nodes) for nodes in community_nodes_map.values())
            span.set_attribute("output.total_community_nodes_found", total_community_nodes)
            return {
                "entry_entity_ids": mentioned_entity_ids,
                "triggered_community_ids": community_ids_to_expand,
                "community_nodes_map": community_nodes_map
            }

    @timer
    def custom_retrieve(self, query_str: str) -> List[NodeWithScore]:
        """
        执行自定义的检索逻辑：
        1. 分别用向量和Cypher进行检索。
        2. 如果开启，进行社区上下文扩展。
        3. 合并并去重结果。
        """
        current_span = trace.get_current_span()
        current_span.set_attribute("input.query", query_str)
        print("--- 开始执行混合检索 ---")

        # 向量检索
        with self._tracer.start_as_current_span("Vector Retrieval") as vector_span:
            print("  - 步骤1: 执行向量检索...")
            vector_nodes = self._vector_retriever.retrieve(query_str)
            for node in vector_nodes:
                node.node.metadata["retrieval_source"] = "Vector Search (Entry)"
            print(f"  - 向量检索找到 {len(vector_nodes)} 个节点。")


        cypher_nodes = []  # 默认cypher_nodes为空列表
        with self._tracer.start_as_current_span("Text-to-Cypher Retrieval") as cypher_span:
            # Cypher检索
            print("  - 步骤2: 执行文本到Cypher检索...")
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
                        if retrieved_cypher_nodes:
                            generated_query = retrieved_cypher_nodes[0].text.split("Cypher Response:")[0].strip()
                            cypher_span.set_attribute("llm.generated_cypher", generated_query)


            except Exception as e:

                cypher_span.record_exception(e)

                cypher_span.set_status(Status(StatusCode.ERROR, str(e)))

                print(f"  - Cypher检索时发生未知错误: {e}")
        # --- 社区扩展逻辑 ---
        all_community_nodes = []
        if self._community_expansion and vector_nodes:
            with self._tracer.start_as_current_span("Community Expansion") as expansion_span:
                print("  - 步骤3: (已开启) 执行社区上下文扩展...")
                # expansion_results 是一个包含多个信息的字典
                expansion_results = self._expand_with_community(vector_nodes)

                if expansion_results:
                    community_nodes_map = expansion_results.get("community_nodes_map", {})

                    # 为节点打上标签，并把社区ID存入元数据
                    for comm_id, nodes in community_nodes_map.items():
                        for node_with_score in nodes:
                            node_with_score.node.metadata["retrieval_source"] = "Community Expansion"
                            node_with_score.node.metadata["community_id"] = comm_id

                    # 将所有社区节点收集到一个列表中，用于后续合并
                    all_community_nodes = [node for nodes in community_nodes_map.values() for node in nodes]

                    if all_community_nodes:
                        first_node_metadata = all_community_nodes[0].node.metadata
                        first_node_metadata["expansion_entry_entities"] = list(
                            expansion_results.get("entry_entity_ids", []))
                        first_node_metadata["expansion_triggered_communities"] = list(
                            expansion_results.get("triggered_community_ids", []))

                    # 打印统计信息
                    total_community_nodes = len(all_community_nodes)
                    print(f"  - 社区扩展找到 {len(community_nodes_map)} 个社区，共 {total_community_nodes} 个额外节点。")

            # ------------------------

        # 合并与去重
        with self._tracer.start_as_current_span("Merge and Purify") as merge_span:
            print("  - 步骤4: 合并与去重...")
            combined_nodes = {}
            all_retrieved_nodes = all_community_nodes + vector_nodes + cypher_nodes
            for node in all_retrieved_nodes:
                if node.node.node_id not in combined_nodes:
                    combined_nodes[node.node.node_id] = node

            final_nodes_raw = list(combined_nodes.values())
            final_nodes_purified = []
            for node_with_score in final_nodes_raw:
                original_node = node_with_score.node

                # 1. 明确地只提取文本内容
                text_content = original_node.get_content(metadata_mode="none")

                # 2. 检查文本内容有效性
                if not isinstance(text_content, str) or not text_content.strip():
                    print(f"  - [过滤] 丢弃了一个文本内容为空或无效的节点: ID={original_node.node_id}")
                    continue

                # 3. 移除 'embedding' 键
                clean_metadata = original_node.metadata.copy()  # 创建副本以避免修改原始对象
                if 'embedding' in clean_metadata:
                    del clean_metadata['embedding']
                    print(f"  - [净化] 从节点 {original_node.node_id} 的元数据中移除了 embedding。")

                # 4. 创建一个全新的纯净的 TextNode 对象
                purified_node = TextNode(
                    id_=original_node.node_id,
                    text=text_content,
                    metadata=clean_metadata,
                )

                # 5. 用纯净的节点和原始分数重新组装 NodeWithScore
                final_nodes_purified.append(NodeWithScore(
                    node=purified_node,
                    score=node_with_score.score
                ))
            merge_span.set_attribute("input.combined_nodes_count", len(final_nodes_raw))
            merge_span.set_attribute("output.purified_nodes_count", len(final_nodes_purified))
            print(f"--- 净化完成，节点数从 {len(final_nodes_raw)} 变为 {len(final_nodes_purified)} ---\n")

        current_span.set_attribute("output.final_nodes_count", len(final_nodes_purified))
        return final_nodes_purified

    @timer
    def print_categorized_source_nodes(self, response):
        """
        分类别地打印源节点，清晰地区分入口节点和社区扩展节点。
        """
        if not response.source_nodes:
            print("[上下文] 未找到源节点。")
            return

        print("\n" + "=" * 20 + " [上下文详细分析] " + "=" * 20)

        # --- 步骤1: 按来源对节点进行分组 ---
        entry_nodes = []
        cypher_nodes = []
        community_groups = {}  # {community_id: [nodes...]}

        for node in response.source_nodes:
            source = node.node.metadata.get("retrieval_source")
            if source == "Vector Search (Entry)":
                entry_nodes.append(node)
            elif source == "Text-to-Cypher":
                cypher_nodes.append(node)
            elif source == "Community Expansion":
                comm_id = node.node.metadata.get("community_id", "unknown_community")
                if comm_id not in community_groups:
                    community_groups[comm_id] = []
                community_groups[comm_id].append(node)

        # --- 步骤2: 打印入口节点 (Vector Search (Entry)) ---
        if entry_nodes:
            print(f"\n--- 来源: Vector Search (Entry) ({len(entry_nodes)}个节点) ---")
            # 1. 打印每个入口节点的元文本
            for i, node in enumerate(entry_nodes):
                print(f"\n  入口 {i + 1} (ID: {node.node.node_id}, 分数: {node.score:.4f}):")
                print("    -> [元 Chunk 文本]:")
                self._print_chunks_by_ids([node.node.node_id])

            # 2. 基于所有入口节点，查询并打印它们触发的社区关系网络
            print("\n  [由所有入口节点触发的社区关系网络]:")
            try:
                # 从入口节点中提取它们所属的社区ID
                entry_community_ids = set()
                with self.graph_store._driver.session(database=AURA_DATABASE) as session:
                    chunk_ids = [n.node.node_id for n in entry_nodes]
                    cypher_get_comm_ids = """
                    UNWIND $chunk_ids AS c_id
                    MATCH (:Chunk {id: c_id})-[:MENTIONS]->(e:__Entity__)
                    WHERE e.leidenCommunityId IS NOT NULL
                    RETURN DISTINCT e.leidenCommunityId AS commId
                    """
                    results = session.run(cypher_get_comm_ids, chunk_ids=chunk_ids)
                    for record in results:
                        entry_community_ids.add(record["commId"])

                if not entry_community_ids:
                    print("    - 入口节点未关联到任何社区。")
                else:
                    print(f"    - 入口节点关联到社区: {list(entry_community_ids)}")
                    with self.graph_store._driver.session(database=AURA_DATABASE) as session:
                        cypher_relations = """
                        UNWIND $comm_ids AS c_id
                        MATCH (n:__Entity__ {leidenCommunityId: c_id})-[r]-(m:__Entity__ {leidenCommunityId: c_id})
                        RETURN DISTINCT n.name AS source, type(r) AS relation, m.name AS target
                        LIMIT 50
                        """
                        rel_results = session.run(cypher_relations, comm_ids=list(entry_community_ids))
                        if not self._print_records(rel_results, "        - ({source})-[{relation}]->({target})"):
                            print("        - 未在这些社区内找到关系。")
            except Exception as e:
                print(f"        - 查询入口社区关系时出错: {e}")

        # --- 步骤3: 打印社区扩展的节点 (如果存在) ---
        if community_groups:
            print(f"\n--- 来源: Community Expansion (扩展节点) ({len(community_groups)}个社区) ---")
            for i, (community_id, group) in enumerate(community_groups.items()):
                # 过滤掉已经是入口的节点，只打印纯粹的扩展节点
                expansion_only_nodes = [n for n in group if
                                        n.node.node_id not in {en.node.node_id for en in entry_nodes}]
                if not expansion_only_nodes:
                    print(f"\n  社区 {i + 1} (ID: {community_id}): 所有节点均为入口节点，无纯扩展节点。")
                    continue

                print(f"\n  社区 {i + 1} (ID: {community_id}, 包含 {len(expansion_only_nodes)} 个纯扩展节点):")
                print("    -> [扩展出的元 Chunk 文本]:")
                chunk_ids = {node.node.node_id for node in expansion_only_nodes}
                self._print_chunks_by_ids(list(chunk_ids))

        # --- 步骤4: 打印 Text-to-Cypher 的结果 ---
        if cypher_nodes:
            print(f"\n--- 来源: Text-to-Cypher ({len(cypher_nodes)}个节点) ---")
            for i, node in enumerate(cypher_nodes):
                print(f"  源 {i + 1} (ID: {node.node.node_id}, 分数: {node.score:.4f}):")
                cypher_content = node.text.strip()
                print(f"    -> [整理的实体关系内容]: {cypher_content}")
                # ... (您原来的Cypher解析和打印逻辑) ...
                unique_chunk_ids = set()
                try:
                    response_part = cypher_content.split("Cypher Response:", 1)[1].strip()
                    match = re.search(r'\[.*\]', response_part, re.DOTALL)
                    if match:
                        response_str = match.group(0)
                        response_data = ast.literal_eval(response_str)
                        if isinstance(response_data, list):
                            for item in response_data:
                                if isinstance(item, dict):
                                    for key, value in item.items():
                                        if 'triplet_source_id' in key and value:
                                            unique_chunk_ids.add(value)
                except (IndexError, SyntaxError, ValueError) as e:
                    print(f"        - 解析Cypher响应出错: {e}")

                if not unique_chunk_ids:
                    print("    -> [元 Chunk 文本]: 未能从Cypher响应中提取到有效的源Chunk ID。")
                    continue

                print("    -> [元 Chunk 文本 (来自Cypher响应)]: ")
                self._print_chunks_by_ids(list(unique_chunk_ids))

    # 批量打印Chunk文本
    def _print_chunks_by_ids(self, id_list: list):
        if not id_list:
            return
        try:
            with self.graph_store._driver.session(database=AURA_DATABASE) as session:
                cypher_query = """
                UNWIND $id_list AS chunk_id_param
                MATCH (c:Chunk {id: chunk_id_param})
                RETURN c.id AS chunk_id, c.text AS original_text
                """
                results = session.run(cypher_query, id_list=id_list)
                format_string = "        - [ID: {chunk_id}] {original_text}"

                if not self._print_records(results, format_string):
                    print("        - 数据库中未找到任何对应ID的Chunk。")
        except Exception as e:
            print(f"        - 批量查询元文本时出错: {e}")

    # 辅助方法2: 通用打印查询结果
    def _print_records(self, records, format_str: str) -> bool:
        found_count = 0
        for record in records:
            # 使用 record.data() 将记录转换为字典，以便格式化字符串
            print(format_str.format(**record.data()))
            found_count += 1
        return found_count > 0


@timer
def load_existing_graph_index():
    """从现有的 Neo4j 数据库中加载 PropertyGraphIndex。"""
    print("正在配置 LLM 和嵌入模型...")


    print("正在连接到 Neo4j 并加载现有图谱索引...")
    graph_store = Neo4jPropertyGraphStore(
        username=AURA_DB_USER_NAME,  # 使用实际的用户名变量
        password=AURA_DB_PASSWORD,
        url=AURA_URI,
        database=AURA_DATABASE,
    )
    index = PropertyGraphIndex.from_existing(property_graph_store=graph_store, llm=llm, embed_model=embed_model)
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
    query = "分析一下国内半导体行业发展方向"
    print(f"问题: {query}")

    # --- 1. 自定义混合检索查询引擎 ---
    print("\n 正在构建自定义混合检索查询引擎...")
    hybrid_retriever = HybridGraphRetriever(
        graph_store=index.property_graph_store,
        tracer=tracer,
        llm=llm,
        index=index,
        similarity_top_k=1,
        community_expansion=True,
        # verbose=True
    )
    custom_query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        llm=llm,
    )


    # 执行自定义查询
    print("\n" + "-" * 25 + " 执行自定义混合查询 " + "-" * 25)
    custom_response = custom_query_engine.query(query)
    print(custom_response.response)

    hybrid_retriever.print_categorized_source_nodes(custom_response)


if __name__ == "__main__":
    graph_index = load_existing_graph_index()
    if graph_index:
        run_queries(graph_index)
    print('----------')
