from typing import List

from llama_index.core import PropertyGraphIndex, Settings
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.retrievers import (
    CustomPGRetriever,
    VectorContextRetriever,
    TextToCypherRetriever,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore

from llm import VolcengineLLM
from rag_milvus.config import get_embedding_model
from config import *
from tool import timer


class HybridGraphRetriever(CustomPGRetriever):
    """
    一个混合图检索器，融合了向量检索和文本到Cypher查询。
    """

    def init(
            self,
            index: PropertyGraphIndex,
            similarity_top_k: int = 4,
            path_depth: int = 2,
            **kwargs,
    ):
        """
        初始化两个子检索器：向量检索器和Cypher检索器。

        Args:
            index (PropertyGraphIndex): 已加载的图谱索引。
            similarity_top_k (int): 向量检索返回的最相似节点数。
            path_depth (int): 从向量检索到的节点出发，在图中探索的深度。
        """
        # 调用父类的初始化方法，确保 self.graph_store 等属性被设置

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
        )

    def custom_retrieve(self, query_str: str) -> List[NodeWithScore]:
        """
        执行自定义的检索逻辑：
        1. 分别用向量和Cypher进行检索。
        2. 合并并去重结果。
        """
        print("--- 开始执行混合检索 ---")

        # 向量检索
        print("  - 步骤1: 执行向量检索...")
        vector_nodes = self._vector_retriever.retrieve(query_str)
        print(f"  - 向量检索找到 {len(vector_nodes)} 个节点。")

        # Cypher检索
        print("  - 步骤2: 执行文本到Cypher检索...")
        cypher_nodes = self._cypher_retriever.retrieve(query_str)
        print(f"  - Cypher检索找到 {len(cypher_nodes)} 个节点。")

        # 合并与去重
        combined_nodes = {}
        for node in vector_nodes + cypher_nodes:
            if node.node.node_id not in combined_nodes:
                combined_nodes[node.node.node_id] = node

        final_nodes = list(combined_nodes.values())
        print(f"--- 混合检索完成，合并去重后共 {len(final_nodes)} 个节点。 ---\n")

        return final_nodes


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
    query = "国内地产销售和居民中长期贷款有什么关系？"
    print(f"问题: {query}")

    # --- 1. 自定义混合检索查询引擎 ---
    print("\n 正在构建自定义混合检索查询引擎...")
    hybrid_retriever = HybridGraphRetriever(
        graph_store=index.property_graph_store,
        index=index,
        similarity_top_k=5
    )
    custom_query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        llm=Settings.llm,
    )

    # --- 2. 基础向量检索查询引擎 ---
    print(" 正在构建基础向量检索查询引擎...")
    base_vector_retriever = VectorContextRetriever(
        index.property_graph_store,
        vector_store=index.vector_store,
        include_text=True,
        similarity_top_k=5
    )
    base_query_engine = RetrieverQueryEngine.from_args(
        retriever=base_vector_retriever,
        llm=Settings.llm
    )

    # --- 对比查询结果 ---

    # 执行自定义查询
    print("\n" + "-" * 25 + " 执行自定义混合查询 " + "-" * 25)
    custom_response = custom_query_engine.query(query)
    print("\n[结果] 自定义混合查询的最终答案:")
    print(custom_response.response)
    print("\n[上下文] 自定义混合查询的源节点:")
    for i, node in enumerate(custom_response.source_nodes):
        print(f"  源 {i + 1} (ID: {node.node_id}, 分数: {node.score:.4f}):")
        print(f"    -> {node.text.strip().replace('\n', ' ')}")

    # 执行基础向量查询
    print("\n" + "-" * 25 + " 执行基础向量查询 " + "-" * 25)
    base_response = base_query_engine.query(query)
    print("\n[结果] 基础向量查询的最终答案:")
    print(base_response.response)
    print("\n[上下文] 基础向量查询的源节点:")
    for i, node in enumerate(base_response.source_nodes):
        print(f"  源 {i + 1} (ID: {node.node_id}, 分数: {node.score:.4f}):")
        print(f"    -> {node.text.strip().replace('\n', ' ')}")


if __name__ == "__main__":
    graph_index = load_existing_graph_index()
    if graph_index:
        run_queries(graph_index)