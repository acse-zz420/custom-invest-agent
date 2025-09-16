import asyncio
from typing import List
from llama_index.core import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.settings import Settings
from llama_index.core.tools import FunctionTool

from graph.graph_query import HybridGraphRetriever
from rag_milvus.rag_pipeline_raw import get_sentence_embedding,get_reranker_model,retrieve_and_rerank_pipeline_test
from prompt import CUSTOM_QA_TEMPLATE, CUSTOM_REFINE_TEMPLATE
from llm_ali import QwenToolLLM
from config import *
from llama_index.core.schema import NodeWithScore, TextNode

llm = QwenToolLLM()
def custom_graph_search(query: str) -> List[NodeWithScore]:
    """
    使用混合图谱检索策略（向量+社区扩展）来回答关于实体关系的问题。
    适用于需要深度图谱遍历和关联分析的复杂查询。
    """
    print(f"\n[图谱检索工具] 接收到查询: '{query}'")

    # a. 从已有的 Property Graph Store 创建索引对象
    graph_store = Neo4jPropertyGraphStore(
        username=AURA_DB_USER_NAME,  # 使用实际的用户名变量
        password=AURA_DB_PASSWORD,
        url=AURA_URI,
        database=AURA_DATABASE,
    )
    index = PropertyGraphIndex.from_existing(property_graph_store=graph_store)

    # b. 构建您的自定义混合检索器
    hybrid_retriever = HybridGraphRetriever(
        graph_store=index.property_graph_store,
        index=index,
        similarity_top_k=10,
        community_expansion=True,
    )

    retrieved_nodes = hybrid_retriever.retrieve(query)

    print(f"[图谱检索工具] 检索到 {len(retrieved_nodes)} 个节点。")
    return retrieved_nodes


async def custom_milvus_search(query: str) -> List[NodeWithScore]:
    """
    使用BM25增强的混合搜索策略，从Milvus向量数据库的金融文档中检索信息。
    适用于需要从大量非结构化文本中寻找答案的问题。
    """
    print(f"\n[Milvus检索工具] 接收到查询: '{query}'")

    filter_fields = [
        "institution",
        "report_type",
        "authors",
        # "date_range"
    ]

    retrieved_nodes = await retrieve_and_rerank_pipeline_test(
        query=query,
        llm=llm,
        collection_name="financial_reports",
        embedding_model_path=EMBEDDING_MODEL_PATH,
        search_strategy="bm25_enhanced",
        search_threshold=0.4,
        filter_fields=filter_fields,
        top_k_retrieval=10,
        top_k_rerank=3,
        dense_embedding_function=get_sentence_embedding,
        reranker_model_function=get_reranker_model()
    )

    print(f"[Milvus检索工具] 检索并重排后，返回 {len(retrieved_nodes)} 个节点。")

    # 直接返回结构化的节点列表
    return retrieved_nodes

custom_graph_rag_tool = FunctionTool.from_defaults(
    fn=custom_graph_search, # 同步函数
    name="knowledge_graph_query",
    description=(
        "用于查询实体之间的复杂关系和结构化信息。例如“哪些公司与A公司有合作关系？”或“B高管在哪些公司任职？”"
    )
)

custom_vector_rag_tool = FunctionTool.from_defaults(
    async_fn=custom_milvus_search, # 异步函数
    name="financial_document_search",
    description=(
        "用于从金融文档和研究报告中检索具体信息。适用于回答关于市场分析、公司财报、宏观经济等方面的问题。"
    )
)

# 导出新的工具列表
rag_tools = [custom_vector_rag_tool, custom_graph_rag_tool]
