import asyncio
from llama_index.core import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.settings import Settings
from llama_index.core.tools import FunctionTool

from graph.graph_query import HybridGraphRetriever
from rag_milvus.run import execute_rag_pipeline,get_sentence_embedding,get_reranker_model
from prompt import CUSTOM_QA_TEMPLATE, CUSTOM_REFINE_TEMPLATE
from llm import VolcengineLLM
from Agent.config import *

llm = VolcengineLLM(api_key=API_KEY)
def custom_graph_search(query: str) -> str:
    """
    使用混合图谱检索策略（向量+社区扩展）来回答关于实体关系的问题。
    适用于需要深度图谱遍历和关联分析的复杂查询。
    """
    print(f"\n[图谱检索工具] 接收到查询: '{query}'")

    # a. 从已有的 Property Graph Store 创建索引对象
    graph_store = Neo4jPropertyGraphStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URI,
        database=NEO4J_DATABASE
    )
    index = PropertyGraphIndex.from_existing(property_graph_store=graph_store)

    # b. 构建您的自定义混合检索器
    hybrid_retriever = HybridGraphRetriever(
        graph_store=index.property_graph_store,
        index=index,
        similarity_top_k=10,
        community_expansion=True,
    )

    # c. 构建响应合成器
    synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode=ResponseMode.COMPACT,
        text_qa_template=CUSTOM_QA_TEMPLATE,
        refine_template=CUSTOM_REFINE_TEMPLATE
    )

    # d. 构建最终的查询引擎
    custom_query_engine = RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=synthesizer
    )

    # e. 执行查询并返回结果
    response = custom_query_engine.query(query)
    return str(response)


async def custom_milvus_search(query: str) -> str:
    """
    使用BM25增强的混合搜索策略，从Milvus向量数据库的金融文档中检索信息。
    适用于需要从大量非结构化文本中寻找答案的问题。
    """
    print(f"\n[Milvus检索工具] 接收到查询: '{query}'")

    filter_fields = [
        "institution",
        "report_type",
        "authors",
        "date_range"
    ]

    response = await asyncio.to_thread(
        execute_rag_pipeline,
        query=query,
        llm=llm,
        collection_name="financial_reports",
        embedding_model_path=EMBEDDING_MODEL_PATH,
        search_strategy="bm25_enhanced",
        search_threshold=0.4,
        filter_fields=filter_fields,
        top_k_retrieval=50,
        top_k_llm=5,
        dense_embedding_function=get_sentence_embedding,
        reranker_model_function=get_reranker_model()
    )

    return str(response)

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