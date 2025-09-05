import json
import asyncio
import time
import pandas as pd
import phoenix as px
from typing import Dict, List, Optional, Literal, Any
from llama_index.core import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from watchfiles import awatch

from rag_graph.graph_query import HybridGraphRetriever
from opentelemetry import trace
from phoenix.client import Client
from opentelemetry.trace import Status, StatusCode
from llm import VolcengineLLM
from rag_milvus.query_split import parse_query_to_json
from rag_milvus.milvus_filter import query_milvus, bm25_enhanced_search,get_sentence_embedding
from reranker import rerank_results, rerank_nodes
from rag_milvus.config import  *
from Agent.config import *
from rag_milvus import tracing
from rag_milvus.tracing import tracer,shutdown_tracer
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.settings import Settings

# _, _, Settings.embed_model = get_embedding_model()




def build_complex_filters(
        parsed_result: Dict[str, Any],
        filter_fields: List[str]
) -> List[Dict[str, Any]]:
    """
    将LLM解析出的字典，转换成一个复杂的、适配下游处理的筛选器列表。

    Args:
        parsed_result (Dict[str, Any]): 从LLM解析出的键值对结果。
        filter_fields (List[str]): 一个列表，包含应该被用于创建筛选器的字段名称。

    Returns:
        List[Dict[str, Any]]: 一个结构化的筛选器列表。
    """
    current_span = trace.get_current_span()
    current_span.set_attribute("input.parsed_result", json.dumps(parsed_result, ensure_ascii=False))
    current_span.set_attribute("input.filter_fields", filter_fields)

    filters = []
    # 定义用于分割多值字段的分隔符
    delimiters = [",", "、", "/", " "]

    for field, value in parsed_result.items():
        # 1. 跳过不需要或无效的字段
        if field not in filter_fields:
            continue
        if not value or value in ["无", "不明确", "未提及", "None"]:
            continue

        # 2. 特殊处理：日期范围 (date_range)
        # 日期范围的字段名为 'date_range'，实际作用于 'date' 字段
        if field == "date_range" and "至" in str(value):
            try:
                start_date, end_date = str(value).split("至")
                start_date, end_date = start_date.strip(), end_date.strip()
                if start_date:
                    filters.append({"field": "date", "operator": ">=", "value": start_date})
                if end_date:
                    filters.append({"field": "date", "operator": "<=", "value": end_date})
            except ValueError:
                print(f"警告：无法解析日期范围 '{value}'")
            # 处理完日期范围后跳过后续的通用逻辑
            continue

        # 3. 通用逻辑：处理其他字段

        # 尝试按分隔符拆分，以处理多值情况
        query_values = []
        # 将值转换为字符串以使用 in 操作符
        str_value = str(value)
        used_delimiter = next((d for d in delimiters if d in str_value), None)

        if used_delimiter:
            query_values = [v.strip() for v in str_value.split(used_delimiter) if v.strip()]
        else:
            # 如果没有分隔符，说明是单值，也统一放入列表中处理
            query_values = [str_value.strip()]

        # 根据值的数量决定使用 'in' 还是 'like'
        if not query_values:
            continue

        if len(query_values) > 1:
            # --- 多值情况：使用 'in' 操作符 ---
            filters.append({"field": field, "operator": "in", "value": query_values})
        else:
            # --- 单值情况：使用 'like' 操作符---
            filters.append({"field": field, "operator": "like", "value": f"%{query_values[0]}%"})

    current_span.set_attribute("output.filters", json.dumps(filters, ensure_ascii=False))
    return filters


def generate_llm_answer(
        query: str,
        retrieved_results: List[Dict],
        llm,
        max_results: int = 5
) -> str:
    """
    使用 LLM 根据检索到的文本块生成最终答案

    Args:
        query (str): 用户的原始查询。
        retrieved_results (List[Dict]): 从Milvus检索后经过重排的结果列表。
        llm: 已初始化的LLM客户端。
        max_results (int): 用于生成答案的最多上下文片段数量。
        retrieval_type (str): 检索类型 ('normal', 'bm25_enhanced', 'graph_enhanced')，用于微调提示。

    Returns:
        str: LLM生成的最终答案。
    """

    current_span = trace.get_current_span()
    current_span.set_attribute("input.query", query)
    current_span.set_attribute("input.retrieved_results_count", len(retrieved_results))
    current_span.set_attribute("input.max_results", max_results)

    # 1. 处理没有检索到结果的情况
    if not retrieved_results or (len(retrieved_results) == 1 and "message" in retrieved_results[0]):
        return "抱歉，未能在知识库中找到符合筛选条件的高度相关的文本信息来生成答案。"

    # 2. 构建上下文信息 (Context String)
    context_str = ""
    # 只取前 max_results 个结果作为上下文
    for i, result in enumerate(retrieved_results[:max_results], 1):
        context_str += f"--- [参考资料 {i}] ---\n"

        # 包含有用的元数据，让LLM了解上下文
        score = result.get('rerank_score', result.get('similarity_score'))
        if score is not None:
            score_type = "重排相关分" if 'rerank_score' in result else "向量相似分"
            context_str += f"来源文档ID: {result.get('file_id', '未知')}, {score_type}: {score:.4f}\n"

        context_str += f"内容: {result.get('text', '').strip()}\n\n"

    # # 3. 设计 Prompt 模板
    # final_prompt = f"""你是一位顶级的金融分析专家，任务是根据提供的参考资料来精准、深入地回答用户的问题。
    #
    # ### 用户问题:
    # {query}
    #
    # ### 参考资料:
    # {context_str}
    # ### 任务要求:
    # 1.  请仔细阅读并综合所有【参考资料】的内容。
    # 2.  你的回答必须完全基于所提供的资料，禁止编造或引入外部信息。
    # 3.  如果资料内容不足以回答问题，请如实说明“根据当前资料无法回答该问题”。
    # 4.  在回答中，你可以通过 `[参考资料 n]` 的格式来引用信息的来源，以增强回答的可信度。
    # 5.  请条理清晰、逻辑严谨地组织你的答案。
    #
    # 请开始你的回答：
    # """

    # 4. 调用 LLM 生成答案
    try:
        px_client = Client()

        prompt_name = "rag-answer-generator"

        # 拉取最新版本的提示词
        print(f"  正在从 Phoenix 拉取提示词 '{prompt_name}'...")
        prompt_template = px_client.prompts.get(prompt_identifier=prompt_name)
        # 定义要填充到模板中的变量
        prompt_vars = {
            "query": query,
            "context_str": context_str
        }

        # 格式化提示词，得到最终可以发送给 LLM 的内容
        # prompt.format() 返回的是一个字典，符合 OpenAI 的 API 格式
        formatted_prompt_dict = prompt_template.format(variables=prompt_vars)

        # 从返回的字典中提取出用户消息的内容，作为最终的 prompt 字符串
        final_prompt = formatted_prompt_dict["messages"][0]["content"]

        current_span.set_attribute("prompt.template_name", prompt_name)


    except Exception as e:

        print(f"从 Phoenix 拉取或格式化提示词时出错: {e}")

        current_span.record_exception(e)

        current_span.set_status(Status(StatusCode.ERROR, "Failed to retrieve or format prompt from Phoenix"))

        return "抱歉，准备提示词时遇到错误。"

    try:
        response = llm.complete(final_prompt).text.strip()
        current_span.set_attribute("output.final_answer", response)
        return response
    except Exception as e:
        print(f"LLM 生成答案时出错: {e}")
        current_span.record_exception(e)
        current_span.set_status(Status(StatusCode.ERROR, "Exception during LLM completion"))
        return "抱歉，在生成答案的过程中遇到了一个内部错误。"



def retrieve_from_graph(
        query: str,
        graph_index: PropertyGraphIndex
) -> List[Dict]:
    """
    使用 HybridGraphRetriever 从知识图谱中检索信息，并转换为标准字典格式。

    Args:
        query (str): 用户查询。
        graph_index (PropertyGraphIndex): 已加载的图谱索引对象。

    Returns:
        List[Dict]: 转换成与Milvus结果兼容的字典列表。
    """
    current_span = trace.get_current_span()
    current_span.set_attribute("input.query", query)

    print("  执行知识图谱检索...")
    # --- 1. 初始化混合检索器 ---
    hybrid_retriever = HybridGraphRetriever(
        graph_store=graph_index.property_graph_store,
        index=graph_index,
        similarity_top_k=5,
        community_expansion=True
    )

    # --- 2. 执行检索 ---
    # custom_retrieve 返回的是 List[NodeWithScore]
    graph_nodes = hybrid_retriever.custom_retrieve(query)

    # --- 3. 转换为标准字典格式，以便与Milvus结果合并 ---
    results = []
    for node_with_score in graph_nodes:
        results.append({
            "file_id": node_with_score.node.metadata.get("file_name", "graph_node"),
            "text": node_with_score.node.get_content(),
            "similarity_score": node_with_score.score,  # 使用图检索器返回的分数
            "source_type": "knowledge_graph"  # 添加一个来源标识
        })
    print(f"  知识图谱检索到 {len(results)} 个相关节点。")
    current_span.set_attribute("output.retrieved_nodes_count", len(results))
    return results


def execute_rag_pipeline(
        query: str,
        llm: object,  # LLM 客户端实例
        collection_name: str,
        embedding_model_path: str,  # 嵌入模型路径
        graph_index: Optional[PropertyGraphIndex]=None,
        search_strategy: Literal["normal", "bm25_enhanced", "graph_enhanced"] = "normal",  # 选择检索策略
        filter_fields: Optional[List[str]] = None,
        top_k_retrieval: int = 50,  # 初始检索数量
        top_k_llm: int = 5,  # 交给LLM的数量
        search_threshold: float = 0.5,  # 语义搜索阈值
        use_reranker: bool = True,  # 是否启用重排器
        dense_embedding_function: Optional[object] = None,
        reranker_model_function: Optional[object] = None
):
    """
    执行一个完整的、可配置的 RAG（检索增强生成）流程。

    Args:
        query (str): 用户的原始查询。
        llm (object): LLM 客户端实例。
        collection_name (str): Milvus 集合名称。
        embedding_model_path (str): 密集嵌入模型的路径 (用于 query_milvus)。
        graph_index: PropertyGraphIndex的图索引
        search_strategy (Literal["normal", "bm25_enhanced"]):
            - "normal": 使用纯语义搜索结合元数据过滤 (调用 query_milvus)。
            - "bm25_enhanced": 使用语义+关键词混合搜索结合元数据过滤 (调用 bm25_enhanced_search)。
            - "graph_enhanced": 使用bm25_enhanced+graph并行检索
        filter_fields (Optional[List[str]]): 用于从查询中提取的元数据字段列表。
        top_k_retrieval (int): 初始检索返回的结果数量。
        top_k_rerank (int): 重排后保留的结果数量。
        top_k_llm (int): 最终交给 LLM 用于生成答案的上下文数量。
        search_threshold (float): 语义搜索的相似度阈值。
        use_reranker (bool): 是否启用 Reranker 进行结果重排。
        dense_embedding_function (Optional[object]): 混合搜索所需的密集编码器实例。

    Returns:
        str: LLM 生成的最终答案。
    """
    with tracer.start_as_current_span("execute_rag_pipeline") as root_span:
        root_span.set_attribute("input.query", query)
        root_span.set_attribute("input.search_strategy", search_strategy)
        with tracer.start_as_current_span("Phase 1: Parse Query") as parse_phase_span:
            print(f"--- 阶段 1: 解析查询 ---")
            if filter_fields:
                # 步骤 1.1: LLM 解析
                with tracer.start_as_current_span("LLM Parse for Filters") as llm_parse_span:
                    parse_result_json = parse_query_to_json(query, llm)
                    parsed_result = parse_result_json.get("parsed_result", {})
                    llm_parse_span.set_attribute("output.parsed_result", json.dumps(parsed_result, ensure_ascii=False))

                # 步骤 1.2: 构建筛选器
                with tracer.start_as_current_span("Build Filters from Parsed Result") as build_filter_span:
                    filters = build_complex_filters(parsed_result, filter_fields)
                    build_filter_span.set_attribute("output.filters", json.dumps(filters, ensure_ascii=False))
            else:
                filters = ""
            print(f"生成的Milvus筛选器: {filters}")

        with tracer.start_as_current_span("Phase 2: Retrieve") as retrieve_span:
            # --- 2. 根据策略选择并执行检索 ---
            print(f"\n--- 阶段 2: 执行检索 (策略: {search_strategy}) ---")
            all_retrieved_results = []
            with tracer.start_as_current_span("Initial Retrieval") as initial_retrieval_span:
                # -- 路径 1: Milvus 检索 (语义或混合) --
                if search_strategy in ["normal", "bm25_enhanced", "graph_enhanced"]:
                    print("  执行 Milvus 检索...")
                    milvus_results = []
                    if search_strategy == "normal" or search_strategy == "graph_enhanced":
                        milvus_results = query_milvus(
                            collection_name=collection_name, filters=filters, query_text=query,
                            embedding_model_path=embedding_model_path, top_k=top_k_retrieval,
                            search_threshold=search_threshold
                        )
                    elif search_strategy == "bm25_enhanced":
                        milvus_results = bm25_enhanced_search(
                            collection_name=collection_name, query_text=query,
                            embedding_model_path=embedding_model_path, filters=filters,
                            top_k=top_k_retrieval, dense_embedding_function=dense_embedding_function
                        )
                    # 为Milvus结果添加来源标识
                    for res in milvus_results:
                        res['source_type'] = 'milvus'
                    all_retrieved_results.extend(milvus_results)
                    print(f"  Milvus 检索到 {len(milvus_results)} 条结果。")

                    # -- 路径 2: 知识图谱检索 --
                    if search_strategy == "graph_enhanced" and graph_index:
                        graph_results = retrieve_from_graph(query, graph_index)
                        all_retrieved_results.extend(graph_results)

                    # -- 结果去重 --
                    # 根据文本内容进行去重，避免完全相同的内容被多次处理
                    unique_results = {}
                    for res in all_retrieved_results:
                        text_content = res.get("text", "").strip()
                        if text_content and text_content not in unique_results:
                            unique_results[text_content] = res
                    final_initial_results = list(unique_results.values())
                    print(f"\n初步检索并去重后，共获得 {len(final_initial_results)} 条候选结果。")
                    retrieve_span.set_attribute("output.unique_results_count", len(final_initial_results))

            # --- 3. 结果重排 ---
            final_retrieved_docs = []
            if use_reranker and final_initial_results and "message" not in final_initial_results[0]:
                with tracer.start_as_current_span("Phase 3: Rerank") as rerank_span:
                    print(f"\n--- 阶段 3: 对混合结果执行重排 ---")
                    reranker_model, reranker_tokenizer = reranker_model_function
                    final_retrieved_docs = rerank_results(
                        results=final_initial_results,
                        query=query,
                        model=reranker_model,
                        tokenizer=reranker_tokenizer
                    )
                    rerank_span.set_attribute("output.reranked_docs_count", len(final_retrieved_docs))
                    print(f"重排后保留 {len(final_retrieved_docs)} 条结果。")
            else:

                final_retrieved_docs = final_initial_results[:top_k_llm * 2]  # 留多一点给LLM选择

            # --- 4. LLM 生成最终答案 ---
        with tracer.start_as_current_span("Phase 4: Generate Final Answer") as generation_span:
            print(f"\n--- 阶段 4: 生成最终答案 ---")
            final_answer = generate_llm_answer(
                query=query,
                retrieved_results=final_retrieved_docs,
                llm=llm,
                max_results=top_k_llm
            )
            root_span.set_attribute("output.final_answer", final_answer)
            return final_retrieved_docs, final_answer


def _milvus_result_to_nodes(milvus_results: List[Dict]) -> List[NodeWithScore]:
    """将 Milvus 的原始字典结果转换为 NodeWithScore 对象列表"""
    nodes = []
    for res in milvus_results:
        # 创建一个 TextNode 对象
        node = TextNode(
            text=res.get("text", ""),
            doc_id=res.get("doc_id",""),
            # 将所有 Milvus 的元数据（包括 file_id）都放入 metadata
            metadata={
                "source_type": "milvus",  # 添加来源标识
                **res.get("metadata", {})  # 将其他元数据也加进去
            }
        )
        nodes.append(NodeWithScore(node=node, score=res.get("similarity_score")))
    return nodes


def _graph_result_to_nodes(graph_results: List[Dict]) -> List[NodeWithScore]:
    """将图谱检索的字典结果转换为 NodeWithScore 对象列表"""
    nodes = []
    for res in graph_results:
        node = TextNode(
            text=res.get("text", ""),
            id = res.get(("id","")),
            metadata={
                "source_type": "knowledge_graph",
            }
        )
        nodes.append(NodeWithScore(node=node, score=res.get("similarity_score")))
    return nodes



async def retrieve_and_rerank_pipeline(
        query: str,
        llm: object,  # LLM 客户端实例
        collection_name: str,
        embedding_model_path: str,
        graph_index: Optional[PropertyGraphIndex] = None,
        search_strategy: str = "graph_enhanced",
        filter_fields: Optional[List[str]] = None,
        top_k_retrieval: int = 10,
        top_k_rerank: int = 5,  # 重排后保留的数量
        search_threshold: float = 0.5,  # 语义搜索阈值
        use_reranker: bool = True,  # 是否启用重排器
        dense_embedding_function: Optional[object] = None,
        reranker_model_function: Optional[object] = None
) -> List[NodeWithScore]:
    """
    一个只负责“检索”和“重排”的新 pipeline。
    它的最终输出是一个经过排序和筛选的 NodeWithScore 对象列表。
    """
    with tracer.start_as_current_span("retrieve_and_rerank_pipeline") as root_span:
        root_span.set_attribute("input.query", query)
        root_span.set_attribute("input.search_strategy", search_strategy)
        with tracer.start_as_current_span("Phase 1: Parse Query") as parse_phase_span:
            print(f"--- 阶段 1: 解析查询 ---")
            if filter_fields:
                # 步骤 1.1: LLM 解析
                with tracer.start_as_current_span("LLM Parse for Filters") as llm_parse_span:
                    parse_result_json = parse_query_to_json(query, llm)
                    parsed_result = parse_result_json.get("parsed_result", {})
                    llm_parse_span.set_attribute("output.parsed_result", json.dumps(parsed_result, ensure_ascii=False))

                # 步骤 1.2: 构建筛选器
                with tracer.start_as_current_span("Build Filters from Parsed Result") as build_filter_span:
                    filters = build_complex_filters(parsed_result, filter_fields)
                    build_filter_span.set_attribute("output.filters", json.dumps(filters, ensure_ascii=False))
            else:
                filters = ""
            print(f"生成的Milvus筛选器: {filters}")
        with tracer.start_as_current_span("Initial Retrieval") as retrieve_span:
            all_retrieved_results = []  # 用于收集所有来源的原始字典结果

            # 1. 基础检索：总是先从 Milvus 开始
            print(f"\n--- 阶段 2: 执行基础检索 (策略: {search_strategy}) ---")

            if search_strategy == "normal" or search_strategy == "graph_enhanced":
                # 对于 'normal' 和 'graph_enhanced'，使用语义搜索
                print("  执行 Milvus 语义搜索...")
                milvus_raw_results = await asyncio.to_thread(
                    query_milvus,
                    collection_name=collection_name, filters=filters, query_text=query,
                    embedding_model_path=embedding_model_path, top_k=top_k_retrieval,
                    search_threshold=search_threshold
                )
                for res in milvus_raw_results:
                    res['source_type'] = 'milvus'
                all_retrieved_results.extend(milvus_raw_results)
                print(f"  Milvus 语义搜索找到 {len(milvus_raw_results)} 条结果。")

            elif search_strategy == "bm25_enhanced":
                print("  执行 Milvus BM25 增强混合搜索...")
                milvus_raw_results = await asyncio.to_thread(
                    bm25_enhanced_search,
                    collection_name=collection_name, query_text=query,
                    embedding_model_path=embedding_model_path, filters=filters,
                    top_k=top_k_retrieval, dense_embedding_function=dense_embedding_function
                )
                all_retrieved_results.extend(milvus_raw_results)
                print(f"  Milvus 混合搜索找到 {len(milvus_raw_results)} 条结果。")

            # 2. 增强检索：如果策略是 graph_enhanced，则加入图谱结果
            if search_strategy == "graph_enhanced" and graph_index:
                print("\n  --- 使用知识图谱进行结果增强 ---")
                graph_raw_results = await asyncio.to_thread(
                    retrieve_from_graph, query, graph_index
                )
                all_retrieved_results.extend(graph_raw_results)
                print(f"  知识图谱找到 {len(graph_raw_results)} 条补充结果。")


            # 去重
            unique_results_dict = {}
            for res in all_retrieved_results:
                content = res.get("text", "").strip()
                if content and content not in unique_results_dict:
                    unique_results_dict[content] = res

            unique_initial_results = list(unique_results_dict.values())
            print(f"\n初步检索并去重后，共获得 {len(unique_initial_results)} 条候选结果。")

            # 将所有来源的、去重后的字典，统一转换为 NodeWithScore
            all_initial_nodes: List[NodeWithScore] = []
            milvus_nodes = _milvus_result_to_nodes(
                [r for r in unique_initial_results if r.get('source_type') == 'milvus'])
            graph_nodes = _graph_result_to_nodes(
                [r for r in unique_initial_results if r.get('source_type') == 'knowledge_graph'])
            all_initial_nodes.extend(milvus_nodes)
            all_initial_nodes.extend(graph_nodes)

            retrieve_span.set_attribute("output.unique_nodes_count", len(all_initial_nodes))
        # --- 阶段 3: 结果重排 ---
        final_nodes: List[NodeWithScore] = []
        if use_reranker and all_initial_nodes:
            with tracer.start_as_current_span("Rerank") as rerank_span:
                print(f"\n--- 阶段 3: 对 {len(all_initial_nodes)} 个候选节点执行重排 ---")

                # rerank_results 需要被改造，以接收和返回 NodeWithScore
                reranker_model, reranker_tokenizer = reranker_model_function
                final_nodes = rerank_nodes(
                    nodes=all_initial_nodes,
                    query=query,
                    top_n=top_k_rerank,
                    model= reranker_model,
                    tokenizer= reranker_tokenizer
                )
                print(f"重排后保留 {len(final_nodes)} 个节点。")
        else:
            # 如果不使用重排，就按初始分数排序并截断
            all_initial_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
            final_nodes = all_initial_nodes[:top_k_rerank]

        root_span.set_attribute("output.final_nodes_count", len(final_nodes))

        # --- 最终返回 NodeWithScore 列表 ---
        return final_nodes

async def main():
    query = "美债高压的成因及其对实体经济的影响是什么？"
    llm = VolcengineLLM(api_key=API_KEY)

    filter_fields = [
        "institution",
        "report_type",
        "authors",
        "date_range"
    ]
    print("\n--- 正在初始化知识图谱索引 ---")
    graph_index = None
    try:
        graph_store = Neo4jPropertyGraphStore(
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            url=NEO4J_URI,
            database=NEO4J_DATABASE,
        )
        graph_index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            embed_model=Settings.embed_model
        )
        print("知识图谱索引初始化成功。")
    except Exception as e:
        print(f"知识图谱索引初始化失败: {e}")

    result_nodes = await retrieve_and_rerank_pipeline(
            query=query,
            llm=llm,
            collection_name="financial_reports",
            graph_index=graph_index,
            embedding_model_path=EMBEDDING_MODEL_PATH,
            search_strategy="graph_enhanced",
            search_threshold=0.5,
            filter_fields=filter_fields,
            top_k_retrieval=20,
            top_k_rerank=10,
            dense_embedding_function=get_sentence_embedding,
            reranker_model_function=get_reranker_model())
    print(f"The retrieved nodes are {result_nodes}")

# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         print("\n测试被用户中断。")
#     tracing.shutdown_tracer()