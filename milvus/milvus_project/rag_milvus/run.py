import json
from typing import Dict, List, Optional, Literal, Any
from tool import timer
from llm import VolcengineLLM
from query_split import parse_query_to_json
from milvus_filter import query_milvus, run_hybrid_search, get_sentence_embedding
from reranker import rerank_results
from config import *


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

    return filters


@timer
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
        retrieval_type (str): 检索类型 ('semantic', 'metadata', 'hybrid')，用于微调提示。

    Returns:
        str: LLM生成的最终答案。
    """
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

    # 3. 设计 Prompt 模板
    final_prompt = f"""你是一位顶级的金融分析专家，任务是根据提供的参考资料来精准、深入地回答用户的问题。

    ### 用户问题:
    {query}

    ### 参考资料:
    {context_str}
    ### 任务要求:
    1.  请仔细阅读并综合所有【参考资料】的内容。
    2.  你的回答必须完全基于所提供的资料，禁止编造或引入外部信息。
    3.  如果资料内容不足以回答问题，请如实说明“根据当前资料无法回答该问题”。
    4.  在回答中，你可以通过 `[参考资料 n]` 的格式来引用信息的来源，以增强回答的可信度。
    5.  请条理清晰、逻辑严谨地组织你的答案。

    请开始你的回答：
    """

    # 4. 调用 LLM 生成答案
    try:
        response = llm.complete(final_prompt).text.strip()
        return response
    except Exception as e:
        print(f"LLM 生成答案时出错: {e}")
        return "抱歉，在生成答案的过程中遇到了一个内部错误。"


@timer
def execute_rag_pipeline(
        query: str,
        llm: object,  # LLM 客户端实例
        collection_name: str,
        embedding_model_path: str,  # 嵌入模型路径
        search_strategy: Literal["normal", "hybrid"] = "hybrid",  # 关键参数：选择检索策略
        filter_fields: Optional[List[str]] = None,
        top_k_retrieval: int = 50,  # 初始检索数量
        top_k_llm: int = 5,  # 交给LLM的数量
        search_threshold: float = 0.5,  # 语义搜索阈值
        use_reranker: bool = True,  # 是否启用重排器
        # --- 混合搜索专用参数 ---
        dense_embedding_function: Optional[object] = None,  # 仅在 hybrid 模式下需要
):
    """
    执行一个完整的、可配置的 RAG（检索增强生成）流程。

    Args:
        query (str): 用户的原始查询。
        llm (object): LLM 客户端实例。
        collection_name (str): Milvus 集合名称。
        embedding_model_path (str): 密集嵌入模型的路径 (用于 query_milvus)。
        search_strategy (Literal["semantic", "hybrid"]):
            - "semantic": 使用纯语义搜索结合元数据过滤 (调用 query_milvus)。
            - "hybrid": 使用语义+关键词混合搜索结合元数据过滤 (调用 run_hybrid_search)。
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

    # --- 1. LLM 解析查询以提取元数据 ---
    print(f"--- 阶段 1: 解析查询 ---")
    print(f"原始查询: {query}")
    if filter_fields:
        parse_result_json = parse_query_to_json(query, llm)
        parsed_result = parse_result_json.get("parsed_result", {})
        print(f"LLM解析结果: {json.dumps(parsed_result, ensure_ascii=False, indent=2)}")
        filters = build_complex_filters(parsed_result, filter_fields)
        print(f"生成的Milvus筛选器: {filters}")
    else:
        filters = ""  # 如果不提供 filter_fields，则不进行元数据筛选
        print("未配置元数据筛选。")

    # --- 2. 根据策略选择并执行检索 ---
    print(f"\n--- 阶段 2: 执行检索 (策略: {search_strategy}) ---")
    retrieved_results = []

    if search_strategy == "semantic":
        retrieved_results = query_milvus(
            collection_name=collection_name,
            filters=filters,
            query_text=query,
            embedding_model_path=embedding_model_path,
            top_k=top_k_retrieval,
            search_threshold=search_threshold
        )

    elif search_strategy == "hybrid":
        retrieved_results = run_hybrid_search(
            collection_name=collection_name,
            query_text=query,
            embedding_model_path=embedding_model_path,  # 假设 hybrid search 也需要
            filters=filters,
            top_k=top_k_retrieval,
            dense_embedding_function=dense_embedding_function
        )

    else:
        raise ValueError(f"不支持的 search_strategy: '{search_strategy}'. 请选择 'semantic' 或 'hybrid'。")

    print(f"初步检索到 {len(retrieved_results)} 条结果。")

    # --- 3. 结果重排 (可选) ---
    if use_reranker and retrieved_results and "message" not in retrieved_results[0]:
        print(f"\n--- 阶段 3: 执行重排 ---")
        reranker_model, reranker_tokenizer = get_reranker_model()
        final_retrieved_docs = rerank_results(
            results=retrieved_results,
            query=query,
            model=reranker_model,
            tokenizer=reranker_tokenizer
        )
        print(f"重排后保留 {len(final_retrieved_docs)} 条结果。")
    else:
        # 如果不使用重排器，直接使用原始检索结果
        final_retrieved_docs = retrieved_results[:top_k_rerank]

    # --- 4. LLM 生成最终答案 ---
    print(f"\n--- 阶段 4: 生成最终答案 ---")
    final_answer = generate_llm_answer(
        query=query,
        retrieved_results=final_retrieved_docs,
        llm=llm,
        max_results=top_k_llm
    )

    return final_answer


if __name__ == "__main__":
    query = "根据宏观研究,分析亚洲货币升值意味着什么？"
    llm = VolcengineLLM(api_key=API_KEY)

    filter_fields = [
        "institution",
        "report_type",
        "authors",
        "date_range"
    ]

    print("开始执行RAG流程 (策略: Hybrid)...")
    print("=" * 50)
    hybrid_answer = execute_rag_pipeline(
        query=query,
        llm=llm,
        collection_name="financial_reports",
        embedding_model_path=EMBEDDING_MODEL_PATH,
        search_strategy="hybrid",
        filter_fields=filter_fields,
        top_k_retrieval=50,
        top_k_llm=5,
        dense_embedding_function=get_sentence_embedding
    )

    print("\n\n【混合搜索】LLM 生成的最终答案：\n")
    print(hybrid_answer)

