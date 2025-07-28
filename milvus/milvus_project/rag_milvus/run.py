import json
import time
from typing import Dict, List, Optional, Any
from llm import VolcengineLLM
from query_split import parse_query_to_json
from milvus_filter import query_milvus
from reranker import rerank_results, load_reranker


def build_filters_from_llm_result(
        parsed_result: Dict,
        filterable_fields: List[str]
) -> List[Dict]:
    """
    根据query_split解析出的字典，为指定的字段构建适配 query_milvus 的筛选器列表。

    Args:
        parsed_result (Dict): 从LLM解析出的键值对结果。
        filterable_fields (List[str]): 一个列表，包含应该被用于创建筛选器的字段名称。

    Returns:
        List[Dict]: 用于 milvus_filter.query_milvus 的筛选器列表。
    """
    filters = []
    delimiters = [",", "、", "/", " "]

    for field, value in parsed_result.items():
        if field not in filterable_fields:
            continue
        if not value or value in ["无", "不明确", "未提及"]:
            continue

        # --- 日期范围 ---
        if field == "date_range" and "至" in value:
            try:
                start_date, end_date = value.split("至")
                filters.append({"field": "date", "operator": ">=", "value": start_date.strip()})
                filters.append({"field": "date", "operator": "<=", "value": end_date.strip()})
            except ValueError:
                print(f"警告：无法解析日期范围 '{value}'")
            continue

        # 1. 无论输入是 "A" 还是 "A,B"，都先将其拆分为一个值列表。
        query_values = []
        # 找到第一个存在的分隔符
        used_delimiter = next((d for d in delimiters if d in str(value)), None)

        if used_delimiter:
            query_values = [v.strip() for v in str(value).split(used_delimiter) if v.strip()]
        else:
            # 如果没有分隔符，说明是单值，也放入列表中
            query_values = [str(value).strip()]

        # 2. 如果列表不为空，则创建一个 'like_any' 筛选器
        if query_values:
            # 它的值是一个列表，代表列表中任一元素通过LIKE匹配即可。
            filters.append({"field": field, "operator": "like_any", "value": query_values})

    return filters

def deduplicate_results(results: List[dict]) -> List[dict]:
    """根据 chunk_id 去重，保留第一个出现的记录。"""
    seen_chunk_ids = set()
    deduped_results = []
    for result in results:
        chunk_id = result.get("chunk_id")
        if chunk_id and chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            deduped_results.append(result)
    return deduped_results


def generate_llm_answer(
        query: str,
        retrieved_results: List[Dict],
        llm,
        max_results: int = 5,
        retrieval_type: str = "hybrid"  # 新增参数，说明检索类型
) -> str:
    """
    使用 LLM 根据检索到的文本块生成最终答案（已移除图谱RAG）。

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
        if retrieval_type == "metadata":
            return "根据您提供的筛选条件，未能找到相关的文档。"
        else:
            return "抱歉，未能在知识库中找到与您问题高度相关的文本信息来生成答案。"

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

        context_str += f"内容: {result.get('chunk_text', '').strip()}\n\n"

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



if __name__ == "__main__":

    start_time = time.time()

    # --- 1. 初始化 ---
    llm = VolcengineLLM(api_key="ff6acab6-c747-49d7-b01c-2bea59557b8d")
    embedding_model_path = r"D:\yechuan\work\cjsx\model\Qwen3-Embedding-0.6B"
    reranker_model_path = r"D:\yechuan\work\cjsx\model\bge-reranker-large"

    query = "针对中国房地产政策的相关宏观研究，哪些不足和可以改善的地方"

    # --- 2. LLM 解析查询 ---
    print(f"原始查询: {query}")
    parse_result_json = parse_query_to_json(query, llm)
    parsed_result = parse_result_json.get("parsed_result", {})
    print(f"LLM解析结果: {json.dumps(parsed_result, ensure_ascii=False, indent=2)}")

    # --- 3. 构建 Milvus 筛选器 ---
    filter_fields = [
        "institution",
        "report_type",
        "industry",
        "authors",
        "date_range"
    ]
    filters = build_filters_from_llm_result(parsed_result,filter_fields)
    print(f"生成的Milvus筛选器: {filters}")

    # --- 4. Milvus 混合搜索 ---
    # 同时使用语义查询 (query_text) 和元数据筛选 (filters)
    milvus_results = query_milvus(
        collection_name="financial_reports",
        filters=filters,
        logic_operator="and",  # 多个筛选条件之间使用 AND
        query_text=query,  # 使用原始查询进行语义搜索
        embedding_model_path=embedding_model_path,
        top_k=50,
        search_threshold=0.5  # IP相似度阈值
    )

    # --- 5. 结果去重和重排 ---
    deduped_results = deduplicate_results(milvus_results)
    print(f"Milvus 返回 {len(milvus_results)} 个结果, 去重后剩 {len(deduped_results)} 个。")

    if deduped_results and "message" not in deduped_results[0]:
        print("正在使用Reranker对结果进行重排...")
        reranker_model, reranker_tokenizer = load_reranker(reranker_model_path)
        reranked_results = rerank_results(
            results=deduped_results,
            query=query,
            model=reranker_model,
            tokenizer=reranker_tokenizer,
            top_k=10
        )
        print("重排完成。")
    else:
        reranked_results = []

    # --- 6. LLM 生成最终答案 ---
    print("正在生成最终答案...")

    # 可以根据执行的步骤来确定 retrieval_type
    # 在这个完整的流程中，执行了混合搜索，所以是 'hybrid'
    final_answer = generate_llm_answer(
        query=query,
        retrieved_results=reranked_results,  # 传入重排后的结果
        llm=llm,
        max_results=5,  # 使用重排后得分最高的5个结果
        retrieval_type="hybrid"
    )

    # ... (后面的打印和输出部分保持不变) ...
    print("\nLLM 生成的最终答案：\n")
    print(final_answer)
    end_time = time.time()
    print(f"✅ 查询完成 (总耗时: {end_time - start_time:.2f} 秒)")