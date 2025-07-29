import numpy as np
from tool import timer
from pymilvus import Collection, connections, utility
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Any
from milvus_construct import get_sentence_embedding
from llm import VolcengineLLM
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers.fusion_retriever import QueryFusionRetriever
from llama_index.core.schema import TextNode,NodeWithScore
from llama_index.core import VectorStoreIndex,Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def normalize_vector(vector: np.ndarray) -> List[float]:
    """归一化向量以支持 IP 度量，使其等效于余弦相似度。"""
    if np.linalg.norm(vector) == 0:
        return vector.tolist()
    return (vector / np.linalg.norm(vector)).tolist()

@timer
def _build_filter_expression(filters: List[Dict], logic_operator: str) -> Optional[str]:
    """根据筛选条件列表构建 Milvus 查询表达式 (expr)。"""
    if not filters:
        return None

    expressions = []
    for f in filters:
        field = f.get("field")
        op = f.get("operator")
        value = f.get("value")

        if not all([field, op, value]):
            continue

        milvus_field = f'metadata["{field}"]'
        # 对字符串值进行处理，避免注入和语法错误
        if op in ["==", "!="]:
            if isinstance(value, str):
                expressions.append(f'{milvus_field} {op} "{value}"')
            else:  # for numbers or booleans
                expressions.append(f'{milvus_field} {op} {value}')
        elif op == "like":
            expressions.append(f'{milvus_field} like "{value}"')
        elif op in ["in", "not in"]:
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                value_list = ",".join(f'"{v}"' for v in value)
                expressions.append(f"{milvus_field} {op} [{value_list}]")
            elif isinstance(value, list):  # for lists of numbers
                value_list = ",".join(str(v) for v in value)
                expressions.append(f"{milvus_field} {op} [{value_list}]")
        elif op in [">", ">=", "<", "<="]:
            if isinstance(value, str):
                expressions.append(f'{milvus_field} {op} "{value}"')
            else:
                expressions.append(f'{milvus_field} {op} {value}')
        elif op == "like_any":
            # 处理自定义的 'like_any' 操作符
            if isinstance(value, list) and value:
                sub_expressions = [f'{milvus_field} like "%{v}%"' for v in value]
                or_expression = " or ".join(sub_expressions)
                expressions.append(f"({or_expression})")

    if not expressions:
        return None

    separator = f" {logic_operator} "
    return separator.join(f"({expr})" for expr in expressions)

@timer
def query_milvus(
        collection_name: str,
        filters: Optional[List[Dict]] = None,
        logic_operator: str = "and",
        query_text: Optional[str] = None,
        embedding_model_path: Optional[str] = None,
        top_k: int = 10,
        search_threshold: float = 0.5,
        host: str = "localhost",
        port: str = "19530",
) -> List[Dict[str, Any]]:
    """
    查询 Milvus 集合，支持复杂的元数据筛选和可选的语义向量搜索。

    Args:
        collection_name (str): 要查询的集合名称。
        filters (Optional[List[Dict]]): 元数据筛选条件列表。每个字典包含:
            - "field" (str): 要筛选的字段名。
            - "operator" (str): 操作符, 支持 "==", "!=", "in", "not in", "like", ">", "<", ">=", "<=".
            - "value" (Any): 筛选的值。
        logic_operator (str): 多个筛选条件间的逻辑关系, "and" 或 "or"。
        query_text (Optional[str]): 用于语义搜索的查询文本。如果为 None, 则只进行元数据筛选。
        embedding_model_path (Optional[str]): 嵌入模型的本地路径。仅在提供了 query_text 时需要。
        top_k (int): 返回结果的最大数量。
        search_threshold (float): 语义搜索的相似度阈值 (对于IP度量, 越高越相关)。
        host (str): Milvus 服务主机。
        port (str): Milvus 服务端口。

    Returns:
        List[Dict[str, Any]]: 查询结果列表，或在无结果时返回空列表。
    """

    if query_text and not embedding_model_path:
        raise ValueError("当提供 query_text 进行语义搜索时，必须指定 embedding_model_path。")

    print(f"开始连接 Milvus server at {host}:{port}...")
    connections.connect(alias="default", host=host, port=port)
    print("Milvus 连接成功。")

    try:
        if not utility.has_collection(collection_name):
            print(f"错误：集合 '{collection_name}' 不存在。")
            return []

        collection = Collection(collection_name)
        collection.load()

        # 构建元数据筛选表达式
        filter_expr = _build_filter_expression(filters, logic_operator)
        print(f"生成的筛选表达式 (expr): {filter_expr}")

        # 获取所有字段名用于输出，除了向量字段
        output_fields = [field.name for field in collection.schema.fields if
                         not field.is_primary and field.name != "embedding"]

        # --- 根据是否有 query_text 决定执行逻辑 ---
        if query_text:
            # 一：语义搜索 (可结合元数据筛选)
            print(f"正在执行语义搜索: '{query_text}'")
            tokenizer = AutoTokenizer.from_pretrained(embedding_model_path, local_files_only=True)
            model = AutoModel.from_pretrained(embedding_model_path, local_files_only=True)

            query_embedding_raw = get_sentence_embedding(query_text, model, tokenizer)
            query_embedding = normalize_vector(query_embedding_raw)

            search_params = {"metric_type": "IP", "params": {"nprobe": 16}}  # nprobe 可根据索引类型调整

            search_results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=output_fields
            )

            # 处理并过滤搜索结果
            final_results = []
            for hit in search_results[0]:
                if hit.distance >= search_threshold:
                    record = {field: hit.entity.get(field) for field in output_fields}
                    record["similarity_score"] = hit.distance  # 返回相似度分数
                    final_results.append(record)

            if not final_results:
                print("根据阈值筛选后，无满足条件的结果。")
                return [{"message": "无满足条件的结果"}]
            return final_results

        else:
            # 二: 仅元数据筛选
            if not filter_expr:
                print("错误：未提供语义查询文本也未提供任何筛选条件。")
                return []

            print("正在执行元数据筛选查询...")
            results = collection.query(
                expr=filter_expr,
                limit=top_k,
                output_fields=output_fields
            )
            return results

    finally:
        connections.disconnect("default")
        print("Milvus 连接已断开。")

@timer
def get_all_nodes_from_milvus(collection_name: str, host: str, port: str) -> List[TextNode]:
    """从Milvus中获取所有文档，并转换为LlamaIndex的TextNode对象。"""
    print("正在从 Milvus 加载所有节点以初始化BM25...")
    nodes = []
    try:
        connections.connect( host=host, port=port)
        if not utility.has_collection(collection_name):
            return []

        collection = Collection(collection_name)
        collection.load()

        # 获取所有字段，除了向量
        output_fields = [f.name for f in collection.schema.fields if f.name != "embedding"]

        # 使用迭代器获取所有实体
        iterator = collection.query_iterator(expr="doc_id != ''", output_fields=output_fields)
        while True:
            batch = iterator.next()
            if not batch:
                break
            for entity in batch:
                node = TextNode(
                    id_=entity.pop("doc_id"),
                    text=entity.pop("text"),
                    metadata=entity  # 剩下的所有字段都作为元数据
                )
                nodes.append(node)
        return nodes
    finally:
        if "default" in connections.list_connections():
            connections.disconnect()




@timer
def run_hybrid_search(
        collection_name: str,
        query_text: str,
        logic_operator: str,
        embedding_model_path,
        host: str = "localhost",
        port: str = "19530",
        filters: Optional[List[Dict]] = None,
        top_k: int = 10,
):
    """
    执行一个优化的混合搜索：语义+关键词+元数据过滤。
    """
    print("--- 启动混合搜索流程 ---")

    # --- 1. 初始化向量存储和索引 ---
    vector_search_results = query_milvus(
        collection_name=collection_name,
        filters=filters,
        logic_operator=logic_operator,
        query_text=query_text,
        embedding_model_path=embedding_model_path,
        top_k=top_k,
        host=host,
        port=port
    )
    if not vector_search_results or (
            "message" in vector_search_results[0] and vector_search_results[0]["message"] == "无满足条件的结果"):
        print("第一阶段（向量检索）未找到任何结果，流程终止。")
        return []

    filtered_nodes: Dict[str, TextNode] = {}
    for res_dict in vector_search_results:
        non_metadata_keys = {"doc_id", "text", "similarity_score", "id"}
        metadata = {k: v for k, v in res_dict.items() if k not in non_metadata_keys}

        node_id = res_dict.get("doc_id")
        node = TextNode(id_=node_id, text=res_dict.get("text", ""), metadata=metadata)
        filtered_nodes[node_id] = node
    print(f"向量检索完成，找到 {len(filtered_nodes)} 个结果。")

    # b) BM25 检索器 (处理关键字搜索)
    filtered_nodes_list = list(filtered_nodes.values())  # 转换为 TextNode 列表
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=filtered_nodes_list,
        similarity_top_k=top_k
    )
    bm25_results: List[NodeWithScore] = bm25_retriever.retrieve(query_text)
    print(f"BM25检索完成，找到 {len(bm25_results)} 个结果。")

    final_results: List[NodeWithScore] = []
    for bm25_hit in bm25_results:
        # BM25的分数大于0，说明是有效匹配
        if bm25_hit.score >= 0:
            final_results.append(bm25_hit)

    # --- 4. 执行检索 ---
    if not final_results:
        print("在候选节点中，没有找到与关键词也匹配的结果。")

    for i, res in enumerate(final_results):
        print(f"[{i + 1}] BM25 Score: {res.score:.4f} (关键词匹配分数)")
        print(f"    DOC ID: {res.node.id_}")
        print(f"    Text: {res.node.text[:150].strip()}...")
        print("-" * 20)

    return final_results


if __name__ == "__main__":
    EMBEDDING_MODEL_PATH = r"D:\yechuan\work\cjsx\model\Qwen3-Embedding-0.6B"
    api_key = "ff6acab6-c747-49d7-b01c-2bea59557b8d"
    COLLECTION = "financial_reports"

    # 示例 1: 元数据筛选 (AND 逻辑) + 语义搜索
    print("\n--- 示例 1: 语义搜索 + 复合筛选 (AND) ---")
    complex_filters = [
        {"field": "institution", "operator": "in", "value": ["中金公司", "华泰证券"]},
        {"field": "report_type", "operator": "like", "value": "%宏观%"},
        {"field": "date", "operator": ">=", "value": "2024-01-01"}
    ]
    results1 = query_milvus(
        collection_name=COLLECTION,
        filters=complex_filters,
        logic_operator="and",
        query_text="针对中国房地产政策，有哪些不足和可以改善的地方",
        embedding_model_path=EMBEDDING_MODEL_PATH,
        top_k=5,
        search_threshold=0.4  # 假设IP相似度阈值为0.6
    )
    for res in results1:
        print(res)

    # 示例 2: 仅元数据筛选 (OR 逻辑)
    print("\n--- 示例 2: 仅元数据筛选 (OR) ---")
    or_filters = [
        {"field": "institution", "operator": "==", "value": "华创证券"},
        {"field": "title", "operator": "like", "value": "%宏观%"}
    ]
    results2 = query_milvus(
        collection_name=COLLECTION,
        filters=or_filters,
        logic_operator="or",
        top_k=5
    )
    for res in results2:
        print(res)

    # 示例 3: 仅语义搜索，无元数据筛选
    print("\n--- 示例 3: 仅语义搜索 ---")
    results3 = query_milvus(
        collection_name=COLLECTION,
        query_text="美债对美国CPI影响",
        embedding_model_path=EMBEDDING_MODEL_PATH,
        top_k=3,
        search_threshold=0.5
    )
    for res in results3:
        print(res)

    #示例 : 混合搜索(BM25)
    print("\n--- 示例 4: 混合搜索(BM25)")
    results4 = run_hybrid_search(
        collection_name=COLLECTION,
            query_text="针对中国房地产政策，有哪些改善的地方",
        embedding_model_path=EMBEDDING_MODEL_PATH,
        logic_operator="and",
        filters=complex_filters,
        top_k=5
    )
    for res in results4:
        print(res)