import numpy as np
from tool import timer
from typing import Dict, List, Optional, Any
from milvus_construct import get_sentence_embedding
from pymilvus import MilvusClient, connections, Collection, utility
from llama_index.core.schema import TextNode
from config import *

model, tokenizer, _= get_embedding_model()

def normalize_vector(vector: np.ndarray) -> List[float]:
    """归一化向量以支持 IP 度量，使其等效于余弦相似度。"""
    if np.linalg.norm(vector) == 0:
        return vector.tolist()
    return (vector / np.linalg.norm(vector)).tolist()


@timer
def _build_filter_expression(filters: Optional[List[Dict]]) -> Optional[str]:
    """
    将列表形式的过滤条件转换为能正确访问 JSON 动态字段的 Milvus 布尔表达式。

    所有条件默认使用 "and" 连接。
    """
    if not filters:
        return None

    expressions = []
    for f in filters:
        field = f.get("field")
        op = f.get("operator")
        value = f.get("value")

        if not field or not op:
            raise ValueError(f"过滤条件格式错误，缺少 'field' 或 'operator': {f}")

        # 所有元数据都存储在名为 'metadata' 的JSON字段下
        milvus_field = f'metadata["{field}"]'

        expr = ""
        # 1. 操作符: ==, !=, >, >=, <, <=
        if op in ["==", "!=", ">", ">=", "<", "<="]:
            if isinstance(value, str):
                expr = f"{milvus_field} {op} '{value}'"
            else:
                expr = f"{milvus_field} {op} {value}"

        # 2. 操作符: like
        elif op == "like":
            if not isinstance(value, str):
                raise ValueError(f"'like' 操作符的值必须是字符串: {f}")
            expr = f"{milvus_field} like '{value}'"

        # 3. 操作符: in, not in
        elif op in ["in", "not in"]:
            if not isinstance(value, list):
                raise ValueError(f"'{op}' 操作符的值必须是列表: {f}")

            if not value:
                expr = "false" if op == "in" else "true"
            else:
                formatted_values = [f"'{v}'" if isinstance(v, str) else str(v) for v in value]
                value_list_str = ",".join(formatted_values)
                expr = f"{milvus_field} {op} [{value_list_str}]"

        # 4. 自定义操作符: like_any
        elif op == "like_any":
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError(f"'like_any' 操作符的值必须是字符串列表: {f}")

            if not value:
                expr = "false"
            else:
                sub_expressions = [f"{milvus_field} like '%{v}%'" for v in value]
                or_expression = " or ".join(sub_expressions)
                expr = f"({or_expression})"

        else:
            raise ValueError(f"不支持的运算符: '{op}'")

        expressions.append(expr)

    if not expressions:
        return None

    return " and ".join(expressions)

@timer
def query_milvus(
        collection_name: str,
        filters: Optional[List[Dict]] = None,
        query_text: Optional[str] = None,
        embedding_model_path: Optional[str] = None,
        top_k: int = 10,
        search_threshold: float = 0.5,
        uri: str = ZILLIZ_URI,
        token: str = ZILLIZ_TOKEN,
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
        uri (str): Zilliz uri。
        token (str): Zilliz token。

    Returns:
        List[Dict[str, Any]]: 查询结果列表，或在无结果时返回空列表。
    """

    if query_text and not embedding_model_path:
        raise ValueError("当提供 query_text 进行语义搜索时，必须指定 embedding_model_path。")

    print(f"开始连接 Milvus server ")
    connections.connect(alias="default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    print("Milvus 连接成功。")

    try:
        if not utility.has_collection(collection_name):
            print(f"错误：集合 '{collection_name}' 不存在。")
            return []

        collection = Collection(collection_name)
        collection.load()

        # 构建元数据筛选表达式
        filter_expr = _build_filter_expression(filters)
        print(f"生成的筛选表达式 (expr): {filter_expr}")

        # 获取所有字段名用于输出，除了向量字段
        fields_to_exclude = {"embedding", "sparse_embedding"}
        output_fields = [field.name for field in collection.schema.fields if field.name not in fields_to_exclude ]

        # --- 根据是否有 query_text 决定执行逻辑 ---
        if query_text:
            # 一：语义搜索 (可结合元数据筛选)
            print(f"正在执行语义搜索: '{query_text}'")

            query_embedding_raw = get_sentence_embedding(query_text, model, tokenizer)
            query_embedding = normalize_vector(query_embedding_raw)

            search_params = {"metric_type": "IP", "params": {"M": 16, "efConstruction": 200}}

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
                    record = hit.entity
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
def get_all_nodes_from_milvus(collection_name: str, uri: str, token: str) -> List[TextNode]:
    """从Milvus中获取所有文档，并转换为LlamaIndex的TextNode对象。"""
    print("正在从 Milvus 加载所有节点以初始化BM25...")
    nodes = []
    try:
        connections.connect( uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
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
def bm25_enhanced_search(
        collection_name: str,
        query_text: str,
        dense_embedding_function: object,  # 密集向量编码器实例
        embedding_model_path: str,
        top_k: int = 10,
        filters: Optional[List[Dict]] = None,
        uri: str = ZILLIZ_URI,
        token: str = ZILLIZ_TOKEN,
        pk_field: str = "doc_id",
        text_field: str = "text",
        metadata_field:  str = "metadata"
):
    """
    使用 PyMilvus 原生稀疏向量功能执行带有元数据过滤的混合搜索。

    此函数执行两次独立的、经过预过滤的搜索（一次密集，一次稀疏），
    然后合并结果，以分别显示语义分数和BM25分数。

    Args:
        collection_name (str): Milvus 集合名称。
        query_text (str): 用户查询。
        dense_embedding_function: 密集向量编码器实例。
        embedding_model path: 密集向量编码器本地路径
        filters (Optional[str]): Milvus 的布尔表达式，用于在搜索前过滤数据。
            例如: "category == 'tech' and year > 2023"
            这个过滤器会应用到语义搜索和关键词搜索两个阶段。
    """
    print("--- 启动 PyMilvus 原生混合搜索流程 (稀疏向量模型 + 过滤器) ---")

    # --- 1. 初始化连接和功能组件 ---
    try:
        client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
        if not connections.has_connection("default"):
            connections.connect("default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
            print("Milvus 连接成功。")
        collection = Collection(collection_name)
    except Exception as e:
        print(f"连接或加载 Milvus 集合时出错: {e}")
        return []


    # --- 生成查询向量 ---
    print(f"查询文本: '{query_text}'")
    filter_expr = _build_filter_expression(filters)
    if filters:
        print(f"应用过滤器: '{filter_expr}'")

    # a) 生成密集向量
    dense_query_vector = get_sentence_embedding(query_text, model, tokenizer)

    # --- 执行两次独立的、带过滤的搜索 ---

    # a) 密集向量（语义）搜索
    print("\n--- 阶段 1: 执行经过滤的密集（语义）搜索 ---")
    dense_results = client.search(
        data=[normalize_vector(dense_query_vector)],
        anns_field="embedding",
        limit=top_k,
        filter=filter_expr,
        search_params={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
        collection_name=collection_name,
        output_fields=[pk_field, text_field, metadata_field]
    )
    print(f"在满足过滤器的文档中，密集搜索找到 {len(dense_results[0])} 个语义相关结果。")

    # b) 稀疏向量（关键词）搜索
    print("\n--- 阶段 2: 执行经过滤的稀疏（BM25）搜索 ---")
    sparse_results = client.search(
        data=[query_text],  # 直接传入原始查询文本
        anns_field="sparse_embedding",
        limit=top_k,
        filter=filter_expr,
        search_params={"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}},
        collection_name=collection_name,
        output_fields=[pk_field, text_field, metadata_field]
    )
    print(f"在满足过滤器的文档中，BM25搜索找到 {len(sparse_results[0])} 个关键词匹配结果。")

    # --- 4. 合并并展示结果 ---x
    print("\n--- 阶段 3: 合并和格式化两次搜索的结果 ---")

    final_results: Dict[str, Dict] = {}

    # 处理密集搜索结果
    for hit in dense_results[0]:
        doc_id = hit["doc_id"]
        entity_data = hit.get("entity", {})
        final_results[doc_id] = {
            "doc_id": doc_id,
            "text": hit["entity"][text_field],
            "semantic_score": hit["distance"],
            "bm25_score": 0.0,
            "found_by": ["semantic"],
            "metadata": entity_data

        }

    # 处理并合并稀疏搜索结果
    for hit in sparse_results[0]:
        doc_id = hit["doc_id"]
        bm25_score = hit["distance"]
        entity_data = hit.get("entity", {})

        if doc_id in final_results:
            final_results[doc_id]["bm25_score"] = bm25_score
            final_results[doc_id]["found_by"].append("bm25")
        else:
            final_results[doc_id] = {
                "doc_id": doc_id,
                "text": hit["entity"][text_field],
                "semantic_score": -1.0,
                "bm25_score": bm25_score,
                "found_by": ["bm25"],
                "metadata": entity_data
            }

    results_list = list(final_results.values())
    formatted_results = []
    for res in results_list:
        file_name = res.get("metadata", {}).get("metadata", {}).get("file_name", "Unknown File")

        # 构建只包含所需字段的新字典
        clean_result = {
            "doc_id": res.get("doc_id"),
            "text": res.get("text"),
            "file_name": file_name,
            "semantic_score": res.get("semantic_score", 0.0),
            "bm25_score": res.get("bm25_score", 0.0)
        }
        formatted_results.append(clean_result)

        # 3. 对格式化后的结果进行排序
    sorted_results = sorted(
        formatted_results,  # <--- 对新的、干净的列表进行排序
        key=lambda x: x["bm25_score"] + x["semantic_score"],
        reverse=True
    )

    print("\n--- 混合搜索完成，最终结果 (所有结果均满足过滤器条件): ---")
    if not sorted_results:
        print("没有找到任何满足所有条件的结果。")

    for i, res in enumerate(sorted_results):
        print(f"File_name: {res['file_name']})")
        print(f"    Semantic Score (IP distance): {res['semantic_score']:.4f}")
        print(f"    BM25 Score: {res['bm25_score']:.4f}")
        print(f"    Text: {res['text'][:150].strip() if res['text'] else 'N/A'}...")
        print("-" * 20)

    return sorted_results


if __name__ == "__main__":

    COLLECTION = "financial_reports"

    # 示例 1: 元数据筛选  + 语义搜索
    print("\n--- 示例 1: 语义搜索 + 元数据过滤 ---")
    complex_filters = [
        {"field": "institution", "operator": "in", "value": ["中金公司", "华泰证券"]},
        {"field": "report_type", "operator": "like", "value": "%宏观%"},
        {"field": "date", "operator": ">=", "value": "2024-01-01"}
    ]
    results1 = query_milvus(
        collection_name=COLLECTION,
        filters=complex_filters,
        query_text="针对中国房地产政策，有哪些不足和可以改善的地方",
        embedding_model_path=EMBEDDING_MODEL_PATH,
        top_k=5,
        search_threshold=0.4  # 假设IP相似度阈值为0.6
    )
    for res in results1:
        print(res)

    # 示例 2: 仅元数据筛选
    print("\n--- 示例 2: 仅元数据筛选 ---")
    or_filters = [
        {"field": "institution", "operator": "==", "value": "华创证券"},
        {"field": "title", "operator": "like", "value": "%宏观%"}
    ]
    results2 = query_milvus(
        collection_name=COLLECTION,
        filters=or_filters,
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

    results4 = bm25_enhanced_search(
        collection_name=COLLECTION,
        query_text="针对房地产政策，有哪些可以完善的地方",
        dense_embedding_function=get_sentence_embedding,
        embedding_model_path=EMBEDDING_MODEL_PATH,
        filters=complex_filters,
        top_k=5
    )
    for res in results4:
        print(res)