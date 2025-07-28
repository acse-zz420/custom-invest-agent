import numpy as np
import time
from pymilvus import Collection, connections, utility
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Any
from milvus_construct import get_sentence_embedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

def normalize_vector(vector: np.ndarray) -> List[float]:
    """归一化向量以支持 IP 度量，使其等效于余弦相似度。"""
    if np.linalg.norm(vector) == 0:
        return vector.tolist()
    return (vector / np.linalg.norm(vector)).tolist()


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

        # 对字符串值进行处理，避免注入和语法错误
        if op in ["==", "!="]:
            if isinstance(value, str):
                expressions.append(f'{field} {op} "{value}"')
            else:  # for numbers or booleans
                expressions.append(f'{field} {op} {value}')
        elif op == "like":
            expressions.append(f'{field} like "{value}"')
        elif op in ["in", "not in"]:
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                value_list = ",".join(f'"{v}"' for v in value)
                expressions.append(f"{field} {op} [{value_list}]")
            elif isinstance(value, list):  # for lists of numbers
                value_list = ",".join(str(v) for v in value)
                expressions.append(f"{field} {op} [{value_list}]")
        elif op in [">", ">=", "<", "<="]:
            if isinstance(value, str):
                expressions.append(f'{field} {op} "{value}"')
            else:
                expressions.append(f'{field} {op} {value}')

        elif op == "like_any":
            # 处理自定义的 'like_any' 操作符
            if isinstance(value, list) and value:
                sub_expressions = [f'{field} like "%{v}%"' for v in value]
                # 用 "or" 将所有子表达式连接起来，并用括号包围
                # 例如：(industry like "%A%" or industry like "%B%")
                or_expression = " or ".join(sub_expressions)
                expressions.append(f"({or_expression})")

    if not expressions:
        return None

    separator = f" {logic_operator} "
    return separator.join(f"({expr})" for expr in expressions)


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
    total_start_time = time.time()

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
        total_time = time.time() - total_start_time
        print(f"\n--- 总处理耗时: {total_time:.2f} 秒 ---")
        print("Milvus 连接已断开。")

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

        # 使用迭代器安全地获取所有实体
        iterator = collection.query_iterator(expr="id != 0", output_fields=output_fields)
        while True:
            batch = iterator.next()
            if not batch:
                break
            for entity in batch:
                node = TextNode(
                    id_=entity.pop("id"),  # LlamaIndex节点需要id_
                    text=entity.pop("text"),
                    metadata=entity  # 剩下的所有字段都作为元数据
                )
                nodes.append(node)
        return nodes
    finally:
        if "default" in connections.list_connections():
            connections.disconnect()





def run_full_hybrid_search(
        collection_name: str,
        query_text: str,
        embedding_model_path: str,
        filters: Optional[List[Dict]] = None,
        logic_operator: str = "and",
        top_k: int = 10
):
    """
    执行一个完整的混合搜索：语义+关键词+元数据过滤。
    """
    print("--- 启动完整混合搜索流程 ---")

    # 1. 初始化 BM25 检索器
    all_nodes = get_all_nodes_from_milvus(collection_name, "localhost", "19530")
    if not all_nodes:
        print("无法从Milvus获取节点，无法初始化BM25，流程终止。")
        return
    bm25_retriever = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=top_k)
    print("BM25 检索器初始化完成。")

    # 2. 初始化我们的自定义检索器（语义+元数据过滤）
    milvus_filter_retriever = MilvusFilterRetriever(
        collection_name=collection_name,
        embedding_model_path=embedding_model_path,
        filters=filters,
        logic_operator=logic_operator,
        top_k=top_k
    )
    print("自定义Milvus检索器初始化完成。")

    # 3. 使用 QueryFusionRetriever 将两者结合
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[milvus_filter_retriever, bm25_retriever],
        mode="reciprocal_rerank"  # 推荐的融合策略
    )
    print("混合检索器创建完成。")

    # 4. 执行检索
    print(f"\n正在执行混合检索，查询: '{query_text}'")
    final_results = hybrid_retriever.retrieve(query_text)

    # 5. 打印结果
    print("\n--- 混合搜索最终结果 ---")
    if not final_results:
        print("未找到任何结果。")
    for i, res in enumerate(final_results):
        print(f"[{i + 1}] Score: {res.score:.4f}")
        print(f"    Text: {res.text[:150]}...")
        # 你可以打印元数据来验证来源
        # print(f"    Metadata: {res.node.metadata}")
        print("-" * 20)

    return final_results



if __name__ == "__main__":
    EMBEDDING_MODEL_PATH = r"D:\yechuan\work\cjsx\model\Qwen3-Embedding-0.6B"
    COLLECTION = "financial_reports"

    # 示例 1: 复杂的元数据筛选 (AND 逻辑) + 语义搜索
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