from llama_index.core.schema import TextNode, NodeWithScore
from pymilvus import Collection, connections, utility
import time



from milvus_filter import query_milvus
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.query_bundle import QueryBundle
from typing import Any, List, Dict, Optional


class MilvusFilterRetriever(BaseRetriever):
    """一个自定义的检索器，用于包装强大的 query_milvus 函数。"""

    def __init__(
            self,
            collection_name: str,
            embedding_model_path: str,
            filters: Optional[List[Dict]] = None,
            logic_operator: str = "and",
            top_k: int = 10,
            search_threshold: float = 0.5,
            **kwargs: Any,
    ) -> None:
        self._collection_name = collection_name
        self._embedding_model_path = embedding_model_path
        self._filters = filters
        self._logic_operator = logic_operator
        self._top_k = top_k
        self._search_threshold = search_threshold
        super().__init__(**kwargs)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # 调用已有的 query_milvus 函数
        results_dict = query_milvus(
            collection_name=self._collection_name,
            filters=self._filters,
            logic_operator=self._logic_operator,
            query_text=query_bundle.query_str,
            embedding_model_path=self._embedding_model_path,
            top_k=self._top_k,
            search_threshold=self._search_threshold,
        )

        # 将返回的字典列表转换为 NodeWithScore 列表
        nodes_with_scores = []
        if results_dict and "message" not in results_dict[0]:
            for res in results_dict:
                score = res.pop("similarity_score", 0.0)
                node = TextNode(
                    text=res.get("chunk_text", ""),
                    metadata=res
                )
                nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores