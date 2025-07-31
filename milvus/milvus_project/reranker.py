from rag_milvus.config import get_reranker_model
import torch
from typing import List, Dict


def rerank_results(results: List[Dict], query: str, model, tokenizer) -> List[Dict]:
    """使用 reranker 模型对结果重新排序"""
    scored_results = []

    with torch.no_grad():  # 禁用梯度计算
        for result in results:
            if "text" not in result or "message" in result:
                continue  # 跳过无效结果（如 {"message": "无"}）

            # 准备输入：query 和 text 的配对
            inputs = tokenizer(
                query,
                result["text"],
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )

            # 获取 reranker 分数
            outputs = model(**inputs)
            score = torch.sigmoid(outputs.logits).item()  # 转换为 0-1 之间的分数

            # 添加 reranker 分数到结果
            result["reranker_score"] = score
            scored_results.append(result)

    # 按 reranker 分数降序排序
    scored_results = sorted(scored_results, key=lambda x: x["reranker_score"], reverse=True)

    return scored_results