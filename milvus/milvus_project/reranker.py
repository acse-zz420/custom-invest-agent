from config import get_reranker_model
import torch
from typing import List, Dict
from llama_index.core.schema import NodeWithScore


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


def rerank_nodes(
        nodes: List[NodeWithScore],
        query: str,
        model,
        tokenizer,
        top_n: int = 10,
) -> List[NodeWithScore]:
    """
    使用 reranker 模型对 NodeWithScore 对象列表进行重新排序。

    Args:
        nodes (List[NodeWithScore]): 从初始检索中获得的节点列表。
        query (str): 用户的原始查询。
        model: 已加载的 reranker 模型。
        tokenizer: reranker 模型对应的 tokenizer。
        top_n (int): 重排后返回的 top N 个结果。

    Returns:
        List[NodeWithScore]: 经过重排并按新分数排序的 NodeWithScore 对象列表。
    """
    if not nodes:
        return []

    # 我们从每个 NodeWithScore 对象中提取出文本内容
    sentence_pairs = []
    for node_with_score in nodes:
        sentence_pairs.append([query, node_with_score.node.get_content()])

    print(f"  Reranker 正在为 {len(sentence_pairs)} 个节点对计算相关性分数...")

    with torch.no_grad():
        inputs = tokenizer(
            sentence_pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model(**inputs)
        scores = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()

    # 如果只有一个结果，确保 scores 是一个数组
    if scores.ndim == 0:
        scores = [scores.item()]

    # --- 将新分数更新回 NodeWithScore 对象 ---
    for node_with_score, new_score in zip(nodes, scores):
        # 我们将 reranker 的分数更新到 .score 属性中
        node_with_score.score = float(new_score)

    reranked_nodes = sorted(nodes, key=lambda x: x.score or 0.0, reverse=True)

    print(f"  重排完成。")

    # 返回分数最高的 top_n 个节点
    return reranked_nodes[:top_n]