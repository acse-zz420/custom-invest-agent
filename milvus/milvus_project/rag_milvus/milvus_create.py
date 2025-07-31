import re
import os
from pathlib import Path
from typing import List, Tuple


from tool import timer

# 本地模块
from llm import VolcengineLLM
from milvus_construct import get_md5, try_extract_json
from hybrid_chunking import custom_chunk_pipeline

# LlamaIndex 模块
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.base import BM25BuiltInFunction
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Transformers 模块
from transformers import AutoConfig
from pymilvus.model.sparse import BM25EmbeddingFunction
from prompt import chunk_extract_prompt_new

os.environ['MODELSCOPE_CACHE'] = 'D:/my_custom_cache/modelscope'

# 设置 ModelScope 数据集缓存子目录（可选）
os.environ['MODELSCOPE_DATASETS_CACHE'] = 'D:/my_custom_cache/modelscope/datasets'

# 设置 ModelScope 模型缓存子目录（可选）
os.environ['MODELSCOPE_MODELS_CACHE'] = 'D:/my_custom_cache/modelscope/models'

from config import *

config = AutoConfig.from_pretrained(EMBEDDING_MODEL_PATH, local_files_only=True)
embedding_dim = config.hidden_size
print(f"自动检测到嵌入维度: {embedding_dim}")

Settings.llm = VolcengineLLM(api_key=API_KEY)
_, _, Settings.embed_model = get_embedding_model()

@timer
def process_all_md_files(
        md_dir: str,
        chunk_prompt: str,
        embedding_model_path: str,
        collection_name: str = "financial_reports",
        host: str = "localhost",
        port: str = "19530",
        api_key: str = None
):
    """
    处理指定目录下的所有Markdown文件，提取、分块，并将结果通过LlamaIndex存入Milvus Standalone。

    支持稠密向量（embedding）和稀疏向量（sparse_embedding）

    Args:
        md_dir (str): Markdown文件所在的目录。
        chunk_prompt (str): 用于指示LLM提取结构化信息的提示模板。
        embedding_model_path (str): 本地HuggingFace嵌入模型的路径。
        collection_name (str, optional): 在Milvus中创建的集合名称。
        host (str, optional): Milvus Standalone服务的主机地址。
        port (str, optional): Milvus Standalone服务的端口。
        api_key (str, optional): Volcengine LLM的API密钥。

    Returns:
        VectorStoreIndex: 构建好的LlamaIndex索引对象。
    """


    bm25_function = BM25BuiltInFunction(
        input_field_names="text",
        output_field_names="sparse_embedding",
        analyzer_params={"type":"chinese"},
        enable_match=True  # 启用精确匹配
    )

    prompt = PromptTemplate(chunk_prompt)
    all_documents = []

    file_count = len(list(Path(md_dir).rglob("*.md")))
    print(f"发现 {file_count} 个 Markdown 文件，开始逐一处理...")

    for file_idx, file in enumerate(Path(md_dir).rglob("*.md")):
        print(f"\n[{file_idx + 1}/{file_count}] 正在处理文件: {file.name}")
        # 加载和预处理文本
        filename = file.name
        file_id = get_md5(filename)
        content = file.read_text(encoding="utf-8")
        content = re.sub(r"\n{2,}", "\n", content).strip()

        # LLM 推理以抽取整个文件的结构化元数据
        content_with_filename = f"文件名: {file.name}\n正文: {content}"
        formatted_prompt = prompt.format(full_report_text=content_with_filename)
        response = Settings.llm.complete(
            formatted_prompt,
            system_prompt="你是财经领域智能助手，提取结构化信息并输出 JSON。"
        )
        full_struct = try_extract_json(response.text.strip())
        if full_struct is None:
            print(f"无法从 LLM 输出中解析结构化 JSON：{file.name}")
            continue
        print(f"已从 {file.name} 提取结构化信息。")

        # 使用自定义逻辑进行分块
        chunks: List[Tuple[int, str]] = custom_chunk_pipeline(content)
        valid_chunks = [chunk for chunk in chunks if chunk[0] == 0]  # 过滤掉非文本块

        for i, (_, chunk_text) in enumerate(valid_chunks):
            chunk_id = f"{file_id}_{i}"
            # 构造 Document 节点
            node = Document(
                text=chunk_text,
                id_=chunk_id,
                metadata={
                    "metadata": {
                        "file_name": filename,
                        "file_id": file_id,
                        **full_struct  # 数据结构
                    },
                }
            )
            all_documents.append(node)

    print(f"\n--- 所有文件处理完毕，共生成 {len(all_documents)} 个文本块 (documents) ---")

    if not all_documents:
        print("没有生成任何文档，流程终止。")
        return None


    # 初始化 Milvus 向量存储
    print(f"正在连接到 Milvus: http://{host}:{port}")
    vector_store = MilvusVectorStore(
        uri=f"http://{host}:{port}",
        collection_name=collection_name,
        dim=embedding_dim,
        overwrite=True,
        enable_dense=True,
        embedding_field="embedding", #稠密索引
        enable_sparse=True,  # 启用稀疏向量支持
        sparse_embedding_field="sparse_embedding",
        sparse_embedding_function=bm25_function,
        index_config={"index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 200}},
        sparse_index_config={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25"},
        similarity_metric="IP"
    )

    print("MilvusVectorStore 初始化成功。")

    # 5. 构建索引并存入 Milvus
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        all_documents,
        storage_context=storage_context,
        show_progress=True
    )

    print(f"索引构建完成！集合 '{collection_name}' 已保存 {len(all_documents)} 个 chunk。")

    return index


if __name__ == '__main__':

    index_object = process_all_md_files(
        md_dir=MD_DIR,
        chunk_prompt= chunk_extract_prompt_new,
        embedding_model_path=EMBEDDING_MODEL_PATH,
        collection_name="financial_reports",
        api_key=API_KEY
    )