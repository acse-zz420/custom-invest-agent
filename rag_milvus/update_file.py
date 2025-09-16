import re
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

from llm import VolcengineLLM
from hybrid_chunking import custom_chunk_pipeline
from milvus_construct import get_md5,try_extract_json
from llama_index.core import (Document, Settings, StorageContext,
                              VectorStoreIndex, PromptTemplate)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import Collection, connections, utility
from transformers import AutoConfig
from tool import  timer
from prompt import chunk_extract_prompt_new

# --- 全局配置 (请根据你的环境修改) ---
MD_DIR = r"D:\yechuan\work\cjsx\RAG\report"
EMBEDDING_MODEL_PATH = r"D:\yechuan\work\cjsx\model\Qwen3-Embedding-0.6B"
VOLCENGINE_API_KEY = "ff6acab6-c747-49d7-b01c-2bea59557b8d"
COLLECTION_NAME = "financial_reports"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# 用于跟踪文件状态的JSON文件
STATE_FILE = Path("./data_processing_state.json")

# --- 辅助函数 ---
# 这是唯一需要修改的函数

from pymilvus import Collection, connections, utility


def get_file_ids_from_milvus(collection_name: str, host: str, port: str) -> set:
    """
    从Milvus集合中获取所有唯一的 file_id。

    Returns:
        一个包含所有 file_id 字符串的集合 (set)。
    """
    print("正在从 Milvus 获取所有已存在的 file_id...")

    if "default" in connections.list_connections():
        connections.disconnect("default")

    try:
        connections.connect(host=host, port=port)

        if not utility.has_collection(collection_name):
            print("集合不存在，返回空的 file_id 集合。")
            return set()

        # Collection() 在没有指定 using 参数时，会自动使用 "default" 连接
        collection = Collection(collection_name)
        collection.load()

        # 我们假设 file_id 不会为空
        results = collection.query(
            expr="file_id != ''",
            output_fields=["file_id"]
        )

        existing_ids = {item['file_id'] for item in results}
        print(f"在 Milvus 中找到 {len(existing_ids)} 个唯一的 file_id。")
        return existing_ids

    except Exception as e:
        print(f"从 Milvus 获取 file_id 时出错: {e}")
        return set()
    finally:
        if "default" in connections.list_connections():
            connections.disconnect("default")

def process_file_to_nodes(file_path: Path, prompt_template: PromptTemplate) -> List[Document]:
    """
    处理单个文件，返回LlamaIndex节点列表。
    """
    print(f"  -> 正在处理: {file_path.name}")

    file_id = get_md5(file_path.name)

    content = file_path.read_text(encoding="utf-8")
    content = re.sub(r"\n{2,}", "\n", content).strip()

    content_with_filename = f"文件名: {file_path.name}\n正文: {content}"
    formatted_prompt = prompt_template.format(full_report_text=content_with_filename)
    response = Settings.llm.complete(
        formatted_prompt, system_prompt="你是财经领域智能助手..."
    )
    full_struct = try_extract_json(response.text.strip())
    if full_struct is None:
        print(f"❌ 在 {file_path.name} 中无法解析JSON，跳过此文件。")
        return []

    chunks = custom_chunk_pipeline(content)
    valid_chunks = [c for c in chunks if c[0] == 0]

    nodes = []
    for i, (_, chunk_text) in enumerate(valid_chunks):
        nodes.append(Document(
            text=chunk_text,
            metadata={
                "file_name": file_path.name,
                "file_id": file_id,
                **full_struct
            }
        ))
    return nodes

#--场景一：md文件的更新与删减--
@timer
def run_sync_by_file_id():
    """
    通过对比本地文件ID和Milvus中已存在的ID来处理文件的增/删。
    此方法不关心文件内容是否变化
    """
    print("--- 场景一：启动基于 file_id 的同步流程 ---")
    start_time = time.time()

    Settings.llm = VolcengineLLM(api_key=VOLCENGINE_API_KEY)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_PATH)
    config = AutoConfig.from_pretrained(EMBEDDING_MODEL_PATH, local_files_only=True)

    vector_store = MilvusVectorStore(
        uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}",
        collection_name=COLLECTION_NAME,
        dim=config.hidden_size,
        overwrite=False,
        enable_dynamic_field=True
    )
    index = VectorStoreIndex.from_vector_store(vector_store)

    # --- 2. 获取本地和Milvus的状态 ---
    local_file_map = {
        get_md5(p.name): p for p in Path(MD_DIR).rglob("*.md")
    }
    local_ids = set(local_file_map.keys())
    print(f"在本地发现 {len(local_ids)} 个文件。")

    milvus_ids = get_file_ids_from_milvus(COLLECTION_NAME, MILVUS_HOST, MILVUS_PORT)

    # --- 3. 计算差异 ---
    ids_to_add = local_ids - milvus_ids
    ids_to_delete = milvus_ids - local_ids
    print(f"差异计算完成: {len(ids_to_add)} 个新增, {len(ids_to_delete)} 个删除。")

    # --- 4. 执行删除操作---
    if ids_to_delete:
        print("\n--- 正在从Milvus删除节点 ---")
        try:
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            collection = Collection(COLLECTION_NAME)

            id_list_str = ",".join(f'"{id_str}"' for id_str in ids_to_delete)
            delete_expr = f"file_id in [{id_list_str}]"

            print(f"  - 准备执行批量删除，表达式: {delete_expr}")

            res = collection.delete(expr=delete_expr)
            print(f"  - 已提交批量删除请求，影响的主键数量: {res.delete_count}")

            collection.flush()

        except Exception as e:
            print(f"删除操作时出错: {e}")
        finally:
            if "delete_ops" in connections.list_connections():
                connections.disconnect()

    # --- 5. 执行新增操作 ---
    if ids_to_add:
        print("\n--- 正在处理并插入新节点 ---")
        from prompt import chunk_extract_prompt_new  # 初始化的prompt
        prompt = PromptTemplate(chunk_extract_prompt_new)

        all_new_nodes = []
        for file_id in ids_to_add:
            file_path = local_file_map[file_id]
            # 调用你原有的文件处理函数
            # 注意：确保 process_file_to_nodes 内部的 file_id 生成逻辑一致
            nodes = process_file_to_nodes(file_path, prompt)
            all_new_nodes.extend(nodes)

        if all_new_nodes:
            index.insert_nodes(all_new_nodes, show_progress=True)
            print(f"成功插入 {len(all_new_nodes)} 个新节点。")

    if not ids_to_add and not ids_to_delete:
        print("\n本地与Milvus数据一致，无需操作。")

    print(f"--- 基于 file_id 的同步完成，耗时: {time.time() - start_time:.2f} 秒 ---")


# --- 场景二：元数据刷新 ---
@timer
def run_metadata_refresh(new_prompt_str:str):
    """仅刷新元数据，不重新计算embedding"""
    print("--- 场景二：启动元数据刷新流程 ---")
    start_time = time.time()

    if not new_prompt_str:
        print("错误：执行元数据刷新必须通过 --prompt 参数提供一个新的prompt。")
        return

    Settings.llm = VolcengineLLM(api_key=VOLCENGINE_API_KEY)
    prompt_template = PromptTemplate(new_prompt_str)

    print(f"正在使用新Prompt刷新元数据...")

    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(COLLECTION_NAME)
        collection.load()

        iterator = collection.query_iterator(output_fields=["id", "text", "doc_id", "embedding"])

        upsert_data = []
        count = 0
        total_chunks = collection.num_entities
        print(f"共需处理 {total_chunks} 个 chunks.")

        while True:
            batch = iterator.next()

            if not batch:
                break  # 如果是空列表，跳出 while 循环

            # c. 如果批次不为空，则处理这一批数据
            count += len(batch)
            print(f"  - 正在处理批次，已处理 {count}/{total_chunks} 个chunk...")
            for chunk in batch:
                content_with_filename = f"文件名: {chunk.get('file_name', '未知')}\n正文: {chunk['text']}"
                formatted_prompt = prompt_template.format(full_report_text=content_with_filename)
                response = Settings.llm.complete(
                    formatted_prompt, system_prompt="你是财经领域智能助手..."
                )
                new_struct = try_extract_json(response.text.strip())

                if new_struct:
                    # 准备upsert数据：主键 + 新的元数据
                    upsert_data.append({"id": chunk["id"],"doc_id": chunk["doc_id"],"text": chunk["text"],"embedding":chunk["embedding"] ,**new_struct})
        # 2. 批量执行Upsert
        if upsert_data:
            print(f"\n准备向Milvus Upsert {len(upsert_data)} 条记录的元数据...")
            collection.upsert(upsert_data)
            collection.flush()
            print("元数据Upsert和Flush操作完成。")

    finally:
        connections.disconnect("metadata_refresh")

    print(f"--- 元数据刷新完成，耗时: {time.time() - start_time:.2f} 秒 ---")


#--场景三：根据指定的 doc_id 更新单个 chunk 的元数据。--
@timer
def run_batch_chunk_update(updates: Dict[str, Dict]):
    """
    场景四：批量更新多个Chunk的元数据（打补丁）。

    此函数高效地处理一个包含多个chunk更新请求的字典，
    通过一次查询和一次upsert完成所有操作。

    Args:
        updates (Dict[str, Dict]): 一个字典，键是目标 chunk 的 ID (doc_id),
                                   值是包含要新增或修改的键值对的"补丁"字典。
                                   示例:
                                   {
                                       "doc_id_1": {"authors": "张三"},
                                       "doc_id_2": {"institution": "中金公司", "date": "2025-5-31"}
                                   }
    """
    print("--- 场景三：批量Chunk元数据更新 ---")

    if not updates:
        print("提供了一个空的更新字典，无需操作。")
        return

    doc_ids_to_update = list(updates.keys())
    print(f"  - 准备更新 {len(doc_ids_to_update)} 个Chunk。")

    conn_alias = "default"
    try:
        # --- 1. 连接并获取集合 ---
        if conn_alias not in connections.list_connections():
            connections.connect(alias=conn_alias, host=MILVUS_HOST, port=MILVUS_PORT)

        if not utility.has_collection(COLLECTION_NAME):
            print(f"错误: 集合 '{COLLECTION_NAME}' 不存在。")
            return

        collection = Collection(COLLECTION_NAME)
        collection.load()

        # --- 2. 一次性批量查询所有相关的 Chunks ---
        query_expr = f'doc_id in {json.dumps(doc_ids_to_update)}'
        print(f"  - 正在批量查询 Milvus: {query_expr}")

        # 确保查询所有字段，以便能完整地重建它们
        results = collection.query(expr=query_expr, output_fields=["id", "doc_id", "embedding", "text", "metadata"])

        if not results:
            print("警告: 在Milvus中未找到任何提供的 doc_id。")
            return

        # --- 3. 在内存中批量处理并准备 Upsert 数据 ---
        upsert_batch = []
        found_ids = set()

        for original_chunk in results:
            doc_id = original_chunk['doc_id']
            found_ids.add(doc_id)

            # 获取对应的元数据补丁
            patch = updates[doc_id]

            current_metadata = original_chunk.get('metadata', {}) or {}
            print(f"  - 处理 {doc_id}:\n    - 旧元数据: {current_metadata}")

            # 应用补丁：这会修改或添加新值，同时保留旧值
            current_metadata.update(patch)
            print(f"    - 新元数据: {current_metadata}")

            # 构建完整的Upsert payload，必须包含所有字段
            upsert_payload = {
                "id": original_chunk["id"],  # 主键必须提供
                "doc_id": doc_id,
                "embedding": original_chunk["embedding"],
                "text": original_chunk["text"],
                "metadata": current_metadata  # 使用更新后的元数据
            }
            upsert_batch.append(upsert_payload)

        # --- 4. 一次性批量执行 Upsert 操作 ---
        if upsert_batch:
            print(f"\n  - 正在向 Milvus 批量 Upsert {len(upsert_batch)} 条记录...")
            res = collection.upsert(upsert_batch)
            collection.flush()
            print(f"  - 批量 Upsert 操作成功。影响的主键数量: {res.upsert_count}")

        # --- 5. 报告未找到的ID ---
        missing_ids = set(doc_ids_to_update) - found_ids
        if missing_ids:
            print(f"警告: 以下 {len(missing_ids)} 个 doc_id 在 Milvus 中未找到，已被忽略: {missing_ids}")

    except Exception as e:
        print(f"更新 chunks 时发生严重错误: {e}")
    finally:
        if conn_alias in connections.list_connections():
            connections.disconnect(conn_alias)

if __name__ == "__main__":

    # --- 选择你的运行模式 ---
    # 'sync'      : 用于文件内容同步 (增、删、改)
    # 'new_prompt'   : 用于刷新全部元数据
    # 'update_chunk'      : 更新特定chunk的metadata
    MODE = "update_chunk"

    # --- 如果是 'refresh' 模式，在这里提供你的新Prompt ---
    NEW_PROMPT_FOR_REFRESH = """
    这是我的新的、用于刷新元数据的Prompt。
    请根据以下文本内容，只提取报告的类型。
    JSON输出格式为: {"report_type": "提取的类型"}
    ---
    文本内容:
    {full_report_text}
    ---
    JSON输出:
     {{
      "report_type": "",
      }}
    """

    metadata_patch= {
        "578f3b5db32dec7185d0f77bedc2d967_2": {
            "file_name": "AB",
        },
        # "cbd731cf9b43a171fff209409dd35240_17": {
        #     "authors": "张三",
        #     "institution": "中金公司"
        # },
        # "id_that_does_not_exist": { # 这个ID将被报告为未找到
        #     "status": "failed"
        # }
    }
    # --- 主逻辑 ---
    if MODE == "sync":
        run_sync_by_file_id()
    elif MODE == "new_prompt":
        run_metadata_refresh(new_prompt_str=chunk_extract_prompt_new)
    elif MODE == "update_chunk":
        run_batch_chunk_update(metadata_patch)
    else:
        print(f"错误：未知的模式 '{MODE}'。请选择 'sync', 'refresh', 或 'update_chunk'。")