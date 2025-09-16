from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from milvus import MilvusServer
from pathlib import Path
from typing import List, Tuple, Literal, Optional
import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
from llm import VolcengineLLM
from hybrid_chunking import custom_chunk_pipeline
import hashlib
import re
from datetime import datetime
from  prompt import chunk_extract_prompt

def get_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def get_sentence_embedding(text: str, embedding_model, embedding_tokenizer) -> List[float]:
    with torch.no_grad():
        inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        outputs = embedding_model(**inputs)

        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask

        return embedding.squeeze().tolist()



def extract_info_from_filename(filename: str) -> Optional[dict]:
    stem = Path(filename).stem
    # 解析文件名格式，例如：20250430-国金证券-宏观经济点评：川式化债，越化越糟？
    match = re.match(r"(\d{8})-([\u4e00-\u9fa5A-Za-z]+)-(.+?)[:：](.+)", stem)
    if not match:
        print(f"无法解析文件名：{filename}")
        return None

    date_str, company, report_type_raw, title = match.groups()
    try:
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        date_fmt = date_obj.strftime("%Y-%m-%d")
    except Exception:
        print(f"日期解析失败：{date_str}")
        return None

    def map_report_type(raw_type: str) -> str:
        for key in ["宏观", "行业", "公司", "策略", "基金", "专题"]:
            if key in raw_type:
                return key + "研究" if key != "公司" else "公司点评"
        return "其他"

    report_type = map_report_type(report_type_raw)

    file_id = get_md5(filename)

    return {
        "file_id": file_id,
        "date": date_fmt,
        "institution": company,
        "title": title.strip(),
        "report_type": report_type,
    }



def try_extract_json(text: str) -> Optional[dict]:
    try:
        # 尝试直接加载
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试从 markdown 中提取 ```json ... ```
        match = re.search(r"```json\n(.*?)```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass

        # 尝试从字符串中提取 { ... } 部分
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass

    return None



def save_to_milvus(
        data: List[dict],
        collection_name: str = "financial_report",
        host: str = "localhost",
        port: str = "19530"
):
    # 连接 Milvus server
    print(f"开始连接 Milvus server at {host}:{port}...")
    connections.connect(alias="default", host=host, port=port)
    print("Milvus 连接成功。")

    try:
        # 检查集合是否存在
        print("写入前collections:", utility.list_collections())
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"已删除原有 collection: {collection_name}")

        # 创建集合模式
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="chunk_embedding", dtype=DataType.FLOAT_VECTOR, dim=len(data[0]["chunk_embedding"])),
            FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="institution", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=300),
            FieldSchema(name="report_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="date_range", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="target_entity", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="industry", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="authors", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="fund_codes", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="fund_names", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="topic", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=1000),
        ]

        schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
        collection = Collection(name=collection_name, schema=schema)
        print(f"集合 '{collection_name}' 创建成功。")

        # 准备插入数据
        insert_data = []
        for idx, chunk in enumerate(data):
            # 构建每条记录，包含固定字段和动态字段
            embedding = np.array(chunk["chunk_embedding"])
            embedding = embedding / np.linalg.norm(embedding)
            record = {
                "chunk_id": chunk["chunk_id"],
                "chunk_text": chunk["chunk_text"],
                "chunk_embedding": chunk["chunk_embedding"],
                "file_id": chunk["file_id"],
                "date": chunk.get("date", ""),
                "institution": chunk.get("institution", ""),
                "title": chunk.get("title", ""),
                "report_type": chunk.get("report_type", ""),
                "date_range": chunk.get("date_range", ""),
                "target_entity": chunk.get("target_entity", ""),
                "industry": chunk.get("industry", ""),
                "authors": chunk.get("authors", ""),
                "fund_codes": chunk.get("fund_codes", ""),
                "fund_names": chunk.get("fund_names", ""),
                "topic": chunk.get("topic", ""),
                "keywords": chunk.get("keywords", ""),
            }

            # 添加动态字段
            for key in chunk:
                if key not in ["chunk_id", "chunk_text", "chunk_embedding", "file_id"]:
                    record[key] = chunk[key]

            insert_data.append(record)

            if (idx + 1) % 50 == 0:
                print(f"已添加 {idx + 1} 条数据...")
                # 插入批量数据
                collection.insert(insert_data)
                insert_data = []  # 清空批量数据

        # 插入剩余数据
        if insert_data:
            print(f"插入剩余 {len(insert_data)} 条数据")
            collection.insert(insert_data)

        # 确保数据持久化
        collection.flush()
        print("数据插入成功。")

        # 创建索引
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="chunk_embedding", index_params=index_params)
        collection.load()
        print("索引创建并加载成功。")

        print("写入后collections:", utility.list_collections())
        print(f"✅ Mil.third_party集合 `{collection_name}` 已保存 {len(data)} 个 chunk")

    finally:
        # 断开 Milvus 连接
        connections.disconnect("default")
        print("Milvus 连接已断开。")

def process_all_md_files(
    md_dir: str,
    chunk_prompt: str,
    embedding_model_path: str,
    collection_name: str = "financial_report"
):
    server = MilvusServer()
    server.set_base_dir("D:/yechuan/work/cjsx/RAG/milvus_data")  # 指定存储路径
    server.start()
    print("MilvusServer 已启动，数据存储路径: D:/yechuan/work/cjsx/RAG/milvus_data")

    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_path, local_files_only=True)
    embedding_model = AutoModel.from_pretrained(embedding_model_path, local_files_only=True)

    llm = VolcengineLLM(api_key="ff6acab6-c747-49d7-b01c-2bea59557b8d")
    all_chunks = []

    for file in Path(md_dir).glob("*.md"):
        meta = extract_info_from_filename(file.name)
        if not meta:
            print(f"跳过文件 {file.name}")
            continue

        content = file.read_text(encoding="utf-8")
        content = re.sub(r"\n{2,}", "\n", content).strip()
        file_id = meta["file_id"]

        full_prompt = chunk_prompt.format(full_report_text=content)
        try:
            full_struct_raw = llm.complete(full_prompt).text.strip()
            print(f"结构抽取原始输出（{file.name}）:\n{full_struct_raw}\n")

            full_struct = try_extract_json(full_struct_raw)
            if full_struct is None:
                print(f"❌ 无法从 LLM 输出中解析结构化 JSON：{file.name}")
                continue
        except Exception as e:
            print(f"提取结构信息失败: {file.name}，错误: {e}")
            continue

        chunks: List[Tuple[int, str]] = custom_chunk_pipeline(content)
        valid_chunks = [chunk for chunk in chunks if chunk[0] == 0]  # 过滤 is_table==0 的chunk

        for i, chunk_tuple in enumerate(valid_chunks):
            chunk_text = chunk_tuple[1]  # 改成1而不是2，取文本部分

            chunk_id = f"{file_id}_{i}"
            chunk_embedding = get_sentence_embedding(chunk_text, embedding_model, embedding_tokenizer)

            all_chunks.append({
                "chunk_id": chunk_id,
                "file_id": file_id,
                "chunk_text": chunk_text,
                **meta,
                **full_struct,
                "chunk_embedding": chunk_embedding,
            })
    # # 保存所有 chunks 到 JSON 文件，路径和名称你可以自定义
    # json_output_path = Path(md_dir) / "all_chunks.json"
    # with open(json_output_path, "w", encoding="utf-8") as f:
    #     json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    #
    # print(f"✅ 所有 chunk 已保存到文件: {json_output_path}")
    save_to_milvus(all_chunks, collection_name=collection_name)


if __name__ == "__main__":
    embed_model_path= r"D:\yechuan\work\cjsx\model\Qwen3-Embedding-0.6B"
    md_folder = r"D:\yechuan\work\cjsx\RAG\milvus_project\report"
    collection_name = "financial_report"

    process_all_md_files(
        md_folder,
        chunk_extract_prompt,
        embed_model_path,
        collection_name
    )
