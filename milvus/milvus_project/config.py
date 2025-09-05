import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

MD_DIR = r"D:\yechuan\work\cjsx\RAG\report"
MD_TEST_DIR = r"D:\yechuan\work\cjsx\Docker_milvus\milvus\report_test"

TOOL_CALL_MODEL = "ep-20250826172947-ntwcf"
CHAT_MODEL = "ep-20250422130700-hfw6r"

EMBEDDING_MODEL_PATH = r"D:\yechuan\work\cjsx\model\Qwen3-Embedding-0.6B"
RERANK_MODEL_PATH = r"D:\yechuan\work\cjsx\model\bge-reranker-large"

# 火山
API_KEY = "ff6acab6-c747-49d7-b01c-2bea59557b8d"
VOL_URI = "https://ark.cn-beijing.volces.com/api/v3"

# 阿里
ALI_API_KEY = "sk-390066d0ba8745cd94817d83668f0440"
QWEN_MODEL = "qwen3-235b-a22b-thinking-2507"
QWEN_URI = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MILVUS_URI = "http://localhost:19530"
MILVUS_COLLECTION = "financial_reports"
MILVUS_DB_NAME = "default"

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "yc123456"
NEO4J_DATABASE = "neo4j"

# Global variables to store the loaded models
_embed_tokenizer = None
_embed_model = None
_embedding_model = None


def get_embedding_model():
    global _embed_tokenizer, _embed_model, _embedding_model

    if _embed_tokenizer is None or _embed_model is None or _embedding_model is None:
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model_path = EMBEDDING_MODEL_PATH
        # Load tokenizer
        _embed_tokenizer = AutoTokenizer.from_pretrained(
            embedding_model_path,
            local_files_only=True
        )

        # Load model
        _embed_model = AutoModel.from_pretrained(
            embedding_model_path,
            local_files_only=True
        ).to(device)

        # Set up HuggingFaceEmbedding
        _embedding_model = HuggingFaceEmbedding(model_name=embedding_model_path)

        print(f"Loaded tokenizer, model, and embedding from {embedding_model_path} on {device}")

    return _embed_model, _embed_tokenizer, _embedding_model


_rerank_tokenizer = None
_rerank_model = None
def get_reranker_model():
    global _rerank_model, _rerank_tokenizer

    if _rerank_model is None or _rerank_tokenizer is None :
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rerank_model_path = RERANK_MODEL_PATH
        # Load tokenizer
        _rerank_tokenizer = AutoTokenizer.from_pretrained(
            RERANK_MODEL_PATH,
        )

        # Load model
        _rerank_model = AutoModelForSequenceClassification.from_pretrained(
            RERANK_MODEL_PATH,
        ).to(device)
        _rerank_model.eval()
        print(f"Loaded reranker tokenizer, model, from {rerank_model_path} on {device}")

    return _rerank_model, _rerank_tokenizer