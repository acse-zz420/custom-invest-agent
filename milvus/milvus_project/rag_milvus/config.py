import torch
from transformers import AutoTokenizer, AutoModel
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


EMBEDDING_MODEL_PATH = r"D:\yechuan\work\cjsx\model\Qwen3-Embedding-0.6B"
API_KEY = "ff6acab6-c747-49d7-b01c-2bea59557b8d"
MD_DIR = r"D:\yechuan\work\cjsx\RAG\report"


# Global variables to store the loaded models
_tokenizer = None
_model = None
_embedding_model = None


def get_embedding_model():
    global _tokenizer, _model, _embedding_model

    if _tokenizer is None or _model is None or _embedding_model is None:
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model_path = EMBEDDING_MODEL_PATH
        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            embedding_model_path,
            local_files_only=True
        )

        # Load model
        _model = AutoModel.from_pretrained(
            embedding_model_path,
            local_files_only=True
        ).to(device)

        # Set up HuggingFaceEmbedding
        _embedding_model = HuggingFaceEmbedding(model_name=embedding_model_path)

        print(f"Loaded tokenizer, model, and embedding from {embedding_model_path} on {device}")

    return _tokenizer, _model, _embedding_model