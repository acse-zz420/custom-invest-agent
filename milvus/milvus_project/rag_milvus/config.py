# global_vector_model.py
from sentence_transformers import SentenceTransformer
import torch


EMBEDDING_MODEL_PATH = r"D:\python_model\bge-large-zh-v1.5"
API_KEY = "ff6acab6-c747-49d7-b01c-2bea59557b8d"
MD_DIR = r"E:\financial_reports"

_t2v_model = None


def get_t2v_model():
    global _t2v_model
    if _t2v_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _t2v_model = SentenceTransformer(EMBEDDING_MODEL_PATH).to(device)
        print("load T2V_MODEL ok!")
    return _t2v_model