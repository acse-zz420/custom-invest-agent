import os

from torch.cuda import temperature

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from pydantic import Field, SecretStr, PrivateAttr
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from openai import OpenAI



# ========= 1. 自定义火山引擎 LLM 类 =========
class VolcengineLLM(CustomLLM):
    api_key: SecretStr = Field(default='ff6acab6-c747-49d7-b01c-2bea59557b8d', description="Volcengine API Key")
    base_url: str = Field(
        default="https://ark.cn-beijing.volces.com/api/v3",
        description="火山引擎基础地址"
    )
    model: str = Field(
        default="ep-20250422130700-hfw6r",
        description="火山引擎 LLM 模型名称"
    )
    context_window: int = Field(
        default=32768,
        description="上下文窗口大小"
    )
    temperature: float = Field(
        default=0.9,
        description="生成温度"
    )

    _client: OpenAI = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._client = OpenAI(
            api_key=self.api_key.get_secret_value(),
            base_url=self.base_url,
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            model_name=self.model,
            is_chat_model=True
        )

    def complete(self, prompt: str, system_prompt: str = None, **kwargs) -> CompletionResponse:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", 1024)

        system_content = system_prompt or "你是财经领域的智能助手，请结合背景知识提供专业简洁的回答。"

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response_text = completion.choices[0].message.content
            return CompletionResponse(text=response_text)
        except Exception as e:
            return CompletionResponse(text=f"[Error] {str(e)}")

    def stream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("Streaming not supported by VolcengineLLM yet")







