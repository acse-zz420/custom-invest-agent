import json
from typing import Any, List, Optional, AsyncGenerator

from openai import OpenAI, AsyncOpenAI
from pydantic import PrivateAttr, SecretStr, Field

from llama_index.core.llms import (
    CustomLLM, LLMMetadata, ChatMessage, ChatResponse,
    ChatResponseAsyncGen,
    CompletionResponse,
    CompletionResponseGen
)
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.tools.types import BaseTool

from config import ALI_API_KEY, QWEN_MODEL, QWEN_URI


class QwenToolLLM(CustomLLM):
    """
    一个专门为阿里云千问模型设计的、支持 Tool Calling 和异步操作的
    LlamaIndex LLM 类。
    """
    model: str = Field(default=QWEN_MODEL, description="千问模型名称")
    api_key: str = Field(default=ALI_API_KEY, description="阿里云 DashScope API Key")
    base_url: str = Field(default=QWEN_URI, description="千问 API base url")
    temperature: float = Field(default=0.1, description="生成温度")
    max_tokens: int = Field(default=4096, description="最大生成 token 数")
    context_window: int = Field(default=32768, description="模型上下文窗口大小")
    verbose: bool = Field(default=False, description="是否打印详细输出")

    enable_thinking: bool = Field(
        default=False,
        description="是否启用深度思考模式。注意：某些模型会强制开启。"
    )

    _client: OpenAI = PrivateAttr()
    _aclient: AsyncOpenAI = PrivateAttr()

    def __init__(self, **data):

        super().__init__(**data)

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        self._aclient = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model,
            is_chat_model=True,
            is_function_calling_model=True,
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

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """
        同步流式完成。我们明确地禁用它，引导用户使用异步版本。
        """
        raise NotImplementedError("流式输出 (stream_complete) 在此 LLM 中未实现，请使用 astream_chat。")

    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        """ 同步聊天，支持并正确处理 Tool Calling """

        #  将 LlamaIndex ChatMessage 转换为 OpenAI 字典格式
        openai_messages = []
        for msg in messages:
            openai_msg = {"role": msg.role.value, "content": msg.content or ""}
            if msg.additional_kwargs:
                if "tool_calls" in msg.additional_kwargs:
                    openai_msg["content"] = None
                    openai_msg["tool_calls"] = msg.additional_kwargs["tool_calls"]
                if msg.role == MessageRole.TOOL and "tool_call_id" in msg.additional_kwargs:
                    openai_msg["tool_call_id"] = msg.additional_kwargs["tool_call_id"]
            openai_messages.append(openai_msg)

        api_kwargs = kwargs.copy()
        if "tools" in api_kwargs:
            llama_tools: List[BaseTool] = api_kwargs.pop("tools")
            api_kwargs["tools"] = [tool.metadata.to_openai_tool() for tool in llama_tools]

        if self.enable_thinking:
            api_kwargs["extra_body"] = {"enable_thinking": True}

        if self.verbose:
            self._print_payload("QwenToolLLM Sync Chat", openai_messages, api_kwargs)

        try:
            # 使用同步客户端
            response = self._client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                **api_kwargs,
            )
        except Exception as e:
            return ChatResponse(message=ChatMessage(role="assistant", content=f"千问 API (同步) 调用失败: {e}"))

        return self._parse_openai_response_to_chat_response(response)


    async def achat(
            self, messages: List[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """ 异步聊天，支持并正确处理 Tool Calling """

        # 1. 将 LlamaIndex ChatMessage 转换为 OpenAI 字典格式
        openai_messages = []
        for msg in messages:
            openai_msg = {"role": msg.role.value, "content": msg.content or ""}
            if msg.additional_kwargs:
                if "tool_calls" in msg.additional_kwargs:
                    openai_msg["content"] = None
                    openai_msg["tool_calls"] = msg.additional_kwargs["tool_calls"]
                if msg.role == MessageRole.TOOL and "tool_call_id" in msg.additional_kwargs:
                    openai_msg["tool_call_id"] = msg.additional_kwargs["tool_call_id"]
            openai_messages.append(openai_msg)

        #  从 kwargs 提取并转换 tools
        api_kwargs = kwargs.copy()
        if "tools" in api_kwargs:
            llama_tools: List[BaseTool] = api_kwargs.pop("tools")
            api_kwargs["tools"] = [tool.metadata.to_openai_tool() for tool in llama_tools]

        if self.enable_thinking:
            api_kwargs["extra_body"] = {"enable_thinking": True}

        # 打印调试信息
        if self.verbose:
            self._print_payload("QwenToolLLM Sync Chat", openai_messages, api_kwargs)

        # 3. 调用千问 API
        try:
            response = await self._aclient.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                **api_kwargs,
            )
        except Exception as e:
            return ChatResponse(message=ChatMessage(role="assistant", content=f"千问 API 调用失败: {e}"))

        # 将 OpenAI 响应转换回 LlamaIndex 格式
        response_message = response.choices[0].message
        final_additional_kwargs = {}
        if response_message.tool_calls:
            llama_index_tool_calls = [
                {
                    "id": call.id,
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                } for call in response_message.tool_calls
            ]
            final_additional_kwargs["tool_calls"] = llama_index_tool_calls

        return ChatResponse(
            message=ChatMessage(
                role=response_message.role or "assistant",
                content=response_message.content or "",
                additional_kwargs=final_additional_kwargs,
            ),
            raw=response.model_dump(),
        )

