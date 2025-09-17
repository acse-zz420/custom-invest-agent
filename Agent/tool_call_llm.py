import json
import logging
from typing import Any, List, Optional

from openai import AsyncOpenAI, OpenAI
from pydantic import PrivateAttr, SecretStr

from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    CompletionResponseAsyncGen,
    LLMMetadata,
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen
)

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    ChatResponseAsyncGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.custom import CustomLLM


from llama_index.core.llms.llm import ToolSelection
from llama_index.core.tools.types import BaseTool, ToolMetadata
from config import VOL_URI, TOOL_CALL_MODEL
from config import API_KEY

# ========= 火山豆包 Tool Calling 专用 LLM 类 =========


class DoubaoToolLLM(CustomLLM):
    """
    一个专门为火山引擎豆包模型设计的、支持 Tool Calling 和异步操作的
    LlamaIndex LLM 类。
    """

    model: str = Field(
        default=TOOL_CALL_MODEL,
        description="火山引擎豆包模型名称，例如 doubao-seed-1.6, doubao-pro-32k 等"
    )
    api_key: SecretStr = Field(description="火山引擎 API Key")
    base_url: str = Field(
        default=VOL_URI,
        description="火山引擎 API 基础地址"
    )
    temperature: float = Field(default=0.1, description="生成温度")
    max_tokens: int = Field(default=4096, description="最大生成 token 数")
    context_window: int = Field(default=32768, description="模型上下文窗口大小")
    verbose: bool = Field(default=False, description="是否打印详细输出")

    _aclient: AsyncOpenAI = PrivateAttr()

    def __init__(
            self,
            model: str = TOOL_CALL_MODEL,
            api_key: str = API_KEY,
            base_url: str = VOL_URI,
            temperature: float = 0.1,
            max_tokens: int = 4096,
            context_window: int = 32768,
            verbose: bool = False,
            callback_manager: Optional[CallbackManager] = None,
            **kwargs: Any,
    ) -> None:
        # Pydantic v2 requires passing api_key as a SecretStr
        api_key_secret = SecretStr(api_key)

        super().__init__(
            model=model,
            api_key=api_key_secret,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window,
            callback_manager=callback_manager,
            **kwargs,
        )
        self.verbose = True
        self._client = OpenAI(
            api_key=self.api_key.get_secret_value(),
            base_url=self.base_url,
        )
        self._aclient = AsyncOpenAI(
            api_key=self.api_key.get_secret_value(),
            base_url=self.base_url,
        )

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model,
            is_chat_model=True,
            is_function_calling_model=True,  # 明确声明支持 Tool Calling
        )

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        必需的同步非流式方法。
        """
        kwargs.pop("formatted", None)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return CompletionResponse(
            text=response.choices[0].message.content or "",
            raw=response.model_dump()
        )

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """
        必需的同步流式方法。
        我们明确地告诉用户，请使用异步版本。
        """
        kwargs.pop("formatted", None)
        raise NotImplementedError(
            "DoubaoToolLLM sync streaming is not implemented. Please use 'astream_complete' or 'astream_chat'.")

    # --- LlamaIndex 的核心异步方法 ---

    async def achat(
            self, messages: List[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """ 异步聊天，支持并正确处理 Tool Calling """

        kwargs.pop("formatted", None)
        openai_messages = []
        for msg in messages:
            # 1. 构造基础的 OpenAI 消息字典，使用 .content 属性来获取文本
            openai_msg = {
                "role": msg.role.value,
                "content": msg.content  # .content 属性会自动处理 blocks
            }

            # 2. 处理 Tool Calling 相关的 additional_kwargs
            if msg.additional_kwargs:
                # 处理模型发起的 tool_calls
                tool_calls_from_kw = msg.additional_kwargs.get("tool_calls")
                if tool_calls_from_kw:
                    # 如果有 tool_calls，OpenAI 期望 content 为 None 或空
                    openai_msg["content"] = None

                    # 将 LlamaIndex 的 ToolSelection 字典转换为 OpenAI 格式
                    openai_msg["tool_calls"] = [
                        {
                            "id": tc.get("id"),
                            "type": "function",
                            "function": {
                                "name": tc.get("function", {}).get("name"),
                                "arguments": tc.get("function", {}).get("arguments"),
                            },
                        }
                        for tc in tool_calls_from_kw
                    ]

                # 处理工具返回的结果 (role=tool)
                if msg.role == MessageRole.TOOL and "tool_call_id" in msg.additional_kwargs:
                    openai_msg["tool_call_id"] = msg.additional_kwargs["tool_call_id"]

            openai_messages.append(openai_msg)

        # 提取并转换 tools
        api_kwargs = kwargs.copy()
        if "tools" in api_kwargs:
            llama_tools = api_kwargs.pop("tools")
            api_kwargs["tools"] = [tool.metadata.to_openai_tool() for tool in llama_tools]

        # 打印调试信息 (保持不变)
        if self.verbose:
            payload_to_print = {"model": self.model, "messages": openai_messages, **api_kwargs}
            logging.info("\n" + "=" * 20 + " [DoubaoToolLLM Debug: Final OpenAI Payload] " + "=" * 20)
            logging.info(json.dumps(payload_to_print, indent=2, ensure_ascii=False))
            logging.info("=" * 80 + "\n")

        # 调用 API
        try:
            response = await self._aclient.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                **api_kwargs,
            )
        except Exception as e:
            # 如果 API 调用本身就失败了，直接返回错误信息
            return ChatResponse(
                message=ChatMessage(role="assistant", content=f"API 调用失败: {e}")
            )
        if not response.choices:
            return ChatResponse(message=ChatMessage(role="assistant", content="API 返回了空的 choices 列表。"))

        response_message = response.choices[0].message

        if not response_message:
            return ChatResponse(message=ChatMessage(role="assistant", content="API 返回的 choice 中没有 message 对象。"))

        # 2. 初始化 additional_kwargs 和 llama_index_tool_calls
        final_additional_kwargs = {}
        llama_index_tool_calls = []

        # 3. 先检查 tool_calls 是否存在且是一个列表，然后再遍历
        if response_message.tool_calls and isinstance(response_message.tool_calls, list):
            if self.verbose:
                print("--- [DoubaoToolLLM Debug]: OpenAI response contains tool calls. Converting... ---")

            for call in response_message.tool_calls:
                llama_index_tool_calls.append(
                    {
                        "id": call.id,
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                )

            # 只有当成功处理了 tool_calls 后，才将其放入 additional_kwargs
            final_additional_kwargs["tool_calls"] = llama_index_tool_calls

        # 4. 最终返回一个结构正确的 ChatResponse
        return ChatResponse(
            message=ChatMessage(
                role=response_message.role or "assistant",
                content=response_message.content or "",  # 确保 content 不为 None
                additional_kwargs=final_additional_kwargs,
            ),
            raw=response.model_dump(),
        )

    async def astream_chat(
            self, messages: List[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """ 异步聊天，流式 """
        kwargs.pop("formatted", None)
        openai_messages = [msg.dict() for msg in messages]

        async def gen() -> ChatResponseAsyncGen:
            stream = await self._aclient.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                **kwargs,
            )

            collected_content = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta
                content = delta.content or ""
                collected_content += content

                yield ChatResponse(
                    message=ChatMessage(
                        role="assistant",
                        content=collected_content,
                    ),
                    delta=content,
                    raw=chunk.model_dump(),
                )

        return gen()


    # --- Tool Calling 的核心异步方法 ---

    async def achat_with_tools(
            self,
            tools: List[BaseTool],
            user_message: ChatMessage,
            chat_history: Optional[List[ChatMessage]] = None,
    ) -> ChatResponse:
        """ 专为 Tool Calling 设计的异步聊天方法 """

        # 将 LlamaIndex 的 Tool 转换为 OpenAI API 兼容的格式
        tool_descs = [tool.metadata.to_openai_tool() for tool in tools]

        # 准备消息历史
        messages = chat_history or []
        messages.append(user_message)
        openai_messages = [msg.dict() for msg in messages]

        response = await self._aclient.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            tools=tool_descs,
            tool_choice="auto",  # 让模型自己决定是否以及调用哪个工具
        )

        response_message = response.choices[0].message

        # 检查模型是否决定调用工具
        tool_calls = response_message.tool_calls
        if tool_calls:
            # 如果需要调用工具，LlamaIndex Agent 会期望一个包含 tool_calls 的消息
            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content=response_message.content or "",  # 可能为空
                    additional_kwargs={
                        "tool_calls": [
                            ToolSelection(
                                tool_id=call.id,
                                tool_name=call.function.name,
                                tool_args_str=call.function.arguments,
                            ).dict()
                            for call in tool_calls
                        ]
                    }
                ),
                raw=response.model_dump()
            )
        else:
            # 如果模型决定不调用工具，直接返回它的文本回答
            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content=response_message.content,
                ),
                raw=response.model_dump()
            )

    async def achat_with_tool_result(
            self,
            messages_with_tool_history: List[ChatMessage],
            **kwargs: Any
    ) -> ChatResponse:
        """
        异步聊天，但这次的消息列表包含了之前的工具调用和工具结果。
        用于 Tool Calling 的第二阶段。
        """
        openai_messages = [msg.dict(exclude_none=True) for msg in messages_with_tool_history]

        if self.verbose:
            print("\n" + "=" * 20 + " [Raw Request Sent to LLM API - Step 2] " + "=" * 20)
            print("--- Request to: achat (with tool result) ---")
            print("--- Messages (including tool history) ---")
            print(json.dumps(openai_messages, indent=2, ensure_ascii=False))
            print("=" * 80 + "\n")

        # 这次调用不应该再包含 tools 参数
        response = await self._aclient.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )

        return ChatResponse(
            message=ChatMessage(
                role=response.choices[0].message.role or "assistant",
                content=response.choices[0].message.content,
            ),
            raw=response.model_dump(),
        )