import asyncio
import json
from typing import Dict, List
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage
import re
import xml.etree.ElementTree as ET
from pydantic import BaseModel, Field
from typing import Any, Optional, Dict

from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.workflow import (
    Event, StartEvent, StopEvent, step,
    Workflow
)
from llama_index.core.agent.workflow import AgentWorkflow
from pyexpat.errors import messages

from tools.calculator import *
from tools.rag_tools import *
from llama_index.core.llms import ChatMessage
from llama_index.core.agent.workflow import FunctionAgent
from tool_call_llm import DoubaoToolLLM
from llama_index.utils.workflow import draw_all_possible_flows



# --- 1. 定义工作流中的数据结构 (Events and Plan) ---

class UserInputEvent(StartEvent):
    user_msg: str
    chat_history: List[Dict[str, Any]] = Field(default_factory=list)

class FinalOutputEvent(StopEvent):
    response: str

class CalculationEvent(Event):
    user_msg: str
    calculation_details: str

class DocumentSearchEvent(Event):
    user_msg: str
    query: str

class GraphQueryEvent(Event):
    user_msg: str
    query: str

class SimpleChatEvent(Event):
    user_msg: str

# --- 汇总事件 ---
class SummaryRequestEvent(Event):
    user_msg: str
    tool_result: str

# --- 2. 定义规划者的 Prompt ---

DISPATCHER_PROMPT = """你是一个任务分发机器人。你的任务是分析用户的请求，并结合“聊天历史”的上下文，并判断它属于哪一类任务。

**聊天历史**
{chat_history}

## 任务类别及详细说明 ##

1.  **"calculation" (计算任务)**
    *   **何时使用:** 当用户的请求是一个明确的、可以代入公式进行计算的数学问题时。
    *   **关键词:** “计算”、“收益率”、“波动率是多少”、“百分比”、“增长了多少”。
    *   **示例:** "股价从100涨到105，收益率是多少？" 
    *   **反例:** "A公司的收入增长率是多少？" -> 这不是计算任务，因为需要先查找数据，所以应归类为 "document_search"。

2.  **"document_search" (文档检索任务)**
    *   **何时使用:** 当用户的问题需要通过**阅读和理解**文档、报告、新闻等非结构化文本来寻找**事实、摘要、原因、影响、观点或解释**时。这是最常用的类别。
    *   **关键词:** “是什么”、“为什么”、“分析”、“总结”、“介绍一下”、“有哪些”、“的原因”、“的影响”。
    *   **示例:** "分析一下美债对中国进出口的影响" 

3.  **"graph_query" (图谱查询任务)**
    *   **何时使用:** 当用户的问题是关于**两个或多个特定实体之间**的、**非常明确的、已知的结构化关系**时,具有主谓宾的结构。
    *   **关键词:** “和...的关系”、“半导体市场销量是...”、“哪些公司投资了...”、“...的股东是谁”。
    *   **示例:** "胜利证券和OSL数字证券是什么关系？"

4.  **"simple_chat" (简单聊天)**
    *   **何时使用:** 当用户的请求是简单的问候、闲聊，或者是一个不属于以上任何类别的常识性问题。
    *   **示例:** "你好"

对于非计算的问题，优先使用"document_search"
请根据用户的请求和聊天历史，只返回一个包含 `task_type` 和 `query` 的、严格的 JSON 对象。`query` 应该是传递给下一步专家的、精炼后的指令。



例如：
用户请求: "A公司股票A的2025-8-25收盘价为56.7，在2025-8-26收盘价为58.4，请问他的收益率是多少？"
你的输出:
```json
{{
  "task_type": "calculation",
  "query": "使用起始价格56.7和结束价格58.4，调用calculate_return_rate工具计算收益率"
}}

## 当前用户请求 ##
{user_msg}
"""


# --- 3. 创建自定义工作流 ---

class FinancialWorkflow(Workflow):
    def __init__(self, llm: LLM, agents: Dict[str, AgentWorkflow], verbose: bool = False):
        super().__init__(
            timeout=300.0,
            verbose=verbose,
        )

        self.llm = llm
        self.agents = agents
        self.verbose = verbose

        self.tools = {
            "Calculator": [
                return_rate_tool,
                volatility_tool
            ],
            "DocumentSearch": [custom_vector_rag_tool],
            "KnowledgeGraph": [custom_graph_rag_tool],
        }
    # 步骤 1: 规划 (Plan)
    @step
    async def dispatcher_step(
            self, ev: UserInputEvent
    ) -> CalculationEvent | DocumentSearchEvent | GraphQueryEvent | SimpleChatEvent:
        if self.verbose: print(f"--- [Dispatcher]: 正在分析请求 '{ev.user_msg}'... ---")

        formatted_history_parts = []
        for msg in ev.chat_history:
            # 获取 role
            role = msg.get("role", "unknown").capitalize()

            # 从 blocks 列表中提取文本内容
            content_text = ""
            blocks = msg.get("blocks", [])
            if blocks and isinstance(blocks, list):

                text_blocks = [b.get("text") for b in blocks if b.get("block_type") == "text" and b.get("text")]
                content_text = "\n".join(text_blocks)

            if content_text:
                formatted_history_parts.append(f"{role}: {content_text}")

        formatted_history = "\n".join(formatted_history_parts)
        if not formatted_history:
            formatted_history = "无历史记录。"

        prompt = DISPATCHER_PROMPT.format( chat_history=formatted_history, user_msg=ev.user_msg)

        response = await self.llm.achat(
            messages=[ChatMessage(role="user", content=prompt)],
        )

        try:

            # 1. 首先，获取原始 content
            raw_content = response.message.content
            if self.verbose: print(
                f"---  [Dispatcher Step 1]: LLM 返回的原始 content 类型: {type(raw_content)}, 内容: '{raw_content}' ---")

            # 2. 检查 content 是否为 None 或空
            if not raw_content or not raw_content.strip():
                raise ValueError("LLM returned empty content.")

            # 3. 执行清理操作
            cleaned_response = raw_content.strip().replace("```json", "").replace("```", "")

            final_json_str = cleaned_response.strip()
            if self.verbose: print(f"---  [Dispatcher Step 3]: 最终准备解析的字符串: '{final_json_str}' ---")

            decision = json.loads(final_json_str)
            task_type = decision.get("task_type")
            query = decision.get("query")

            if self.verbose: print(f"--- [Dispatcher]: 决策完成，任务类型 = {task_type} ---")

            if task_type == "calculation":
                return CalculationEvent(user_msg=ev.user_msg, calculation_details=query)
            elif task_type == "document_search":
                return DocumentSearchEvent(user_msg=ev.user_msg, query=query)
            elif task_type == "graph_query":
                return GraphQueryEvent(user_msg=ev.user_msg, query=query)
            else:
                return SimpleChatEvent(user_msg=ev.user_msg)
        except Exception as e:
            error_context = "cleaned_response 未定义"
            if 'cleaned_response' in locals():
                error_context = f"cleaned_response 的值为: '{cleaned_response}'"

            if self.verbose: print(f"--- ❌ [Dispatcher]: 决策解析失败 ({e})，上下文: {error_context}，转为简单聊天。 ---")
            return SimpleChatEvent(user_msg=ev.user_msg)

    async def _execute_tool_calling_step(self, tool_category: str, user_input: str) -> str:
        """一个通用的、执行 Tool Calling 的辅助函数"""
        target_tools = self.tools[tool_category]
        if self.verbose: print(f"--- [{tool_category}]: 接收到任务 '{user_input}' ---")

        # 阶段 1: 模型决策
        response = await self.llm.achat(
            messages=[ChatMessage(role="user", content=user_input)],
            tools=target_tools,
            tool_choice="auto",
        )

        tool_calls = response.message.additional_kwargs.get("tool_calls")

        if not tool_calls:
            if self.verbose: print(f"--- [{tool_category}]: LLM 决定不调用工具，直接回答。---")
            return response.message.content

        # 阶段 2: 本地执行
        if self.verbose: print(f"--- 🔧 [{tool_category}]: LLM 决定调用工具... ---")
        tool_outputs = []
        for call in tool_calls:
            tool_name = call.get("function", {}).get("name")
            arguments_str = call.get("function", {}).get("arguments")

            if not tool_name or arguments_str is None:
                tool_outputs.append({"output": f"错误: LLM 返回的 tool_call 格式不完整: {call}"})
                continue
            arguments = json.loads(arguments_str)

            target_tool = next((t for t in target_tools if t.metadata.name == tool_name), None)
            if not target_tool:
                tool_outputs.append({"output": f"错误: 找不到名为 '{tool_name}' 的工具对象"})
                continue

            target_tool_fn = target_tool.fn

            #   我们现在要判断 target_tool_fn 是同步还是异步
            if asyncio.iscoroutinefunction(target_tool_fn):
                # 如果是异步函数 (async def)，就直接 await
                if self.verbose: print(f"      - (Async) 执行函数: {tool_name}(**{arguments})")
                tool_result = await target_tool_fn(**arguments)
            else:
                # 如果是同步函数 (def)，就用 asyncio.to_thread 在独立线程中运行
                if self.verbose: print(f"      - (Sync via Thread) 执行函数: {tool_name}(**{arguments})")
                tool_result = await asyncio.to_thread(target_tool_fn, **arguments)

            tool_outputs.append({"tool_name": tool_name, "output": str(tool_result)})


        final_result = "\n".join([f"{r['tool_name']} 返回: {r['output']}" for r in tool_outputs])
        return final_result

    # 步骤 2a: 计算分支

    @step
    async def calculation_step(self, ev: CalculationEvent) -> SummaryRequestEvent:
        result = await self._execute_tool_calling_step("Calculator", ev.calculation_details)
        return SummaryRequestEvent(user_msg=ev.user_msg, tool_result=result)

    # 步骤 2b: 文档检索分支
    @step
    async def doc_search_step(self, ev: DocumentSearchEvent) -> SummaryRequestEvent:
        result = await self._execute_tool_calling_step("DocumentSearch", ev.query)
        return SummaryRequestEvent(user_msg=ev.user_msg, tool_result=result)

    # 步骤 2c: 图谱查询分支
    @step
    async def graph_query_step(self, ev: GraphQueryEvent) -> SummaryRequestEvent:
        result = await self._execute_tool_calling_step("KnowledgeGraph", ev.query)
        return SummaryRequestEvent(user_msg=ev.user_msg, tool_result=result)

    # 步骤 2d: 简单聊天分支
    @step
    async def simple_chat_step(self, ev: SimpleChatEvent) -> FinalOutputEvent:
        if self.verbose: print(f"--- 💬 [Chat]: 正在直接回答... ---")
        messages_to_send = [ChatMessage(role="user", content=ev.user_msg)]
        response = await self.llm.achat(messages_to_send)
        return FinalOutputEvent(response=response.message.content)

    # 步骤 3: 总结器 (Summarizer) - 接收所有专家分支的结果
    @step
    async def summarizer_step(self, ev: SummaryRequestEvent) -> FinalOutputEvent:
        if self.verbose: print(f"--- [Summarizer]: 正在总结专家结果... ---")
        prompt = (
            f"你是一个总结专家。请根据用户的“原始请求”和下面专家的“工作报告”，生成一个最终的、流畅连贯的中文回答。\n\n"
            f"## 用户的原始请求 ##\n{ev.user_msg}\n\n"
            f"## 专家工作报告 ##\n{ev.tool_result}\n\n"
            f"## 最终回答 ##"
        )
        response = await self.llm.achat([ChatMessage(role="user", content=prompt)])
        return FinalOutputEvent(response=response.message.content)

# draw_all_possible_flows(FinancialWorkflow, filename="multi_step_workflow.html")