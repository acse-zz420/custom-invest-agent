from rag_milvus import tracing
import json
import re


from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.workflow import (
    Event, StartEvent, StopEvent, step,
    Workflow, Context
)
from llama_index.core.agent.workflow import AgentWorkflow
from pyexpat.errors import messages

from tools.calculator import *
from tools.rag_tools import *
from llama_index.core.llms import ChatMessage

from agent_prompts import *
from events import *
from llama_index.utils.workflow import draw_all_possible_flows

class FinancialWorkflow(Workflow):
    def __init__(self, llm: LLM, agents: Dict[str, AgentWorkflow], verbose: bool = False, max_loops:int=3):
        super().__init__(
            timeout=300.0,
            verbose=verbose,
        )

        self.llm = llm
        self.agents = agents
        self.verbose = verbose
        self.max_loops = max_loops
        self.tools = {
            "Calculator": [
                return_rate_tool,
                volatility_tool
            ],
            "DocumentSearch": [custom_vector_rag_tool],
            "KnowledgeGraph": [custom_graph_rag_tool],
        }

    def _extract_json_from_response(self, response_content: str) -> str:
        """
        一个健壮的辅助函数，使用正则表达式从 LLM 的响应中提取出 JSON 字符串。
        """
        # 正则表达式：匹配从第一个 '{' 到最后一个 '}' 之间的所有内容
        # re.DOTALL 标志让 '.' 可以匹配包括换行符在内的任何字符
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)

        if json_match:
            json_str = json_match.group(0)
            if self.verbose: print(f"--- [JSON Extractor]: 成功提取到 JSON: '{json_str}' ---")
            return json_str
        else:
            if self.verbose: print(f"--- [JSON Extractor]: 未在响应中找到有效的 JSON 对象。 ---")
            raise ValueError("No valid JSON object found in LLM response.")

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
    # 步骤 1: 规划 (Plan)
    @step
    async def dispatcher_step(self, ev: UserInputEvent) -> RAGTriggerEvent | SimpleChatTriggerEvent:
        if self.verbose: print(f"--- [Dispatcher]: 分析请求 '{ev.user_msg}'... ---")
        prompt = DISPATCHER_PROMPT.format(user_msg=ev.user_msg, chat_history=str(ev.chat_history))
        response = await self.llm.achat([ChatMessage(role="user", content=prompt)])
        try:
            raw_content = response.message.content or ""
            if self.verbose: print(f"--- [Dispatcher]: LLM 返回的原始决策文本: '{raw_content}' ---")

            cleaned_response = self._extract_json_from_response(raw_content)
            decision = json.loads(cleaned_response)

            if decision.get("task_type") == "retrieval":

                return RAGTriggerEvent(user_msg=ev.user_msg, query=ev.user_msg)
            else:
                return SimpleChatTriggerEvent(user_msg=ev.user_msg)
        except Exception as e:
            if self.verbose: print(f"--- [Dispatcher]: 决策解析失败 ({e})，转为简单聊天。 ---")
            return SimpleChatTriggerEvent(user_msg=ev.user_msg)

    # --- 路线 A: 简单聊天 ---
    @step
    async def simple_chat_step(self, ev: SimpleChatTriggerEvent) -> FinalOutputEvent:
        if self.verbose: print(f"--- [SimpleChat]: 直接回答... ---")
        response = await self.llm.achat([ChatMessage(role="user", content=ev.user_msg)])
        return FinalOutputEvent(response=response.message.content)

    # --- 路线 B: RAG 流程 ---

    # 步骤 B.1: 并行检索 (Fan-Out)
    @step
    async def milvus_retrieval_step(self, ev: RAGTriggerEvent) -> SearchResultEvent:
        if self.verbose: print(f"--- [Milvus]: (循环 {ev.current_loop}) 开始并行检索... ---")
        nodes = await custom_milvus_search(ev.query)
        return SearchResultEvent(source="milvus", results=nodes, user_msg=ev.user_msg, query=ev.query,
                                 current_loop=ev.current_loop)

    @step
    async def graph_retrieval_step(self, ev: RAGTriggerEvent) -> SearchResultEvent:
        if self.verbose: print(f"--- [Graph]: (循环 {ev.current_loop}) 开始并行查询... ---")
        nodes = await asyncio.to_thread(custom_graph_search, ev.query)
        return SearchResultEvent(source="rag_graph", results=nodes, user_msg=ev.user_msg, query=ev.query,
                                 current_loop=ev.current_loop)


    @step
    async def aggregator_step(self, ev: SearchResultEvent, ctx: Context) -> AggregatedContextEvent| None:

        search_results: List[SearchResultEvent] | None = ctx.collect_events(
            ev, expected=[SearchResultEvent, SearchResultEvent]
        )
        if search_results is None:
            # 如果还没有收集齐（比如只有一个分支完成了），
            # 就返回 None，工作流会暂停此步骤并等待下一个事件
            if self.verbose: print(f"--- [Aggregator]: 已收到来自 '{ev.source}' 的结果，等待其他并行结果... ---")
            return None
        if self.verbose: print(f"--- [Aggregator]: 开始融合去重... ---")

        all_nodes = []
        for result_event in search_results:
            all_nodes.extend(result_event.results)

        # (去重逻辑)
        unique_nodes = {(n.node.metadata.get("file_name"), n.node.get_content()): n for n in all_nodes}
        final_nodes = list(unique_nodes.values())

        fused_context = "\n\n".join(
            [f"来源: {n.node.metadata.get('file_name', 'N/A')}\n内容: {n.node.get_content()}" for n in final_nodes])
        ref_event = search_results[0]

        return AggregatedContextEvent(
            user_msg=ref_event.user_msg, query=ref_event.query, current_loop=ref_event.current_loop,
            retrieved_nodes=final_nodes, fused_context=fused_context
        )

    # 步骤 B.3: 判断是否需要计算
    @step
    async def calculation_check_step(self, ev: AggregatedContextEvent) -> CalculationEvent | SummarizationEvent:
        if self.verbose: print(f"--- [CalcCheck]: 检查是否需要计算... ---")
        prompt = CALCULATION_CHECK_PROMPT.format(user_msg=ev.user_msg, context=ev.fused_context)
        response = await self.llm.achat([ChatMessage(role="user", content=prompt)])
        raw_content = response.message.content or ""
        if self.verbose: print(f"--- [Dispatcher]: LLM 返回的原始决策文本: '{raw_content}' ---")

        cleaned_response = self._extract_json_from_response(raw_content)
        decision = json.loads(cleaned_response)

        if decision.get("calculation_needed"):
            if self.verbose: print(f"--- [CalcCheck]: 需要计算。进入计算步骤... ---")
            # 将所有状态传递给计算事件
            return CalculationEvent(user_msg=ev.user_msg, calculation_details=decision.get("calculation_query"),
                                    fused_context=ev.fused_context,current_loop=ev.current_loop)
        else:
            if self.verbose: print(f"--- [CalcCheck]: 无需计算。直接进入总结步骤... ---")
            # 如果不需要计算，就直接用融合的上下文去生成答案
            return SummarizationEvent(user_msg=ev.user_msg, final_context=ev.fused_context,current_loop=ev.current_loop)

    # 步骤 B.4: 计算 (可选路径)
    @step
    async def calculation_step(self, ev: CalculationEvent) -> SummarizationEvent:
        if self.verbose: print(f"--- [Calculator]: 执行计算... ---")
        # (你需要一个 _execute_tool_calling_step 辅助函数)
        calc_result = await self._execute_tool_calling_step("Calculator", ev.calculation_details)

        final_context = f"检索到的信息:\n{ev.fused_context}\n\n计算结果:\n{calc_result}"
        return SummarizationEvent(user_msg=ev.user_msg, final_context=final_context, current_loop=ev.current_loop)

    # 步骤 B.5: 生成最终答案
    @step
    async def summarizer_step(self, ev: SummarizationEvent) -> CritiqueEvent:
        if self.verbose: print(f"---  [Summarizer]: 生成草稿回答... ---")
        prompt = SUMMARIZER_PROMPT.format(context=ev.final_context, user_msg=ev.user_msg)
        response = await self.llm.achat([ChatMessage(role="user", content=prompt)])
        draft_answer = response.message.content
        return CritiqueEvent(user_msg=ev.user_msg, final_context=ev.final_context, preliminary_answer=draft_answer,
                             current_loop=ev.current_loop)

    # 步骤 B.6: 质量评估与循环
    @step
    async def quality_check_step(self, ev: CritiqueEvent) -> RAGTriggerEvent | FinalOutputEvent:
        if self.verbose: print(f"--- [Critique]: 评估草稿回答... ---")
        prompt = CRITIQUE_PROMPT.format(user_msg=ev.user_msg, draft_answer=ev.preliminary_answer)
        response = await self.llm.achat([ChatMessage(role="user", content=prompt)])
        raw_content = response.message.content or ""
        if self.verbose: print(f"--- [Dispatcher]: LLM 返回的原始决策文本: '{raw_content}' ---")

        cleaned_response = self._extract_json_from_response(raw_content)
        critique = json.loads(cleaned_response)

        current_loop = ev.current_loop if hasattr(ev, 'current_loop') else 1

        if critique.get("is_sufficient") or current_loop >= self.max_loops:
            if not critique.get("is_sufficient"):
                print(f"---  [Critique]: 已达到最大循环次数 ({self.max_loops})，强制结束。---")
            if self.verbose: print(f"--- [Critique]: 回答令人满意。工作流结束。 ---")
            return FinalOutputEvent(response=ev.preliminary_answer)
        else:
            if self.verbose: print(f"--- [Critique]: 回答不理想。将基于修正建议重新开始检索... ---")
            new_query = f"原始问题: {ev.user_msg}\n修正建议: {critique.get('missing_information')}"
            return RAGTriggerEvent(user_msg=ev.user_msg, query=new_query, current_loop=current_loop + 1)


draw_all_possible_flows(FinancialWorkflow, filename="multi_step_workflow.html")