import asyncio
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llama_index.core.settings import Settings
from llama_index.core.llms import ChatMessage

# 导入您的核心组件
from llm_ali import QwenToolLLM
from .planner import FinancialWorkflow
from config import *
from .specialists import get_specialist_agents



# --- 定义 API 的请求和响应 Pydantic 模型 ---


class ChatRequest(BaseModel):
    user_msg: str = Field(..., description="用户的聊天输入内容")
    session_id: str = Field(..., description="用于区分不同对话会话的唯一ID")
    # chat_history 现在是一个可选字段
    chat_history: List[Dict[str, Any]] = Field(default_factory=list, description="之前的对话历史")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Agent生成的最终回答")
    session_id: str = Field(..., description="当前对话的会话ID")
    # 返回更新后的历史记录，方便客户端继续下一轮对话
    new_chat_history: List[Dict[str, Any]] = Field(..., description="包含了最新一轮对话的完整历史")


# --- 创建一个全局变量来存储“应用状态” ---

app_state = {}


# --- 使用 lifespan 来管理应用的生命周期 ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 服务器启动时执行的代码 ---
    print("--- FastAPI Application startup... ---")

    # b. 加载模型和资源
    print("   - Loading models and resources (this may take a while)...")
    _,_,Settings.embed_model = get_embedding_model()
    llm = QwenToolLLM()

    # c. 创建工作流的单例实例
    print("   - Initializing FinancialWorkflow...")
    workflow = FinancialWorkflow(
        llm=llm,
        agents=get_specialist_agents(),
        verbose=True
    )

    # d. 将核心对象存入全局状态，以便在 API 请求中访问
    app_state["workflow"] = workflow
    # 使用一个线程安全的字典来管理多用户的聊天历史
    app_state["chat_histories"] = {}

    print("--- Application startup complete. Server is ready to accept requests. ---")

    yield



# --- 创建 FastAPI 应用，并应用 lifespan ---
app = FastAPI(
    title="金融分析智能体 API",
    description="一个集成了计算、文档检索和知识图谱查询的复杂 Agent 工作流。",
    version="1.0.0",
    lifespan=lifespan
)


# --- 创建 API 端点 ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    接收用户聊天请求，调用 Agent Workflow，并返回结果。
    """
    workflow = app_state.get("workflow")
    if not workflow:
        raise HTTPException(status_code=503, detail="Workflow is not initialized. Please wait for startup to complete.")

    # 从全局状态中获取或创建该 session 的历史记录
    # 这是一个简单的内存实现，生产环境建议使用 Redis 等
    chat_histories = app_state["chat_histories"]
    session_history = chat_histories.get(request.session_id, [])

    print(f"\n--- 收到来自 Session '{request.session_id}' 的请求 ---")
    print(f"   - 用户问题: {request.user_msg}")
    print(f"   - 传入历史记录: {len(session_history)} 条")

    try:
        # a. 运行工作流
        import nest_asyncio
        nest_asyncio.apply()

        async def run_task():
            chat_history_for_run = [msg.model_dump() for msg in session_history]
            handler = workflow.run(
                user_msg=request.user_msg,
                chat_history=chat_history_for_run
            )
            return await handler

        # FastAPI 已经在事件循环中，直接 await 即可
        # 无需 asyncio.run()
        result = await run_task()
        final_response_text = result.response

        # b. 更新该 session 的历史记录
        session_history.append(ChatMessage(role="user", content=request.user_msg))
        session_history.append(ChatMessage(role="assistant", content=final_response_text))
        chat_histories[request.session_id] = session_history  # 写回全局状态

        # c. 将更新后的历史记录（对象）转换回字典用于返回
        new_chat_history_dicts = [msg.model_dump() for msg in session_history]

        # d. 返回成功的响应
        return ChatResponse(
            response=final_response_text,
            session_id=request.session_id,
            new_chat_history=new_chat_history_dicts
        )

    except Exception as e:
        print(f"\n❌ 工作流执行出错: {e}")
        traceback.print_exc()
        # 返回一个标准的 HTTP 500 错误
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {e}")