# api.py (最终的、简化版)

import asyncio
import traceback
import uuid
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 导入 LlamaIndex 核心组件
from llama_index.core.settings import Settings
from llama_index.core.llms import ChatMessage

# 导入您的自定义核心组件
from llm_ali import QwenToolLLM
from Agent.planner import FinancialWorkflow
from config import *
from Agent.specialists import get_specialist_agents


# --- 定义 API 的请求和响应 Pydantic 模型 ---

class ChatRequest(BaseModel):
    user_msg: str = Field(..., description="用户的聊天输入内容")
    session_id: str = Field(..., description="用于区分不同对话会话的唯一ID")
    chat_history: List[Dict[str, Any]] = Field(default_factory=list, description="之前的对话历史")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Agent生成的最终回答")
    session_id: str = Field(..., description="当前对话的会话ID")
    new_chat_history: List[Dict[str, Any]] = Field(..., description="包含了最新一轮对话的完整历史")


# --- 创建 FastAPI ---
app = FastAPI(
    title="金融分析智能体 API",
    description="一个集成了计算、文档检索和知识图谱查询的 Agent 工作流。",
    version="1.0.0",
)


app_state = {
    "workflow": None,
    "chat_histories": {},
    "is_initialized": False
}


# --- 初始化 ---
def initialize_workflow():
    """
    加载所有模型和资源，并创建工作流的单例实例。

    """
    # 使用 global 关键字来修改全局的 app_state
    global app_state

    # 防止重复初始化
    if app_state["is_initialized"]:
        return

    print("--- 首次请求，开始初始化工作流... ---")

    # a. 加载模型和资源
    print("   - Loading models and resources...")
    Settings.embed_model = get_embedding_model()
    llm = QwenToolLLM()

    # b. 创建工作流实例
    print("   - Initializing FinancialWorkflow...")
    workflow = FinancialWorkflow(
        llm=llm,
        agents=get_specialist_agents(),
        verbose=True
    )

    # c. 将核心对象存入全局状态
    app_state["workflow"] = workflow
    app_state["is_initialized"] = True  # 标记为已初始化

    print("--- 工作流初始化完成。---")


# --- 创建 API 端点 ---
@app.get("/")
def read_root():
    return {"Hello": "金融分析智能体"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    接收用户聊天请求，调用 Agent Workflow，并返回结果。
    """
    # --- 懒加载：在第一次请求时，才执行初始化 ---
    if not app_state["is_initialized"]:
        initialize_workflow()

    workflow = app_state["workflow"]
    chat_histories = app_state["chat_histories"]

    session_history = chat_histories.get(request.session_id, [])

    print(f"\n--- 收到来自 Session '{request.session_id}' 的请求 ---")
    print(f"   - 用户问题: {request.user_msg}")

    try:
        import nest_asyncio
        nest_asyncio.apply()

        async def run_task():
            chat_history_for_run = [msg.model_dump() for msg in session_history]
            handler = workflow.run(
                user_msg=request.user_msg,
                chat_history=chat_history_for_run
            )
            return await handler

        result = await run_task()  # FastAPI 已经在事件循环中
        final_response_text = result.response

        # 更新历史记录
        session_history.append(ChatMessage(role="user", content=request.user_msg))
        session_history.append(ChatMessage(role="assistant", content=final_response_text))
        chat_histories[request.session_id] = session_history

        new_chat_history_dicts = [msg.model_dump() for msg in session_history]

        return ChatResponse(
            response=final_response_text,
            session_id=request.session_id,
            new_chat_history=new_chat_history_dicts
        )

    except Exception as e:
        print(f"\n 工作流执行出错: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {e}")


if __name__ == '__main__':
    import uvicorn

    SERVER_IP = '0.0.0.0'
    SERVER_PORT = 8000

    print(f"--- 启动 FastAPI 服务器，监听在 {SERVER_IP}:{SERVER_PORT} ---")
    uvicorn.run(app, host=SERVER_IP, port=SERVER_PORT)