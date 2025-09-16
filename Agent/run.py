import traceback
import asyncio
from llama_index.core.settings import Settings
from Agent.tool_call_llm import DoubaoToolLLM
from llama_index.core.llms import ChatMessage
from Agent.planner import FinancialWorkflow
from config import *
from Agent.specialists import get_specialist_agents
from timer_tool import timer


import nest_asyncio

# 应用补丁
nest_asyncio.apply()
@timer
def main():
    _, _, Settings.embed_model = get_embedding_model()
    llm = DoubaoToolLLM(api_key=API_KEY, model=TOOL_CALL_MODEL)
    workflow = FinancialWorkflow(llm=llm, verbose=True, agents=get_specialist_agents())

    print("\n🤖 金融分析工作流已准备就绪！请输入您的问题 (输入 'exit' 退出)。\n")

    chat_history = []
    while True:
        user_input = input("👤 你: ")
        if user_input.lower() == 'exit':
            break

        try:
            async def run_task():
                # 1. 启动工作流并获取 handler
                chat_history_for_run = [msg.model_dump() for msg in chat_history]
                handler = workflow.run(
                    user_msg=user_input,
                    chat_history=chat_history_for_run
                )

                final_result_object = await handler

                return final_result_object

            result = asyncio.run(run_task())
            final_response_text = result.response

            chat_history.append(ChatMessage(role="user", content=user_input))
            chat_history.append(ChatMessage(role="assistant", content=final_response_text))

            print(f"\n🤖 Agent 最终回答: {final_response_text}\n")

        except Exception as e:
            print(f"\n❌ 工作流执行出错:")
            traceback.print_exc()


if __name__ == "__main__":
    main()
    print('-----------------')
