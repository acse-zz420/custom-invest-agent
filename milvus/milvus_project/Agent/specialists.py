from openai import api_key

from llm import VolcengineLLM
from .tools.financial_calculator import financial_calculator_tool
from .tools.rag_tools import rag_tools

from llama_index.core.agent.workflow import FunctionAgent

from llm_ali import QwenToolLLM


def get_specialist_agents() -> dict[str: FunctionAgent]:
    """
    创建并返回一个包含所有专家智能体的列表。
    """
    tool_llm = QwenToolLLM()

    # 1. 计算专家
    calculator_agent = FunctionAgent(
        tools=[financial_calculator_tool],
        llm=tool_llm,
        name="CalculatorAgent",
        description="用于执行精确的金融计算，如收益率、波动率等。",
        is_async=True
    )

    # 2. 向量数据库检索专家
    vector_rag_agent = FunctionAgent(
        tools=[rag_tools[0]],
        llm=tool_llm,
        name="DocumentSearchAgent",
        description="用于从金融文档和研究报告中检索具体信息。",
        is_async=True
    )

    # 3. 知识图谱查询专家
    graph_rag_agent = FunctionAgent(
        tools=[rag_tools[1]],
        llm=tool_llm,
        name="KnowledgeGraphAgent",
        description="用于查询实体之间的复杂关系和结构化信息。",
        is_async=True
    )

    return {"CalculatorAgent":calculator_agent,"DocumentSearchAgent":vector_rag_agent,"KnowledgeGraphAgent":graph_rag_agent}


