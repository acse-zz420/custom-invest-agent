from typing import List, Dict, Any, Optional
from pydantic import Field
from llama_index.core.workflow import Event, StartEvent, StopEvent
from llama_index.core.schema import NodeWithScore


class UserInputEvent(StartEvent):
    user_msg: str
    chat_history: List[Dict[str, Any]] = Field(default_factory=list)
    current_loop: int = 1
    previous_attempt: Optional[str] = None # 上一轮的草稿答案
    critique_feedback: Optional[str] = None # 上一轮的批评意见

class FinalOutputEvent(StopEvent):
    response: str

class RAGTriggerEvent(Event):
    user_msg: str
    query: str
    current_loop: int = 1
    previous_attempt: Optional[str] = None
    critique_feedback: Optional[str] = None

class SimpleChatTriggerEvent(Event):
    user_msg: str

class CalculatorTriggerEvent(Event):
    user_msg: str
    query: str
    current_loop: int = 1

class SearchResultEvent(Event):
    source: str
    results: List[NodeWithScore]
    user_msg: str
    query: str
    current_loop: int

class AggregatedContextEvent(Event):
    user_msg: str
    query: str
    current_loop: int
    fused_context: str


# --- 核心修改 1: 统一事件名称 ---
#    我们将所有“触发XX步骤”的事件都命名为 XxxEvent
class SummarizationEvent(Event): # 不再是 Trigger
    user_msg: str
    final_context: str
    current_loop: int

# --- 核心修改 2: 补全并重命名事件 ---
class CritiqueEvent(Event): # 之前叫 AnswerWithContextEvent
    user_msg: str
    final_context: str
    preliminary_answer: str
    current_loop: int
