from typing import Dict, TypedDict, Annotated, Sequence, List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages

class MultiRoleAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_plan: str          # Tracks the structured, high-level debugging roadmap
    test_traceback: str        # Captures the raw pytest terminal stdout/stderr errors
    loop_count: int            # Tracks execution cycles to prevent runaway local runs
    test_files: list[str]