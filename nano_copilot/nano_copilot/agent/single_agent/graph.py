import os
from langgraph.graph import StateGraph, END
from nano_copilot.agent.single_agent.state import AgentState
from nano_copilot.agent.single_agent.nodes import call_model, should_continue
from nano_copilot.tools.registry import tool_node

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

# Add conditional execution paths
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
workflow.add_edge("tools", "agent")

single_agent = workflow.compile()