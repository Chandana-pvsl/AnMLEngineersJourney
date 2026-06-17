# from nano_copilot.tools.registry import file_tool_node
from nano_copilot.agent.multi_role_single_agent.nodes import planner_node, executor_node, verifier_node, route_verification, tool_node
from nano_copilot.agent.multi_role_single_agent.state import MultiRoleAgentState
from langgraph.graph import StateGraph, END


workflow = StateGraph(MultiRoleAgentState)

# 1. Register our three roles + tool execution layer
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("tools", tool_node)
workflow.add_node("verifier", verifier_node)

# 2. Build the execution flow paths
workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")


def route_executor(state: MultiRoleAgentState) -> str:
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", [])
    
    print("\n" + "="*40)
    print("🛰️  [EDGE ROUTER] EVALUATING EXECUTOR OUTPUT")
    print("="*40)
    
    if tool_calls and len(tool_calls) > 0:
        for call in tool_calls:
            print(f"  🛠️  Requested Tool : {call['name']}")
            print(f"  📥  Arguments     : {call.get('args', {})}")
        print("="*40 + "\n➡️  Routing Target: [TOOLS]\n")
        return "tools"
        
    print("  📝  No tool calls detected. Content summary:")
    print(f"  \"{last_msg.content[:150]}...\"")
    print("="*40 + "\n➡️  Routing Target: [VERIFIER]\n")
    return "verifier"

def route_tools(state: MultiRoleAgentState) -> str:
    last_msg = state["messages"][-1]
    tool_name = getattr(last_msg, "name", "unknown_tool")
    
    print("\n" + "~"*40)
    print(f"⚙️  [EDGE ROUTER] TOOL EXECUTION COMPLETED: {tool_name}")
    print("~"*40)
    
    if tool_name == "complete_task":
        print("  🏁  'complete_task' intercepted! Shifting to system validation suite.")
        print("~"*40 + "\n➡️  Routing Target: [VERIFIER]\n")
        return "verifier"
        
    print(f"  🔄  File modifications made by {tool_name}. Returning control to LLM.")
    print("~"*40 + "\n➡️  Routing Target: [EXECUTOR]\n")
    return "executor"


workflow.add_conditional_edges(
    "executor",
    route_executor,
    {
        "tools": "tools",
        "verifier": "verifier"
    }
)

workflow.add_conditional_edges(
    "tools",
    route_tools,
    {
        "verifier": "verifier",
        "executor": "executor"
    }
)

# After tools run, they must always hand control back to the executor to finish its code run
# workflow.add_edge("tools", "executor")

# Route the verifier node's analysis outputs
workflow.add_conditional_edges(
    "verifier",
    route_verification,
    {
        "planner": "planner", # Loop back to rewrite strategy
        END: END              # Terminate workflow
    }
)

compiled_agent = workflow.compile()