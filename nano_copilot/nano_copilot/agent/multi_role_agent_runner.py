import os
from nano_copilot.agent.multi_role_single_agent.graph import compiled_agent
from langchain_core.messages import SystemMessage, HumanMessage

def execute_query(query: str, workspace_path: str) -> dict:
    """Adapter to feed evaluation harness tasks directly into the LangGraph loop."""
    os.environ["WORKSPACE_ROOT"] = os.path.abspath(workspace_path)
    print(os.path.abspath(workspace_path))
    
    # Initialize state inputs
    initial_input = {
        "messages": [
            HumanMessage(content=f"Task Query: {query}")
        ]
    }

    config = {
        "recursion_limit": 50  # Limits the graph to 20 node transitions total
    }
    
    # Run the graph synchronously
    output_state = compiled_agent.invoke(initial_input, config=config)
    print(output_state)

    # Extract structural performance stats for telemetry tracking
    messages_delivered = output_state["messages"]
    tool_calls_count = sum(1 for m in messages_delivered if hasattr(m, 'tool_calls') and m.tool_calls)
    
    return {
        "tokens_used": 0,  # Map tracking variables via model metrics if needed
        "graph_turns": tool_calls_count
    }

if __name__ == "__main__":
    # Create a dummy file to test our agent on
    from dotenv import load_dotenv
    load_dotenv()
    test_file = "calculator.py"
    with open(test_file, "w") as f:
        f.write("def add(a, b):\n    return a + b\n")

    print(f"Initial '{test_file}' content:")
    print(open(test_file).read())

    # Set up our initial input state
    workspace_path = test_file
    query = "Add a subtract function and a multiply function to this file."

    # Run the graph!
    final_output = execute_query(query, workspace_path)

    print("\n--- Final Graph Execution Complete ---")
    print(f"Status: {final_output}")
    print(f"\nUpdated '{test_file}' content:")
    print(open(test_file).read())
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)