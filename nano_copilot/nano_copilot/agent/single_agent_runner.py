import os
from nano_copilot.agent.single_agent.graph import single_agent
from langchain_core.messages import SystemMessage, HumanMessage

def execute_query(query: str, workspace_path: str) -> dict:
    """Adapter to feed evaluation harness tasks directly into the LangGraph loop."""
    os.environ["WORKSPACE_ROOT"] = os.path.abspath(workspace_path)
    print(os.path.abspath(workspace_path))
    system_prompt = (
        "You are an expert staff senior software engineering agent specialized in debugging.\n"
        "Your objective is to fix codebases by invoking tools to read, edit, and test files.\n"
        "Follow this execution strategy:\n"
        "1. If it is a file read the file\n"
        "2. If it is a directory: List the directory layout to locate files\n"
        "3. View the contents of files related to the problem.\n"
        "4. Run the verification test command using `execute_system_tests` to see the failure trace.\n"
        "5. Edit files by completely rewriting them via `patch_file_contents`.\n"
        "6. Run the test command again. Do not stop until tests pass successfully.\n"
    )
    
    # Initialize state inputs
    initial_input = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Workspace Path: {os.path.abspath(workspace_path)} Task Query: {query}")
        ]
    }

    config = {
        "recursion_limit": 30  # Limits the graph to 20 node transitions total
    }
    
    # Run the graph synchronously
    output_state = single_agent.invoke(initial_input, config=config)
    
    # print(output_state)
    # Extract structural performance stats for telemetry tracking
    messages_delivered = output_state["messages"]
    tool_calls_count = sum(1 for m in messages_delivered if hasattr(m, 'tool_calls') and m.tool_calls)
    
    return {
        "tokens_used": 0,  # Map tracking variables via model metrics if needed
        "graph_turns": tool_calls_count
    }

if __name__ == "__main__":
    # Create a dummy file to test our agent on
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