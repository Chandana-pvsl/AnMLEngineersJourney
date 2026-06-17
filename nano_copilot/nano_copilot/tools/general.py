from langchain_core.tools import tool

@tool
def complete_task(explanation: str) -> str:
    """Call this tool ONLY when you have fully completed all modifications 
    requested in the plan.
    
    Args:
        explanation: A brief summary of the changes you implemented.
    """
    return f"Task marked as complete. Summary: {explanation}"