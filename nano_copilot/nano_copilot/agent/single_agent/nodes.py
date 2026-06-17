import os
from nano_copilot.agent.single_agent.state import NanoCopilotState, AgentState
# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from nano_copilot.tools.registry import ALL_TOOLS, tool_node
from nano_copilot.model.llm import llm
from typing import Dict
from langgraph.graph import END
from langchain_core.messages import AIMessage
import json
import uuid

# =====================================================================
# 2. THE NODES (Functions that do work and update the state)
# =====================================================================

# def researcher_node(state: NanoCopilotState):
#     """Reads the targeted file and asks the LLM to write the new version."""
#     print("\n[Node 1] Researcher is reading the file and prompting the LLM...")
    
#     file_path = state["target_file"]
    
#     # Read the local file
#     if os.path.exists(file_path):
#         with open(file_path, "r") as f:
#             current_code = f.read()
#     else:
#         current_code = "# File does not exist yet. Creating a new one."

#     # Prompt tailored to return raw code only
#     prompt = f"""
#     You are an expert developer.
#     We have a file located at: {file_path}
    
#     Here is the current content of the file:
#     ---
#     {current_code}
#     ---
    
#     User Goal: {state['instructions']}
    
#     Output the ENTIRE modified file. Do not include any markdown wrappers like ```python. 
#     Just give me the raw code ready to save.
#     """
    
#     response = llm.invoke(prompt)
    
#     # Return a dictionary containing the keys we want to update in the State
#     return {
#         "file_content": current_code,
#         "proposed_edit": response.content.strip()
#     }


# def editor_node(state: NanoCopilotState):
#     """Takes the proposed edit from the state and writes it back to disk."""
#     print("\n[Node 2] Editor is overwriting the file on disk...")
    
#     file_path = state["target_file"]
#     new_code = state["proposed_edit"]
    
#     try:
#         with open(file_path, "w") as f:
#             f.write(new_code)
#         return {"status": "success"}
#     except Exception as e:
#         print(f"Error writing file: {e}")
#         return {"status": f"failed: {str(e)}"}
def idx_or_generated_uuid():
    return str(uuid.uuid4())

def add_tool_call_to_response(response):
    if isinstance(response, AIMessage) and not response.tool_calls:
        cleaned_content = response.content.strip()
        
        # If it looks like a JSON block, attempt an explicit extraction
        if cleaned_content.startswith("{") and cleaned_content.endswith("}"):
            try:
                parsed = json.loads(cleaned_content)
                
                # Check if it contains standard tool signature properties
                if "name" in parsed:
                    print(f"\n[PATCH] Intercepted raw string tool call from local LLM: {parsed['name']}")
                    
                    # Map the raw string fields back into standard LangGraph tool call objects
                    response.tool_calls = [{
                        "name": parsed["name"],
                        "args": parsed.get("arguments", parsed.get("args", {})),
                        "id": f"call_{idx_or_generated_uuid()}" if not hasattr(response, 'id') else response.id
                    }]
            except json.JSONDecodeError:
                pass # Not valid JSON text, continue treating it as ordinary text conversation
                
    # Logging debug trace to terminal to confirm it successfully converted
    print(f"➡️ In-Flight Graph State -> Active Tool Calls: {getattr(response, 'tool_calls', [])}\n")
    
    return response

def call_model(state: AgentState) -> Dict:
    messages = state["messages"]
    tool_llm = llm.bind_tools(ALL_TOOLS, strict=True)
    response = tool_llm.invoke(messages)
    response = add_tool_call_to_response(response)
    return {"messages": [response]}

# 4. Conditional Routing Router Logic
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    # If model didn't call a tool, it is done thinking and providing its final answer
    if not last_message.tool_calls:
        return END
    return "tools"