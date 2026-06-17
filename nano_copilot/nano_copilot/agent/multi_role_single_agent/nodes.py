import subprocess
import os, json
from nano_copilot.agent.multi_role_single_agent.state import MultiRoleAgentState
from typing import Dict
from langchain_core.messages import SystemMessage
from nano_copilot.model.llm import get_agent_model
from langchain_core.messages import AIMessage
from langgraph.graph import END
from nano_copilot.tools.registry import FILE_TOOLS, TEST_TOOLS
from nano_copilot.tools.general import complete_task
from nano_copilot.tools.file_ops import list_directory_structure
from langgraph.prebuilt import ToolNode


WORKSPACE_ROOT = os.environ.get("WORKSPACE_ROOT", "/Users/chandana/Documents/LearnMlCoding/nano_copilot/coding_eval_playground")

tool_node = ToolNode(FILE_TOOLS+[complete_task])

def planner_node(state: MultiRoleAgentState) -> Dict:
    messages = state["messages"]
    traceback = state.get("test_traceback", "None. This is the initial run.")
    # "You are an elite software architect and PLANNER.\n"
    #     "Your task is to review the user's issue and design a bulletproof debugging strategy.\n"
    #     "CRITICAL: Do not attempt to patch files or run tests yourself right now.\n"
    #     f"LATEST TEST FAILURES:\n{traceback}\n\n"
    #     "Formulate a precise, step-by-step text plan explaining which files need to be modified, "
    #     "what types should be changed, and how to avoid side-effects. If a previous plan failed, "
    #     "analyze the traceback above and pivot to a completely new strategy."
    planner_prompt = SystemMessage(content=(
         "You are an elite software architect and PLANNER.\n"
        "Your ONLY job is to analyze the repository history and build a code modification roadmap.\n"
        f"LATEST CODE BASE TEST ERRORS:\n{traceback}\n\n"
        "Output a clear, step-by-step text plan specifying which source files need code changes.\n\n"
        "CRITICAL RULES FOR PLAN GENERATION:\n"
        "1. Do NOT include ANY steps about running tests, executing pytest, or verifying code.\n"
    ))
    
    # We pass the history along with the fresh planning directive
    llm = get_agent_model()
    llm = llm.bind_tools([list_directory_structure], strict=True)
    llm = get_agent_model()
    response = llm.invoke([planner_prompt] + list(messages))
    return {
        "current_plan": response.content,
        "messages": [response]
    }


def executor_node(state: MultiRoleAgentState) -> Dict:
    messages = state["messages"]
    current_plan = state["current_plan"]
    # "You are a precise production systems systems developer and EXECUTOR.\n"
        # f"YOUR ASSIGNED TARGET PLAN:\n{current_plan}\n\n"
        # "Your objective is to read the target files and apply clean modifications using "
        # "the `patch_file_contents` tool to fulfill the plan exactly.\n"
        # "CRITICAL: Do not change the overall strategy. Focus entirely on syntax, accurate "
        # "imports, type alignment, and writing comprehensive logic updates."
    executor_prompt = SystemMessage(content=(
        "You are a precise production systems developer and EXECUTOR.\n"
        f"YOUR ASSIGNED STRATEGY TO IMPLEMENT:\n{current_plan}\n\n"
        "Your ONLY objective is to read the codebase files and write code modifications using your tools."
        "When you are done with the task output the string - \"DONE\" \n\n"
        "⚠️ CRITICAL EXECUTION GUARDRAILS:\n"
        "1. You are strictly forbidden from writing or running tests.\n"
        "2. If the plan mentions testing, verifying, or running pytest, IGNORE IT COMPLETELY.\n"
        "3. Do not guess paths or create placeholder files to satisfy testing steps.\n"
        "4. Once you have applied the structural code fixes to the source files, STOP and yield control immediately. Do not attempt to verify your work. Output - \"DONE\". Nothing else "
    ))
    
    llm = get_agent_model()
    llm = llm.bind_tools(FILE_TOOLS+[complete_task], strict=True)
    response = llm.invoke([executor_prompt] + list(messages))
    
    # --- AUTO-PARSING COMPATIBILITY PATCH (From our previous step) ---
    import json
    if isinstance(response, AIMessage) and not response.tool_calls:
        cleaned = response.content.strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            try:
                parsed = json.loads(cleaned)
                if "name" in parsed:
                    response.tool_calls = [{
                        "name": parsed["name"],
                        "args": parsed.get("arguments", parsed.get("args", {})),
                        "id": "call_exec_patch_1"
                    }]
            except json.JSONDecodeError:
                pass

    return {"messages": [response]}


# def verifier_node(state: MultiRoleAgentState) -> Dict:
#     # Instead of letting the LLM decide whether to run tests, this node 
#     # executes the validation suite automatically against the current codebase state.
#     print("🧪 Verifier active: Booting pytest engine...")
#     env = os.environ.copy()
#     env["PYTHONPATH"] = os.path.join(WORKSPACE_ROOT, "src")
    
#     # Run the tests at the repository root level
#     result = subprocess.run(
#         ["pytest"],
#         cwd=WORKSPACE_ROOT,
#         env=env,
#         capture_output=True,
#         text=True
#     )
    
#     output = result.stdout + "\n" + result.stderr
#     current_loops = state.get("loop_count", 0) + 1
    
#     if result.returncode == 0:
#         return {
#             "test_traceback": "SUCCESS",
#             "loop_count": current_loops
#         }
        
#     return {
#         "test_traceback": output,
#         "loop_count": current_loops
#     }

def verifier_node(state: MultiRoleAgentState) -> Dict:
    print("🧪 Verifier active: Isolating exact test methods...")
    
    original_query = state["messages"][0].content
    current_loops = state.get("loop_count", 0) + 1
    
    # 1. To identify exact methods, the LLM needs to see what's inside the test files.
    # We gather the test code context dynamically.
    test_context_dump = ""
    tests_dir = os.path.join(WORKSPACE_ROOT, "tests")
    
    if os.path.exists(tests_dir):
        for root, _, files in os.walk(tests_dir):
            for file in files:
                if file.startswith("test_") and file.endswith(".py"):
                    rel_path = os.path.relpath(os.path.join(root, file), WORKSPACE_ROOT)
                    try:
                        with open(os.path.join(root, file), "r") as f:
                            # We only extract lines containing 'def test_' to keep context clean
                            test_methods = [line.strip() for line in f if "def test_" in line]
                        test_context_dump += f"\nFile: {rel_path}\nFound Methods:\n" + "\n".join(test_methods) + "\n"
                    except Exception:
                        pass

    # 2. Instruct the LLM to map files to exact methods using structured JSON
    verifier_selector_prompt = SystemMessage(content=(
        "You are an expert QA Automation Engineer specialized in test isolation.\n"
        "Review the user's objective and the available test files along with their functions. "
        "Identify the EXACT test functions/methods that must be executed to validate this change.\n\n"
        f"USER OBJECTIVE:\n{original_query}\n\n"
        f"AVAILABLE TEST METHODS CONTEXT:\n{test_context_dump}\n\n"
        "CRITICAL RULES:\n"
        "1. Return a raw JSON object where keys are relative file paths and values are arrays of specific test function names.\n"
        "   Example format:\n"
        "   {\n"
        "     \"tests/test_unit.py\": [\"test_addition_positive_numbers\", \"test_subtraction\"],\n"
        "     \"tests/test_integration.py\": [\"test_api_v1_flow\"]\n"
        "   }\n"
        "2. Do not include markdown formatting, code blocks (like ```json), or text explanations.\n"
        "3. Only target functions that actually exist in the context above.\n"
        "4. If no specific methods match, return an empty object: {}\n"
    ))
    
    llm = get_agent_model()
    selector_response = llm.invoke([verifier_selector_prompt])
    cleaned_output = selector_response.content.strip()
    
    print("FIles for testing ", cleaned_output)
    # Strip markdown code blocks if the local model leaked them
    if "```" in cleaned_output:
        cleaned_output = cleaned_output.split("```")[1]
        if cleaned_output.startswith("json"):
            cleaned_output = cleaned_output[4:]
    cleaned_output = cleaned_output.strip()

    # 3. Parse the target map safely
    targets_map = {}
    try:
        if cleaned_output and cleaned_output != "{}":
            targets_map = json.loads(cleaned_output)
    except Exception as e:
        print(f"⚠️ Failed to parse LLM method targets JSON. Error: {e}. Falling back to global pytest.")
        targets_map = {}

    # 4. Execute targeted tests sequentially using pytest syntax: file.py::method_name
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(WORKSPACE_ROOT, "src")
    
    combined_output = ""
    all_passed = True

    # Scenario A: Fallback to blanket pytest if no specific target targets were parsed
    if not targets_map:
        print("⚡ Running entire test suite as fallback...")
        result = subprocess.run(["pytest"], cwd=WORKSPACE_ROOT, env=env, capture_output=True, text=True)
        combined_output = result.stdout + "\n" + result.stderr
        all_passed = (result.returncode == 0)
        
    # Scenario B: Target specific methods precisely
    else:
        for file_path, methods in targets_map.items():
            full_file_path = os.path.join(WORKSPACE_ROOT, file_path)
            if not os.path.exists(full_file_path):
                print(f"⏩ Skipping invalid test file path: {file_path}")
                continue
                
            for method in methods:
                # Build the explicit target string syntax: path/to/file.py::test_method
                pytest_target = f"{file_path}::{method}"
                print(f"🎯 Running isolated verification: pytest {pytest_target}")
                
                result = subprocess.run(
                    ["pytest", pytest_target], 
                    cwd=WORKSPACE_ROOT, 
                    env=env, 
                    capture_output=True, 
                    text=True
                )
                
                if result.returncode != 0:
                    all_passed = False
                    output = result.stdout + "\n" + result.stderr
                    
                    if "======= FAILURES =======" in output:
                        failure_summary = output.split("======= FAILURES =======")[-1]
                    else:
                        failure_summary = output[-1200:]
                        
                    combined_output += f"\n--- FAILURES IN {pytest_target} ---\n{failure_summary}\n"

    if all_passed:
        return {
            "test_traceback": "SUCCESS",
            "loop_count": current_loops
        }
        
    return {
        "test_traceback": combined_output.strip(),
        "loop_count": current_loops
    }

# def verifier_node(state: MultiRoleAgentState) -> Dict:
#     print("🧪 Verifier active: Identifying all relevant test targets...")
    
#     # 1. Gather repository context
#     directory_layout = list_directory_structure.invoke({})
#     original_query = state["messages"][0].content
    
#     # 2. Instruct the LLM to return a strict JSON array of file paths
#     verifier_selector_prompt = SystemMessage(content=(
#         "You are an expert QA Automation Engineer specialized in impact analysis.\n"
#         "Review the user's objective and the repository layout, then identify ALL test files "
#         "that must be executed to fully validate this change and check for regressions.\n\n"
#         f"USER OBJECTIVE:\n{original_query}\n\n"
#         f"AVAILABLE FILES IN REPOSITORY:\n{directory_layout}\n\n"
#         "CRITICAL RULES:\n"
#         "1. Return a raw JSON array of strings containing relative file paths.\n"
#         "   Example format: {\"files\": [\"tests/test_unit.py\", \"tests/test_integration.py\"]}\n"
#         "2. Do not include markdown formatting, code blocks (like ```json), or text explanations.\n"
#         "3. Only include files that actually exist in the layout above.\n"
#         "4. If the entire suite needs to be checked, return an empty array: []"
#     ))
    
#     llm = get_agent_model()
#     selector_response = llm.invoke([verifier_selector_prompt])
#     cleaned_output = selector_response.content.strip()
    
#     # Simple fallback parsing if local model includes markdown code blocks
#     if "```" in cleaned_output:
#         cleaned_output = cleaned_output.split("```")[1]
#         if cleaned_output.startswith("json"):
#             cleaned_output = cleaned_output[4:]
#     cleaned_output = cleaned_output.strip()

#     # 3. Parse target files safely
#     test_files = []
#     try:
#         test_files = json.loads(cleaned_output)
#         if isinstance(test_files, dict):
#             test_files = test_files["files"]
#         if not isinstance(test_files, list):
#             test_files = [str(test_files)]
#     except Exception as e:
#         print(f"⚠️ Failed to parse LLM test list JSON. Falling back to running all tests. Error: {e}")
#         test_files = [] # Empty list defaults to global pytest execution

#     # 4. Execute the targeted test suite sequentially
#     env = os.environ.copy()
#     env["PYTHONPATH"] = os.path.join(WORKSPACE_ROOT, "src")
    
#     combined_output = ""
#     all_passed = True
#     current_loops = state.get("loop_count", 0) + 1

#     # Scenario A: Run full test suite if list is empty or parsing failed
#     if not test_files:
#         print("⚡ Running entire pytest suite...")
#         result = subprocess.run(["pytest"], cwd=WORKSPACE_ROOT, env=env, capture_output=True, text=True)
#         combined_output = result.stdout + "\n" + result.stderr
#         all_passed = (result.returncode == 0)
    
#     # Scenario B: Target explicit files sequentially
#     else:
#         for test_file in test_files:
#             full_test_path = os.path.join(WORKSPACE_ROOT, test_file)
#             if not os.path.exists(full_test_path):
#                 print(f"⏩ Skipping hallucinated file path: {full_test_path}")
#                 continue
                
#             print(f"⚡ Running targeted verification: pytest {test_file}")
#             result = subprocess.run(["pytest", test_file], cwd=WORKSPACE_ROOT, env=env, capture_output=True, text=True)
            
#             if result.returncode != 0:
#                 all_passed = False
#                 # Isolate the core failure block to prevent context window bloating
#                 output = result.stdout + "\n" + result.stderr
#                 if "======= FAILURES =======" in output:
#                     failure_summary = output.split("======= FAILURES =======")[-1]
#                 else:
#                     failure_summary = output[-1500:]
                
#                 combined_output += f"\n--- FAILURES IN {test_file} ---\n{failure_summary}\n"

#     # 5. Return aggregated loop evaluations back to the graph state
#     if all_passed:
#         return {
#             "test_traceback": "SUCCESS",
#             "loop_count": current_loops
#         }
        
#     return {
#         "test_traceback": combined_output.strip(),
#         "loop_count": current_loops
#     }



def route_verification(state: MultiRoleAgentState) -> str:
    # Exit Rule A: The tests passed cleanly
    if state["test_traceback"] == "SUCCESS":
        print("🎉 SUCCESS: All tests passed cleanly. Exiting graph!")
        return END
        
    # Exit Rule B: Safety valve to prevent local resource depletion
    if state.get("loop_count", 0) >= 4:
        print("🚨 CRITICAL: Runaway loop safety valve triggered. Exiting graph.")
        return END
        
    # Standard Rule: Tests failed, route back to the Planner with the new traceback details
    print(f"❌ Tests failed (Cycle {state['loop_count']}). Routing back to PLANNER to adapt...")
    return "planner"