import os
import subprocess
from langchain_core.tools import tool

WORKSPACE_ROOT = os.environ.get("WORKSPACE_ROOT", "/Users/chandana/Documents/LearnMlCoding/nano_copilot/coding_eval_playground")

@tool
def execute_system_tests(test_command: str) -> str:
    """Runs a specific verification command via pytest and returns stdout/stderr logs. 
    Use this to see if your fix worked, or to read traceback errors if it failed.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(WORKSPACE_ROOT, "src")
    
    result = subprocess.run(
        test_command.split(),
        cwd=WORKSPACE_ROOT,
        env=env,
        capture_output=True,
        text=True
    )
    
    output = result.stdout + "\n" + result.stderr
    if result.returncode == 0:
        return f"SUCCESS: All target test cases passed!\n{output}"
    return f"FAILURE: Tests broken. Traceback analysis:\n{output}"