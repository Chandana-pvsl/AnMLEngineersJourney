import os
from langchain_core.tools import tool

WORKSPACE_ROOT = os.environ.get("WORKSPACE_ROOT", "/Users/chandana/Documents/LearnMlCoding/nano_copilot/coding_eval_playground")

@tool
def list_directory_structure() -> str:
    """Recursively lists all python file paths within the workspace to understand the codebase layout."""
    paths = []
    print("workspace root is", WORKSPACE_ROOT)
    if not os.path.isdir(WORKSPACE_ROOT):
        return WORKSPACE_ROOT
    for root, _, files in os.walk(WORKSPACE_ROOT):
        if ".git" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(".py") or file.endswith(".ini"):
                rel_path = os.path.relpath(os.path.join(root, file), WORKSPACE_ROOT)
                paths.append(rel_path)
    return "\n".join(paths)

@tool
def view_file_contents(relative_path: str) -> str:
    """Reads and returns the absolute contents of a specific file within the workspace repository."""
    full_path = os.path.join(WORKSPACE_ROOT, relative_path)
    if not os.path.exists(full_path):
        return f"Error: File {relative_path} does not exist."
    with open(full_path, "r") as f:
        return f.read()

@tool
def patch_file_contents(relative_path: str, new_contents: str) -> str:
    """Overwrites the target file completely with the provided code block text. Use this to apply fixes."""
    full_path = os.path.join(WORKSPACE_ROOT, relative_path)
    if not os.path.exists(full_path):
        return f"Error: Target path {relative_path} missing."
    with open(full_path, "w") as f:
        f.write(new_contents)
    return f"Successfully updated {relative_path}."