from nano_copilot.tools.file_ops import list_directory_structure, view_file_contents, patch_file_contents
from nano_copilot.tools.testing import execute_system_tests
from nano_copilot.tools.general import complete_task
from langgraph.prebuilt import ToolNode


FILE_TOOLS = [
    list_directory_structure,
    view_file_contents,
    patch_file_contents,
]

TEST_TOOLS = [
    execute_system_tests
]

GENERAL_TOOLS = [
    complete_task
]

ALL_TOOLS = FILE_TOOLS + TEST_TOOLS + GENERAL_TOOLS

# tool_node = ToolNode(ALL_TOOLS)

# file_tool_node = ToolNode(FILE_TOOLS)