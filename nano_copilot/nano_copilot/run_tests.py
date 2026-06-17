from nano_copilot.eval.eval_runner import LocalEvalHarness
# from nano_copilot.agent.single_agent_runner import execute_query
from nano_copilot.agent.multi_role_agent_runner import execute_query
import json
from dotenv import load_dotenv
load_dotenv()

target_repo_path = "/Users/chandana/Documents/LearnMlCoding/nano_copilot/coding_eval_playground"
# execute_query_fn = lambda x: execute_query(x, target_repo_path)
harness = LocalEvalHarness(target_repo_path=target_repo_path)
output = harness.run_suite(execute_query)

with open("/Users/chandana/Documents/LearnMlCoding/nano_copilot/data/results/multi_role_agent_results.json", "w") as f:
    json.dump(output, f)