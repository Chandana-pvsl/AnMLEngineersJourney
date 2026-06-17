import argparse
import os
import json
import subprocess
import time
from typing import List, Dict, Any, Callable
from nano_copilot.eval.eval_dataset import EVAL_DATASET

class LocalEvalHarness:
    def __init__(self, target_repo_path: str):
        self.repo_path = os.path.abspath(target_repo_path)

    def reset_environment(self):
        """Rollbacks any mutations executed by the agent back to baseline git head."""
        subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "clean", "-fd"], cwd=self.repo_path, capture_output=True)

    def run_verification(self, test_command: str) -> bool:
        """Executes targeted testing framework inside sandbox environment."""
        # Run through virtualenv subprocess invocation
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(self.repo_path, "src")
        result = subprocess.run(
            test_command.split(),
            cwd=self.repo_path,
            env=env,
            capture_output=True,
            text=True
        )
        print(result)
        return result.returncode == 0

    def run_suite(self, agent_runner: Callable[[str, str], dict]) -> Dict[str, Any]:
        scores = {"easy": [], "medium": [], "medium_hard": [], "hard": []}
        telemetry_logs = []

        print("\n=== STARTING AGENTIC ARCHITECTURE EVALUATION SUITE ===\n")
        
        for task in EVAL_DATASET:
            print(f"📋 Task ID: {task['task_id']} | Complexity: {task['complexity']}")
            self.reset_environment()
            
            # 1. Double check that the test fails *before* the agent touches it
            if self.run_verification(task["test_command"]):
                print(f"⚠️ Warning: Test passed before agent intervention for {task['task_id']}. Skipping.")
                continue

            start_time = time.time()
            
            # 2. Invoke the target agent loop (this will hook into LangGraph in Post 1)
            error = None
            try:
                agent_metrics = agent_runner(task["query"], self.repo_path)
            except Exception as e:
                error = str(e)
            
            duration = time.time() - start_time
            
            if not error:
                # 3. Check if the agent successfully fixed the code
                is_resolved = self.run_verification(task["test_command"])
                
                # 4. Extract generated Git diff patch
                diff_patch = subprocess.run(
                    ["git", "diff"], cwd=self.repo_path, capture_output=True, text=True
                ).stdout
            else:
                is_resolved = False
                diff_patch = ""
                agent_metrics = {}

            # Record metrics
            scores[task["complexity"]].append(1 if is_resolved else 0)
            
            log_entry = {
                "task_id": task["task_id"],
                "complexity": task["complexity"],
                "resolved": is_resolved,
                "duration_seconds": round(duration, 2),
                "tokens_used": agent_metrics.get("tokens_used", 0),
                "graph_turns": agent_metrics.get("graph_turns", 0),
                "patch": diff_patch,
                "error": error
            }
            telemetry_logs.append(log_entry)
            
            status_icon = "✅ SUCCESS" if is_resolved else "❌ FAILED"
            print(f"Status: {status_icon} | Time: {log_entry['duration_seconds']}s | Turns: {log_entry['graph_turns']}\n")

        # Compile final telemetry summary
        summary = {
            "total_tasks": len(telemetry_logs),
            "success_rate": sum(1 for log in telemetry_logs if log["resolved"]) / len(telemetry_logs) * 100,
            "breakdown": {comp: (sum(v)/len(v)*100 if v else 0) for comp, v in scores.items()}
        }
        
        # print("=== FINAL EVALUATION SUMMARY ===")
        # print(json.dumps(summary, indent=2))
        return {"summary": summary, "detail": telemetry_logs}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A coding agent eval harness"
    )
    parser.add_argument(
        "--target_repo_path", 
        required=False,
        type=str,
        default="/Users/chandana/Documents/LearnMlCoding/nano_copilot/coding_eval_playground",
        help="The path to the folder you want to evaluate the agent on",
    )
    args = parser.parse_args()
    # Standard dummy mock agent to test the harness functionality on day one
    def sample_failing_agent(query: str, workspace_path: str) -> dict:
        # This mirrors a dummy agent structure returning token logs
        # It does nothing to modify code, so tests should continue to fail
        return {"tokens_used": 4200, "graph_turns": 4}

    harness = LocalEvalHarness(target_repo_path=args.target_repo_path)
    harness.run_suite(sample_failing_agent)