"""
Example: Basic SWE-bench Evaluation

This example shows how to run a basic evaluation on SWE-bench tasks
using the RFSN Benchmark framework.
"""

import asyncio
from pathlib import Path

from eval.run import run_eval, EvalConfig
from eval.swebench import load_tasks


async def main():
    """Run a basic SWE-bench evaluation."""
    
    # Configuration
    config = EvalConfig(
        dataset="swebench_lite",
        max_tasks=5,  # Start small
        output_dir=Path("runs"),
        parallel_workers=1,  # Serial for debugging
    )
    
    # Load tasks
    tasks = load_tasks(
        dataset=config.dataset,
        max_tasks=config.max_tasks,
    )
    print(f"Loaded {len(tasks)} tasks")
    
    # Run evaluation
    print("Starting evaluation...")
    results = await run_eval(config)
    
    # Print summary
    passed = sum(1 for r in results if r.success)
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(results)} passed ({passed/len(results):.1%})")
    print(f"{'='*50}")
    
    for result in results:
        status = "✅" if result.success else "❌"
        print(f"  {status} {result.task_id}: {result.status.value}")


if __name__ == "__main__":
    asyncio.run(main())
