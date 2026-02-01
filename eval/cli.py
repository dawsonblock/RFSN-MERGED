"""Command-line entrypoint for SWE-bench evaluation.

This is a thin wrapper around eval.run.run_eval plus the dataset adapter in
eval.swebench.

Usage examples:
  python -m eval --dataset swebench_lite --max-tasks 5
  python -m eval --dataset path/to/swebench.jsonl --task-ids inst1 inst2

Outputs:
  - eval_results/results.jsonl     (rich per-task results)
  - eval_results/swebench.jsonl    (SWE-bench scoring format)
"""

from __future__ import annotations

import argparse
import json
import asyncio
from pathlib import Path

from .run import EvalConfig, run_eval
from .swebench import export_results_to_swebench_format, load_tasks


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RFSN SWE-bench evaluation")
    p.add_argument("--dataset", default="swebench_lite", help="Dataset name or path to JSONL")
    p.add_argument("--max-tasks", type=int, default=None, help="Max tasks to run")
    p.add_argument("--task-ids", nargs="*", default=None, help="Optional list of task instance_ids")
    p.add_argument("--profile", default="swebench_lite", help="Agent profile to use")
    p.add_argument("--parallel", type=int, default=1, help="Parallel tasks")
    p.add_argument("--work-dir", default="./eval_runs", help="Working directory for cloned repos")
    p.add_argument("--results-dir", default="./eval_results", help="Directory for results")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = EvalConfig(
        dataset=args.dataset,
        task_ids=args.task_ids or None,
        max_tasks=args.max_tasks,
        profile_name=args.profile,
        parallel_tasks=max(1, int(args.parallel)),
        work_dir=Path(args.work_dir),
        results_dir=Path(args.results_dir),
    )

    # Ensure directories exist
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load tasks to fail early on path issues.
    _ = load_tasks(cfg.dataset, task_ids=cfg.task_ids, max_tasks=cfg.max_tasks)

    results = asyncio.run(run_eval(cfg))

    # Write rich results
    results_path = cfg.results_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")

    # Write SWE-bench scoring format
    swe_path = cfg.results_dir / "swebench.jsonl"
    export_results_to_swebench_format(
        results=[{
            "task_id": r.task_id,
            "success": r.success,
            # NOTE: model_patch is populated by the agent when available.
            "final_patch": (r.patch_history[-1].get("diff") if r.patch_history else ""),
        } for r in results],
        output_path=swe_path,
    )

    print(f"Wrote: {results_path}")
    print(f"Wrote: {swe_path}")


if __name__ == "__main__":
    main()
