"""CI Entrypoint - GitHub Action Integration.

Adapts the Controller to run within a CI environment (GitHub Actions).
Reads inputs from environment variables, executes the plan, and 
outputs results to GITHUB_OUTPUT and summary to GITHUB_STEP_SUMMARY.

This module enforces RFSN_BENCH_STRICT=1 for benchmark mode and
writes machine-readable results to results.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rfsn_controller.structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class CIBenchmarkConfig:
    """Configuration for CI benchmark runs."""
    
    dataset: str = "swebench_lite"
    max_tasks: int | None = None
    parallel_tasks: int = 1
    output_dir: Path = Path("runs")
    results_dir: Path = Path("eval_results")
    strict_mode: bool = True
    
    @classmethod
    def from_env(cls) -> "CIBenchmarkConfig":
        """Load configuration from environment variables."""
        max_tasks_str = os.environ.get("INPUT_MAX_TASKS", "0")
        max_tasks = int(max_tasks_str) if max_tasks_str != "0" else None
        
        return cls(
            dataset=os.environ.get("INPUT_DATASET", "swebench_lite"),
            max_tasks=max_tasks,
            parallel_tasks=int(os.environ.get("INPUT_PARALLEL_TASKS", "1")),
            output_dir=Path(os.environ.get("INPUT_OUTPUT_DIR", "runs")),
            results_dir=Path(os.environ.get("INPUT_RESULTS_DIR", "eval_results")),
            strict_mode=os.environ.get("RFSN_BENCH_STRICT", "").lower() in {"1", "true", "yes"},
        )


def verify_strict_mode() -> bool:
    """Verify strict mode is enabled for benchmarks.
    
    Returns:
        True if strict mode is enabled, False otherwise.
    """
    strict = os.environ.get("RFSN_BENCH_STRICT", "").lower() in {"1", "true", "yes"}
    if not strict:
        logger.warning("[CI] RFSN_BENCH_STRICT not set - results may not be valid")
        return False
    
    logger.info("[CI] RFSN_BENCH_STRICT=1 - strict mode enabled")
    return True


def write_github_output(key: str, value: str) -> None:
    """Write a value to GITHUB_OUTPUT for use in subsequent steps.
    
    Args:
        key: Output variable name.
        value: Output value.
    """
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{key}={value}\n")
    else:
        # Fallback for local testing
        logger.info(f"[CI Output] {key}={value}")


def write_step_summary(markdown: str) -> None:
    """Write markdown summary to GITHUB_STEP_SUMMARY.
    
    Args:
        markdown: Markdown content to write.
    """
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(markdown)
    else:
        # Fallback for local testing
        print("[CI Summary]")
        print(markdown)


def generate_results_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate a summary of benchmark results.
    
    Args:
        results: List of result dictionaries.
        
    Returns:
        Summary dictionary with counts and breakdowns.
    """
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total": len(results),
        "passed": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "by_status": {},
        "avg_resolution_time": 0.0,
        "total_llm_calls": 0,
        "total_gate_rejections": 0,
        "security_violations": [],
    }
    
    for r in results:
        # Count by status
        status = r.get("status", "UNKNOWN")
        summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
        
        # Aggregate metrics
        summary["total_llm_calls"] += r.get("llm_calls", 0)
        summary["total_gate_rejections"] += r.get("gate_rejections", 0)
        
        # Collect security violations
        for v in r.get("security_violations", []):
            summary["security_violations"].append({
                "task_id": r.get("task_id"),
                "violation": v,
            })
    
    # Calculate averages
    if results:
        total_time = sum(r.get("resolution_time", 0) for r in results)
        summary["avg_resolution_time"] = total_time / len(results)
    
    return summary


def run_ci_benchmark(config: CIBenchmarkConfig) -> int:
    """Run RFSN benchmark in CI mode.
    
    Args:
        config: Benchmark configuration.
        
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    start_time = time.time()
    
    # Verify strict mode
    is_strict = verify_strict_mode()
    if config.strict_mode and not is_strict:
        logger.error("FATAL: RFSN_BENCH_STRICT must be set for benchmark mode")
        write_github_output("success", "false")
        write_github_output("error", "strict_mode_not_set")
        return 1
    
    # Create directories
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[CI] Starting benchmark: dataset={config.dataset}")
    logger.info(f"[CI] Max tasks: {config.max_tasks or 'all'}")
    logger.info(f"[CI] Parallel tasks: {config.parallel_tasks}")
    
    try:
        # Import and run the evaluation
        import asyncio
        from eval.run import EvalConfig, run_eval
        
        eval_config = EvalConfig(
            dataset=config.dataset,
            max_tasks=config.max_tasks,
            parallel_tasks=config.parallel_tasks,
            work_dir=config.output_dir,
            results_dir=config.results_dir,
        )
        
        # Run evaluation
        results = asyncio.run(run_eval(eval_config))
        
        # Convert results to dicts
        result_dicts = [r.to_dict() for r in results]
        
        # Generate summary
        summary = generate_results_summary(result_dicts)
        summary["dataset"] = config.dataset
        summary["strict_mode"] = is_strict
        summary["total_runtime"] = time.time() - start_time
        
        # Write results.json
        results_json_path = Path("results.json")
        with open(results_json_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"[CI] Wrote summary to {results_json_path}")
        
        # Write GitHub outputs
        pass_rate = summary["passed"] / max(1, summary["total"]) * 100
        write_github_output("success", str(summary["passed"] == summary["total"]).lower())
        write_github_output("total_tasks", str(summary["total"]))
        write_github_output("passed_tasks", str(summary["passed"]))
        write_github_output("failed_tasks", str(summary["failed"]))
        write_github_output("pass_rate", f"{pass_rate:.1f}")
        
        # Write step summary
        summary_md = f"""## RFSN Benchmark Results

| Metric | Value |
|--------|-------|
| Dataset | {config.dataset} |
| Total Tasks | {summary['total']} |
| Passed | {summary['passed']} |
| Failed | {summary['failed']} |
| Pass Rate | {pass_rate:.1f}% |
| Total Runtime | {summary['total_runtime']:.1f}s |
| LLM Calls | {summary['total_llm_calls']} |
| Gate Rejections | {summary['total_gate_rejections']} |

### Status Breakdown

"""
        for status, count in sorted(summary["by_status"].items()):
            summary_md += f"- **{status}**: {count}\n"
        
        if summary["security_violations"]:
            summary_md += "\n### ⚠️ Security Violations\n\n"
            for v in summary["security_violations"][:10]:  # Limit to 10
                summary_md += f"- `{v['task_id']}`: {v['violation']}\n"
        
        write_step_summary(summary_md)
        
        # Return code based on success
        if summary["passed"] == summary["total"]:
            logger.info("[CI] Benchmark completed: all tasks passed")
            return 0
        else:
            logger.warning(f"[CI] Benchmark completed: {summary['failed']} tasks failed")
            return 0  # Still return 0, failures are expected in benchmarks
            
    except SystemExit as e:
        # Re-raise SystemExit from strict mode violations
        logger.error(f"[CI] SystemExit: {e}")
        write_github_output("success", "false")
        write_github_output("error", str(e))
        return 1
        
    except Exception as e:
        logger.exception(f"[CI] Benchmark failed with error: {e}")
        write_github_output("success", "false")
        write_github_output("error", str(e)[:200])
        return 1


def main() -> int:
    """Main entry point for CI benchmark."""
    config = CIBenchmarkConfig.from_env()
    return run_ci_benchmark(config)


if __name__ == "__main__":
    sys.exit(main())
