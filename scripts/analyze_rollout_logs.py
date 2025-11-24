#!/usr/bin/env python3
"""Analyze rollout logs and generate statistics."""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import sys


def load_logs_from_step(step_dir: Path) -> List[Dict[str, Any]]:
    """Load all rollout logs from a step directory."""
    logs = []
    for rollout_file in sorted(step_dir.glob("*/rollout_*.json")):
        with open(rollout_file) as f:
            log = json.load(f)
            log["_file_path"] = str(rollout_file)
            logs.append(log)
    return logs


def analyze_step(step_dir: Path, verbose: bool = False) -> Dict[str, Any]:
    """Analyze all logs from a training step."""
    logs = load_logs_from_step(step_dir)
    
    if not logs:
        return {}
    
    stats = {
        "step": logs[0]["step"],
        "num_rollouts": len(logs),
        "rewards": [],
        "turns": [],
        "correct_answers": 0,
        "attempted_answers": 0,
        "found_correct_email": 0,
        "read_correct_email": 0,
        "ran_out_of_turns": 0,
        "gave_up_early": 0,
        "repeated_searches": 0,
        "zero_result_searches": 0,
        "avg_duration": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "tool_usage": defaultdict(int),
        "error_types": defaultdict(int),
    }
    
    for log in logs:
        rubric = log.get("rubric", {})
        
        # Basic stats
        stats["rewards"].append(log.get("reward", 0))
        stats["turns"].append(rubric.get("num_turns", 0))
        
        # Success metrics
        if rubric.get("answer_correct"):
            stats["correct_answers"] += 1
        if rubric.get("attempted_answer"):
            stats["attempted_answers"] += 1
        if rubric.get("ever_found_right_email"):
            stats["found_correct_email"] += 1
        if rubric.get("ever_read_right_email"):
            stats["read_correct_email"] += 1
        
        # Failure modes
        if rubric.get("ran_out_of_turns"):
            stats["ran_out_of_turns"] += 1
        if rubric.get("gave_up_too_early"):
            stats["gave_up_early"] += 1
        if rubric.get("num_repeated_searches", 0) > 0:
            stats["repeated_searches"] += 1
        if rubric.get("num_zero_result_searches", 0) > 0:
            stats["zero_result_searches"] += 1
        
        # Error types
        if rubric.get("cant_parse_tool_call"):
            stats["error_types"]["parse_error"] += 1
        if rubric.get("bad_tool_call_name"):
            stats["error_types"]["bad_tool_name"] += 1
        if rubric.get("bad_tool_call_args"):
            stats["error_types"]["bad_tool_args"] += 1
        
        # Resource usage
        stats["avg_duration"] += log.get("duration_seconds", 0)
        stats["total_input_tokens"] += log.get("total_input_tokens", 0)
        stats["total_output_tokens"] += log.get("total_output_tokens", 0)
        
        # Tool usage
        for tool_call in log.get("tool_calls", []):
            tool_name = tool_call.get("tool_name", "unknown")
            stats["tool_usage"][tool_name] += 1
    
    # Calculate averages
    n = len(logs)
    stats["avg_reward"] = sum(stats["rewards"]) / n if n > 0 else 0
    stats["avg_turns"] = sum(stats["turns"]) / n if n > 0 else 0
    stats["accuracy"] = stats["correct_answers"] / n if n > 0 else 0
    stats["attempt_rate"] = stats["attempted_answers"] / n if n > 0 else 0
    stats["found_rate"] = stats["found_correct_email"] / n if n > 0 else 0
    stats["read_rate"] = stats["read_correct_email"] / n if n > 0 else 0
    stats["avg_duration"] /= n if n > 0 else 1
    
    return stats


def print_step_summary(stats: Dict[str, Any]):
    """Print a summary of step statistics."""
    print(f"\n{'='*80}")
    print(f"Step {stats['step']} Summary ({stats['num_rollouts']} rollouts)")
    print(f"{'='*80}")
    
    print(f"\n📊 Performance Metrics:")
    print(f"   Accuracy:           {stats['accuracy']*100:6.2f}% ({stats['correct_answers']}/{stats['num_rollouts']})")
    print(f"   Attempt Rate:       {stats['attempt_rate']*100:6.2f}% ({stats['attempted_answers']}/{stats['num_rollouts']})")
    print(f"   Found Email Rate:   {stats['found_rate']*100:6.2f}% ({stats['found_correct_email']}/{stats['num_rollouts']})")
    print(f"   Read Email Rate:    {stats['read_rate']*100:6.2f}% ({stats['read_correct_email']}/{stats['num_rollouts']})")
    print(f"   Average Reward:     {stats['avg_reward']:6.3f}")
    print(f"   Average Turns:      {stats['avg_turns']:6.2f}")
    
    print(f"\n⚠️  Failure Modes:")
    print(f"   Ran Out of Turns:   {stats['ran_out_of_turns']:4d} ({stats['ran_out_of_turns']/stats['num_rollouts']*100:.1f}%)")
    print(f"   Gave Up Early:      {stats['gave_up_early']:4d} ({stats['gave_up_early']/stats['num_rollouts']*100:.1f}%)")
    print(f"   Repeated Searches:  {stats['repeated_searches']:4d} ({stats['repeated_searches']/stats['num_rollouts']*100:.1f}%)")
    print(f"   Zero-Result Search: {stats['zero_result_searches']:4d} ({stats['zero_result_searches']/stats['num_rollouts']*100:.1f}%)")
    
    if stats["error_types"]:
        print(f"\n❌ Error Types:")
        for error_type, count in stats["error_types"].items():
            print(f"   {error_type:20s} {count:4d} ({count/stats['num_rollouts']*100:.1f}%)")
    
    print(f"\n🔧 Tool Usage:")
    total_tools = sum(stats["tool_usage"].values())
    for tool_name, count in sorted(stats["tool_usage"].items()):
        print(f"   {tool_name:25s} {count:4d} ({count/total_tools*100:.1f}%)")
    
    print(f"\n⏱️  Resource Usage:")
    print(f"   Avg Duration:       {stats['avg_duration']:6.2f}s")
    print(f"   Total Input Tokens: {stats['total_input_tokens']:,}")
    print(f"   Total Output Tokens:{stats['total_output_tokens']:,}")
    print(f"   Avg Tokens/Rollout: {(stats['total_input_tokens'] + stats['total_output_tokens'])/stats['num_rollouts']:,.0f}")


def compare_steps(log_dir: Path, steps: List[int]):
    """Compare statistics across multiple steps."""
    all_stats = []
    
    for step in steps:
        step_dir = log_dir / f"step_{step}"
        if not step_dir.exists():
            print(f"⚠️  Warning: Step {step} not found at {step_dir}")
            continue
        
        stats = analyze_step(step_dir)
        if stats:
            all_stats.append(stats)
    
    if not all_stats:
        print("❌ No valid steps found")
        return
    
    print(f"\n{'='*80}")
    print(f"Comparison Across {len(all_stats)} Steps")
    print(f"{'='*80}")
    
    print(f"\n{'Step':<8} {'Accuracy':<10} {'Avg Reward':<12} {'Avg Turns':<10} {'Found%':<10} {'Read%':<10}")
    print("-" * 80)
    
    for stats in all_stats:
        print(f"{stats['step']:<8} "
              f"{stats['accuracy']*100:>6.2f}%    "
              f"{stats['avg_reward']:>8.3f}    "
              f"{stats['avg_turns']:>6.2f}    "
              f"{stats['found_rate']*100:>6.2f}%   "
              f"{stats['read_rate']*100:>6.2f}%")
    
    # Show improvement
    if len(all_stats) >= 2:
        first = all_stats[0]
        last = all_stats[-1]
        
        print(f"\n📈 Progress from Step {first['step']} to Step {last['step']}:")
        print(f"   Accuracy:    {first['accuracy']*100:.2f}% → {last['accuracy']*100:.2f}% "
              f"({(last['accuracy']-first['accuracy'])*100:+.2f}%)")
        print(f"   Avg Reward:  {first['avg_reward']:.3f} → {last['avg_reward']:.3f} "
              f"({last['avg_reward']-first['avg_reward']:+.3f})")
        print(f"   Avg Turns:   {first['avg_turns']:.2f} → {last['avg_turns']:.2f} "
              f"({last['avg_turns']-first['avg_turns']:+.2f})")


def find_failure_cases(step_dir: Path, limit: int = 5):
    """Find and display common failure cases."""
    logs = load_logs_from_step(step_dir)
    
    # Group failures by type
    failures = {
        "wrong_answer": [],
        "no_answer": [],
        "wrong_email": [],
        "errors": [],
    }
    
    for log in logs:
        rubric = log.get("rubric", {})
        
        if rubric.get("cant_parse_tool_call") or rubric.get("bad_tool_call_name") or rubric.get("bad_tool_call_args"):
            failures["errors"].append(log)
        elif not rubric.get("attempted_answer"):
            failures["no_answer"].append(log)
        elif rubric.get("attempted_answer") and not rubric.get("answer_correct"):
            failures["wrong_answer"].append(log)
        elif not rubric.get("ever_found_right_email"):
            failures["wrong_email"].append(log)
    
    print(f"\n{'='*80}")
    print(f"Failure Cases Analysis (Step {logs[0]['step']})")
    print(f"{'='*80}")
    
    for failure_type, cases in failures.items():
        if not cases:
            continue
        
        print(f"\n{failure_type.replace('_', ' ').title()}: {len(cases)} cases")
        print("-" * 80)
        
        for i, log in enumerate(cases[:limit], 1):
            print(f"\n  {i}. Query: {log['query_question'][:70]}...")
            print(f"     File: {log['_file_path']}")
            print(f"     Reward: {log['reward']:.3f}")
            
            # Show tool calls
            if log.get("tool_calls"):
                print(f"     Tool calls:")
                for tc in log["tool_calls"][:3]:
                    print(f"       - Turn {tc['turn_number']}: {tc['tool_name']}")
                    if tc.get('error'):
                        print(f"         Error: {tc['error']}")
            
            # Show final answer if any
            if log.get("final_answer"):
                print(f"     Agent answer: {log['final_answer'][:60]}...")
                print(f"     Ground truth: {log['query_answer'][:60]}...")
        
        if len(cases) > limit:
            print(f"\n  ... and {len(cases) - limit} more cases")


def main():
    parser = argparse.ArgumentParser(description="Analyze rollout logs")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="outputs/rollout_logs",
        help="Path to rollout logs directory"
    )
    parser.add_argument(
        "--step",
        type=int,
        help="Analyze a specific training step"
    )
    parser.add_argument(
        "--compare",
        type=str,
        help="Compare multiple steps (comma-separated, e.g., '0,5,10')"
    )
    parser.add_argument(
        "--failures",
        action="store_true",
        help="Show common failure cases for specified step"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all available steps"
    )
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    
    if not log_dir.exists():
        print(f"❌ Error: Log directory not found: {log_dir}")
        print(f"   Make sure you've run training with enable_detailed_logging=True")
        sys.exit(1)
    
    # Find available steps
    available_steps = sorted([
        int(d.name.split("_")[1]) 
        for d in log_dir.iterdir() 
        if d.is_dir() and d.name.startswith("step_")
    ])
    
    if not available_steps:
        print(f"❌ Error: No step directories found in {log_dir}")
        sys.exit(1)
    
    print(f"📁 Log directory: {log_dir}")
    print(f"📊 Available steps: {', '.join(map(str, available_steps))}")
    
    # Handle different modes
    if args.compare:
        steps = [int(s.strip()) for s in args.compare.split(",")]
        compare_steps(log_dir, steps)
    
    elif args.all:
        compare_steps(log_dir, available_steps)
    
    elif args.step is not None:
        step_dir = log_dir / f"step_{args.step}"
        if not step_dir.exists():
            print(f"❌ Error: Step {args.step} not found")
            sys.exit(1)
        
        stats = analyze_step(step_dir)
        if stats:
            print_step_summary(stats)
        
        if args.failures:
            find_failure_cases(step_dir)
    
    else:
        # Default: show latest step
        latest_step = available_steps[-1]
        print(f"\nAnalyzing latest step: {latest_step}")
        step_dir = log_dir / f"step_{latest_step}"
        
        stats = analyze_step(step_dir)
        if stats:
            print_step_summary(stats)


if __name__ == "__main__":
    main()

