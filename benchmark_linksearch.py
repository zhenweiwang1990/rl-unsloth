"""Benchmark script for Link Search Agent.

Runs evaluation and saves detailed results to CSV.

Usage:
    python benchmark_linksearch.py --limit 100 --model-path outputs/grpo_linksearch_masked/final
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Link Search Agent")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to fine-tuned model (default: base model)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of queries to benchmark"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    run_id = os.environ.get("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    print("="*60)
    print("Link Search Agent Benchmark")
    print("="*60)
    print(f"Run ID: {run_id}")
    
    # Load model
    from unsloth import FastLanguageModel
    from link_search_agent.config import GRPOConfig, PolicyConfig
    from link_search_agent.data import load_link_search_queries
    from link_search_agent.grpo_utils import execute_rollout
    
    config = GRPOConfig()
    
    model_name = "base"
    if args.model_path:
        print(f"\n📦 Loading fine-tuned model from: {args.model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            dtype=None,
        )
        model_name = Path(args.model_path).name
    else:
        print(f"\n📦 Loading base model: {config.model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            dtype=None,
        )
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=config.seed,
            max_seq_length=config.max_seq_length,
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    print("✓ Model loaded")
    
    # Load benchmark queries
    print(f"\n📚 Loading benchmark queries...")
    queries = load_link_search_queries(
        split="test",
        limit=args.limit,
        shuffle=True,
        seed=42,
    )
    print(f"✓ Loaded {len(queries)} queries")
    
    # Setup policy config
    policy_config = PolicyConfig(
        max_turns=config.max_turns,
        max_tokens=config.max_tokens,
        max_profiles=config.max_profiles,
        verbose=args.verbose,
    )
    
    # Prepare output files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"benchmark_results_linksearch_{run_id}.csv"
    json_path = output_dir / f"benchmark_results_linksearch_{run_id}.json"
    
    # Run benchmark
    print(f"\n🔄 Running benchmark...")
    
    results = []
    loop = asyncio.get_event_loop()
    
    start_time = time.time()
    
    # Open CSV file for streaming results
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'query_id', 'query', 'gold_handles', 'predicted_handles',
            'num_gold', 'num_predicted', 'num_hits', 'score',
            'num_turns', 'duration_seconds', 'input_tokens', 'output_tokens',
            'num_searches', 'num_reads', 'error'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, query in enumerate(tqdm(queries, desc="Benchmarking")):
            query_start = time.time()
            
            try:
                conversation, reward, rubric, rollout_log = loop.run_until_complete(
                    execute_rollout(
                        query=query,
                        model=model,
                        tokenizer=tokenizer,
                        policy_config=policy_config,
                        verbose=args.verbose,
                        log_turns=args.verbose,
                        enable_detailed_logging=True,
                        training_step=0,
                    )
                )
                
                duration = time.time() - query_start
                
                predicted = rubric.predicted_handles if hasattr(rubric, 'predicted_handles') else []
                
                row = {
                    'query_id': query.id,
                    'query': query.query[:200],
                    'gold_handles': json.dumps(query.gold_handles),
                    'predicted_handles': json.dumps(predicted),
                    'num_gold': len(query.gold_handles),
                    'num_predicted': rubric.num_predicted_handles,
                    'num_hits': rubric.num_correct_handles,
                    'score': rubric.score,
                    'num_turns': rubric.num_turns,
                    'duration_seconds': round(duration, 2),
                    'input_tokens': rubric.total_input_tokens,
                    'output_tokens': rubric.total_output_tokens,
                    'num_searches': rubric.num_unique_searches,
                    'num_reads': rubric.num_profiles_read,
                    'error': '',
                }
                
                results.append(row)
                writer.writerow(row)
                csvfile.flush()
                
            except Exception as e:
                logger.error(f"Error benchmarking query {query.id}: {e}")
                row = {
                    'query_id': query.id,
                    'query': query.query[:200],
                    'gold_handles': json.dumps(query.gold_handles),
                    'predicted_handles': '[]',
                    'num_gold': len(query.gold_handles),
                    'num_predicted': 0,
                    'num_hits': 0,
                    'score': 0.0,
                    'num_turns': 0,
                    'duration_seconds': time.time() - query_start,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'num_searches': 0,
                    'num_reads': 0,
                    'error': str(e),
                }
                results.append(row)
                writer.writerow(row)
                csvfile.flush()
    
    total_time = time.time() - start_time
    
    # Calculate summary metrics
    scores = [r['score'] for r in results]
    hits = [r['num_hits'] for r in results]
    turns = [r['num_turns'] for r in results]
    durations = [r['duration_seconds'] for r in results]
    errors = sum(1 for r in results if r['error'])
    
    avg_score = sum(scores) / len(scores) if scores else 0
    avg_hits = sum(hits) / len(hits) if hits else 0
    avg_turns = sum(turns) / len(turns) if turns else 0
    accuracy = sum(1 for s in scores if s > 0) / len(scores) if scores else 0
    perfect_rate = sum(1 for s in scores if s >= 1.0) / len(scores) if scores else 0
    
    summary = {
        "run_id": run_id,
        "model": model_name,
        "model_path": args.model_path,
        "num_queries": len(queries),
        "num_errors": errors,
        "total_time_seconds": round(total_time, 2),
        "avg_time_per_query": round(total_time / len(queries), 2),
        "metrics": {
            "avg_score": round(avg_score, 4),
            "avg_hits": round(avg_hits, 2),
            "avg_turns": round(avg_turns, 2),
            "accuracy": round(accuracy, 4),
            "perfect_rate": round(perfect_rate, 4),
        },
        "score_distribution": {
            "0": sum(1 for s in scores if s == 0),
            "0.1-0.3": sum(1 for s in scores if 0 < s <= 0.3),
            "0.3-0.5": sum(1 for s in scores if 0.3 < s <= 0.5),
            "0.5-0.7": sum(1 for s in scores if 0.5 < s <= 0.7),
            "0.7-1.0": sum(1 for s in scores if 0.7 < s < 1.0),
            "1.0": sum(1 for s in scores if s >= 1.0),
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save JSON summary
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print results
    print("\n" + "="*60)
    print("Benchmark Results")
    print("="*60)
    print(f"\nRun ID: {run_id}")
    print(f"Model: {model_name}")
    print(f"Queries: {len(queries)}")
    print(f"Errors: {errors}")
    print(f"Total time: {total_time:.1f}s ({total_time/len(queries):.2f}s/query)")
    
    print(f"\n📊 Performance Metrics:")
    print(f"   Average Score:      {avg_score:.4f}")
    print(f"   Average Hits:       {avg_hits:.2f}")
    print(f"   Average Turns:      {avg_turns:.2f}")
    print(f"   Accuracy (>0 hits): {accuracy*100:.2f}%")
    print(f"   Perfect Score Rate: {perfect_rate*100:.2f}%")
    
    print(f"\n📈 Score Distribution:")
    for label, count in summary["score_distribution"].items():
        pct = count / len(scores) * 100 if scores else 0
        print(f"   {label:10s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\n📁 Results saved to:")
    print(f"   CSV: {csv_path}")
    print(f"   JSON: {json_path}")
    
    print("\n✅ Benchmark complete!")
    
    return avg_score


if __name__ == "__main__":
    main()

