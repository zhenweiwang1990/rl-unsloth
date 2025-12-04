"""Evaluation script for Link Search Agent.

Usage:
    python eval_linksearch.py --model-path outputs/grpo_linksearch_masked/final --num-queries 100
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Link Search Agent")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to fine-tuned model (default: base model)"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of queries to evaluate"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Link Search Agent Evaluation")
    print("="*60)
    
    # Load model
    from unsloth import FastLanguageModel
    from link_search_agent.config import GRPOConfig, PolicyConfig
    from link_search_agent.data import load_link_search_queries
    from link_search_agent.grpo_utils import execute_rollout
    
    config = GRPOConfig()
    
    if args.model_path:
        print(f"\n📦 Loading fine-tuned model from: {args.model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            dtype=None,
        )
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
    
    # Load evaluation queries
    print(f"\n📚 Loading evaluation queries...")
    queries = load_link_search_queries(
        split="test",
        limit=args.num_queries,
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
    
    # Run evaluation
    print(f"\n🔄 Running evaluation...")
    
    results = []
    scores = []
    hits = []
    total_time = 0
    
    loop = asyncio.get_event_loop()
    
    for i, query in enumerate(tqdm(queries, desc="Evaluating")):
        start_time = time.time()
        
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
            
            duration = time.time() - start_time
            total_time += duration
            
            scores.append(rubric.score)
            hits.append(rubric.num_correct_handles)
            
            result = {
                "query_id": query.id,
                "query": query.query,
                "gold_handles": query.gold_handles,
                "predicted_handles": rubric.predicted_handles if hasattr(rubric, 'predicted_handles') else [],
                "score": rubric.score,
                "num_hits": rubric.num_correct_handles,
                "num_turns": rubric.num_turns,
                "reward": reward,
                "duration": duration,
            }
            results.append(result)
            
            if args.verbose and (i + 1) % 10 == 0:
                print(f"\nProgress: {i+1}/{len(queries)}")
                print(f"  Avg Score: {sum(scores)/len(scores):.3f}")
                print(f"  Avg Hits: {sum(hits)/len(hits):.2f}")
        
        except Exception as e:
            logger.error(f"Error evaluating query {query.id}: {e}")
            results.append({
                "query_id": query.id,
                "query": query.query,
                "error": str(e),
                "score": 0.0,
            })
            scores.append(0.0)
            hits.append(0)
    
    # Calculate metrics
    avg_score = sum(scores) / len(scores) if scores else 0
    avg_hits = sum(hits) / len(hits) if hits else 0
    accuracy = sum(1 for s in scores if s > 0) / len(scores) if scores else 0
    perfect_rate = sum(1 for s in scores if s >= 1.0) / len(scores) if scores else 0
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"\nQueries evaluated: {len(queries)}")
    print(f"Total time: {total_time:.1f}s ({total_time/len(queries):.2f}s/query)")
    print(f"\n📊 Performance Metrics:")
    print(f"   Average Score:      {avg_score:.3f}")
    print(f"   Average Hits:       {avg_hits:.2f}")
    print(f"   Accuracy (>0 hits): {accuracy*100:.2f}%")
    print(f"   Perfect Score Rate: {perfect_rate*100:.2f}%")
    
    # Score distribution
    print(f"\n📈 Score Distribution:")
    bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0, float('inf')]
    for i in range(len(bins)-1):
        count = sum(1 for s in scores if bins[i] <= s < bins[i+1])
        pct = count / len(scores) * 100 if scores else 0
        label = f"{bins[i]:.1f}-{bins[i+1]:.1f}" if bins[i+1] != float('inf') else f"{bins[i]:.1f}+"
        print(f"   {label}: {count:4d} ({pct:5.1f}%)")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            "model_path": args.model_path,
            "num_queries": len(queries),
            "avg_score": avg_score,
            "avg_hits": avg_hits,
            "accuracy": accuracy,
            "perfect_rate": perfect_rate,
            "total_time": total_time,
            "results": results,
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📁 Results saved to: {output_path}")
    
    print("\n✅ Evaluation complete!")
    
    return avg_score


if __name__ == "__main__":
    main()

