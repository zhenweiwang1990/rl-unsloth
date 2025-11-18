"""Benchmark script for Email Agent model."""

import os
import asyncio
import torch
from unsloth import FastLanguageModel
from openai import AsyncOpenAI
import logging
from tqdm import tqdm
import polars as pl
from datetime import datetime
from typing import Optional

from email_agent.config import PolicyConfig
from email_agent.data import load_synthetic_queries
from email_agent.agent import EmailAgent
from email_agent.rollout import calculate_reward

# Setup logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_env_int(key: str, default: str) -> int:
    """Get integer from environment variable, stripping comments."""
    value = os.environ.get(key, default)
    # Remove inline comments (everything after #)
    value = value.split('#')[0].strip()
    return int(value)




async def benchmark_model(
    model_path: Optional[str] = None,
    limit: int = 100,
    verbose: bool = False,
):
    """Benchmark a model on the email agent task.
    
    Args:
        model_path: Path to trained model (LoRA adapter). If None, uses base model from env.
        limit: Number of queries to evaluate
        verbose: Whether to print detailed logs
        
    Returns:
        Polars DataFrame with benchmark results
    """
    logger.info("="*60)
    logger.info("Email Agent Benchmark")
    logger.info("="*60)
    
    # Initialize policy config
    policy_config = PolicyConfig(
        max_turns=get_env_int("MAX_TURNS", "10"),
        max_tokens=get_env_int("MAX_TOKENS", "4096"),
        verbose=verbose,
    )
    
    # Get base model name from environment
    base_model_name = os.environ.get("MODEL_NAME", "unsloth/Qwen3-14B-Base")
    max_seq_length = get_env_int("MAX_SEQ_LENGTH", "8192")
            
    # Load model with unsloth optimization
    if model_path:
        logger.info(f"Loading fine-tuned model from {model_path}")
        logger.info(f"Base model: {base_model_name}")
        logger.info("Using unsloth FastLanguageModel for optimized inference")
        
        # Load base model with unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,  # Use 4-bit quantization for faster inference
            dtype=None,  # Auto-detect
        )
        
        # Load LoRA adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        logger.info("✓ Fine-tuned model loaded successfully")
    else:
        logger.info(f"No model path provided - using base model: {base_model_name}")
        logger.info("Using unsloth FastLanguageModel for optimized inference")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,  # Use 4-bit quantization for faster inference
            dtype=None,  # Auto-detect
        )
        logger.info("✓ Base model loaded successfully")
    
    # Enable inference mode (important for speed!)
    FastLanguageModel.for_inference(model)
    logger.info("✓ Unsloth inference mode enabled (2-5x faster)")
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load queries from test split to avoid overfitting
    logger.info(f"Loading {limit} benchmark queries from test split...")
    queries = load_synthetic_queries(
        split="test",
        limit=limit,
        shuffle=True,
        max_messages=1,
    )
    logger.info(f"✓ Loaded {len(queries)} queries from test split")
    
    # Initialize OpenRouter client (for judge only)
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set (needed for Qwen3-32B judge)")
    
    openai_client = AsyncOpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    logger.info("✓ OpenRouter client initialized (Qwen3-32B judge)")
    
    # Create agent
    agent = EmailAgent(
        model=model,
        tokenizer=tokenizer,
        policy_config=policy_config,
        openai_client=openai_client,
    )
    logger.info("✓ EmailAgent initialized")
    
    # Results storage
    results = []
    
    # Benchmark loop
    logger.info("")
    logger.info("Starting benchmark...")
    logger.info("")
    if verbose:
        logger.info("Verbose mode enabled - detailed logs will be printed for each query")
        logger.info("")
    
    start_time = datetime.now()
    
    # Don't use tqdm since we'll print progress for each query
    for idx, query in enumerate(queries):
        query_start = datetime.now()
        
        # Run agent
        rubric, conversation = await agent.run_query(
            query=query,
            verbose=verbose,
        )
        
        # Calculate reward
        reward = calculate_reward(policy_config, rubric)
        
        # Calculate duration
        query_duration = (datetime.now() - query_start).total_seconds()
        
        # Store result
        result = {
            "query_id": query.id,
            "question": query.question[:100],  # Truncate for display
            "answer": query.answer[:100],
            "reward": reward,
            "answer_correct": int(rubric.answer_correct),
            "sources_correct": int(rubric.sources_correct),
            "num_turns": rubric.num_turns,
            "attempted_answer": int(rubric.attempted_answer),
            "ever_found_right_email": int(rubric.ever_found_right_email),
            "ever_read_right_email": int(rubric.ever_read_right_email),
            "ran_out_of_turns": int(rubric.ran_out_of_turns),
            "returned_i_dont_know": int(rubric.returned_i_dont_know),
            "cant_parse_tool_call": int(rubric.cant_parse_tool_call),
            "bad_tool_call_name": int(rubric.bad_tool_call_name),
            "bad_tool_call_args": int(rubric.bad_tool_call_args),
            "duration_seconds": query_duration,
        }
        results.append(result)
        
        # Print progress after each query
        logger.info(
            f"[{idx+1}/{len(queries)}] "
            f"Reward: {reward:.3f} | "
            f"Correct: {rubric.answer_correct} | "
            f"Turns: {rubric.num_turns} | "
            f"Time: {query_duration:.1f}s | "
            f"Question: {query.question[:60]}..."
        )
    
    # Convert to DataFrame
    df = pl.DataFrame(results)
    
    # Calculate statistics
    total_duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("")
    logger.info("="*60)
    logger.info("BENCHMARK RESULTS")
    logger.info("="*60)
    logger.info(f"Total queries: {len(results)}")
    logger.info(f"Total duration: {total_duration:.2f}s")
    logger.info(f"Avg duration per query: {total_duration/len(results):.2f}s")
    logger.info("")
    logger.info(f"Average reward: {df['reward'].mean():.3f}")
    logger.info(f"Answer accuracy: {df['answer_correct'].mean():.1%}")
    logger.info(f"Source accuracy: {df['sources_correct'].mean():.1%}")
    logger.info(f"Average turns: {df['num_turns'].mean():.2f}")
    logger.info("")
    logger.info(f"Attempted answers: {df['attempted_answer'].sum()}/{len(results)}")
    logger.info(f"Found right email: {df['ever_found_right_email'].sum()}/{len(results)}")
    logger.info(f"Read right email: {df['ever_read_right_email'].sum()}/{len(results)}")
    logger.info("")
    logger.info(f"Ran out of turns: {df['ran_out_of_turns'].sum()}")
    logger.info(f"Returned 'I don't know': {df['returned_i_dont_know'].sum()}")
    logger.info(f"Parse errors: {df['cant_parse_tool_call'].sum()}")
    logger.info(f"Bad tool name: {df['bad_tool_call_name'].sum()}")
    logger.info(f"Bad tool args: {df['bad_tool_call_args'].sum()}")
    logger.info("="*60)
    
    return df


async def main():
    """Main benchmark function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Email Agent model")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model (optional, uses base model from env if not provided)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of queries to benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed logs (shows agent reasoning, tool calls, and judge decisions)",
    )
    
    args = parser.parse_args()
    
    # Override from environment variables
    limit = int(os.environ.get("TEST_SET_SIZE", args.limit))
    verbose = os.environ.get("VERBOSE", "false").lower() == "true" or args.verbose
    
    # Run benchmark
    results = await benchmark_model(
        model_path=args.model_path,
        limit=limit,
        verbose=verbose,
    )
    
    # Save results to outputs directory (mounted from host)
    if args.output:
        output_path = args.output
    else:
        run_id = os.environ.get("RUN_ID", "001")
        # Create outputs directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        output_path = f"outputs/benchmark_results_{run_id}.csv"
    
    results.write_csv(output_path)
    logger.info(f"\n✓ Results saved to {output_path}")
    
    # Print final summary
    logger.info("")
    logger.info("="*60)
    logger.info("FINAL BENCHMARK SUMMARY")
    logger.info("="*60)
    logger.info(f"Model: {args.model_path if args.model_path else 'Base model (no fine-tuning)'}")
    logger.info(f"Total queries evaluated: {len(results)}")
    logger.info(f"Results file: {output_path}")
    logger.info("")
    logger.info("Key Metrics:")
    logger.info(f"  • Average Reward: {results['reward'].mean():.3f}")
    logger.info(f"  • Answer Accuracy: {results['answer_correct'].mean():.1%}")
    logger.info(f"  • Source Accuracy: {results['sources_correct'].mean():.1%}")
    logger.info(f"  • Average Turns: {results['num_turns'].mean():.2f}")
    logger.info("")
    logger.info("Success Indicators:")
    logger.info(f"  • Attempted Answer: {results['attempted_answer'].sum()}/{len(results)} ({results['attempted_answer'].mean():.1%})")
    logger.info(f"  • Found Right Email: {results['ever_found_right_email'].sum()}/{len(results)} ({results['ever_found_right_email'].mean():.1%})")
    logger.info(f"  • Read Right Email: {results['ever_read_right_email'].sum()}/{len(results)} ({results['ever_read_right_email'].mean():.1%})")
    logger.info("")
    logger.info("Error Analysis:")
    logger.info(f"  • Ran Out of Turns: {results['ran_out_of_turns'].sum()} ({results['ran_out_of_turns'].mean():.1%})")
    logger.info(f"  • Returned 'I Don't Know': {results['returned_i_dont_know'].sum()} ({results['returned_i_dont_know'].mean():.1%})")
    logger.info(f"  • Parse Errors: {results['cant_parse_tool_call'].sum()} ({results['cant_parse_tool_call'].mean():.1%})")
    logger.info(f"  • Bad Tool Name: {results['bad_tool_call_name'].sum()} ({results['bad_tool_call_name'].mean():.1%})")
    logger.info(f"  • Bad Tool Args: {results['bad_tool_call_args'].sum()} ({results['bad_tool_call_args'].mean():.1%})")
    logger.info("="*60)


if __name__ == "__main__":
    asyncio.run(main())

