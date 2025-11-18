"""Evaluation script for trained Email Agent model."""

import os
import asyncio
import torch
from unsloth import FastLanguageModel
from openai import AsyncOpenAI
import logging
from tqdm import tqdm

from email_agent.config import PolicyConfig
from email_agent.data import load_synthetic_queries
from email_agent.agent import EmailAgent
from email_agent.rollout import calculate_reward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def evaluate_model(
    model_path: str,
    num_queries: int = 100,
    verbose: bool = False,
):
    """Evaluate a trained model on the email agent task.
    
    Args:
        model_path: Path to the trained model (LoRA adapter)
        num_queries: Number of queries to evaluate on
        verbose: Whether to print detailed logs
    """
    logger.info(f"Loading model from {model_path}")
    
    # Get base model name from environment
    base_model_name = os.environ.get("MODEL_NAME", "unsloth/Qwen3-14B-Base")
    max_seq_length = int(os.environ.get("MAX_SEQ_LENGTH", "8192"))
    
    # Load model with unsloth optimization
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
    
    # Enable inference mode (important for speed!)
    FastLanguageModel.for_inference(model)
    logger.info("✓ Model loaded successfully with unsloth optimization (2-5x faster)")
    
    # Set padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load queries from test split to avoid overfitting
    logger.info(f"Loading {num_queries} evaluation queries from test split...")
    queries = load_synthetic_queries(
        split="test",
        limit=num_queries,
        shuffle=True,
        max_messages=1,
    )
    logger.info(f"✓ Loaded {len(queries)} queries from test split")
    
    # Initialize policy config
    policy_config = PolicyConfig(
        max_turns=10,
        max_tokens=4096,
        verbose=verbose,
    )
    
    # Initialize OpenRouter client for judge
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
    
    # Evaluation loop
    results = []
    
    for query in tqdm(queries, desc="Evaluating"):
        # Run agent
        rubric, conversation = await agent.run_query(
            query=query,
            verbose=verbose,
        )
        
        # Calculate reward
        reward = calculate_reward(policy_config, rubric)
        
        results.append({
            "query_id": query.id,
            "reward": reward,
            "answer_correct": int(rubric.answer_correct),
            "sources_correct": int(rubric.sources_correct),
            "num_turns": rubric.num_turns,
            "attempted_answer": int(rubric.attempted_answer),
        })
        
        if verbose:
            logger.info(f"Query {query.id}: reward={reward:.2f}, correct={rubric.answer_correct}")
    
    # Compute statistics
    avg_reward = sum(r["reward"] for r in results) / len(results)
    accuracy = sum(r["answer_correct"] for r in results) / len(results)
    source_accuracy = sum(r["sources_correct"] for r in results) / len(results)
    avg_turns = sum(r["num_turns"] for r in results) / len(results)
    attempted = sum(r["attempted_answer"] for r in results)
    
    logger.info("="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"Number of queries: {len(results)}")
    logger.info(f"Average reward: {avg_reward:.3f}")
    logger.info(f"Answer accuracy: {accuracy:.1%}")
    logger.info(f"Source accuracy: {source_accuracy:.1%}")
    logger.info(f"Average turns: {avg_turns:.2f}")
    logger.info(f"Attempted answers: {attempted}/{len(results)}")
    logger.info("="*60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Email Agent model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/grpo/final",
        help="Path to trained model",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of queries to evaluate",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed logs",
    )
    
    args = parser.parse_args()
    
    asyncio.run(evaluate_model(
        model_path=args.model_path,
        num_queries=args.num_queries,
        verbose=args.verbose,
    ))

