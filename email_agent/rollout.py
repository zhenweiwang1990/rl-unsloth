"""Rollout logic and evaluation utilities for email agent GRPO training."""

import logging
from typing import Dict
from dataclasses import dataclass, asdict
from openai import AsyncOpenAI

from email_agent.data.types import SyntheticQuery
from email_agent.config import PolicyConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationRubric:
    """Rubric for evaluating agent performance."""
    
    answer_correct: bool = False
    sources_correct: bool = False
    num_turns: int = 0
    attempted_answer: bool = False
    ever_found_right_email: bool = False
    ever_read_right_email: bool = False
    cant_parse_tool_call: bool = False
    bad_tool_call_name: bool = False
    bad_tool_call_args: bool = False
    ran_out_of_turns: bool = False
    returned_i_dont_know: bool = False
    num_sources: int = 0
    ever_tried_to_read_invalid_email: bool = False

    def to_metrics(self) -> Dict[str, float | int]:
        """Convert rubric to metrics dictionary."""
        return {k: int(v) if isinstance(v, bool) else v for k, v in asdict(self).items()}


def calculate_reward(
    policy_config: PolicyConfig, rubric: EvaluationRubric
) -> float:
    """Calculate reward based on rubric.
    
    Args:
        policy_config: Policy configuration
        rubric: Evaluation rubric with performance metrics
        
    Returns:
        Reward value between -2 and 2
    """
    # Simple reward function: 1 for correct, 0 otherwise
    if policy_config.stupid_simple_reward_fn:
        return float(rubric.answer_correct)

    # Complex reward function with partial credit
    partial_rewards = 0.0
    partial_rewards += 0.1 if rubric.ever_found_right_email else 0
    partial_rewards += 0.1 if rubric.ever_read_right_email else 0
    partial_rewards += 0.1 if not rubric.ever_tried_to_read_invalid_email else 0
    partial_rewards += 0.1 if rubric.sources_correct else 0

    # Formatting errors: -2 to -1
    if rubric.cant_parse_tool_call:
        return -2 + partial_rewards

    if rubric.bad_tool_call_name:
        return -1.9 + partial_rewards

    if rubric.bad_tool_call_args:
        return -1.8 + partial_rewards

    # Wrong answer: -1 to 0
    if rubric.attempted_answer and not rubric.answer_correct:
        return -1 + partial_rewards

    # No answer: 0 to 1
    if rubric.returned_i_dont_know or rubric.ran_out_of_turns:
        return 0 + partial_rewards

    # Correct answer: 1 to 2
    if rubric.answer_correct:
        reward = 1.0
        reward += 0.3 if rubric.sources_correct else 0
        reward += 0.1 / rubric.num_sources if rubric.num_sources > 0 else 0
        reward += 0.1 * (1 - rubric.num_turns / max(policy_config.max_turns, 1))
        return reward

    logger.warning(f"Rubric not handled properly: {rubric}")
    return 0.0


async def determine_if_answer_is_correct(
    answer: str, 
    query: SyntheticQuery,
    openai_client: AsyncOpenAI,
    verbose: bool = False
) -> bool:
    """Use Qwen3-32B (via OpenRouter) to determine if the answer is correct.
    
    Args:
        answer: The answer provided by the agent
        query: The synthetic query with ground truth
        openai_client: OpenAI-compatible client (configured for OpenRouter)
        verbose: Whether to print detailed judge logs
        
    Returns:
        True if answer is semantically correct, False otherwise
    """
    system_prompt = (
        "You will be given a question and two different answers to the question: "
        "the correct answer and the answer given by an AI. Your job is to determine "
        "if the answer given by the AI is correct. Return True if the answer is "
        "semantically similar to the correct answer, and False otherwise. "
        "Return only the word True or False, no other text."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Question: {query.question}\n"
                f"Correct answer: {query.answer}\n"
                f"AI answer: {answer}"
            ),
        },
    ]

    if verbose:
        print("\n" + "="*60)
        print("JUDGE EVALUATION (Qwen3-32B via OpenRouter)")
        print("="*60)
        print(f"Question: {query.question}")
        print(f"\nGround Truth: {query.answer}")
        print(f"\nAgent Answer: {answer}")
        print("\nCalling judge model...")

    response = await openai_client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=messages,
        temperature=0,
        max_tokens=10,  # Allow a bit more tokens for True/False response
    )

    judge_result = response.choices[0].message.content.strip().lower().startswith("t")
    
    if verbose:
        print(f"\nJudge Decision: {'✓ CORRECT' if judge_result else '✗ INCORRECT'}")
        print(f"Judge Response: {response.choices[0].message.content.strip()}")
        print("="*60)
    
    logger.info(
        f"Judge evaluation - Question: {query.question[:50]}..., "
        f"Result: {judge_result}"
    )

    return judge_result





