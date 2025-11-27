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
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    # Search repetition tracking
    num_repeated_searches: int = 0  # Number of times the agent repeated the exact same search
    num_zero_result_searches: int = 0  # Number of searches that returned 0 results
    repeated_zero_result_search: bool = False  # Did agent repeat a search that already returned 0 results?
    
    # Search effort tracking
    num_unique_searches: int = 0  # Number of unique searches (different parameters)
    num_total_searches: int = 0  # Total number of search attempts (including repeats)
    num_retry_after_zero: int = 0  # Number of times agent tried different search after getting 0 results (good behavior)
    gave_up_too_early: bool = False  # Did agent give up with "I don't know" after too few unique searches?
    
    # Read tracking (NEW)
    num_total_reads: int = 0  # Total number of read_email calls
    num_unique_reads: int = 0  # Number of unique emails read
    num_repeated_reads: int = 0  # Number of times reading already-read emails
    repeated_correct_email: int = 0  # Number of times re-reading the correct email (bad!)
    
    # Search strategy quality (NEW)
    num_searches_with_zero_results: int = 0  # Searches returning 0 results
    num_searches_with_too_many_results: int = 0  # Searches returning ≥10 results
    num_searches_with_optimal_results: int = 0  # Searches returning 1-9 results (ideal)
    
    broadened_search_after_zero_results: int = 0  # Broadened search after 0 results (good!)
    narrowed_search_after_many_results: int = 0  # Narrowed search after ≥10 results (good!)
    read_after_optimal_search: int = 0  # Read email after 1-9 search results (good!)
    ignored_optimal_results: int = 0  # Continued searching after 1-9 results (bad)
    
    # Sources precision (NEW)
    num_correct_sources: int = 0  # Number of correct sources cited
    num_incorrect_sources: int = 0  # Number of incorrect sources cited
    source_precision: float = 0.0  # Precision of sources (correct / total)
    
    # Timing information (NEW)
    turn_found_right_email: int = -1  # Turn number when correct email was found (-1 if never)
    turn_read_right_email: int = -1  # Turn number when correct email was read (-1 if never)

    def to_metrics(self) -> Dict[str, float | int]:
        """Convert rubric to metrics dictionary."""
        return {k: int(v) if isinstance(v, bool) else v for k, v in asdict(self).items()}


def calculate_reward(
    policy_config: PolicyConfig, rubric: EvaluationRubric
) -> float:
    """Calculate reward based on rubric with comprehensive strategy awareness.
    
    New reward system addresses:
    1. Repeated reading penalty
    2. Search strategy quality (0/10/1-9 results handling)
    3. Source precision (not just inclusion)
    4. Timing bonuses for early discovery
    5. Balanced penalty/reward structure
    
    Args:
        policy_config: Policy configuration
        rubric: Evaluation rubric with performance metrics
        
    Returns:
        Reward value between -3 and +3
    """
    # Simple reward function: 1 for correct, 0 otherwise
    if policy_config.stupid_simple_reward_fn:
        return float(rubric.answer_correct)

    # ========== BASE PARTIAL REWARDS ==========
    partial_rewards = 0.0
    
    # Finding and reading correct email (with timing bonus)
    if rubric.ever_found_right_email:
        base_find_bonus = 0.15
        if rubric.turn_found_right_email > 0:
            # Earlier is better: +0.15 at turn 1, +0.075 at turn 5, +0.03 at turn 9
            timing_bonus = base_find_bonus * (1 - rubric.turn_found_right_email / policy_config.max_turns)
            partial_rewards += max(0.05, timing_bonus)  # At least 0.05
        else:
            partial_rewards += base_find_bonus
    
    if rubric.ever_read_right_email:
        base_read_bonus = 0.15
        if rubric.turn_read_right_email > 0:
            timing_bonus = base_read_bonus * (1 - rubric.turn_read_right_email / policy_config.max_turns)
            partial_rewards += max(0.05, timing_bonus)
        else:
            partial_rewards += base_read_bonus
    
    # Not reading invalid emails
    partial_rewards += 0.1 if not rubric.ever_tried_to_read_invalid_email else 0
    
    # Source precision reward (scaled by precision, not binary)
    if rubric.source_precision > 0:
        source_reward = 0.25 * rubric.source_precision
        partial_rewards += source_reward
    
    # ========== SEARCH STRATEGY REWARDS ==========
    search_strategy_reward = 0.0
    
    # Reward good search strategies
    if rubric.broadened_search_after_zero_results > 0:
        search_strategy_reward += 0.20 * rubric.broadened_search_after_zero_results
    
    if rubric.narrowed_search_after_many_results > 0:
        search_strategy_reward += 0.20 * rubric.narrowed_search_after_many_results
    
    if rubric.read_after_optimal_search > 0:
        search_strategy_reward += 0.25 * rubric.read_after_optimal_search
    
    # Penalties for poor search strategies
    search_strategy_penalty = 0.0
    
    # Penalize not adjusting after problematic results
    if rubric.num_searches_with_zero_results > rubric.broadened_search_after_zero_results:
        missed = rubric.num_searches_with_zero_results - rubric.broadened_search_after_zero_results
        search_strategy_penalty += 0.15 * missed
    
    if rubric.num_searches_with_too_many_results > rubric.narrowed_search_after_many_results:
        missed = rubric.num_searches_with_too_many_results - rubric.narrowed_search_after_many_results
        search_strategy_penalty += 0.15 * missed
    
    # Penalize ignoring optimal search results
    if rubric.ignored_optimal_results > 0:
        search_strategy_penalty += 0.25 * rubric.ignored_optimal_results
    
    search_strategy_reward -= search_strategy_penalty
    
    # ========== REPETITION PENALTIES ==========
    repetition_penalty = 0.0
    
    # Search repetition penalty (reduced from 0.15 to 0.12)
    if rubric.num_repeated_searches > 0:
        repetition_penalty += 0.12 * rubric.num_repeated_searches
    
    if rubric.repeated_zero_result_search:
        repetition_penalty += 0.15
    
    # NEW: Read repetition penalty
    if rubric.num_repeated_reads > 0:
        # Repeated reads are wasteful
        repetition_penalty += 0.20 * rubric.num_repeated_reads
    
    # NEW: Extra penalty for re-reading correct email (very wasteful!)
    if rubric.repeated_correct_email > 0:
        repetition_penalty += 0.25 * rubric.repeated_correct_email
    
    # NEW: Penalty for incorrect sources
    if rubric.num_incorrect_sources > 0:
        # Each wrong source cited: -0.10
        repetition_penalty += 0.10 * rubric.num_incorrect_sources
    
    # ========== FORMATTING ERRORS ==========
    if rubric.cant_parse_tool_call:
        return -2.5 + partial_rewards + search_strategy_reward - repetition_penalty

    if rubric.bad_tool_call_name:
        return -2.3 + partial_rewards + search_strategy_reward - repetition_penalty

    if rubric.bad_tool_call_args:
        return -2.1 + partial_rewards + search_strategy_reward - repetition_penalty

    # ========== WRONG ANSWER ==========
    if rubric.attempted_answer and not rubric.answer_correct:
        return -1.0 + partial_rewards + search_strategy_reward - repetition_penalty

    # ========== NO ANSWER CASES ==========
    if rubric.returned_i_dont_know or rubric.ran_out_of_turns:
        base_reward = 0.0 + partial_rewards + search_strategy_reward - repetition_penalty
        
        # Penalty for early give-up (capped at -1.5)
        if rubric.returned_i_dont_know and not rubric.ran_out_of_turns:
            min_expected_searches = 3
            early_giveup_penalty = 0.0
            
            if rubric.num_unique_searches < min_expected_searches:
                missing_searches = min_expected_searches - rubric.num_unique_searches
                early_giveup_penalty = 0.4 * missing_searches  # Reduced from 0.5
            
            # Base penalty for early give-up (reduced from 1.0 to 0.5)
            early_giveup_penalty += 0.5
            
            # Penalty for unused turns (reduced from 0.05 to 0.03)
            remaining_turns = policy_config.max_turns - rubric.num_turns
            if remaining_turns > 0:
                unused_turn_penalty = 0.03 * remaining_turns
                early_giveup_penalty += unused_turn_penalty
            
            # Cap total penalty at -1.5
            early_giveup_penalty = min(early_giveup_penalty, 1.5)
            
            rubric.gave_up_too_early = True
            logger.warning(
                f"Agent gave up EARLY: {rubric.num_unique_searches} unique searches, "
                f"{rubric.num_turns}/{policy_config.max_turns} turns. "
                f"Penalty: -{early_giveup_penalty:.2f}"
            )
            base_reward -= early_giveup_penalty
        
        # Reward for exhausting turn budget with good effort
        if rubric.ran_out_of_turns:
            effort_bonus = 0.0
            
            if rubric.num_unique_searches >= 3:
                effort_bonus = 0.15
                if rubric.num_unique_searches >= 5:
                    effort_bonus = 0.25
            
            if rubric.num_retry_after_zero > 0:
                effort_bonus += 0.08 * rubric.num_retry_after_zero
            
            # Turn usage bonus (reduced from 0.02 to 0.015)
            turn_usage_bonus = 0.015 * rubric.num_turns
            effort_bonus += turn_usage_bonus
            
            base_reward += effort_bonus
        
        return base_reward

    # ========== CORRECT ANSWER ==========
    if rubric.answer_correct:
        # Perfect execution (relaxed criteria)
        is_perfect = (
            rubric.num_turns <= 4 and  # Allow one exploration/adjustment
            rubric.ever_found_right_email and
            rubric.ever_read_right_email and
            rubric.sources_correct and
            rubric.num_repeated_searches == 0 and
            rubric.num_repeated_reads == 0 and
            rubric.num_searches_with_zero_results <= 1 and  # Allow one exploration
            rubric.read_after_optimal_search > 0  # Must read after optimal search
        )
        
        if is_perfect:
            logger.info(
                f"Perfect execution: {rubric.num_turns} turns, "
                f"optimal strategy, correct answer. Full marks: 3.0"
            )
            return 3.0
        
        # Normal correct answer
        reward = 1.5
        
        # Source precision bonus (scaled, not binary)
        if rubric.source_precision > 0:
            reward += 0.30 * rubric.source_precision
        
        # Strategy bonus for reading after optimal search
        if rubric.read_after_optimal_search > 0:
            reward += 0.20
        
        # Add search strategy rewards
        reward += search_strategy_reward
        
        # Efficiency bonus based on turns used
        efficiency_bonus = 0.25 * (1 - rubric.num_turns / max(policy_config.max_turns, 1))
        reward += efficiency_bonus
        
        # Subtract penalties
        reward -= repetition_penalty
        
        # Cap at 2.8 (perfect is 3.0)
        return min(reward, 2.8)

    logger.warning(f"Rubric not handled properly: {rubric}")
    return 0.0 + search_strategy_reward - repetition_penalty


async def determine_if_answer_is_correct(
    answer: str, 
    query: SyntheticQuery,
    openai_client: AsyncOpenAI,
    verbose: bool = False
) -> bool:
    """Use LLM judge model (via OpenRouter) to determine if the answer is correct.
    
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
        print("JUDGE EVALUATION (DeepSeek V3.2 via OpenRouter)")
        print("="*60)
        print(f"Question: {query.question}")
        print(f"\nGround Truth: {query.answer}")
        print(f"\nAgent Answer: {answer}")
        print("\nCalling judge model...")

    response = await openai_client.chat.completions.create(
        model="deepseek/deepseek-v3.2-exp",
        messages=messages,
        max_tokens=10,  # Allow a bit more tokens for True/False response
        extra_body={
            "thinking": False,  # Explicitly disable thinking/reasoning
        }
    )

    # Extract the final message content (not thinking/reasoning)
    message = response.choices[0].message
    content = message.content
    
    # For models with thinking, content should only contain the final response
    # Extract True/False from the actual response content
    if content:
        content_clean = content.strip().lower()
        # Look for "true" or "false" in the response, case-insensitive
        if "true" in content_clean:
            judge_result = True
        elif "false" in content_clean:
            judge_result = False
        else:
            # Fallback: check if starts with 't'
            judge_result = content_clean.startswith("t")
            logger.warning(f"Judge returned unexpected response: {content}")
    else:
        judge_result = False
        logger.error("Judge returned empty content")
    
    if verbose:
        print(f"\nJudge Decision: {'✓ CORRECT' if judge_result else '✗ INCORRECT'}")
        print(f"Judge Response: {content.strip() if content else 'Empty'}")
        print("="*60)
    
    logger.info(
        f"Judge evaluation - Question: {query.question[:50]}..., "
        f"Result: {judge_result}"
    )

    return judge_result





