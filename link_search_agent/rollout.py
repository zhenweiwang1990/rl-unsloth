"""Evaluation rubric and reward calculation for Link Search Agent."""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Set

from link_search_agent.config import PolicyConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def normalize_handle(handle: str) -> str:
    """Normalize a LinkedIn handle for comparison."""
    if not handle:
        return ""
    return handle.lower().strip()


def compute_score(predicted: List[str], gold: List[str]) -> Dict:
    """Compute matching score between predicted and gold handles.
    
    Scoring:
    - Each predicted handle that matches a gold adds 0.1 point
    - Score is capped at 1.0
    - If all gold handles are hit and gold count < 10, score is forced to 1.0
    
    Args:
        predicted: List of predicted handles
        gold: List of gold (correct) handles
        
    Returns:
        Dict with hits, score, precision, recall
    """
    gold_set = set(normalize_handle(h) for h in gold if h)
    pred_set = set(normalize_handle(h) for h in predicted if h)
    
    hits = list(gold_set & pred_set)
    
    if len(gold_set) == 0:
        return {
            "hits": hits,
            "score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "num_gold": 0,
            "num_predicted": len(pred_set),
            "num_hits": 0,
        }
    
    all_gold_hit = len(hits) >= len(gold_set)
    
    # Score calculation
    if all_gold_hit and len(gold_set) < 10:
        score = 1.0
    else:
        score = min(len(hits) * 0.1, 1.0)
    
    precision = len(hits) / len(pred_set) if pred_set else 0.0
    recall = len(hits) / len(gold_set) if gold_set else 0.0
    
    return {
        "hits": hits,
        "score": score,
        "precision": precision,
        "recall": recall,
        "num_gold": len(gold_set),
        "num_predicted": len(pred_set),
        "num_hits": len(hits),
    }


@dataclass
class LinkSearchRubric:
    """Rubric for evaluating link search agent performance."""
    
    # Core results
    num_correct_handles: int = 0  # Number of correct handles found
    num_predicted_handles: int = 0  # Total handles predicted
    num_gold_handles: int = 0  # Number of gold handles
    score: float = 0.0  # Computed score (0-1)
    precision: float = 0.0
    recall: float = 0.0
    
    # Turn tracking
    num_turns: int = 0
    attempted_answer: bool = False  # Did agent call return_results
    ran_out_of_turns: bool = False
    
    # Search quality
    num_total_searches: int = 0
    num_unique_searches: int = 0  # Different SQL queries
    num_repeated_searches: int = 0
    num_zero_result_searches: int = 0
    num_searches_with_results: int = 0
    
    # Read quality
    num_total_reads: int = 0
    num_unique_reads: int = 0
    num_repeated_reads: int = 0
    num_correct_profiles_read: int = 0  # Reads of correct handles
    num_invalid_reads: int = 0  # Reads that returned error
    
    # Strategy quality
    broadened_search_after_zero: int = 0  # Good: widened search after 0 results
    narrowed_search_after_many: int = 0   # Good: narrowed after many results
    read_after_good_search: int = 0       # Good: read after finding candidates
    
    # Errors
    cant_parse_tool_call: bool = False
    bad_tool_call_name: bool = False
    bad_tool_call_args: bool = False
    sql_error_count: int = 0
    
    # Token usage
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    # Timing
    turn_first_correct_found: int = -1  # Turn when first correct handle found
    turn_first_correct_read: int = -1   # Turn when first correct handle read
    
    def to_metrics(self) -> Dict[str, float | int]:
        """Convert rubric to metrics dictionary."""
        return {k: int(v) if isinstance(v, bool) else v for k, v in asdict(self).items()}


def calculate_reward(
    policy_config: PolicyConfig,
    rubric: LinkSearchRubric,
) -> float:
    """Calculate reward based on rubric.
    
    Reward structure:
    - Base score from handle matching (0-1, scaled to 0-1.5)
    - Bonus for good search strategy
    - Penalties for inefficiency and errors
    
    Args:
        policy_config: Policy configuration
        rubric: Evaluation rubric with performance metrics
        
    Returns:
        Reward value between -2 and +3
    """
    # Simple reward: just use the score
    if policy_config.stupid_simple_reward_fn:
        return rubric.score
    
    # ========== BASE SCORE ==========
    # Map 0-1 score to 0-1.5 base reward
    base_reward = rubric.score * 1.5
    
    # ========== PARTIAL CREDIT FOR FINDING CORRECT HANDLES ==========
    partial_rewards = 0.0
    
    # Early discovery bonus
    if rubric.turn_first_correct_found > 0:
        # Earlier is better
        timing_bonus = 0.15 * (1 - rubric.turn_first_correct_found / policy_config.max_turns)
        partial_rewards += max(0.05, timing_bonus)
    
    if rubric.turn_first_correct_read > 0:
        timing_bonus = 0.10 * (1 - rubric.turn_first_correct_read / policy_config.max_turns)
        partial_rewards += max(0.03, timing_bonus)
    
    # ========== SEARCH STRATEGY REWARDS ==========
    strategy_reward = 0.0
    
    # Good: broadened search after zero results
    if rubric.broadened_search_after_zero > 0:
        strategy_reward += 0.15 * rubric.broadened_search_after_zero
    
    # Good: narrowed search after many results
    if rubric.narrowed_search_after_many > 0:
        strategy_reward += 0.15 * rubric.narrowed_search_after_many
    
    # Good: read after finding candidates
    if rubric.read_after_good_search > 0:
        strategy_reward += 0.20 * rubric.read_after_good_search
    
    # ========== PENALTIES ==========
    penalties = 0.0
    
    # Repeated searches are wasteful
    if rubric.num_repeated_searches > 0:
        penalties += 0.10 * rubric.num_repeated_searches
    
    # Repeated reads are wasteful
    if rubric.num_repeated_reads > 0:
        penalties += 0.15 * rubric.num_repeated_reads
    
    # SQL errors
    if rubric.sql_error_count > 0:
        penalties += 0.08 * rubric.sql_error_count
    
    # Invalid reads
    if rubric.num_invalid_reads > 0:
        penalties += 0.10 * rubric.num_invalid_reads
    
    # ========== FORMATTING ERRORS (severe penalties) ==========
    if rubric.cant_parse_tool_call:
        return -2.0 + partial_rewards + strategy_reward - penalties
    
    if rubric.bad_tool_call_name:
        return -1.8 + partial_rewards + strategy_reward - penalties
    
    if rubric.bad_tool_call_args:
        return -1.5 + partial_rewards + strategy_reward - penalties
    
    # ========== NO ANSWER CASE ==========
    if not rubric.attempted_answer:
        if rubric.ran_out_of_turns:
            # Ran out of turns without answering
            # Give partial credit for effort
            effort_bonus = 0.0
            if rubric.num_unique_searches >= 3:
                effort_bonus += 0.10
            if rubric.num_correct_profiles_read > 0:
                effort_bonus += 0.15 * min(rubric.num_correct_profiles_read, 3)
            return -0.5 + partial_rewards + strategy_reward + effort_bonus - penalties
        else:
            # Gave up early without answering
            return -1.0 + partial_rewards + strategy_reward - penalties
    
    # ========== ANSWERED CASE ==========
    # Perfect execution bonus
    is_perfect = (
        rubric.score >= 1.0 and
        rubric.num_repeated_searches == 0 and
        rubric.num_repeated_reads == 0 and
        rubric.sql_error_count == 0 and
        rubric.num_turns <= 8  # Efficient
    )
    
    if is_perfect:
        logger.info(f"Perfect execution: score={rubric.score}, turns={rubric.num_turns}")
        return 3.0
    
    # Normal case: combine everything
    reward = base_reward + partial_rewards + strategy_reward - penalties
    
    # Efficiency bonus for fewer turns
    if rubric.num_turns < policy_config.max_turns:
        efficiency = 0.20 * (1 - rubric.num_turns / policy_config.max_turns)
        reward += efficiency
    
    # Cap at 2.8 (perfect is 3.0)
    return min(max(reward, -2.0), 2.8)

