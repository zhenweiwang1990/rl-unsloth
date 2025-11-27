"""Simple standalone test for reward function logic.

This bypasses imports and tests the core reward calculation logic.
"""

import sys
sys.path.insert(0, '/home/zhlmmc/rl-unsloth')

from dataclasses import dataclass


@dataclass
class PolicyConfig:
    max_turns: int = 10
    stupid_simple_reward_fn: bool = False


@dataclass  
class EvaluationRubric:
    """Simplified rubric for testing."""
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
    num_repeated_searches: int = 0
    num_zero_result_searches: int = 0
    repeated_zero_result_search: bool = False
    num_unique_searches: int = 0
    num_total_searches: int = 0
    num_retry_after_zero: int = 0
    gave_up_too_early: bool = False
    num_total_reads: int = 0
    num_unique_reads: int = 0
    num_repeated_reads: int = 0
    repeated_correct_email: int = 0
    num_searches_with_zero_results: int = 0
    num_searches_with_too_many_results: int = 0
    num_searches_with_optimal_results: int = 0
    broadened_search_after_zero_results: int = 0
    narrowed_search_after_many_results: int = 0
    read_after_optimal_search: int = 0
    ignored_optimal_results: int = 0
    num_correct_sources: int = 0
    num_incorrect_sources: int = 0
    source_precision: float = 0.0
    turn_found_right_email: int = -1
    turn_read_right_email: int = -1
    total_input_tokens: int = 0
    total_output_tokens: int = 0


def test_scenario(name, rubric, config):
    """Test a scenario."""
    # Simplified reward calculation for testing
    partial = 0.0
    
    # Base rewards
    if rubric.ever_found_right_email:
        if rubric.turn_found_right_email > 0:
            timing = 0.15 * (1 - rubric.turn_found_right_email / config.max_turns)
            partial += max(0.05, timing)
        else:
            partial += 0.15
    
    if rubric.ever_read_right_email:
        if rubric.turn_read_right_email > 0:
            timing = 0.15 * (1 - rubric.turn_read_right_email / config.max_turns)
            partial += max(0.05, timing)
        else:
            partial += 0.15
    
    partial += 0.1 if not rubric.ever_tried_to_read_invalid_email else 0
    
    if rubric.source_precision > 0:
        partial += 0.25 * rubric.source_precision
    
    # Search strategy
    strategy = 0.0
    strategy += 0.20 * rubric.broadened_search_after_zero_results
    strategy += 0.20 * rubric.narrowed_search_after_many_results
    strategy += 0.25 * rubric.read_after_optimal_search
    
    strategy_penalty = 0.0
    if rubric.num_searches_with_zero_results > rubric.broadened_search_after_zero_results:
        strategy_penalty += 0.15 * (rubric.num_searches_with_zero_results - rubric.broadened_search_after_zero_results)
    
    if rubric.num_searches_with_too_many_results > rubric.narrowed_search_after_many_results:
        strategy_penalty += 0.15 * (rubric.num_searches_with_too_many_results - rubric.narrowed_search_after_many_results)
    
    strategy_penalty += 0.25 * rubric.ignored_optimal_results
    strategy -= strategy_penalty
    
    # Repetition penalties
    repetition = 0.0
    repetition += 0.12 * rubric.num_repeated_searches
    if rubric.repeated_zero_result_search:
        repetition += 0.15
    repetition += 0.20 * rubric.num_repeated_reads
    repetition += 0.25 * rubric.repeated_correct_email
    repetition += 0.10 * rubric.num_incorrect_sources
    
    # Calculate final reward
    if rubric.cant_parse_tool_call:
        reward = -2.5 + partial + strategy - repetition
    elif rubric.bad_tool_call_name:
        reward = -2.3 + partial + strategy - repetition
    elif rubric.bad_tool_call_args:
        reward = -2.1 + partial + strategy - repetition
    elif rubric.attempted_answer and not rubric.answer_correct:
        reward = -1.0 + partial + strategy - repetition
    elif rubric.answer_correct:
        is_perfect = (
            rubric.num_turns <= 4 and
            rubric.ever_found_right_email and
            rubric.ever_read_right_email and
            rubric.sources_correct and
            rubric.num_repeated_searches == 0 and
            rubric.num_repeated_reads == 0 and
            rubric.num_searches_with_zero_results <= 1 and
            rubric.read_after_optimal_search > 0
        )
        if is_perfect:
            reward = 3.0
        else:
            reward = 1.5
            if rubric.source_precision > 0:
                reward += 0.30 * rubric.source_precision
            if rubric.read_after_optimal_search > 0:
                reward += 0.20
            reward += strategy
            efficiency = 0.25 * (1 - rubric.num_turns / max(config.max_turns, 1))
            reward += efficiency
            reward -= repetition
            reward = min(reward, 2.8)
    else:
        reward = 0.0 + partial + strategy - repetition
    
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"Reward: {reward:.3f}")
    print(f"  Partial: {partial:.3f}")
    print(f"  Strategy: {strategy:.3f}")
    print(f"  Repetition penalty: {repetition:.3f}")
    
    return reward


def main():
    config = PolicyConfig(max_turns=10)
    
    print("="*70)
    print("REWARD FUNCTION VALIDATION TESTS")
    print("="*70)
    
    # Test 1: Perfect
    r1 = EvaluationRubric(
        answer_correct=True,
        sources_correct=True,
        num_turns=3,
        ever_found_right_email=True,
        ever_read_right_email=True,
        turn_found_right_email=1,
        turn_read_right_email=2,
        source_precision=1.0,
        num_correct_sources=1,
        read_after_optimal_search=1,
        num_searches_with_optimal_results=1,
    )
    test_scenario("✅ Test 1: Perfect Execution", r1, config)
    
    # Test 2: Repeated reading
    r2 = EvaluationRubric(
        answer_correct=True,
        sources_correct=True,
        num_turns=5,
        ever_found_right_email=True,
        ever_read_right_email=True,
        turn_found_right_email=1,
        turn_read_right_email=2,
        source_precision=1.0,
        num_correct_sources=1,
        num_repeated_reads=2,
        repeated_correct_email=1,
    )
    test_scenario("⚠️  Test 2: Repeated Reading (penalty)", r2, config)
    
    # Test 3: Good search strategy
    r3 = EvaluationRubric(
        answer_correct=True,
        sources_correct=True,
        num_turns=5,
        ever_found_right_email=True,
        ever_read_right_email=True,
        turn_found_right_email=3,
        turn_read_right_email=4,
        source_precision=1.0,
        num_correct_sources=1,
        num_searches_with_zero_results=1,
        broadened_search_after_zero_results=1,
        read_after_optimal_search=1,
    )
    test_scenario("✅ Test 3: Good Strategy (broadened after 0)", r3, config)
    
    # Test 4: Bad strategy
    r4 = EvaluationRubric(
        answer_correct=True,
        sources_correct=True,
        num_turns=6,
        ever_found_right_email=True,
        ever_read_right_email=True,
        source_precision=1.0,
        num_correct_sources=1,
        num_searches_with_too_many_results=2,
        narrowed_search_after_many_results=0,
    )
    test_scenario("⚠️  Test 4: Bad Strategy (didn't narrow)", r4, config)
    
    # Test 5: Poor source precision
    r5 = EvaluationRubric(
        answer_correct=True,
        sources_correct=False,
        num_turns=4,
        attempted_answer=True,
        ever_found_right_email=True,
        ever_read_right_email=True,
        turn_found_right_email=1,
        turn_read_right_email=2,
        num_sources=5,
        num_correct_sources=1,
        num_incorrect_sources=4,
        source_precision=0.20,
    )
    test_scenario("⚠️  Test 5: Poor Source Precision (1/5)", r5, config)
    
    print(f"\n{'='*70}")
    print("✅ All tests completed!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

