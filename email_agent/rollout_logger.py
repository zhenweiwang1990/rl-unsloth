"""Detailed rollout logging for debugging and analysis."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from email_agent.data.types import SyntheticQuery
from email_agent.rollout import EvaluationRubric

logger = logging.getLogger(__name__)


@dataclass
class ToolCallLog:
    """Log entry for a single tool call."""
    turn_number: int
    tool_name: str
    tool_arguments: Dict[str, Any]
    tool_result: Any
    is_correct_email_found: bool = False  # For search_emails
    is_correct_email_read: bool = False   # For read_email
    result_count: int = 0  # Number of results for search
    error: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class JudgeLog:
    """Log entry for answer judging."""
    system_prompt: str
    user_prompt: str
    agent_answer: str
    ground_truth_answer: str
    judge_response: str
    is_correct: bool
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class RolloutLog:
    """Complete log for a single rollout."""
    # Query information
    query_id: str
    query_question: str
    query_answer: str
    query_inbox: str
    correct_message_ids: List[str]
    
    # System configuration
    system_prompt: str
    max_turns: int
    policy_config: Dict[str, Any]
    
    # Rollout metadata
    step: int
    rollout_index: int
    temperature: float
    repetition_penalty: float
    
    # Execution trace
    conversation_history: List[Dict[str, Any]]
    tool_calls: List[ToolCallLog]
    
    # Answer and judging
    final_answer: Optional[str] = None
    final_answer_sources: List[str] = None
    judge_log: Optional[JudgeLog] = None
    
    # Evaluation results
    rubric: Optional[Dict[str, Any]] = None
    reward: float = 0.0
    
    # Timing
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    
    # Token usage
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    def __post_init__(self):
        if not self.start_time:
            self.start_time = datetime.now().isoformat()
        if self.final_answer_sources is None:
            self.final_answer_sources = []


class RolloutLogBuilder:
    """Builder for constructing RolloutLog in memory during rollout execution.
    
    This class accumulates log data during a rollout without writing to disk,
    avoiding concurrency issues when multiple rollouts run in parallel.
    """
    
    def __init__(
        self,
        query: SyntheticQuery,
        system_prompt: str,
        max_turns: int,
        policy_config: Dict[str, Any],
        step: int,
        rollout_index: int,
        temperature: float = 0.7,
        repetition_penalty: float = 1.0,
    ):
        """Initialize the log builder.
        
        Args:
            query: The query being processed
            system_prompt: System prompt used
            max_turns: Maximum number of turns
            policy_config: Policy configuration dict
            step: Training step number
            rollout_index: Index of this rollout (0 to num_rollouts-1)
            temperature: Sampling temperature used
            repetition_penalty: Repetition penalty used
        """
        import time
        self.start_time = time.time()
        
        self.log = RolloutLog(
            query_id=query.id,
            query_question=query.question,
            query_answer=query.answer,
            query_inbox=query.inbox_address,
            correct_message_ids=query.message_ids,
            system_prompt=system_prompt,
            max_turns=max_turns,
            policy_config=policy_config,
            step=step,
            rollout_index=rollout_index,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            conversation_history=[],
            tool_calls=[],
            start_time=datetime.now().isoformat(),
        )
    
    def log_conversation_message(self, message: Dict[str, Any]) -> None:
        """Log a conversation message.
        
        Args:
            message: Conversation message dict (OpenAI format)
        """
        # Create a clean copy for logging
        clean_message = {
            "role": message.get("role"),
            "content": message.get("content"),
        }
        
        # Add tool-specific fields if present
        if message.get("tool_calls"):
            clean_message["tool_calls"] = message["tool_calls"]
        
        if message.get("tool_call_id"):
            clean_message["tool_call_id"] = message["tool_call_id"]
        
        self.log.conversation_history.append(clean_message)
    
    def log_tool_call(
        self,
        turn_number: int,
        tool_name: str,
        tool_arguments: Dict[str, Any],
        tool_result: Any,
        correct_message_id: str,
        error: Optional[str] = None,
    ) -> None:
        """Log a tool call execution.
        
        Args:
            turn_number: Turn number (1-indexed)
            tool_name: Name of the tool called
            tool_arguments: Arguments passed to tool
            tool_result: Result returned by tool
            correct_message_id: The correct message ID for this query
            error: Error message if tool call failed
        """
        # Analyze tool call for correctness
        is_correct_email_found = False
        is_correct_email_read = False
        result_count = 0
        
        if tool_name == "search_emails":
            if isinstance(tool_result, list):
                result_count = len(tool_result)
                # Check if correct email is in results
                for result in tool_result:
                    if isinstance(result, dict):
                        if result.get("message_id") == correct_message_id:
                            is_correct_email_found = True
                            break
        
        elif tool_name == "read_email":
            message_id = tool_arguments.get("message_id")
            if message_id == correct_message_id:
                is_correct_email_read = True
        
        # Create tool call log
        tool_log = ToolCallLog(
            turn_number=turn_number,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            tool_result=tool_result,
            is_correct_email_found=is_correct_email_found,
            is_correct_email_read=is_correct_email_read,
            result_count=result_count,
            error=error,
        )
        
        self.log.tool_calls.append(tool_log)
    
    def log_final_answer(
        self,
        answer: str,
        source_message_ids: List[str],
    ) -> None:
        """Log the final answer.
        
        Args:
            answer: Final answer text
            source_message_ids: List of source message IDs cited
        """
        self.log.final_answer = answer
        self.log.final_answer_sources = source_message_ids
    
    def log_judge_evaluation(
        self,
        system_prompt: str,
        user_prompt: str,
        agent_answer: str,
        ground_truth_answer: str,
        judge_response: str,
        is_correct: bool,
    ) -> None:
        """Log the judge evaluation process.
        
        Args:
            system_prompt: System prompt sent to judge
            user_prompt: User prompt sent to judge
            agent_answer: Answer provided by agent
            ground_truth_answer: Ground truth answer
            judge_response: Response from judge model
            is_correct: Whether judge determined answer is correct
        """
        self.log.judge_log = JudgeLog(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            agent_answer=agent_answer,
            ground_truth_answer=ground_truth_answer,
            judge_response=judge_response,
            is_correct=is_correct,
        )
    
    def build(
        self,
        rubric: EvaluationRubric,
        reward: float,
    ) -> RolloutLog:
        """Finalize and return the completed RolloutLog.
        
        Args:
            rubric: Evaluation rubric with metrics
            reward: Final reward value
            
        Returns:
            Complete RolloutLog object
        """
        import time
        
        # Calculate duration
        self.log.duration_seconds = time.time() - self.start_time
        
        # Set end time
        self.log.end_time = datetime.now().isoformat()
        
        # Add rubric and reward
        self.log.rubric = rubric.to_metrics()
        self.log.reward = reward
        
        # Add token usage from rubric
        self.log.total_input_tokens = rubric.total_input_tokens
        self.log.total_output_tokens = rubric.total_output_tokens
        
        return self.log


def save_rollout_logs(rollout_logs: List[RolloutLog], output_dir: str = "outputs/rollout_logs") -> List[str]:
    """Save multiple rollout logs to disk (batch write after step completes).
    
    Args:
        rollout_logs: List of RolloutLog objects to save
        output_dir: Base directory for rollout logs
        
    Returns:
        List of file paths where logs were saved
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for rollout_log in rollout_logs:
        # Create directory structure: step_X/query_Y/
        step_dir = output_path / f"step_{rollout_log.step}"
        query_dir = step_dir / f"query_{rollout_log.query_id}"
        query_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as rollout_Z.json
        filename = f"rollout_{rollout_log.rollout_index}.json"
        filepath = query_dir / filename
        
        # Convert to dict and save
        log_dict = asdict(rollout_log)
        
        # Convert tool_calls to dicts
        log_dict["tool_calls"] = [asdict(tc) for tc in rollout_log.tool_calls]
        
        # Convert judge_log to dict if present
        if rollout_log.judge_log:
            log_dict["judge_log"] = asdict(rollout_log.judge_log)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_dict, f, indent=2, ensure_ascii=False)
        
        saved_paths.append(str(filepath))
    
    if saved_paths:
        logger.info(f"Saved {len(saved_paths)} rollout logs to: {output_dir}")
    
    return saved_paths

