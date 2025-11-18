"""Unified Email Agent for model inference and tool execution."""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

from email_agent.data.types import SyntheticQuery
from email_agent.tools import search_emails, read_email, SearchResult
from email_agent.config import PolicyConfig
from email_agent.prompts import create_system_prompt, get_tools_schema
from email_agent.rollout import (
    EvaluationRubric,
    determine_if_answer_is_correct,
    calculate_reward,
)
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class EmailAgent:
    """Unified email agent that handles model inference and tool execution.
    
    This agent:
    1. Takes a model and tokenizer
    2. Uses transformers' native tool calling support with OpenAI-format tools
    3. Parses OpenAI-formatted tool calls from model output
    4. Executes tools and updates conversation
    5. Tracks evaluation metrics
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        policy_config: PolicyConfig,
        openai_client: Optional[AsyncOpenAI] = None,
        rollout_index: int = 0,
        num_rollouts: int = 4,
    ):
        """Initialize the agent.
        
        Args:
            model: The language model (transformers AutoModelForCausalLM)
            tokenizer: The tokenizer (transformers AutoTokenizer)
            policy_config: Policy configuration
            openai_client: OpenAI client for judge (optional, only needed for evaluation)
            rollout_index: Index of this rollout within its group (0 to num_rollouts-1)
            num_rollouts: Total number of rollouts in the group
        """
        self.model = model
        self.tokenizer = tokenizer
        self.policy_config = policy_config
        self.openai_client = openai_client
        self.tools = get_tools_schema()
        self.rollout_index = rollout_index
        self.num_rollouts = num_rollouts
        
    async def run_query(
        self,
        query: SyntheticQuery,
        verbose: bool = False,
    ) -> Tuple[EvaluationRubric, List[Dict[str, Any]]]:
        """Run the agent on a single query.
        
        Args:
            query: The query to process
            verbose: Whether to print detailed logs
            
        Returns:
            Tuple of (rubric, conversation_history)
        """
        rubric = EvaluationRubric()
        
        # Track tool call history to detect repetitions
        # Key: (tool_name, normalized_args), Value: result (for zero-result detection)
        tool_call_history = {}
        
        # Create initial conversation
        system_prompt = create_system_prompt(query, self.policy_config.max_turns)
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query.question},
        ]
        
        if verbose:
            print("\n" + "="*80)
            print(f"QUERY {query.id}")
            print("="*80)
            print(f"Question: {query.question}")
            print(f"Ground Truth Answer: {query.answer}")
            print(f"Correct Message ID: {query.message_ids[0]}")
            print(f"Inbox: {query.inbox_address}")
            print("="*80)
        
        # Agent loop
        for turn in range(self.policy_config.max_turns):
            rubric.num_turns += 1
            
            if verbose:
                print(f"\n{'─'*80}")
                print(f"TURN {turn + 1}/{self.policy_config.max_turns}")
                print(f"{'─'*80}")
            
            try:
                # Generate model response using LiteLLM
                response_message, raw_content, input_tokens, output_tokens = self._generate_response(conversation, verbose)
                
                # Track token usage
                rubric.total_input_tokens += input_tokens
                rubric.total_output_tokens += output_tokens
                
                # Add assistant message to conversation
                conversation.append({
                    "role": "assistant",
                    "content": response_message.get("content"),
                    "tool_calls": response_message.get("tool_calls"),
                })
                
                # Check if there are tool calls from LiteLLM
                if response_message.get("tool_calls"):
                    # Execute tool calls (OpenAI format from LiteLLM)
                    should_break = await self._execute_tool_calls(
                        response_message["tool_calls"],
                        query,
                        rubric,
                        conversation,
                        tool_call_history,
                        verbose,
                    )
                    if should_break:
                        break
                        
                # Check if this is text response (no tool calls)
                elif response_message.get("content"):
                    # Model returned text instead of tool call
                    # This shouldn't happen with proper tool calling
                    rubric.cant_parse_tool_call = True
                    if verbose:
                        print(f"\n⚠️  Model returned text instead of tool call: {response_message['content'][:100]}")
                    break
                        
                else:
                    # Empty response
                    rubric.cant_parse_tool_call = True
                    if verbose:
                        print(f"\n⚠️  Model returned empty response")
                    break
                    
            except Exception as e:
                logger.error(f"Error in agent loop turn {turn + 1}: {e}")
                if verbose:
                    print(f"\n❌ Exception: {e}")
                    import traceback
                    traceback.print_exc()
                rubric.cant_parse_tool_call = True
                break
        
        # Check if ran out of turns
        if rubric.num_turns >= self.policy_config.max_turns and not rubric.attempted_answer:
            rubric.ran_out_of_turns = True
            if verbose:
                print(f"\n⏱️  Agent ran out of turns ({self.policy_config.max_turns})")
        
        if verbose:
            self._print_evaluation_summary(rubric, query)
        
        return rubric, conversation
    
    def _generate_response(
        self, 
        conversation: List[Dict], 
        verbose: bool
    ) -> Tuple[Dict[str, Any], str, int, int]:
        """Generate a response from the model with OpenAI-format tool calling.
        
        Uses transformers' native chat template with tools support.
        
        Args:
            conversation: Conversation history
            verbose: Whether to print logs
            
        Returns:
            Tuple of (response_message_dict, raw_content, input_tokens, output_tokens)
            - response_message_dict: Contains 'content' and/or 'tool_calls'
            - raw_content: Raw generated text for debugging
            - input_tokens: Number of input tokens
            - output_tokens: Number of output tokens
        """
        # Check if tokenizer supports tool calling via chat template
        try:
            # Format conversation with tools using chat template
            # Many modern tokenizers (Llama 3, Qwen, etc.) support tools parameter
            text = self.tokenizer.apply_chat_template(
                conversation,
                tools=self.tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (TypeError, ValueError) as e:
            # Fallback: tokenizer doesn't support tools parameter
            logger.warning(f"Tokenizer doesn't support tools parameter: {e}")
            # Use regular chat template without tools
            text = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        
        # Generate
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_tokens = inputs.input_ids.shape[1]
        
        # Calculate dynamic temperature and repetition penalty based on rollout index
        # This encourages exploration diversity within each group of rollouts
        if self.policy_config.enable_dynamic_temperature:
            temperature = self.policy_config.base_temperature + (
                self.rollout_index * self.policy_config.temperature_increment
            )
            repetition_penalty = self.policy_config.base_repetition_penalty + (
                self.rollout_index * self.policy_config.repetition_penalty_increment
            )
            
            if verbose:
                print(f"🎲 Rollout {self.rollout_index + 1}/{self.num_rollouts}: "
                      f"temp={temperature:.2f}, rep_penalty={repetition_penalty:.2f}")
        else:
            temperature = 0.7
            repetition_penalty = 1.0
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.policy_config.max_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        output_tokens = outputs.shape[1] - input_tokens
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][input_tokens:],
            skip_special_tokens=True,
        )
        
        # Parse tool calls from response if present
        response_message, parsed_successfully = self._parse_tool_calls_from_response(response, verbose)
        
        if verbose:
            print(f"\n📝 Model Response:")
            if response_message.get("tool_calls"):
                print(f"   Tool Calls: {len(response_message['tool_calls'])}")
                for tc in response_message["tool_calls"]:
                    func_name = tc['function']['name']
                    func_args = tc['function']['arguments'][:100]
                    print(f"   - {func_name}: {func_args}")
            elif response_message.get("content"):
                content = response_message["content"]
                if len(content) > 500:
                    print(f"   {content[:500]}...")
                else:
                    print(f"   {content}")
        
        return response_message, response, input_tokens, output_tokens
    
    def _parse_tool_calls_from_response(
        self,
        response: str,
        verbose: bool
    ) -> Tuple[Dict[str, Any], bool]:
        """Parse tool calls from model response.
        
        Supports multiple formats:
        1. OpenAI-style function calling XML/JSON tags
        2. Direct JSON tool call objects
        3. Plain text (no tool calls)
        
        Args:
            response: Raw model response
            verbose: Whether to print logs
            
        Returns:
            Tuple of (response_message_dict, success)
            - response_message_dict: Contains 'content' and/or 'tool_calls'
            - success: True if parsing was successful
        """
        import re
        
        # Try to parse structured tool calls
        # Look for <tool_call> tags (some models use this)
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        tool_call_matches = re.findall(tool_call_pattern, response, re.DOTALL)
        
        if tool_call_matches:
            # Parse tool calls from XML-like format
            tool_calls = []
            for i, match in enumerate(tool_call_matches):
                try:
                    # Parse JSON inside tool_call tags
                    tool_data = json.loads(match.strip())
                    tool_calls.append({
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tool_data.get("name", ""),
                            "arguments": json.dumps(tool_data.get("arguments", {})),
                        }
                    })
                except json.JSONDecodeError:
                    if verbose:
                        print(f"\n⚠️  Failed to parse tool call: {match[:100]}")
                    continue
            
            if tool_calls:
                return {
                    "content": None,
                    "tool_calls": tool_calls,
                }, True
        
        # Try to find JSON object that looks like a tool call
        # Look for patterns like {"name": "...", "arguments": {...}}
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}', response, re.DOTALL)
            if not json_match:
                # Try more complex nested JSON
                json_match = re.search(r'\{(?:[^{}]|{[^{}]*})*\}', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                
                # Check if it's a tool call format
                if "name" in parsed and "arguments" in parsed:
                    tool_calls = [{
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": parsed["name"],
                            "arguments": json.dumps(parsed["arguments"]),
                        }
                    }]
                    return {
                        "content": None,
                        "tool_calls": tool_calls,
                    }, True
                    
        except (json.JSONDecodeError, KeyError):
            pass
        
        # No tool calls found, return as plain text
        return {
            "content": response,
            "tool_calls": None,
        }, True
    
    async def _handle_return_final_answer(
        self,
        tool_args: Dict[str, Any],
        query: SyntheticQuery,
        rubric: EvaluationRubric,
        verbose: bool,
    ) -> bool:
        """Handle return_final_answer tool call.
        
        Args:
            tool_args: Tool arguments with 'answer' and 'source_message_ids'
            query: The query being processed
            rubric: Evaluation rubric to update
            verbose: Whether to print logs
            
        Returns:
            True (should break the agent loop)
        """
        final_answer = tool_args.get("answer", "")
        source_message_ids = tool_args.get("source_message_ids", [])
        
        if verbose:
            print(f"\n🎯 Agent returning final answer...")
            print(f"   Answer: {final_answer}")
            print(f"   Sources: {source_message_ids}")
            print(f"   Correct source: {query.message_ids[0]}")
        
        if not isinstance(source_message_ids, list):
            source_message_ids = []
        
        rubric.num_sources = len(source_message_ids)
        
        if final_answer == "I don't know":
            rubric.returned_i_dont_know = True
            if verbose:
                print(f"\n   ⚠️  Agent returned: I don't know")
        else:
            rubric.attempted_answer = True
            
            # Check sources
            rubric.sources_correct = query.message_ids[0] in source_message_ids
            
            # Call judge to check answer (if OpenAI client available)
            if self.openai_client:
                if verbose:
                    print(f"\n   Calling judge model to evaluate answer...")
                
                rubric.answer_correct = await self._judge_answer(
                    final_answer,
                    query,
                    verbose,
                )
                
                if verbose:
                    print(f"\n{'─'*60}")
                    print("ANSWER EVALUATION")
                    print(f"{'─'*60}")
                    print(f"Answer correct: {'✓ YES' if rubric.answer_correct else '✗ NO'}")
                    print(f"Sources correct: {'✓ YES' if rubric.sources_correct else '✗ NO'}")
                    print(f"{'─'*60}")
        
        return True  # Break the loop
    
    async def _execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        query: SyntheticQuery,
        rubric: EvaluationRubric,
        conversation: List[Dict],
        tool_call_history: Dict,
        verbose: bool,
    ) -> bool:
        """Execute tool calls and update conversation.
        
        Args:
            tool_calls: List of OpenAI-format tool call dictionaries
            query: The query being processed
            rubric: Evaluation rubric to update
            conversation: Conversation history to update
            tool_call_history: History of previous tool calls for repetition detection
            verbose: Whether to print logs
            
        Returns:
            True if should break the agent loop, False otherwise
        """
        should_break = False
        
        for tool_call in tool_calls:
            # Extract tool call info from OpenAI format
            tool_call_id = tool_call.get("id", "")
            tool_function = tool_call.get("function", {})
            tool_name = tool_function.get("name")
            
            # Parse arguments (they come as JSON string in OpenAI format)
            arguments_str = tool_function.get("arguments", "{}")
            try:
                tool_args = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
            except json.JSONDecodeError as e:
                rubric.bad_tool_call_args = True
                if verbose:
                    print(f"\n❌ Failed to parse tool arguments: {e}")
                    print(f"   Arguments string: {arguments_str}")
                should_break = True
                break
            
            if not tool_name:
                rubric.bad_tool_call_name = True
                if verbose:
                    print(f"\n❌ Missing tool_name in tool call")
                should_break = True
                break
            
            if verbose:
                print(f"\n🔧 Tool Call: {tool_name}")
                print(f"   ID: {tool_call_id}")
                print(f"   Arguments: {json.dumps(tool_args, indent=4)}")
            
            # Execute the tool
            tool_result, should_break_inner = await self._execute_single_tool(
                tool_name,
                tool_args,
                query,
                rubric,
                tool_call_history,
                verbose,
            )
            
            if verbose:
                print(f"\n📊 Tool Result:")
                if isinstance(tool_result, list):
                    print(f"   Returned {len(tool_result)} items")
                    for i, item in enumerate(tool_result[:3]):
                        print(f"   [{i+1}] {json.dumps(item, indent=6)}")
                    if len(tool_result) > 3:
                        print(f"   ... and {len(tool_result) - 3} more")
                elif isinstance(tool_result, dict):
                    if "error" in tool_result:
                        print(f"   ❌ {tool_result}")
                    else:
                        print(f"   ✓ {json.dumps(tool_result, indent=4)[:200]}...")
            
            # Add tool result to conversation in OpenAI format
            conversation.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(tool_result),
            })
            
            if should_break_inner:
                should_break = True
                break
        
        return should_break
    
    async def _execute_single_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        query: SyntheticQuery,
        rubric: EvaluationRubric,
        tool_call_history: Dict,
        verbose: bool,
    ) -> Tuple[Any, bool]:
        """Execute a single tool call.
        
        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments
            query: The query being processed
            rubric: Evaluation rubric to update
            tool_call_history: History of previous tool calls for repetition detection
            verbose: Whether to print logs
            
        Returns:
            Tuple of (tool_result, should_break)
        """
        should_break = False
        
        if tool_name == "search_emails":
            try:
                # Create a normalized key for this search to detect repetitions
                # Sort keywords for consistent comparison
                keywords = tool_args.get('keywords', [])
                if isinstance(keywords, list):
                    keywords = sorted(keywords)
                search_key = (
                    tool_name,
                    tuple(keywords) if isinstance(keywords, list) else keywords,
                    tool_args.get('from_addr', ''),
                    tool_args.get('to_addr', ''),
                )
                
                # Check if this exact search was already performed
                if search_key in tool_call_history:
                    rubric.num_repeated_searches += 1
                    prev_result = tool_call_history[search_key]
                    
                    # Check if repeating a zero-result search
                    if prev_result == 0:
                        rubric.repeated_zero_result_search = True
                    
                    if verbose:
                        print(f"\n⚠️  WARNING: Repeating identical search!")
                        print(f"   This search was already performed and returned {prev_result} results")
                        print(f"   Repetition penalty will be applied")
                
                if verbose:
                    print(f"\n🔍 Executing search_emails...")
                    print(f"   Keywords: {tool_args.get('keywords', [])}")
                    if tool_args.get('from_addr'):
                        print(f"   From: {tool_args.get('from_addr')}")
                    if tool_args.get('to_addr'):
                        print(f"   To: {tool_args.get('to_addr')}")
                
                search_results = search_emails(
                    **tool_args,
                    inbox=query.inbox_address,
                )
                result = [asdict(r) for r in search_results]
                
                # Track this search in history
                tool_call_history[search_key] = len(search_results)
                
                # Track zero-result searches
                if len(search_results) == 0:
                    rubric.num_zero_result_searches += 1
                
                # Check if we found the right email
                found_right = False
                for r in search_results:
                    if r.message_id == query.message_ids[0]:
                        rubric.ever_found_right_email = True
                        found_right = True
                
                if verbose:
                    print(f"\n✓ Search returned {len(search_results)} email(s)")
                    if found_right:
                        print(f"   ✓ CORRECT email found in results!")
                    else:
                        print(f"   ✗ Correct email NOT in results")
                
                return result, should_break
                
            except Exception as e:
                rubric.bad_tool_call_args = True
                logger.error(f"Error searching emails: {e}")
                should_break = True
                return {"error": str(e)}, should_break
                
        elif tool_name == "read_email":
            message_id_to_read = tool_args.get("message_id")
            
            if not isinstance(message_id_to_read, str):
                rubric.bad_tool_call_args = True
                should_break = True
                if verbose:
                    print(f"\n❌ Invalid message_id type: {type(message_id_to_read)}")
                return {"error": "Invalid message_id type"}, should_break
            
            if verbose:
                print(f"\n📧 Reading email: {message_id_to_read}")
            
            is_correct = message_id_to_read == query.message_ids[0]
            if is_correct:
                rubric.ever_read_right_email = True
            
            email_content = read_email(message_id_to_read)
            
            if email_content is None:
                rubric.ever_tried_to_read_invalid_email = True
                if verbose:
                    print(f"   ❌ Email not found!")
                return {"error": "Email not found"}, should_break
            else:
                if verbose:
                    print(f"   ✓ Successfully read email")
                    print(f"   Subject: {email_content.subject[:80]}...")
                    print(f"   From: {email_content.from_address}")
                    print(f"   Date: {email_content.date}")
                    if is_correct:
                        print(f"   ✓ This is the CORRECT email!")
                    else:
                        print(f"   ✗ This is NOT the correct email (correct: {query.message_ids[0]})")
                return email_content.model_dump(), should_break
                
        elif tool_name == "return_final_answer":
            # Handle final answer tool call
            should_break = await self._handle_return_final_answer(
                tool_args,
                query,
                rubric,
                verbose,
            )
            return {"status": "Final answer submitted"}, should_break
            
        else:
            rubric.bad_tool_call_name = True
            should_break = True
            logger.error(f"Unknown tool name: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}, should_break
    
    async def _judge_answer(
        self,
        answer: str,
        query: SyntheticQuery,
        verbose: bool,
    ) -> bool:
        """Use GPT-4o to judge if the answer is correct.
        
        Args:
            answer: The answer provided by the agent
            query: The query with ground truth
            verbose: Whether to print logs
            
        Returns:
            True if answer is correct, False otherwise
        """
        if not self.openai_client:
            logger.warning("OpenAI client not available for judging")
            return False
        
        return await determine_if_answer_is_correct(
            answer=answer,
            query=query,
            openai_client=self.openai_client,
            verbose=verbose,
        )
    
    def _print_evaluation_summary(self, rubric: EvaluationRubric, query: SyntheticQuery):
        """Print evaluation summary."""
        print(f"\n{'='*80}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Turns used: {rubric.num_turns}/{self.policy_config.max_turns}")
        print(f"Attempted answer: {'✓' if rubric.attempted_answer else '✗'}")
        print(f"Found right email: {'✓' if rubric.ever_found_right_email else '✗'}")
        print(f"Read right email: {'✓' if rubric.ever_read_right_email else '✗'}")
        print(f"Answer correct: {'✓' if rubric.answer_correct else '✗'}")
        print(f"Sources correct: {'✓' if rubric.sources_correct else '✗'}")
        
        # Show efficiency metrics
        if rubric.num_repeated_searches > 0 or rubric.num_zero_result_searches > 0:
            print(f"\n⚠️  Efficiency Issues:")
            if rubric.num_repeated_searches > 0:
                print(f"   Repeated identical searches: {rubric.num_repeated_searches}")
            if rubric.repeated_zero_result_search:
                print(f"   Repeated zero-result search: ✗ (extra penalty)")
            if rubric.num_zero_result_searches > 0:
                print(f"   Total zero-result searches: {rubric.num_zero_result_searches}")
        
        # Calculate reward
        reward = calculate_reward(self.policy_config, rubric)
        print(f"\n🎯 Final Reward: {reward:.3f}")
        print(f"{'='*80}\n")

