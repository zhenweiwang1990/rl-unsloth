"""Unified Email Agent for model inference and tool execution."""

import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

from email_agent.data.types import SyntheticQuery
from email_agent.tools import search_emails, read_email, SearchResult
from email_agent.config import PolicyConfig
from email_agent.prompts import create_system_prompt
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
    2. Generates responses using transformers
    3. Parses JSON-formatted tool calls
    4. Executes tools and updates conversation
    5. Tracks evaluation metrics
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        policy_config: PolicyConfig,
        openai_client: Optional[AsyncOpenAI] = None,
    ):
        """Initialize the agent.
        
        Args:
            model: The language model (transformers AutoModelForCausalLM)
            tokenizer: The tokenizer (transformers AutoTokenizer)
            policy_config: Policy configuration
            openai_client: OpenAI client for judge (optional, only needed for evaluation)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.policy_config = policy_config
        self.openai_client = openai_client
        
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
                # Generate model response
                response = self._generate_response(conversation, verbose)
                
                # Add assistant message to conversation
                conversation.append({"role": "assistant", "content": response})
                
                # Parse tool calls or final answer from response
                parsed_data = self._parse_response(response, verbose)
                
                if parsed_data is None:
                    # Could not parse response
                    rubric.cant_parse_tool_call = True
                    if verbose:
                        print(f"\n⚠️  Could not parse JSON from model response")
                    break
                
                # Check if this is a final answer
                if "final_answer" in parsed_data:
                    # Handle final answer
                    should_break = await self._handle_final_answer(
                        parsed_data,
                        query,
                        rubric,
                        verbose,
                    )
                    if should_break:
                        break
                        
                # Check if there are tool calls
                elif "tool_calls" in parsed_data:
                    # Execute tool calls
                    should_break = await self._execute_tool_calls(
                        parsed_data["tool_calls"],
                        query,
                        rubric,
                        conversation,
                        verbose,
                    )
                    if should_break:
                        break
                        
                else:
                    # Invalid format
                    rubric.cant_parse_tool_call = True
                    if verbose:
                        print(f"\n⚠️  Response JSON missing both 'tool_calls' and 'final_answer'")
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
    
    def _generate_response(self, conversation: List[Dict], verbose: bool) -> str:
        """Generate a response from the model.
        
        Args:
            conversation: Conversation history
            verbose: Whether to print logs
            
        Returns:
            Generated response text
        """
        # Format conversation for model
        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Generate
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.policy_config.max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        
        if verbose:
            print(f"\n📝 Model Response:")
            if len(response) > 500:
                print(f"   {response[:500]}...")
            else:
                print(f"   {response}")
        
        return response
    
    def _parse_response(self, response: str, verbose: bool) -> Optional[Dict[str, Any]]:
        """Parse JSON from model response.
        
        Args:
            response: Model response text
            verbose: Whether to print logs
            
        Returns:
            Parsed JSON dictionary, or None if parsing failed
        """
        try:
            # Try to extract JSON from response
            # Look for JSON object (content between { and })
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                return parsed
            else:
                if verbose:
                    print(f"\n⚠️  No JSON object found in response")
                return None
                
        except json.JSONDecodeError as e:
            if verbose:
                print(f"\n⚠️  JSON parse error: {e}")
            return None
    
    async def _handle_final_answer(
        self,
        parsed_data: Dict[str, Any],
        query: SyntheticQuery,
        rubric: EvaluationRubric,
        verbose: bool,
    ) -> bool:
        """Handle final answer from the model.
        
        Args:
            parsed_data: Parsed JSON data with final_answer
            query: The query being processed
            rubric: Evaluation rubric to update
            verbose: Whether to print logs
            
        Returns:
            True (should break the agent loop)
        """
        final_answer = parsed_data.get("final_answer")
        source_message_ids = parsed_data.get("source_message_ids", [])
        
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
                    print(f"\n   Calling GPT-4o judge to evaluate answer...")
                
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
        verbose: bool,
    ) -> bool:
        """Execute tool calls and update conversation.
        
        Args:
            tool_calls: List of tool call dictionaries
            query: The query being processed
            rubric: Evaluation rubric to update
            conversation: Conversation history to update
            verbose: Whether to print logs
            
        Returns:
            True if should break the agent loop, False otherwise
        """
        should_break = False
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool_name")
            tool_args = tool_call.get("arguments", {})
            
            if not tool_name:
                rubric.bad_tool_call_name = True
                if verbose:
                    print(f"\n❌ Missing tool_name in tool call")
                should_break = True
                break
            
            if verbose:
                print(f"\n🔧 Tool Call: {tool_name}")
                print(f"   Arguments: {json.dumps(tool_args, indent=4)}")
            
            # Execute the tool
            tool_result, should_break_inner = await self._execute_single_tool(
                tool_name,
                tool_args,
                query,
                rubric,
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
            
            # Add tool result to conversation
            conversation.append({
                "role": "tool",
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
        verbose: bool,
    ) -> Tuple[Any, bool]:
        """Execute a single tool call.
        
        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments
            query: The query being processed
            rubric: Evaluation rubric to update
            verbose: Whether to print logs
            
        Returns:
            Tuple of (tool_result, should_break)
        """
        should_break = False
        
        if tool_name == "search_emails":
            try:
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
            # This should not happen here (handled separately)
            # But handle it gracefully
            rubric.bad_tool_call_name = True
            logger.warning("return_final_answer called as tool_call instead of final_answer format")
            return {"error": "Use final_answer format instead"}, True
            
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
        
        # Calculate reward
        reward = calculate_reward(self.policy_config, rubric)
        print(f"\n🎯 Final Reward: {reward:.3f}")
        print(f"{'='*80}\n")

