"""Email Agent for GRPO training with unsloth."""

from .config import GRPOConfig, PolicyConfig
from .tools import search_emails, read_email, return_final_answer, SearchResult
from .data import SyntheticQuery, Email, load_synthetic_queries, generate_database

__all__ = [
    "GRPOConfig",
    "PolicyConfig",
    "search_emails",
    "read_email",
    "return_final_answer",
    "SearchResult",
    "SyntheticQuery",
    "Email",
    "load_synthetic_queries",
    "generate_database",
]

