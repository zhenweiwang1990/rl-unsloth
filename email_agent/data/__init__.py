"""Data loading and management for email agent."""

from .types import SyntheticQuery, Email
from .query_loader import load_synthetic_queries
from .local_email_db import generate_database

__all__ = ["SyntheticQuery", "Email", "load_synthetic_queries", "generate_database"]

