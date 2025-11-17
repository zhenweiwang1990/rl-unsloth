"""Data types for email agent."""

from pydantic import BaseModel
from typing import List, Optional


class SyntheticQuery(BaseModel):
    """Represents a synthetic query with ground truth answer."""
    
    id: int
    question: str
    answer: str
    message_ids: List[str]  # Message IDs of referenced emails
    how_realistic: float
    inbox_address: str
    query_date: str


class Email(BaseModel):
    """Represents an email message."""
    
    message_id: str
    date: str  # ISO 8601 string 'YYYY-MM-DD HH:MM:SS'
    subject: Optional[str] = None
    from_address: Optional[str] = None
    to_addresses: List[str] = []
    cc_addresses: List[str] = []
    bcc_addresses: List[str] = []
    body: Optional[str] = None
    file_name: Optional[str] = None

