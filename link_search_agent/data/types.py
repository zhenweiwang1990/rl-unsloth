"""Data types for Link Search Agent."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LinkSearchQuery:
    """A link search query with ground truth handles.
    
    Attributes:
        id: Unique identifier for the query
        query: The natural language search query
        gold_handles: List of correct LinkedIn handles (ground truth)
    """
    id: str
    query: str
    gold_handles: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Ensure gold_handles is always a list
        if self.gold_handles is None:
            self.gold_handles = []
        # Normalize handles to lowercase
        self.gold_handles = [h.lower().strip() for h in self.gold_handles if h]


@dataclass 
class ProfileDetail:
    """Profile details returned by read_profile tool."""
    linkedin_handle: str
    name: Optional[str] = None
    summary: Optional[str] = None
    about: Optional[str] = None
    skills: Optional[str] = None
    experiences: Optional[List[dict]] = None
    education: Optional[List[dict]] = None
    meta: Optional[dict] = None

