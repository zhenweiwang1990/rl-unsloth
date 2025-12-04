"""Load link search queries from HuggingFace dataset."""

import os
import random
from typing import List, Optional

from datasets import load_dataset, Dataset

from link_search_agent.data.types import LinkSearchQuery

# HuggingFace repository containing the link search dataset
HF_REPO_ID = os.environ.get("HF_DATASET_ID", "gboxai/linksearch")
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", None))


def normalize_handle(raw: str) -> str:
    """Normalize a LinkedIn handle."""
    if not raw or not isinstance(raw, str):
        return ""
    s = raw.strip()
    # Extract handle from URL if present
    if "linkedin.com/in/" in s.lower():
        import re
        match = re.search(r'linkedin\.com/in/([^?/#]+)', s, re.IGNORECASE)
        if match:
            s = match.group(1)
    # URL decode
    try:
        import urllib.parse
        s = urllib.parse.unquote(s)
    except:
        pass
    # Clean up
    s = s.replace("?", "").replace("#", "").rstrip("/")
    return s.lower().strip()


def parse_handles(handles_data) -> List[str]:
    """Parse handles from various formats."""
    result = []
    
    if handles_data is None:
        return result
    
    # If it's already a list
    if isinstance(handles_data, list):
        for item in handles_data:
            if isinstance(item, str):
                h = normalize_handle(item)
                if h:
                    result.append(h)
            elif isinstance(item, dict):
                # Handle nested format like {"content": [...]}
                content = item.get("content", [])
                if isinstance(content, list):
                    for c in content:
                        h = normalize_handle(str(c) if c else "")
                        if h:
                            result.append(h)
    
    # If it's a string (possibly semicolon-separated)
    elif isinstance(handles_data, str):
        for part in handles_data.split(";"):
            h = normalize_handle(part)
            if h:
                result.append(h)
    
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for h in result:
        if h not in seen:
            seen.add(h)
            deduped.append(h)
    
    return deduped


def load_link_search_queries(
    split: str = "train",
    limit: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
) -> List[LinkSearchQuery]:
    """Load link search queries from HuggingFace dataset.
    
    Args:
        split: Dataset split to load ('train' or 'test')
        limit: Maximum number of queries to return
        shuffle: Whether to shuffle the queries
        seed: Random seed for shuffling
        
    Returns:
        List of LinkSearchQuery objects
    """
    print(f"Loading link search queries from {HF_REPO_ID} ({split} split)...")
    
    try:
        dataset: Dataset = load_dataset(HF_REPO_ID, split=split, token=HF_TOKEN)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("If the dataset is private, set HF_TOKEN environment variable.")
        raise
    
    queries = []
    
    for row in dataset:
        query_id = str(row.get("id", len(queries)))
        query_text = row.get("query", "")
        
        # Parse handles from the 'handles' field
        handles_data = row.get("handles", [])
        gold_handles = parse_handles(handles_data)
        
        # Skip queries without gold handles
        if not gold_handles:
            continue
        
        queries.append(LinkSearchQuery(
            id=query_id,
            query=query_text,
            gold_handles=gold_handles,
        ))
    
    print(f"Loaded {len(queries)} queries with gold handles")
    
    if shuffle:
        random.seed(seed)
        random.shuffle(queries)
    
    if limit is not None:
        queries = queries[:limit]
    
    return queries

