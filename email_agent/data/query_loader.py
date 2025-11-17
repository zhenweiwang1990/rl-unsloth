"""Load synthetic queries from Hugging Face dataset."""

from .types import SyntheticQuery
from typing import List, Optional
from datasets import load_dataset, Dataset
import random

# Hugging Face repository containing the Enron email dataset
HF_REPO_ID = "corbt/enron_emails_sample_questions"

# Queries that have been manually identified as ambiguous or incorrect
BAD_QUERIES = [49, 101, 129, 171, 208, 266, 327]


def load_synthetic_queries(
    split: str = "train",
    limit: Optional[int] = None,
    max_messages: Optional[int] = 1,
    shuffle: bool = False,
    exclude_known_bad_queries: bool = True,
) -> List[SyntheticQuery]:
    """Load synthetic email queries from the dataset.
    
    Args:
        split: Dataset split to load ('train' or 'test')
        limit: Maximum number of queries to return
        max_messages: Filter queries to those with at most this many messages
        shuffle: Whether to shuffle the queries
        exclude_known_bad_queries: Whether to exclude known problematic queries
        
    Returns:
        List of SyntheticQuery objects
    """
    dataset: Dataset = load_dataset(HF_REPO_ID, split=split)  # type: ignore

    if max_messages is not None:
        dataset = dataset.filter(lambda x: len(x["message_ids"]) <= max_messages)

    if exclude_known_bad_queries:
        dataset = dataset.filter(lambda x: x["id"] not in BAD_QUERIES)

    if shuffle:
        dataset = dataset.shuffle()

    # Convert dataset rows to SyntheticQuery objects
    queries = [SyntheticQuery(**row) for row in dataset]  # type: ignore

    if max_messages is not None:
        queries = [query for query in queries if len(query.message_ids) <= max_messages]

    if shuffle:
        random.shuffle(queries)

    if limit is not None:
        return queries[:limit]
    
    return queries

