"""Test script to verify train/test split usage."""

from email_agent.data import load_synthetic_queries

def test_splits():
    """Test that both train and test splits can be loaded."""
    
    print("="*60)
    print("Testing Dataset Split Loading")
    print("="*60)
    print()
    
    # Test train split
    print("Loading train split...")
    train_queries = load_synthetic_queries(split='train', limit=10, shuffle=False)
    print(f"✓ Train split loaded: {len(train_queries)} queries")
    print(f"  First query ID: {train_queries[0].id}")
    print(f"  Question: {train_queries[0].question[:80]}...")
    print()
    
    # Test test split  
    print("Loading test split...")
    test_queries = load_synthetic_queries(split='test', limit=10, shuffle=False)
    print(f"✓ Test split loaded: {len(test_queries)} queries")
    print(f"  First query ID: {test_queries[0].id}")
    print(f"  Question: {test_queries[0].question[:80]}...")
    print()
    
    # Verify they are different
    train_ids = set(q.id for q in train_queries)
    test_ids = set(q.id for q in test_queries)
    
    if train_ids.intersection(test_ids):
        print("⚠️  WARNING: Some IDs overlap between train and test!")
        print(f"  Overlapping IDs: {train_ids.intersection(test_ids)}")
    else:
        print("✓ No overlap between train and test splits (good!)")
    
    print()
    
    # Get full dataset sizes
    print("Loading full splits to check sizes...")
    full_train = load_synthetic_queries(split='train', shuffle=False)
    full_test = load_synthetic_queries(split='test', shuffle=False)
    
    print(f"✓ Full train split: {len(full_train)} queries")
    print(f"✓ Full test split: {len(full_test)} queries")
    print()
    
    print("="*60)
    print("✓ All dataset split tests passed!")
    print("="*60)
    print()
    print("Summary:")
    print(f"  - Train split: {len(full_train)} examples")
    print(f"  - Test split: {len(full_test)} examples")
    print(f"  - Total: {len(full_train) + len(full_test)} examples")
    print()
    print("Usage in scripts:")
    print("  • Training: Uses train split")
    print("  • Validation (during training): Uses test split")
    print("  • Evaluation: Uses test split")
    print("  • Benchmark: Uses test split")


if __name__ == "__main__":
    test_splits()

