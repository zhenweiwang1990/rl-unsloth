"""Test script to verify the setup is correct."""

import sys
import os

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers: {e}")
        return False
    
    try:
        import datasets
        print(f"✓ Datasets {datasets.__version__}")
    except ImportError as e:
        print(f"✗ Datasets: {e}")
        return False
    
    try:
        import pydantic
        print(f"✓ Pydantic {pydantic.__version__}")
    except ImportError as e:
        print(f"✗ Pydantic: {e}")
        return False
    
    try:
        import trl
        print(f"✓ TRL {trl.__version__}")
    except ImportError as e:
        print(f"✗ TRL: {e}")
        return False
    
    try:
        from unsloth import FastLanguageModel
        print("✓ Unsloth")
    except ImportError as e:
        print(f"✗ Unsloth: {e}")
        return False
    
    try:
        import openai
        print(f"✓ OpenAI {openai.__version__}")
    except ImportError as e:
        print(f"✗ OpenAI: {e}")
        return False
    
    return True


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    
    import torch
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  - Device count: {torch.cuda.device_count()}")
        print(f"  - Current device: {torch.cuda.current_device()}")
        print(f"  - Device name: {torch.cuda.get_device_name()}")
        return True
    else:
        print("⚠ CUDA not available (CPU training will be very slow)")
        return False


def test_email_agent():
    """Test that email_agent package can be imported."""
    print("\nTesting email_agent package...")
    
    try:
        from email_agent.config import GRPOConfig, PolicyConfig
        print("✓ Config imported")
        
        from email_agent.tools import search_emails, read_email
        print("✓ Tools imported")
        
        from email_agent.data import SyntheticQuery, Email
        print("✓ Data types imported")
        
        from email_agent.rollout import calculate_reward, EvaluationRubric
        print("✓ Rollout imported")
        
        return True
    except ImportError as e:
        print(f"✗ Email agent import failed: {e}")
        return False


def test_database():
    """Test database existence."""
    print("\nTesting database...")
    
    db_path = "enron_emails.db"
    if os.path.exists(db_path):
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        print(f"✓ Database exists ({size_mb:.1f} MB)")
        return True
    else:
        print("⚠ Database not found. Run ./scripts/generate_database.sh")
        return False


def test_openai_key():
    """Test OpenAI API key."""
    print("\nTesting OpenAI API key...")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key and api_key != "your_openai_api_key_here":
        print(f"✓ OpenAI API key set ({api_key[:10]}...)")
        return True
    else:
        print("⚠ OpenAI API key not set. Set OPENAI_API_KEY environment variable.")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Email Agent GRPO Training - Setup Test")
    print("="*60)
    print()
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("Email Agent", test_email_agent()))
    results.append(("Database", test_database()))
    results.append(("OpenAI Key", test_openai_key()))
    
    print()
    print("="*60)
    print("Test Results")
    print("="*60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:20s} {status}")
    
    print()
    
    # Check critical components
    critical = ["Imports", "Email Agent"]
    critical_pass = all(result for name, result in results if name in critical)
    
    if critical_pass:
        print("✓ Critical components working! You can start training.")
    else:
        print("✗ Critical components missing. Please fix errors above.")
        sys.exit(1)
    
    # Check optional components
    optional = ["CUDA", "Database", "OpenAI Key"]
    optional_pass = all(result for name, result in results if name in optional)
    
    if not optional_pass:
        print("⚠ Some optional components missing:")
        for name, result in results:
            if name in optional and not result:
                if name == "Database":
                    print("  - Run: ./scripts/generate_database.sh")
                elif name == "OpenAI Key":
                    print("  - Set: export OPENAI_API_KEY='your_key'")
                elif name == "CUDA":
                    print("  - Install CUDA and NVIDIA drivers")
    
    print()
    print("Next steps:")
    print("1. Generate database: ./scripts/generate_database.sh")
    print("2. Set OpenAI key: export OPENAI_API_KEY='your_key'")
    print("3. Start training: python train_grpo.py")
    print("="*60)


if __name__ == "__main__":
    main()

