"""Model loading utilities with compatibility fixes for Qwen3 and other models."""

import os
import logging

logger = logging.getLogger(__name__)


def load_model_with_unsloth(model_name: str, max_seq_length: int = 32768, load_in_4bit: bool = True):
    """Load model using unsloth with compatibility fixes.
    
    Args:
        model_name: HuggingFace model identifier
        max_seq_length: Maximum sequence length
        load_in_4bit: Whether to use 4-bit quantization
    
    Returns:
        Tuple of (model, tokenizer)
    """
    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer
    
    # Set environment variables for compatibility
    original_trust = os.environ.get("TRUST_REMOTE_CODE", "")
    os.environ["TRUST_REMOTE_CODE"] = "true"
    
    try:
        logger.info(f"Loading model: {model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=None,
        )
        logger.info("✓ Model loaded successfully")
        return model, tokenizer
            
    except (AttributeError, TypeError, KeyError, ValueError, RuntimeError) as e:
        error_str = str(e)
        logger.error(f"Failed to load model '{model_name}': {error_str}")
        
        # Check if it's a config file issue
        if "config file" in error_str.lower() or "no config" in error_str.lower():
            logger.error("\nTroubleshooting suggestions:")
            logger.error(f"  1. Verify the model '{model_name}' exists on HuggingFace")
            logger.error("  2. Check network connectivity to huggingface.co")
            logger.error("  3. Try clearing the cache:")
            logger.error(f"     rm -rf ~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}")
            logger.error("  4. Check if the model name is correct (case-sensitive)")
            logger.error("  5. Ensure you have HF_TOKEN set if the model is private")
            raise
        
        # Handle tokenizer compatibility issues
        if "model_type" in error_str or ("dict" in error_str.lower() and "attribute" in error_str.lower()):
            logger.warning("Tokenizer compatibility issue detected. Trying alternative method...")
            
            try:
                logger.info("Loading tokenizer separately...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )
                
                logger.info("Loading model with unsloth...")
                model, _ = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=max_seq_length,
                    load_in_4bit=load_in_4bit,
                    dtype=None,
                )
                
                logger.info("✓ Model loaded with alternative method")
                return model, tokenizer
                
            except Exception as e2:
                logger.error(f"Alternative loading method failed: {e2}")
                logger.error("\nTroubleshooting suggestions:")
                logger.error("  1. Update transformers: pip install --upgrade transformers>=4.40.0")
                logger.error("  2. Try a different model variant")
                logger.error("  3. Check model compatibility with unsloth")
                raise e  # Re-raise original error
        else:
            # Re-raise unknown errors
            raise
    finally:
        # Restore original values
        if original_trust:
            os.environ["TRUST_REMOTE_CODE"] = original_trust
        elif "TRUST_REMOTE_CODE" in os.environ:
            del os.environ["TRUST_REMOTE_CODE"]

