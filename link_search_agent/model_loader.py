"""Model loading utilities using unsloth."""

import logging

logger = logging.getLogger(__name__)


def load_model_with_unsloth(model_name: str, max_seq_length: int = 32768, load_in_4bit: bool = True):
    """Load model using unsloth.
    
    Args:
        model_name: HuggingFace model identifier
        max_seq_length: Maximum sequence length
        load_in_4bit: Whether to use 4-bit quantization
    
    Returns:
        Tuple of (model, tokenizer)
    """
    from unsloth import FastLanguageModel
    
    logger.info(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,
    )
    logger.info("✓ Model loaded successfully")
    return model, tokenizer
