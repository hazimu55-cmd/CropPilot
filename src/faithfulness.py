from src.config import FAITHFULNESS_THRESHOLD
import re


def calculate_word_overlap(response: str, context: str) -> float:
    """
    Calculate word overlap between generated response and retrieved context.
    This implements a simple faithfulness check using word-overlap heuristic.
    
    Args:
        response: Generated response from LLM
        context: Retrieved context documents
        
    Returns:
        Word overlap ratio (0.0 to 1.0)
    """
    # Extract words from both texts (lowercase, remove punctuation)
    response_words = set(re.findall(r'\b\w+\b', response.lower()))
    context_words = set(re.findall(r'\b\w+\b', context.lower()))
    
    if not response_words:
        return 0.0
    
    # Calculate overlap
    overlap = len(response_words & context_words)
    overlap_ratio = overlap / len(response_words)
    
    return overlap_ratio


def check_faithfulness(response: str, chunks: list) -> tuple:
    """
    Check faithfulness of generated response against retrieved documents.
    
    Args:
        response: Generated response from LLM
        chunks: List of retrieved chunks with 'content' field
        
    Returns:
        Tuple of (is_faithful: bool, overlap_ratio: float)
    """
    # Combine all chunk content
    combined_context = " ".join(chunk.get('content', '') for chunk in chunks)
    
    # Calculate word overlap
    overlap_ratio = calculate_word_overlap(response, combined_context)
    
    # Check against threshold
    is_faithful = overlap_ratio >= FAITHFULNESS_THRESHOLD
    
    print(f"Faithfulness check: overlap_ratio={overlap_ratio:.3f}, threshold={FAITHFULNESS_THRESHOLD}, faithful={is_faithful}")
    
    return is_faithful, overlap_ratio


def format_faithfulness_warning(overlap_ratio: float) -> str:
    """
    Format a warning message for low faithfulness responses.
    """
    return f"\n⚠️ Note: This response has lower confidence (faithfulness score: {overlap_ratio:.1%}). Please verify with agricultural experts before taking action."
