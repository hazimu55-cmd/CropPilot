from src.config import MIN_CHUNK_LENGTH


def apply_retrieval_gate(chunks: list) -> list:
    """
    Filter out weak or short chunks from retrieved information.
    This implements the retrieval gate from the architecture.
    
    Args:
        chunks: List of retrieved chunks with 'content' field
        
    Returns:
        Filtered list of chunks that meet quality criteria
    """
    filtered_chunks = []
    
    for chunk in chunks:
        content = chunk.get('content', '')
        
        # Filter 1: Minimum length check
        if len(content) < MIN_CHUNK_LENGTH:
            print(f"Filtering out short chunk (length: {len(content)})")
            continue
        
        # Filter 2: Check for meaningful content (not just whitespace/special chars)
        meaningful_chars = sum(1 for c in content if c.isalnum() or c.isspace())
        if meaningful_chars / len(content) < 0.5:
            print(f"Filtering out chunk with low meaningful content ratio")
            continue
        
        # Filter 3: Check for agricultural keywords (basic quality check)
        agricultural_keywords = [
            'treatment', 'disease', 'pest', 'chemical', 'organic', 
            'dosage', 'prevention', 'management', 'control', 'fungicide',
            'insecticide', 'fertilizer', 'symptom', 'infection'
        ]
        
        has_agri_content = any(keyword.lower() in content.lower() for keyword in agricultural_keywords)
        if not has_agri_content:
            print(f"Filtering out chunk without agricultural content")
            continue
        
        filtered_chunks.append(chunk)
    
    print(f"Retrieval gate: {len(chunks)} -> {len(filtered_chunks)} chunks")
    return filtered_chunks
