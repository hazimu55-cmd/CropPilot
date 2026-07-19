"""
Reliability & Evaluation Layer
Contains confidence gates, retrieval gates, and faithfulness checks
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GateResult:
    """Result of a gate check"""
    passed: bool
    confidence: float
    reason: str
    filtered_data: Optional[Dict] = None


class ConfidenceGate:
    """
    Confidence gate to filter weak diagnoses
    Only allows predictions above a confidence threshold
    """
    
    def __init__(self, threshold: float = 0.70):
        """
        Initialize confidence gate
        
        Args:
            threshold: Minimum confidence threshold (default: 0.70)
        """
        self.threshold = threshold
    
    def check(self, prediction: Dict) -> GateResult:
        """
        Check if prediction passes confidence gate
        
        Args:
            prediction: Dictionary with 'confidence' key
            
        Returns:
            GateResult with pass/fail status
        """
        confidence = prediction.get("confidence", 0.0)
        
        if confidence >= self.threshold:
            return GateResult(
                passed=True,
                confidence=confidence,
                reason=f"Confidence {confidence:.2%} meets threshold {self.threshold:.2%}",
                filtered_data=prediction
            )
        else:
            return GateResult(
                passed=False,
                confidence=confidence,
                reason=f"Confidence {confidence:.2%} below threshold {self.threshold:.2%}",
                filtered_data=None
            )
    
    def filter_predictions(self, predictions: List[Dict]) -> Tuple[List[Dict], List[GateResult]]:
        """
        Filter multiple predictions through confidence gate
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Tuple of (filtered predictions, gate results)
        """
        results = []
        filtered = []
        
        for pred in predictions:
            result = self.check(pred)
            results.append(result)
            if result.passed:
                filtered.append(result.filtered_data)
        
        return filtered, results


class RetrievalGate:
    """
    Retrieval gate to filter weak RAG results
    Filters based on relevance scores and content quality
    """
    
    def __init__(self, min_relevance: float = 0.5, min_content_length: int = 50):
        """
        Initialize retrieval gate
        
        Args:
            min_relevance: Minimum relevance score threshold
            min_content_length: Minimum content length in characters
        """
        self.min_relevance = min_relevance
        self.min_content_length = min_content_length
    
    def check(self, chunk: Dict) -> GateResult:
        """
        Check if retrieved chunk passes retrieval gate
        
        Args:
            chunk: Dictionary with 'content' and optional 'score'
            
        Returns:
            GateResult with pass/fail status
        """
        content = chunk.get("content", "")
        score = chunk.get("score", 1.0)  # Default to high score if not provided
        
        content_length = len(content.strip())
        
        # Check content length
        if content_length < self.min_content_length:
            return GateResult(
                passed=False,
                confidence=score,
                reason=f"Content length {content_length} below minimum {self.min_content_length}",
                filtered_data=None
            )
        
        # Check relevance score
        if score < self.min_relevance:
            return GateResult(
                passed=False,
                confidence=score,
                reason=f"Relevance score {score:.2f} below threshold {self.min_relevance}",
                filtered_data=None
            )
        
        return GateResult(
            passed=True,
            confidence=score,
            reason=f"Content passes all gates (length: {content_length}, score: {score:.2f})",
            filtered_data=chunk
        )
    
    def filter_chunks(self, chunks: List[Dict]) -> Tuple[List[Dict], List[GateResult]]:
        """
        Filter multiple chunks through retrieval gate
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Tuple of (filtered chunks, gate results)
        """
        results = []
        filtered = []
        
        for chunk in chunks:
            result = self.check(chunk)
            results.append(result)
            if result.passed:
                filtered.append(result.filtered_data)
        
        return filtered, results


class FaithfulnessChecker:
    """
    Faithfulness check using RAGAS + citations
    Verifies that LLM outputs are faithful to retrieved context
    """
    
    def __init__(self):
        """Initialize faithfulness checker"""
        # RAGAS evaluation would be initialized here
        # For now, we'll implement a simplified version
        self.citation_keywords = ["according to", "based on", "as per", "source", "document"]
    
    def check_citations(self, response: str, context_chunks: List[Dict]) -> GateResult:
        """
        Check if response properly cites sources
        
        Args:
            response: LLM generated response
            context_chunks: Retrieved context chunks
            
        Returns:
            GateResult with pass/fail status
        """
        response_lower = response.lower()
        
        # Check if response contains citation indicators
        has_citations = any(keyword in response_lower for keyword in self.citation_keywords)
        
        # Check if response mentions sources
        mentions_source = any(
            chunk.get("source", "") in response or 
            f"page {chunk.get('page', '')}" in response
            for chunk in context_chunks
        )
        
        if has_citations or mentions_source:
            return GateResult(
                passed=True,
                confidence=0.8,
                reason="Response contains citations or source references",
                filtered_data={"has_citations": has_citations, "mentions_source": mentions_source}
            )
        else:
            return GateResult(
                passed=False,
                confidence=0.3,
                reason="Response lacks proper citations or source references",
                filtered_data=None
            )
    
    def check_faithfulness_simple(self, response: str, context_chunks: List[Dict]) -> GateResult:
        """
        Simplified faithfulness check
        Verifies that response doesn't hallucinate information not in context
        
        Args:
            response: LLM generated response
            context_chunks: Retrieved context chunks
            
        Returns:
            GateResult with pass/fail status
        """
        # Combine all context
        context_text = " ".join([chunk.get("content", "") for chunk in context_chunks])
        context_words = set(context_text.lower().split())
        
        # Extract key terms from response (simplified)
        response_words = set(response.lower().split())
        
        # Check if response contains many words not in context
        # This is a very simplified check - real RAGAS would be more sophisticated
        unique_response_words = response_words - context_words
        
        # Allow some unique words (connectors, etc.) but flag if too many
        unique_ratio = len(unique_response_words) / max(len(response_words), 1)
        
        if unique_ratio < 0.5:  # If less than 50% of words are unique
            return GateResult(
                passed=True,
                confidence=1.0 - unique_ratio,
                reason=f"Response appears faithful (unique word ratio: {unique_ratio:.2%})",
                filtered_data={"unique_ratio": unique_ratio}
            )
        else:
            return GateResult(
                passed=False,
                confidence=1.0 - unique_ratio,
                reason=f"Response may contain hallucinations (unique word ratio: {unique_ratio:.2%})",
                filtered_data=None
            )
    
    def comprehensive_check(self, response: str, context_chunks: List[Dict]) -> Dict[str, GateResult]:
        """
        Perform comprehensive faithfulness check
        
        Args:
            response: LLM generated response
            context_chunks: Retrieved context chunks
            
        Returns:
            Dictionary of check results
        """
        return {
            "citations": self.check_citations(response, context_chunks),
            "faithfulness": self.check_faithfulness_simple(response, context_chunks)
        }


class ReliabilityLayer:
    """
    Main reliability layer that orchestrates all gates and checks
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.70,
        retrieval_min_relevance: float = 0.5,
        retrieval_min_content_length: int = 50
    ):
        """
        Initialize reliability layer
        
        Args:
            confidence_threshold: Threshold for confidence gate
            retrieval_min_relevance: Minimum relevance for retrieval gate
            retrieval_min_content_length: Minimum content length for retrieval gate
        """
        self.confidence_gate = ConfidenceGate(threshold=confidence_threshold)
        self.retrieval_gate = RetrievalGate(
            min_relevance=retrieval_min_relevance,
            min_content_length=retrieval_min_content_length
        )
        self.faithfulness_checker = FaithfulnessChecker()
    
    def evaluate_diagnosis(self, prediction: Dict) -> GateResult:
        """
        Evaluate diagnosis through confidence gate
        
        Args:
            prediction: Disease prediction
            
        Returns:
            GateResult
        """
        return self.confidence_gate.check(prediction)
    
    def evaluate_retrieval(self, chunks: List[Dict]) -> Tuple[List[Dict], List[GateResult]]:
        """
        Evaluate retrieved chunks through retrieval gate
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            Tuple of (filtered chunks, gate results)
        """
        return self.retrieval_gate.filter_chunks(chunks)
    
    def evaluate_response(self, response: str, context_chunks: List[Dict]) -> Dict[str, GateResult]:
        """
        Evaluate LLM response through faithfulness checks
        
        Args:
            response: LLM generated response
            context_chunks: Retrieved context chunks
            
        Returns:
            Dictionary of check results
        """
        return self.faithfulness_checker.comprehensive_check(response, context_chunks)
