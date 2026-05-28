"""
Agentic RAG - Reasoner Module
Evaluates results and decides next actions
"""
from typing import Dict, Any, List, Optional

class Reasoner:
    """Evaluates retrieval results and decides next steps"""
    
    def __init__(self):
        self.decision_history = []
    
    def evaluate(self, step_result: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Evaluate step result and decide next action
        
        Returns:
            Decision dict with 'action' (continue, retry, refine, or complete)
        """
        retrieved_chunks = step_result.get('retrieved_chunks', [])
        
        # Check if we have results
        if not retrieved_chunks:
            return {
                "action": "retry",
                "reason": "No results found",
                "modified_query": self._refine_query(original_query)
            }
        
        # Check result quality
        best_score = retrieved_chunks[0].get('score', 0) if retrieved_chunks else 0
        
        if best_score < 0.2:
            return {
                "action": "refine",
                "reason": "Low relevance results",
                "modified_query": self._refine_query(original_query)
            }
        elif best_score < 0.5:
            return {
                "action": "continue_with_caution",
                "reason": "Moderate relevance",
                "confidence": "medium"
            }
        else:
            return {
                "action": "complete",
                "reason": "Satisfactory results found",
                "confidence": "high"
            }
    
    def _refine_query(self, query: str) -> str:
        """Refine a query for better retrieval"""
        # Simple refinement strategies
        query_lower = query.lower()
        
        # Remove question words
        question_words = ['what', 'when', 'where', 'why', 'how', 'is', 'are', 'can', 'do']
        words = query_lower.split()
        refined = [w for w in words if w not in question_words]
        
        if not refined:
            return query
        
        return ' '.join(refined[:10])  # Keep first 10 important words
    
    def should_continue(self, current_step: int, total_steps: int, 
                       results: List[Any]) -> bool:
        """Determine if execution should continue"""
        # Stop if we've done too many steps
        if current_step >= total_steps:
            return False
        
        # Stop if we have no results after 2 steps
        if current_step >= 2 and not results:
            return False
        
        return True
    
    def synthesize(self, step_results: List[Dict[str, Any]], 
                  original_query: str) -> Dict[str, Any]:
        """Synthesize multiple step results into coherent answer"""
        all_chunks = []
        
        for result in step_results:
            chunks = result.get('retrieved_chunks', [])
            all_chunks.extend(chunks)
        
        # Remove duplicates by text content
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            text = chunk.get('text', '')[:100]  # Use first 100 chars as key
            if text not in seen:
                seen.add(text)
                unique_chunks.append(chunk)
        
        return {
            "synthesized_chunks": unique_chunks[:10],  # Top 10 unique chunks
            "total_unique_sources": len(unique_chunks),
            "original_query": original_query
        }
