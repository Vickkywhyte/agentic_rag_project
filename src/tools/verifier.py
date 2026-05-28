"""
Verifier Tool - Cross-checks information across sources
"""
from typing import Dict, Any, List

class VerifierTool:
    """Tool for verifying information consistency across chunks"""
    
    def __init__(self):
        self.name = "verify_information"
        self.description = "Verify consistency of information across sources"
    
    def execute(self, chunks: List[Dict[str, Any]], claim: str) -> Dict[str, Any]:
        """Verify a claim against multiple chunks"""
        supporting = []
        contradicting = []
        
        for chunk in chunks:
            chunk_text = chunk.get('text', '').lower()
            claim_lower = claim.lower()
            
            # Simple keyword matching verification
            claim_words = set(claim_lower.split())
            chunk_words = set(chunk_text.split())
            
            overlap = claim_words & chunk_words
            overlap_ratio = len(overlap) / max(len(claim_words), 1)
            
            if overlap_ratio > 0.5:
                supporting.append(chunk)
            elif overlap_ratio < 0.1 and len(overlap) > 0:
                contradicting.append(chunk)
        
        return {
            "success": True,
            "is_consistent": len(contradicting) == 0,
            "supporting_sources": len(supporting),
            "contradicting_sources": len(contradicting),
            "verdict": "consistent" if len(contradicting) == 0 else "inconsistent",
            "details": {
                "supporting": [c.get('source') for c in supporting[:3]],
                "contradicting": [c.get('source') for c in contradicting[:3]]
            }
        }
    
    def get_tool_definition(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "claim": {
                    "type": "string",
                    "description": "The claim to verify against available information"
                }
            }
        }
