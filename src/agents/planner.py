"""
Agentic RAG - Planner Module
Decomposes user queries into multi-step plans
"""
from typing import List, Dict, Any
import json
import re

class Planner:
    """Plans the execution strategy for complex queries"""
    
    def __init__(self):
        self.plan_history = []
    
    def create_plan(self, query: str, available_tools: List[str]) -> Dict[str, Any]:
        """
        Create an execution plan for the query
        
        Args:
            query: User's question or task
            available_tools: List of tool names available to the agent
        
        Returns:
            Plan dictionary with steps and dependencies
        """
        # Analyze query complexity
        complexity = self._assess_complexity(query)
        
        if complexity == "simple":
            return self._simple_plan(query)
        elif complexity == "comparison":
            return self._comparison_plan(query)
        elif complexity == "multi_step":
            return self._multi_step_plan(query)
        else:
            return self._research_plan(query, available_tools)
    
    def _assess_complexity(self, query: str) -> str:
        """Determine query complexity"""
        query_lower = query.lower()
        
        # Indicators of different complexity types
        comparison_words = ['compare', 'versus', 'vs', 'difference between', 'rather than']
        multi_step_words = ['then', 'after that', 'first', 'second', 'finally']
        research_words = ['analyze', 'investigate', 'research', 'find out', 'determine']
        
        if any(word in query_lower for word in comparison_words):
            return "comparison"
        elif any(word in query_lower for word in multi_step_words):
            return "multi_step"
        elif any(word in query_lower for word in research_words):
            return "research"
        else:
            return "simple"
    
    def _simple_plan(self, query: str) -> Dict[str, Any]:
        """Plan for simple single-step queries"""
        return {
            "steps": [
                {
                    "id": 1,
                    "action": "retrieve",
                    "tool": "rag_search",
                    "query": query,
                    "depends_on": []
                }
            ],
            "type": "simple"
        }
    
    def _comparison_plan(self, query: str) -> Dict[str, Any]:
        """Plan for comparison queries"""
        return {
            "steps": [
                {
                    "id": 1,
                    "action": "retrieve",
                    "tool": "rag_search",
                    "query": f"First item for: {query}",
                    "depends_on": []
                },
                {
                    "id": 2,
                    "action": "retrieve",
                    "tool": "rag_search",
                    "query": f"Second item for: {query}",
                    "depends_on": []
                },
                {
                    "id": 3,
                    "action": "synthesize",
                    "tool": "compare",
                    "query": query,
                    "depends_on": [1, 2]
                }
            ],
            "type": "comparison"
        }
    
    def _multi_step_plan(self, query: str) -> Dict[str, Any]:
        """Plan for multi-step queries"""
        # Extract potential sub-queries
        sentences = re.split(r'[.!?]\s+', query)
        
        steps = []
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                steps.append({
                    "id": i + 1,
                    "action": "retrieve",
                    "tool": "rag_search",
                    "query": sentence.strip(),
                    "depends_on": [i] if i > 0 else []
                })
        
        return {
            "steps": steps,
            "type": "multi_step"
        }
    
    def _research_plan(self, query: str, available_tools: List[str]) -> Dict[str, Any]:
        """Plan for complex research queries"""
        plan = {
            "steps": [
                {
                    "id": 1,
                    "action": "retrieve",
                    "tool": "rag_search",
                    "query": query,
                    "depends_on": []
                },
                {
                    "id": 2,
                    "action": "verify",
                    "tool": "verify_information",
                    "query": "Check consistency across sources",
                    "depends_on": [1]
                }
            ],
            "type": "research"
        }
        
        # Add summarization if available
        if "summarize" in available_tools:
            plan["steps"].append({
                "id": 3,
                "action": "summarize",
                "tool": "summarize",
                "depends_on": [2]
            })
        
        return plan
    
    def update_history(self, plan: Dict[str, Any], results: List[Any]):
        """Store plan execution history"""
        self.plan_history.append({
            "plan": plan,
            "results": results,
            "timestamp": None  # Would use datetime in production
        })
