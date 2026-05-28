"""
RAG Tool - Wrapper for the retrieval system
"""
from typing import Dict, Any, List
from rag_engine import retrieve, VectorStore

class RAGTool:
    """Tool for retrieving information from the vector store"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.name = "rag_search"
        self.description = "Search the company documents for information"
    
    def execute(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Execute a retrieval operation"""
        results = retrieve(self.store, query, top_k)
        
        return {
            "success": len(results) > 0,
            "retrieved_chunks": results,
            "query": query,
            "count": len(results)
        }
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return tool definition for LLM"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up in the documents"
                }
            }
        }
