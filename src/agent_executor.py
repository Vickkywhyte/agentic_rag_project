"""
Agentic RAG - Main Executor
Orchestrates the entire agentic workflow
"""
from typing import Dict, Any, List
from .agents.planner import Planner
from .agents.reasoner import Reasoner
from .tools.rag_tool import RAGTool
from .tools.calculator import CalculatorTool
from .tools.verifier import VerifierTool
from rag_engine import generate

class AgenticRAGExecutor:
    """Main executor for Agentic RAG system"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.planner = Planner()
        self.reasoner = Reasoner()
        
        # Initialize tools
        self.tools = {
            "rag_search": RAGTool(vector_store),
            "calculate": CalculatorTool(),
            "verify_information": VerifierTool()
        }
        
        self.execution_history = []
    
    def execute(self, query: str, api_key: str) -> Dict[str, Any]:
        """
        Execute a query using agentic workflow
        
        Returns:
            Dictionary with final answer and execution trace
        """
        # Step 1: Plan
        plan = self.planner.create_plan(query, list(self.tools.keys()))
        
        # Step 2: Execute steps
        step_results = []
        for step in plan['steps']:
            result = self._execute_step(step, step_results)
            step_results.append(result)
            
            # Step 3: Reason about result
            decision = self.reasoner.evaluate(result, query)
            
            if decision['action'] == 'retry':
                # Retry with refined query
                refined_step = step.copy()
                refined_step['query'] = decision.get('modified_query', step.get('query', query))
                result = self._execute_step(refined_step, step_results)
                step_results[-1] = result
            elif decision['action'] == 'complete':
                break
        
        # Step 4: Synthesize final answer
        if plan['type'] == 'comparison' or len(step_results) > 1:
            synthesis = self.reasoner.synthesize(step_results, query)
            final_answer = self._generate_final_answer(query, synthesis, api_key)
        else:
            # Simple answer from first step
            final_answer = self._generate_final_answer(
                query, 
                {"synthesized_chunks": step_results[0].get('retrieved_chunks', [])},
                api_key
            )
        
        return {
            "answer": final_answer,
            "execution_trace": {
                "plan": plan,
                "steps": step_results,
                "tools_used": list(set([s.get('tool_used', '') for s in step_results if s.get('tool_used')]))
            }
        }
    
    def _execute_step(self, step: Dict[str, Any], previous_results: List) -> Dict[str, Any]:
        """Execute a single step using the appropriate tool"""
        tool_name = step.get('tool')
        tool = self.tools.get(tool_name)
        
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "step": step
            }
        
        # Execute based on tool type
        if tool_name == "rag_search":
            query = step.get('query', '')
            result = tool.execute(query)
            result['tool_used'] = tool_name
            return result
        elif tool_name == "calculate":
            query = step.get('query', '')
            result = tool.execute(query)
            result['tool_used'] = tool_name
            return result
        elif tool_name == "verify_information":
            # Get previous chunks for verification
            all_chunks = []
            for prev in previous_results:
                all_chunks.extend(prev.get('retrieved_chunks', []))
            result = tool.execute(all_chunks, step.get('query', ''))
            result['tool_used'] = tool_name
            return result
        elif tool_name == "compare":
            # Comparison tool uses multiple previous results
            return self._compare_results(previous_results, step.get('query', ''))
        else:
            return {
                "success": False,
                "error": f"Unsupported tool: {tool_name}",
                "step": step
            }
    
    def _compare_results(self, previous_results: List, query: str) -> Dict[str, Any]:
        """Compare multiple retrieval results"""
        comparisons = []
        
        for i, result in enumerate(previous_results):
            chunks = result.get('retrieved_chunks', [])
            if chunks:
                comparisons.append({
                    "source_set": i + 1,
                    "top_result": chunks[0].get('text', '')[:200],
                    "score": chunks[0].get('score', 0)
                })
        
        return {
            "success": True,
            "tool_used": "compare",
            "comparisons": comparisons,
            "count": len(comparisons)
        }
    
    def _generate_final_answer(self, query: str, synthesis: Dict[str, Any], api_key: str) -> str:
        """Generate final answer using LLM"""
        chunks = synthesis.get('synthesized_chunks', [])
        
        # Generate using existing RAG engine
        answer, _ = generate(query, chunks, api_key)
        return answer
