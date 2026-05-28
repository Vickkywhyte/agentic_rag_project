"""
Calculator Tool - For arithmetic operations
"""
from typing import Dict, Any
import ast
import operator

class CalculatorTool:
    """Tool for performing calculations"""
    
    def __init__(self):
        self.name = "calculate"
        self.description = "Perform mathematical calculations"
    
    def execute(self, expression: str) -> Dict[str, Any]:
        """Evaluate a mathematical expression safely"""
        try:
            # Safe evaluation
            result = self._safe_eval(expression)
            return {
                "success": True,
                "result": result,
                "expression": expression
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "expression": expression
            }
    
    def _safe_eval(self, expr: str):
        """Safely evaluate mathematical expressions"""
        # Allowed operations
        allowed_ops = {
            operator.add: '+',
            operator.sub: '-',
            operator.mul: '*',
            operator.truediv: '/',
            operator.pow: '**',
        }
        
        # Parse and evaluate safely
        tree = ast.parse(expr, mode='eval')
        code = compile(tree, '<string>', 'eval')
        
        # Restrict to safe operations
        allowed_names = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
        }
        
        return eval(code, {"__builtins__": {}}, allowed_names)
    
    def get_tool_definition(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '100 * 0.2')"
                }
            }
        }
